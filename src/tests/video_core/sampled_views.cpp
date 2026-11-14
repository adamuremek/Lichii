// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/sampled_views.h"
#include "video_core/texture_cache/texture_cache_base.h"

namespace {

using VideoCommon::ImageBase;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageId;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewBase;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
using VideoCore::Surface::PixelFormat;

ImageInfo Info() {
    ImageInfo info;
    info.format = PixelFormat::A8B8G8R8_UNORM;
    info.type = ImageType::e2D;
    info.size = {64, 64, 1};
    info.resources.levels = 1;
    info.resources.layers = 1;
    return info;
}

struct View {
    ImageViewId id;
};

/// Images and views laid out as TextureCache makes them: the null image and view first.
struct Cache {
    Cache() {
        REQUIRE(images.insert(VideoCommon::NullImageParams{}) == VideoCommon::NULL_IMAGE_ID);
        REQUIRE(views.insert(VideoCommon::NullImageViewParams{}) ==
                VideoCommon::NULL_IMAGE_VIEW_ID);
    }

    ImageId AddImage() {
        const ImageId id = images.insert(Info(), 0x10000, 0x10000);
        images[id].flags &= ~ImageFlagBits::CpuModified; // uploaded
        return id;
    }

    ImageViewId AddView(ImageId image) {
        const ImageViewInfo info{ImageViewType::e2D, PixelFormat::A8B8G8R8_UNORM};
        return views.insert(info, images[image].info, image, 0x10000);
    }

    bool Change(std::vector<View> sampled) const {
        return VideoCommon::SampledViewsChange(sampled, views, images);
    }

    Common::SlotVector<ImageBase> images;
    Common::SlotVector<ImageViewBase> views;
};

} // namespace

TEST_CASE("The null view is never prepared", "[video_core][texture_cache]") {
    Cache cache;
    // Its image has no valid format and stays CpuModified: preparing it uploads it (#904).
    REQUIRE(True(cache.images[VideoCommon::NULL_IMAGE_ID].flags & ImageFlagBits::CpuModified));
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, VideoCommon::NULL_IMAGE_VIEW_ID));
    CHECK_FALSE(cache.Change({{VideoCommon::NULL_IMAGE_VIEW_ID}}));

    // A draw sampling an uploaded texture beside a null one changes nothing.
    const ImageViewId view = cache.AddView(cache.AddImage());
    CHECK_FALSE(cache.Change({{VideoCommon::NULL_IMAGE_VIEW_ID}, {view}}));
}

TEST_CASE("Buffer views and empty ids are never prepared", "[video_core][texture_cache]") {
    Cache cache;
    ImageInfo info;
    info.type = ImageType::Buffer;
    info.format = PixelFormat::R32_UINT;
    info.size = {256, 1, 1};
    const ImageViewInfo view_info{ImageViewType::Buffer, PixelFormat::R32_UINT};
    const ImageViewId buffer = cache.views.insert(info, view_info, 0x20000);
    // A buffer view points at the null image.
    REQUIRE(cache.views[buffer].image_id == VideoCommon::NULL_IMAGE_ID);
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, buffer));
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, ImageViewId{}));
    CHECK_FALSE(cache.Change({{buffer}, {ImageViewId{}}}));
}

TEST_CASE("Empty descriptor views with their own ids never prepare the null image",
          "[video_core][texture_cache]") {
    Cache cache;
    const ImageViewId empty = cache.views.insert(VideoCommon::NullImageViewParams{});
    REQUIRE(empty != VideoCommon::NULL_IMAGE_VIEW_ID);
    REQUIRE_FALSE(cache.views[empty].IsBuffer());
    REQUIRE(cache.views[empty].image_id == VideoCommon::NULL_IMAGE_ID);
    CHECK_FALSE(VideoCommon::PreparesImage(cache.views, empty));
    CHECK_FALSE(cache.Change({{empty}, {cache.AddView(cache.AddImage())}}));
}

TEST_CASE("A sampled image with an upload pending changes", "[video_core][texture_cache]") {
    Cache cache;
    const ImageId image = cache.AddImage();
    const ImageViewId view = cache.AddView(image);
    CHECK(VideoCommon::PreparesImage(cache.views, view));
    CHECK_FALSE(cache.Change({{view}}));
    cache.images[image].flags |= ImageFlagBits::CpuModified;
    CHECK(cache.Change({{view}}));
}

TEST_CASE("A sampled image changes while a newer alias has not been copied in",
          "[video_core][texture_cache]") {
    Cache cache;
    const ImageId sampled = cache.AddImage();
    const ImageId target = cache.AddImage();
    const ImageViewId view = cache.AddView(sampled);
    REQUIRE(AddImageAlias(cache.images[sampled], cache.images[target], sampled, target));
    VideoCommon::SetModificationTick(cache.images, cache.images[sampled], 4);
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 3);
    CHECK_FALSE(cache.Change({{view}}));

    // A draw renders into the alias (TextureCache::MarkModification): the next sample copies it.
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 5);
    CHECK(cache.Change({{view}}));

    // As SynchronizeAliases leaves it after the copy.
    VideoCommon::SetModificationTick(cache.images, cache.images[sampled], 5);
    cache.images[sampled].newest_alias_tick = 5;
    CHECK_FALSE(cache.Change({{view}}));

    // A newer alias deleted before it was copied leaves nothing to copy.
    VideoCommon::SetModificationTick(cache.images, cache.images[target], 6);
    cache.images[sampled].aliased_images.clear();
    CHECK_FALSE(cache.Change({{view}}));
}

TEST_CASE("Prepared render targets need the current frame and no pending image work",
          "[video_core][texture_cache][prepared_targets]") {
    Cache cache;
    const ImageId target = cache.AddImage();
    auto& image = cache.images[target];
    constexpr u64 frame = 41;
    image.last_use_frame = frame;
    CHECK_FALSE(VideoCommon::RenderTargetStillPrepared(image, frame));
    image.flags |= ImageFlagBits::GpuModified;
    CHECK(VideoCommon::RenderTargetStillPrepared(image, frame));
    CHECK_FALSE(VideoCommon::RenderTargetStillPrepared(image, frame + 1));
    for (const auto pending :
         {ImageFlagBits::CpuModified, ImageFlagBits::SparseTiles, ImageFlagBits::SparseLayers,
          ImageFlagBits::UnwrittenSparse, ImageFlagBits::Remapped}) {
        image.flags |= pending;
        CHECK_FALSE(VideoCommon::RenderTargetStillPrepared(image, frame));
        image.flags &= ~pending;
        CHECK(VideoCommon::RenderTargetStillPrepared(image, frame));
    }
    const ImageId alias = cache.AddImage();
    REQUIRE(AddImageAlias(cache.images[target], cache.images[alias], target, alias));
    CHECK_FALSE(VideoCommon::RenderTargetStillPrepared(cache.images[target], frame));
}

TEST_CASE("Prepared target keys distinguish attachment, frame, and mapping changes",
          "[video_core][texture_cache][prepared_targets]") {
    int address_space{};
    int other_space{};
    const VideoCommon::RenderTargetPreparationKey key{&address_space, 4, 18, 22, 219, 15};
    CHECK(key == VideoCommon::RenderTargetPreparationKey{&address_space, 4, 18, 22, 219, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&other_space, 4, 18, 22, 219, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&address_space, 5, 18, 22, 219, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&address_space, 4, 19, 22, 219, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&address_space, 4, 18, 23, 219, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&address_space, 4, 18, 22, 220, 15});
    CHECK(key != VideoCommon::RenderTargetPreparationKey{&address_space, 4, 18, 22, 219, 16});
}

TEST_CASE("Prepared target bookkeeping cost in a repeated draw sequence",
          "[.prepared_targets_benchmark]") {
    struct LruTraits {
        using ObjectType = ImageId;
        using TickType = u64;
    };
    constexpr size_t draws = 500'000;
    for (const size_t draws_per_pass : {1, 16, 128}) {
        for (size_t trial = 0; trial < 5; ++trial) {
            std::array<double, 2> timing{};
            for (size_t step = 0; step < 2; ++step) {
                const bool skip_prepared = (trial + step) % 2 != 0;
                Cache cache;
                Common::LeastRecentlyUsedCache<LruTraits> lru;
                const std::array images{cache.AddImage(), cache.AddImage()};
                for (const ImageId id : images) {
                    cache.images[id].lru_index = lru.Insert(id, 0);
                }
                std::optional<VideoCommon::RenderTargetPreparationKey> prepared;
                u64 modification = 0;
                size_t hits = 0;
                const auto begin = std::chrono::steady_clock::now();
                for (size_t draw = 0; draw < draws; ++draw) {
                    const u64 pass = draw / draws_per_pass;
                    const u64 frame = pass / 110 + 1;
                    const auto key = [&] {
                        return VideoCommon::RenderTargetPreparationKey{&cache,       0, pass, frame,
                                                                       modification, 1};
                    };
                    if (skip_prepared && prepared == key() &&
                        std::ranges::all_of(images, [&](ImageId id) {
                            return VideoCommon::RenderTargetStillPrepared(cache.images[id], frame);
                        })) {
                        ++hits;
                        continue;
                    }
                    for (const ImageId id : images) {
                        auto& image = cache.images[id];
                        image.flags |= ImageFlagBits::GpuModified;
                        VideoCommon::SetModificationTick(cache.images, image, ++modification);
                        lru.Touch(image.lru_index, frame);
                        image.last_use_frame = frame;
                    }
                    if (skip_prepared) {
                        prepared = key();
                    }
                }
                const std::chrono::duration<double, std::nano> elapsed =
                    std::chrono::steady_clock::now() - begin;
                timing[skip_prepared] = elapsed.count() / draws;
                const size_t passes = (draws + draws_per_pass - 1) / draws_per_pass;
                CHECK(hits == (skip_prepared ? draws - passes : 0));
                CHECK(modification == (draws - hits) * images.size());
            }
            std::printf("draws per pass %zu trial %zu: %.2f -> %.2f ns/draw\n", draws_per_pass,
                        trial + 1, timing[0], timing[1]);
        }
    }
}

TEST_CASE("Texture memos distinguish alternating handles and descriptor contents",
          "[video_core][texture_cache]") {
    using VideoCommon::SampledTextureMemo;
    constexpr std::array<GPUVAddr, 2> addresses{0x1000, 0x2000};
    std::array<u32, 3> uniform_words{0x100001, 0, 0x200002};
    const std::array handles{&uniform_words[0], &uniform_words[2]};
    std::array<std::array<u8, sizeof(VideoCommon::TICEntry)>, 2> tics{};
    std::array<std::array<u8, sizeof(VideoCommon::TSCEntry)>, 2> tscs{};
    const SampledTextureMemo::Context context{&handles, &tics, 10, 20, 30, 40, false};

    const auto capture = [&] {
        SampledTextureMemo memo;
        memo.context = context;
        memo.valid = true;
        for (size_t i = 0; i < addresses.size(); ++i) {
            auto& entry = memo.entries.emplace_back();
            entry.handle_address = addresses[i];
            entry.handle_host = reinterpret_cast<const u8*>(handles[i]);
            entry.tic_host = tics[i].data();
            entry.tsc_host = tscs[i].data();
            entry.raw = *handles[i];
            std::memcpy(&entry.tic, tics[i].data(), tics[i].size());
            std::memcpy(&entry.tsc, tscs[i].data(), tscs[i].size());
        }
        return memo;
    };

    const auto first = capture();
    CHECK(first.Matches(context, addresses));
    CHECK_FALSE(first.Matches(context, std::span(addresses).first(1)));
    CHECK_FALSE(first.Matches(context, std::array{addresses[1], addresses[0]}));

    // A range can reach a texture handle even when only its neighboring uniform changed.
    VideoCommon::Nx2FastDraw::Delta write;
    write.write_begin[0] = addresses[0];
    write.write_end[0] = addresses[0] + 8;
    write.num_writes = 1;
    constexpr std::array handle_word{std::pair{0U, 4U}};
    CHECK(write.ReachesWords(addresses[0], handle_word, {}));
    const std::array upload{first.entries[0].raw, 42U};
    std::memcpy(uniform_words.data(), upload.data(), sizeof(upload));
    CHECK(uniform_words[1] == 42);
    CHECK(first.Matches(context, addresses));

    *handles[1] ^= 1;
    tics[1].back() ^= 1;
    tscs[1].back() ^= 1;
    const auto second = capture();
    CHECK_FALSE(first.Matches(context, addresses));
    CHECK(second.Matches(context, addresses));

    *handles[1] ^= 1;
    CHECK_FALSE(first.Matches(context, addresses));
    CHECK_FALSE(second.Matches(context, addresses));
    tics[1].back() ^= 1;
    CHECK_FALSE(first.Matches(context, addresses));
    tscs[1].back() ^= 1;
    CHECK(first.Matches(context, addresses));
    CHECK_FALSE(second.Matches(context, addresses));

    auto invalid = first;
    invalid.entries[0].handle_host = nullptr;
    for (const auto member : {&SampledTextureMemo::Context::host_serial,
                              &SampledTextureMemo::Context::image_changes,
                              &SampledTextureMemo::Context::sampler_changes}) {
        auto changed = context;
        ++(changed.*member);
        CHECK_FALSE(invalid.Matches(changed, addresses));
    }
    auto changed = context;
    changed.memory = &tscs;
    CHECK_FALSE(invalid.Matches(changed, addresses));
    changed = context;
    changed.image_table = &tscs;
    CHECK_FALSE(invalid.Matches(changed, addresses));
    changed = context;
    ++changed.bind_id;
    CHECK_FALSE(invalid.Matches(changed, addresses));
    changed = context;
    changed.via_header_index = true;
    CHECK_FALSE(invalid.Matches(changed, addresses));
    invalid.valid = false;
    CHECK_FALSE(invalid.Matches(context, addresses));
}
