// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_policy.h"

namespace {

using namespace Common::Literals;
using namespace Vulkan::StagingBufferPolicy;

TEST_CASE("Vulkan staging size classes round large requests in quarter steps",
          "[video_core][vulkan][staging]") {
    CHECK(ClassSize(SizeClass(17_MiB, false), false) == 32_MiB);
    CHECK(ClassSize(SizeClass(17_MiB, true), true) == 20_MiB);
    CHECK(ClassSize(SizeClass(20_MiB + 1, true), true) == 24_MiB);
    CHECK(SizeClass(0, true) == 0);

    for (bool fine : {false, true}) {
        const u32 last = fine ? NumClasses - 1 : SizeBits;
        for (u32 bucket = 1; bucket <= last; ++bucket) {
            const auto previous_capacity = ClassSize(bucket - 1, fine);
            const auto capacity = ClassSize(bucket, fine);
            CHECK(capacity > previous_capacity);
            CHECK(SizeClass(previous_capacity + 1, fine) == bucket);
            CHECK(SizeClass(capacity, fine) == bucket);
            CHECK(LastReuseClass(bucket, fine) >= bucket);
            CHECK(LastReuseClass(bucket, fine) <= last);
        }
        CHECK(ClassSize(last, fine) == (std::numeric_limits<u64>::max)());
    }
}

TEST_CASE("Vulkan staging keeps recent buffers and respects deferred and in-flight use",
          "[video_core][vulkan][staging]") {
    CHECK_FALSE(CanRelease(100, 40, false, true));
    CHECK(CanRelease(101, 40, false, true));
    CHECK_FALSE(CanRelease(101, 40, true, true));
    CHECK_FALSE(CanRelease(101, 40, false, false));

    const u32 large = SizeClass(33_MiB, true);
    CHECK(ClassSize(large, true) == 40_MiB);
    CHECK(ClassSize(LastReuseClass(large, true), true) == 56_MiB);
    CHECK(LastReuseClass(SizeClass(1_MiB, true), true) == SizeClass(1_MiB, true));
    CHECK(LastReuseClass(SizeClass(33_MiB, false), false) == SizeClass(33_MiB, false));
}

TEST_CASE("Vulkan upload slices remain disjoint until their submissions finish",
          "[video_core][vulkan][staging]") {
    UploadArenaCursor block;
    constexpr size_t alignment = 768;
    constexpr size_t capacity = UploadArenaBlockSize;
    CHECK(block.TryReserve(768 + 1, alignment, capacity, 5, 4) == 0);
    CHECK(block.TryReserve(1, alignment, capacity, 5, 4) == 1536);
    CHECK(block.TryReserve(1024, alignment, capacity, 6, 4) == 2304);
    CHECK(block.TryReserve(capacity - 3840, alignment, capacity, 6, 5) == 3840);
    CHECK_FALSE(block.TryReserve(1, alignment, capacity, 7, 5));
    CHECK(block.last_tick == 6);
    CHECK(block.TryReserve(capacity, alignment, capacity, 7, 6) == 0);
    CHECK_FALSE(block.TryReserve(capacity + 1, alignment, capacity, 8, 7));
    CHECK(block.TryReserve(512, alignment, capacity, 8, 7) == 0);
}

TEST_CASE("Vulkan upload arena absorbs a burst with several submissions in flight",
          "[video_core][vulkan][staging]") {
    struct Slice {
        size_t offset;
        size_t size;
        u64 tick;
    };
    std::array<UploadArenaCursor, UploadArenaBlockCount> blocks{};
    std::array<std::vector<Slice>, UploadArenaBlockCount> live{};
    constexpr size_t request_count = 928;
    size_t slabs_used = 0;
    size_t fallbacks = 0;
    size_t overlaps = 0;
    for (size_t i = 0; i < request_count; ++i) {
        const size_t size = (1 + (i * 7) % 15) * 192_KiB;
        const u64 tick = 1 + i / 8;
        const u64 done = tick > 6 ? tick - 6 : 0;
        bool reserved = false;
        for (size_t j = 0; j < blocks.size(); ++j) {
            const auto offset = blocks[j].TryReserve(size, 768, UploadArenaBlockSize, tick, done);
            if (!offset) {
                continue;
            }
            reserved = true;
            slabs_used = (std::max)(slabs_used, j + 1);
            auto& slices = live[j];
            std::erase_if(slices, [done](const Slice& slice) { return slice.tick <= done; });
            for (const Slice& slice : slices) {
                overlaps += *offset < slice.offset + slice.size && slice.offset < *offset + size;
            }
            slices.push_back({*offset, size, tick});
            break;
        }
        fallbacks += !reserved;
    }
    CHECK(overlaps == 0);
    CHECK(fallbacks == 0);
    CHECK(slabs_used <= 3);
}

TEST_CASE("Vulkan large uploads consume free arena space and reduce the staging size class",
          "[video_core][vulkan][staging]") {
    constexpr size_t capacity = WarmUploadArenaBlockSize;
    constexpr size_t request = 1024_MiB;
    constexpr u64 tick = 12;
    constexpr u64 done = 10;
    constexpr size_t alignment = 768;
    std::array<UploadArenaReservation, UploadArenaBlockCount> blocks{{
        {{128_MiB, 10}, capacity},
        {{192_MiB, 11}, capacity},
        {{capacity, 11}, capacity},
        {{}, capacity},
    }};

    const size_t spill = PlanLargeUpload(blocks, request, alignment, tick, done, false);
    CHECK(blocks[0].offset == 0);
    CHECK(blocks[0].size == capacity);
    CHECK(blocks[1].offset == Common::AlignUp(192_MiB, alignment));
    CHECK(blocks[1].cursor.used == capacity);
    CHECK(blocks[2].size == 0);
    CHECK(blocks[2].cursor.last_tick == 11);
    CHECK(blocks[3].size == capacity);
    CHECK(spill == request - blocks[0].size - blocks[1].size - blocks[3].size);
    CHECK(ClassSize(SizeClass(spill, false), false) == 512_MiB);

    for (auto& block : blocks) {
        block = {{capacity, 11}, capacity};
    }
    blocks.back().cursor = {};
    CHECK(PlanLargeUpload(blocks, request, alignment, tick, done, false) == request);
    for (auto& block : blocks) {
        block = {{capacity, 10}, capacity};
    }
    CHECK(PlanLargeUpload(blocks, request, alignment, tick, done, false) == 0);
    for (const auto& block : blocks) {
        CHECK(block.size == capacity);
        CHECK(block.cursor.last_tick == tick);
    }
}

} // namespace
