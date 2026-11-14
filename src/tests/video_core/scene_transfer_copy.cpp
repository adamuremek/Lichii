// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/util.h"

namespace {

using namespace VideoCommon;
using VideoCore::Surface::PixelFormat;

constexpr Extent3D SceneExtent{2560, 1440, 1};
constexpr GPUVAddr SourceAddress = 0x01000000;
constexpr GPUVAddr DestinationAddress = 0x03000000;
constexpr VAddr SourcePhysical = 0x04000000;
constexpr VAddr DestinationPhysical = 0x06000000;

ImageInfo SceneInfo() {
    ImageInfo info;
    info.format = PixelFormat::B10G11R11_FLOAT;
    info.type = ImageType::e2D;
    info.resources = {1, 1};
    info.size = SceneExtent;
    return info;
}

TEST_CASE("Scene transfer requires exact color image bases and full extents",
          "[video_core][texture_cache][scene_transfer]") {
    const ImageBase src{SceneInfo(), SourceAddress, SourcePhysical};
    const ImageBase dst{SceneInfo(), DestinationAddress, DestinationPhysical};
    CHECK(CanTransferWholeColorImage(dst, src, DestinationAddress, SourceAddress, SceneExtent));
    CHECK_FALSE(
        CanTransferWholeColorImage(dst, src, DestinationAddress + 4, SourceAddress, SceneExtent));
    CHECK_FALSE(CanTransferWholeColorImage(dst, src, DestinationAddress, SourceAddress,
                                           Extent3D{2560, 1439, 1}));
    CHECK_FALSE(
        CanTransferWholeColorImage(dst, dst, DestinationAddress, DestinationAddress, SceneExtent));
    const ImageBase overlapping{SceneInfo(), DestinationAddress, SourcePhysical + 4096};
    CHECK_FALSE(CanTransferWholeColorImage(overlapping, src, DestinationAddress, SourceAddress,
                                           SceneExtent));
}

TEST_CASE("Scene transfer keeps aliased and partially backed images on the blit path",
          "[video_core][texture_cache][scene_transfer]") {
    const ImageBase src{SceneInfo(), SourceAddress, SourcePhysical};
    ImageBase dst{SceneInfo(), DestinationAddress, DestinationPhysical};
    const auto can_copy = [&] {
        return CanTransferWholeColorImage(dst, src, DestinationAddress, SourceAddress, SceneExtent);
    };
    CHECK(can_copy());
    dst.info.resources.layers = 2;
    CHECK_FALSE(can_copy());
    dst.info.resources.layers = 1;
    dst.info.num_samples = 2;
    CHECK_FALSE(can_copy());
    dst.info.num_samples = 1;
    dst.info.format = PixelFormat::R16_FLOAT;
    CHECK_FALSE(can_copy());
    dst.info.format = PixelFormat::B10G11R11_FLOAT;
    for (const auto flag :
         {ImageFlagBits::Sparse, ImageFlagBits::TrimmedLevels, ImageFlagBits::Rescaled,
          ImageFlagBits::Unbacked, ImageFlagBits::BadOverlap}) {
        dst.flags |= flag;
        CHECK_FALSE(can_copy());
        dst.flags &= ~flag;
    }
    dst.aliased_images.emplace_back();
    CHECK_FALSE(can_copy());
    dst.aliased_images.clear();
    dst.overlapping_images.emplace_back();
    CHECK_FALSE(can_copy());
}

} // namespace
