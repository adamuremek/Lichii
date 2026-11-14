// SPDX-License-Identifier: GPL-3.0-or-later

#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/vk_draw_bindings.h"

namespace {

template <class Handle>
Handle HandleOf(uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(value);
    } else {
        return static_cast<Handle>(value);
    }
}

} // namespace

TEST_CASE("Push descriptors compare every template field and array element",
          "[video_core][draw_bindings]") {
    using Vulkan::DescriptorUpdateEntry;
    constexpr std::array types{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER};
    const auto buffer = HandleOf<VkBuffer>(1);
    const auto image = HandleOf<VkImageView>(2);
    const auto sampler = HandleOf<VkSampler>(3);
    const auto texel = HandleOf<VkBufferView>(4);
    const VkDescriptorImageInfo image_info{sampler, image, VK_IMAGE_LAYOUT_GENERAL};
    const std::array<DescriptorUpdateEntry, types.size()> previous{
        VkDescriptorBufferInfo{buffer, 0, 256}, VkDescriptorBufferInfo{buffer, 256, 4096},
        image_info, image_info, texel};
    auto current = previous;
    const auto equal = [&] { return Vulkan::EqualPushDescriptors(types, previous, current); };
    CHECK(equal());
    current[0].buffer.buffer = HandleOf<VkBuffer>(7);
    CHECK_FALSE(equal());
    current = previous;
    current[1].buffer.offset = 512;
    CHECK_FALSE(equal());
    current = previous;
    current[1].buffer.range = 2048;
    CHECK_FALSE(equal());
    current = previous;
    current[3].image.sampler = HandleOf<VkSampler>(5);
    CHECK_FALSE(equal());
    current = previous;
    current[3].image.imageView = HandleOf<VkImageView>(6);
    CHECK_FALSE(equal());
    current = previous;
    current[3].image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    CHECK_FALSE(equal());
    current = previous;
    current[4].texel_buffer = VK_NULL_HANDLE;
    CHECK_FALSE(equal());
    CHECK_FALSE(Vulkan::EqualPushDescriptors(types, previous, std::span{current}.first(4)));
}

TEST_CASE("Draw push constants track exact bytes and refresh after a pipeline rebind",
          "[video_core][draw_bindings]") {
    Vulkan::DrawPushConstants cache;
    Vulkan::DrawPushConstants::Scaling scaling{};
    CHECK(cache.Update(false, scaling));
    CHECK_FALSE(cache.Update(false, scaling));
    scaling.back() = 1;
    CHECK(cache.Update(false, scaling));
    auto merged = scaling;
    const std::array area{-0.0f, 1.0f, 2.0f, 3.0f};
    Vulkan::DrawPushConstants::ApplyRenderArea(merged, area);
    CHECK(merged[0] == std::bit_cast<u32>(-0.0f));
    CHECK(merged[3] == std::bit_cast<u32>(3.0f));
    CHECK(merged.back() == 1);
    CHECK(cache.Update(false, merged));
    CHECK_FALSE(cache.Update(false, merged));
    CHECK(cache.Update(true, merged));
    CHECK(cache.Update(false, scaling));
}
