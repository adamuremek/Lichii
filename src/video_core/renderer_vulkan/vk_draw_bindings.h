// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <span>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"

namespace Vulkan {

/// Compare only the fields that the Vulkan update template consumes.
[[nodiscard]] inline bool EqualPushDescriptors(std::span<const VkDescriptorType> types,
                                               std::span<const DescriptorUpdateEntry> previous,
                                               std::span<const DescriptorUpdateEntry> current) {
    if (types.size() != previous.size() || types.size() != current.size()) {
        return false;
    }
    for (size_t index = 0; index < types.size(); ++index) {
        const auto& a = previous[index];
        const auto& b = current[index];
        switch (types[index]) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            if (a.buffer.buffer != b.buffer.buffer || a.buffer.offset != b.buffer.offset ||
                a.buffer.range != b.buffer.range) {
                return false;
            }
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            if (a.texel_buffer != b.texel_buffer) {
                return false;
            }
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            if (a.image.sampler != b.image.sampler || a.image.imageView != b.image.imageView ||
                a.image.imageLayout != b.image.imageLayout) {
                return false;
            }
            break;
        default:
            return false;
        }
    }
    return true;
}

class DrawPushConstants {
public:
    using Scaling = std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>;

    static void ApplyRenderArea(Scaling& words, const std::array<f32, 4>& area) {
        using namespace Shader::Backend::SPIRV;
        // The render area overwrites the beginning of the scaling words in the same stages.
        static_assert(RENDERAREA_LAYOUT_OFFSET == RESCALING_LAYOUT_WORDS_OFFSET);
        static_assert(sizeof(Scaling) >= sizeof(area));
        static_assert(RESCALING_LAYOUT_DOWN_FACTOR_OFFSET >= sizeof(area));
        std::ranges::copy(std::bit_cast<std::array<u32, 4>>(area), words.begin());
    }

    [[nodiscard]] bool Update(bool pipeline_rebound, const Scaling& words) {
        if (!pipeline_rebound && defined && last_words == words) {
            return false;
        }
        last_words = words;
        defined = true;
        return true;
    }

private:
    Scaling last_words{};
    bool defined{};
};

} // namespace Vulkan
