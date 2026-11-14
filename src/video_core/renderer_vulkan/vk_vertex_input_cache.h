// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include <boost/container/static_vector.hpp>

#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

struct VertexInputLayout {
    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;
};

struct VertexInputKey {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    static_assert(Regs::NumVertexAttributes == 32 && Regs::NumVertexArrays == 32);

    std::array<u32, 32> attributes{};
    std::array<u32, 32> divisors{};
    std::array<u16, 32> strides{};
    u32 instanced{};
    u32 repacked{};

    static VertexInputKey Capture(const Regs& regs, u32 max_attributes, u32 max_bindings,
                                  u32 repacked_mask) {
        VertexInputKey key{};
        for (u32 i = 0; i < max_attributes; ++i) {
            key.attributes[i] = regs.vertex_attrib_format[i].hex;
        }
        for (u32 i = 0; i < max_bindings; ++i) {
            const auto& stream = regs.vertex_streams[i];
            const bool instance = regs.vertex_stream_instances.IsInstancingEnabled(i);
            key.strides[i] = static_cast<u16>(stream.stride.Value());
            key.divisors[i] = instance ? stream.frequency : 1;
            key.instanced |= u32{instance} << i;
        }
        key.repacked = repacked_mask & static_cast<u32>((u64{1} << max_bindings) - 1);
        return key;
    }

    bool operator==(const VertexInputKey&) const = default;
};
static_assert(std::has_unique_object_representations_v<VertexInputKey>);

class VertexInputCache {
public:
    static constexpr size_t MaxLayouts = 1024;

    struct Statistics {
        u64 built{};
        u64 reused{};
        u64 redundant_commands{};
        u64 uncached{};
    };

    template <typename Build>
    const VertexInputLayout* FindOrBuild(const VertexInputKey& key, Build&& build) {
        for (const size_t slot : {recent_slot, recent_slot ^ 1}) {
            if (recent[slot].layout && key == recent[slot].key) {
                recent_slot = slot;
                ++stats.reused;
                return recent[slot].layout;
            }
        }
        const VertexInputLayout* result;
        const auto it = layouts.find(key);
        if (it != layouts.end()) {
            result = it->second.get();
            ++stats.reused;
        } else {
            if (layouts.size() >= MaxLayouts) {
                ++stats.uncached;
                return nullptr;
            }
            auto layout = std::make_unique<VertexInputLayout>();
            build(*layout);
            result = layouts.emplace(key, std::move(layout)).first->second.get();
            ++stats.built;
        }
        recent_slot ^= 1;
        recent[recent_slot] = {key, result};
        return result;
    }

    bool NeedsRecording(const VertexInputLayout* layout, u64 epoch, u64 stride_serial) {
        // A buffer bind can set strides after the last full vertex-input command.
        if (recorded_layout == layout && recorded_epoch == epoch &&
            recorded_stride_serial == stride_serial) {
            ++stats.redundant_commands;
            return false;
        }
        recorded_layout = layout;
        recorded_epoch = epoch;
        recorded_stride_serial = stride_serial;
        return true;
    }

    void InvalidateRecording() {
        recorded_layout = nullptr;
    }

    Statistics TakeStatistics() {
        return std::exchange(stats, {});
    }

private:
    struct Hash {
        size_t operator()(const VertexInputKey& key) const noexcept {
            return static_cast<size_t>(
                Common::CityHash64(reinterpret_cast<const char*>(&key), sizeof(key)));
        }
    };
    struct RecentLayout {
        VertexInputKey key{};
        const VertexInputLayout* layout{};
    };

    // Commands keep these addresses until the Vulkan worker records them.
    Common::unordered_map<VertexInputKey, std::unique_ptr<VertexInputLayout>, Hash> layouts;
    std::array<RecentLayout, 2> recent{};
    size_t recent_slot{};
    const VertexInputLayout* recorded_layout{};
    u64 recorded_epoch{};
    u64 recorded_stride_serial{};
    Statistics stats{};
};

} // namespace Vulkan
