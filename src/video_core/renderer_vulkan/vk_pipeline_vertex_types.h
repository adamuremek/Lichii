// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include <boost/container/small_vector.hpp>

#include "common/container/unordered_map.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Vulkan {

// Entries are owned by the graphics cache. Erase them before retiring a pipeline.
template <typename Pipeline>
class VertexTypePipelineIndex {
public:
    void Insert(Pipeline& pipeline) {
        if (pipeline.CanReuseVertexTypes()) {
            groups[pipeline.Key().VertexTypeGroupHash()].push_back(&pipeline);
        }
    }

    void Erase(Pipeline& pipeline) {
        if (!pipeline.CanReuseVertexTypes()) {
            return;
        }
        const auto it = groups.find(pipeline.Key().VertexTypeGroupHash());
        if (it != groups.end()) {
            auto& candidates = it->second;
            const auto position = std::ranges::find(candidates, &pipeline);
            if (position != candidates.end()) {
                candidates.erase(position);
            }
            if (candidates.empty()) {
                groups.erase(it);
            }
        }
    }

    Pipeline* Find(const GraphicsPipelineCacheKey& key) const {
        if (groups.empty() || !key.state.dynamic_vertex_input) {
            return nullptr;
        }
        const auto it = groups.find(key.VertexTypeGroupHash());
        if (it == groups.end()) {
            return nullptr;
        }
        const auto& candidates = it->second;
        const auto match = std::ranges::find_if(candidates, [&](const Pipeline* candidate) {
            return candidate->IsBuilt() && !candidate->Failed() &&
                   candidate->MatchesVertexTypeVariant(key);
        });
        return match == candidates.end() ? nullptr : *match;
    }

    void Clear() {
        groups.clear();
    }

private:
    Common::unordered_map<u64, boost::container::small_vector<Pipeline*, 2>> groups;
};

} // namespace Vulkan
