// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/vk_vertex_input_cache.h"

namespace {

using Vulkan::VertexInputCache;
using Vulkan::VertexInputKey;
using Vulkan::VertexInputLayout;

TEST_CASE("Vulkan vertex input keys follow formats, instancing, strides and repacking",
          "[video_core][vulkan][vertex_input_cache]") {
    auto regs = std::make_unique<VertexInputKey::Regs>();
    const auto capture = [&] { return VertexInputKey::Capture(*regs, 16, 16, 0); };
    const auto base = capture();
    auto& stream = regs->vertex_streams[3];
    stream.address_low = 0x1234;
    stream.frequency = 5;
    regs->vertex_attrib_format[17].hex = 0x1000;
    CHECK(capture() == base);

    stream.stride.Assign(69);
    const auto with_stride = capture();
    CHECK(with_stride != base);
    CHECK(with_stride.strides[3] == 69);
    CHECK(VertexInputKey::Capture(*regs, 16, 16, 1u << 17) == with_stride);
    CHECK(VertexInputKey::Capture(*regs, 16, 16, 1u << 3) != with_stride);

    regs->vertex_stream_instances.is_instanced[3] = 1;
    const auto instanced = capture();
    CHECK(instanced != with_stride);
    CHECK(instanced.instanced == 1u << 3);
    CHECK(instanced.divisors[3] == 5);
    stream.frequency = 0;
    CHECK(capture() != instanced);
    regs->vertex_attrib_format[3].hex = 0x1000;
    CHECK(capture().attributes[3] == 0x1000);
}

TEST_CASE("Vulkan vertex layouts survive cache growth and record again after invalidation",
          "[video_core][vulkan][vertex_input_cache]") {
    VertexInputCache cache;
    u32 builds = 0;
    auto build = [&](VertexInputLayout& layout) {
        layout.bindings.push_back({.binding = ++builds});
    };
    VertexInputKey key{};
    const auto* first = cache.FindOrBuild(key, build);
    REQUIRE(first);
    CHECK(first->bindings.front().binding == 1);
    CHECK(cache.NeedsRecording(first, 1, 0));
    CHECK_FALSE(cache.NeedsRecording(first, 1, 0));
    CHECK(cache.NeedsRecording(first, 2, 0));
    CHECK(cache.NeedsRecording(first, 2, 1));
    cache.InvalidateRecording();
    CHECK(cache.NeedsRecording(first, 2, 1));

    key.strides[0] = 1;
    const auto* second = cache.FindOrBuild(key, build);
    REQUIRE(second);
    CHECK(second != first);
    CHECK(cache.NeedsRecording(second, 2, 1));
    CHECK(cache.NeedsRecording(first, 2, 1));
    CHECK(cache.FindOrBuild(key, build) == second);
    for (u16 i = 2; i < VertexInputCache::MaxLayouts; ++i) {
        key.strides[0] = i;
        REQUIRE(cache.FindOrBuild(key, build));
    }
    key.strides[0] = VertexInputCache::MaxLayouts;
    CHECK(cache.FindOrBuild(key, build) == nullptr);
    key.strides[0] = 0;
    CHECK(cache.FindOrBuild(key, build) == first);
    CHECK(first->bindings.front().binding == 1);
    CHECK(builds == VertexInputCache::MaxLayouts);
    const auto stats = cache.TakeStatistics();
    CHECK(stats.built == VertexInputCache::MaxLayouts);
    CHECK(stats.reused == 2);
    CHECK(stats.redundant_commands == 1);
    CHECK(stats.uncached == 1);
    CHECK(cache.TakeStatistics().built == 0);
}

} // namespace
