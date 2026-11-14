// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "shader_recompiler/shader_info.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_vertex_types.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace FastDraw = VideoCommon::Nx2FastDraw;
constexpr u32 ViewportScaleX = MAXWELL3D_REG_INDEX(viewport_transform);

class NoDma final : public Tegra::Engines::AccelerateDMAInterface {
public:
    bool BufferCopy(GPUVAddr, GPUVAddr, u64) override { return true; }
    bool BufferClear(GPUVAddr, u64, u32) override { return true; }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy&, const Tegra::DMA::ImageOperand&,
                       const Tegra::DMA::BufferOperand&) override { return true; }
    bool BufferToImage(const Tegra::DMA::ImageCopy&, const Tegra::DMA::BufferOperand&,
                       const Tegra::DMA::ImageOperand&) override { return true; }
};

/// Records the fast draw serial each draw sees.
class SerialRasterizer final : public VideoCore::RasterizerInterface {
public:
    Maxwell3D* engine{};
    std::vector<u64> draws;

    void Draw(bool, u32) override { draws.push_back(engine->ObserveFastDrawSerial()); }
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&& func) override { func(); }
    void SyncOperation(std::function<void()>&& func) override { func(); }
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override { return false; }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(PAddr, u64) override {}
    bool OnCPUWrite(PAddr, u64) override { return false; }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override { return dma; }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    NoDma dma;
};

struct FastDrawFixture {
    FastDrawFixture() : memory{system, device_memory, 32, 1ULL << 31}, maxwell3d{memory} {
        SetSwitch(true);
        FastDraw::g_raster_state_reuse = {};
        FastDraw::g_delta = {};
        rasterizer.engine = &maxwell3d;
        maxwell3d.BindRasterizer(&rasterizer);
    }
    ~FastDrawFixture() {
        FastDraw::SetEnabledForTest(previous);
        FastDraw::g_track_full_uniforms.store(previous_full_uniform_tracking,
                                             std::memory_order_relaxed);
        FastDraw::g_raster_state_reuse = previous_raster_state;
        FastDraw::g_delta = {};
    }

    static void SetSwitch(bool on) {
        FastDraw::SetEnabledForTest(on);
    }

    u32 Reg(u32 method) const {
        return maxwell3d.regs.reg_array[method];
    }

    void Write(u32 method, u32 value) {
        maxwell3d.CallMethod(system, method, value, true);
    }

    void DrawArrays(u32 first, u32 count) {
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.first), first);
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.count), count);
        Write(MAXWELL3D_REG_INDEX(draw.begin), 0x4); // first instance, triangles
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    /// Whether the last two draws saw the same serial (the second could reuse the first).
    bool Reusable() const {
        const auto& draws = rasterizer.draws;
        REQUIRE(draws.size() >= 2);
        return draws[draws.size() - 1] == draws[draws.size() - 2];
    }

    const bool previous = FastDraw::Enabled();
    const bool previous_full_uniform_tracking =
        FastDraw::g_track_full_uniforms.load(std::memory_order_relaxed);
    const FastDraw::RasterStateReuse previous_raster_state = FastDraw::g_raster_state_reuse;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory;
    SerialRasterizer rasterizer;
    Maxwell3D maxwell3d;
};

} // namespace

TEST_CASE("Fast draw keeps the serial across draws that change only draw parameters",
          "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.DrawArrays(3, 6);
    REQUIRE(f.rasterizer.draws.size() == 2);
    CHECK(f.Reusable());

    f.Write(ViewportScaleX, f.Reg(ViewportScaleX));
    f.DrawArrays(9, 3);
    CHECK(f.Reusable());

    f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), 0x40);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), 0x1000);
    f.DrawArrays(12, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw breaks reuse on anything that changes what a draw binds", "[fast_draw]") {
    FastDrawFixture f;
    const auto breaks = [&](auto&& change) {
        f.DrawArrays(0, 3);
        change();
        f.DrawArrays(0, 3);
        return !f.Reusable();
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.Write(ViewportScaleX,
                f.Reg(ViewportScaleX) + 1);
    }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(clear_surface), 0x3c); }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(tiled_cache_barrier), 0); }));
    CHECK(breaks([&] { FastDraw::Break(FastDraw::Reason::Fence); }));
}

TEST_CASE("Full draw uniform serial tracks effects apart from pure register writes", "[fast_draw]") {
    FastDrawFixture f;
    FastDraw::g_track_full_uniforms.store(true, std::memory_order_relaxed);
    const u64 uniform_serial = FastDraw::UniformSerial();
    const u64 draw_serial = FastDraw::Serial();
    f.Write(ViewportScaleX, f.Reg(ViewportScaleX) ^ 1);
    CHECK(FastDraw::Serial() == draw_serial + 1);
    CHECK(FastDraw::UniformSerial() == uniform_serial);
    FastDraw::Break(FastDraw::Reason::Map);
    FastDraw::Break(FastDraw::Reason::OtherEngine);
    CHECK(FastDraw::UniformSerial() == uniform_serial + 2);
}

TEST_CASE("Shader register counts preserve draw reuse while program changes break it",
          "[fast_draw]") {
    FastDrawFixture f;
    VideoCommon::Dirty::SetupDirtyFlags(f.maxwell3d.dirty.tables);
    auto& shader_dirty = f.maxwell3d.dirty.flags;
    constexpr u32 vertex_count = 0x813;
    constexpr u32 pixel_count = 0x853;
    f.DrawArrays(0, 3);
    shader_dirty.reset(VideoCommon::Dirty::Shaders);
    f.Write(vertex_count, 12);
    f.Write(pixel_count, 20);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
    CHECK_FALSE(shader_dirty[VideoCommon::Dirty::Shaders]);
    CHECK(f.Reg(vertex_count) == 12);
    CHECK(f.Reg(pixel_count) == 20);

    constexpr u32 vertex_enable = 0x810;
    constexpr u32 vertex_offset = 0x811;
    constexpr u32 pixel_address_high = 0x855;
    constexpr u32 pixel_address_low = 0x856;
    for (const u32 method : {vertex_enable, vertex_offset, pixel_address_high, pixel_address_low,
                             u32{MAXWELL3D_REG_INDEX(program_region.address_low)},
                             u32{MAXWELL3D_REG_INDEX(rasterize_enable)}}) {
        CAPTURE(method);
        shader_dirty.reset(VideoCommon::Dirty::Shaders);
        f.Write(method, f.Reg(method) ^ 1);
        f.Write(pixel_count, f.Reg(pixel_count) + 1);
        f.DrawArrays(0, 3);
        CHECK_FALSE(f.Reusable());
        CHECK(shader_dirty[VideoCommon::Dirty::Shaders]);
    }
}

TEST_CASE("Dynamic vertex types can differ only in attributes the shader cannot read",
          "[fast_draw]") {
    using Vulkan::FixedPipelineState;
    using Shader::IR::Attribute;
    Shader::Info vertex{};
    vertex.loads.Set(Attribute::Generic0X);
    vertex.loads.Set(Attribute::Generic31W);
    vertex.passthrough.Set(Attribute::Generic7Y);
    const u64 mask = FixedPipelineState::UsedAttributeTypeMask(vertex);
    CHECK(mask == (u64{3} | (u64{3} << 62) | (u64{3} << 14)));

    FixedPipelineState first{};
    first.dynamic_vertex_input.Assign(1);
    first.attribute_types = 1 | (u64{2} << 62);
    auto second = first;
    second.attribute_types |= u64{3} << 12;
    CHECK(first != second);
    CHECK(first.MatchesWithAttributeTypeMask(second, mask));
    for (u32 slot : {0, 7, 31}) {
        const u64 type_bit = u64{1} << (2 * slot);
        second.attribute_types ^= type_bit;
        CHECK_FALSE(first.MatchesWithAttributeTypeMask(second, mask));
        second.attribute_types ^= type_bit;
    }
    second.attachments[2].raw = 1;
    CHECK_FALSE(first.MatchesWithAttributeTypeMask(second, mask));
    second.attachments[2].raw = 0;
    second.dynamic_state.raw1 = 1;
    CHECK_FALSE(first.MatchesWithAttributeTypeMask(second, mask));
    second.dynamic_state.raw1 = 0;
    second.dynamic_vertex_input.Assign(0);
    CHECK_FALSE(first.MatchesWithAttributeTypeMask(second, mask));

    vertex.loads_indexed_attributes = true;
    CHECK(FixedPipelineState::UsedAttributeTypeMask(vertex) == ~u64{0});
    vertex.loads_indexed_attributes = false;
    vertex.loads.Set(Attribute::FixedFncTexture0S);
    CHECK(FixedPipelineState::UsedAttributeTypeMask(vertex) == ~u64{0});
}

TEST_CASE("Vertex type cache keeps built variants through pipeline switches and eviction",
          "[fast_draw]") {
    using Vulkan::GraphicsPipelineCacheKey;
    struct Pipeline {
        GraphicsPipelineCacheKey key{};
        u64 mask{3};
        bool built{true};
        bool failed{};

        const GraphicsPipelineCacheKey& Key() const { return key; }
        bool CanReuseVertexTypes() const { return mask != ~u64{0}; }
        bool IsBuilt() const { return built; }
        bool Failed() const { return failed; }
        bool MatchesVertexTypeVariant(const GraphicsPipelineCacheKey& other) const {
            return key.unique_hashes == other.unique_hashes &&
                   key.state.MatchesWithAttributeTypeMask(other.state, mask);
        }
    };

    Pipeline first{};
    first.key.unique_hashes[1] = 1;
    first.key.state.dynamic_vertex_input.Assign(1);
    first.key.state.attribute_types = 1;
    auto switched = first;
    switched.key.unique_hashes[5] = 2;
    auto different_type = first;
    different_type.key.state.attribute_types = 2;
    auto fallback = first;
    fallback.mask = ~u64{0};
    fallback.key.unique_hashes[5] = 3;
    auto requested = first.key;
    requested.state.attribute_types |= u64{3} << 8;
    CHECK(first.key.VertexTypeGroupHash() == requested.VertexTypeGroupHash());

    Vulkan::VertexTypePipelineIndex<Pipeline> index;
    index.Insert(first);
    index.Insert(switched);
    index.Insert(different_type);
    index.Insert(fallback);
    CHECK(index.Find(switched.key) == &switched);
    CHECK(index.Find(requested) == &first);
    first.built = false;
    CHECK(index.Find(requested) == nullptr);
    first.built = true;
    first.failed = true;
    CHECK(index.Find(requested) == nullptr);
    first.failed = false;
    CHECK(index.Find(fallback.key) == nullptr);
    auto different_state = requested;
    different_state.state.dynamic_state.raw1 ^= 1;
    CHECK(index.Find(different_state) == nullptr);
    index.Erase(first);
    CHECK(index.Find(requested) == nullptr);
    CHECK(index.Find(different_type.key) == &different_type);
    index.Clear();
    CHECK(index.Find(different_type.key) == nullptr);
}

TEST_CASE("Fast draw reuses only raster state that leaves the pipeline unchanged", "[fast_draw]") {
    FastDrawFixture f;
    constexpr u32 viewport = MAXWELL3D_REG_INDEX(viewports);
    using Transform = Maxwell3D::Regs::ViewportTransform;
    constexpr u32 last_transform = MAXWELL3D_REG_INDEX(viewport_transform) +
                                   (Maxwell3D::Regs::NumViewports - 1) *
                                       sizeof(Transform) / sizeof(u32);
    constexpr u32 scale = last_transform + offsetof(Transform, scale_y) / sizeof(u32);
    constexpr u32 swizzle = last_transform + offsetof(Transform, swizzle) / sizeof(u32);
    constexpr u32 shader_viewport = MAXWELL3D_REG_INDEX(viewport_scale_offset_enabled);
    constexpr u32 cull = MAXWELL3D_REG_INDEX(gl_cull_face);
    constexpr u32 clip = MAXWELL3D_REG_INDEX(viewport_clip_control);
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[2]);
    const auto change = [&](u32 method, u32 value) {
        f.Write(method, value);
        f.DrawArrays(0, 3);
        CHECK(f.Reg(method) == value);
        return f.Reusable();
    };

    f.DrawArrays(0, 3);
    CHECK_FALSE(change(viewport, f.Reg(viewport) ^ 1));
    FastDraw::g_raster_state_reuse.enabled = true;
    CHECK(change(viewport, f.Reg(viewport) ^ 1));
    CHECK(change(scale, f.Reg(scale) ^ 1));
    CHECK_FALSE(change(swizzle, f.Reg(swizzle) ^ 1));
    CHECK_FALSE(change(shader_viewport, f.Reg(shader_viewport) ^ 1));
    CHECK(change(blend, f.Reg(blend) ^ 1));
    constexpr u32 constant_color = MAXWELL3D_REG_INDEX(const_color_rendering.enabled);
    constexpr u32 color_mask = MAXWELL3D_REG_INDEX(color_mask) + 2;
    constexpr u32 equation = MAXWELL3D_REG_INDEX(blend_per_target) +
                             offsetof(Maxwell3D::Regs::BlendPerTarget, color_dest) / sizeof(u32);
    constexpr u32 per_target = MAXWELL3D_REG_INDEX(blend_per_target_enabled);
    CHECK(change(constant_color, f.Reg(constant_color) ^ 1));
    CHECK(change(color_mask, f.Reg(color_mask) ^ 1));
    CHECK(change(equation, f.Reg(equation) ^ 1));
    constexpr auto format = static_cast<u32>(Tegra::RenderTargetFormat::A8B8G8R8_UNORM);
    using Target = Maxwell3D::Regs::RenderTargetConfig;
    constexpr u32 target_format = MAXWELL3D_REG_INDEX(rt) +
                                  (3 * sizeof(Target) + offsetof(Target, format)) / sizeof(u32);
    CHECK_FALSE(change(target_format, format));
    CHECK_FALSE(change(blend, f.Reg(blend) ^ 1)); // A hole before the last attachment.
    CHECK_FALSE(change(color_mask, f.Reg(color_mask) ^ 1));
    constexpr u32 common_mask = MAXWELL3D_REG_INDEX(color_mask_common);
    if (f.Reg(common_mask) == 0) {
        CHECK_FALSE(change(common_mask, 1));
    }
    CHECK(change(color_mask, f.Reg(color_mask) ^ 1));
    CHECK_FALSE(change(target_format, 0));
    CHECK(change(blend, f.Reg(blend) ^ 1));
    constexpr u32 blend_zero = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK_FALSE(change(blend_zero, f.Reg(blend_zero) ^ 1));
    CHECK_FALSE(change(per_target, f.Reg(per_target) ^ 1));
    CHECK_FALSE(change(equation, f.Reg(equation) ^ 1));
    CHECK_FALSE(change(cull, f.Reg(cull) ^ 1));
    FastDraw::g_raster_state_reuse.cull_dynamic = true;
    CHECK(change(cull, f.Reg(cull) ^ 1));

    f.Write(clip, 1u << 11); // Passthrough
    f.DrawArrays(0, 3);
    CHECK(change(clip, 3u << 11)); // FrustumXYZ also disables depth clamp.
    CHECK_FALSE(change(clip, 2u << 11)); // FrustumXY enables depth clamp.
    FastDraw::g_raster_state_reuse.depth_clamp_dynamic = true;
    CHECK(change(clip, 5u << 11));
}

TEST_CASE("Vulkan pipeline reuse ignores blend state after its last attachment", "[fast_draw]") {
    FastDrawFixture fixture;
    Vulkan::FixedPipelineState original{};
    original.color_formats[0] = static_cast<u8>(Tegra::RenderTargetFormat::A8B8G8R8_UNORM);
    original.color_formats[2] = original.color_formats[0];
    REQUIRE(original.ColorAttachmentCount() == 3);
    auto next = original;
    next.attachments[3].enable.Assign(1);
    CHECK_FALSE(original == next);
    CHECK(original.MatchesWithoutTrailingBlend(next));
    next.attachments[1].enable.Assign(1);
    CHECK_FALSE(original.MatchesWithoutTrailingBlend(next));
    next.attachments[1].enable.Assign(0);
    next.attributes[0].raw ^= 1;
    CHECK_FALSE(original.MatchesWithoutTrailingBlend(next));
    next.attributes[0].raw ^= 1;
    next.attachment0_dual_source_blend.Assign(1);
    CHECK_FALSE(original.MatchesWithoutTrailingBlend(next));

    original.dynamic_vertex_input.Assign(1);
    next = original;
    next.attachments[3].enable.Assign(1);
    next.attribute_types ^= 3;
    CHECK_FALSE(original.MatchesWithAttributeTypeMask(next, ~u64{3}));
    FastDraw::g_raster_state_reuse.enabled = true;
    CHECK(original.MatchesWithAttributeTypeMask(next, ~u64{3}));
    CHECK_FALSE(original.MatchesWithAttributeTypeMask(next, ~u64{0}));
}

TEST_CASE("Fast draw serial never moves while disabled", "[fast_draw]") {
    FastDrawFixture f;
    FastDrawFixture::SetSwitch(false);
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(blend.enable[0]), f.Reg(MAXWELL3D_REG_INDEX(blend.enable[0])) ^ 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw publishes one register break at each draw boundary", "[fast_draw]") {
    FastDrawFixture f;
    f.maxwell3d.SetDrawRegisterCoalescing(true);
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    constexpr u32 cull = MAXWELL3D_REG_INDEX(gl_cull_face);
    f.DrawArrays(0, 3);
    const u64 before = FastDraw::Serial();
    f.Write(blend, f.Reg(blend) ^ 1);
    f.Write(cull, f.Reg(cull) ^ 1);
    CHECK(FastDraw::Serial() == before);
    f.DrawArrays(0, 3);
    CHECK_FALSE(f.Reusable());
    CHECK(FastDraw::Serial() == before + 1);
    CHECK((FastDraw::g_last_break.load() >> 8) == blend);

    f.DrawArrays(1, 3);
    CHECK(f.Reusable());
    FastDraw::Break(FastDraw::Reason::Invalidation);
    f.DrawArrays(2, 3);
    CHECK_FALSE(f.Reusable());
    f.Write(cull, f.Reg(cull) ^ 1);
    f.DrawArrays(0, 3);
    CHECK_FALSE(f.Reusable());
}

namespace {

struct DeltaFixture : FastDrawFixture {
    void SetConstantBuffer(GPUVAddr address, u32 size, u32 offset) {
        Write(MAXWELL3D_REG_INDEX(const_buffer.size), size);
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_high), static_cast<u32>(address >> 32));
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), static_cast<u32>(address));
        Write(MAXWELL3D_REG_INDEX(const_buffer.offset), offset);
    }
};

} // namespace

TEST_CASE("Fast draw records bind and constant data changes, keeps index buffer changes light",
          "[fast_draw]") {
    DeltaFixture f;
    f.DrawArrays(0, 3);
    const u64 serial = FastDraw::Serial();
    FastDraw::g_delta = {};

    f.Write(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low),
            f.Reg(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low)) + 0x100);
    f.Write(MAXWELL3D_REG_INDEX(index_buffer.format), f.Reg(MAXWELL3D_REG_INDEX(index_buffer.format)) ^ 1);
    CHECK(FastDraw::Serial() == serial);
    CHECK(FastDraw::g_delta.Empty());

    f.SetConstantBuffer(0x10000, 0x100, 0);
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31); // slot 3, valid
    CHECK(FastDraw::g_delta.binds[4] == 1U << 3);
    FastDraw::g_delta.binds[4] = 0;
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31);
    CHECK(FastDraw::g_delta.binds[4] == 0);

    f.SetConstantBuffer(0x20000, 0x100, 0x10);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1, 2);
    REQUIRE(FastDraw::g_delta.num_writes == 1);
    CHECK(FastDraw::g_delta.write_begin[0] == 0x20010);
    CHECK(FastDraw::g_delta.write_end[0] == 0x20018);

    CHECK(FastDraw::Serial() == serial);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
    CHECK_FALSE(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw breaks on state and texture headers", "[fast_draw]") {
    DeltaFixture f;
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), 0x40000);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_low), 0x50000);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.limit), 7);
    const auto breaks = [&](auto&& change) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        change();
        return FastDraw::Serial() != serial;
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0xe0); // the last texture header
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x50000, 0x1000, 0);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK_FALSE(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0x100); // just past the headers
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    // More separate writes than the delta holds widen its ranges instead of ending draw reuse.
    CHECK_FALSE(breaks([&] {
        f.SetConstantBuffer(0x60000, 0x1000, 0);
        for (u32 i = 0; i <= FastDraw::Delta::MaxWrites; ++i) {
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), i * 0x20);
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), i);
        }
    }));
    const auto& delta = FastDraw::g_delta;
    REQUIRE(delta.num_writes == FastDraw::Delta::MaxWrites);
    CHECK(delta.no_uniform_keeps);
    for (u32 i = 0; i <= FastDraw::Delta::MaxWrites; ++i) {
        const GPUVAddr address = 0x60000 + i * 0x20;
        bool covered = false;
        for (u32 write = 0; write < delta.num_writes; ++write) {
            covered |= delta.write_begin[write] <= address && address + 4 <= delta.write_end[write];
        }
        CHECK(covered);
    }
}

TEST_CASE("Fast draw breaks on texture header data written through another mapping",
          "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr headers = 0x40000;
    constexpr GPUVAddr alias = 0x90000;
    f.memory.Map(headers, 0x100000, 0x1000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, 0x100000, 0x1000, Tegra::PTEKind::PITCH, false);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), static_cast<u32>(headers));
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    const auto breaks = [&](u32 offset) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.SetConstantBuffer(alias, 0x1000, offset);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
        return FastDraw::Serial() != serial;
    };
    CHECK(breaks(0xe0));        // the last texture header
    CHECK_FALSE(breaks(0x100)); // just past the headers
}

TEST_CASE("Fast draw finds texture handle data written through another mapping",
          "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr handles = 0x100000; // a buffer of texture handles, 0x100 bytes
    constexpr GPUVAddr alias = 0x300000;   // its device memory, mapped again
    constexpr GPUVAddr other = 0x500000;   // other device memory
    constexpr GPUVAddr unmapped = 0x700000;
    constexpr auto Kind = Tegra::PTEKind::PITCH;
    f.memory.Map(handles, 0x200000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x200000, 0x1000, Kind, false);
    f.memory.Map(other, 0x400000, 0x1000, Kind, false);
    FastDraw::DeviceSpans spans;
    const auto reaches = [&](GPUVAddr handle_buffer, GPUVAddr buffer, u32 offset,
                             std::vector<u32> words) {
        FastDraw::g_delta = {};
        f.SetConstantBuffer(buffer, 0x2000, offset);
        f.maxwell3d.CallMultiMethod(f.system, MAXWELL3D_REG_INDEX(const_buffer.buffer),
                                    words.data(), static_cast<u32>(words.size()),
                                    static_cast<u32>(words.size()));
        spans.Update(f.memory, handle_buffer, 0x100);
        return FastDraw::g_delta.Reaches(handle_buffer, 0x100, spans);
    };
    CHECK(reaches(handles, handles, 0x10, {1}));
    CHECK(reaches(handles, alias, 0x10, {1})); // missed when compared by GPU address only
    CHECK(reaches(handles, alias, 0xfc, {1, 2}));
    CHECK_FALSE(reaches(handles, alias, 0x100, {1})); // just past the handles
    CHECK_FALSE(reaches(handles, other, 0x10, {1}));
    // Nothing mapped there: compared by GPU address, as before.
    CHECK(reaches(unmapped, unmapped, 0x10, {1}));
    CHECK_FALSE(reaches(unmapped, other, 0x10, {1}));

    // A write across two pages that map device memory apart lands on both.
    f.memory.Map(alias, 0x201000, 0x1000, Kind, false);
    f.memory.Map(alias + 0x1000, 0x200000, 0x1000, Kind, false);
    CHECK(reaches(handles, alias, 0xffc, {1, 2}));
    CHECK(FastDraw::g_delta.num_device_writes == 2);
    CHECK_FALSE(reaches(handles, alias, 0xff8, {1, 2}));

    // Mapping changes translate the handle buffer again.
    CHECK_FALSE(reaches(handles, other, 0x10, {1}));
    f.memory.Map(handles, 0x400000, 0x1000, Kind, false);
    CHECK(reaches(handles, other, 0x10, {1}));
    CHECK_FALSE(reaches(handles, alias + 0x1000, 0x10, {1}));
}

TEST_CASE("Fast draw compares texture handle data word by word", "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr handles = 0x100000; // handles at 0x10 and 0x80, draw data between
    constexpr GPUVAddr alias = 0x300000;
    constexpr GPUVAddr split = 0x500000; // two pages that map device memory apart
    constexpr auto Kind = Tegra::PTEKind::PITCH;
    f.memory.Map(handles, 0x200000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x200000, 0x1000, Kind, false);
    const std::array<std::pair<u32, u32>, 2> words{{{0x10, 0x14}, {0x80, 0x84}}};
    FastDraw::DeviceSpans spans;
    const auto reaches = [&](GPUVAddr handle_buffer, GPUVAddr buffer, u32 offset) {
        FastDraw::g_delta = {};
        f.SetConstantBuffer(buffer, 0x2000, offset);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
        spans.Update(f.memory, handle_buffer + 0x10, 0x74);
        return FastDraw::g_delta.Reaches(handle_buffer + 0x10, 0x74, spans) &&
               FastDraw::g_delta.ReachesWords(handle_buffer, words, spans);
    };
    CHECK(reaches(handles, handles, 0x10));
    CHECK(reaches(handles, handles, 0x80));
    CHECK_FALSE(reaches(handles, handles, 0x40)); // draw data between the handles
    CHECK(reaches(handles, alias, 0x80));
    CHECK_FALSE(reaches(handles, alias, 0x40));

    // Not one run of device memory: a device write anywhere in the range counts.
    f.memory.Map(split, 0x401000, 0x1000, Kind, false);
    f.memory.Map(split + 0x1000, 0x400000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x401000, 0x1000, Kind, false);
    CHECK(reaches(split + 0xf80, alias, 0xfc0));
}

TEST_CASE("Fast draw records and breaks nothing while switched off", "[fast_draw]") {
    DeltaFixture f;
    FastDrawFixture::SetSwitch(false);
    const u64 serial = FastDraw::Serial();
    f.Write(MAXWELL3D_REG_INDEX(blend.enable[0]), f.Reg(MAXWELL3D_REG_INDEX(blend.enable[0])) ^ 1);
    f.SetConstantBuffer(0x10000, 0x100, 0);
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config), 0x11);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    FastDraw::Break(FastDraw::Reason::Fence);
    CHECK(FastDraw::Serial() == serial);
    CHECK(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw keeps base vertex and instance as draw parameters", "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(global_base_vertex_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_vertex_index)) + 64);
    f.Write(MAXWELL3D_REG_INDEX(global_base_instance_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_instance_index)) + 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw keeps counter resets and the render enable condition light", "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    constexpr u32 clear_report = MAXWELL3D_REG_INDEX(clear_report_value);
    f.Write(clear_report, f.Reg(clear_report) ^ 1);
    f.DrawArrays(3, 3);
    CHECK(f.Reusable());
    f.Write(MAXWELL3D_REG_INDEX(render_enable.mode), 1); // True: the next draws run
    f.DrawArrays(6, 3);
    CHECK(f.rasterizer.draws.size() == 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw records a report's memory write instead of breaking on it", "[fast_draw]") {
    DeltaFixture f;
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), 0x40000);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    const auto report = [&](u32 address) {
        f.Write(MAXWELL3D_REG_INDEX(report_semaphore.address_high), 0);
        f.Write(MAXWELL3D_REG_INDEX(report_semaphore.address_low), address);
        f.Write(MAXWELL3D_REG_INDEX(report_semaphore.payload),
                f.Reg(MAXWELL3D_REG_INDEX(report_semaphore.payload)) + 1);
        f.Write(MAXWELL3D_REG_INDEX(report_semaphore.query), 0); // long release
    };
    f.DrawArrays(0, 3);
    const u64 serial = FastDraw::Serial();
    FastDraw::g_delta = {};
    f.Write(0x155, f.Reg(0x155) + 0x40); // SET_MME_MEM_ADDRESS_B
    report(0x30000);
    CHECK(FastDraw::Serial() == serial);
    REQUIRE(FastDraw::g_delta.num_writes == 1);
    CHECK(FastDraw::g_delta.write_begin[0] == 0x30000);
    CHECK(FastDraw::g_delta.write_end[0] == 0x30010);
    f.DrawArrays(3, 3);
    CHECK(f.Reusable());

    // Into the texture headers a delta draw keeps: a full resolve.
    report(0x40020);
    CHECK(FastDraw::Serial() != serial);
}

TEST_CASE("Fast draw keeps the vertex id base as a draw parameter", "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(vertex_id_base), f.Reg(MAXWELL3D_REG_INDEX(vertex_id_base)) + 3);
    f.DrawArrays(3, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw keeps vertex stream addresses and sizes light", "[fast_draw]") {
    DeltaFixture f;
    const auto stream = [](u32 index, u32 word) {
        return static_cast<u32>(MAXWELL3D_REG_INDEX(vertex_streams) + index * 4 + word);
    };
    const auto light = [&](u32 method) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.Write(method, f.Reg(method) + 0x40);
        return FastDraw::Serial() == serial;
    };
    CHECK(light(stream(0, 1)));
    CHECK(light(stream(3, 2)));
    CHECK(light(0x600 / 4 + 1));          // SET_VERTEX_STREAM_SIZE_B(0)
    CHECK(light(0x600 / 4 + 31 * 2));     // SET_VERTEX_STREAM_SIZE_A(31)
    CHECK(light(MAXWELL3D_REG_INDEX(vertex_stream_limits) + 5 * 2 + 1));
    CHECK_FALSE(light(stream(0, 0)));     // stride and enable
    CHECK_FALSE(light(stream(0, 3)));     // frequency
    CHECK_FALSE(light(0x600 / 4 + 64));   // past the sizes
}
