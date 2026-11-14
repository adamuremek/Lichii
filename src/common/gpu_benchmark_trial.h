// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>

#include "common/common_types.h"

namespace Common::GpuBenchmarkTrial {

enum class Experiment {
    None,
    AdaptiveSubmit,
    Renderer,
    CpuPlacement,
    TextureReads,
    BufferReads,
    AcquireWaits,
};

enum class Mode : u32 {
    Restore,
    Baseline,
    AdaptiveSubmit,
    RasterStateReuse,
    Both,
    CacheLocal,
    TextureReadFilter,
    BufferReadFilter,
    PreciseAcquireWait,
};

struct Policies {
    bool adaptive_submit;
    bool raster_state_reuse;
};

constexpr Policies ResolvePolicies(Experiment experiment, Mode mode, Policies configured) {
    if (mode == Mode::Restore || experiment == Experiment::CpuPlacement) {
        return configured;
    }
    return {
        .adaptive_submit = mode == Mode::AdaptiveSubmit || mode == Mode::Both,
        .raster_state_reuse = experiment == Experiment::Renderer
                                  ? mode == Mode::RasterStateReuse || mode == Mode::Both
                                  : configured.raster_state_reuse,
    };
}

inline Experiment ConfiguredExperiment() {
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value && std::string_view{value} == "1";
    };
    if (enabled("EDEN_NX2_TAS_RENDERER_AB")) {
        return Experiment::Renderer;
    }
    if (enabled("EDEN_NX2_TAS_CPU_PLACEMENT_AB")) {
        return Experiment::CpuPlacement;
    }
    if (enabled("EDEN_NX2_TAS_TEXTURE_READS_AB")) {
        return Experiment::TextureReads;
    }
    if (enabled("EDEN_NX2_TAS_BUFFER_READS_AB")) {
        return Experiment::BufferReads;
    }
    if (enabled("EDEN_NX2_TAS_ACQUIRE_WAITS_AB")) {
        return Experiment::AcquireWaits;
    }
    return enabled("EDEN_NX2_TAS_ADAPTIVE_SUBMIT_AB") ? Experiment::AdaptiveSubmit
                                                     : Experiment::None;
}

constexpr bool ResolveToggle(Mode mode, Mode candidate, bool configured) {
    return mode == Mode::Restore ? configured : mode == candidate;
}

inline constexpr u32 ModeMask = 15;
static_assert(static_cast<u32>(Mode::PreciseAcquireWait) <= ModeMask);
inline std::atomic<u32> requested{};
inline std::atomic<u32> applied{};
inline std::atomic<u64> rendered_frames{};
inline std::atomic<u64> read_checks_skipped{};
inline std::atomic<u64> acquire_waits_shortened{};

inline void NoteReadCheckSkipped() noexcept {
    read_checks_skipped.fetch_add(1, std::memory_order_relaxed);
}

inline u64 ReadChecksSkipped() noexcept {
    return read_checks_skipped.load(std::memory_order_relaxed);
}

inline void NoteRenderedFrame() noexcept {
    rendered_frames.fetch_add(1, std::memory_order_relaxed);
}

inline u64 RenderedFrames() noexcept {
    return rendered_frames.load(std::memory_order_relaxed);
}

inline u32 Request(Mode mode) noexcept {
    u32 previous = requested.load(std::memory_order_relaxed);
    u32 ticket;
    do {
        ticket = ((previous & ~ModeMask) + ModeMask + 1) | static_cast<u32>(mode);
    } while (!requested.compare_exchange_weak(previous, ticket, std::memory_order_release,
                                              std::memory_order_relaxed));
    return ticket;
}

inline u32 Requested() noexcept {
    return requested.load(std::memory_order_acquire);
}

constexpr Mode GetMode(u32 ticket) noexcept {
    return static_cast<Mode>(ticket & ModeMask);
}

inline void Acknowledge(u32 ticket) noexcept {
    applied.store(ticket, std::memory_order_release);
}

inline bool Applied(u32 ticket) noexcept {
    return applied.load(std::memory_order_acquire) == ticket;
}

} // namespace Common::GpuBenchmarkTrial
