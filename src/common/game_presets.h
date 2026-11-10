// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/settings.h"

namespace Settings {

inline constexpr u64 MarioKartWorldProgramId = 0x0400C3F00006E000ULL;
inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

inline void SetMarioKartWorldRecommendedSettings(Values& settings, bool global) {
    const auto session = [global](auto& setting, auto value) {
        setting.SetGlobal(global);
        setting.SetValue(value);
    };
    session(settings.renderer_backend, RendererBackend::Vulkan);
    session(settings.resolution_setup, ResolutionSetup::Res1X);
    session(settings.gpu_accuracy, GpuAccuracy::Low);
    session(settings.use_asynchronous_shaders, true);
    session(settings.astc_recompression, AstcRecompression::Bc3);
    session(settings.vram_usage_mode, VramUsageMode::Conservative);
    session(settings.use_reactive_flushing, false);
    session(settings.vsync_mode, VSyncMode::FifoRelaxed);
    session(settings.use_docked_mode, ConsoleMode::Handheld);
    session(settings.scaling_filter, ScalingFilter::ScaleForce);
    session(settings.anti_aliasing, AntiAliasing::Fxaa);
}

inline bool ApplyMarioKartWorldPreset(Values& settings, u64 program_id, bool accepted) {
    if (!accepted || program_id != MarioKartWorldProgramId) {
        return false;
    }
    SetMarioKartWorldRecommendedSettings(settings, false);
    return true;
}

/// Ocarina of Time meets hundreds of new pipelines on entering an area. Waiting for each one held
/// frames for up to 11 s (Ryzen 9 9900X, AMD driver); Bounded waited 0.1 s for a 152-pipeline
/// burst and kept 32 fps with no visible missing draws. While native_async_draws is still at its
/// Wait default, Ocarina uses Bounded for the session; saved settings are not changed. It only
/// takes effect with asynchronous shaders on.
inline bool ApplyOcarinaOfTimeAsyncDraws(Values& settings, u64 program_id) {
    if (program_id != OcarinaOfTimeProgramId ||
        settings.native_async_draws.GetValue() != NativeAsyncDraws::Wait) {
        return false;
    }
    settings.native_async_draws.SetGlobal(false);
    settings.native_async_draws.SetValue(NativeAsyncDraws::Bounded);
    return true;
}

} // namespace Settings
