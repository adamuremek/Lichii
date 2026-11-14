// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

/// EDEN_NX2_PASS_STATS=1: cuts each frame into passes (runs of draws into the same render
/// targets, runs of compute dispatches) and records which earlier pass of the frame wrote each
/// image a pass uses. Every 600 frames it logs how many passes a frame has, how they depend on
/// each other and how long the frame would take with the passes run on 2, 4, 8 or any number of
/// threads, in draws. With GPU profiling or stutter tracing, also models the measured time
/// with cache work, clears and compute kept serial.
class PassStats {
public:
    struct WorkSample {
        u64 start{};
        u64 cache{};
    };

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    [[nodiscard]] WorkSample BeginWork() {
        if (!enabled || !Nx2Profile::g_zones_enabled) {
            return {};
        }
        if (timing_start_ns == 0) {
            timing_start_ns = Nx2Profile::Now();
            timing_start_ticks = Nx2Profile::Ticks();
        }
        const u64 cache = CacheTicks();
        return {Nx2Profile::Ticks(), cache};
    }

    void Read(u32 image) {
        if (enabled) {
            reads.push_back(image);
        }
    }

    void Write(u32 image) {
        if (enabled) {
            writes.push_back(image);
        }
    }

    /// The draw being prepared renders into the targets of `serial`.
    void Targets(u64 serial) {
        if (!enabled) {
            return;
        }
        if (passes.empty() || passes.back().compute || passes.back().serial != serial) {
            passes.push_back(Pass{.serial = serial});
        }
        Flush();
    }

    void Draw(WorkSample sample, bool serial = false) {
        if (enabled && !passes.empty()) {
            ++passes.back().work;
            AccountWork(sample, serial);
        }
    }

    void Compute(WorkSample sample) {
        if (!enabled) {
            return;
        }
        if (passes.empty() || !passes.back().compute) {
            passes.push_back(Pass{.compute = true});
        }
        ++passes.back().work;
        AccountWork(sample, true);
        Flush();
    }

    void EndFrame() {
        if (!enabled) {
            return;
        }
        Flush();
        Account();
        passes.clear();
        writer.clear();
        if (++frames < 600) {
            return;
        }
        const double n = static_cast<double>(frames);
        const auto speedup = [this](size_t index) {
            return total.span[index] != 0
                       ? static_cast<double>(total.work) / static_cast<double>(total.span[index])
                       : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Pass stats per frame: {:.1f} passes ({:.1f} compute), {:.1f} draws and "
                 "dispatches, {:.1f} dependencies between passes, {:.1f} passes with none | "
                 "largest pass {:.1f}% of the frame | frame length in draws: serial {:.0f}, "
                 "2 threads {:.0f} ({:.2f}x), 4 threads {:.0f} ({:.2f}x), 8 threads {:.0f} "
                 "({:.2f}x), unlimited {:.0f} ({:.2f}x)",
                 total.passes / n, total.compute_passes / n, total.work / n, total.edges / n,
                 total.independent / n,
                 total.work != 0 ? 100.0 * static_cast<double>(total.largest) /
                                       static_cast<double>(total.work)
                                 : 0.0,
                 total.work / n, total.span[0] / n, speedup(0), total.span[1] / n, speedup(1),
                 total.span[2] / n, speedup(2), total.span[3] / n, speedup(3));
        if (timing_start_ns != 0) {
            const u64 ticks = Nx2Profile::Ticks() - timing_start_ticks;
            const u64 ns = Nx2Profile::Now() - timing_start_ns;
            const double ms_per_tick_per_frame = static_cast<double>(ns) / ticks / n / 1e6;
            const auto ms = [ms_per_tick_per_frame](u64 value) {
                return static_cast<double>(value) * ms_per_tick_per_frame;
            };
            LOG_INFO(HW_GPU,
                     "Pass timing per frame: {:.2f} ms of draw, clear and dispatch work | "
                     "serial {:.2f} ms (cache {:.2f} ms), remaining graphics {:.2f} ms | "
                     "estimated total: 2 workers {:.2f} ms, 4 workers {:.2f} ms, 8 workers "
                     "{:.2f} ms (image dependencies only; excludes methods, waits outside draws "
                     "and worker overhead)",
                     ms(total.serial_ticks + total.parallel_ticks), ms(total.serial_ticks),
                     ms(total.cache_ticks), ms(total.parallel_ticks),
                     ms(total.serial_ticks + total.timed_span[0]),
                     ms(total.serial_ticks + total.timed_span[1]),
                     ms(total.serial_ticks + total.timed_span[2]));
            timing_start_ns = 0;
        }
        total = {};
        frames = 0;
    }

private:
    struct Pass {
        u64 serial{};
        bool compute{};
        u32 work{};
        u64 parallel_ticks{};
        std::vector<u32> deps;
    };

    struct Totals {
        u64 passes{};
        u64 compute_passes{};
        u64 work{};
        u64 edges{};
        u64 independent{};
        u64 largest{};
        u64 cache_ticks{};
        u64 serial_ticks{};
        u64 parallel_ticks{};
        std::array<u64, 4> span{}; // 2, 4, 8 threads, unlimited
        std::array<u64, 3> timed_span{};
    };

    static u64 CacheTicks() {
        using Nx2Profile::Zone;
        constexpr std::array zones{
            Zone::PipelineLookup, Zone::GraphicsPipelineMiss, Zone::ComputePipelineMiss,
            Zone::ShaderCapture, Zone::SyncDescriptors, Zone::ReadHandles, Zone::FillImageViews,
            Zone::TextureBuffers, Zone::UpdateBuffers, Zone::BindStageBuffers, Zone::RenderTargets,
            Zone::TextureUpload, Zone::TextureCache, Zone::UniformBuffers, Zone::GeometryBuffers,
            Zone::StorageBindings,
        };
        u64 ticks = 0;
        for (const Zone zone : zones) {
            ticks += Nx2Profile::t_data.exclusive_ns[static_cast<size_t>(zone)];
        }
        return ticks;
    }

    void AccountWork(WorkSample sample, bool serial) {
        if (sample.start == 0) {
            return;
        }
        const u64 elapsed = Nx2Profile::Ticks() - sample.start;
        const u64 cache = (std::min)(elapsed, CacheTicks() - sample.cache);
        const u64 parallel = serial ? 0 : elapsed - cache;
        total.cache_ticks += cache;
        total.serial_ticks += elapsed - parallel;
        total.parallel_ticks += parallel;
        passes.back().parallel_ticks += parallel;
    }

    void Depend(Pass& pass, u32 index, u32 image) {
        const auto it = writer.find(image);
        if (it != writer.end() && it->second != index &&
            std::ranges::find(pass.deps, it->second) == pass.deps.end()) {
            pass.deps.push_back(it->second);
        }
    }

    /// Gives the images used since the last draw or dispatch to the current pass.
    void Flush() {
        if (!passes.empty()) {
            const u32 index = static_cast<u32>(passes.size() - 1);
            Pass& pass = passes.back();
            for (const u32 image : reads) {
                Depend(pass, index, image);
            }
            for (const u32 image : writes) {
                Depend(pass, index, image);
                writer[image] = index;
            }
        }
        reads.clear();
        writes.clear();
    }

    void Account() {
        total.passes += passes.size();
        u64 largest = 0;
        for (const Pass& pass : passes) {
            total.compute_passes += pass.compute ? 1 : 0;
            total.work += pass.work;
            total.edges += pass.deps.size();
            total.independent += pass.deps.empty() ? 1 : 0;
            largest = (std::max<u64>)(largest, pass.work);
        }
        total.largest += largest;
        // Passes start in stream order, each on the thread that is free first, once the passes
        // it depends on have finished.
        static constexpr std::array<size_t, 4> Threads{2, 4, 8, 0};
        for (size_t index = 0; index < Threads.size(); ++index) {
            total.span[index] += EstimateSpan(Threads[index], false);
        }
        if (timing_start_ns != 0) {
            for (size_t index = 0; index < total.timed_span.size(); ++index) {
                total.timed_span[index] += EstimateSpan(Threads[index], true);
            }
        }
    }

    u64 EstimateSpan(size_t workers, bool timed) {
        std::array<u64, 8> free_at{};
        finish.resize(passes.size());
        u64 span = 0;
        u64 barrier = 0;
        for (size_t index = 0; index < passes.size(); ++index) {
            const Pass& pass = passes[index];
            if (timed && pass.compute) {
                barrier = span;
            }
            u64 start = barrier;
            for (const u32 dep : pass.deps) {
                start = (std::max)(start, finish[dep]);
            }
            const u64 work = timed ? pass.parallel_ticks : pass.work;
            if (workers != 0 && work != 0) {
                const auto thread = std::min_element(free_at.begin(), free_at.begin() + workers);
                start = (std::max)(start, *thread);
                *thread = start + work;
            }
            finish[index] = start + work;
            span = (std::max)(span, finish[index]);
        }
        return span;
    }

    bool enabled = std::getenv("EDEN_NX2_PASS_STATS") != nullptr;
    std::vector<Pass> passes;
    std::vector<u32> reads;
    std::vector<u32> writes;
    std::vector<u64> finish;
    std::unordered_map<u32, u32> writer; // image -> the frame's last pass that wrote it
    Totals total;
    u64 timing_start_ticks{};
    u64 timing_start_ns{};
    u32 frames{};
};

} // namespace VideoCommon
