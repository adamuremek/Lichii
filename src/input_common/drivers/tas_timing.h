// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "common/core_pinning.h"
#include "common/gpu_benchmark_trial.h"
#include "common/logging.h"

namespace InputCommon::TasInput {

class PlaybackTiming {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr u64 Near120Ns = 100'000'000'000ULL / (120 * 97);

    void Start(Clock::time_point now, size_t frames) {
        intervals.clear();
        intervals.reserve(frames);
        started = now;
        previous = now;
    }

    void Frame(Clock::time_point now) {
        if (previous) {
            intervals.push_back(static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - *previous).count()));
            previous = now;
        }
    }

    void Reset() {
        intervals.clear();
        previous.reset();
    }

    struct Summary {
        u64 elapsed_ns{};
        u64 p50_ns{};
        u64 p95_ns{};
        u64 p99_ns{};
        u64 max_ns{};
        size_t near_120{};
        size_t over_50_ms{};
        size_t over_100_ms{};
    };

    std::optional<Summary> Finish(size_t expected) {
        if (!previous || intervals.empty() || intervals.size() != expected) {
            Reset();
            return std::nullopt;
        }
        const u64 elapsed_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(*previous - started).count());
        std::ranges::sort(intervals);
        const auto quantile = [&](size_t percent) {
            return intervals[(intervals.size() * percent + 99) / 100 - 1];
        };
        const auto within = [&](u64 ns) {
            return static_cast<size_t>(std::ranges::upper_bound(intervals, ns) - intervals.begin());
        };
        const Summary summary{elapsed_ns, quantile(50), quantile(95), quantile(99), intervals.back(),
                              within(Near120Ns), intervals.size() - within(50'000'000),
                              intervals.size() - within(100'000'000)};
        Reset();
        return summary;
    }

private:
    std::vector<u64> intervals;
    Clock::time_point started{};
    std::optional<Clock::time_point> previous;
};

class GpuBenchmark {
public:
    using Experiment = Common::GpuBenchmarkTrial::Experiment;
    using Mode = Common::GpuBenchmarkTrial::Mode;
    static constexpr size_t WarmupFrames = 120;

    struct Phase {
        size_t index;
        size_t frames;
        Mode mode;
        std::optional<PlaybackTiming::Summary> timing;
        u64 read_checks_skipped{};
        u64 acquire_waits_shortened{};
    };

    bool Start(size_t frames, Experiment experiment_) {
        Reset();
        experiment = experiment_;
        modes = Modes(experiment);
        if (modes.empty() || frames % modes.size() != 0 || frames / modes.size() <= WarmupFrames) {
            return false;
        }
        phase_length = frames / modes.size();
        phase_index = 0;
        BeginPhase();
        return true;
    }

    std::optional<Phase> Frame(PlaybackTiming::Clock::time_point now) {
        if (ticket == 0) {
            return std::nullopt;
        }
        const size_t measured = phase_length - WarmupFrames;
        if (++frame == WarmupFrames) {
            if (!Common::GpuBenchmarkTrial::Applied(ticket)) {
                const Phase unavailable{phase_index, measured, modes[phase_index], std::nullopt};
                Reset();
                return unavailable;
            }
            timing.Start(now, measured);
            checks_at_warmup = Common::GpuBenchmarkTrial::ReadChecksSkipped();
            acquire_waits_at_warmup =
                Common::GpuBenchmarkTrial::acquire_waits_shortened.load(std::memory_order_relaxed);
        } else if (frame > WarmupFrames) {
            timing.Frame(now);
        }
        if (frame != phase_length) {
            return std::nullopt;
        }
        const bool applied = Common::GpuBenchmarkTrial::Applied(ticket);
        Phase result{phase_index, measured, modes[phase_index],
                     applied ? timing.Finish(measured) : std::nullopt};
        result.read_checks_skipped = Common::GpuBenchmarkTrial::ReadChecksSkipped() - checks_at_warmup;
        result.acquire_waits_shortened =
            Common::GpuBenchmarkTrial::acquire_waits_shortened.load(std::memory_order_relaxed) -
            acquire_waits_at_warmup;
        if (!applied || ++phase_index == modes.size()) {
            Reset();
        } else {
            BeginPhase();
        }
        return result;
    }

    void Reset() {
        if (ticket != 0) {
            if (experiment == Experiment::CpuPlacement) {
                if (!Common::ApplyCpuPlacementTrial(Common::CpuPlacementTrialMode::Restore, 0)) {
                    LOG_WARNING(Input, "TAS A/B could not restore the original CPU sets");
                }
            } else {
                Common::GpuBenchmarkTrial::Request(Mode::Restore);
            }
            ticket = 0;
            timing.Reset();
        }
    }

    bool Active() const {
        return ticket != 0;
    }

    static std::span<const Mode> Modes(Experiment experiment) {
        static constexpr Mode adaptive_submit[]{
            Mode::Baseline, Mode::AdaptiveSubmit, Mode::AdaptiveSubmit, Mode::Baseline,
        };
        static constexpr Mode renderer[]{
            Mode::Baseline, Mode::AdaptiveSubmit, Mode::Both, Mode::RasterStateReuse,
            Mode::RasterStateReuse, Mode::Both, Mode::AdaptiveSubmit, Mode::Baseline,
        };
        static constexpr Mode cpu_placement[]{
            Mode::Baseline, Mode::CacheLocal, Mode::CacheLocal, Mode::Baseline,
        };
        static constexpr Mode texture_reads[]{
            Mode::Baseline, Mode::TextureReadFilter, Mode::TextureReadFilter, Mode::Baseline,
        };
        static constexpr Mode buffer_reads[]{
            Mode::Baseline, Mode::BufferReadFilter, Mode::BufferReadFilter, Mode::Baseline,
        };
        static constexpr Mode acquire_waits[]{
            Mode::Baseline, Mode::PreciseAcquireWait, Mode::PreciseAcquireWait, Mode::Baseline,
        };
        switch (experiment) {
        case Experiment::AdaptiveSubmit:
            return adaptive_submit;
        case Experiment::Renderer:
            return renderer;
        case Experiment::CpuPlacement:
            return cpu_placement;
        case Experiment::TextureReads:
            return texture_reads;
        case Experiment::BufferReads:
            return buffer_reads;
        case Experiment::AcquireWaits:
            return acquire_waits;
        default:
            return {};
        }
    }

private:
    void BeginPhase() {
        const auto mode = modes[phase_index];
        ticket = Common::GpuBenchmarkTrial::Request(mode);
        if (experiment == Experiment::CpuPlacement) {
            using PlacementMode = Common::CpuPlacementTrialMode;
            Common::ApplyCpuPlacementTrial(
                mode == Mode::CacheLocal ? PlacementMode::CacheLocal : PlacementMode::Baseline,
                ticket);
        }
        frame = 0;
    }

    PlaybackTiming timing;
    std::span<const Mode> modes;
    Experiment experiment{Experiment::None};
    size_t phase_length{};
    size_t phase_index{};
    size_t frame{};
    u32 ticket{};
    u64 checks_at_warmup{};
    u64 acquire_waits_at_warmup{};
};

} // namespace InputCommon::TasInput
