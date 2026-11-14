// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/dkb_frame_trace.h"
#include "input_common/drivers/tas_timing.h"

TEST_CASE("DKB frame trace requires the exact environment opt-in", "[common][dkb_frame_trace]") {
    const char* value = std::getenv("EDEN_DKB_FRAME_TRACE");
    REQUIRE(Common::DkbFrameTrace::Enabled() == (value && std::strcmp(value, "1") == 0));
}

TEST_CASE("DKB frame trace limits reports by elapsed time", "[common][dkb_frame_trace]") {
    using namespace std::chrono_literals;
    Common::DkbFrameTrace::RateLimit rate;
    const auto start = std::chrono::steady_clock::time_point{10s};
    REQUIRE(rate.Ready(start));
    REQUIRE_FALSE(rate.Ready(start));
    REQUIRE_FALSE(rate.Ready(start + 999ms));
    REQUIRE(rate.Ready(start + 1s));
    REQUIRE(rate.Ready(start + 20s));
    REQUIRE_FALSE(rate.Ready(start + 20s));
}

TEST_CASE("DKB GPU timestamp differences respect valid bits", "[common][dkb_frame_trace]") {
    using Common::DkbFrameTrace::TimestampDelta;
    REQUIRE(TimestampDelta(100, 145, 64) == 45);
    REQUIRE(TimestampDelta(~u64{0} - 2, 4, 64) == 7);
    REQUIRE(TimestampDelta(250, 5, 8) == 11);
    REQUIRE(TimestampDelta(0x1fa, 0x305, 8) == 11);
    REQUIRE(TimestampDelta(10, 10, 36) == 0);
    REQUIRE(TimestampDelta((u64{1} << 36) - 3, 4, 36) == 7);
    REQUIRE(TimestampDelta(0, 1, 0) == 0);
    REQUIRE(TimestampDelta(0, 1, 65) == 0);
}

TEST_CASE("Frame timing windows retain rare long frames and exact nearest-rank quantiles",
          "[common][dkb_frame_trace]") {
    Common::DkbFrameTrace::FrameTimeWindow window;
    REQUIRE(window.Summarize().max_ms == 0.0);
    for (u64 i = 1; i <= window.Capacity; ++i) {
        REQUIRE(window.Add(i * 1'000'000) == (i == window.Capacity));
    }
    const auto summary = window.Summarize();
    REQUIRE(summary.mean_ms == 150.5);
    REQUIRE(summary.p50_ms == 150.0);
    REQUIRE(summary.p95_ms == 285.0);
    REQUIRE(summary.p99_ms == 297.0);
    REQUIRE(summary.max_ms == 300.0);
    REQUIRE(summary.over_50_ms == 250);
    REQUIRE(summary.over_100_ms == 200);
    window.count = 0;
    window.Add(1'000'000'000);
    REQUIRE(window.Summarize().p99_ms == 1000.0);
    REQUIRE(window.Summarize().over_100_ms == 1);
}

TEST_CASE("TAS timing includes every route interval and clears partial runs", "[common][dkb_frame_trace]") {
    using namespace std::chrono_literals;
    using InputCommon::TasInput::PlaybackTiming;
    PlaybackTiming timing;
    const auto start = PlaybackTiming::Clock::time_point{20s};
    timing.Start(start, 100);
    for (int frame = 1; frame < 100; ++frame) {
        timing.Frame(start + 8ms * frame);
    }
    timing.Frame(start + 8ms * 99 + 130ms);
    const auto summary = timing.Finish(100);
    REQUIRE(summary);
    CHECK(summary->elapsed_ns == 922'000'000);
    CHECK(summary->p50_ns == 8'000'000);
    CHECK(summary->p95_ns == 8'000'000);
    CHECK(summary->p99_ns == 8'000'000);
    CHECK(summary->max_ns == 130'000'000);
    CHECK(summary->near_120 == 99);
    CHECK(summary->over_50_ms == 1);
    CHECK(summary->over_100_ms == 1);

    timing.Start(start, 2);
    timing.Frame(start + 9ms);
    CHECK_FALSE(timing.Finish(2));
    timing.Start(start + 1s, 1);
    timing.Frame(start + 1s + 9ms);
    const auto next = timing.Finish(1);
    REQUIRE(next);
    CHECK(next->elapsed_ns == 9'000'000);
    CHECK(next->near_120 == 0);
}

TEST_CASE("TAS A/B measures equal idle phases and excludes switch warmup",
          "[common][dkb_frame_trace]") {
    using namespace std::chrono_literals;
    namespace Trial = Common::GpuBenchmarkTrial;
    using InputCommon::TasInput::GpuBenchmark;
    using InputCommon::TasInput::PlaybackTiming;
    using Mode = Trial::Mode;
    const auto experiment = GENERATE(Trial::Experiment::AdaptiveSubmit, Trial::Experiment::Renderer,
                                     Trial::Experiment::TextureReads, Trial::Experiment::BufferReads,
                                     Trial::Experiment::AcquireWaits);
    constexpr std::array adaptive{Mode::Baseline, Mode::AdaptiveSubmit, Mode::AdaptiveSubmit,
                                  Mode::Baseline};
    constexpr std::array renderer{Mode::Baseline, Mode::AdaptiveSubmit, Mode::Both,
                                  Mode::RasterStateReuse, Mode::RasterStateReuse, Mode::Both,
                                  Mode::AdaptiveSubmit, Mode::Baseline};
    constexpr std::array texture_reads{Mode::Baseline, Mode::TextureReadFilter,
                                      Mode::TextureReadFilter, Mode::Baseline};
    constexpr std::array buffer_reads{Mode::Baseline, Mode::BufferReadFilter,
                                      Mode::BufferReadFilter, Mode::Baseline};
    constexpr std::array acquire_waits{Mode::Baseline, Mode::PreciseAcquireWait,
                                       Mode::PreciseAcquireWait, Mode::Baseline};
    std::span<const Mode> sequence{adaptive};
    if (experiment == Trial::Experiment::Renderer) {
        sequence = renderer;
    } else if (experiment == Trial::Experiment::TextureReads) {
        sequence = texture_reads;
    } else if (experiment == Trial::Experiment::BufferReads) {
        sequence = buffer_reads;
    } else if (experiment == Trial::Experiment::AcquireWaits) {
        sequence = acquire_waits;
    }
    const size_t phases = sequence.size();
    GpuBenchmark comparison;
    constexpr size_t warmup = GpuBenchmark::WarmupFrames;
    CHECK_FALSE(comparison.Start(phases * warmup, experiment));
    CHECK_FALSE(comparison.Start(phases * (warmup + 1) + 1, experiment));
    REQUIRE(comparison.Start(phases * (warmup + 3), experiment));
    auto now = PlaybackTiming::Clock::time_point{};
    u32 previous_ticket{};
    for (size_t phase = 0; phase < phases; ++phase) {
        const Mode mode = sequence[phase];
        const u32 ticket = Trial::Requested();
        CHECK(ticket != previous_ticket);
        CHECK(Trial::GetMode(ticket) == mode);
        CHECK_FALSE(Trial::Applied(ticket));
        previous_ticket = ticket;
        const bool skips_checks = mode == Mode::TextureReadFilter || mode == Mode::BufferReadFilter;
        const bool shortens_waits = mode == Mode::PreciseAcquireWait;
        for (size_t frame = 0; frame < warmup; ++frame) {
            if (frame == 3) {
                Trial::Acknowledge(ticket);
            }
            if (skips_checks) {
                Trial::NoteReadCheckSkipped();
            }
            if (shortens_waits) {
                Trial::acquire_waits_shortened.fetch_add(1, std::memory_order_relaxed);
            }
            now += 40ms;
            CHECK_FALSE(comparison.Frame(now));
        }
        const auto interval = 40ms - std::chrono::milliseconds{4 * static_cast<u32>(mode)};
        for (size_t frame = 0; frame < 2; ++frame) {
            if (skips_checks) {
                Trial::NoteReadCheckSkipped();
            }
            if (shortens_waits) {
                Trial::acquire_waits_shortened.fetch_add(1, std::memory_order_relaxed);
            }
            now += interval;
            CHECK_FALSE(comparison.Frame(now));
        }
        const auto result = comparison.Frame(now += interval);
        REQUIRE(result);
        REQUIRE(result->timing);
        CHECK(result->index == phase);
        CHECK(result->mode == mode);
        CHECK(result->frames == 3);
        CHECK(result->read_checks_skipped == (skips_checks ? 2 : 0));
        CHECK(result->acquire_waits_shortened == (shortens_waits ? 2 : 0));
        const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count();
        CHECK(result->timing->elapsed_ns == 3 * ns);
        CHECK(result->timing->p99_ns == ns);
    }
    CHECK(Trial::GetMode(Trial::Requested()) == Trial::Mode::Restore);
    CHECK_FALSE(comparison.Frame(now + 5ms));
}

TEST_CASE("TAS A/B restores the configured mode after a missing acknowledgement or reset",
          "[common][dkb_frame_trace]") {
    using namespace std::chrono_literals;
    namespace Trial = Common::GpuBenchmarkTrial;
    using InputCommon::TasInput::GpuBenchmark;
    using InputCommon::TasInput::PlaybackTiming;
    GpuBenchmark comparison;
    constexpr size_t warmup = GpuBenchmark::WarmupFrames;
    REQUIRE(comparison.Start(8 * (warmup + 3), Trial::Experiment::Renderer));
    auto now = PlaybackTiming::Clock::time_point{};
    for (size_t frame = 1; frame < warmup; ++frame) {
        CHECK_FALSE(comparison.Frame(now += 16ms));
    }
    const auto unavailable = comparison.Frame(now += 16ms);
    REQUIRE(unavailable);
    CHECK_FALSE(unavailable->timing);
    CHECK(Trial::GetMode(Trial::Requested()) == Trial::Mode::Restore);
    REQUIRE(comparison.Start(8 * (warmup + 3), Trial::Experiment::Renderer));
    Trial::Acknowledge(Trial::Requested());
    CHECK_FALSE(comparison.Frame(now += 16ms));
    comparison.Reset();
    CHECK(Trial::GetMode(Trial::Requested()) == Trial::Mode::Restore);
    CHECK_FALSE(comparison.Frame(now += 16ms));
}

TEST_CASE("TAS A/B applies only the requested renderer policies", "[common][dkb_frame_trace]") {
    using namespace Common::GpuBenchmarkTrial;
    constexpr Policies configured{true, true};
    const auto baseline = ResolvePolicies(Experiment::Renderer, Mode::Baseline, configured);
    CHECK_FALSE(baseline.adaptive_submit);
    CHECK_FALSE(baseline.raster_state_reuse);
    const auto submit = ResolvePolicies(Experiment::Renderer, Mode::AdaptiveSubmit, configured);
    CHECK(submit.adaptive_submit);
    CHECK_FALSE(submit.raster_state_reuse);
    const auto raster = ResolvePolicies(Experiment::Renderer, Mode::RasterStateReuse, configured);
    CHECK_FALSE(raster.adaptive_submit);
    CHECK(raster.raster_state_reuse);
    const auto both = ResolvePolicies(Experiment::Renderer, Mode::Both, {false, false});
    CHECK(both.adaptive_submit);
    CHECK(both.raster_state_reuse);
    const auto restored = ResolvePolicies(Experiment::Renderer, Mode::Restore, {true, false});
    CHECK(restored.adaptive_submit);
    CHECK_FALSE(restored.raster_state_reuse);
    const auto legacy = ResolvePolicies(Experiment::AdaptiveSubmit, Mode::Baseline, configured);
    CHECK_FALSE(legacy.adaptive_submit);
    CHECK(legacy.raster_state_reuse);
    for (const Mode candidate :
         {Mode::TextureReadFilter, Mode::BufferReadFilter, Mode::PreciseAcquireWait}) {
        CHECK_FALSE(ResolveToggle(Mode::Baseline, candidate, true));
        CHECK(ResolveToggle(candidate, candidate, false));
        CHECK(ResolveToggle(Mode::Restore, candidate, true));
        CHECK_FALSE(ResolveToggle(Mode::Restore, candidate, false));
    }
}
