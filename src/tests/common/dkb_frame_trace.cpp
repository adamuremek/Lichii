// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/dkb_frame_trace.h"

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
