// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"

namespace Common::DkbFrameTrace {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_DKB_FRAME_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

struct RateLimit {
    std::chrono::steady_clock::time_point next{};

    bool Ready(std::chrono::steady_clock::time_point now) {
        if (now < next) {
            return false;
        }
        next = now + std::chrono::seconds{1};
        return true;
    }
};

// Bounded windows preserve exact frame quantiles without per-frame log writes or allocation.
struct FrameTimeWindow {
    static constexpr size_t Capacity = 300;
    std::array<u64, Capacity> intervals_ns{};
    size_t count{};

    bool Add(u64 interval_ns) {
        if (count < Capacity) {
            intervals_ns[count++] = interval_ns;
        }
        return count == Capacity;
    }

    struct Summary {
        double mean_ms{};
        double p50_ms{};
        double p95_ms{};
        double p99_ms{};
        double max_ms{};
        u64 over_50_ms{};
        u64 over_100_ms{};
    };

    Summary Summarize() const {
        if (count == 0) {
            return {};
        }
        auto sorted = intervals_ns;
        std::sort(sorted.begin(), sorted.begin() + count);
        u64 sum{}, over_50{}, over_100{};
        for (size_t i = 0; i < count; ++i) {
            sum += sorted[i];
            over_50 += sorted[i] > 50'000'000;
            over_100 += sorted[i] > 100'000'000;
        }
        const auto percentile = [&](size_t percent) {
            return sorted[(count * percent + 99) / 100 - 1] / 1e6;
        };
        return {static_cast<double>(sum) / count / 1e6, percentile(50), percentile(95),
                percentile(99), sorted[count - 1] / 1e6, over_50, over_100};
    }
};

constexpr u64 TimestampDelta(u64 begin, u64 end, u32 valid_bits) {
    if (valid_bits == 0 || valid_bits > 64) {
        return 0;
    }
    const u64 mask = valid_bits == 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    return (end - begin) & mask;
}

} // namespace Common::DkbFrameTrace
