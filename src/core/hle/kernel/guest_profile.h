// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

#include "common/common_types.h"

/// EDEN_NX2_GUEST_PROFILE=1: splits each guest core thread's wall time into JIT execution, system
/// calls (by call), the host kernel between them, and idle spinning or blocking. Logged every 10 s.
namespace Kernel::GuestProfile {

enum class State : u8 { Kernel, Jit, Svc, IdleSpin, IdleWait, Count };

/// SleepThread with a zero or negative timeout is a yield; it gets its own slot.
inline constexpr u32 YieldSlot = 0x100;
inline constexpr size_t SvcSlots = YieldSlot + 1;
inline constexpr u64 WindowNs = 10'000'000'000ULL;

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GUEST_PROFILE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

struct Data {
    State state{State::Kernel};
    u32 svc{};
    u64 last_ns{};
    u64 window_start_ns{};
    std::optional<u64> cpu_start_ns;
    std::array<u64, static_cast<size_t>(State::Count)> ns{};
    std::array<u64, static_cast<size_t>(State::Count)> count{};
    std::array<u64, SvcSlots> svc_ns{};
    std::array<u64, SvcSlots> svc_count{};
};

/// Charges the time since the last transition to the previous state and enters the next one.
void Record(Data& data, State next, u32 svc, u64 now_ns);

/// One log line's text for a window of window_ns wall time; cpu_ns is the thread's CPU time in it.
[[nodiscard]] std::string Format(const Data& data, u64 window_ns, std::optional<u64> cpu_ns);

/// Labels this host thread's lines ("Guest core <n>", or "Guest single core" for core < 0).
void SetCore(int core);

/// Records a transition on this host thread and logs its window once 10 s have passed.
void Enter(State next, u32 svc = 0);

} // namespace Kernel::GuestProfile
