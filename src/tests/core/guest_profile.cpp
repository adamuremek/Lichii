// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/kernel/guest_profile.h"

using namespace Kernel::GuestProfile;

namespace {
constexpr u64 Ms = 1'000'000;
size_t Index(State state) {
    return static_cast<size_t>(state);
}
} // namespace

TEST_CASE("Guest profile charges each interval to the state it leaves", "[guest_profile]") {
    Data data;
    Record(data, State::Jit, 0, 1 * Ms);
    Record(data, State::Svc, 0x18, 5 * Ms);
    Record(data, State::Kernel, 0, 7 * Ms);
    Record(data, State::IdleSpin, 0, 8 * Ms);
    Record(data, State::IdleWait, 0, 8 * Ms + Ms / 10);
    Record(data, State::Kernel, 0, 10 * Ms);

    CHECK(data.ns[Index(State::Jit)] == 4 * Ms);
    CHECK(data.ns[Index(State::Svc)] == 2 * Ms);
    CHECK(data.ns[Index(State::Kernel)] == 1 * Ms);
    CHECK(data.ns[Index(State::IdleSpin)] == Ms / 10);
    CHECK(data.ns[Index(State::IdleWait)] == 2 * Ms - Ms / 10);
    CHECK(data.svc_ns[0x18] == 2 * Ms);
    CHECK(data.svc_count[0x18] == 1);
    CHECK(data.count[Index(State::Jit)] == 1);
}

TEST_CASE("Guest profile keeps yields apart and clamps unknown calls", "[guest_profile]") {
    Data data;
    Record(data, State::Svc, YieldSlot, 1 * Ms);
    Record(data, State::Svc, 0x0b, 2 * Ms);
    Record(data, State::Svc, 0x9999, 4 * Ms);
    Record(data, State::Kernel, 0, 5 * Ms);
    CHECK(data.svc_ns[YieldSlot] == 1 * Ms + 1 * Ms);
    CHECK(data.svc_ns[0x0b] == 2 * Ms);
    CHECK(data.svc_count[YieldSlot] == 2);
}

TEST_CASE("Guest profile lines report per second and sort calls by time", "[guest_profile]") {
    Data data;
    Record(data, State::Jit, 0, 1);
    Record(data, State::Svc, 0x1a, 1 + 600 * Ms);
    Record(data, State::Svc, YieldSlot, 1 + 700 * Ms);
    Record(data, State::Svc, 0x7f, 1 + 1000 * Ms);
    Record(data, State::Kernel, 0, 1 + 1001 * Ms);
    const std::string line = Format(data, 2000 * Ms, 900 * Ms);
    CHECK(line ==
          " | kernel 0.0 ms | jit 300.0 ms (0/s) | svc 200.5 ms (2/s) | idle spin 0.0 ms (0/s)"
          " | idle wait 0.0 ms | host_cpu_busy_ms_per_s=450.0 window_wall_ms=2000.0"
          " | svcs: SleepThread yield 150.0 ms (0/s), ArbitrateLock 50.0 ms (0/s),"
          " svc 0x7f 0.5 ms (0/s)");
}
