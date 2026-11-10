// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/thread_cpu_clock.h"
#include "core/hle/kernel/guest_profile.h"

namespace Kernel::GuestProfile {
namespace {

constexpr std::array<const char*, static_cast<size_t>(State::Count)> state_names{
    "kernel", "jit", "svc", "idle spin", "idle wait"};

const char* SvcName(u32 slot) {
    switch (slot) {
    case 0x01: return "SetHeapSize";
    case 0x06: return "QueryMemory";
    case 0x08: return "CreateThread";
    case 0x0b: return "SleepThread";
    case 0x0d: return "SetThreadPriority";
    case 0x0f: return "SetThreadCoreMask";
    case 0x10: return "GetCurrentProcessorNumber";
    case 0x11: return "SignalEvent";
    case 0x12: return "ClearEvent";
    case 0x16: return "CloseHandle";
    case 0x17: return "ResetSignal";
    case 0x18: return "WaitSynchronization";
    case 0x19: return "CancelSynchronization";
    case 0x1a: return "ArbitrateLock";
    case 0x1b: return "ArbitrateUnlock";
    case 0x1c: return "WaitProcessWideKeyAtomic";
    case 0x1d: return "SignalProcessWideKey";
    case 0x1e: return "GetSystemTick";
    case 0x20: return "SendSyncRequestLight";
    case 0x21: return "SendSyncRequest";
    case 0x22: return "SendSyncRequestWithUserBuffer";
    case 0x24: return "GetProcessId";
    case 0x25: return "GetThreadId";
    case 0x27: return "OutputDebugString";
    case 0x29: return "GetInfo";
    case 0x2c: return "MapPhysicalMemory";
    case 0x2d: return "UnmapPhysicalMemory";
    case 0x34: return "WaitForAddress";
    case 0x35: return "SignalToAddress";
    case 0x36: return "SynchronizePreemptionState";
    case 0x43: return "ReplyAndReceive";
    case 0x45: return "CreateEvent";
    case 0x4c: return "ControlCodeMemory";
    case 0x5f: return "FlushProcessDataCache";
    case YieldSlot: return "SleepThread yield";
    default: return nullptr;
    }
}

struct ThreadState {
    Data data;
    std::string label{"Guest core ?"};
};

thread_local ThreadState t_state;

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

} // namespace

void Record(Data& data, State next, u32 svc, u64 now_ns) {
    if (data.last_ns != 0 && now_ns > data.last_ns) {
        const u64 elapsed = now_ns - data.last_ns;
        data.ns[static_cast<size_t>(data.state)] += elapsed;
        if (data.state == State::Svc) {
            data.svc_ns[data.svc] += elapsed;
        }
    }
    data.last_ns = now_ns;
    data.state = next;
    ++data.count[static_cast<size_t>(next)];
    if (next == State::Svc) {
        data.svc = std::min<u32>(svc, YieldSlot);
        ++data.svc_count[data.svc];
    }
}

std::string Format(const Data& data, u64 window_ns, std::optional<u64> cpu_ns) {
    const double window = static_cast<double>(std::max<u64>(window_ns, 1));
    const auto ms = [&](u64 ns) { return static_cast<double>(ns) * 1000.0 / window; };
    const auto per_s = [&](u64 n) { return static_cast<double>(n) * 1e9 / window; };
    std::string line;
    for (size_t i = 0; i < data.ns.size(); ++i) {
        line += fmt::format(" | {} {:.1f} ms", state_names[i], ms(data.ns[i]));
        if (i != static_cast<size_t>(State::Kernel) && i != static_cast<size_t>(State::IdleWait)) {
            line += fmt::format(" ({:.0f}/s)", per_s(data.count[i]));
        }
    }
    if (cpu_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.1f}", ms(*cpu_ns));
    } else {
        line += " | host_cpu_busy_ms_per_s=unavailable";
    }
    line += fmt::format(" window_wall_ms={:.1f}", window / 1e6);
    std::vector<std::pair<u64, u32>> calls;
    for (u32 slot = 0; slot < SvcSlots; ++slot) {
        if (data.svc_count[slot] != 0) {
            calls.emplace_back(data.svc_ns[slot], slot);
        }
    }
    std::sort(calls.begin(), calls.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (!calls.empty()) {
        line += " | svcs:";
        for (size_t i = 0; i < std::min<size_t>(calls.size(), 8); ++i) {
            const u32 slot = calls[i].second;
            const char* name = SvcName(slot);
            line += fmt::format(" {} {:.1f} ms ({:.0f}/s),",
                                name ? std::string{name} : fmt::format("svc {:#x}", slot),
                                ms(calls[i].first), per_s(data.svc_count[slot]));
        }
        line.pop_back();
    }
    return line;
}

void SetCore(int core) {
    t_state.label = core < 0 ? std::string{"Guest single core"} : fmt::format("Guest core {}", core);
}

void Enter(State next, u32 svc) {
    auto& data = t_state.data;
    const u64 now = NowNs();
    if (data.window_start_ns == 0) {
        data.window_start_ns = now;
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
    }
    Record(data, next, svc, now);
    if (now - data.window_start_ns < WindowNs) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    std::optional<u64> cpu;
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        cpu = *cpu_now - *data.cpu_start_ns;
    }
    LOG_INFO(Kernel, "{} per second:{}", t_state.label,
             Format(data, now - data.window_start_ns, cpu));
    data = Data{.state = next, .svc = data.svc, .last_ns = now, .window_start_ns = now,
                .cpu_start_ns = cpu_now};
}

} // namespace Kernel::GuestProfile
