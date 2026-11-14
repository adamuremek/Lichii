// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Common {

namespace DkbCpuTrace {
enum class Role : std::size_t;
}

/// One logical processor as the host reports it.
struct LogicalProcessor {
    /// What the host's affinity call takes (a CPU set id on Windows).
    u32 id;
    /// The physical core: equal for SMT siblings.
    u32 core;
    /// Higher is faster. Equal on every processor of a CPU that is not hybrid.
    u8 efficiency_class;
    std::optional<u32> last_level_cache{};
};

/// Physical cores of the fastest class needed before the GPU threads are pinned.
constexpr std::size_t MinCoresForGpuThreadPinning = 6;

/// Logical processor ids for each pinned role. All three are empty when the host has fewer than
/// MinCoresForGpuThreadPinning performance cores.
struct GpuThreadPinning {
    std::size_t performance_cores{};
    std::vector<u32> front;
    std::vector<u32> back;
    std::vector<u32> guest;
    std::vector<u32> vulkan_worker;
    std::vector<std::vector<u32>> guest_by_core;
};

/// The split GPU thread's front and back each get a whole physical performance core, so no SMT
/// sibling shares it, and the guest cores get the remaining performance cores. Core 0 takes most
/// interrupts, so the GPU threads get the next two.
/// With a guest core count, use a separate cache for those cores when the topology allows it.
[[nodiscard]] GpuThreadPinning PlanGpuThreadPinning(
    std::span<const LogicalProcessor> processors, std::size_t cache_local_guest_cores = 0);

void PinCurrentGuestCore(std::size_t core);
void PinCurrentVulkanWorker();

class CpuPlacementTrialThread {
public:
    CpuPlacementTrialThread(DkbCpuTrace::Role role, bool pin_gpu_threads);
    ~CpuPlacementTrialThread();
    CpuPlacementTrialThread(const CpuPlacementTrialThread&) = delete;
    CpuPlacementTrialThread& operator=(const CpuPlacementTrialThread&) = delete;

private:
    std::optional<std::size_t> slot;
};

enum class CpuPlacementTrialMode { Baseline, CacheLocal, Restore };
bool ApplyCpuPlacementTrial(CpuPlacementTrialMode mode, u32 ticket);

/// The CPUs of a Linux sysfs CPU list such as "0-3,8,10-11", in order. Empty for an empty or
/// malformed list.
[[nodiscard]] std::vector<u32> ParseCpuList(std::string_view list);

} // namespace Common
