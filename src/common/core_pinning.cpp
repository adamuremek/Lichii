// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <charconv>
#include <iterator>
#include <map>

#include "common/core_pinning.h"

namespace Common {

GpuThreadPinning PlanGpuThreadPinning(std::span<const LogicalProcessor> processors,
                                     size_t cache_local_guest_cores) {
    GpuThreadPinning plan;
    if (processors.empty()) {
        return plan;
    }
    const u8 fastest =
        std::ranges::max(processors, {}, &LogicalProcessor::efficiency_class).efficiency_class;
    std::vector<u32> cores;
    for (const LogicalProcessor& processor : processors) {
        if (processor.efficiency_class == fastest) {
            cores.push_back(processor.core);
        }
    }
    std::ranges::sort(cores);
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    plan.performance_cores = cores.size();
    if (cores.size() < MinCoresForGpuThreadPinning) {
        return plan;
    }
    const u32 front_core = cores[1];
    const u32 back_core = cores[2];
    for (const LogicalProcessor& processor : processors) {
        if (processor.efficiency_class != fastest) {
            continue;
        }
        if (processor.core == front_core) {
            plan.front.push_back(processor.id);
        } else if (processor.core == back_core) {
            plan.back.push_back(processor.id);
        } else {
            plan.guest.push_back(processor.id);
        }
    }
    if (cache_local_guest_cores == 0) {
        return plan;
    }

    // Keep the default front and back cores. The worker shares their cache, and the guest cores
    // share a second cache with enough distinct physical cores for every guest thread.
    std::map<u32, std::map<u32, std::vector<u32>>> caches;
    std::optional<u32> front_cache;
    std::optional<u32> back_cache;
    for (const auto& cpu : processors) {
        if (cpu.efficiency_class != fastest) {
            continue;
        }
        if (!cpu.last_level_cache) {
            return plan;
        }
        const u32 cache = *cpu.last_level_cache;
        caches[cache][cpu.core].push_back(cpu.id);
        if (cpu.core == front_core) {
            front_cache = cache;
        } else if (cpu.core == back_core) {
            back_cache = cache;
        }
    }
    if (caches.size() != 2 || !front_cache || front_cache != back_cache) {
        return plan;
    }
    const auto gpu_cache = caches.find(*front_cache);
    const auto guest_cache = gpu_cache == caches.begin() ? std::next(gpu_cache) : caches.begin();
    if (guest_cache->second.size() < cache_local_guest_cores) {
        return plan;
    }
    const auto worker = std::ranges::find_if(gpu_cache->second, [&](const auto& entry) {
        return entry.first != cores.front() && entry.first != front_core && entry.first != back_core;
    });
    if (worker == gpu_cache->second.end()) {
        return plan;
    }
    plan.vulkan_worker = worker->second;
    plan.guest.clear();
    for (const auto& [core, ids] : guest_cache->second) {
        if (plan.guest_by_core.size() == cache_local_guest_cores) {
            break;
        }
        plan.guest_by_core.push_back(ids);
        plan.guest.insert(plan.guest.end(), ids.begin(), ids.end());
    }
    return plan;
}

std::vector<u32> ParseCpuList(std::string_view list) {
    std::vector<u32> cpus;
    while (!list.empty() && (list.back() == '\n' || list.back() == ' ')) {
        list.remove_suffix(1);
    }
    const auto number = [](std::string_view text, u32& value) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc{} && end == text.data() + text.size();
    };
    while (!list.empty()) {
        const size_t comma = list.find(',');
        const std::string_view range = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        const size_t dash = range.find('-');
        u32 first{};
        u32 last{};
        if (!number(range.substr(0, dash), first) ||
            !number(dash == std::string_view::npos ? range.substr(0, dash) : range.substr(dash + 1),
                    last) ||
            last < first || last - first >= 4096) {
            return {};
        }
        // 64-bit, so a range ending at the largest u32 doesn't wrap to 0 and loop forever.
        for (u64 cpu = first; cpu <= last; ++cpu) {
            cpus.push_back(static_cast<u32>(cpu));
        }
    }
    return cpus;
}

} // namespace Common
