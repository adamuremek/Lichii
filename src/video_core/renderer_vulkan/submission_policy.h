// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <charconv>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Vulkan::SubmissionPolicy {

inline constexpr u32 DefaultNativeBudget = 128;
/// Draws per submission on AMD's proprietary driver (see RasterizerVulkan::FlushWork).
inline constexpr u32 AmdNativeBudget = 512;

enum class Action {
    None,
    Dispatch,
    Flush,
};

struct WorkSinceSubmit {
    /// Count the work before this call. Any submission, including a fence or a download, advances
    /// the tick and starts a new count.
    constexpr u32 BeforeWork(u64 current_tick) noexcept {
        if (tick != current_tick) {
            tick = current_tick;
            work = 0;
        }
        return work++;
    }

    u64 tick{};
    u32 work{};
};

constexpr bool CheckFrontWait(u32 work, u32 budget) noexcept {
    return budget > 1 && work < budget - 1 && work >= (std::max)(64U, budget / 4) &&
           work % 32 == 0;
}

constexpr bool CheckGpuIdle(u32 work, u32 budget) noexcept {
    return budget > 1 && work < budget - 1 && work >= (std::max)(128U, budget / 2) &&
           work % 64 == 0;
}

inline std::optional<u32> ParseBudget(std::string_view value) {
    if (value.empty()) {
        return std::nullopt;
    }
    u32 budget{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), budget);
    if (error != std::errc{} || end != value.data() + value.size() || budget == 0) {
        return std::nullopt;
    }
    return budget;
}

constexpr Action Advance(u32& counter, bool native, bool android,
                         u32 native_budget = DefaultNativeBudget, u32 legacy_budget = 4096) {
    ++counter;
    if (native && !android) {
        if (counter >= native_budget) {
            counter = 0;
            return Action::Flush;
        }
        return counter % 32 == 0 ? Action::Dispatch : Action::None;
    }
    const u32 check_mask = android ? 3 : 31;
    if ((counter & check_mask) != check_mask) {
        return Action::None;
    }
    if (counter < (android ? 512U : legacy_budget)) {
        return Action::Dispatch;
    }
    counter = 0;
    return Action::Flush;
}

} // namespace Vulkan::SubmissionPolicy
