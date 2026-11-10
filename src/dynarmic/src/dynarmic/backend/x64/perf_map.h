// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <bit>

namespace Dynarmic::Backend::X64 {

#if (defined(__linux__) && !defined(__ANDROID__)) || defined(_WIN32)
namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name);
}  // namespace detail
template<typename T>
void PerfMapRegister(T start, const void* end, std::string_view friendly_name) noexcept {
    detail::PerfMapRegister(std::bit_cast<const void*>(start), end, friendly_name);
}
void PerfMapClear();
#else
// Resolve to no-op (compiler thinks fmt has side effects)
template<typename T> inline void PerfMapRegister(T, const void*, std::string_view) noexcept {}
inline void PerfMapClear() noexcept {}
#endif

#if defined(_WIN32)
/// EDEN_NX2_JIT_CODE=<file>: each A64 block's host code and guest instruction words, for
/// disassembling the blocks a CPU capture with EDEN_NX2_JIT_MAP finds hot.
bool PerfMapCodeEnabled() noexcept;
void PerfMapRecordCode(const void* host, std::size_t host_size, std::uint64_t guest_pc,
                       std::span<const std::uint32_t> guest_words) noexcept;
#else
inline bool PerfMapCodeEnabled() noexcept {
    return false;
}
inline void PerfMapRecordCode(const void*, std::size_t, std::uint64_t,
                              std::span<const std::uint32_t>) noexcept {}
#endif

}  // namespace Dynarmic::Backend::X64
