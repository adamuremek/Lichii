// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <span>

#include "common/alignment.h"
#include "common/bit_util.h"
#include "common/common_types.h"

namespace Vulkan::StagingBufferPolicy {

constexpr u32 FineFrom = 20;
constexpr u32 SizeBits = std::numeric_limits<u64>::digits;
constexpr u32 NumClasses = FineFrom + (SizeBits - FineFrom) * 4 + 1;
constexpr u64 KeepFrames = 60;
constexpr size_t DeletionsPerFrame = 32;
constexpr size_t UploadArenaBlockSize = 64 * 1024 * 1024;
constexpr size_t WarmUploadArenaBlockSize = 256 * 1024 * 1024;
constexpr size_t UploadArenaBlockCount = 4;

struct UploadArenaCursor {
    [[nodiscard]] constexpr size_t NextOffset(size_t alignment, u64 tick,
                                              u64 completed_tick) const {
        // Unfinished submissions may share a block, but their byte ranges must stay disjoint.
        const size_t start = last_tick != tick && last_tick <= completed_tick ? 0 : used;
        return Common::AlignUp(start, alignment);
    }

    [[nodiscard]] constexpr std::optional<size_t> TryReserve(size_t size, size_t alignment,
                                                             size_t capacity, u64 tick,
                                                             u64 completed_tick) {
        const size_t offset = NextOffset(alignment, tick, completed_tick);
        if (offset > capacity || size > capacity - offset) {
            return std::nullopt;
        }
        used = offset + size;
        last_tick = tick;
        return offset;
    }

    size_t used{};
    u64 last_tick{};
};

constexpr u32 SizeClass(u64 size, bool fine) {
    const auto log2 = static_cast<u32>(Common::Log2Ceil(size));
    if (!fine || log2 <= FineFrom) {
        return log2;
    }
    const u32 base = log2 - 1;
    const u64 excess = size - (u64{1} << base);
    const u64 step = u64{1} << (base - 2);
    const auto steps = static_cast<u32>((excess - 1) / step + 1);
    return FineFrom + (base - FineFrom) * 4 + steps;
}

constexpr u64 ClassSize(u32 size_class, bool fine) {
    if (!fine || size_class <= FineFrom) {
        return size_class == SizeBits ? (std::numeric_limits<u64>::max)() : u64{1} << size_class;
    }
    if (size_class == NumClasses - 1) {
        return (std::numeric_limits<u64>::max)();
    }
    const u32 fine_class = size_class - FineFrom - 1;
    const u32 base = FineFrom + fine_class / 4;
    return (u64{1} << base) + (fine_class % 4 + 1) * (u64{1} << (base - 2));
}

constexpr u32 LastReuseClass(u32 size_class, bool fine) {
    return fine && size_class > FineFrom ? (std::min)(size_class + 2, NumClasses - 1) : size_class;
}

struct UploadArenaReservation {
    UploadArenaCursor cursor;
    size_t capacity;
    size_t offset{};
    size_t size{};
};

// Plan on copies of the cursors so a request that would still allocate the same size class can
// leave the arena available to other uploads. Return the bytes still needed from staging.
constexpr size_t PlanLargeUpload(std::span<UploadArenaReservation> blocks, size_t size,
                                 size_t alignment, u64 tick, u64 completed_tick, bool fine) {
    size_t remaining = size;
    for (auto& block : blocks) {
        const size_t offset = block.cursor.NextOffset(alignment, tick, completed_tick);
        if (offset >= block.capacity || remaining == 0) {
            continue;
        }
        block.offset = offset;
        block.size = (std::min)(remaining, block.capacity - offset);
        block.cursor = {offset + block.size, tick};
        remaining -= block.size;
    }
    return remaining == 0 || SizeClass(remaining, fine) < SizeClass(size, fine) ? remaining : size;
}

constexpr bool CanRelease(u64 frame, u64 last_use, bool deferred, bool gpu_done) {
    return !deferred && gpu_done && frame - last_use > KeepFrames;
}

} // namespace Vulkan::StagingBufferPolicy
