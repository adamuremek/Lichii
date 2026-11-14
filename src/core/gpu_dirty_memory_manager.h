// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "core/device_memory_manager.h"

namespace Core {

class GPUDirtyMemoryManager {
public:
    GPUDirtyMemoryManager() : current{default_transform} {
        back_buffer.reserve(256);
        front_buffer.reserve(256);
    }

    ~GPUDirtyMemoryManager() = default;

    void Collect(PAddr address, size_t size) {
        while (size) {
            const size_t bytes = (std::min<size_t>)(size, page_size - (address & page_mask));
            CollectTransform(BuildTransform(address, bytes));
            address += bytes;
            size -= bytes;
        }
    }

    [[nodiscard]] bool HasCollectedWrites() const noexcept {
        // A rollover adds to the back buffer while publishing a nonempty current under guard.
        return current.load(std::memory_order_acquire).mask != 0;
    }

    /// Calls `callback` for each write collected and not yet applied, without consuming them: those
    /// still collecting and those a Gather is handing out.
    void ForEachPending(const std::function<void(PAddr, size_t)>& callback) {
        std::vector<TransformAddress> pending;
        {
            std::scoped_lock gather_lk(gather_guard);
            std::scoped_lock lk(guard);
            pending = back_buffer;
            pending.insert(pending.end(), front_buffer.begin(), front_buffer.end());
            const TransformAddress t = current.load(std::memory_order_acquire);
            if (t.mask) {
                pending.push_back(t);
            }
        }
        for (const auto& transform : pending) {
            ForEachRun(transform, callback);
        }
    }

    void Gather(std::function<void(PAddr, size_t)>& callback, bool skip_empty = false) {
        if (skip_empty && !HasCollectedWrites()) {
            return;
        }
        // FlushRegion also gathers off the GPU thread (host1x video decode reads); serialize
        // gatherers so one cannot swap or clear front_buffer while another iterates it.
        std::scoped_lock gather_lk(gather_guard);
        {
            std::scoped_lock lk(guard);
            const TransformAddress t = current.exchange(default_transform, std::memory_order_acq_rel);
            front_buffer.swap(back_buffer);
            if (t.mask) {
                front_buffer.emplace_back(t);
            }
        }
        for (auto& transform : front_buffer) {
            ForEachRun(transform, callback);
        }
        front_buffer.clear();
    }

    /// Returns true when collected writes not yet gathered overlap [address, address + size).
    bool Overlaps(PAddr address, size_t size) {
        if (size == 0) {
            return false;
        }
        const PAddr end = address + size;
        const auto overlaps = [&](TransformAddress t) {
            if (!t.mask) {
                return false;
            }
            const PAddr begin = static_cast<PAddr>(t.address) << page_bits;
            if (address >= begin + page_size || end <= begin) {
                return false;
            }
            const size_t minor_bit = address > begin ? (address - begin) >> align_bits : 0;
            const size_t top_bit = ((std::min)(end - begin, PAddr{page_size}) + align_mask) >> align_bits;
            return (t.mask & CreateMask<u32>(top_bit, minor_bit)) != 0;
        };
        std::scoped_lock lk(guard);
        if (overlaps(current.load(std::memory_order_acquire))) {
            return true;
        }
        for (const TransformAddress& t : back_buffer) {
            if (overlaps(t)) {
                return true;
            }
        }
        return false;
    }

private:
    struct alignas(8) TransformAddress {
        u32 address;
        u32 mask;
    };

    void CollectTransform(TransformAddress write) {
        TransformAddress previous = current.load(std::memory_order_acquire);
        do {
            if (previous.mask && previous.address != write.address) {
                std::scoped_lock lock{guard};
                // Reserve the entry before changing current, then keep the value we actually
                // replaced. A gather or another collector may have moved the earlier snapshot.
                auto& entry = back_buffer.emplace_back();
                entry = current.exchange(write, std::memory_order_acq_rel);
                if (!entry.mask) {
                    back_buffer.pop_back();
                }
                return;
            }
            const TransformAddress merged{write.address, write.mask | previous.mask};
            if (merged.mask == previous.mask) {
                return;
            }
            if (current.compare_exchange_weak(previous, merged, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
                return;
            }
        } while (true);
    }

    template <typename Func>
    static void ForEachRun(const TransformAddress& transform, Func&& callback) {
        size_t offset = 0;
        u64 mask = transform.mask;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits << align_bits;
            mask = mask >> empty_bits;
            const size_t continuous_bits = std::countr_one(mask);
            callback((static_cast<PAddr>(transform.address) << page_bits) + offset,
                     continuous_bits << align_bits);
            mask = continuous_bits < align_size ? (mask >> continuous_bits) : 0;
            offset += continuous_bits << align_bits;
        }
    }

    constexpr static size_t page_bits = DEVICE_PAGEBITS - 1;
    constexpr static size_t page_size = 1ULL << page_bits;
    constexpr static size_t page_mask = page_size - 1;

    constexpr static size_t align_bits = 6U;
    constexpr static size_t align_size = 1U << align_bits;
    constexpr static size_t align_mask = align_size - 1;
    constexpr static TransformAddress default_transform = {.address = ~0U, .mask = 0U};

    template <typename T>
    T CreateMask(size_t top_bit, size_t minor_bit) {
        T mask = ~T(0);
        mask <<= (sizeof(T) * 8 - top_bit);
        mask >>= (sizeof(T) * 8 - top_bit);
        mask >>= minor_bit;
        mask <<= minor_bit;
        return mask;
    }

    TransformAddress BuildTransform(PAddr address, size_t size) {
        const size_t minor_address = address & page_mask;
        const size_t minor_bit = minor_address >> align_bits;
        const size_t top_bit = (minor_address + size + align_mask) >> align_bits;
        TransformAddress result{};
        result.address = static_cast<u32>(address >> page_bits);
        result.mask = CreateMask<u32>(top_bit, minor_bit);
        return result;
    }

    std::atomic<TransformAddress> current{};
    std::mutex guard;
    std::mutex gather_guard;
    std::vector<TransformAddress> back_buffer;
    std::vector<TransformAddress> front_buffer;
};

} // namespace Core
