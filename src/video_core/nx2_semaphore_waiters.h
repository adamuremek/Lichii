// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "common/common_types.h"

namespace Tegra {

/// A release latches every suspended acquire it meets, even if another release overwrites it
/// before the front polls. The front owns the latches and the registry serializes their writers.
class Nx2SemaphoreWaiters {
public:
    static constexpr u32 AcquireEqual = 0x1;
    static constexpr u32 AcquireGequal = 0x4;
    static constexpr u32 AcquireMask = 0x8;

    enum class Source : u8 {
        Unknown,
        PayloadRelease,
        QueryResult,
        QmdReduction,
        QmdMax,
        Count,
    };

    class Latch {
    public:
        ~Latch() {
            if (registry) {
                registry->Remove(*this);
            }
        }

        [[nodiscard]] bool Latched() const noexcept {
            return source.load(std::memory_order_acquire) != Source::Count;
        }

        [[nodiscard]] Source LatchSource() const noexcept {
            const auto current = source.load(std::memory_order_acquire);
            return current == Source::Count ? Source::Unknown : current;
        }

    private:
        friend class Nx2SemaphoreWaiters;
        Nx2SemaphoreWaiters* registry{};
        std::atomic<Source> source{Source::Count};
    };

    static Nx2SemaphoreWaiters& Get() {
        static Nx2SemaphoreWaiters waiters;
        return waiters;
    }

    ~Nx2SemaphoreWaiters() {
        for (const auto& waiter : waiters) {
            waiter.latch->registry = nullptr;
            waiter.latch->source.store(Source::Count, std::memory_order_relaxed);
        }
    }

    void Add(Latch& latch, u64 device_address, u32 value, u32 operation) {
        if (latch.registry && latch.registry != this) {
            latch.registry->Remove(latch);
        }
        std::scoped_lock lock{mutex};
        const Waiter waiter{&latch, device_address, value, operation};
        const auto it = std::ranges::find(waiters, &latch, &Waiter::latch);
        if (it == waiters.end()) {
            waiters.push_back(waiter);
            count.store(waiters.size(), std::memory_order_release);
        } else {
            *it = waiter;
        }
        latch.registry = this;
        latch.source.store(Source::Count, std::memory_order_relaxed);
    }

    void Remove(Latch& latch) {
        if (latch.registry != this) {
            return;
        }
        std::scoped_lock lock{mutex};
        std::erase_if(waiters, [&latch](const Waiter& waiter) { return waiter.latch == &latch; });
        count.store(waiters.size(), std::memory_order_release);
        latch.registry = nullptr;
        latch.source.store(Source::Count, std::memory_order_relaxed);
    }

    void OnWrite(u64 device_address, u32 value, Source source = Source::Unknown) {
        if (count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lock{mutex};
        for (const auto& waiter : waiters) {
            auto& current = waiter.latch->source;
            if (waiter.address == device_address &&
                current.load(std::memory_order_relaxed) == Source::Count && Matches(waiter, value)) {
                // Publish the writes preceding the semaphore to the front that polls this latch.
                current.store(source, std::memory_order_release);
            }
        }
    }

private:
    struct Waiter {
        Latch* latch;
        u64 address;
        u32 value;
        u32 operation;
    };

    static bool Matches(const Waiter& waiter, u32 value) {
        switch (waiter.operation) {
        case AcquireEqual:
            return value == waiter.value;
        case AcquireGequal:
            return value >= waiter.value;
        case AcquireMask:
            return (value & waiter.value) != 0;
        default:
            return false;
        }
    }

    std::mutex mutex;
    std::vector<Waiter> waiters;
    std::atomic<size_t> count{};
};

} // namespace Tegra
