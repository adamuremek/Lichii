// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/gpu_dirty_memory_manager.h"

namespace {

using Core::GPUDirtyMemoryManager;
using Range = std::pair<PAddr, size_t>;
constexpr size_t PageSize = size_t{1} << (Core::DEVICE_PAGEBITS - 1);

std::vector<Range> Gather(GPUDirtyMemoryManager& manager, bool skip_empty) {
    std::vector<Range> runs;
    std::function<void(PAddr, size_t)> collect = [&](PAddr address, size_t bytes) {
        runs.emplace_back(address, bytes);
    };
    manager.Gather(collect, skip_empty);
    return runs;
}

TEST_CASE("Dirty GPU memory reports writes on either side of a tracking page",
          "[core][gpu_dirty_memory]") {
    for (const bool skip_empty : {false, true}) {
        GPUDirtyMemoryManager manager;
        CHECK_FALSE(manager.HasCollectedWrites());
        manager.Collect(0, 0);
        CHECK(Gather(manager, skip_empty).empty());
        manager.Collect(PageSize - 2, 4);
        CHECK(manager.HasCollectedWrites());
        CHECK(manager.Overlaps(PageSize - 1, 1));
        CHECK(manager.Overlaps(PageSize, 1));
        CHECK_FALSE(manager.Overlaps(PageSize, 0));
        const std::vector<Range> expected{{PageSize - 64, 64}, {PageSize, 64}};
        CHECK(Gather(manager, skip_empty) == expected);
        CHECK_FALSE(manager.HasCollectedWrites());
        CHECK(Gather(manager, skip_empty).empty());
    }
}

TEST_CASE("Dirty GPU memory rollover preserves writes racing with a gather",
          "[core][gpu_dirty_memory]") {
    constexpr size_t producer_count = 2;
    constexpr size_t writes_per_producer = 16384;
    constexpr size_t writes = producer_count * writes_per_producer;
    GPUDirtyMemoryManager manager;
    std::atomic<size_t> remaining{producer_count};
    std::array<std::thread, producer_count> producers;
    for (size_t producer = 0; producer < producer_count; ++producer) {
        producers[producer] = std::thread([&, producer] {
            for (size_t i = 0; i < writes_per_producer; ++i) {
                // Distinct pages and runs make omissions or duplicates visible.
                manager.Collect((producer + i * producer_count) * PageSize, 4);
            }
            remaining.fetch_sub(1, std::memory_order_release);
        });
    }
    std::vector<unsigned> seen(writes);
    size_t invalid_runs = 0;
    std::function<void(PAddr, size_t)> collect = [&](PAddr address, size_t size) {
        if (address % PageSize || address / PageSize >= writes || size != 64) {
            ++invalid_runs;
            return;
        }
        ++seen[address / PageSize];
    };
    while (remaining.load(std::memory_order_acquire)) {
        manager.Gather(collect, true);
    }
    for (auto& producer : producers) {
        producer.join();
    }
    manager.Gather(collect, true);
    CHECK(invalid_runs == 0);
    CHECK(std::ranges::count(seen, 1u) == writes);
    CHECK_FALSE(manager.HasCollectedWrites());
}

TEST_CASE("Dirty GPU memory gather cost", "[.dirty_gather_benchmark]") {
    constexpr size_t iterations = 500'000;
    for (const size_t period : {0, 10, 1}) {
        for (size_t trial = 0; trial < 5; ++trial) {
            std::array<double, 2> timing{};
            for (size_t step = 0; step < 2; ++step) {
                const bool skip_empty = ((trial + step) % 2) != 0;
                std::array<GPUDirtyMemoryManager, 6> managers;
                std::mutex applying;
                u64 total_bytes = 0;
                std::function<void(PAddr, size_t)> collect = [&](PAddr, size_t bytes) {
                    total_bytes += bytes;
                };
                const auto begin = std::chrono::steady_clock::now();
                for (size_t i = 0; i < iterations; ++i) {
                    if (period && i % period == 0) {
                        managers[i % managers.size()].Collect(PageSize, 4);
                    }
                    if (skip_empty && std::ranges::none_of(
                                          managers, &GPUDirtyMemoryManager::HasCollectedWrites)) {
                        continue;
                    }
                    std::scoped_lock lock{applying};
                    for (auto& manager : managers) {
                        manager.Gather(collect, skip_empty);
                    }
                }
                const std::chrono::duration<double, std::nano> elapsed =
                    std::chrono::steady_clock::now() - begin;
                timing[skip_empty] = elapsed.count() / iterations;
                CHECK(total_bytes == (period ? iterations / period * 64 : 0));
            }
            std::printf("period %zu trial %zu: %.2f -> %.2f ns/fence\n", period, trial + 1,
                        timing[0], timing[1]);
        }
    }
}

} // namespace
