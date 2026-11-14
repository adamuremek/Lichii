// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/puller.h"
#include "video_core/nx2_semaphore_waiters.h"

using Tegra::Nx2SemaphoreWaiters;
using Latch = Nx2SemaphoreWaiters::Latch;

TEST_CASE("Semaphore waits include unplaced writers and later overlapping writes",
          "[video_core][nx2_waiters]") {
    using Acquire = Tegra::Engines::Puller::PendingAcquire;
    CHECK(Acquire::UnnotedWriterPosition(0, 0, 0) == 0);
    CHECK(Acquire::UnnotedWriterPosition(0, 70, 80) == 0);
    CHECK(Acquire::UnnotedWriterPosition(80, 70, 70) == 80);
    CHECK(Acquire::UnnotedWriterPosition(20, 90, 80) == 90);
    CHECK(Acquire::UnnotedWriterPosition(110, 90, 80) == 110);
}

TEST_CASE("Semaphore waiters latch on the writes that meet them", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    Latch equal, gequal, mask;
    REQUIRE_FALSE(equal.Latched());
    waiters.Add(equal, 0x1000, 5, Nx2SemaphoreWaiters::AcquireEqual);
    waiters.Add(gequal, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal);
    waiters.Add(mask, 0x2000, 0x4, Nx2SemaphoreWaiters::AcquireMask);

    waiters.OnWrite(0x1000, 4);
    waiters.OnWrite(0x2000, 0x3);
    waiters.OnWrite(0x3000, 0xffffffff);
    REQUIRE_FALSE(equal.Latched());
    REQUIRE_FALSE(gequal.Latched());
    REQUIRE_FALSE(mask.Latched());

    waiters.OnWrite(0x1000, 6, Nx2SemaphoreWaiters::Source::QmdMax);
    REQUIRE_FALSE(equal.Latched());
    REQUIRE(gequal.Latched());
    waiters.OnWrite(0x1000, 5, Nx2SemaphoreWaiters::Source::PayloadRelease);
    REQUIRE(equal.Latched());
    CHECK(gequal.LatchSource() == Nx2SemaphoreWaiters::Source::QmdMax);
    CHECK(equal.LatchSource() == Nx2SemaphoreWaiters::Source::PayloadRelease);
    // A later write that no longer meets it keeps the latch.
    waiters.OnWrite(0x1000, 0);
    REQUIRE(equal.Latched());
    waiters.OnWrite(0x2000, 0x6);
    REQUIRE(mask.Latched());

    // Adding again starts unlatched, on the new word.
    waiters.Add(equal, 0x1004, 7, Nx2SemaphoreWaiters::AcquireEqual);
    REQUIRE_FALSE(equal.Latched());
    CHECK(equal.LatchSource() == Nx2SemaphoreWaiters::Source::Unknown);
    waiters.OnWrite(0x1000, 7);
    REQUIRE_FALSE(equal.Latched());
    waiters.OnWrite(0x1004, 7);
    REQUIRE(equal.Latched());

    waiters.Remove(gequal);
    REQUIRE_FALSE(gequal.Latched());
    waiters.Remove(equal);
    waiters.Remove(mask);
    waiters.Remove(mask);
    REQUIRE_FALSE(equal.Latched());
    REQUIRE_FALSE(mask.Latched());
}

TEST_CASE("Semaphore waiters retain more than 64 owners and replace existing owners",
          "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    std::array<Latch, 141> owners{};
    for (size_t i = 0; i + 1 < owners.size(); ++i) {
        waiters.Add(owners[i], 0x1000 + i * 4, 1, Nx2SemaphoreWaiters::AcquireGequal);
    }
    auto& extra = owners.back();
    waiters.Add(extra, 0x8000, 1, Nx2SemaphoreWaiters::AcquireGequal);
    waiters.OnWrite(0x8000, 1);
    REQUIRE(extra.Latched());
    // An owner already in the table moves to its new word.
    waiters.Add(owners[3], 0x9000, 2, Nx2SemaphoreWaiters::AcquireGequal);
    waiters.OnWrite(0x100c, 1);
    REQUIRE_FALSE(owners[3].Latched());
    waiters.OnWrite(0x9000, 2);
    REQUIRE(owners[3].Latched());

    waiters.Remove(owners[5]);
    waiters.Add(extra, 0x8000, 1, Nx2SemaphoreWaiters::AcquireGequal);
    waiters.OnWrite(0x8000, 1);
    REQUIRE(extra.Latched());
    REQUIRE_FALSE(owners[5].Latched());
}

TEST_CASE("Semaphore registrations end with either the channel or registry", "[video_core][nx2_waiters]") {
    Latch survivor;
    {
        Nx2SemaphoreWaiters waiters;
        {
            Latch channel;
            waiters.Add(channel, 0x1000, 1, Nx2SemaphoreWaiters::AcquireEqual);
            REQUIRE_FALSE(channel.Latched());
        }
        waiters.OnWrite(0x1000, 1);
        waiters.Add(survivor, 0x1000, 1, Nx2SemaphoreWaiters::AcquireEqual);
        waiters.OnWrite(0x1000, 1);
        REQUIRE(survivor.Latched());
    }
    CHECK_FALSE(survivor.Latched());
}

TEST_CASE("Destroying a semaphore waiter is serialized with release writes",
          "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    std::atomic<u32> writes{};
    std::jthread writer([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            waiters.OnWrite(0x1000, 1);
            writes.fetch_add(1, std::memory_order_release);
            writes.notify_one();
        }
    });
    for (u32 round = 0; round < 1000; ++round) {
        Latch channel;
        waiters.Add(channel, 0x1000, 1, Nx2SemaphoreWaiters::AcquireEqual);
        u32 observed = writes.load(std::memory_order_acquire);
        for (u32 poll = 0; poll < 8 && !channel.Latched(); ++poll) {
            writes.wait(observed, std::memory_order_acquire);
            observed = writes.load(std::memory_order_acquire);
        }
        REQUIRE(channel.Latched());
    }
}

TEST_CASE("Semaphore waiters see what the writer wrote before the latch",
          "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    constexpr u32 Rounds = 2000;
    std::array<u32, 4> words{};
    std::atomic<u32> round{};
    std::jthread writer([&] {
        for (u32 value = 1; value <= Rounds; ++value) {
            while (round.load(std::memory_order_acquire) < value) {
                std::this_thread::yield();
            }
            for (u32 word = 0; word < words.size(); ++word) {
                words[word] = value;
                waiters.OnWrite(0x1000 + word * 4, value);
            }
        }
    });
    std::array<Latch, 4> owners{};
    for (u32 value = 1; value <= Rounds; ++value) {
        for (u32 word = 0; word < words.size(); ++word) {
            waiters.Add(owners[word], 0x1000 + word * 4, value, Nx2SemaphoreWaiters::AcquireGequal);
        }
        round.store(value, std::memory_order_release);
        for (u32 word = 0; word < words.size(); ++word) {
            while (!owners[word].Latched()) {
                std::this_thread::yield();
            }
            REQUIRE(words[word] == value);
            waiters.Remove(owners[word]);
        }
    }
}
