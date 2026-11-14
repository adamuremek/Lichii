// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/uniform_stream_reuse.h"

namespace {
void Check(bool condition, const char* message) {
    INFO(message);
    REQUIRE(condition);
}

struct Memory {
    std::array<u8, 8192> bytes{};
    u32 gathered{}; ///< Reads through ReadBlockUnsafe
    u32 in_place{}; ///< Ranges GetSpan handed out
    void ReadBlockUnsafe(u64 address, void* destination, u32 size) {
        Check(address + size <= bytes.size(), "read outside guest memory");
        std::memcpy(destination, bytes.data() + address, size);
        ++gathered;
    }
    /// As the device memory manager: a range in one piece of host memory, here one that doesn't
    /// cross a 4 KiB page (as if the pages after it were mapped elsewhere).
    u8* GetSpan(u64 address, size_t size) {
        if (address / 4096 != (address + size - 1) / 4096) {
            return nullptr;
        }
        ++in_place;
        return bytes.data() + address;
    }
};

struct Runtime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef last{}, bound{};
    u64 tick{1};
    u64 CurrentTick() const {
        return tick;
    }
    StreamUniformRef LastStreamUniformRef() const {
        return last;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        last = bound = StreamUniformRef{allocations.size() - 1};
        return allocations.back();
    }
    void RebindStreamUniform(StreamUniformRef ref, u32 size) {
        Check(ref.allocation < allocations.size(), "reused a recycled allocation");
        Check(allocations[ref.allocation].size() == size, "reused wrong allocation size");
        bound = ref;
    }
    void NextTick() {
        ++tick;
        allocations.clear();
    }
};

void CheckBound(const Runtime& runtime, const Memory& memory, u64 address, u32 size) {
    const auto& bytes = runtime.allocations.at(runtime.bound.allocation);
    Check(bytes.size() == size &&
              std::memcmp(bytes.data(), memory.bytes.data() + address, size) == 0,
          "bound uniform data differs from guest memory");
}
} // namespace

TEST_CASE("Uniform stream reuse preserves data through mutation and session resets",
          "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    auto bind = [&](size_t stage, u32 slot, u64 address, u32 size) {
        Check(BindSmallUniformReusing(runtime, memory, stage, slot, slot, address, size),
              "valid small uniform rejected");
        CheckBound(runtime, memory, address, size);
    };
    bind(0, 0, 0, 64);
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "identical bytes were copied twice");
    memory.bytes[7] = 23;
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "changed bytes reused stale data");
    bind(0, 1, 0, 64);
    bind(1, 0, 0, 64);
    Check(runtime.allocations.size() == 4, "stage or slot shared the wrong allocation");
    runtime.NextTick();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 1, "a new command buffer reused an old allocation");
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    bind(0, 0, 0, 64);
    Check(runtime.allocations.size() == 2, "session reset retained an old allocation");
    bind(4, 17, 0, 4096);
    const auto before = runtime.allocations.size();
    for (auto [stage, slot, size] :
         {std::array<u32, 3>{5, 0, 64}, {0, 18, 64}, {0, 0, 0}, {0, 0, 4097}}) {
        Check(!BindSmallUniformReusing(runtime, memory, stage, slot, slot, 0, size),
              "invalid uniform range did not use the fallback");
    }
    Check(runtime.allocations.size() == before, "fallback allocated stream memory");

    u32 random = 17;
    for (u32 index = 0; index < 10000; ++index) {
        random = random * 1664525U + 1013904223U;
        if (index % 37 == 0)
            runtime.NextTick();
        if (index % 503 == 0)
            VideoCommon::ResetUniformStreamReuse<Runtime>();
        const u32 size = 1 + random % 4096;
        const u64 address = (random >> 12) % (memory.bytes.size() - 4096);
        if (index % 3 == 0)
            memory.bytes[address + random % size] ^= u8{0x5a};
        const size_t stage = (random >> 20) % 5;
        const u32 slot = (random >> 24) % 18;
        bind(stage, slot, address, size);
        bind(stage, slot, address, size);
    }
    Check(memory.gathered > 1000 && memory.in_place > 1000,
          "the reads did not take both the gathered and the in-place path");
}

TEST_CASE("Uniform stream keeps require the same range and live allocation", "[video_core]") {
    using VideoCommon::BindSmallUniformReusing;
    using VideoCommon::RebindKeptUniform;
    Memory memory;
    Runtime runtime;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 0, 64));
    REQUIRE(BindSmallUniformReusing(runtime, memory, 0, 0, 0, 0, 64));
    const u32 gathered = memory.gathered;
    const u32 in_place = memory.in_place;
    REQUIRE(RebindKeptUniform(runtime, 0, 0, 0, 64));
    CheckBound(runtime, memory, 0, 64);
    CHECK(memory.gathered == gathered);
    CHECK(memory.in_place == in_place);
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 64, 64));
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 0, 32));
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 1, 0, 64));
    CHECK_FALSE(RebindKeptUniform(runtime, 1, 0, 0, 64));
    CHECK_FALSE(RebindKeptUniform(runtime, 5, 0, 0, 64));
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 18, 0, 64));
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 0, 0));
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 0, 4097));

    // Equal bytes at a different address reuse the allocation, but update the keep's range.
    REQUIRE(BindSmallUniformReusing(runtime, memory, 0, 0, 0, 64, 64));
    CHECK(runtime.allocations.size() == 1);
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 0, 64));
    REQUIRE(RebindKeptUniform(runtime, 0, 0, 64, 64));
    CHECK(VideoCommon::KeptUniformHolds<Runtime>(memory, 0, 0, 64, 64));
    memory.bytes[67] = 9;
    CHECK_FALSE(VideoCommon::KeptUniformHolds<Runtime>(memory, 0, 0, 64, 64));

    runtime.NextTick();
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 64, 64));
    REQUIRE(BindSmallUniformReusing(runtime, memory, 0, 0, 0, 64, 64));
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    CHECK_FALSE(RebindKeptUniform(runtime, 0, 0, 64, 64));
}

TEST_CASE("Compute stream reuse preserves dispatch snapshots and leaves GPU patches fresh",
          "[video_core]") {
    using VideoCommon::BindSmallComputeUniformReusing;
    using VideoCommon::BindSmallUniformReusing;
    Memory memory;
    Runtime runtime;
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    constexpr u32 size = 256;
    REQUIRE(BindSmallUniformReusing(runtime, memory, 0, 0, 0, 0, size));
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    const auto first = runtime.bound;
    REQUIRE(first.allocation == 1);
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    REQUIRE(runtime.bound.allocation == first.allocation);

    REQUIRE_FALSE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, true));
    auto patched = runtime.BindMappedUniformBuffer(0, 0, size);
    memory.ReadBlockUnsafe(0, patched.data(), size);
    patched[4] = 0xA5;
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    REQUIRE(runtime.bound.allocation == first.allocation);
    CheckBound(runtime, memory, 0, size);

    memory.bytes[4] = 0x59;
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    REQUIRE(runtime.bound.allocation == 3);
    CHECK(runtime.allocations[first.allocation][4] == 0);
    CheckBound(runtime, memory, 0, size);
    runtime.NextTick();
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    REQUIRE(runtime.allocations.size() == 1);
    CheckBound(runtime, memory, 0, size);
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    REQUIRE(BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0, size, false));
    REQUIRE(runtime.allocations.size() == 2);
    REQUIRE_FALSE(BindSmallComputeUniformReusing(runtime, memory, 8, 0, 0, size, false));
}

TEST_CASE("Compute stream reuse allocation benchmark", "[.compute_uniform_benchmark]") {
    struct BenchMemory {
        std::array<u8, 256> bytes{};
        const u8* GetSpan(u64 address, size_t) const {
            return bytes.data() + address;
        }
        void ReadBlockUnsafe(u64 address, void* dest, u32 size) const {
            std::memcpy(dest, bytes.data() + address, size);
        }
    } memory;
    struct BenchRuntime {
        struct StreamUniformRef {
            size_t allocation;
        };
        u64 CurrentTick() const { return tick; }
        StreamUniformRef LastStreamUniformRef() const { return last; }
        std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
            last = bound = StreamUniformRef{next++};
            return std::span{allocations[last.allocation]}.first(size);
        }
        void RebindStreamUniform(StreamUniformRef ref, u32) { bound = ref; }
        std::vector<std::array<u8, 256>> allocations{1024};
        StreamUniformRef last{}, bound{};
        size_t next{};
        u64 tick{};
    } runtime;
    constexpr u32 size = 256;
    constexpr size_t dispatches = 1024 * 512;
    for (const bool changing : {false, true}) {
        for (size_t run = 0; run < 10; ++run) {
            const bool reuse = (run / 2 + run % 2) % 2 != 0;
            size_t allocations = 0;
            const auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < dispatches; ++i) {
                if ((i & 1023) == 0) {
                    allocations += runtime.next;
                    runtime.next = 0;
                    ++runtime.tick;
                }
                if (changing && (i & 15) == 0) {
                    memory.bytes[4] ^= 1;
                }
                if (reuse) {
                    (void)VideoCommon::BindSmallComputeUniformReusing(runtime, memory, 0, 0, 0,
                                                                     size, false);
                } else {
                    auto span = runtime.BindMappedUniformBuffer(0, 0, size);
                    memory.ReadBlockUnsafe(0, span.data(), size);
                }
            }
            const auto elapsed = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - start).count();
            allocations += runtime.next;
            runtime.next = 0;
            REQUIRE(allocations == (reuse ? (changing ? dispatches / 16 : dispatches / 1024)
                                           : dispatches));
            REQUIRE(runtime.allocations[runtime.bound.allocation] == memory.bytes);
            std::printf("Compute uniforms %s %s: %.2f ns per bind, %zu allocations\n",
                        changing ? "changing" : "stable", reuse ? "reuse" : "original",
                        elapsed / dispatches, allocations);
        }
    }
}
