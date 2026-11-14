// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ordered (LDAR, LDAPR) and exclusive (LDAXR) loads through the JIT's fastmem path. On x86-64 a
// load is already ordered before later loads and stores, so these are plain loads: they must not
// write the page they read, which a locked read-modify-write did.
//
// The microbenchmark is hidden from normal runs; start it with
//   tests "[.jit_ordered_load_benchmark]"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "common/common_types.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace {

constexpr u32 AddressSpaceBits = 20;
constexpr size_t ArenaSize = size_t{1} << AddressSpaceBits;
constexpr size_t PageSize = 0x10000; // A multiple of every host's page size
constexpr u64 Writable = PageSize;
constexpr u64 ReadOnly = 2 * PageSize;
constexpr u64 CodeBase = 8 * PageSize;

constexpr u32 Svc = 0xD4000001;
constexpr u32 AddX1X1One = 0x91000421;  // ADD X1, X1, #1
constexpr u32 SubsX2X2One = 0xF1000442; // SUBS X2, X2, #1
constexpr u32 BranchNotEqualBack2 = 0x54FFFFC1; // B.NE to two instructions back
constexpr u32 CasaW3W4 = 0x88E37C04; // CASA W3, W4, [X0]
constexpr u32 CaslW3W4 = 0x88A3FC04; // CASL W3, W4, [X0]

constexpr u32 Ldar(u32 size, u32 rn, u32 rt) {
    return 0x08DFFC00 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Ldapr(u32 size, u32 rn, u32 rt) {
    return 0x38BFC000 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Ldaxr(u32 size, u32 rn, u32 rt) {
    return 0x085FFC00 | (size << 30) | (rn << 5) | rt;
}
constexpr u32 Stlxr(u32 size, u32 rs, u32 rn, u32 rt) {
    return 0x0800FC00 | (size << 30) | (rs << 16) | (rn << 5) | rt;
}
/// LDADDAL: adds Rs to the memory at Rn and loads the old value into Rt.
constexpr u32 Ldaddal(u32 size, u32 rs, u32 rn, u32 rt) {
    return 0x38E00000 | (size << 30) | (rs << 16) | (rn << 5) | rt;
}

/// Host memory the guest addresses [0, ArenaSize) map onto.
class Arena {
public:
    Arena() {
#ifdef _WIN32
        base = static_cast<u8*>(
            VirtualAlloc(nullptr, ArenaSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        REQUIRE(base != nullptr);
#else
        void* const mapping = mmap(nullptr, ArenaSize, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(mapping != MAP_FAILED);
        base = static_cast<u8*>(mapping);
#endif
    }
    ~Arena() {
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, ArenaSize);
#endif
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void ProtectReadOnly(u64 address) {
#ifdef _WIN32
        DWORD old{};
        REQUIRE(VirtualProtect(base + address, PageSize, PAGE_READONLY, &old) != 0);
#else
        REQUIRE(mprotect(base + address, PageSize, PROT_READ) == 0);
#endif
    }

    u8* base{};
};

class Environment final : public Dynarmic::A64::UserCallbacks {
public:
    Dynarmic::A64::Jit* jit{};
    u8* memory{};
    std::vector<u32> code;
    /// Memory accesses the JIT made through a callback instead of inline.
    u64 callbacks{};
    u64 alias_address = ~u64{};
    u64 alias_target{};
    u32 retry_writes{};
    bool abort_writes{};

    std::optional<u32> MemoryReadCode(u64 address) override {
        const u64 index = (address - CodeBase) / 4;
        if (address < CodeBase || index >= code.size()) {
            return std::nullopt;
        }
        return code[index];
    }
    u64 MemoryRead(u64 address, size_t size) override {
        ++callbacks;
        u64 value{};
        std::memcpy(&value, MemoryPointer(address), size);
        return value;
    }
    Dynarmic::A64::Vector MemoryRead128(u64) override {
        FAIL("Unexpected 128-bit read");
        return {};
    }
    void MemoryWrite(u64 address, u64 value, size_t size) override {
        ++callbacks;
        std::memcpy(memory + address, &value, size);
    }
    void MemoryWrite128(u64, Dynarmic::A64::Vector) override {
        FAIL("Unexpected 128-bit write");
    }
    bool MemoryWriteExclusive32(u64 address, u32 value, u32 expected) override {
        return ExclusiveWrite(address, value, expected);
    }
    bool MemoryWriteExclusive64(u64 address, u64 value, u64 expected) override {
        return ExclusiveWrite(address, value, expected);
    }
    void CallSVC(u32) override {
        jit->HaltExecution();
    }
    void ExceptionRaised(u64, Dynarmic::A64::Exception) override {
        FAIL("Unexpected exception");
    }
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override {
        return 1000;
    }
    u64 GetCNTPCT() override {
        return 0;
    }

private:
    u8* MemoryPointer(u64 address) const {
        return memory + (address == alias_address ? alias_target : address);
    }

    template <typename T>
    bool ExclusiveWrite(u64 address, T value, T expected) {
        ++callbacks;
        if (abort_writes) {
            jit->HaltExecution(Dynarmic::HaltReason::MemoryAbort);
            return false;
        }
        if (retry_writes) {
            --retry_writes;
            return false;
        }
        return std::atomic_ref{*reinterpret_cast<T*>(MemoryPointer(address))}
            .compare_exchange_strong(expected, value);
    }
};

/// The emulator's configuration at the default CPU accuracy, over a small arena.
Dynarmic::A64::UserConfig Config(Environment& env, Dynarmic::ExclusiveMonitor& monitor,
                                 size_t processor) {
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    config.enable_cycle_counting = false;
    config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(env.memory);
    config.fastmem_address_space_bits = AddressSpaceBits;
    config.silently_mirror_fastmem = false;
    config.fastmem_exclusive_access = true;
    config.global_monitor = &monitor;
    config.processor_id = static_cast<u8>(processor);
    config.unsafe_optimizations = true;
    config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
    return config;
}

void Store(Arena& arena, u64 address, u64 value) {
    std::memcpy(arena.base + address, &value, sizeof(value));
}
u64 Load(const Arena& arena, u64 address) {
    u64 value{};
    std::memcpy(&value, arena.base + address, sizeof(value));
    return value;
}

void Run(Environment& env, Dynarmic::A64::Jit& jit) {
    env.jit = &jit;
    jit.SetPC(CodeBase);
    jit.Run();
}

} // namespace

TEST_CASE("Ordered loads read a read-only page inline", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, ReadOnly, 0x8877665544332211ULL);
    arena.ProtectReadOnly(ReadOnly);

    Environment env;
    env.memory = arena.base;
    env.code = {Ldar(0, 0, 1),  Ldar(1, 0, 2),  Ldar(2, 0, 3),  Ldar(3, 0, 4), Ldapr(0, 0, 5),
                Ldapr(1, 0, 6), Ldapr(2, 0, 7), Ldapr(3, 0, 8), Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, ReadOnly);
    for (size_t reg = 1; reg <= 8; ++reg) {
        jit.SetRegister(reg, ~0ULL);
    }
    Run(env, jit);

    constexpr std::array<u64, 4> expected{0x11, 0x2211, 0x44332211, 0x8877665544332211ULL};
    for (size_t size = 0; size < expected.size(); ++size) {
        REQUIRE(jit.GetRegister(1 + size) == expected[size]);
        REQUIRE(jit.GetRegister(5 + size) == expected[size]);
    }
    REQUIRE(Load(arena, ReadOnly) == 0x8877665544332211ULL);
#ifdef ARCHITECTURE_x86_64
    REQUIRE(env.callbacks == 0);
#endif
}

TEST_CASE("Exclusive loads read a read-only page inline", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, ReadOnly, 0x0123456789ABCDEFULL);
    arena.ProtectReadOnly(ReadOnly);

    Environment env;
    env.memory = arena.base;
    env.code = {Ldaxr(3, 0, 1), Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, ReadOnly);
    Run(env, jit);

    REQUIRE(jit.GetRegister(1) == 0x0123456789ABCDEFULL);
#ifdef ARCHITECTURE_x86_64
    REQUIRE(env.callbacks == 0);
#endif
}

TEST_CASE("Exclusive and atomic updates still store through plain loads", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, Writable, 41);

    Environment env;
    env.memory = arena.base;
    // An exclusive increment, then an atomic add of X3 that loads the old value into X4.
    env.code = {Ldaxr(3, 0, 1), AddX1X1One, Stlxr(3, 2, 0, 1), Ldaddal(3, 3, 0, 4), Ldar(3, 0, 5),
                Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, Writable);
    jit.SetRegister(2, ~0ULL);
    jit.SetRegister(3, 100);
    Run(env, jit);

    REQUIRE(jit.GetRegister(1) == 42);
    REQUIRE(jit.GetRegister(2) == 0);
    REQUIRE(jit.GetRegister(4) == 42);
    REQUIRE(jit.GetRegister(5) == 142);
    REQUIRE(Load(arena, Writable) == 142);
}

TEST_CASE("Each processor keeps its own reservation", "[jit_ordered_loads]") {
    Arena arena;
    Store(arena, Writable, 7);

    // Inline exclusives on the last processor of the monitor.
    Environment env;
    env.memory = arena.base;
    env.code = {Ldaxr(3, 0, 1), AddX1X1One, Stlxr(3, 2, 0, 1), Svc};
    Dynarmic::ExclusiveMonitor monitor{4};
    Dynarmic::A64::Jit jit{Config(env, monitor, 3)};
    jit.SetRegister(0, Writable);
    jit.SetRegister(2, ~0ULL);
    Run(env, jit);
    REQUIRE(jit.GetRegister(2) == 0);
    REQUIRE(Load(arena, Writable) == 8);

    // The monitor's own interface, as the kernel uses it: a store through one processor's
    // reservation ends the others' on that address and leaves the rest.
    const auto mark = [&](size_t processor, u64 address) {
        return monitor.ReadAndMark<u64>(processor, address, [&] { return Load(arena, address); });
    };
    const auto store = [&](size_t processor, u64 address) {
        u64 saved{};
        const bool stored =
            monitor.DoExclusiveOperation<u64>(processor, address, [&](u64 expected) {
                saved = expected;
                return true;
            });
        return stored ? saved : ~0ULL;
    };
    Store(arena, Writable + 8, 21);
    REQUIRE(mark(0, Writable) == 8);
    REQUIRE(mark(1, Writable) == 8);
    REQUIRE(mark(2, Writable + 8) == 21);
    REQUIRE(store(1, Writable) == 8);
    REQUIRE(store(0, Writable) == ~0ULL);
    REQUIRE(store(2, Writable + 8) == 21);
    REQUIRE(store(2, Writable + 8) == ~0ULL);
}

TEST_CASE("Ordered compare-and-swap updates exactly one 32-bit word", "[jit_ordered_loads]") {
    Arena arena;
    constexpr u64 sentinel = 0xB7F0A6D100000000ULL;
    Store(arena, Writable, sentinel | 12);
    Environment env;
    env.memory = arena.base;
    env.code = {CasaW3W4, CaslW3W4, Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    jit.SetRegister(0, Writable);
    jit.SetRegister(3, 12);
    jit.SetRegister(4, 37);
    Run(env, jit);
    CHECK(jit.GetRegister(3) == 37);
    CHECK(Load(arena, Writable) == (sentinel | 37));
    jit.SetRegister(4, 41);
    Run(env, jit);
    CHECK(jit.GetRegister(3) == 41);
    CHECK(Load(arena, Writable) == (sentinel | 41));
}

#ifdef ARCHITECTURE_x86_64
TEST_CASE("Native LSE atomics preserve widths, CAS results and discarded destinations",
          "[jit_ordered_loads][native_lse]") {
    Arena arena;
    constexpr u64 sentinel = 0xB7F0A6D100000000ULL;
    Store(arena, Writable, sentinel | 0xfffffffe);
    Store(arena, Writable + 8, 0x100000000ULL);

    Environment env;
    env.memory = arena.base;
    env.code = {Ldaddal(2, 1, 0, 5), CasaW3W4, CaslW3W4, Ldaddal(2, 1, 0, 31),
                Ldaddal(3, 1, 8, 6), CasaW3W4 | (1u << 30) | (8u << 5), Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    auto config = Config(env, monitor, 0);
    config.native_lse_atomics = true;
    Dynarmic::A64::Jit jit{config};
    jit.SetRegister(0, Writable);
    jit.SetRegister(1, 3);
    jit.SetRegister(3, 1);
    jit.SetRegister(4, 7);
    jit.SetRegister(8, Writable + 8);
    Run(env, jit);

    CHECK(jit.GetRegister(3) == 0x100000003ULL);
    CHECK(jit.GetRegister(5) == 0xfffffffe);
    CHECK(jit.GetRegister(6) == 0x100000000ULL);
    CHECK(Load(arena, Writable) == (sentinel | 10));
    CHECK(Load(arena, Writable + 8) == 0x100000003ULL);
    CHECK(env.callbacks == 0);
}

TEST_CASE("Native LSE fastmem faults preserve atomic callback retries and aborts",
          "[jit_ordered_loads][native_lse]") {
    Arena arena;
    arena.ProtectReadOnly(ReadOnly);
    Store(arena, Writable, 40);

    Environment env;
    env.memory = arena.base;
    env.alias_address = ReadOnly;
    env.alias_target = Writable;
    env.retry_writes = 1;
    env.code = {Ldaddal(3, 1, 0, 4), CasaW3W4, Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    auto config = Config(env, monitor, 0);
    config.native_lse_atomics = true;
    config.check_halt_on_memory_access = true;
    Dynarmic::A64::Jit jit{config};
    jit.SetRegister(0, ReadOnly);
    jit.SetRegister(1, 3);
    jit.SetRegister(3, 43);
    Run(env, jit);
    CHECK(jit.GetRegister(4) == 40);
    CHECK(jit.GetRegister(3) == 43);
    CHECK(Load(arena, Writable) == 40);
    CHECK(env.callbacks == 6);

    env.abort_writes = true;
    jit.SetPC(CodeBase);
    jit.SetRegister(4, 100);
    const auto result = jit.Run();
    CHECK(Dynarmic::Has(result, Dynarmic::HaltReason::MemoryAbort));
    CHECK(jit.GetPC() == CodeBase);
    CHECK(jit.GetRegister(4) == 100);
    CHECK(Load(arena, Writable) == 40);
}

TEST_CASE("Native LSE adds keep every update under contention", "[jit_ordered_loads][native_lse]") {
    constexpr size_t threads = 4;
    constexpr u64 iterations = 100'000;
    Arena arena;
    Dynarmic::ExclusiveMonitor monitor{threads};
    std::atomic<bool> start{};
    std::array<std::thread, threads> workers;
    for (size_t index = 0; index < threads; ++index) {
        workers[index] = std::thread([&, index] {
            Environment env;
            env.memory = arena.base;
            env.code = {Ldaddal(3, 3, 0, 4), SubsX2X2One, BranchNotEqualBack2, Svc};
            auto config = Config(env, monitor, index);
            config.native_lse_atomics = true;
            Dynarmic::A64::Jit jit{config};
            jit.SetRegister(0, Writable);
            jit.SetRegister(2, iterations);
            jit.SetRegister(3, 1);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            Run(env, jit);
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    CHECK(Load(arena, Writable) == threads * iterations);
}
#endif

TEST_CASE("JIT unsigned division preserves ARM results across widths and zero divisors",
          "[jit_unsigned_div]") {
    Arena arena;
    Environment env;
    env.memory = arena.base;
    env.code = {0x1ac10802, 0x9ac10803, Svc}; // udiv w2, w0, w1; udiv x3, x0, x1
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    constexpr std::array<u64, 11> values{
        0, 1, 2, 3, 127, 128, 129, 0xffffffff, 0x100000000, 0x8000000000000000, ~u64{0},
    };
    for (const u64 lhs : values) {
        for (const u64 rhs : values) {
            jit.SetRegister(0, lhs);
            jit.SetRegister(1, rhs);
            Run(env, jit);
            const u32 divisor = static_cast<u32>(rhs);
            CHECK(jit.GetRegister(2) == (divisor ? static_cast<u32>(lhs) / divisor : 0));
            CHECK(jit.GetRegister(3) == (rhs ? lhs / rhs : 0));
        }
    }
}

TEST_CASE("JIT unsigned division in a guest queue wrap loop", "[.jit_unsigned_div_benchmark]") {
    Arena arena;
    Environment env;
    env.memory = arena.base;
    // add w0, w0, #1; udiv w2, w0, w1; msub w0, w2, w1, w0
    // subs x3, x3, #1; b.ne to add; svc #0
    env.code = {0x11000400, 0x1ac10802, 0x1b018040, 0xf1000463, 0x54ffff81, Svc};
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit jit{Config(env, monitor, 0)};
    for (const u32 capacity : {1, 2, 32, 128, 1024}) {
        jit.SetRegister(0, 0);
        jit.SetRegister(1, capacity);
        jit.SetRegister(3, 32);
        Run(env, jit);
        constexpr u64 iterations = 8'000'003;
        jit.SetRegister(0, 0);
        jit.SetRegister(3, iterations);
        const auto begin = std::chrono::steady_clock::now();
        Run(env, jit);
        const std::chrono::duration<double, std::nano> elapsed =
            std::chrono::steady_clock::now() - begin;
        std::printf("queue capacity %u: %.3f ns/step\n", capacity,
                    elapsed.count() / static_cast<double>(iterations));
        CHECK(jit.GetRegister(0) == iterations % capacity);
    }
}

TEST_CASE("Ordered load and atomic microbenchmark", "[.jit_ordered_load_benchmark]") {
    const bool native_lse = std::getenv("TEST_NATIVE_LSE") != nullptr;
    constexpr u64 Iterations = 4'000'000;
    struct Loop {
        const char* name;
        u32 instruction;
    };
    constexpr std::array loops{
        Loop{"LDAR", Ldar(3, 0, 1)},
        Loop{"LDAPR", Ldapr(3, 0, 1)},
        Loop{"LDADDAL", Ldaddal(3, 3, 0, 4)},
        Loop{"CASA", CasaW3W4},
        Loop{"CASL", CaslW3W4},
    };
    struct Case {
        size_t threads;
        bool shared;
    };
    constexpr std::array cases{Case{1, true}, Case{2, true}, Case{4, true}, Case{2, false},
                               Case{4, false}, Case{6, false}};
    for (const Loop& loop : loops) {
        // Sharing one word shows the instruction's own cost between cores; a word each, a page
        // apart, shows what the cores share besides guest memory.
        for (const Case& test : cases) {
            const size_t threads = test.threads;
            const bool shared = test.shared;
            Arena arena;
            Dynarmic::ExclusiveMonitor monitor{threads};
            std::atomic<bool> start{};
            std::vector<double> nanoseconds(threads);
            std::vector<std::thread> workers;
            for (size_t thread = 0; thread < threads; ++thread) {
                workers.emplace_back([&, thread] {
                    Environment env;
                    env.memory = arena.base;
                    env.code = {loop.instruction, SubsX2X2One, BranchNotEqualBack2, Svc};
                    auto config = Config(env, monitor, thread);
                    config.native_lse_atomics = native_lse;
                    Dynarmic::A64::Jit jit{config};
                    jit.SetRegister(0, shared ? Writable : Writable + thread * 0x1000);
                    jit.SetRegister(2, Iterations);
                    jit.SetRegister(3, 1);
                    while (!start.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    const auto begin = std::chrono::steady_clock::now();
                    Run(env, jit);
                    const std::chrono::duration<double, std::nano> elapsed =
                        std::chrono::steady_clock::now() - begin;
                    nanoseconds[thread] = elapsed.count() / static_cast<double>(Iterations);
                });
            }
            start.store(true, std::memory_order_release);
            double total{};
            for (size_t thread = 0; thread < threads; ++thread) {
                workers[thread].join();
                total += nanoseconds[thread];
            }
            std::printf("%-8s %zu thread(s) on %s: %.2f ns per iteration\n", loop.name, threads,
                        shared ? "one word" : "a word each", total / static_cast<double>(threads));
        }
    }
}
