// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <optional>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"

namespace {
std::array<u32, 3> EmitInstruction(std::optional<u32> method, bool end,
                                   std::optional<u32> read = std::nullopt) {
    std::array<u32, 3> bits{};
    const auto field = [&](u32 offset, u32 count, u32 value) {
        for (u32 bit = 0; bit < count; ++bit) {
            bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
        }
    };
    field(0, 1, end);
    for (u32 lane = 0; lane < 2; ++lane) {
        field(15 + lane * 36, 5, 24);
        field(20 + lane * 36, 5, lane == 0 && read ? 25 : 24);
        field(25 + lane * 36, 5, 24);
    }
    if (method && read) {
        field(10, 5, 22);
        field(30, 16, *read);
        field(66, 16, *method);
        field(82, 3, 6);
        field(85, 4, 1);
    } else if (method) {
        field(30, 16, *method);
        field(82, 3, 5);
        field(85, 4, 3);
    }
    return {bits[2], bits[1], bits[0]};
}
} // namespace

TEST_CASE("Owned native macro FIFOs preserve loads across DMA appends and repeated calls",
          "[video_core][native_macro]") {
    std::vector<u32> code;
    for (const auto instruction : {EmitInstruction(0x158, false), EmitInstruction(0x200, true),
                                   EmitInstruction(0x201, false)}) {
        code.insert(code.end(), instruction.begin(), instruction.end());
    }
    Tegra::TuringMacroJit jit;
    jit.Prepare(code, 1);
    std::array<u32, 0xc00> ram{};
    std::vector<u32> parameters;
    std::vector<u32> reference;
    for (u32 invocation = 0; invocation < 1000; ++invocation) {
        const u32 count = 2 + invocation % 127;
        parameters.push_back(count);
        reference.assign(parameters.begin(), parameters.end());
        const auto run = [&](std::vector<u32>& input) {
            std::vector<std::pair<u32, u32>> writes;
            const auto result = jit.Execute(
                0, input, ram, [](u32) { return 0U; },
                [&](u32 method, u32 value) {
                    writes.emplace_back(method, value);
                    if (method == 0x158) {
                        for (u32 word = 0; word < value; ++word) {
                            input.push_back(invocation * 256 + word);
                        }
                    }
                    return true;
                });
            CHECK(result.success);
            CHECK(result.consumed == 3);
            return writes;
        };
        const auto expected = run(reference);
        const auto actual = run(parameters);
        REQUIRE(actual.size() == 3);
        CHECK(actual == expected);
        CHECK(parameters == reference);
        CHECK(actual[1] == std::pair<u32, u32>{0x200, invocation * 256});
        CHECK(actual[2] == std::pair<u32, u32>{0x201, invocation * 256 + 1});
        parameters.clear();
    }
}

TEST_CASE("Macro batches preserve write order across full batches, reads and DMA",
          "[video_core][native_macro]") {
    constexpr u32 write_count = 40;
    constexpr u32 source = 0x200 + write_count - 1;
    std::vector<u32> code;
    const auto append = [&](const std::array<u32, 3>& instruction) {
        code.insert(code.end(), instruction.begin(), instruction.end());
    };
    for (u32 i = 0; i < write_count; ++i) {
        append(EmitInstruction(0x200 + i, false));
    }
    append(EmitInstruction(0x300, false, source));
    append(EmitInstruction(0x158, false));
    append(EmitInstruction(0x400, true));
    append(EmitInstruction(0x401, false));

    Tegra::TuringMacroJit jit;
    for (u64 generation = 1; generation <= 2; ++generation) {
        const u32 first_method = generation == 1 ? 0x200 : 0x240;
        const auto first = EmitInstruction(first_method, false);
        std::ranges::copy(first, code.begin());
        for (const bool batch : {false, true, false, true}) {
            jit.Prepare(code, generation, batch);
            std::array<u32, 0xc00> ram{};
            std::array<u32, 0xe00> registers{};
            std::vector<std::pair<u32, u32>> writes;
            std::vector<u32> fifo;
            for (u32 i = 0; i < write_count; ++i) {
                fifo.push_back(101 + i);
            }
            fifo.push_back(2);
            const auto result = jit.Execute(
                0, fifo, ram, [&](u32 method) { return registers[method]; },
                [&](u32 method, u32 value) {
                    writes.emplace_back(method, value);
                    if (method == 0x158) {
                        CHECK(registers[0x300] == 140);
                        fifo.reserve(fifo.capacity() + 1);
                        fifo.push_back(0xabc);
                        fifo.push_back(0xdef);
                    }
                    registers[method] = value;
                    return true;
                });
            REQUIRE(result.success);
            REQUIRE(result.consumed == write_count + 3);
            REQUIRE(writes.size() == write_count + 4);
            for (u32 i = 0; i < write_count; ++i) {
                CHECK(writes[i] == std::pair<u32, u32>{i == 0 ? first_method : 0x200 + i, 101 + i});
            }
            CHECK(writes[write_count] == std::pair<u32, u32>{0x300, 140});
            CHECK(writes[write_count + 1] == std::pair<u32, u32>{0x158, 2});
            CHECK(writes[write_count + 2] == std::pair<u32, u32>{0x400, 0xabc});
            CHECK(writes[write_count + 3] == std::pair<u32, u32>{0x401, 0xdef});
        }
    }
}

TEST_CASE("Macro batches apply earlier writes when a macro fails", "[video_core][native_macro]") {
    for (const u32 failing_method : {0x201, 0xe00}) {
        std::vector<u32> code;
        for (const auto instruction :
             {EmitInstruction(0x200, true), EmitInstruction(failing_method, false)}) {
            code.insert(code.end(), instruction.begin(), instruction.end());
        }
        Tegra::TuringMacroJit jit;
        for (const bool batch : {false, true}) {
            jit.Prepare(code, 1, batch);
            std::array<u32, 0xc00> ram{};
            std::vector<u32> fifo{123};
            if (failing_method == 0xe00) {
                fifo.push_back(456);
            }
            std::vector<std::pair<u32, u32>> writes;
            const auto result = jit.Execute(
                0, fifo, ram, [](u32) { return 0U; },
                [&](u32 method, u32 value) {
                    if (method >= 0xe00) {
                        return false;
                    }
                    writes.emplace_back(method, value);
                    return true;
                });
            CHECK_FALSE(result.success);
            CHECK(result.error == (failing_method == 0xe00 ? "unsupported emitted method"
                                                           : "parameter FIFO underflow"));
            CHECK(writes == std::vector<std::pair<u32, u32>>{{0x200, 123}});
        }
    }
}

TEST_CASE("Macro callback cost by number of emitted writes", "[.macro_batch_benchmark]") {
    constexpr u32 calls = 400'000;
    for (const u32 count : {1, 3, 8, 32}) {
        std::vector<u32> code;
        for (u32 i = 0; i < count; ++i) {
            const auto instruction = EmitInstruction(0x200 + i, count == 1 || i + 2 == count);
            code.insert(code.end(), instruction.begin(), instruction.end());
        }
        if (count == 1) {
            const auto delay = EmitInstruction(std::nullopt, false);
            code.insert(code.end(), delay.begin(), delay.end());
        }
        Tegra::TuringMacroJit jit;
        for (u32 trial = 0; trial < 5; ++trial) {
            std::array<double, 2> timing{};
            for (u32 step = 0; step < 2; ++step) {
                const bool batch = (trial + step) % 2 != 0;
                jit.Prepare(code, 1, batch);
                std::array<u32, 0xc00> ram{};
                std::vector<u32> fifo(count, 7);
                u64 checksum{};
                size_t failures{};
                const auto begin = std::chrono::steady_clock::now();
                for (u32 call = 0; call < calls; ++call) {
                    fifo[0] = call;
                    const auto result = jit.Execute(
                        0, fifo, ram, [](u32) { return 0U; },
                        [&](u32 method, u32 value) {
                            checksum += method + value;
                            return true;
                        });
                    failures += !result.success;
                }
                const std::chrono::duration<double, std::nano> elapsed =
                    std::chrono::steady_clock::now() - begin;
                timing[batch] = elapsed.count() / calls;
                const u64 expected =
                    u64{calls} * (count * 0x200 + count * (count - 1) / 2 + (count - 1) * 7) +
                    u64{calls} * (calls - 1) / 2;
                CHECK(failures == 0);
                CHECK(checksum == expected);
            }
            std::printf("%u writes trial %u: %.2f -> %.2f ns per macro\n", count, trial + 1,
                        timing[0], timing[1]);
        }
    }
}
#endif
