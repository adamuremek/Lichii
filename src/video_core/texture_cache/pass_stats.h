// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon {

/// EDEN_NX2_PASS_STATS=1: cuts each frame into passes (runs of draws into the same render
/// targets, runs of compute dispatches) and records which earlier pass of the frame wrote each
/// image a pass uses. Every 600 frames it logs how many passes a frame has, how they depend on
/// each other and how long the frame would take with the passes run on 2, 4, 8 or any number of
/// threads, in draws. It measures what a back that runs passes in parallel could gain; it
/// changes nothing. The cost of looking images and buffers up in the caches is not modelled.
class PassStats {
public:
    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    void Read(u32 image) {
        if (enabled) {
            reads.push_back(image);
        }
    }

    void Write(u32 image) {
        if (enabled) {
            writes.push_back(image);
        }
    }

    /// The draw being prepared renders into the targets of `serial`.
    void Targets(u64 serial) {
        if (!enabled) {
            return;
        }
        if (passes.empty() || passes.back().compute || passes.back().serial != serial) {
            passes.push_back(Pass{.serial = serial});
        }
        Flush();
    }

    void Draw() {
        if (enabled && !passes.empty()) {
            ++passes.back().work;
        }
    }

    void Compute() {
        if (!enabled) {
            return;
        }
        if (passes.empty() || !passes.back().compute) {
            passes.push_back(Pass{.compute = true});
        }
        ++passes.back().work;
        Flush();
    }

    void EndFrame() {
        if (!enabled) {
            return;
        }
        Flush();
        Account();
        passes.clear();
        writer.clear();
        if (++frames < 600) {
            return;
        }
        const double n = static_cast<double>(frames);
        const auto speedup = [this](size_t index) {
            return total.span[index] != 0
                       ? static_cast<double>(total.work) / static_cast<double>(total.span[index])
                       : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Pass stats per frame: {:.1f} passes ({:.1f} compute), {:.1f} draws and "
                 "dispatches, {:.1f} dependencies between passes, {:.1f} passes with none | "
                 "largest pass {:.1f}% of the frame | frame length in draws: serial {:.0f}, "
                 "2 threads {:.0f} ({:.2f}x), 4 threads {:.0f} ({:.2f}x), 8 threads {:.0f} "
                 "({:.2f}x), unlimited {:.0f} ({:.2f}x)",
                 total.passes / n, total.compute_passes / n, total.work / n, total.edges / n,
                 total.independent / n,
                 total.work != 0 ? 100.0 * static_cast<double>(total.largest) /
                                       static_cast<double>(total.work)
                                 : 0.0,
                 total.work / n, total.span[0] / n, speedup(0), total.span[1] / n, speedup(1),
                 total.span[2] / n, speedup(2), total.span[3] / n, speedup(3));
        total = {};
        frames = 0;
    }

private:
    struct Pass {
        u64 serial{};
        bool compute{};
        u32 work{};
        std::vector<u32> deps;
    };

    struct Totals {
        u64 passes{};
        u64 compute_passes{};
        u64 work{};
        u64 edges{};
        u64 independent{};
        u64 largest{};
        std::array<u64, 4> span{}; // 2, 4, 8 threads, unlimited
    };

    void Depend(Pass& pass, u32 index, u32 image) {
        const auto it = writer.find(image);
        if (it != writer.end() && it->second != index &&
            std::ranges::find(pass.deps, it->second) == pass.deps.end()) {
            pass.deps.push_back(it->second);
        }
    }

    /// Gives the images used since the last draw or dispatch to the current pass.
    void Flush() {
        if (!passes.empty()) {
            const u32 index = static_cast<u32>(passes.size() - 1);
            Pass& pass = passes.back();
            for (const u32 image : reads) {
                Depend(pass, index, image);
            }
            for (const u32 image : writes) {
                Depend(pass, index, image);
                writer[image] = index;
            }
        }
        reads.clear();
        writes.clear();
    }

    void Account() {
        total.passes += passes.size();
        u64 largest = 0;
        for (const Pass& pass : passes) {
            total.compute_passes += pass.compute ? 1 : 0;
            total.work += pass.work;
            total.edges += pass.deps.size();
            total.independent += pass.deps.empty() ? 1 : 0;
            largest = (std::max<u64>)(largest, pass.work);
        }
        total.largest += largest;
        // Passes start in stream order, each on the thread that is free first, once the passes
        // it depends on have finished.
        static constexpr std::array<size_t, 4> Threads{2, 4, 8, 0};
        for (size_t setup = 0; setup < Threads.size(); ++setup) {
            std::vector<u64> free_at(Threads[setup]);
            finish.assign(passes.size(), 0);
            u64 span = 0;
            for (size_t index = 0; index < passes.size(); ++index) {
                u64 start = 0;
                for (const u32 dep : passes[index].deps) {
                    start = (std::max)(start, finish[dep]);
                }
                if (!free_at.empty()) {
                    const auto thread = std::ranges::min_element(free_at);
                    start = (std::max)(start, *thread);
                    *thread = start + passes[index].work;
                }
                finish[index] = start + passes[index].work;
                span = (std::max)(span, finish[index]);
            }
            total.span[setup] += span;
        }
    }

    bool enabled = std::getenv("EDEN_NX2_PASS_STATS") != nullptr;
    std::vector<Pass> passes;
    std::vector<u32> reads;
    std::vector<u32> writes;
    std::vector<u64> finish;
    std::unordered_map<u32, u32> writer; // image -> the frame's last pass that wrote it
    Totals total;
    u32 frames{};
};

} // namespace VideoCommon
