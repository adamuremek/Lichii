#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame_budget  # noqa: E402


def line(t, message):
    return f"[{t:11.6f}] Render.Vulkan <Info> video_core\\renderer_vulkan\\x.cpp:1:F: {message}\n"


def cpu(t, frames, guest_ms, back_ms, wall_ms=1000.0):
    roles = "".join(f" guest_core_{i}_host_cpu_ms={guest_ms:.3f} guest_core_{i}_host_cpu_ms_per_frame=0"
                    for i in range(6))
    return line(t, f"DKB cpu through_frame=1 frames={frames} window_wall_ms={wall_ms:.3f}{roles} "
                   f"guest_single_core_host=inactive gpu_command_cpu_ms=100.000 "
                   f"gpu_back_cpu_ms={back_ms:.3f} vulkan_worker_cpu_ms=50.000")


def gpu(t, busy_ms, submits):
    return line(t, f"DKB gpu_frame renderer_frame=1 last_submission_tick=1 gpu_frame_span_ms=20.0 "
                   f"gpu_submission_sum_ms={busy_ms} measured_submissions={submits} dropped_submissions=0")


LOG = (
    "[   0.000800] Frontend <Info> qt_common.cpp:245:Init: Lichii Version: Test | nx2-abc\n"
    "[   1.000000] Frontend <Info> main_window.cpp:2363:BootGame: Booting game: 0400486024DEC000 | Zelda | \n"
    + cpu(2.0, 60, 400, 100) + gpu(1.9, 1.5, 8)
    + cpu(3.0, 40, 950, 500) + gpu(2.9, 10.0, 60)
    + cpu(4.0, 40, 500, 920) + gpu(3.9, 9.0, 50)
    + cpu(5.0, 40, 500, 500) + gpu(4.9, 5.0, 40)
    + cpu(6.0, 5, 900, 900)
    + line(6.1, "Nx2StutterSummary window_s=10.00 frames=165 over33=1 over100=1 hist=16:1 "
                "cat_ms=texture:400.0,buffer:200.0 builds=gfx:0,compute:0 "
                "sites=image_create:100x500.0,memory_allocate:10x100.0")
    + line(6.2, "Fast draw per second: 100 delta | full: 1 state changed")
    + line(6.2, "Fast draw ineligible per second: | prepare us per draw: full 5.00, delta 1.00, "
                "all 4.00 (1000 draws/s)")
    + line(6.3, "GPU back thread per second: | other 5.0 ms (0/s) | split wait 300.0 ms (10/s) "
                "| method 150.0 ms (4000/s) | bind stage buffers 70.0 ms (900/s) | buffer downloads 0 KiB "
                "in 0 copies (0 KiB on pages MME DMA reads) | GPU waits by zone: fence 2.0 ms (5/s) "
                "| host_cpu_busy_ms_per_s=600.0 window_wall_ms=10000.0")
    + "[   6.400000] Kernel <Info> core/hle/kernel/guest_profile.cpp:1:Enter: Guest core 2 per second: "
      "| kernel 10.0 ms | jit 600.0 ms (900/s) | svc 100.0 ms (5000/s) | idle spin 200.0 ms (8000/s) "
      "| idle wait 90.0 ms | host_cpu_busy_ms_per_s=900.0 window_wall_ms=10000.0 "
      "| svcs: SleepThread yield 60.0 ms (3000/s), WaitSynchronization 30.0 ms (200/s)\n"
)


class FrameBudgetTest(unittest.TestCase):
    def run_tool(self, *args):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(LOG)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(frame_budget.main([f.name, *args]), 0)
            return out.getvalue()
        finally:
            os.unlink(f.name)

    def test_classifies_each_second(self):
        result = json.loads(self.run_tool("--json"))
        verdicts = [r["verdict"] for r in result["per_second"]]
        self.assertEqual(verdicts, ["on target", "guest CPU core saturated",
                                    "GPU back thread saturated",
                                    "no thread saturated (waits or sync)", "hitch or loading"])
        self.assertEqual(result["slow_seconds"], 3)
        self.assertEqual(result["hitch_seconds"], 1)
        self.assertEqual(result["program"], "0400486024DEC000")
        self.assertEqual(result["build"], "Test | nx2-abc")
        self.assertAlmostEqual(result["per_second"][1]["host_gpu_ms"], 10.0)
        self.assertEqual(result["per_second"][1]["submits_per_frame"], 60)
        # Hitches stay out of the averages.
        self.assertAlmostEqual(result["threads"]["gpu_back"]["slow"]["cpu_ms_per_frame"],
                               (12.5 + 23.0 + 12.5) / 3)

    def test_ten_second_lines_follow_the_slow_seconds(self):
        result = json.loads(self.run_tool("--json"))
        self.assertEqual(result["slow_sites"][0], ["image_create", 50.0])
        self.assertEqual(result["slow_categories"][0], ["texture", 40.0])
        zones = dict(result["slow_zones"]["GPU back thread"])
        self.assertNotIn("split wait", zones)
        self.assertEqual(zones["method"], 150.0)
        self.assertEqual(result["slow_waits"]["GPU back thread"], [["GPU waits by zone: fence", 2.0]])
        self.assertEqual(result["draws"]["slow"]["delta_percent"], 10.0)
        self.assertEqual(result["missing"], ["EDEN_NX2_HOSTTIMING_STATS=1"])
        guest = result["guest_cores"]
        self.assertEqual(guest["states_ms_per_s"]["idle spin"], 200.0)
        self.assertEqual(guest["svcs"][0], ["SleepThread yield", 60.0])

    def test_markdown_report(self):
        text = self.run_tool("--windows")
        self.assertIn("| guest CPU core saturated | 1 |", text)
        self.assertIn("## GPU back thread: busiest zones in slow windows", text)
        self.assertIn("## Per second", text)
        self.assertIn("System calls: SleepThread yield 60.0, WaitSynchronization 30.0", text)

    def test_start_and_end_select_seconds(self):
        result = json.loads(self.run_tool("--json", "--start-s", "2.5", "--end-s", "3.5"))
        self.assertEqual([r["t"] for r in result["per_second"]], [3.0])


if __name__ == "__main__":
    unittest.main()
