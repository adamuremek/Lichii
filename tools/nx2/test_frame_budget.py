#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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
    + line(6.05, "Frame timing window renderer_frame=601 samples=300 partial=false mean_ms=17.4 "
                 "p50_ms=16.7 p95_ms=31.2 p99_ms=65.0 max_ms=132.5 over_50_ms=4 over_100_ms=1")
    + line(6.06, "Nx2Stutter frame=602 ms=148.0 cause=texture")
    + line(6.1, "Nx2StutterSummary window_s=10.00 frames=165 over33=1 over100=1 hist=16:1 "
                "cat_ms=texture:400.0,buffer:200.0 builds=gfx:0,compute:0 "
                "sites=image_create:100x500.0,memory_allocate:10x100.0")
    + line(6.2, "Fast draw per second: 100 delta | full: 1 state changed | 42 unchanged-handle reuses")
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
    def test_tas_ab_requires_four_complete_ordered_phases(self):
        start = "TAS A/B started experiment=adaptive-submit frames=3600 warmup=120"
        phases = []
        for index, mode in enumerate(("baseline", "candidate", "candidate", "baseline"), 1):
            elapsed = 6_500_000_000 if mode == "candidate" else 13_000_000_000
            phases.append(f"TAS A/B phase experiment=adaptive-submit index={index} mode={mode} "
                          f"frames=780 elapsed_ns={elapsed} p50_ns=9000000 p95_ns=10000000 "
                          "p99_ns=12000000 max_ns=50000000 near_120=400 over_50_ms=0 "
                          "over_100_ms=0")
        end = "TAS A/B completed experiment=adaptive-submit frames=3600"
        entries = list(enumerate([start, *phases, end]))
        result = frame_budget.last_tas_ab(entries)
        self.assertEqual(result["fps"], {"baseline": 60, "candidate": 120})
        self.assertEqual(result["fps_delta"], 60)
        self.assertIn("120.00 FPS", frame_budget.report_tas_ab(result, "Kokiri Forest"))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "eden_log.txt"
            log.write_text("".join(line(t, text) for t, text in entries), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(frame_budget.main([str(log), "--tas-ab", "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue())["fps"], result["fps"])
        self.assertIsNone(frame_budget.last_tas_ab(entries[:-1]))
        for malformed in ([start, *phases[:-1], end],
                          [start, phases[1], phases[0], *phases[2:], end],
                          [start, *phases[:2], "TAS playback interrupted", *phases[2:], end],
                          [start, *phases[:-1], phases[-1].replace("frames=780", "frames=779"), end]):
            self.assertIsNone(frame_budget.last_tas_ab(list(enumerate(malformed))))

    def test_renderer_benchmark_aggregates_each_policy_in_both_orders(self):
        start = "TAS A/B started experiment=renderer frames=7200 warmup=120"
        end = "TAS A/B completed experiment=renderer frames=7200"
        sequence = ("baseline", "adaptive-submit", "both", "raster-state",
                    "raster-state", "both", "adaptive-submit", "baseline")
        intervals = {"baseline": 20_000_000, "adaptive-submit": 16_000_000,
                     "raster-state": 18_000_000, "both": 8_000_000}
        phases = []
        for index, mode in enumerate(sequence, 1):
            ns = intervals[mode]
            near_120 = 780 if mode == "both" else 0
            phases.append(f"TAS A/B phase experiment=renderer index={index} mode={mode} "
                          f"frames=780 elapsed_ns={780 * ns} p50_ns={ns} p95_ns={ns} "
                          f"p99_ns={ns} max_ns={ns} near_120={near_120} over_50_ms=0 "
                          "over_100_ms=0")
        result = frame_budget.last_tas_ab(list(enumerate([start, *phases, end])))
        self.assertEqual(result["experiment"], "renderer")
        self.assertEqual(result["fps"]["baseline"], 50)
        self.assertEqual(result["fps"]["adaptive-submit"], 62.5)
        self.assertAlmostEqual(result["fps"]["raster-state"], 1000 / 18)
        self.assertEqual(result["fps"]["both"], 125)
        self.assertEqual(result["near_120"]["both"], 1)
        report = frame_budget.report_tas_ab(result, "Kokiri Forest")
        self.assertIn("| both | 125.00 FPS | +75.00 FPS | 100.0% |", report)
        for invalid in (
            [start, *phases, end.replace("renderer", "adaptive-submit")],
            [start, *phases[:3], "TAS A/B aborted experiment=renderer because a frame was not rendered",
             *phases[3:], end],
            [start, *phases[:5], phases[5].replace("experiment=renderer", "experiment=adaptive-submit"),
             *phases[6:], end],
        ):
            self.assertIsNone(frame_budget.last_tas_ab(list(enumerate(invalid))))

    def test_texture_reads_benchmark_requires_and_reports_skip_counts(self):
        start = "TAS A/B started experiment=texture-reads frames=3600 warmup=120"
        end = "TAS A/B completed experiment=texture-reads frames=3600"
        phases = []
        for index, mode in enumerate(("baseline", "candidate", "candidate", "baseline"), 1):
            skips = 1234 if mode == "candidate" else 0
            phases.append(f"TAS A/B phase experiment=texture-reads index={index} mode={mode} "
                          f"frames=780 elapsed_ns=13000000000 p50_ns=16000000 p95_ns=17000000 "
                          f"p99_ns=19000000 max_ns=20000000 near_120=0 over_50_ms=0 "
                          f"over_100_ms=0 texture_checks_skipped={skips}")
        result = frame_budget.last_tas_ab(list(enumerate([start, *phases, end])))
        self.assertIn("skipped 2,468 texture cache checks over 1,560 measured candidate frames",
                      frame_budget.report_tas_ab(result, "Kokiri Forest"))
        buffer_run = [text.replace("texture-reads", "buffer-reads").replace(
            "texture_checks_skipped=", "texture_checks_skipped=0 buffer_checks_skipped=")
            for text in [start, *phases, end]]
        buffer_result = frame_budget.last_tas_ab(list(enumerate(buffer_run)))
        self.assertIn("skipped 2,468 buffer cache checks over 1,560 measured candidate frames",
                      frame_budget.report_tas_ab(buffer_result, "Kokiri Forest"))
        acquire_run = [text.replace("buffer-reads", "acquire-waits").replace(
            "buffer_checks_skipped=", "buffer_checks_skipped=0 acquire_waits_shortened=")
            for text in buffer_run]
        acquire_result = frame_budget.last_tas_ab(list(enumerate(acquire_run)))
        self.assertIn("selected 2,468 shorter wait targets over 1,560 measured frames",
                      frame_budget.report_tas_ab(acquire_result, "Kokiri Forest"))
        acquire_run[2] = acquire_run[2].split(" acquire_waits_shortened=")[0]
        self.assertIsNone(frame_budget.last_tas_ab(list(enumerate(acquire_run))))
        buffer_run[2] = buffer_run[2].split(" buffer_checks_skipped=")[0]
        self.assertIsNone(frame_budget.last_tas_ab(list(enumerate(buffer_run))))
        phases[1] = phases[1].split(" texture_checks_skipped=")[0]
        self.assertIsNone(frame_budget.last_tas_ab(list(enumerate([start, *phases, end]))))

    def test_tas_uses_route_identity_and_full_interval_summary(self):
        route = "d0ab000000000001"
        start = f"TAS playback started frames=100 route={route}"
        finish = (f"TAS playback completed frames=100 route={route} elapsed_ns=922000000 "
                  "p50_ns=8000000 p95_ns=8000000 p99_ns=8000000 max_ns=130000000 "
                  "near_120=99 over_50_ms=1 over_100_ms=1")
        parsed = frame_budget.last_tas_playback([(1.0, start), (2.0, finish)])
        self.assertEqual(parsed["route"], route)
        self.assertEqual(parsed["seconds"], 0.922)
        self.assertEqual(parsed["timing"]["near_120"], 99)
        self.assertAlmostEqual(parsed["fps"], 100 / 0.922)
        self.assertIsNone(frame_budget.last_tas_playback([(1.0, start), (2.0, finish.replace(
            route, "d0ab000000000002"))]))

    def test_tas_compare_requires_the_same_input_and_game(self):
        route = "d0ab000000000001"
        title = "[   1.000000] Frontend <Info> BootGame: Booting game: 0400486024DEC000 | Zelda | \n"

        def replay(directory, mode, route_id=route, program="0400486024DEC000", slower=False):
            directory.mkdir(exist_ok=True)
            begin = line(2.0, f"TAS playback started frames=100 route={route_id}")
            elapsed_ns = 2_000_000_000 if slower else 1_000_000_000
            interval_ns = elapsed_ns // 100
            end = line(4.0, f"TAS playback completed frames=100 route={route_id} "
                       f"elapsed_ns={elapsed_ns} p50_ns={interval_ns} p95_ns={interval_ns} "
                       f"p99_ns={interval_ns} max_ns={interval_ns} near_120=0 "
                       "over_50_ms=0 over_100_ms=0")
            sites = ("staging_create:20x8.0" if slower else
                     "buffer_create:5x2.0,scene_transfer:75x2.0")
            before_route = line(1.95, "Buffer cache: 900 buffers 2500 MiB; largest (2147483648, 4096)")
            crossing = line(2.25, "Nx2StutterSummary window_s=1.0 sites=staging_create:99x99.0")
            crossing += line(2.26, "GPU waits in 1.0 s: 120 frames (120.0/s) | "
                                   "reactive flushes 900 waited 900.0 ms")
            crossing += line(2.5, "Memory trace: backed Vulkan images since last sample: 999 created, "
                                  "999 reused, 999 destroyed, 999 evicted by pressure trim; "
                                  "recycler holds 4 images (16 MiB)")
            crossing += line(2.6, "Uniform keeps in 600 frames: 999 kept, 999 streamed, "
                                  "0 verified, 0 verify mismatches, 999 uniform binds kept on full draws")
            if not slower:
                crossing += line(2.61, "Storage upload scans in 600 frames: 999 of 999 skipped, "
                                       "999 verify mismatches")
            inside = line(3.5, f"Nx2StutterSummary window_s=1.0 sites={sites}")
            reads, wait_ms, reader_ms = (150, 120.0, 90.0) if slower else (150, 120.0, 30.0)
            inside += line(3.5, "GPU waits in 1.0 s: 100 frames (100.0/s) | "
                                f"reactive flushes {reads} waited {wait_ms} ms; downloaded 12800 KiB, "
                                f"then copies on the reading thread {reader_ms} ms")
            full_keeps = "" if slower else ", 28 uniform binds kept on full draws"
            inside += line(3.5, "Uniform keeps in 600 frames: 34 kept, 123 streamed, "
                                f"0 verified, 0 verify mismatches{full_keeps}")
            created, reused, evicted, kept = (200, 60, 100, 12) if slower else (50, 210, 0, 500)
            inside += line(3.51, f"Memory trace: backed Vulkan images since last sample: "
                                 f"{created} created, {reused} reused, 30 destroyed, "
                                 f"{evicted} evicted by pressure trim; recycler holds 10 images "
                                 f"({kept} MiB)")
            if slower:
                inside += line(3.55, "Buffer cache: 759 buffers 1576 MiB; largest (1073676288, 7243497472)")
                inside += line(3.56, "Slow device memory allocate: 1024.0 MiB, type 1, "
                               "dedicated buffer, 235.3 ms")
                inside += line(3.56, "Slow device memory allocate: 128.0 MiB, type 1, "
                               "dedicated buffer, 11.2 ms")
                inside += line(3.57, "Nx2StutterDetail frame=11923 ms=455.1 zones=back:compute 376.5/66 "
                               "sites=memory_allocate:21x294.1,staging_create:24x246.7,buffer_create: "
                               "long=back:staging_create 235.4")
            else:
                inside += line(3.55, "Buffer cache: 100 buffers 88 MiB; largest (8388608, 7243497472)")
            gpu_time = 9.2 if slower else 7.5
            inside += line(3.6, "DKB gpu_frame gpu_frame_span_ms=12.5 "
                           f"gpu_submission_sum_ms={gpu_time} measured_submissions=2 "
                           "dropped_submissions=0")
            inside += line(3.7, "DKB gpu_frame gpu_frame_span_ms=40.0 "
                           "gpu_submission_sum_ms=36.0 measured_submissions=2 "
                           "dropped_submissions=1")
            if not slower:
                inside += line(3.52, "Storage upload scans in 600 frames: 80 of 100 skipped, "
                                     "0 verify mismatches")
                inside += line(3.75, "Vertex input cache: window_s=1.25 built=3 reused=80 "
                               "commands_skipped=60 uncached=2")
            (directory / "eden_log.txt").write_text(title.replace("0400486024DEC000", program)
                                                    + before_route + begin + crossing + inside + end,
                                                    encoding="utf-8")
            (directory / "launch.json").write_text(json.dumps({"Mode": mode,
                                                              "SettingsSnapshotSHA256": "test"}),
                                                   encoding="utf-8-sig")

        with tempfile.TemporaryDirectory() as tmp:
            baseline, candidate = Path(tmp) / "before", Path(tmp) / "after"
            replay(baseline, "reference", slower=True)
            replay(candidate, "experiment")
            report = frame_budget.compare_tas_runs(baseline, candidate)
            self.assertIn("| Average FPS | 50.00 fps | 100.00 fps | +50.00 fps |", report)
            self.assertIn("The launcher settings snapshots match.", report)
            self.assertIn("| Staging trace coverage | 1.00 s | 1.00 s | +0.00 s |", report)
            self.assertIn("| Staging allocations | 20.00 calls/s | 0.00 calls/s | -20.00 calls/s |",
                          report)
            self.assertIn("| Time spent allocating staging buffers | 8.00 ms/s | 0.00 ms/s |",
                          report)
            self.assertIn("| Reader wait and landing | 0.90 ms/frame | 0.30 ms/frame | -0.60 ms/frame |",
                          report)
            self.assertIn("| Reactive download size | 128.00 KiB/frame | 128.00 KiB/frame |", report)
            self.assertIn("| Scene transfers | 0.00 calls/s | 75.00 calls/s | +75.00 calls/s |",
                          report)
            self.assertIn("| p95 sampled GPU submission time | 9.20 ms | 7.50 ms | -1.70 ms |",
                          report)
            self.assertIn("| Peak sampled buffer cache | 1576.00 MiB | 88.00 MiB |", report)
            self.assertIn("| Largest sampled GPU buffer | 1023.94 MiB | 8.00 MiB |", report)
            self.assertIn("| Time in logged large buffer allocations | 235.30 ms | 0.00 ms |", report)
            self.assertIn("| Vulkan images created | 198.02 images/s | 49.50 images/s |", report)
            self.assertIn("| Recycler pressure evictions | 99.01 images/s | 0.00 images/s |", report)
            self.assertIn("| Peak sampled recycled images | 16.00 MiB | 500.00 MiB |", report)
            self.assertIn("Candidate streamed uniform bindings over 600 sampled frames. "
                          "28 kept on full draws, 6 kept on delta draws, 123 sent through", report)
            self.assertIn("Candidate reused 80 of 100 eligible storage scans over 600 sampled "
                          "frames, with 0 verification mismatches reported.", report)
            self.assertIn("| memory_allocate | 294.1 ms | — |", report)
            self.assertIn("| staging_create | 246.7 ms | — |", report)
            self.assertIn("3 layouts built, 80 reused, 60 redundant commands skipped, 2 uncached.",
                          report)
            replay(candidate, "experiment", route_id="d0ab000000000002")
            with self.assertRaisesRegex(ValueError, "different input scripts"):
                frame_budget.compare_tas_runs(baseline, candidate)
            replay(candidate, "experiment", program="0400245021D2C000")
            with self.assertRaisesRegex(ValueError, "different game IDs"):
                frame_budget.compare_tas_runs(baseline, candidate)

    def test_last_complete_tas_route_ignores_interruptions(self):
        messages = [
            (1.0, "TAS playback started frames=240"),
            (2.0, "TAS playback interrupted"),
            (3.0, "TAS playback completed frames=240"),
            (4.0, "TAS playback started frames=360"),
            (10.0, "TAS playback completed frames=360"),
            (11.0, "TAS playback started frames=360"),
        ]
        self.assertIsNone(frame_budget.last_tas_playback(messages[:3]))
        self.assertEqual(frame_budget.last_tas_playback(messages),
                         {"start_s": 4.0, "end_s": 10.0, "frames": 360,
                          "seconds": 6.0, "fps": 60.0})

    def test_tas_report_bounds_the_samples(self):
        before = cpu(3.0, 40, 950, 500)
        log = LOG.replace(before, line(2.5, "TAS playback started frames=210") + before)
        log += line(6.5, "TAS playback completed frames=210")
        result = json.loads(self.run_tool("--tas", "--target-fps", "120", "--json", log=log))
        self.assertEqual(result["tas"]["fps"], 52.5)
        self.assertEqual(result["frame_windows"], [])
        self.assertEqual(result["longest_stutter"]["frame"], 602)
        self.assertEqual([row["t"] for row in result["per_second"]], [3.0, 4.0, 5.0, 6.0])

    def test_texture_memo_report_uses_complete_windows(self):
        log = LOG + line(8.0, "TAS playback started frames=1800")
        log += line(10.5, "Texture memo: window_s=10.500 hits=500 full=200 second_hits=90 mismatches=0")
        log += cpu(12.0, 60, 750, 550)
        log += line(20.7, "Texture memo: window_s=10.200 hits=900 full=100 second_hits=60 mismatches=1")
        log += line(38.0, "TAS playback completed frames=1800")
        log += line(40.0, "Texture memo: window_s=19.300 hits=800 full=300 second_hits=80 mismatches=2")
        result = json.loads(self.run_tool("--tas", "--json", log=log))
        self.assertEqual(result["texture_memo"], {"window_s": 10.2, "hits": 900, "full": 100,
                                                  "second_hits": 60, "mismatches_since_startup": 1})
        self.assertIn("60 hits used the second lookup", self.run_tool("--tas", log=log))

    def run_tool(self, *args, log=LOG):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(log)
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
        self.assertEqual(result["on_target_seconds"], 1)
        self.assertEqual(result["hitch_seconds"], 1)
        self.assertEqual(result["frame_windows"][0]["p99"], 65.0)
        self.assertEqual(result["longest_stutter"]["ms"], 148.0)
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
        self.assertEqual(result["draws"]["slow"]["same_handle_draws_per_s"], 42)
        self.assertEqual(result["missing"], ["EDEN_NX2_HOSTTIMING_STATS=1"])
        guest = result["guest_cores"]
        self.assertEqual(guest["states_ms_per_s"]["idle spin"], 200.0)
        self.assertEqual(guest["svcs"][0], ["SleepThread yield", 60.0])

    def test_pass_timing_reports_serial_work_and_estimated_savings(self):
        prefix = "Pass timing per frame: "
        log = LOG + line(6.5, prefix + "7.65 ms of draw, clear and dispatch work | serial 4.69 ms "
                         "(cache 3.88 ms), remaining graphics 2.96 ms | estimated total: "
                         "2 workers 7.46 ms, 4 workers 7.37 ms, 8 workers 7.35 ms")
        log += line(6.6, prefix + "13.32 ms of draw, clear and dispatch work | serial 10.01 ms "
                    "(cache 7.85 ms), remaining graphics 3.30 ms | estimated total: "
                    "2 workers 13.08 ms, 4 workers 12.97 ms, 8 workers 12.94 ms")
        result = json.loads(self.run_tool("--target-fps", "120", "--json", log=log))
        self.assertEqual(result["pass_timing"]["serial_over_budget"], 1)
        self.assertEqual(result["pass_timing"]["heaviest"]["cache_ms"], 7.85)
        self.assertIn("| 4 | 12.97 | 0.35 ms |", self.run_tool("--target-fps", "120", log=log))
        self.assertNotIn("Draw, clear and dispatch work", self.run_tool())

    def test_markdown_report(self):
        log = LOG + line(8.0, "Vertex type cache reused 35 built pipelines in 600 frames")
        text = self.run_tool("--windows", log=log)
        self.assertIn("| guest CPU core saturated | 1 |", text)
        self.assertIn("## GPU back thread: busiest zones in slow windows", text)
        self.assertIn("## Per second", text)
        self.assertIn("System calls: SleepThread yield 60.0, WaitSynchronization 30.0", text)
        self.assertIn("Vertex type cache reused a built pipeline for 35 logged cache misses.", text)

    def test_start_and_end_select_seconds(self):
        result = json.loads(self.run_tool("--json", "--start-s", "2.5", "--end-s", "3.5"))
        self.assertEqual([r["t"] for r in result["per_second"]], [3.0])


if __name__ == "__main__":
    unittest.main()
