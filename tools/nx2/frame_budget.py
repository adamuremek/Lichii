#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where each second's frame time went, and which thread limited it.

  frame_budget.py <log> [--session N] [--tas | --start-s S --end-s S]
                 [--target-fps 60] [--windows] [--json]
  frame_budget.py <candidate-run> --compare-to <baseline-run> [--session N]
  frame_budget.py <log> --tas-ab [--session N] [--json]

Reads the lines of a profiling run (see docs/nx2/profiling.md):
  EDEN_DKB_FRAME_TRACE=1    DKB cpu (per-thread CPU time each second), DKB gpu_frame, Frame timing window
  EDEN_NX2_STUTTER_TRACE=1  Nx2StutterSummary (costly sites per 10 s), Fast draw per second
  EDEN_NX2_GPU_PROFILE=1    "<thread> per second:" zone times and wait sites per 10 s
  EDEN_NX2_HOSTTIMING_STATS=1  HostTiming wait lines
  EDEN_NX2_GUEST_PROFILE=1  "Guest core <n> per second:" JIT, system call, kernel and idle time per 10 s
  EDEN_NX2_PASS_STATS=1     pass timings when GPU profiling or stutter tracing is enabled
Each second is classified by the busiest thread relative to wall time. Thread CPU time counts spinning
as busy, so a saturated thread is the limit only when the GPU profile does not show it waiting.

Standard library only.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TIME = re.compile(r"^\[\s*(\d+\.\d+)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SESSION_START = re.compile(r"Booting game: ([0-9A-Fa-f]{16})(?: \| ([^|]*))?")
TAS_PLAYBACK = re.compile(r"TAS playback (?:(started|completed) frames=(\d+)"
                          r"(?: route=([0-9a-f]{16}))?(.*)|(interrupted))$")
TAS_TIMING = re.compile(r" elapsed_ns=(?P<elapsed_ns>\d+) p50_ns=(?P<p50_ns>\d+) "
                        r"p95_ns=(?P<p95_ns>\d+) p99_ns=(?P<p99_ns>\d+) max_ns=(?P<max_ns>\d+) "
                        r"near_120=(?P<near_120>\d+) over_50_ms=(?P<over_50_ms>\d+) "
                        r"over_100_ms=(?P<over_100_ms>\d+)")
TAS_PHASE_TIMING = re.compile(TAS_TIMING.pattern +
                              r"(?: texture_checks_skipped=(?P<texture_checks_skipped>\d+))?"
                              r"(?: buffer_checks_skipped=(?P<buffer_checks_skipped>\d+))?"
                              r"(?: acquire_waits_shortened=(?P<acquire_waits_shortened>\d+))?")
TAS_AB_START = re.compile(r"TAS A/B started experiment=(\S+) frames=(\d+) warmup=(\d+)$")
TAS_AB_PHASE = re.compile(r"TAS A/B phase experiment=(\S+) index=(\d+) "
                          r"mode=(\S+) frames=(\d+)(.*)$")
TAS_AB_END = re.compile(r"TAS A/B completed experiment=(\S+) frames=(\d+)$")
TAS_AB_MODES = {
    "adaptive-submit": ("baseline", "candidate", "candidate", "baseline"),
    "renderer": ("baseline", "adaptive-submit", "both", "raster-state",
                 "raster-state", "both", "adaptive-submit", "baseline"),
    "cpu-placement": ("baseline", "cache-local", "cache-local", "baseline"),
    "texture-reads": ("baseline", "candidate", "candidate", "baseline"),
    "buffer-reads": ("baseline", "candidate", "candidate", "baseline"),
    "acquire-waits": ("baseline", "candidate", "candidate", "baseline"),
}
BUILD = re.compile(r"Lichii Version: (.*)$")
CPU = re.compile(r"DKB cpu through_frame=\d+ frames=(\d+) window_wall_ms=([\d.]+)(.*)$")
CPU_ROLE = re.compile(r"(\w+?)(?:_host)?_cpu_ms=([\d.]+)")
GPU_FRAME = re.compile(r"DKB gpu_frame .*gpu_frame_span_ms=([\d.]+) gpu_submission_sum_ms=([\d.]+) "
                       r"measured_submissions=(\d+)")
GPU_DROPS = re.compile(r"\bdropped_submissions=(\d+)")
FRAME_WINDOW = re.compile(r"Frame timing window .*samples=(\d+) .*mean_ms=([\d.]+) "
                          r"p50_ms=([\d.]+) p95_ms=([\d.]+) p99_ms=([\d.]+) max_ms=([\d.]+)")
STUTTER = re.compile(r"Nx2Stutter frame=(\d+) ms=([\d.]+) cause=(\S+)")
PASS_TIMING = re.compile(r"Pass timing per frame: (?P<total_ms>[\d.]+) ms of draw, clear and dispatch work \| "
                         r"serial (?P<serial_ms>[\d.]+) ms \(cache (?P<cache_ms>[\d.]+) ms\), "
                         r"remaining graphics [\d.]+ ms \| estimated total: "
                         r"2 workers (?P<two_ms>[\d.]+) ms, 4 workers (?P<four_ms>[\d.]+) ms, "
                         r"8 workers (?P<eight_ms>[\d.]+) ms")
STUTTER_DETAIL_SITES = re.compile(r"Nx2StutterDetail frame=(\d+) ms=([\d.]+) .*?sites=(\S+)")
STUTTER_SITE = re.compile(r"(?:^|,)(\w+):(\d+)x(\d+(?:\.\d+)?)(?=,|$)")
BUFFER_CACHE = re.compile(r"Buffer cache: (\d+) buffers (\d+) MiB; largest"
                          r"(?: \((\d+), (\d+)\))?")
LARGE_BUFFER_ALLOC = re.compile(r"Slow device memory allocate: ([\d.]+) MiB, type \d+, "
                                r"dedicated buffer, ([\d.]+) ms")
IMAGE_RECYCLER = re.compile(r"backed Vulkan images since last sample: (\d+) created, "
                            r"(\d+) reused, (\d+) destroyed, (\d+) evicted by pressure trim; "
                            r"recycler holds \d+ images \((\d+) MiB\)")
UNIFORM_KEEPS = re.compile(r"Uniform keeps in (\d+) frames: (\d+) kept, (\d+) streamed, "
                           r"(\d+) verified, (\d+) verify mismatches"
                           r"(?:, (\d+) uniform binds kept on full draws)?")
STORAGE_SCANS = re.compile(r"Storage upload scans in (\d+) frames: (\d+) of (\d+) skipped, "
                           r"(\d+) verify mismatches")
ALLOCATION_SITES = ("staging_create", "buffer_create", "memory_allocate", "memory_free",
                    "buffer_collect", "texture_collect")
FAST_DRAW = re.compile(r"Fast draw per second: (\d+) delta")
SAME_HANDLES = re.compile(r"(\d+) unchanged-handle reuses")
VERTEX_TYPE_CACHE = re.compile(r"Vertex type cache reused (\d+) built pipelines in \d+ frames")
FAST_PREPARE = re.compile(r"prepare us per draw: full ([\d.]+), delta ([\d.]+), all ([\d.]+) \((\d+) draws/s\)")
TEXTURE_MEMO = re.compile(r"Texture memo: window_s=([\d.]+) hits=(\d+) full=(\d+) "
                          r"second_hits=(\d+) mismatches=(\d+)")
VERTEX_INPUT_CACHE = re.compile(r"Vertex input cache: window_s=([\d.]+) built=(\d+) "
                                r"reused=(\d+) commands_skipped=(\d+) uncached=(\d+)")
SUMMARY_SITES = re.compile(r"Nx2StutterSummary window_s=([\d.]+) .*?sites=(\S+)")
SUMMARY_CATS = re.compile(r"Nx2StutterSummary .*?cat_ms=(\S+)")
STAGING_CREATES = re.compile(r"(?:^|,)staging_create:(\d+)x([\d.]+)(?:,|$)")
SCENE_TRANSFERS = re.compile(r"(?:^|,)scene_transfer:(\d+)x[\d.]+(?:,|$)")
READBACK_WINDOW = re.compile(r"GPU waits in ([\d.]+) s: (\d+) frames .*?\| "
                             r"reactive flushes (\d+) waited ([\d.]+) ms")
READBACK_BYTES = re.compile(r"\bdownloaded (\d+) KiB")
READBACK_READER = re.compile(r"copies on the reading thread ([\d.]+) ms")
PROFILE = re.compile(r"(GPU thread|GPU back thread|Vulkan worker) per second:(.*)$")
PROFILE_ZONE = re.compile(r"\| ([a-zA-Z/ ]+?) ([\d.]+) ms \((\d+)/s\)")
PROFILE_WALL = re.compile(r"window_wall_ms=([\d.]+)")
PROFILE_SITES = re.compile(r"\| (waits for the back by site|GPU waits by zone):([^|]*)")
SITE = re.compile(r" ?([a-zA-Z/ ]+?) ([\d.]+) ms \((\d+)/s\)")
GUEST_PROFILE = re.compile(r"(Guest core \d+|Guest single core) per second:(.*)$")
GUEST_STATE = re.compile(r"\| (kernel|jit|svc|idle spin|idle wait) ([\d.]+) ms")
GUEST_SVCS = re.compile(r"\| svcs:(.*)$")
HOSTTIMING = re.compile(r"HostTiming wait: .*OS sleep ([\d.]+) ms/s, WaitFor ([\d.]+) ms/s; "
                        r"wake late p50/p99/max (\d+)/(\d+)/(\d+) us.*thread CPU ([\d.]+)%")

# Time a thread spends waiting, not working.
WAIT_ZONES = {"idle", "worker idle", "split wait", "split drain", "writer wait", "pending sleep",
              "present wait", "list submit wait", "list GPU wait", "GPU wait", "pipeline wait"}
SATURATED = 0.85
# Seconds below this are loads or long hitches; they are counted but kept out of the averages.
HITCH_FPS = 20.0
GUEST = "guest"
ROLE_LABELS = {GUEST: "guest CPU core", "gpu_back": "GPU back thread", "gpu_command": "GPU thread",
               "vulkan_worker": "Vulkan worker", "host_gpu": "host GPU"}


def read_sessions(path):
    sessions = [{"start": None, "title": None, "program": None, "lines": []}]
    build = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            text = ANSI.sub("", raw.rstrip("\n"))
            if build is None and (m := BUILD.search(text)):
                build = m.group(1).strip()
            t = TIME.match(text)
            if not t:
                continue
            seconds = float(t.group(1))
            if m := SESSION_START.search(text):
                sessions.append({"start": seconds, "program": m.group(1).upper(),
                                 "title": (m.group(2) or "").strip(), "lines": []})
            sessions[-1]["lines"].append((seconds, text))
    return build, [s for s in sessions if s["start"] is not None] or sessions


def last_tas_ab(lines):
    start = None
    phases = []
    result = None
    for _, text in lines:
        if match := TAS_AB_START.search(text):
            name, frame_count, warmup_count = match.groups()
            frames, warmup = int(frame_count), int(warmup_count)
            modes = TAS_AB_MODES.get(name, ())
            start = (name, frames, warmup) if (modes and frames % len(modes) == 0 and
                                               frames // len(modes) > warmup) else None
            phases = []
        elif "TAS A/B aborted " in text or "TAS playback interrupted" in text:
            start = None
        elif start is not None and (match := TAS_AB_PHASE.search(text)):
            experiment, index, mode, frames, suffix = match.groups()
            index, frames = int(index), int(frames)
            timing = TAS_PHASE_TIMING.fullmatch(suffix)
            if not (experiment == start[0] and index == len(phases) + 1 and index <= len(modes)
                    and modes[index - 1] == mode and frames == start[1] // len(modes) - start[2]
                    and timing):
                start = None
                continue
            values = {key: int(value) for key, value in timing.groupdict().items()
                      if value is not None}
            counter = {"texture-reads": "texture_checks_skipped",
                       "buffer-reads": "buffer_checks_skipped",
                       "acquire-waits": "acquire_waits_shortened"}.get(experiment)
            if counter and counter not in values:
                start = None
                continue
            if values["elapsed_ns"] == 0 or any(values[k] > frames for k in
                                                 ("near_120", "over_50_ms", "over_100_ms")):
                start = None
                continue
            phases.append({"index": index, "mode": mode, "frames": frames, **values,
                           "fps": frames * 1e9 / values["elapsed_ns"]})
        elif start is not None and (match := TAS_AB_END.search(text)):
            if match.group(1) == start[0] and int(match.group(2)) == start[1] and len(phases) == len(modes):
                fps = {}
                near_120 = {}
                for mode in dict.fromkeys(modes):
                    selected = [phase for phase in phases if phase["mode"] == mode]
                    frames = sum(phase["frames"] for phase in selected)
                    fps[mode] = (frames * 1e9 /
                                 sum(phase["elapsed_ns"] for phase in selected))
                    near_120[mode] = sum(phase["near_120"] for phase in selected) / frames
                result = {"experiment": start[0], "frames": start[1], "warmup_per_phase": start[2],
                          "phases": phases, "fps": fps, "near_120": near_120}
                if "candidate" in fps:
                    result["fps_delta"] = fps["candidate"] - fps["baseline"]
            start = None
    return result


def report_tas_ab(result, title):
    benchmark = {"adaptive-submit": "Adaptive Vulkan submission A/B",
                 "renderer": "Vulkan renderer comparison",
                 "cpu-placement": "CPU cache placement A/B",
                 "texture-reads": "Vulkan texture read filter A/B",
                 "buffer-reads": "Vulkan buffer read filter A/B",
                 "acquire-waits": "Semaphore acquire wait A/B"}[result["experiment"]]
    out = [f"# {benchmark} · {title}", "",
           f"{len(result['phases'])} idle phases with {result['warmup_per_phase']} warmup "
           "frames discarded per phase.",
           "", "| Phase | Mode | Frames | FPS | p50 ms | p95 ms | p99 ms | Max ms | >50 ms |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase in result["phases"]:
        out.append(f"| {phase['index']} | {phase['mode']} | {phase['frames']} | "
                   f"{phase['fps']:.2f} | {phase['p50_ns'] / 1e6:.2f} | "
                   f"{phase['p95_ns'] / 1e6:.2f} | {phase['p99_ns'] / 1e6:.2f} | "
                   f"{phase['max_ns'] / 1e6:.2f} | {phase['over_50_ms']} |")
    fps = result["fps"]
    out += ["", "| Mode | Combined FPS | Change from baseline | Frames within 3% of 120 FPS or faster |",
            "|---|---:|---:|---:|"]
    for mode, mean_fps in fps.items():
        out.append(f"| {mode} | {mean_fps:.2f} FPS | {mean_fps - fps['baseline']:+.2f} FPS | "
                   f"{result['near_120'][mode]:.1%} |")
    if result["experiment"] in ("texture-reads", "buffer-reads"):
        cache = result["experiment"].split("-")[0]
        candidate = [phase for phase in result["phases"] if phase["mode"] == "candidate"]
        skipped = sum(phase[f"{cache}_checks_skipped"] for phase in candidate)
        frames = sum(phase["frames"] for phase in candidate)
        out += ["", f"The filter skipped {skipped:,} {cache} cache checks over {frames:,} "
                    "measured candidate frames."]
    elif result["experiment"] == "acquire-waits":
        candidate = [phase for phase in result["phases"] if phase["mode"] == "candidate"]
        shortened = sum(phase["acquire_waits_shortened"] for phase in candidate)
        frames = sum(phase["frames"] for phase in candidate)
        out += ["", f"The candidate selected {shortened:,} shorter wait targets over "
                    f"{frames:,} measured frames."]
    out += ["", "The frame-time budget at 120 FPS is 8.33 ms."]
    return "\n".join(out)


def last_tas_playback(lines):
    start = None
    result = None
    for t, text in lines:
        match = TAS_PLAYBACK.search(text)
        if not match:
            continue
        if match.group(5):
            start = None
            continue
        frames, route = int(match.group(2)), match.group(3)
        if match.group(1) == "started":
            start = (t, frames, route)
            continue
        if start and start[1:] == (frames, route) and frames > 0 and t > start[0]:
            seconds = t - start[0]
            result = {"start_s": start[0], "end_s": t, "frames": frames,
                      "seconds": seconds, "fps": frames / seconds}
            if route is not None:
                result["route"] = route
            if timing_match := TAS_TIMING.fullmatch(match.group(4)):
                timing = {key: int(value) for key, value in timing_match.groupdict().items()}
                if (0 < timing["p50_ns"] <= timing["p95_ns"] <= timing["p99_ns"] <=
                        timing["max_ns"] <= timing["elapsed_ns"] and
                        timing["near_120"] <= frames and
                        timing["over_100_ms"] <= timing["over_50_ms"] <= frames):
                    seconds = timing["elapsed_ns"] / 1e9
                    result.update(seconds=seconds, fps=frames / seconds, timing=timing)
        start = None
    return result


def allocation_diagnostics(lines):
    result = {}
    sites = {}
    recycler = Counter()
    previous_recycler_sample = None
    uniforms = Counter()
    previous_uniform_sample = False
    storage_scans = Counter()
    previous_storage_sample = False
    large_buffer_allocations = Counter()
    for t, text in lines:
        if match := BUFFER_CACHE.search(text):
            count, mib = map(int, match.groups()[:2])
            size = int(match.group(3) or 0)
            result["buffer_cache_samples"] = result.get("buffer_cache_samples", 0) + 1
            if mib > result.get("peak_buffer_cache", {}).get("mib", -1):
                result["peak_buffer_cache"] = {"t": t, "mib": mib, "count": count}
            if size > result.get("largest_buffer", {}).get("bytes", 0):
                result["largest_buffer"] = {"t": t, "bytes": size, "device_addr": int(match.group(4))}
        elif match := LARGE_BUFFER_ALLOC.search(text):
            size, ms = map(float, match.groups())
            if size >= 512:
                large_buffer_allocations.update(count=1, ms=ms)
                large_buffer_allocations["max_ms"] = max(large_buffer_allocations["max_ms"], ms)
        elif match := STUTTER_DETAIL_SITES.search(text):
            frame, interval = int(match.group(1)), float(match.group(2))
            for name, count, duration in STUTTER_SITE.findall(match.group(3)):
                ms = float(duration)
                if name in ALLOCATION_SITES and ms > sites.get(name, {}).get("ms", 0):
                    sites[name] = {"t": t, "frame": frame, "interval_ms": interval,
                                   "ms": ms, "count": int(count)}
        elif match := IMAGE_RECYCLER.search(text):
            created, reused, destroyed, evicted, kept_mib = map(int, match.groups())
            if previous_recycler_sample is not None and t > previous_recycler_sample:
                recycler.update(seconds=t - previous_recycler_sample, created=created,
                                reused=reused, destroyed=destroyed, pressure_evictions=evicted)
            recycler["peak_kept_mib"] = max(recycler["peak_kept_mib"], kept_mib)
            previous_recycler_sample = t
        elif match := UNIFORM_KEEPS.search(text):
            frames, kept, streamed, verified, mismatches, full = (
                int(value or 0) for value in match.groups())
            if previous_uniform_sample:
                uniforms.update(frames=frames, kept=kept, streamed=streamed, verified=verified,
                                mismatches=mismatches, kept_full=full)
            previous_uniform_sample = True
        elif match := STORAGE_SCANS.search(text):
            frames, skipped, total, mismatches = map(int, match.groups())
            if previous_storage_sample:
                storage_scans.update(frames=frames, skipped=skipped, total=total,
                                     mismatches=mismatches)
            previous_storage_sample = True
    if sites:
        result["hitch_sites"] = sites
    if recycler["seconds"]:
        result["recycler"] = dict(recycler)
    if uniforms["frames"]:
        result["uniforms"] = dict(uniforms)
    if storage_scans["frames"]:
        result["storage_scans"] = dict(storage_scans)
    if large_buffer_allocations:
        result["large_buffer_allocations"] = dict(large_buffer_allocations)
    return result


def readback_diagnostics(lines, start_s):
    totals = Counter()
    for t, text in lines:
        match = READBACK_WINDOW.search(text)
        if not match:
            continue
        window = float(match.group(1))
        if window <= 0 or (start_s is not None and t - window < start_s):
            continue
        totals["seconds"] += window
        totals["frames"] += int(match.group(2))
        totals["flushes"] += int(match.group(3))
        totals["flush_ms"] += float(match.group(4))
        downloaded = READBACK_BYTES.search(text)
        reader = READBACK_READER.search(text)
        totals["download_kib"] += int(downloaded.group(1)) if downloaded else 0
        totals["reader_ms"] += float(reader.group(1)) if reader else 0.0
    return totals


def read_tas_run(path, session):
    path = Path(path)
    log = path / "eden_log.txt" if path.is_dir() else path
    build, sessions = read_sessions(log)
    try:
        game = sessions[session]
    except IndexError as error:
        raise ValueError(f"{log} has {len(sessions)} session(s)") from error
    replay = last_tas_playback(game["lines"])
    if not replay or not game["program"] or "route" not in replay or "timing" not in replay:
        raise ValueError(f"{log} needs a completed TAS replay with route identity and frame timing")
    launch_file = log.parent / "launch.json"
    launch = json.loads(launch_file.read_text(encoding="utf-8-sig")) if launch_file.is_file() else {}
    staging_window = staging_count = staging_ms = scene_transfers = 0
    gpu_times = []
    vertex_window = 0.0
    vertex_counts = Counter()
    route_lines = [(t, text) for t, text in game["lines"] if replay["start_s"] <= t <= replay["end_s"]]
    for t, text in route_lines:
        if match := SUMMARY_SITES.search(text):
            window = float(match.group(1))
            if window > 0 and t - window >= replay["start_s"]:
                staging_window += window
                if site := STAGING_CREATES.search(match.group(2)):
                    staging_count += int(site.group(1))
                    staging_ms += float(site.group(2))
                if site := SCENE_TRANSFERS.search(match.group(2)):
                    scene_transfers += int(site.group(1))
        elif (match := GPU_FRAME.search(text)) and int(match.group(3)):
            if (drops := GPU_DROPS.search(text)) and int(drops.group(1)) == 0:
                gpu_times.append(float(match.group(2)))
        elif match := VERTEX_INPUT_CACHE.search(text):
            window = float(match.group(1))
            if window > 0 and t - window >= replay["start_s"]:
                vertex_window += window
                vertex_counts.update(dict(zip(("built", "reused", "skipped", "uncached"),
                                              map(int, match.groups()[1:]))))
    if staging_window:
        replay["staging"] = (staging_window, staging_count / staging_window,
                             staging_ms / staging_window)
        replay["scene_transfers"] = scene_transfers
    if gpu_times:
        gpu_times.sort()
        replay["gpu_submissions"] = (len(gpu_times), statistics.median(gpu_times),
                                     gpu_times[-1 - len(gpu_times) // 20])
    if vertex_window:
        replay["vertex_input"] = vertex_window, vertex_counts
    replay["allocations"] = allocation_diagnostics(route_lines)
    replay["readbacks"] = readback_diagnostics(route_lines, replay["start_s"])
    return game["program"], game["title"], replay, launch.get("Mode") or build or log.name, launch


def compare_tas_runs(baseline, candidate, session=-1):
    game_a, title, a, mode_a, launch_a = read_tas_run(baseline, session)
    game_b, _, b, mode_b, launch_b = read_tas_run(candidate, session)
    if game_a != game_b:
        raise ValueError(f"different game IDs in the two replays ({game_a} and {game_b})")
    if (a["route"], a["frames"]) != (b["route"], b["frames"]):
        raise ValueError("the two replays used different input scripts or frame counts")

    def milliseconds(replay, key):
        return replay["timing"][key] / 1e6

    def near_target(replay):
        return 100 * replay["timing"]["near_120"] / replay["frames"]

    metrics = [
        ("Average FPS", a["fps"], b["fps"], "fps"),
        ("Playback duration", a["seconds"], b["seconds"], "s"),
        ("p50 frame interval", milliseconds(a, "p50_ns"), milliseconds(b, "p50_ns"), "ms"),
        ("p95 frame interval", milliseconds(a, "p95_ns"), milliseconds(b, "p95_ns"), "ms"),
        ("p99 frame interval", milliseconds(a, "p99_ns"), milliseconds(b, "p99_ns"), "ms"),
        ("Worst frame interval", milliseconds(a, "max_ns"), milliseconds(b, "max_ns"), "ms"),
        ("Frames at 116.4 FPS pace or faster", near_target(a), near_target(b), "%"),
        ("Frames over 50 ms", a["timing"]["over_50_ms"], b["timing"]["over_50_ms"], "frames"),
        ("Frames over 100 ms", a["timing"]["over_100_ms"], b["timing"]["over_100_ms"], "frames"),
    ]
    lines = [f"# TAS comparison for {title or game_a}", "",
             f"Route `{a['route']}` with {a['frames']:,} frames.", "",
             f"- Baseline build or mode `{mode_a}`",
             f"- Candidate build or mode `{mode_b}`", ""]
    hash_a = launch_a.get("SettingsSnapshotSHA256")
    hash_b = launch_b.get("SettingsSnapshotSHA256")
    if hash_a and hash_b:
        lines += ["The launcher settings snapshots match." if hash_a == hash_b else
                  "The launcher settings snapshots differ. Check the settings used for the run.", ""]
    if "staging" in a and "staging" in b:
        lines += ["Staging allocation rates cover complete trace windows within each replay.", ""]
        metrics += [
            ("Staging trace coverage", a["staging"][0], b["staging"][0], "s"),
            ("Staging allocations", a["staging"][1], b["staging"][1], "calls/s"),
            ("Time spent allocating staging buffers", a["staging"][2], b["staging"][2], "ms/s"),
        ]
        if a["scene_transfers"] or b["scene_transfers"]:
            metrics.append(("Scene transfers", a["scene_transfers"] / a["staging"][0],
                            b["scene_transfers"] / b["staging"][0], "calls/s"))
    if "gpu_submissions" in a and "gpu_submissions" in b:
        samples_a, median_a, p95_a = a["gpu_submissions"]
        samples_b, median_b, p95_b = b["gpu_submissions"]
        lines += [f"GPU submission timings use {samples_a:,} baseline and {samples_b:,} candidate "
                  "frame samples with no dropped measurements.", ""]
        metrics += [
            ("Median sampled GPU submission time", median_a, median_b, "ms"),
            ("p95 sampled GPU submission time", p95_a, p95_b, "ms"),
        ]
    alloc_a, alloc_b = a["allocations"], b["allocations"]
    reads_a, reads_b = a["readbacks"], b["readbacks"]
    if reads_a["frames"] and reads_b["frames"]:
        lines += ["Readback metrics use complete trace windows and sum times across guest threads.", ""]
        metrics.append(("Readback trace coverage", reads_a["seconds"], reads_b["seconds"], "s"))
        for key, label, unit in (
            ("flushes", "Reactive flushes", "calls/frame"),
            ("flush_ms", "Reactive flush wait", "ms/frame"),
            ("reader_ms", "Reader wait and landing", "ms/frame"),
            ("download_kib", "Reactive download size", "KiB/frame"),
        ):
            metrics.append((label, reads_a[key] / reads_a["frames"],
                            reads_b[key] / reads_b["frames"], unit))
    if "peak_buffer_cache" in alloc_a and "peak_buffer_cache" in alloc_b:
        lines += ["Buffer cache peaks are sampled every 600 frames.", ""]
        metrics.append(("Peak sampled buffer cache", alloc_a["peak_buffer_cache"]["mib"],
                        alloc_b["peak_buffer_cache"]["mib"], "MiB"))
        if "largest_buffer" in alloc_a and "largest_buffer" in alloc_b:
            metrics.append(("Largest sampled GPU buffer", alloc_a["largest_buffer"]["bytes"] / 2**20,
                            alloc_b["largest_buffer"]["bytes"] / 2**20, "MiB"))
    large_a, large_b = (alloc.get("large_buffer_allocations", {})
                        for alloc in (alloc_a, alloc_b))
    if large_a or large_b:
        lines += ["Large allocation metrics count logged slow dedicated buffer allocations of 512 MiB or more.", ""]
        metrics += [
            ("Logged large buffer allocations", large_a.get("count", 0), large_b.get("count", 0), "calls"),
            ("Time in logged large buffer allocations", large_a.get("ms", 0), large_b.get("ms", 0), "ms"),
            ("Longest logged large buffer allocation", large_a.get("max_ms", 0), large_b.get("max_ms", 0), "ms"),
        ]
    if "recycler" in alloc_a and "recycler" in alloc_b:
        recycler_a, recycler_b = alloc_a["recycler"], alloc_b["recycler"]
        lines += ["Image recycler rates cover complete trace windows within each replay.", ""]
        metrics.append(("Recycler trace coverage", recycler_a["seconds"], recycler_b["seconds"], "s"))
        for key, label in (("created", "Vulkan images created"),
                           ("reused", "Vulkan images reused"),
                           ("pressure_evictions", "Recycler pressure evictions")):
            metrics.append((label, recycler_a[key] / recycler_a["seconds"],
                            recycler_b[key] / recycler_b["seconds"], "images/s"))
        metrics.append(("Peak sampled recycled images", recycler_a["peak_kept_mib"],
                        recycler_b["peak_kept_mib"], "MiB"))
    lines += ["| Metric | Baseline | Candidate | Change |", "|---|---:|---:|---:|"]
    for label, before, after, unit in metrics:
        if unit == "frames":
            lines.append(f"| {label} | {before:,} | {after:,} | {after - before:+,} |")
        else:
            change_unit = "pp" if unit == "%" else unit
            lines.append(f"| {label} | {before:.2f} {unit} | {after:.2f} {unit} | "
                         f"{after - before:+.2f} {change_unit} |")
    sites_a, sites_b = alloc_a.get("hitch_sites", {}), alloc_b.get("hitch_sites", {})
    if sites_a or sites_b:
        lines += ["", "Largest site cost on a logged hitch in each replay. Sites can overlap.",
                  "", "| Site | Baseline | Candidate |", "|---|---:|---:|"]
        for name in ALLOCATION_SITES:
            if name in sites_a or name in sites_b:
                before = f"{sites_a[name]['ms']:.1f} ms" if name in sites_a else "—"
                after = f"{sites_b[name]['ms']:.1f} ms" if name in sites_b else "—"
                lines.append(f"| {name} | {before} | {after} |")
    for label, replay in (("Baseline", a), ("Candidate", b)):
        if uniforms := replay["allocations"].get("uniforms"):
            lines += ["", f"{label} streamed uniform bindings over {uniforms['frames']:,} sampled "
                      f"frames. {uniforms['kept_full']:,} kept on full draws, "
                      f"{uniforms['kept'] - uniforms['kept_full']:,} kept on delta draws, "
                      f"{uniforms['streamed']:,} sent through the usual stream path."]
        if storage := replay["allocations"].get("storage_scans"):
            lines += ["", f"{label} reused {storage['skipped']:,} of {storage['total']:,} eligible "
                      f"storage scans over {storage['frames']:,} sampled frames, with "
                      f"{storage['mismatches']:,} verification mismatches reported."]
        if vertex_input := replay.get("vertex_input"):
            window, counts = vertex_input
            lines += ["", f"{label} vertex input cache over {window:.1f} s of replay. "
                      f"{counts['built']:,} layouts built, {counts['reused']:,} reused, "
                      f"{counts['skipped']:,} redundant commands skipped, "
                      f"{counts['uncached']:,} uncached."]
    return "\n".join(lines)


def parse(lines):
    data = defaultdict(list)
    for t, text in lines:
        if m := CPU.search(text):
            roles = {}
            for name, ms in CPU_ROLE.findall(m.group(3)):
                if not name.endswith("_per_frame"):
                    roles[name] = float(ms)
            data["cpu"].append({"t": t, "frames": int(m.group(1)), "wall_ms": float(m.group(2)),
                                "roles": roles})
        elif m := GPU_FRAME.search(text):
            data["gpu"].append({"t": t, "span_ms": float(m.group(1)), "busy_ms": float(m.group(2)),
                                "submits": int(m.group(3))})
        elif m := FRAME_WINDOW.search(text):
            data["frames"].append({"t": t, "samples": int(m.group(1)), "mean": float(m.group(2)),
                                   "p50": float(m.group(3)), "p95": float(m.group(4)),
                                   "p99": float(m.group(5)), "max": float(m.group(6))})
        elif m := STUTTER.search(text):
            data["stutters"].append({"t": t, "frame": int(m.group(1)), "ms": float(m.group(2)),
                                     "cause": m.group(3)})
        elif m := PASS_TIMING.search(text):
            data["passes"].append({"t": t, **{k: float(v) for k, v in m.groupdict().items()}})
        elif m := FAST_DRAW.search(text):
            data["fast"].append({"t": t, "delta": int(m.group(1))})
            if reuses := SAME_HANDLES.search(text):
                data["fast"][-1]["same_handle_draws_per_s"] = int(reuses.group(1))
        elif m := FAST_PREPARE.search(text):
            if data["fast"]:
                data["fast"][-1].update(full_us=float(m.group(1)), delta_us=float(m.group(2)),
                                        all_us=float(m.group(3)), draws=int(m.group(4)))
        elif m := TEXTURE_MEMO.search(text):
            data["texture_memo"].append({"t": t, "window_s": float(m.group(1)),
                                         "hits": int(m.group(2)), "full": int(m.group(3)),
                                         "second_hits": int(m.group(4)), "mismatches": int(m.group(5))})
        elif m := VERTEX_TYPE_CACHE.search(text):
            data["vertex_type_cache"].append(int(m.group(1)))
        elif "Nx2StutterSummary" in text and (m := SUMMARY_SITES.search(text)):
            window = float(m.group(1)) or 10.0
            sites = {}
            for item in m.group(2).split(","):
                name, _, rest = item.partition(":")
                count, _, ms = rest.partition("x")
                if ms:
                    sites[name] = {"ms_per_s": float(ms) / window, "per_s": int(count) / window}
            cats = {}
            if c := SUMMARY_CATS.search(text):
                for item in c.group(1).split(","):
                    name, _, ms = item.partition(":")
                    cats[name] = float(ms) / window
            data["sites"].append({"t": t, "window_s": window, "sites": sites, "cat_ms_per_s": cats})
        elif m := PROFILE.search(text):
            body = m.group(2)
            wall = PROFILE_WALL.search(body)
            zones = {}
            for name, ms, calls in PROFILE_ZONE.findall(body.split("| buffer downloads")[0]):
                zones[name.strip()] = {"ms_per_s": float(ms), "per_s": int(calls)}
            waits = {}
            for kind, items in PROFILE_SITES.findall(body):
                for name, ms, calls in SITE.findall(items):
                    waits[f"{kind}: {name.strip()}"] = {"ms_per_s": float(ms), "per_s": int(calls)}
            data["profile"].append({"t": t, "thread": m.group(1), "zones": zones, "waits": waits,
                                    "window_s": float(wall.group(1)) / 1000 if wall else 10.0})
        elif m := GUEST_PROFILE.search(text):
            body = m.group(2)
            wall = PROFILE_WALL.search(body)
            svcs = {}
            if c := GUEST_SVCS.search(body):
                for name, ms, calls in SITE.findall(c.group(1)):
                    svcs[name.strip()] = {"ms_per_s": float(ms), "per_s": int(calls)}
            data["guest"].append({"t": t, "core": m.group(1),
                                  "states": {k: float(v) for k, v in GUEST_STATE.findall(body)},
                                  "svcs": svcs,
                                  "window_s": float(wall.group(1)) / 1000 if wall else 10.0})
        elif m := HOSTTIMING.search(text):
            data["hosttiming"].append({"t": t, "sleep_ms_per_s": float(m.group(1)),
                                       "waitfor_ms_per_s": float(m.group(2)),
                                       "wake_late_p99_us": int(m.group(4)),
                                       "cpu_percent": float(m.group(6))})
    return data


def seconds_table(data, target_fps):
    """One row per DKB cpu line: frame rate, each thread's share of wall time, host GPU time."""
    rows = []
    gpu = data["gpu"]
    gi = 0
    previous_t = None
    for cpu in data["cpu"]:
        if cpu["frames"] == 0 or cpu["wall_ms"] <= 0:
            previous_t = cpu["t"]
            continue
        frame_ms = cpu["wall_ms"] / cpu["frames"]
        util = {}
        guest = []
        for name, ms in cpu["roles"].items():
            share = ms / cpu["wall_ms"]
            if name.startswith("guest_"):
                guest.append(share)
            else:
                util[name] = share
        if guest:
            util[GUEST] = max(guest)
        # The latest host GPU frame sampled in this second.
        sample = None
        while gi < len(gpu) and gpu[gi]["t"] <= cpu["t"]:
            if previous_t is None or gpu[gi]["t"] > previous_t:
                sample = gpu[gi]
            gi += 1
        if sample:
            util["host_gpu"] = min(sample["busy_ms"] / frame_ms, 1.0)
        fps = 1000.0 / frame_ms
        busiest = max(util, key=util.get) if util else None
        if fps >= target_fps * 0.97:
            verdict = "on target"
        elif fps < HITCH_FPS:
            verdict = "hitch or loading"
        elif busiest and util[busiest] >= SATURATED:
            verdict = f"{ROLE_LABELS.get(busiest, busiest)} saturated"
        else:
            verdict = "no thread saturated (waits or sync)"
        rows.append({"t": cpu["t"], "fps": fps, "frame_ms": frame_ms, "util": util,
                     "cpu_ms_per_frame": {k: v * frame_ms for k, v in util.items() if k != "host_gpu"},
                     "guest_mean": statistics.fmean(guest) if guest else None,
                     "host_gpu_ms": sample["busy_ms"] if sample else None,
                     "submits_per_frame": sample["submits"] if sample else None,
                     "slow": HITCH_FPS <= fps < target_fps * 0.97, "hitch": fps < HITCH_FPS,
                     "verdict": verdict})
        previous_t = cpu["t"]
    return rows


def windowed(items, rows, default_window=10.0):
    """Splits 10 s lines into those covering mostly slow seconds and the rest."""
    slow, ok = [], []
    for item in items:
        window = item.get("window_s", default_window)
        inside = [r for r in rows if item["t"] - window < r["t"] <= item["t"]]
        if not inside:
            continue
        (slow if sum(r["slow"] for r in inside) * 2 > len(inside) else ok).append(item)
    return slow, ok


def mean_of(rows, get):
    values = [v for v in (get(r) for r in rows) if v is not None]
    return statistics.fmean(values) if values else None


def top(entries, key, limit):
    totals = defaultdict(list)
    for entry in entries:
        for name, value in entry[key].items():
            totals[name].append(value["ms_per_s"])
    averaged = {name: sum(v) / len(entries) for name, v in totals.items()}
    return sorted(averaged.items(), key=lambda kv: -kv[1])[:limit]


def analyse(data, target_fps):
    rows = seconds_table(data, target_fps)
    slow = [r for r in rows if r["slow"]]
    ok = [r for r in rows if not r["slow"] and not r["hitch"]]
    threads = sorted({k for r in rows for k in r["util"]})
    groups = {"slow": slow, "on_target": ok}
    result = {
        "seconds": len(rows),
        "on_target_seconds": len(ok),
        "slow_seconds": len(slow),
        "hitch_seconds": sum(r["hitch"] for r in rows),
        "mean_fps": mean_of(rows, lambda r: r["fps"]),
        "verdicts": Counter(r["verdict"] for r in rows if r["slow"] or r["hitch"]).most_common(),
        "threads": {},
        "host_gpu": {},
        "frame_windows": data["frames"],
        "longest_stutter": max(data["stutters"], key=lambda item: item["ms"], default=None),
        "missing": [],
    }
    for name in threads:
        result["threads"][name] = {
            g: {"busy_percent": mean_of(rs, lambda r, n=name: r["util"].get(n) and r["util"][n] * 100),
                "cpu_ms_per_frame": mean_of(rs, lambda r, n=name: r["cpu_ms_per_frame"].get(n))}
            for g, rs in groups.items()}
    if passes := data["passes"]:
        budget_ms = 1000 / target_fps
        result["pass_timing"] = {
            "samples": len(passes),
            "serial_over_budget": sum(p["serial_ms"] > budget_ms for p in passes),
            "heaviest": max(passes, key=lambda p: p["total_ms"]),
        }
    for g, rs in groups.items():
        result["host_gpu"][g] = {"busy_ms_per_frame": mean_of(rs, lambda r: r["host_gpu_ms"]),
                                 "submits_per_frame": mean_of(rs, lambda r: r["submits_per_frame"]),
                                 "frame_ms": mean_of(rs, lambda r: r["frame_ms"])}
    slow_fast, ok_fast = windowed([f for f in data["fast"] if "draws" in f], rows)
    result["draws"] = {}
    for g, fs in (("slow", slow_fast), ("on_target", ok_fast)):
        if fs:
            frame_ms = result["host_gpu"][g]["frame_ms"] or 1000 / target_fps
            result["draws"][g] = {
                "draws_per_s": statistics.fmean(f["draws"] for f in fs),
                "draws_per_frame": statistics.fmean(f["draws"] for f in fs) * frame_ms / 1000,
                "delta_percent": statistics.fmean(100 * f["delta"] / f["draws"] if f["draws"] else 0
                                                  for f in fs),
                "full_prepare_us": statistics.fmean(f["full_us"] for f in fs)}
            same_handle = [f["same_handle_draws_per_s"] for f in fs if "same_handle_draws_per_s" in f]
            if same_handle:
                result["draws"][g]["same_handle_draws_per_s"] = statistics.fmean(same_handle)
    slow_sites, _ = windowed(data["sites"], rows)
    result["slow_sites"] = top(slow_sites, "sites", 8) if slow_sites else []
    result["slow_categories"] = []
    if slow_sites:
        cats = defaultdict(float)
        for s in slow_sites:
            for name, ms in s["cat_ms_per_s"].items():
                cats[name] += ms / len(slow_sites)
        result["slow_categories"] = sorted(cats.items(), key=lambda kv: -kv[1])
    result["slow_zones"] = {}
    result["slow_waits"] = {}
    for thread in ("GPU thread", "GPU back thread", "Vulkan worker"):
        entries = [p for p in data["profile"] if p["thread"] == thread]
        slow_entries, _ = windowed(entries, rows)
        if slow_entries:
            zones = [(n, ms) for n, ms in top(slow_entries, "zones", 20) if n not in WAIT_ZONES]
            result["slow_zones"][thread] = zones[:8]
            result["slow_waits"][thread] = top(slow_entries, "waits", 6)
    result["guest_cores"] = {}
    slow_guest, _ = windowed(data["guest"], rows)
    if slow_guest:
        states = defaultdict(float)
        for g in slow_guest:
            for name, ms in g["states"].items():
                states[name] += ms / len(slow_guest)
        result["guest_cores"] = {"states_ms_per_s": dict(states),
                                 "svcs": top(slow_guest, "svcs", 8)}
    if data["hosttiming"]:
        result["hosttiming"] = {
            "cpu_percent": statistics.fmean(h["cpu_percent"] for h in data["hosttiming"]),
            "wake_late_p99_us": max(h["wake_late_p99_us"] for h in data["hosttiming"])}
    for key, switch in (("cpu", "EDEN_DKB_FRAME_TRACE=1"), ("sites", "EDEN_NX2_STUTTER_TRACE=1"),
                        ("profile", "EDEN_NX2_GPU_PROFILE=1"),
                        ("hosttiming", "EDEN_NX2_HOSTTIMING_STATS=1"),
                        ("guest", "EDEN_NX2_GUEST_PROFILE=1")):
        if not data[key]:
            result["missing"].append(switch)
    return result, rows


def fmt(value, spec=".1f", none="-"):
    return none if value is None else format(value, spec)


def report(result, rows, header, show_windows, target_fps):
    out = [f"# Frame budget: {header}", ""]
    if timing := result.get("tas", {}).get("timing"):
        out.append(f"Whole-route p99 {timing['p99_ns'] / 1e6:.2f} ms, "
                   f"worst interval {timing['max_ns'] / 1e6:.2f} ms. "
                   f"{timing['near_120']:,} of {result['tas']['frames']:,} frames were within "
                   "the 116.4 FPS frame budget.")
        out.append("")
    if not rows:
        out.append("No `DKB cpu` lines: run with EDEN_DKB_FRAME_TRACE=1 (see docs/nx2/profiling.md).")
        return "\n".join(out)
    out.append(f"{result['seconds']} s analysed, mean {fmt(result['mean_fps'])} fps, "
               f"{result['on_target_seconds']} s on target, "
               f"{result['slow_seconds']} s slow ({HITCH_FPS:.0f}-{target_fps * 0.97:.0f} fps), "
               f"{result['hitch_seconds']} s of hitches or loading below {HITCH_FPS:.0f} fps "
               f"(budget {1000 / target_fps:.2f} ms per frame). Thread averages exclude hitch seconds.")
    if windows := result["frame_windows"]:
        out.append(f"Worst sampled frame window p99 {max(window['p99'] for window in windows):.2f} ms, "
                   f"peak frame interval {max(window['max'] for window in windows):.2f} ms.")
    if hitch := result["longest_stutter"]:
        out.append(f"Longest logged stutter {hitch['ms']:.1f} ms at {hitch['t']:.2f} s "
                   f"(frame {hitch['frame']}, {hitch['cause']}).")
    if reads := result.get("readbacks"):
        out.append(f"Reactive readbacks over {reads['seconds']:.1f} s and {reads['frames']:,} frames. "
                   f"{reads['flushes']:,} flushes downloaded {reads['download_kib'] / 1024:,.1f} MiB. "
                   f"Guest threads spent {reads['flush_ms']:,.1f} ms waiting for flushes and "
                   f"{reads['reader_ms']:,.1f} ms waiting for and landing the copies, summed across threads.")
    allocations = result["allocations"]
    if peak := allocations.get("peak_buffer_cache"):
        out.append(f"Peak sampled buffer cache {peak['mib']:,} MiB at {peak['t']:.2f} s "
                   f"({peak['count']:,} buffers across the cache at that sample).")
    if largest := allocations.get("largest_buffer"):
        out.append(f"Largest sampled buffer {largest['bytes'] / 2**20:,.1f} MiB "
                   f"at {largest['device_addr']:#x}, sampled at {largest['t']:.2f} s.")
    if large := allocations.get("large_buffer_allocations"):
        out.append(f"{large['count']:,} logged slow dedicated buffer allocations of at least 512 MiB "
                   f"took {large['ms']:,.1f} ms total. The longest took {large['max_ms']:,.1f} ms.")
    if recycler := allocations.get("recycler"):
        out.append(f"Vulkan image recycler over {recycler['seconds']:.1f} s: "
                   f"{recycler['created']:,} created, {recycler['reused']:,} reused, "
                   f"{recycler['pressure_evictions']:,} pressure evictions, "
                   f"{recycler['peak_kept_mib']:,} MiB peak kept.")
    if uniforms := allocations.get("uniforms"):
        out.append(f"Uniform bindings over {uniforms['frames']:,} sampled frames. "
                   f"{uniforms['kept_full']:,} kept on full draws, {uniforms['kept']:,} kept total, "
                   f"{uniforms['verified']:,} checked with {uniforms['mismatches']:,} mismatches.")
    if storage := allocations.get("storage_scans"):
        out.append(f"Storage upload scans over {storage['frames']:,} sampled frames. "
                   f"{storage['skipped']:,} of {storage['total']:,} eligible scans reused, "
                   f"{storage['mismatches']:,} verification mismatches reported.")
    if sites := allocations.get("hitch_sites"):
        out += ["", "| Site | Largest cost | Calls | Frame interval | Time |",
                "|---|---:|---:|---:|---:|"]
        for name, site in sorted(sites.items(), key=lambda item: -item[1]["ms"]):
            out.append(f"| {name} | {site['ms']:.1f} ms | {site['count']} | "
                       f"{site['interval_ms']:.1f} ms | {site['t']:.2f} s |")
    if result["missing"]:
        out.append(f"Not in this log: {', '.join(result['missing'])}.")
    if passes := result.get("pass_timing"):
        sample = passes["heaviest"]
        out += ["", "## Draw, clear and dispatch work", "",
                f"The heaviest sample ended at {sample['t']:.2f} s and averaged "
                f"{sample['total_ms']:.2f} ms per frame, including {sample['serial_ms']:.2f} ms "
                f"of serial work and {sample['cache_ms']:.2f} ms in serial caches. "
                f"Serial work alone exceeded the frame budget in {passes['serial_over_budget']} "
                f"of {passes['samples']} pass samples.", "",
                "| Workers | Estimated ms per frame | Estimated savings |",
                "|---:|---:|---:|"]
        for workers, key in ((2, "two_ms"), (4, "four_ms"), (8, "eight_ms")):
            out.append(f"| {workers} | {sample[key]:.2f} | {sample['total_ms'] - sample[key]:.2f} ms |")
        out += ["", "The model covers image dependencies. Total frame time also includes methods, "
                "waits outside draws and worker overhead."]
    out += ["", "## What limited the slow seconds", "", "| Verdict | Seconds |", "|---|---|"]
    out += [f"| {v} | {n} |" for v, n in result["verdicts"]] or ["| none slow | 0 |"]
    out += ["", "## Threads (CPU time; spinning counts as busy)", "",
            "| Thread | Busy % slow | ms/frame slow | Busy % on target | ms/frame on target |",
            "|---|---|---|---|---|"]
    for name, g in sorted(result["threads"].items(),
                          key=lambda kv: -(kv[1]["slow"]["busy_percent"] or 0)):
        label = "guest CPU core (busiest)" if name == GUEST else ROLE_LABELS.get(name, name)
        out.append(f"| {label} | {fmt(g['slow']['busy_percent'], '.0f')} | "
                   f"{fmt(g['slow']['cpu_ms_per_frame'])} | {fmt(g['on_target']['busy_percent'], '.0f')} | "
                   f"{fmt(g['on_target']['cpu_ms_per_frame'])} |")
    hs, ho = result["host_gpu"]["slow"], result["host_gpu"]["on_target"]
    out += ["", "## Host GPU and draws", "", "| | Slow | On target |", "|---|---|---|",
            f"| frame ms | {fmt(hs['frame_ms'])} | {fmt(ho['frame_ms'])} |",
            f"| host GPU busy ms/frame | {fmt(hs['busy_ms_per_frame'])} | {fmt(ho['busy_ms_per_frame'])} |",
            f"| Vulkan submits/frame | {fmt(hs['submits_per_frame'], '.0f')} | "
            f"{fmt(ho['submits_per_frame'], '.0f')} |"]
    ds, do = result["draws"].get("slow", {}), result["draws"].get("on_target", {})
    if ds or do:
        out += [f"| draws/frame | {fmt(ds.get('draws_per_frame'), '.0f')} | {fmt(do.get('draws_per_frame'), '.0f')} |",
                f"| delta (fast) draws % | {fmt(ds.get('delta_percent'), '.0f')} | {fmt(do.get('delta_percent'), '.0f')} |",
                f"| full draw prepare us | {fmt(ds.get('full_prepare_us'), '.2f')} | "
                f"{fmt(do.get('full_prepare_us'), '.2f')} |"]
        if "same_handle_draws_per_s" in ds or "same_handle_draws_per_s" in do:
            out.append(f"| unchanged-handle reuses/s | {fmt(ds.get('same_handle_draws_per_s'), '.0f')} | "
                       f"{fmt(do.get('same_handle_draws_per_s'), '.0f')} |")
    if memo := result.get("texture_memo"):
        total = memo["hits"] + memo["full"]
        if total:
            out += ["", "## Sampled textures", "",
                    f"Memo hits on {memo['hits']:,} of {total:,} eligible draws "
                    f"({100 * memo['hits'] / total:.1f}%) over {memo['window_s']:.1f} s. "
                    f"{memo['second_hits']:,} hits used the second lookup."]
    if aliases := result.get("vertex_type_cache_hits"):
        out += ["", f"Vertex type cache reused a built pipeline for {aliases:,} logged cache misses."]
    if result["slow_categories"]:
        out += ["", "## Stutter categories in slow windows (ms per second)", "",
                ", ".join(f"{n} {ms:.1f}" for n, ms in result["slow_categories"] if ms >= 0.1)]
    if result["slow_sites"]:
        out += ["", "## Rare costly sites in slow windows (ms per second)", "",
                ", ".join(f"{n} {ms:.1f}" for n, ms in result["slow_sites"])]
    for thread, zones in result["slow_zones"].items():
        out += ["", f"## {thread}: busiest zones in slow windows (ms per second)", "",
                ", ".join(f"{n} {ms:.1f}" for n, ms in zones)]
        if result["slow_waits"].get(thread):
            out.append("Waits: " + ", ".join(f"{n} {ms:.1f}" for n, ms in result["slow_waits"][thread]))
    if result["guest_cores"]:
        g = result["guest_cores"]
        out += ["", "## Guest cores in slow windows (mean of one core, ms per second)", "",
                ", ".join(f"{n} {ms:.1f}" for n, ms in sorted(g["states_ms_per_s"].items(),
                                                              key=lambda kv: -kv[1])),
                "System calls: " + ", ".join(f"{n} {ms:.1f}" for n, ms in g["svcs"]),
                "JIT is guest code (spin loops included); idle spin is host polling before an idle "
                "core blocks (EDEN_NX2_IDLE_SPIN_US)."]
    if "hosttiming" in result:
        h = result["hosttiming"]
        out += ["", f"HostTiming thread CPU {h['cpu_percent']:.0f}%, worst wake late p99 "
                    f"{h['wake_late_p99_us']} us."]
    if show_windows:
        names = sorted({k for r in rows for k in r["util"]})
        out += ["", "## Per second", "", "| t | fps | " + " | ".join(names) + " | verdict |",
                "|" + "---|" * (len(names) + 3)]
        for r in rows:
            cells = " | ".join(fmt(r["util"].get(n) and r["util"][n] * 100, ".0f") for n in names)
            out.append(f"| {r['t']:.1f} | {r['fps']:.1f} | {cells} | {r['verdict']} |")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--session", type=int, default=-1, help="game session index (default: last)")
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tas", action="store_true", help="measure the last completed TAS playback")
    mode.add_argument("--tas-ab", action="store_true", help="compare the last completed idle A/B")
    mode.add_argument("--compare-to", metavar="BASELINE",
                      help="compare a TAS run log or directory with a baseline log or directory")
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--windows", action="store_true", help="also print one row per second")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.compare_to is not None:
        if args.start_s is not None or args.end_s is not None or args.json or args.windows:
            parser.error("--compare-to reports both whole TAS replays as Markdown")
        try:
            print(compare_tas_runs(args.compare_to, args.log, args.session))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return 0

    build, sessions = read_sessions(args.log)
    try:
        session = sessions[args.session]
    except IndexError:
        parser.error(f"the log has {len(sessions)} session(s)")
    if args.tas_ab:
        if args.start_s is not None or args.end_s is not None or args.windows:
            parser.error("--tas-ab reports the complete A/B phases")
        result = last_tas_ab(session["lines"])
        if result is None:
            parser.error("no complete TAS A/B in this game session")
        result.update(build=build, program=session["program"], title=session["title"])
        print(json.dumps(result, indent=2) if args.json else report_tas_ab(
            result, session["title"] or session["program"] or args.log))
        return 0
    tas = None
    if args.tas:
        if args.start_s is not None or args.end_s is not None:
            parser.error("--tas supplies the measurement bounds")
        tas = last_tas_playback(session["lines"])
        if tas is None:
            parser.error("no complete TAS playback in this game session")
        args.start_s, args.end_s = tas["start_s"], tas["end_s"]
    lines = [(t, text) for t, text in session["lines"]
             if (args.start_s is None or t >= args.start_s) and (args.end_s is None or t <= args.end_s)]
    data = parse(lines)
    result, rows = analyse(data, args.target_fps)
    if data["vertex_type_cache"]:
        result["vertex_type_cache_hits"] = sum(data["vertex_type_cache"])
    memo_windows = [w for w in data["texture_memo"]
                    if args.start_s is None or w["t"] - w["window_s"] >= args.start_s]
    if memo_windows:
        result["texture_memo"] = {key: sum(w[key] for w in memo_windows)
                                  for key in ("window_s", "hits", "full", "second_hits")}
        result["texture_memo"]["mismatches_since_startup"] = memo_windows[-1]["mismatches"]
    result.update(build=build, program=session["program"], title=session["title"],
                  allocations=allocation_diagnostics(lines))
    if reads := readback_diagnostics(lines, args.start_s):
        result["readbacks"] = dict(reads)
    if tas:
        result["tas"] = tas
        # A timing window logged after playback began can still contain pre-route frames.
        result["frame_windows"] = [
            window for window in result["frame_windows"]
            if window["t"] - window["samples"] * window["mean"] / 1000 >= tas["start_s"]
        ]
    if args.json:
        result["per_second"] = rows
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        header = " | ".join(x for x in (session["title"], session["program"], build) if x)
        if tas:
            header += (f" | TAS {tas['frames']} frames in {tas['seconds']:.2f} s "
                       f"({tas['fps']:.2f} FPS, including loads)")
        print(report(result, rows, header or args.log, args.windows, args.target_fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
