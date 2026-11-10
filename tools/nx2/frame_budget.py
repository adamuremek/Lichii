#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where each second's frame time went, and which thread limited it.

  frame_budget.py <log> [--session N] [--start-s S] [--end-s S] [--target-fps 60] [--windows] [--json]

Reads the lines of a profiling run (see docs/nx2/profiling.md):
  EDEN_DKB_FRAME_TRACE=1    DKB cpu (per-thread CPU time each second), DKB gpu_frame, Frame timing window
  EDEN_NX2_STUTTER_TRACE=1  Nx2StutterSummary (costly sites per 10 s), Fast draw per second
  EDEN_NX2_GPU_PROFILE=1    "<thread> per second:" zone times and wait sites per 10 s
  EDEN_NX2_HOSTTIMING_STATS=1  HostTiming wait lines
  EDEN_NX2_GUEST_PROFILE=1  "Guest core <n> per second:" JIT, system call, kernel and idle time per 10 s
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

TIME = re.compile(r"^\[\s*(\d+\.\d+)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
SESSION_START = re.compile(r"Booting game: ([0-9A-Fa-f]{16})(?: \| ([^|]*))?")
BUILD = re.compile(r"Lichii Version: (.*)$")
CPU = re.compile(r"DKB cpu through_frame=\d+ frames=(\d+) window_wall_ms=([\d.]+)(.*)$")
CPU_ROLE = re.compile(r"(\w+?)(?:_host)?_cpu_ms=([\d.]+)")
GPU_FRAME = re.compile(r"DKB gpu_frame .*gpu_frame_span_ms=([\d.]+) gpu_submission_sum_ms=([\d.]+) "
                       r"measured_submissions=(\d+)")
FRAME_WINDOW = re.compile(r"Frame timing window .*samples=(\d+) .*p50_ms=([\d.]+) p95_ms=([\d.]+) "
                          r"p99_ms=([\d.]+) max_ms=([\d.]+)")
FAST_DRAW = re.compile(r"Fast draw per second: (\d+) delta")
FAST_PREPARE = re.compile(r"prepare us per draw: full ([\d.]+), delta ([\d.]+), all ([\d.]+) \((\d+) draws/s\)")
SUMMARY_SITES = re.compile(r"Nx2StutterSummary window_s=([\d.]+) .*?sites=(\S+)")
SUMMARY_CATS = re.compile(r"Nx2StutterSummary .*?cat_ms=(\S+)")
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
            data["frames"].append({"t": t, "samples": int(m.group(1)), "p50": float(m.group(2)),
                                   "p95": float(m.group(3)), "p99": float(m.group(4)),
                                   "max": float(m.group(5))})
        elif m := FAST_DRAW.search(text):
            data["fast"].append({"t": t, "delta": int(m.group(1))})
        elif m := FAST_PREPARE.search(text):
            if data["fast"]:
                data["fast"][-1].update(full_us=float(m.group(1)), delta_us=float(m.group(2)),
                                        all_us=float(m.group(3)), draws=int(m.group(4)))
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
        "slow_seconds": len(slow),
        "hitch_seconds": sum(r["hitch"] for r in rows),
        "mean_fps": mean_of(rows, lambda r: r["fps"]),
        "verdicts": Counter(r["verdict"] for r in rows if r["slow"] or r["hitch"]).most_common(),
        "threads": {},
        "host_gpu": {},
        "frame_windows": data["frames"],
        "missing": [],
    }
    for name in threads:
        result["threads"][name] = {
            g: {"busy_percent": mean_of(rs, lambda r, n=name: r["util"].get(n) and r["util"][n] * 100),
                "cpu_ms_per_frame": mean_of(rs, lambda r, n=name: r["cpu_ms_per_frame"].get(n))}
            for g, rs in groups.items()}
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
    if not rows:
        out.append("No `DKB cpu` lines: run with EDEN_DKB_FRAME_TRACE=1 (see docs/nx2/profiling.md).")
        return "\n".join(out)
    out.append(f"{result['seconds']} s analysed, mean {fmt(result['mean_fps'])} fps, "
               f"{result['slow_seconds']} s slow ({HITCH_FPS:.0f}-{target_fps * 0.97:.0f} fps), "
               f"{result['hitch_seconds']} s of hitches or loading below {HITCH_FPS:.0f} fps "
               f"(budget {1000 / target_fps:.2f} ms per frame). Averages leave the hitches out.")
    if result["missing"]:
        out.append(f"Not in this log: {', '.join(result['missing'])}.")
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
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--windows", action="store_true", help="also print one row per second")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    build, sessions = read_sessions(args.log)
    try:
        session = sessions[args.session]
    except IndexError:
        parser.error(f"the log has {len(sessions)} session(s)")
    lines = [(t, text) for t, text in session["lines"]
             if (args.start_s is None or t >= args.start_s) and (args.end_s is None or t <= args.end_s)]
    result, rows = analyse(parse(lines), args.target_fps)
    result.update(build=build, program=session["program"], title=session["title"])
    if args.json:
        result["per_second"] = rows
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        header = " | ".join(x for x in (session["title"], session["program"], build) if x)
        print(report(result, rows, header or args.log, args.windows, args.target_fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
