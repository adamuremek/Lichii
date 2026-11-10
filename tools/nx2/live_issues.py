#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Follow a running Lichii log and record performance problems as they happen.

  live_issues.py <eden_log.txt> [--out issues.md] [--window 10] [--from-start]

Every window (default 10 s of game time) it prints one status line: frames per second, frames
over 50/100 ms, shader pipelines built and the time spent waiting on them, the busiest guest core
and GPU thread, and the worst re-read file. It also records each new problem once, with its game
time, in the markdown file: hitch bursts, shader build bursts, a saturated thread, files the guest
keeps re-reading, and each new warning or error kind. Uses the lines of the frame budget tracers
(docs/nx2/profiling.md); missing tracers just leave their columns empty. Reads the log only.

Standard library only.
"""

import argparse
import os
import re
import sys
import time
from collections import Counter

TIME = re.compile(r"^\[\s*(\d+\.\d+)\]")
BOOT = re.compile(r"Booting game: ([0-9A-Fa-f]{16})")
STUTTER = re.compile(r"Nx2Stutter frame=\d+ ms=([\d.]+) cause=(\w+)")
SUMMARY = re.compile(r"Nx2StutterSummary .*?frames=(\d+) over33=(\d+) over100=(\d+)")
SUMMARY_FIELD = re.compile(r"(gfx_miss|pipeline_wait|shader_capture|texture|buffer|query_fence):([\d.]+)")
BUILDS = re.compile(r"builds=gfx:(\d+),compute:(\d+)")
NEW = re.compile(r"gfx_new:(\d+)")
CPU = re.compile(r"DKB cpu through_frame=\d+ frames=(\d+) window_wall_ms=([\d.]+)(.*)$")
ROLE = re.compile(r"(\w+?)(?:_host)?_cpu_ms=([\d.]+)")
REUSE_FILE = re.compile(r"Guest read reuse file (\S+) \(([\d.]+) MiB at 0x[0-9a-f]+\): (\d+) reads, "
                        r"([\d.]+) MiB/s, re-read ([\d.]+) MiB/s")
# Area names in streamed paths: scene, terrain, bake and placement files, and landscape models.
AREA = re.compile(r"/(?:Scene|TerrainArc|Bake/Probe/Scene|Foliage/Placement/Scene|PreComputedCulling/Scene)"
                  r"/([A-Za-z]+[A-Za-z0-9]*?)(?:_\d+|_Child|_Adult)?[/._]|/Model/Land_([A-Za-z]+)")
BOUNDED = re.compile(r"Bounded pipeline wait: ([\d.]+) ms spent in the last 10 s")
SKIPPED = re.compile(r"Async shaders: (\d+) pipeline lookups skipped")
NATIVE_MODE = re.compile(r"LogSettings: .. Renderer\.native_async_draws: (\w+)")
LEVEL = re.compile(r"^\[\s*[\d.]+\] (\S+) <(Warning|Error|Critical)> ([^:]+):\d+:(\w+): (.*)$")
# Numbers and hex words, so repeats of one warning with other values count as one kind.
DIGITS = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]*\d[0-9a-fA-F]*\b")

HITCH_MS = 100.0
REREAD_MIB_S = 2.0
SATURATED = 0.85


class Window:
    def __init__(self, start):
        self.start = start
        self.hitches = []
        self.causes = Counter()
        self.cpu = []
        self.summary = None
        self.reuse = []
        self.bounded_ms = None


class Monitor:
    def __init__(self, out_path, window_s):
        self.out_path = out_path
        self.window_s = window_s
        self.window = None
        self.seen_kinds = Counter()
        self.reported = set()
        self.issues = 0
        self.skipped = None
        self.reported_skipped = None
        self.native_mode = None
        self.loops = {}
        self.loop_quiet = {}
        self.recent_areas = []
        self.out = open(out_path, "a", encoding="utf-8")
        self.out.write(f"\n## Live session started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.out.flush()

    def note(self, t, text):
        self.issues += 1
        line = f"- **{t:8.1f} s** {text}"
        print("  ISSUE " + line[2:], flush=True)
        self.out.write(line + "\n")
        self.out.flush()

    def feed(self, line):
        m = TIME.match(line)
        if not m:
            return
        t = float(m.group(1))
        if BOOT.search(line):
            self.flush(t)
            self.window = None
            self.reported.clear()
            self.note(t, f"game booted ({BOOT.search(line).group(1)})")
        if self.window is None:
            self.window = Window(t)
        elif t - self.window.start >= self.window_s:
            self.flush(t)
            self.window = Window(t)
        w = self.window
        if m2 := STUTTER.search(line):
            ms = float(m2.group(1))
            if ms >= HITCH_MS:
                w.hitches.append(ms)
                w.causes[m2.group(2)] += 1
        elif "Nx2StutterSummary" in line and (m2 := SUMMARY.search(line)):
            fields = {k: float(v) for k, v in SUMMARY_FIELD.findall(line.split("builds=")[0])}
            b = BUILDS.search(line)
            n = NEW.search(line)
            w.summary = {"frames": int(m2.group(1)), "over33": int(m2.group(2)),
                         "over100": int(m2.group(3)), "gfx": int(b.group(1)) if b else 0,
                         "compute": int(b.group(2)) if b else 0, "new": int(n.group(1)) if n else 0,
                         **fields}
        elif m2 := CPU.search(line):
            frames, wall = int(m2.group(1)), float(m2.group(2))
            if frames and wall > 0:
                roles = {k: float(v) / wall for k, v in ROLE.findall(m2.group(3))
                         if not k.endswith("_per_frame")}
                guest = max((v for k, v in roles.items() if k.startswith("guest_")), default=0.0)
                w.cpu.append({"fps": frames * 1000.0 / wall, "guest": guest,
                              "back": roles.get("gpu_back", 0.0), "front": roles.get("gpu_command", 0.0)})
        elif m2 := BOUNDED.search(line):
            w.bounded_ms = float(m2.group(1))
        elif m2 := SKIPPED.search(line):
            self.skipped = int(m2.group(1))
        elif m2 := NATIVE_MODE.search(line):
            if m2.group(1) != self.native_mode:
                self.native_mode = m2.group(1)
                self.note(t, f"Switch 2 draws with unbuilt pipelines: `{m2.group(1)}`")
        elif m2 := REUSE_FILE.search(line):
            w.reuse.append((m2.group(1), float(m2.group(5)), int(m2.group(3))))
        elif m2 := LEVEL.match(line):
            kind = f"{m2.group(1)} {m2.group(2)} {m2.group(3)}:{m2.group(4)}: " + \
                   DIGITS.sub("N", m2.group(5))[:120]
            self.seen_kinds[kind] += 1
            if self.seen_kinds[kind] == 1:
                self.note(t, f"new {m2.group(2).lower()}: `{m2.group(5)[:160]}` ({m2.group(3)})")

    def flush(self, now):
        w = self.window
        if w is None:
            return
        fps = sum(c["fps"] for c in w.cpu) / len(w.cpu) if w.cpu else None
        low = min((c["fps"] for c in w.cpu), default=None)
        guest = max((c["guest"] for c in w.cpu), default=None)
        back = max((c["back"] for c in w.cpu), default=None)
        s = w.summary or {}
        worst = max(w.reuse, key=lambda r: r[1], default=None)
        parts = [f"[{w.start:7.1f}-{now:7.1f} s]",
                 f"fps {fps:5.1f} (low {low:4.1f})" if fps is not None else "fps    -",
                 f"hitches>{HITCH_MS:.0f}ms {len(w.hitches)}",
                 f"builds {s.get('gfx', 0)}+{s.get('compute', 0)} ({s.get('new', 0)} new) "
                 f"wait {s.get('pipeline_wait', 0) / 1000:.1f}s" if s else "builds -",
                 f"guest core {guest * 100:3.0f}% back {back * 100:3.0f}%" if guest is not None else ""]
        if w.bounded_ms is not None:
            parts.append(f"bounded wait {w.bounded_ms / 1000:.2f}s")
        if self.skipped is not None and self.skipped != self.reported_skipped:
            parts.append(f"skipped lookups {self.skipped}")
            self.reported_skipped = self.skipped
        if worst and worst[1] >= REREAD_MIB_S:
            parts.append(f"re-read {worst[0].rsplit('/', 1)[-1]} {worst[1]:.1f} MiB/s")
        print(" | ".join(p for p in parts if p), flush=True)
        # Issues, each kind once until it clears.
        if len(w.hitches) >= 3:
            cause = w.causes.most_common(1)[0][0]
            self.note(w.start, f"{len(w.hitches)} hitches over {HITCH_MS:.0f} ms (worst "
                               f"{max(w.hitches):.0f} ms), mostly `{cause}`")
        if s and (s.get("pipeline_wait", 0) >= 500 or s.get("new", 0) >= 10):
            self.note(w.start, f"shader builds: {s['gfx']} graphics + {s['compute']} compute "
                               f"pipelines ({s['new']} never cached), {s.get('pipeline_wait', 0) / 1000:.1f} s "
                               f"waiting, {s.get('gfx_miss', 0):.0f} ms building on the GPU thread")
        if fps is not None and fps < 55 and not w.hitches and guest is not None:
            who = "guest CPU core" if guest >= max(back, SATURATED) else \
                  "GPU back thread" if back >= SATURATED else None
            if who and ("slow", who) not in self.reported:
                self.reported.add(("slow", who))
                self.note(w.start, f"steady {fps:.0f} fps with the {who} saturated "
                                   f"(guest {guest * 100:.0f}%, back {back * 100:.0f}%)")
        elif fps is not None and fps >= 58:
            self.reported.discard(("slow", "guest CPU core"))
            self.reported.discard(("slow", "GPU back thread"))
        self.track_loops(w, now)

    def track_loops(self, w, now):
        """Re-read loops as episodes: when one starts or stops, with the area files around it."""
        hints = sorted({m.group(1) or m.group(2) for path, rate, _ in w.reuse
                        if rate < REREAD_MIB_S and (m := AREA.search(path))})
        if hints:
            self.recent_areas.append((w.start, hints))
            self.recent_areas = self.recent_areas[-3:]
        context = "; ".join(f"{t:.0f} s: {', '.join(h[:4])}" for t, h in self.recent_areas) or "none"
        looping = {path: (rate, reads) for path, rate, reads in w.reuse if rate >= REREAD_MIB_S}
        for path, (rate, reads) in looping.items():
            if path not in self.loops:
                self.loops[path] = w.start
                self.note(w.start, f"re-read loop **started**: `{path}` ({rate:.1f} MiB/s, {reads} "
                                   f"reads in 10 s). Areas loading lately: {context}")
            self.loop_quiet[path] = 0
        for path in list(self.loops):
            if path in looping:
                continue
            # The reuse report skips quiet 10 s windows, so two quiet windows end an episode.
            self.loop_quiet[path] = self.loop_quiet.get(path, 0) + 1
            if self.loop_quiet[path] >= 2:
                began = self.loops.pop(path)
                self.note(now, f"re-read loop **stopped** after {now - began:.0f} s: `{path}`. "
                               f"Areas loading lately: {context}")


def follow(path, start_at_end):
    f = open(path, encoding="utf-8", errors="replace")
    if start_at_end:
        f.seek(0, os.SEEK_END)
    inode = os.stat(path).st_size
    buf = ""
    while True:
        chunk = f.readline()
        if chunk:
            buf += chunk
            if buf.endswith("\n"):
                yield buf.rstrip("\n")
                buf = ""
            continue
        time.sleep(0.5)
        try:
            size = os.stat(path).st_size
        except OSError:
            continue
        if size < f.tell():  # a new boot rotated or truncated the log
            f.close()
            f = open(path, encoding="utf-8", errors="replace")
        inode = size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--out", default="live_issues.md")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--from-start", action="store_true", help="read the existing log first")
    args = parser.parse_args(argv)
    monitor = Monitor(args.out, args.window)
    print(f"following {args.log}; issues go to {args.out}", flush=True)
    try:
        for line in follow(args.log, not args.from_start):
            monitor.feed(line)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
