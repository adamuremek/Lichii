# Frame budget profiling

To find what holds a game below its target frame rate, run it with the frame budget tracers, then let
`tools/nx2/frame_budget.py` name the limiting thread for each second.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File tools\nx2\profile_run.ps1 -Exe <dir>\lichii.exe
```

The script sets the tracers below, starts Lichii and waits for it to exit. It then copies
`%APPDATA%\eden\log\eden_log.txt` and writes `frame_budget.md` to `<dir>\profile-runs\<time>\`. Play
the slow scene for at least 30 s. `-Light` leaves out the GPU profile and `-Extra "A=1 B=2"` adds
other switches, such as an experiment to A/B.

| Switch | What the report uses |
|---|---|
| `EDEN_DKB_FRAME_TRACE=1` | `DKB cpu` (each thread's CPU time per second, the six guest cores included), `DKB gpu_frame` (host GPU time and Vulkan submits per frame), `Frame timing window` (p50/p95/p99) |
| `EDEN_NX2_STUTTER_TRACE=1` | `Nx2StutterSummary` categories and rare costly sites (image and memory allocation, staging, `Scheduler::Finish`), and `Fast draw per second` (draws per frame, delta share, prepare cost) |
| `EDEN_NX2_GPU_PROFILE=1` | Per-zone time of the GPU thread, the GPU back thread and the Vulkan worker, plus what each waits for |
| `EDEN_NX2_HOSTTIMING_STATS=1` | HostTiming thread CPU and timer lateness |
| `EDEN_NX2_GUEST_PROFILE=1` | One `Guest core <n> per second` line per core every 10 s. It splits the core thread's wall time into `jit` (guest code), `svc` (system calls, with the eight costliest by name and SleepThread yields on their own), `kernel` (scheduling between them), `idle spin` (polling before an idle core blocks, set by `EDEN_NX2_IDLE_SPIN_US`) and `idle wait` (blocked) |

To analyse an existing log, use:

```sh
python3 -I tools/nx2/frame_budget.py eden_log.txt [--session N] [--start-s S --end-s S] [--windows] [--json]
```

## Reading the report

The report sorts each second into one of these groups:

- *On target*: the second ran at 97% or more of `--target-fps` (default 60).
- *Hitch or loading*: under 20 fps. These seconds are counted but left out of the averages.
- *Slow*: everything between. The verdict names the busiest thread, if it used at least 85% of wall
  time. Guest cores count as one, the busiest of the six. Host GPU busy time is set against the frame
  time.

A verdict is where to start looking, not proof:

- Thread CPU time includes spinning. A saturated thread is the limit only if its GPU profile zones
  show work rather than waits (`split wait`, `writer wait`, `waits for the back by site`).
- *No thread saturated* means the threads are taking turns waiting on each other. Read the waits
  sections and the per-frame `Nx2StutterDetail` lines.
- *Guest CPU core saturated*: read the guest cores section. If `idle spin` is large, the cores are
  polling for work, not running the game. Compare with `-Extra "EDEN_NX2_IDLE_SPIN_US=0"`. If
  SleepThread yields or synchronization calls are large, guest threads are waiting on each other or
  on the GPU. If `jit` takes the time, the game's own code takes the frame. That includes guest spin
  loops that make no system calls. To see which guest functions are hot, take a CPU capture with
  Windows Performance Recorder.
- Slow and on-target seconds usually come from different scenes. To measure a change, compare the
  same scene with `cache_metrics.py compare` or two `--json` reports. Don't compare the slow and
  on-target columns of one run.

## Watching a session live

To record problems as they happen while you play, follow the log:

```sh
python3 -I tools/nx2/live_issues.py eden_log.txt --out live-issues.md [--from-start]
```

It prints one status line every 10 s of game time: fps, hitches, shader builds and the time spent
waiting on them, Bounded wait time and skipped draws, and the busiest guest core and GPU back
thread. Each new problem goes once, with its game time, to the markdown file: hitch bursts, shader
build bursts, a saturated thread, each new warning or error kind, and files the guest keeps
re-reading (with `EDEN_NX2_FS_READ_TRACE=2`). Re-read loops are logged as episodes, when they start
and stop, with the areas whose files were loading around then. It only reads the log.
