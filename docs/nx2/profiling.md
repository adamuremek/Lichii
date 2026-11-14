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

## Replaying a gameplay route

On Windows, launch the OotPgo package with `-RecordRoute`. Load a save just before the area
transition, select **TAS → Record**, and play through the transition and the slow section. Select
**TAS → Stop Recording** before closing Lichii. The launcher prints the path to the run's
`route.txt`.

For subsequent runs, use `-ReplayRoute <path-to-route.txt>`. Load the same save, face the same
direction, and select **TAS → Start** once. The launcher saves a copy of the route alongside the
log. Add `-CompareTo <baseline-run-directory>` on the candidate run to compare complete replays.
Recording and replay enable TAS for that launch.

### Using the regular TAS configuration

In **TAS → Configure TAS**, enable TAS features and turn off **Loop script**. Choose a separate
script directory if you already have a player 1 script. Load the same save and face the same
direction before each run. Use **TAS → Record** to capture a route once, including the area
transition and the heavy scene. Stop recording and save it as the player 1 script.

For each build, launch the game, load the starting save, then select **TAS → Reset** and
**TAS → Start**. Keep the game window visible while the route plays. Use the same graphics
settings and comparable shader-cache conditions. Check the ending position when comparing runs
with different frame rates, since TAS advances on presented frames. The playback markers let
the report select the last complete route and include loads in its route FPS.

```sh
python3 -I tools/nx2/frame_budget.py eden_log.txt --tas --target-fps 120 --windows
```

For a stationary benchmark, the local launcher accepts `-IdleReplayFrames 3600`. Enable TAS
features, load a heavy scene with the same camera and save on each run, then select
**TAS → Start** and leave the controls idle. This uses neutral input for a fixed number of frames
without touching the saved TAS files. Close the emulator after TAS completes to save the results.

`-TextureReadBenchmark` compares the Vulkan texture read filter in one stationary run. Load a
save in the heavy scene, keep the render window visible, and start TAS once. The four phases run
with the filter off, on, on, then off, discarding 120 warmup frames in each phase. The launcher
saves `texture_read_benchmark.md` with frame-time percentiles and the number of texture checks
the filter skipped. Use a recorded route when measuring area transitions.

`-BufferReadBenchmark` runs the same four phases for the Switch 2 buffer read filter. Its report,
`buffer_read_benchmark.md`, counts the lookups that bypassed the buffer cache lock. Use
`-BufferReadFilter` to enable the filter for a full recorded route or normal play.

`-AcquireWaitBenchmark` compares semaphore acquire wait targets in four idle phases.
Its report, `acquire_wait_benchmark.md`, includes frame times and how often the candidate
selected an earlier wait position. Use `-PreciseAcquireWait` to enable the candidate during
normal play or a recorded route.

New builds record the interval for every frame in a completed replay. To compare two saved runs,
pass `-CompareTo` with the baseline run directory when launching the candidate, or run the report
offline. Both runs must have the same game ID and input script. Use the same save, settings, and
camera and check that both runs end in the same scene.
The replay comparison includes sampled GPU buffer sizes and the largest allocation and
collection costs on logged hitches when those trace lines are available.
The local launcher saves `frame_budget.md` for runs with frame timing and
`frame_budget_tas.md` for completed replays.

```sh
python3 -I tools/nx2/frame_budget.py <candidate-run> --compare-to <baseline-run>
```

Use `-ReuseStaging` with the OotPgo launcher to test the Vulkan staging pool's quarter-step size
classes and idle-buffer reuse. Replay a route across an area boundary, once with and once without
the switch. The comparison reports staging allocations per second and the time they took across
complete trace windows inside each replay.

`-UploadArena` lets non-deferred uploads share up to four 64 MiB staging buffers when the stream
and size-class caches miss. Use the same route and staging settings with and without the switch.

`-WarmUploadArena` reserves up to four 256 MiB mapped host buffers at Vulkan startup on
discrete GPUs with sufficient host memory. Buffer-cache uploads over 256 MiB can use multiple
arena slices when they replace a dedicated allocation or reduce its size class. Compare the
same route across an area boundary, starting each replay from a fresh launch. Check the
`Large buffer upload` log lines for arena and spill sizes, then compare `staging_create` time
and maximum frame time during the transition.

Use `-RetainLargeBuffers` to keep one 768 MiB to 1,152 MiB Switch 2 buffer in the cache across
brief trips out of a scene on a discrete GPU with sufficient memory budget. For a TAS
comparison, record a route that visits a heavy area, leaves it, then returns. Keep the upload
arena settings the same across both runs. The report includes the count and time of logged
slow dedicated buffer allocations of at least 512 MiB.

Use `-BoundComputeReads` to trim read-only Vulkan storage bindings when the native compute shader's
loads have a proven bound. The launcher can compare the same TAS route with this switch on and
off. Compare the frame intervals and peak buffer cache size with the other switches held fixed.

`-RecycleHeadroom` lets the Vulkan image recycler retain images up to its existing pool cap
within the driver and host-memory budgets. Run the same TAS route with the switch on and off,
using `-MemoryTrace` both times. The TAS comparison reports created and reused images, pressure
evictions, and peak recycler memory over complete sample windows.

`-FastLookup` tests the word-at-a-time comparisons in Ocarina's resource-name search. The
boot log records whether the three comparison sites were patched. Use `-GuestTiming` for both route replays
to compare guest CPU time and frame times.

`-JitScratchCopy` lets the x64 JIT use an empty host register for a scratch result when it
still needs the source value. Compare guest CPU time on the same TAS route with the switch on
and off.

`-FastUnsignedDiv` lets the x64 JIT return zero for an unsigned quotient when the dividend is
smaller than the divisor. Use the same TAS route and `-GuestTiming` for both settings.

`-CacheVertexInput` tests the Vulkan dynamic vertex input layout cache. The `Vertex input cache`
log reports newly built layouts, reused layouts, redundant commands skipped within a recording
epoch, and requests beyond the cache's capacity. Use the same heavy scene and TAS input in both runs.

`-ReuseVertexTypes` lets a Vulkan pipeline with dynamic vertex input reuse its shader when
only unread attribute types change. On an exact cache miss, the renderer can also find
an equivalent built pipeline after switching shaders. Draws still record the current
vertex input layout. The fast-draw log counts consecutive reuse, and `Vertex type cache`
reports how often the cache miss found an equivalent pipeline.

`-CacheShaderStages` reuses native Vulkan shader stages across boot and background pipeline
builds when the environment reads, runtime info, and descriptor bindings match. The `Stage cache`
log reports how many stages skipped both translation and SPIR-V emission. Compare shader load
time and frame pacing across the same area transitions with the switch on and off.

`-ReuseRasterState` keeps the resolved Vulkan draw across changes to viewport scale, offset
and bounds, dynamic culling, and depth-clip changes that preserve the pipeline's depth clamp
setting. It can also keep the draw across trailing color masks, unused per-target blend
equations, and writes to the currently unemulated constant-color registers. Slot zero still
tracks dual-source blending. Compare full and delta draws per second on the same TAS route.
`EDEN_NX2_FAST_DRAW_VERIFY=1` checks the reused pipeline and bindings against a full resolve.

`-CoalesceDrawRegisters` records one fast-draw invalidation for the 3D register changes between
draws. The engine still applies each register write and its dirty flags. When the log attributes
a reuse break to a register, it names the first register that invalidated the draw.

`-BatchState` processes runs of direct and queued 3D state writes in the split GPU back thread.
Queued writes are applied when the engine normally consumes them. The replay stops for draws,
waiters, requested ticks, and ring publication. Use it with the launcher's normal profiling mode.

`-NativeLse` tests native x64 lowering of Ocarina's 32- and 64-bit guest LDADD and CAS. It applies
when fastmem exclusives and the existing CPU setting for bypassing the global monitor are active.
The log reports whether the path was enabled. Use the same CPU settings for both replays.
Add `-GuestTiming` to both launches to include guest JIT and kernel time in the frame-budget
reports without collecting JIT code dumps.

`-SceneTransfer` uses a Vulkan transfer for Ocarina's No Sharpening scene copy when the source
and destination are exact 1:1 color images. Use the same heavy scene in both runs. The comparison
reports transfer counts and sampled GPU submission times.

`-FastDirtyGather` skips GPU fence locks for CPU write trackers with no pending writes. Repeat the
same TAS replay with and without the switch to measure the effect on frame times.

`-MapQueryCache` resolves GPU address ranges on demand and reuses recently queried intervals
until a map changes. Storage-buffer setup can reuse layout and device-contiguous sizes in a
single lookup. Replay the same heavy scene with and without the switch to compare back-thread
time. The hidden `[.mapping_query_benchmark]` test measures separate and combined queries with
steady mappings and periodic remaps.

`-PreparedTargets` reuses prepared color and depth attachments within a frame when their mapping
and image state allow it. The log reports how many draws reused preparation.

`-AdaptiveSubmit` can submit recorded Vulkan work before the normal native draw budget when the
command thread waits or the host GPU catches up. Compare replay frame times and the
`Adaptive native submissions` counts on the same route.

`-AdaptiveSubmitAB` measures both modes in one launch. Enable TAS, load a heavy scene, select
**TAS → Start**, and leave the controls idle until playback ends. The 3600-frame default runs
baseline, candidate, candidate, baseline with 120 warmup frames excluded from each phase.
Close the emulator to write `adaptive_submit_ab.md` next to the archived log.

`-RendererBenchmark` measures baseline, adaptive submission, raster-state reuse, and both
together in a single launch. Enable TAS, load a heavy scene, and select **TAS → Start**.
Keep the game window visible and leave the controls idle until playback ends. The default
7200 frames run the four modes in forward and reverse order with 120 warmup frames per phase.
The launcher writes `renderer_benchmark.md` when the emulator closes. Use a recorded TAS
route for a separate test of area transitions.

`-LocalCpuCaches` tests Windows CPU placement when **Pin GPU threads** is enabled. On a
compatible two-cache CPU, the six guest threads get separate physical cores in one
last-level cache. The GPU command, GPU back, and Vulkan worker threads share the other
cache on three separate cores. The startup log reports the placement.

`-CpuPlacementBenchmark` compares the original and cache-local CPU sets in one
launch on supported Windows CPUs. Leave **Pin GPU threads** enabled, load a heavy
scene, select **TAS → Start**, and leave the visible game idle. The 3600-frame
default runs original, cache-local, cache-local, original with 120 warmup frames
excluded per phase. Other selected optimization flags apply throughout the run.
Close the emulator to write `cpu_placement_benchmark.md` next to the archived log.

`-BatchMacroWrites` buffers register writes from the x64 GPU command thread's Turing macros.
The JIT delivers them in order before register reads and macro DMA. Compare the same heavy
scene with and without the switch, including GPU command-thread time.

`EDEN_NX2_TEXTURE_MEMO_TWO=1` enables a second sampled-texture memo for pipelines with up to
32 sampled textures. Set `EDEN_NX2_TEXTURE_MEMO_STATS=1` in both timed runs to report full
resolves, hits, and hits that needed the second lookup. The TAS report counts complete texture
memo windows inside the route. Use `EDEN_NX2_TEXTURE_MEMO_VERIFY=1` on a separate correctness
run to compare views, samplers and prepared images against a full resolve.

`EDEN_NX2_OOT_NATIVE_ZSTD=1` enables host decompression for the supported Ocarina of Time
build on x86-64. Use the `Ocarina zstd` log lines to see how many bytes and streamed blocks
used the host decoder, and compare the same route with the switch off and on.

`EDEN_NX2_DELTA_SAME_HANDLES=1` allows Vulkan to reuse a draw after a write to a sampled-texture
handle if its handle and descriptor bytes still match the previous resolve. The report includes
the number of reuses per second. `EDEN_NX2_FAST_DRAW_VERIFY=1` checks the reused bindings
against the full resolve in a separate correctness run.

`-KeepFullUniforms` reuses streamed graphics uniforms when a full draw follows register
changes and the bound ranges remain unwritten. Compare the same TAS route with the switch on
and off. `Uniform keeps` logs the number of bindings kept on full draws. Set
`EDEN_NX2_UBO_KEEP_VERIFY=1` for a separate correctness run that compares the reused bytes
against guest memory and takes the normal binding path. The hidden
`[.full_uniform_keep_bench]` test measures the binding path without a Vulkan device.

`EDEN_NX2_COMPUTE_UBOREUSE=1` reuses Vulkan compute uniform allocations when the bytes match
within a command buffer. GPU list patches get a fresh allocation. The `Compute uniforms` log
line gives the number of allocations saved per 600 frames. The offline
`[.compute_uniform_benchmark]` test compares allocation counts and CPU cost with a mock stream.

## Measuring command dispatch offline

Run the hidden Catch2 test `[maxwell_fast_method_bench]` in a Release test build. It reports
nanoseconds per method for direct Maxwell dispatch and for the same prequeued commands replayed
through the GPU split dispatcher. Compare repeated runs on the same machine. The fixture uses a
null rasterizer and synthetic commands, so it measures CPU dispatch cost without booting a game.

On x86-64, `[jit_ordered_loads]` checks Dynarmic's ordered and exclusive memory instructions.
The hidden `[.jit_ordered_load_benchmark]` times translated LDAR, LDAPR, LDADDAL, CASA and CASL
loops with several guest cores. Its shared-word and separate-word cases distinguish guest
memory contention from contention in the emulator's reservation data.

The hidden `[.jit_unsigned_div_benchmark]` test times a translated add, divide and
multiply-subtract loop that wraps a queue index. Run it with `EDEN_NX2_FAST_UNSIGNED_DIV=0`
and `EDEN_NX2_FAST_UNSIGNED_DIV=1` in separate test processes. Queue capacity 1 exercises the
fallback on every step.

The hidden `[.shader_register_refresh_bench]` test measures shader-cache lookup after a
register-count write, with and without the old dirty flag. Its mixed case also changes the
pixel program address every 16 iterations. Both cases use registered shader records and
run without a Vulkan device.

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
