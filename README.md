# triton-perlane-retire

An out-of-tree [Triton](https://github.com/triton-lang/triton) plugin pass
that lets the lanes of a warp **retire independently from divergent
`while` loops**, instead of iterating in lock-step to the warp's maximum
trip count.

Proposed upstream and moved out-of-tree following maintainer guidance:
RFC [triton-lang/triton#10773](https://github.com/triton-lang/triton/issues/10773),
draft PR [triton-lang/triton#10774](https://github.com/triton-lang/triton/pull/10774).

## What it does

Triton lowers a data-dependent loop such as

```python
while tl.max((j < trip).to(tl.int32)) > 0:
    active = j < trip
    acc += tl.where(active, j, 0)
    j += 1
```

to a warp-lock-step loop: the latch warp-reduces the per-lane predicate
(`nvvm.redux.sync max`) and branches on the uniform result, so every lane
iterates until the slowest lane in its warp finishes, and pays a
warp-collective reduction each iteration. On sm_70+ (independent thread
scheduling) that schedule is a codegen choice, not a hardware requirement.

The pass (`nv-per-lane-loop-retirement`, an LLVM-dialect module pass)
redirects such latches to the per-lane predicate — each lane exits as soon
as its own condition fails — deletes the dead cross-lane reduction, and
reconverges the entering lanes (`activemask` captured at the preheader)
with `bar.warp.sync` at the loop exit.

**A static verifier proves observational equivalence before rewriting**
(loops it cannot prove safe are left untouched):

1. body free of NVVM ops (collectives, barriers, nested redux latches),
   calls, atomics, unpredicated `llvm.store`, and inline asm that stores or
   synchronizes — predicated gathers (masked `tl.load`) are safe;
2. single loop exit; unconditionally-entered preheader;
3. every live-out loop-carried value **frozen** on lane-inactive iterations
   (`select(pred, x, old)` / masked-identity updates, followed through the
   struct-typed block-argument projections of the real lowering);
4. the per-lane predicate is **monotone**: once false it cannot re-arm
   under continued lock-step execution. (Live-out freezing alone is *not*
   sufficient — an unmasked induction on a lock-step-inactive lane could
   re-arm a non-monotone predicate; the tests include the counterexample.)

## Measured (RTX 4070, sm_89; every run oracle-checked bit-exact)

| | |
|---|---|
| canonical divergent-trip loop | **2.5–4.2×** end-to-end |
| issued instructions (Nsight) | **39× fewer** (work = `sum(trip)` not `32×warp-max`) |
| SASS | `REDUX.MAX.S32` eliminated, 48→40 static instructions |
| masked-cost law | `t = 50.3 + 1.08·E[warp-max trip]` µs, R²=0.998, holds out-of-sample |
| gather-bound kernels with the same latch (CSR SpMV, MoE top-k) | 1.14–1.25× |

## Build

Against an installed Triton wheel (3.9.0 or later, the release `pip install
triton` gives you; Python 3.11 or later) and the LLVM/MLIR artifact that
release was built with. For Triton 3.9.0 that is `llvm-b010a18d-ubuntu-x64-1`,
sha256 `a8f2966f...e78428`, from
`https://oaitriton.blob.core.windows.net/public/llvm-builds/`
(`ci/download_llvm.py` in [triton-ext](https://github.com/triton-lang/triton-ext)
fetches and checks it). The wheel supplies the headers and `libtriton.so`;
nothing of Triton is compiled.

```bash
pip install triton==3.9.0
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_WHEEL_DIR=$(python -c 'import triton,os;print(os.path.dirname(triton.__file__))') \
  -DTRITON_LLVM_DIR=/path/to/llvm-b010a18d-ubuntu-x64-1
ninja -C build
```

Against a Triton source/build tree compiled with `TRITON_EXT_ENABLED=1`
(Triton main) the old recipe still works: replace `TRITON_WHEEL_DIR` with
`-DTRITON_SRC_DIR=/path/to/triton` (and `-DTRITON_BUILD_DIR` if it is not an
in-source build).

## Use

```python
import perlane_retire          # python/ on PYTHONPATH
perlane_retire.enable("/path/to/build/libPerLaneLoopRetirement.so")
# ... define and launch Triton kernels as usual
```

The library path can also come from `PERLANE_RETIRE_LIB`. On Triton 3.9 and
later `enable()` registers the plugin with `passes.plugin.extend_with`; that
release no longer reads `TRITON_PLUGIN_PATHS`, so setting the variable alone does
nothing there. The compile-cache key includes the plugin library, so rebuilding
it recompiles cached kernels.

`bench/bench_perlane_retire.py` reproduces the headline result end-to-end
(bit-exact oracle + PTX `redux.sync` 1→0); it needs a GPU.

## Tests (no GPU)

```bash
python test/run_tests.py build/libPerLaneLoopRetirement.so        # 9 lit chunks
PERLANE_RETIRE_LIB=build/libPerLaneLoopRetirement.so \
  PYTHONPATH=python python test/test_compile_ptx.py               # PTX for sm_89
```

`run_tests.py` replays `test/nv_per_lane_loop_retirement.mlir` with the
`FileCheck` binary bundled in the wheel and checks the `expected-remark` lines
(the wheel has no `triton-opt`). `test_compile_ptx.py` compiles the canonical
loop for sm_89 with and without the plugin and requires `redux.sync` to go from
1 to 0 and a `bar.warp.sync` to appear; ptxas accepts the result.

## Tested versions

| Triton | LLVM | status |
|---|---|---|
| 3.9.0 (release wheel, 2026-10-27) | `b010a18d` | built and tested as above: 9 lit chunks pass, the PTX check passes. Bench on an RTX 4070 (driver 580.196.07, torch 2.15.0+cu132, GPU otherwise unused), 9 separate runs: bit-exact against the oracle in every run, `redux.sync` 1 to 0; speedup median 2.84x, min 2.19x, max 3.48x, the spread set by the CPU core the run lands on (see below); kernel only 3.70x to 3.71x |
| main @ `81a46fa` (3.8.0-dev) | its own | the build the measurements come from; source-tree recipe |
| 3.8.0 (release wheel) | `5f07f818` | the wheel ships no headers: built with the source-tree recipe against the `v3.8.0` tag (`include/`, `python/src/`), a hand-written `triton/Version.h` and the wheel's own library (`-DTRITON_LIB=<wheel>/_C/libtriton.so`); loaded with `TRITON_PLUGIN_PATHS` set before `import triton`, where the pass binds as `nv_per_lane_loop_retirement`. `test_compile_ptx.py` passes as it is; `run_tests.py` passes (9 chunks) once a shim outside the repo maps `passes.plugin.extend_with`, a 3.9 API, to a no-op. Bench, same card, torch 2.14.0+cu132, 9 runs: bit-exact, `redux.sync` 1 to 0; speedup median 2.26x, min 2.20x, max 2.92x; kernel only 3.70x |

### Reading the bench (15 Nov 2026)

`bench_perlane_retire.py` records a CUDA event, launches the kernel from
Python, records a second event and synchronizes, nine times per build, after
three warm-up launches. For launches 2 to 9, which start on an idle GPU, the
window between the two events holds the host side of Triton's launch as well
as the kernel. On the measuring machine (Intel i7-12700: 8 performance cores,
4 efficiency cores) that host side costs 16 to 18 us more on an efficiency
core, for both builds alike, so the ratio depends on where the process runs.
Same card, both releases, every run bit-exact and without `redux.sync`:

| Triton | runs | baseline us | retire us | speedup, median (min to max) |
|---|---|---|---|---|
| 3.9.0 | 9, unpinned | 4 at 92.2 to 92.9, 5 at 108.5 to 110.6 | 5 at 31.7 to 32.5, 4 at 49.2 to 50.0 | 2.84x (2.19x to 3.48x) |
| 3.9.0 | 5, on a performance core | 93.0 (92.2 to 93.2) | 31.7 (31.7 to 32.4) | 2.91x (2.84x to 2.94x) |
| 3.9.0 | 5, on an efficiency core | 109.5 (108.8 to 118.5) | 49.2 (48.1 to 50.9) | 2.27x (2.19x to 2.36x) |
| 3.9.0 | kernel only, CUDA-graph replay | 76.3 | 20.6 | 3.70x to 3.71x |
| 3.8.0 | 9, unpinned | 4 at 91.9 to 92.2, 5 at 107.5 to 108.5 | 4 at 31.5 to 31.7, 5 at 47.8 to 49.0 | 2.26x (2.20x to 2.92x) |
| 3.8.0 | 5, on a performance core | 92.2 (91.2 to 92.2) | 31.6 (30.7 to 31.7) | 2.91x (2.90x to 2.97x) |
| 3.8.0 | 5, on an efficiency core | 108.6 (107.5 to 109.6) | 48.1 (47.9 to 58.4) | 2.26x (1.87x to 2.28x) |
| 3.8.0 | kernel only, CUDA-graph replay | 76.3 | 20.6 | 3.70x |

Pinned runs alternated, one on each kind of core, five times; on each release
their ranges do not overlap. The unpinned runs fall into the same two states,
and their median only says how many runs landed on each kind of core: 5 of 9
were slow on 3.8.0 and 4 of 9 on 3.9.0 (plus one, the 3.48x, that timed all
nine baseline launches slow and all nine plugin launches fast, as if the
process moved between the builds; the core was not recorded), hence 2.26x
against 2.84x for the same kernel. The 2.91x, 2.83x and 2.16x of the first
measurement on this card sit in or just outside the same two states (93.0/32.0
and 92.3/32.6 fast, 110.6/51.1 slow; no per-launch times were kept).

The GPU side is the same in both states. The first timed baseline launch is
queued behind the warm-ups, so it times the kernel alone: 85.0 to 86.0 us in
all 38 runs, on either core and either release. The GPU was otherwise unused
(P8 and 0 % when each of the three jobs started; `nvidia-smi pmon` shows SM
activity only from the bench's own processes), and the clock, sampled every
200 ms, read P2, 2505 MHz graphics and 10251 MHz memory while bench processes
ran, fast and slow alike.

The kernel-only rows use `triton.testing.do_bench_cudagraph` on the same
kernel, inputs and oracle. The replay keeps the GPU busy, at 2640 to 2730 MHz
for most of it, so its absolute times are at a higher clock than the bench's.
The 2.5x to 4.2x at the top is called end to end and came from a bench of
this form on main; if it used the same launch-inclusive window, it may depend
on the core as well. On this machine that window gives about 2.9x on a
performance core and 2.3x on an efficiency core, and the kernels alone differ
by 3.7x.

The pipeline hook interposes on `add_warp_specialize_to_llvm` inside
`make_llir` (the first point where the warp-uniform latch exists); if a
future Triton renames that anchor, `enable()` fails loudly. The anchor and the
`add_stages_inspection_hook` contract are unchanged in 3.9.0.

## License

MIT.
