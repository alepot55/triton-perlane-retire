"""End-to-end check without a GPU: compile the canonical divergent-trip loop to
PTX for sm_89 with and without the plugin and compare the PTX.

Needs only the Triton wheel (ptxas ships in it) and the built plugin:

  PERLANE_RETIRE_LIB=<build>/libPerLaneLoopRetirement.so \
  PYTHONPATH=<this repo>/python python test/test_compile_ptx.py

Checks what the static test cannot: that the pipeline hook finds its anchor in
this Triton's `make_llir`, that the pass fires on the real lowering (not only on
hand-written LLVM dialect), and that the result is accepted by ptxas.
"""

from __future__ import annotations

import os
import sys
import tempfile

# A fresh cache: the baseline and the plugin build must not share entries.
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="perlane-cache-")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import perlane_retire  # noqa: E402

BLOCK = 32
TARGET = GPUTarget("cuda", 89, 32)


@triton.jit
def perlane_while(inp, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    trip = tl.load(inp + i, mask=valid, other=0)
    acc = tl.zeros((BLOCK,), tl.int32)
    j = tl.zeros((BLOCK,), tl.int32)
    while tl.max((j < trip).to(tl.int32)) > 0:
        active = j < trip
        acc = acc + tl.where(active, j, 0)
        j = j + 1
    tl.store(out + i, acc, mask=valid)


def ptx() -> str:
    src = ASTSource(
        fn=perlane_while,
        signature={"inp": "*i32", "out": "*i32", "n": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK},
    )
    return triton.compile(src, target=TARGET).asm["ptx"]


def main() -> int:
    base = ptx()
    perlane_retire.enable()
    patched = ptx()
    n_base = base.count("redux.sync")
    n_patched = patched.count("redux.sync")
    print(f"triton {triton.__version__}: redux.sync baseline={n_base} plugin={n_patched}; "
          f"bar.warp.sync baseline={base.count('bar.warp.sync')} plugin={patched.count('bar.warp.sync')}")
    ok = n_base >= 1 and n_patched == 0 and "bar.warp.sync" in patched
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
