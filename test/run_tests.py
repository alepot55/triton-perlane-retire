"""Run test/nv_per_lane_loop_retirement.mlir against an installed Triton wheel.

The lit test needs `triton-opt`, which the wheel does not ship. This runner
does the same two checks with what the wheel has:

  * the printed module of every `// -----` chunk goes through the `FileCheck`
    binary bundled with Triton, against the CHECK lines of the test file;
  * the remarks the pass emits must be exactly the `expected-remark @below`
    ones (the `-verify-diagnostics` half of the RUN line).

Usage:
  python test/run_tests.py <build>/libPerLaneLoopRetirement.so [test.mlir]

CPU only; no GPU and no Triton source tree.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TEST = HERE / "nv_per_lane_loop_retirement.mlir"
SPLIT = re.compile(r"^// -----[ \t]*$", re.M)
EXPECTED = re.compile(r"expected-remark\s*@below\s*\{\{(.*)\}\}")
ACTUAL = re.compile(r"^[^\n]*?:(\d+):\d+: remark: (.*)$", re.M)


def worker(lib: str, chunk: str) -> int:
    """Parse one chunk, run the pass, print the module (child process)."""
    from triton._C.libtriton import ir, nvidia, passes

    passes.plugin.extend_with(lib)
    add_pass = getattr(passes.plugin, "add_nv_per_lane_loop_retirement")
    ctx = ir.context()
    ir.load_dialects(ctx)
    nvidia.load_dialects(ctx)
    mod = ir.parse_mlir_module(chunk, ctx)
    pm = ir.pass_manager(ctx)
    add_pass(pm)
    pm.run(mod, "nv-per-lane-loop-retirement")
    print(mod.str() if hasattr(mod, "str") else str(mod))
    return 0


def expected_remarks(chunk_text: str) -> set[tuple[int, str]]:
    out = set()
    for i, line in enumerate(chunk_text.splitlines(), start=1):
        m = EXPECTED.search(line)
        if m:
            out.add((i + 1, m.group(1)))
    return out


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "--worker":
        return worker(argv[2], argv[3])
    if len(argv) < 2:
        print(__doc__)
        return 2
    lib = str(pathlib.Path(argv[1]).resolve())
    test = pathlib.Path(argv[2]) if len(argv) > 2 else DEFAULT_TEST
    import triton  # only to find the bundled FileCheck

    filecheck = pathlib.Path(triton.__file__).parent / "FileCheck"
    chunks = SPLIT.split(test.read_text())
    failures = 0
    printed = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, text in enumerate(chunks):
            path = pathlib.Path(tmp) / f"chunk{n}.mlir"
            path.write_text(text)
            env = dict(os.environ, MLIR_ENABLE_DIAGNOSTICS="remarks")
            r = subprocess.run([sys.executable, "-I", __file__, "--worker", lib, str(path)],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0:
                print(f"chunk {n}: pass failed\n{r.stderr}")
                failures += 1
                continue
            printed.append(r.stdout)
            got = {(int(m.group(1)), m.group(2)) for m in ACTUAL.finditer(r.stderr)}
            want = expected_remarks(text)
            if got != want:
                failures += 1
                print(f"chunk {n}: remarks differ\n  expected: {sorted(want)}\n  got:      {sorted(got)}")
        out = pathlib.Path(tmp) / "out.mlir"
        out.write_text("\n// -----\n".join(printed))
        fc = subprocess.run([str(filecheck), str(test), "--input-file", str(out),
                             "--dump-input-context=50"], capture_output=True, text=True)
        if fc.returncode != 0:
            failures += 1
            print("FileCheck failed\n" + fc.stdout + fc.stderr)
    print(f"{len(chunks)} chunks, {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
