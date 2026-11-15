"""Enable the per-lane loop retirement plugin pass in Triton's NVIDIA pipeline.

Usage:

    import perlane_retire
    perlane_retire.enable()   # loads the plugin, installs the pipeline hook

Where the plugin library comes from (first hit wins): the `library` argument
of `enable()`, the `PERLANE_RETIRE_LIB` environment variable, the entries of
`TRITON_PLUGIN_PATHS` that name `libPerLaneLoopRetirement.so`, a copy next to
this file.

How it is loaded depends on the Triton:

* Triton >= 3.9 and main: plugins are registered explicitly with
  `passes.plugin.extend_with(<library>)`, which `enable()` calls. The pass is
  bound as `add_nv_per_lane_loop_retirement`. `TRITON_PLUGIN_PATHS` is no
  longer read by the Python bindings.
* Triton 3.8: the library named in `TRITON_PLUGIN_PATHS` is loaded while
  `triton` is imported (set it before the import) and the pass is bound under
  its bare name. This path is kept from the 3.8-dev build the plugin was
  written against and has not been re-run on a 3.8.0 release.

The pass runs inside `make_llir`, right after the post-conversion
canonicalize/CSE (immediately before `add_warp_specialize_to_llvm`) -- the
first point where the warp-uniform latch (`nvvm.redux.sync` feeding the
loop branch) exists, and before NVVM ops are lowered further.  Insertion is
done by wrapping the backend's `make_llir` and momentarily interposing on
`add_warp_specialize_to_llvm`, so the upstream pipeline is not duplicated
here; if a future Triton renames that pass, `enable()` fails loudly.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

from triton import knobs
from triton._C.libtriton import passes

_ANCHOR = "add_warp_specialize_to_llvm"
_LIB_NAME = "libPerLaneLoopRetirement.so"
_PASS = "nv_per_lane_loop_retirement"
_LIB_ENV = "PERLANE_RETIRE_LIB"
_PATHS_ENV = "TRITON_PLUGIN_PATHS"


def _find_library(library=None) -> "pathlib.Path | None":
    candidates = []
    if library is not None:
        candidates.append(library)
    if os.environ.get(_LIB_ENV):
        candidates.append(os.environ[_LIB_ENV])
    candidates += [p for p in os.environ.get(_PATHS_ENV, "").split(os.pathsep)
                   if p.endswith(_LIB_NAME)]
    candidates.append(pathlib.Path(__file__).resolve().parent / _LIB_NAME)
    for c in candidates:
        if pathlib.Path(c).is_file():
            return pathlib.Path(c).resolve()
    return None


def _pass_adder():
    """The bound `add` function of the pass, whichever naming this Triton uses."""
    return (getattr(passes.plugin, "add_" + _PASS, None)
            or getattr(passes.plugin, _PASS, None))


def register_plugin(library=None) -> pathlib.Path | None:
    """Make the pass visible in `passes.plugin`; returns the library used.

    Idempotent. Raises if the pass is not available afterwards.
    """
    lib = _find_library(library)
    if _pass_adder() is None:
        if hasattr(passes.plugin, "extend_with"):
            if lib is None:
                raise RuntimeError(
                    f"{_LIB_NAME} not found; pass library=..., or set "
                    f"{_LIB_ENV} to its path")
            passes.plugin.extend_with(str(lib))
        if _pass_adder() is None:
            raise RuntimeError(
                f"plugin pass not loaded; this Triton loads plugins from "
                f"{_PATHS_ENV} while it is imported: set it to {_LIB_NAME} "
                "before `import triton`")
    return lib


def _cache_key(lib) -> str:
    # Computed once: the hook is re-evaluated on every kernel launch to build
    # the compilation cache key, so it must be cheap. The library's bytes are
    # part of it, so rebuilding the pass invalidates cached kernels.
    parts = ["perlane-retire-v0.1.0", pathlib.Path(__file__).read_text()]
    if lib is not None:
        parts.append(hashlib.sha256(lib.read_bytes()).hexdigest())
    return ":".join(parts)


def enable(library=None) -> None:
    lib = register_plugin(library)
    add_pass = _pass_adder()
    key = _cache_key(lib)
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    def hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(a is None for a in (stages, options, language, capability)):
            return key, key_hash

        make_llir = self.make_llir

        def make_llir_with_retire(src, metadata, opt, cap):
            from triton._C.libtriton import nvidia
            sub = nvidia.passes.ttnvgpuir
            if not hasattr(sub, _ANCHOR):
                raise RuntimeError(
                    f"pipeline anchor '{_ANCHOR}' not found; this Triton "
                    "version needs the insertion point updated")
            real = getattr(sub, _ANCHOR)

            def patched(pm):
                add_pass(pm)
                real(pm)

            setattr(sub, _ANCHOR, patched)
            try:
                return make_llir(src, metadata, opt, cap)
            finally:
                setattr(sub, _ANCHOR, real)

        stages["llir"] = lambda src, metadata: make_llir_with_retire(
            src, metadata, options, capability)
        return key, key_hash

    knobs.runtime.add_stages_inspection_hook = hook
