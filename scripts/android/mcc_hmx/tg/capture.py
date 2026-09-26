"""Record every DSP kernel call tinygrad makes under MOCKDSP, so the same kernels can be replayed outside
tinygrad (hexagon-sim with real HMX, or a FastRPC skel on the phone).

A call = the rendered kernel source + its argument buffers (by device address). Each buffer's content is
snapshotted when it first appears in a call: for the graph's real inputs (weights, the input activation,
constants) that is their value; for intermediates it is whatever was there, and the replay overwrites
it in the same order tinygrad did. save() writes <dir>/k<i>.c (one per distinct kernel), calls.txt
(kernel index, then buffer ids), bufs.txt (id size) and b<id>.bin (first-seen contents).
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from tinygrad.helpers import to_mv
from tinygrad.runtime import ops_dsp


@dataclass
class Call:
    src: str
    args: list  # (va_addr, size)
    sink: object = None  # the kernel's AST (for verify.py's per-kernel CPU check)
    globals: tuple = ()  # which of the call's buffers each kernel parameter is


@dataclass
class _State:
    on: bool = False
    run: bool = True  # False: record only (qemu can't run the qfloat / HMX code; replay on hexagon-sim instead)
    calls: list = field(default_factory=list)
    first: dict = field(default_factory=dict)  # va_addr -> (size, bytes)


_st = _State()
# tinygrad lowers / compiles in a worker pool (other processes), so the source is taken from the PROGRAM uop
# when the runtime is created, not from a renderer hook
from tinygrad.engine import realize as _realize  # noqa: E402
from tinygrad.uop.ops import Ops  # noqa: E402

# the kernel's AST before lowering (the PROGRAM's own src[0] is already optimized for the DSP, which the
# CPU backend can't re-lower): to_program's input, keyed by its output (needs PARALLEL=0: in-process)
_orig_to_program = _realize.to_program
_pre_ast: dict = {}


def _cap_to_program(ast, renderer):
    prg = _orig_to_program(ast, renderer)
    _pre_ast[prg.key] = ast
    return prg


_realize.to_program = _cap_to_program
_orig_get_runtime = _realize.get_runtime


def _cap_get_runtime(device, ast, cache=True):
    rt = _orig_get_runtime(device, ast, cache)
    if not hasattr(rt, "_src"):
        rt._src = next((u.arg for u in ast.src if u.op is Ops.SOURCE), "")
        rt._sink, rt._globals = _pre_ast.get(ast.key), tuple(ast.arg.globals)
    return rt


_realize.get_runtime = _cap_get_runtime
_orig_call = ops_dsp.MockDSPProgram.__call__


def _cap_call(self, *bufs, vals=(), **kw):
    if _st.on:
        assert not vals, "symbolic kernel arguments are not replayable"
        for b in bufs:
            if b.va_addr not in _st.first:
                _st.first[b.va_addr] = (b.size, bytes(to_mv(b.va_addr, b.size)))
        _st.calls.append(Call(self._src, [(b.va_addr, b.size) for b in bufs], getattr(self, "_sink", None), getattr(self, "_globals", ())))
        if not _st.run:
            return 0.0
    return _orig_call(self, *bufs, vals=vals, **kw)


ops_dsp.MockDSPProgram.__call__ = _cap_call

# record-only runs never execute a kernel, so skip MOCKDSP's qemu build too (CAPTURE_NO_COMPILE=1: an
# environment variable, because the compile happens in tinygrad's worker processes)
_orig_compile = ops_dsp.DSPCompiler.compile


def _cap_compile(self, src):
    if os.environ.get("CAPTURE_LOG"):
        with open(os.environ["CAPTURE_LOG"], "a") as f:
            f.write(f"{kernel_name(src)} {len(src)} chars\n")
    return b"" if (os.environ.get("CAPTURE_NO_COMPILE") and getattr(self, "mock", False)) else _orig_compile(self, src)


ops_dsp.DSPCompiler.compile = _cap_compile


def start(run=True):
    _st.on, _st.run, _st.calls, _st.first = True, run, [], {}


def stop():
    _st.on = False
    return list(_st.calls)


def addr_of(t):
    """device address of a realized tensor's buffer"""
    buf = t.uop.base.buffer
    return buf._buf.va_addr


def kernel_name(src):
    """the kernel function: the last top-level (non-static) void function defined before the boilerplate"""
    body = src.split("/* DSP boilerplate */")[0]
    names = re.findall(r"^(?:__attribute__\(\([^)]*\)\)\s*)?void\s+(\w+)\(", body, flags=re.M)
    assert names, "no kernel function found"
    return names[-1]


def save(calls, out: Path, out_addr):
    """tinygrad's memory planner reuses one allocation for several logical buffers (of different sizes) and
    buffers can be views into others, so the replay reproduces the memory layout: overlapping address
    intervals merge into regions, a buffer is (region, offset, size), and a region's initial image is its
    intervals' first-seen contents in order of first appearance.
    Writes k<i>.c, calls.txt (kernel, then buffer ids), bufs.txt (id region offset size), regions.txt
    (region size), r<region>.bin, out.txt, vtcm_kb.txt, calls.pkl."""
    out.mkdir(parents=True, exist_ok=True)
    kern, ids = {}, {}
    for c in calls:
        h = hashlib.sha1(c.src.encode()).hexdigest()
        if h not in kern:
            kern[h] = len(kern)
            body = c.src.split("/* DSP boilerplate */")[0]
            name = kernel_name(c.src)
            (out / f"k{kern[h]}.c").write_text(f"/* tinygrad kernel {name} */\n" + re.sub(rf"\b{name}\b", f"tgk{kern[h]}", body))
        for addr, size in c.args:
            if (addr, size) not in ids:
                ids[(addr, size)] = len(ids)
    ivs = sorted(ids)  # (addr, size)
    regions, cur = [], None
    for addr, size in ivs:
        if cur and addr < cur[1]:
            cur[1] = max(cur[1], addr + size)
        else:
            cur = [addr, addr + size]
            regions.append(cur)
    def region_of(addr):
        return next(i for i, (lo, hi) in enumerate(regions) if lo <= addr < hi)
    images = [bytearray(hi - lo) for lo, hi in regions]
    seen = set()
    for c in calls:  # first appearance order
        for addr, size in c.args:
            if (addr, size) in seen:
                continue
            seen.add((addr, size))
            r = region_of(addr)
            first = _st.first.get(addr)
            if first is not None:
                n = min(size, len(first[1]))
                images[r][addr - regions[r][0] : addr - regions[r][0] + n] = first[1][:n]
    for i, img in enumerate(images):
        (out / f"r{i}.bin").write_bytes(bytes(img))
    (out / "regions.txt").write_text("\n".join(f"{i} {hi - lo}" for i, (lo, hi) in enumerate(regions)) + "\n")
    (out / "bufs.txt").write_text(
        "\n".join(f"{i} {region_of(a)} {a - regions[region_of(a)][0]} {s}" for (a, s), i in sorted(ids.items(), key=lambda t: t[1])) + "\n"
    )
    (out / "calls.txt").write_text(
        "\n".join(f"{kern[hashlib.sha1(c.src.encode()).hexdigest()]} " + " ".join(str(ids[(a, s)]) for a, s in c.args) for c in calls) + "\n"
    )
    out_id = next(i for (a, s), i in ids.items() if a == out_addr)
    (out / "out.txt").write_text(f"{out_id}\n")
    from tinygrad.runtime import ops_dsp

    # the VTCM the kernels' tile cache was laid out for (the replay acquires it, 256 KB aligned)
    (out / "vtcm_kb.txt").write_text(f"{getattr(ops_dsp, 'HMX_VTCM_KB', 256)}\n")
    import pickle

    with open(out / "calls.pkl", "wb") as f:  # per call: kernel AST + parameter -> call-buffer order (verify.py)
        pickle.dump([(c.sink, c.globals) for c in calls], f)
