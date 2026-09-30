"""Generate matmul / conv compute kernels with tinygrad for both WebGPU (WGSL) and Vulkan (GLSL -> SPIR-V).

Library used by gen.py (default kernel + one-step Opt candidates) and tune.py (device-timed beam search).
Needs tinygrad 0.14.x on PYTHONPATH and glslangValidator on PATH. No GPU is needed to generate: tinygrad's
lowering runs offline, the "WEBGPU" device string is only a tag threaded through the UOp graph (see
onnxsim/webgpu_tinygrad_codegen.py for the details of this trick).
"""

import os
import re
import subprocess

import numpy as np
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.helpers import Target
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops

from glsl_renderer import GLSLRenderer


def det(shape, seed):
    """Deterministic pseudo-random float32 data in [-0.5, 0.5)."""
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.int64)
    return (
        (((idx * (7919 + 2 * seed) + seed * 104729) % 1000) / 1000.0 - 0.5)
        .astype(np.float32)
        .reshape(shape)
    )


class Problem:
    """A single-kernel problem: named input arrays, a tinygrad graph builder and a numpy reference."""

    def __init__(self, name, inputs, build, ref, flops):
        self.name, self.inputs, self.build, self.ref, self.flops = (
            name,
            inputs,
            build,
            ref,
            flops,
        )


def matmul_problem(M, N, K):
    a, b = det((M, K), 1), det((K, N), 2)
    return Problem(
        f"mm_{M}x{N}x{K}",
        {"a": a, "b": b},
        lambda t: t["a"] @ t["b"],
        lambda: (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32),
        2.0 * M * N * K,
    )


def conv_problem(n, cin, h, w, cout, kh, kw, stride=1, pad=0):
    x, wt = det((n, cin, h, w), 3), det((cout, cin, kh, kw), 4) * 0.2
    ho, wo = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1

    def ref():
        xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))[
            :, :, ::stride, ::stride
        ]
        return np.einsum("nchwij,ocij->nohw", win, wt.astype(np.float64)).astype(
            np.float32
        )

    return Problem(
        f"conv_{n}x{cin}x{h}x{w}_{cout}x{kh}x{kw}_s{stride}p{pad}",
        {"x": x, "w": wt},
        lambda t: t["x"].conv2d(t["w"], stride=stride, padding=pad),
        ref,
        2.0 * n * cout * ho * wo * cin * kh * kw,
    )


def parse_problem(spec):
    kind, _, rest = spec.partition(":")
    v = [int(s) for s in re.split(r"[x,]", rest)]
    if kind == "mm":
        return matmul_problem(*v)
    if kind == "conv":
        return conv_problem(*v)
    raise ValueError(
        f"unknown problem {spec!r}: use mm:MxNxK or conv:N,Cin,H,W,Cout,kh,kw[,stride[,pad]]"
    )


class Lowered:
    """The one scheduled compute kernel of a problem: its AST and the slot -> named-buffer map."""

    def __init__(self, problem):
        named = {k: Tensor(v, device="WEBGPU") for k, v in problem.inputs.items()}
        out = problem.build(named)
        named["out"] = out
        lin = out.schedule_linear()
        calls = [
            u for u in lin.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
        ]
        if len(calls) != 1:
            raise RuntimeError(
                f"{problem.name}: tinygrad scheduled {len(calls)} kernels; this tool handles single-kernel problems"
            )
        base = lambda t: next(u for u in t.uop.toposort() if u.op is Ops.BUFFER)
        by_buf = {base(t): k for k, t in named.items()}
        self.ast = calls[0].src[0]
        self.slots = []  # (slot, name, elements)
        for slot, b in enumerate(calls[0].src[1:]):
            name = by_buf.get(next((u for u in b.toposort() if u.op is Ops.BUFFER), b))
            if name is None:
                raise RuntimeError(
                    f"{problem.name}: kernel buffer {slot} is an intermediate; unsupported"
                )
            self.slots.append((slot, name, int(b.size()[0])))


WGSL, GLSL = WGSLRenderer(Target()), GLSLRenderer(Target())


def render(ast, renderer):
    """(ProgramInfo, source) for a kernel AST."""
    prg = to_program(ast, renderer)
    return prg.arg, next(s for s in prg.src if s.op is Ops.SOURCE).arg


def launch(info):
    g = [int(x) for x in info.global_size] + [1, 1, 1]
    l = [int(x) for x in (info.local_size or [])] + [1, 1, 1]
    return g[:3], l[:3]


def base_scheduler(ast, renderer=GLSL):
    """The unoptimized kernel with its outer loops turned into the workgroup grid (what tinygrad's apply_opts starts from)."""
    s = Scheduler(ast, renderer)
    s.convert_loop_to_global()
    return s


def default_scheduler(ast, renderer=GLSL):
    """The state tinygrad's own hand-coded heuristics reach (what to_program renders when no Opts are given)."""
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations

    return hand_coded_optimizations(base_scheduler(ast, renderer))


def one_step_candidates(state):
    """(applied_opts_repr, Scheduler) for every single further Opt tinygrad's autotuner would try from `state` (BEAM's action set)."""
    return [
        (repr(c.applied_opts), c)
        for c in get_kernel_actions(state, include_0=True).values()
    ]


def compile_spirv(glsl_path, spv_path):
    r = subprocess.run(
        ["glslangValidator", "-V", "-S", "comp", glsl_path, "-o", spv_path],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]


def write_problem(problem, low, outdir):
    """inputs, reference, and the manifest header for the runners."""
    os.makedirs(outdir, exist_ok=True)
    for slot, name, n in low.slots:
        if name == "out":
            continue
        problem.inputs[name].astype(np.float32).tofile(f"{outdir}/in{slot}.bin")
    problem.ref().astype(np.float32).tofile(f"{outdir}/ref.bin")
    lines = [f"problem {problem.name} flops {problem.flops:.0f}"]
    for slot, name, n in low.slots:
        lines.append(f"buf {slot} {n} {'out' if name == 'out' else 'in'}")
    return lines


def write_variant(outdir, vname, ast, opts=""):
    """Render ast to WGSL and GLSL, compile SPIR-V; returns a manifest line or None if any step fails."""
    try:
        info_w, wgsl = render(ast, WGSL)
        info_g, glsl = render(ast, GLSL)
    except Exception as e:  # some Opt combos are not renderable by one of the backends
        return None, f"render failed: {type(e).__name__}: {str(e)[:120]}"
    gw, lw = launch(info_w)
    gg, lg = launch(info_g)
    if (gw, lw) != (gg, lg):
        return None, f"WGSL/GLSL launch mismatch {gw},{lw} vs {gg},{lg}"
    open(f"{outdir}/{vname}.wgsl", "w").write(wgsl)
    open(f"{outdir}/{vname}.comp", "w").write(glsl)
    ok, msg = compile_spirv(f"{outdir}/{vname}.comp", f"{outdir}/{vname}.spv")
    if not ok:
        return None, f"glslang: {msg}"
    entry = re.sub(r"\x1b\[[0-9;]*m", "", info_w.function_name)
    return (
        f"variant {vname} {entry} {gw[0]} {gw[1]} {gw[2]} {lw[0]} {lw[1]} {lw[2]} | {opts}",
        "",
    )
