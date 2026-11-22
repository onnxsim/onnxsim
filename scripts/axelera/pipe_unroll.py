#!/usr/bin/env python3
"""Double-buffered row-tiled int8 GEMM with a fully unrolled, static pipeline.

Each row tile is an ordinary full-size DDR buffer. Two static L1 slots alternate between
tiles (tile i uses slot i % 2), so no loop variable is used in any index. The input copy for
tile i+1 is issued before the matmul of tile i, so the DMA for the next tile can run while
the current matmul executes. The weights are loaded into IMC once per call.

This avoids the `lanes > 1` error that the loop-based pipeline hits (the loop variable in
an index), at the cost of a larger unrolled program.

Usage (Voyager SDK environment active):
    python pipe_unroll.py M K N [--tiles 8] [--frames 100] [--out DIR]
"""

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("M", type=int)
ap.add_argument("K", type=int)
ap.add_argument("N", type=int)
ap.add_argument("--tiles", type=int, default=8)
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--out", default="")
args = ap.parse_args()

M, K, N, NT = args.M, args.K, args.N, args.tiles
assert M % NT == 0 and K % 64 == 0 and N % 64 == 0
TILE = M // NT
KB, NB = K // 64, N // 64
SCALE = 1.0 / 4096.0
NOWAIT = os.environ.get('NOWAIT', '0') == '1'
NOID = os.environ.get('NOID', '0') == '1'

WORK = Path(args.out or tempfile.mkdtemp(prefix="pipe-"))
WORK.mkdir(parents=True, exist_ok=True)
TRACE = WORK / "host_trace.log"
os.environ["AXE_LZ_PERF_LOG_PATH"] = str(TRACE)
os.environ["AXE_PROFILING_CONFIG"] = "HOST_TEMPLATE"

import numpy as np  # noqa: E402
import axelera.compiler.alto.script as A  # noqa: E402
from axelera.compiler.alto import axol  # noqa: E402
from axelera.compiler.alto.compiler.config.config import AltoConfig  # noqa: E402
from axelera.compiler.alto.compiler.driver import compile as alto_compile  # noqa: E402
from axelera.config import HardwareGeneration  # noqa: E402
from axelera.runtime import Context  # noqa: E402


def build_source():
    params = ", ".join(f'x{i}: A.Buffer(({TILE}, {KB}), "int8x64", scope=A.DDR)' for i in range(NT))
    body = [
        f'        w_l1 = A.alloc_buffer(({K}, {NB}), "int8x64", scope=A.L1)',
        "        axol.atomic.copy(w, output=w_l1)",
        "        w_imc = axol.atomic.load_imc(w_l1)",
        f'        s0 = A.alloc_buffer(({TILE}, {KB}), "int8x64", scope=A.L1)',
        f'        s1 = A.alloc_buffer(({TILE}, {KB}), "int8x64", scope=A.L1)',
        f"        axol.atomic.copy(x0, output=s0{'' if NOID else ', id=' + repr('c0')})",
    ]
    outs = []
    for i in range(NT):
        slot = f"s{i % 2}"
        if i + 1 < NT:
                body.append(f"        axol.atomic.copy(x{i + 1}, output=s{(i + 1) % 2}{'' if NOID else ', id=' + repr('c' + str(i + 1))})")
        body += [
            *([] if NOWAIT else [f"        A.waits('c{i}')"]),
            f"        acc{i} = axol.atomic.matmul(matrix_ifd={slot}, matrix_imc=w_imc, id='mm{i}')",
            f"        sc{i} = axol.atomic.mul(acc{i}, {SCALE!r})",
            f'        o{i} = A.alloc_buffer(({TILE}, {NB}), "int8x64", scope=A.DDR)',
            f"        axol.atomic.dpu_cast(sc{i}, o{i})",
        ]
        outs.append(f"o{i}")
    body.append("        A.output(" + ", ".join(outs) + ")")
    return (
        "@A.ir_module\n"
        "class PipeUnroll:\n"
        "    @A.prim_func\n"
        f"    def main({params}, w: A.Buffer(({K}, {NB}), \"int8x64\", scope=A.DDR)):\n"
        + "\n".join(body) + "\n"
    )


SRC = build_source()
(WORK / "module_src.py").write_text(SRC)
ns = {"A": A, "axol": axol}
exec(compile(SRC, str(WORK / "module_src.py"), "exec"), ns)
PipeUnroll = ns["PipeUnroll"]


def read_trace(offset):
    if not TRACE.exists():
        return [], offset
    rows, consumed = [], 0
    for raw in TRACE.read_bytes()[offset:].splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        consumed += len(raw)
        f = dict(p.split(":", 1) for p in raw.decode().split()[1:] if ":" in p)
        if "krn" in f:
            rows.append(int(f["krn"], 16))
    return rows, offset + consumed


axm = alto_compile(PipeUnroll, AltoConfig(output_dir=WORK / "build", clean_output_dir=True,
                                          generation=HardwareGeneration.OMEGA, num_cores=1))
ctx = Context()
m = ctx.load_model(str(axm))
conn = ctx.device_connect(None, num_sub_devices=1)
inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, host_profiling=True)
rng = np.random.default_rng(0)
xs = [rng.integers(-128, 128, (TILE, KB, 64), dtype=np.int8) for _ in range(NT)]
w = rng.integers(-128, 128, (K, NB, 64), dtype=np.int8)
ins = xs + [w]
outs = [np.zeros((TILE, NB, 64), dtype=np.int8) for _ in range(NT)]
for _ in range(10):
    inst.run(ins, outs)
_, offset = read_trace(TRACE.stat().st_size if TRACE.exists() else 0)
walls = []
for _ in range(args.frames):
    t0 = time.perf_counter()
    inst.run(ins, outs)
    walls.append((time.perf_counter() - t0) * 1e6)
inst.run(ins, outs)
if os.environ.get("SAVE"):
    np.savez(str(WORK / "outputs.npz"), *[o.copy() for o in outs], x=np.stack(xs), w=w)
krn, _ = read_trace(offset)
kernel_us = statistics.median(k / 1000 for k in krn)
ops = 2 * M * K * N
print(f"pipe-unroll M={M} K={K} N={N} tiles={NT}: kernel {kernel_us:.1f} us, wall {statistics.median(walls):.1f} us, "
      f"{ops / (kernel_us * 1e-6) / 1e12:.3f} TOPS (kernel)", flush=True)
