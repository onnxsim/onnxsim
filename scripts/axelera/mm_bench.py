#!/usr/bin/env python3
"""Custom AxelerScript kernel: int8 matmul + requantize, compiled for Metis and timed on the card.

Usage (Voyager SDK environment active):
    python mm_bench.py M K N [--frames 300] [--out DIR]

The kernel is written with the axol operator library: load_imc, matmul, mul, dpu_cast.
Each call does 2*M*K*N operations. Kernel time comes from the runtime host trace.
"""

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("M", type=int)
ap.add_argument("K", type=int)
ap.add_argument("N", type=int)
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--out", default="")
ap.add_argument("--cores", type=int, default=1, help="AI cores one kernel is compiled for and run on")
args = ap.parse_args()

M, K, N = args.M, args.K, args.N
assert K % 64 == 0 and N % 64 == 0, "K and N must be multiples of 64"
KB, NB = K // 64, N // 64
SCALE = 1.0 / 4096.0

WORK = Path(args.out or tempfile.mkdtemp(prefix="mm-"))
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


@A.ir_module
class MatMul:
    @A.prim_func
    def main(x: A.Buffer((M, KB), "int8x64", scope=A.DDR),
             w: A.Buffer((K, NB), "int8x64", scope=A.DDR)):
        w_imc = axol.atomic.load_imc(w)
        acc = axol.atomic.matmul(x, w_imc)
        scaled = axol.atomic.mul(acc, SCALE)
        out = A.alloc_buffer((M, NB), "int8x64", scope=A.DDR)
        axol.atomic.dpu_cast(scaled, out)
        A.output(out)


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


axm = alto_compile(MatMul, AltoConfig(output_dir=WORK / "build", clean_output_dir=True,
                                      generation=HardwareGeneration.OMEGA, num_cores=args.cores))
ctx = Context()
m = ctx.load_model(str(axm))
conn = ctx.device_connect(None, num_sub_devices=1)
inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=args.cores, host_profiling=True)
rng = np.random.default_rng(0)
x = [rng.integers(-128, 128, (M, KB, 64), dtype=np.int8)]
w = [rng.integers(-128, 128, (K, NB, 64), dtype=np.int8)]
ins = x + w
outs = [np.zeros((M, NB, 64), dtype=np.int8)]
for _ in range(10):
    inst.run(ins, outs)
_, offset = read_trace(TRACE.stat().st_size if TRACE.exists() else 0)
walls = []
for _ in range(args.frames):
    t0 = time.perf_counter()
    inst.run(ins, outs)
    walls.append((time.perf_counter() - t0) * 1e6)
krn, _ = read_trace(offset)
kernel_us = statistics.median(k / 1000 for k in krn)
ops = 2 * M * K * N
print(f"M={M} K={K} N={N}: ops={ops/1e6:.2f} M, kernel {kernel_us:.1f} us, wall {statistics.median(walls):.1f} us, "
      f"{ops / (kernel_us * 1e-6) / 1e12:.3f} TOPS (kernel), {ops / (statistics.median(walls) * 1e-6) / 1e12:.3f} TOPS (wall)",
      flush=True)
