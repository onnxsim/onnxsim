#!/usr/bin/env python3
"""Int8 GEMM through the SDK's core matmul with the software pipeline forced on (M > 1).

The SDK chooses its pipelined schedule only for M == 1 by default. This script calls the
core matmul directly with pipeline=True, n_outer=False, tiled=True, so the M-row GEMM uses
the same three-stage pipeline template (_n_outer_pipelined) with A resident in L1 and the
N columns streamed.

Usage (Voyager SDK environment active):
    python core_pipelined_mm.py M K N [--frames 200] [--out DIR]
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
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--out", default="")
ap.add_argument("--pipeline", type=int, default=1, help="1 = force the pipelined schedule")
ap.add_argument("--bt", type=int, default=0, help="1 = B given transposed, layout (N blocks, K)")
args = ap.parse_args()

M, K, N = args.M, args.K, args.N
assert K % 64 == 0 and N % 64 == 0
KB, NB = K // 64, N // 64

WORK = Path(args.out or tempfile.mkdtemp(prefix="cpmm-"))
WORK.mkdir(parents=True, exist_ok=True)
TRACE = WORK / "host_trace.log"
os.environ["AXE_LZ_PERF_LOG_PATH"] = str(TRACE)
os.environ["AXE_PROFILING_CONFIG"] = "HOST_TEMPLATE"

import numpy as np  # noqa: E402
import axelera.compiler.alto.script as A  # noqa: E402
from axelera.compiler.alto import axol  # noqa: E402
from axelera.compiler.alto.axol.core.matmul.matmul import _matmul as core_matmul  # noqa: E402
from axelera.compiler.alto.compiler.config.config import AltoConfig  # noqa: E402
from axelera.compiler.alto.compiler.device import BufferScope  # noqa: E402
from axelera.compiler.alto.compiler.driver import compile as alto_compile  # noqa: E402
from axelera.config import HardwareGeneration  # noqa: E402
from axelera.runtime import Context  # noqa: E402

PIPE = args.pipeline == 1
BT = args.bt == 1
PIPE_ARG = None if not PIPE else True


@A.ir_module
class CoreMM:
    @A.prim_func
    def main(a: A.Buffer((M, KB), "int8x64", scope=A.DDR),
             b: A.Buffer((NB, K) if BT else (K, NB), "int8x64", scope=A.DDR)):
        a_l1 = A.alloc_buffer((M, KB), "int8x64", scope=BufferScope.L1)
        axol.atomic.copy(a, output=a_l1)
        c = A.alloc_buffer((M, NB), "float32x64", scope=A.DDR)
        core_matmul(a_l1, b, c, pipeline=PIPE, n_outer=False, tiled=True, is_b_transposed=BT)
        A.output(c)


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


axm = alto_compile(CoreMM, AltoConfig(output_dir=WORK / "build", clean_output_dir=True,
                                      generation=HardwareGeneration.OMEGA, num_cores=1))
ctx = Context()
m = ctx.load_model(str(axm))
conn = ctx.device_connect(None, num_sub_devices=1)
inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, host_profiling=True)
rng = np.random.default_rng(0)
ins = [rng.integers(-128, 128, (M, KB, 64), dtype=np.int8),
       rng.integers(-128, 128, ((NB, K, 64) if BT else (K, NB, 64)), dtype=np.int8)]
outs = [np.zeros(t.shape, t.dtype) for t in m.outputs()]
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
print(f"core matmul pipeline={PIPE} bt={BT} M={M} K={K} N={N}: kernel {kernel_us:.1f} us, wall {statistics.median(walls):.1f} us, "
      f"{ops / (kernel_us * 1e-6) / 1e12:.3f} TOPS (kernel)", flush=True)
