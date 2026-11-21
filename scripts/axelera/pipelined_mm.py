#!/usr/bin/env python3
"""Hand-written software-pipelined int8 GEMM for Metis (AxelerScript).

Follows the structure of the SDK's _n_outer_pipelined matmul template:
A (M rows) stays resident in L1; the N columns stream through three stages
(load a B tile and its IMC load, compute on the previous tile, store the tile
before that), with double-buffered L1 slots and A.tiles(..., drain=2) so the
stages overlap. Output is float32, as in the template.

Usage (Voyager SDK environment active):
    python pipelined_mm.py M K N [--frames 200] [--out DIR] [--tile TILE]
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
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--out", default="")
ap.add_argument("--tile", type=int, default=0, help="N columns per pipeline step (0 = tiles pick)")
args = ap.parse_args()

M, K, N = args.M, args.K, args.N
assert K % 64 == 0 and N % 64 == 0
KB, NB = K // 64, N // 64
TILE_NB = max(1, args.tile // 64) if args.tile else max(1, NB // 8)
assert NB % TILE_NB == 0, 'N blocks must divide into whole pipeline tiles'
NT = NB // TILE_NB

WORK = Path(args.out or tempfile.mkdtemp(prefix="pmm-"))
WORK.mkdir(parents=True, exist_ok=True)
TRACE = WORK / "host_trace.log"
os.environ["AXE_LZ_PERF_LOG_PATH"] = str(TRACE)
os.environ["AXE_PROFILING_CONFIG"] = "HOST_TEMPLATE"

import numpy as np  # noqa: E402
import axelera.compiler.alto.script as A  # noqa: E402
from axelera.compiler.alto import axol  # noqa: E402
from axelera.compiler.alto.compiler.config.config import AltoConfig  # noqa: E402
from axelera.compiler.alto.compiler.device import BufferScope  # noqa: E402
from axelera.compiler.alto.compiler.driver import compile as alto_compile  # noqa: E402
from axelera.config import HardwareGeneration  # noqa: E402
from axelera.compiler.alto.axol.core.matmul._common import pipeline_slot, pipeline_tile  # noqa: E402
from axelera.runtime import Context  # noqa: E402
from tvm import tir  # noqa: E402

LOAD_ID, LOAD_IMC_ID, COMPUTE_ID, STORE_ID = "pmm_load", "pmm_load_imc", "pmm_compute", "pmm_store"


@A.ir_module
class PipelinedMatMul:
    @A.prim_func
    def main(a: A.Buffer((M, KB), "int8x64", scope=A.DDR),
             b: A.Buffer((K, NB), "int8x64", scope=A.DDR)):
        a_l1 = A.alloc_buffer((M, KB), "int8x64", scope=BufferScope.L1)
        b_l1 = A.alloc_buffer((2, K, NB), "int8x64", scope=BufferScope.L1)
        c_l1 = A.alloc_buffer((2, M, NB), "float32x64", scope=BufferScope.L1)
        imc = A.alloc_imc((KB, NB), dtype="int8x4096", core=None)
        out = A.alloc_buffer((M, NB), "float32x64", scope=A.DDR)
        axol.atomic.copy(a, output=a_l1, id="pmm_load_a")
        A.sync(waits=["pmm_load_a"])

        for s in A.tiles(NT, drain=2):
            count = NT
            full = TILE_NB
            load = pipeline_tile(s * full, full, NB)
            compute = pipeline_tile((s - 1) * full, full, NB)
            store = pipeline_tile((s - 2) * full, full, NB)
            is_load = tir.And(s >= 0, s < count)
            is_compute = tir.And(s >= 1, s < count + 1)
            is_store = tir.And(s >= 2, s < count + 2)
            axol.atomic.copy(b.tile(load, dim=1), output=pipeline_slot(b_l1, s, K, A.Ramp(0, 1, load.lanes)),
                             id=LOAD_ID, where=is_load)
            axol.atomic.load_imc(pipeline_slot(b_l1, s - 1, K, A.Ramp(0, 1, compute.lanes)),
                                 imc.tile(compute, dim=1), id=LOAD_IMC_ID, where=is_compute)
            axol.atomic.matmul(matrix_ifd=a_l1, matrix_imc=imc.tile(compute, dim=1),
                               output=pipeline_slot(c_l1, s - 1, M, A.Ramp(0, 1, compute.lanes)),
                               id=COMPUTE_ID, where=is_compute)
            axol.atomic.copy(pipeline_slot(c_l1, s - 2, M, A.Ramp(0, 1, store.lanes)),
                             output=out.tile(store, dim=1), id=STORE_ID, where=is_store)
            A.waits(COMPUTE_ID)
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


axm = alto_compile(PipelinedMatMul, AltoConfig(output_dir=WORK / "build", clean_output_dir=True,
                                               generation=HardwareGeneration.OMEGA, num_cores=1))
ctx = Context()
m = ctx.load_model(str(axm))
conn = ctx.device_connect(None, num_sub_devices=1)
inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, host_profiling=True)
rng = np.random.default_rng(0)
ins = [rng.integers(-128, 128, (M, KB, 64), dtype=np.int8), rng.integers(-128, 128, (K, NB, 64), dtype=np.int8)]
outs = [np.zeros((M, NB, 64), dtype=np.float32)]
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
print(f"pipelined M={M} K={K} N={N}: kernel {kernel_us:.1f} us, wall {statistics.median(walls):.1f} us, "
      f"{ops / (kernel_us * 1e-6) / 1e12:.3f} TOPS (kernel)", flush=True)
