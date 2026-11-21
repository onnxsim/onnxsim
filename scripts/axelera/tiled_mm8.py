#!/usr/bin/env python3
"""Tiled int8 GEMM for Metis with one DDR buffer per row tile (no buffer slicing).

The M rows are split across NT separate input buffers and NT separate output buffers,
so every tile is an ordinary full-size buffer. The weights are copied to L1 and loaded
into IMC once per call. Each tile runs its own matmul, requantizes, and writes its own output.

Usage (Voyager SDK environment active):
    python tiled_mm8.py M K N [--frames 200] [--out DIR]
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
ap.add_argument("--tiles", type=int, default=8)
ap.add_argument("--reload", type=int, default=0, help="1 = reload the weights into IMC for every row tile")
args = ap.parse_args()

M, K, N, NT = args.M, args.K, args.N, args.tiles
RELOAD = args.reload == 1
assert M % NT == 0 and K % 64 == 0 and N % 64 == 0
TILE = M // NT
KB, NB = K // 64, N // 64
SCALE = 1.0 / 4096.0
USE_IDS = os.environ.get("USE_IDS", "1") == "1"
INLINE_WAIT = os.environ.get("INLINE_WAIT", "1") == "1"
USE_MUL = os.environ.get("MUL", "1") == "1"
NO_COPY = os.environ.get("NOCOPY", "0") == "1"  # 1 = skip the per-tile input copy (timing only: L1 contents undefined)  # 0 = skip the per-tile scale (timing only, wrong numerics)

WORK = Path(args.out or tempfile.mkdtemp(prefix="tmm8-"))
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


def _build_module():
    """Build the eight-tile module with the tile count fixed at build time."""
    xs = [f"x{i}" for i in range(NT)]
    os_ = [f"o{i}" for i in range(NT)]
    ns = {}
    params = ", ".join(f"{n}: A.Buffer(({TILE}, {KB}), \"int8x64\", scope=A.DDR)" for n in xs)
    body = [
        "        w_l1 = A.alloc_buffer(({K}, {NB}), \"int8x64\", scope=A.L1)",
        "        axol.atomic.copy(w, output=w_l1)",
        "        w_imc = axol.atomic.load_imc(w_l1)",
    ]
    for i in range(NT):
        body += [
            f"        x{i}_l1 = A.alloc_buffer(({TILE}, {KB}), \"int8x64\", scope=A.L1)",
            (f"        axol.atomic.copy(x{i}, output=x{i}_l1)" if not NO_COPY else f"        x{i}_l1 = x{i}_l1"),
            (f"        w_imc{i} = axol.atomic.load_imc(w_l1)" if RELOAD else f"        w_imc{i} = w_imc"),
            f"        acc{i} = axol.atomic.matmul(matrix_ifd=x{i}_l1, matrix_imc=w_imc{i}, id='mm{i}')",
            (f"        s{i} = axol.atomic.mul(acc{i}, {SCALE!r})" if USE_MUL else f"        s{i} = acc{i}"),
            (f"        A.waits('mm{i}')" if INLINE_WAIT else ""),
            f"        o{i} = A.alloc_buffer(({TILE}, {NB}), \"int8x64\", scope=A.DDR)",
            f"        axol.atomic.dpu_cast(s{i}, o{i})",
        ]
    waits = ", ".join(f"'mm{i}'" for i in range(NT))
    if USE_IDS and not INLINE_WAIT:
        body.append("        A.waits(" + waits + ")")
    body.append("        A.output(" + ", ".join(os_) + ")")
    src = (
        "@A.ir_module\n"
        "class MatMulTiled:\n"
        "    @A.prim_func\n"
        f"    def main({params}, w: A.Buffer(({K}, {NB}), \"int8x64\", scope=A.DDR)):\n"
        + "\n".join(line.replace("{K}", str(K)).replace("{NB}", str(NB)) for line in body)
        + "\n"
    )
    return src


SRC = _build_module()
(WORK / "module_src.py").write_text(SRC)
ns = {"A": A, "axol": axol, "M": M, "K": K, "N": N, "KB": KB, "NB": NB, "TILE": TILE, "SCALE": SCALE}
exec(compile(SRC, str(WORK / "module_src.py"), "exec"), ns)
MatMulTiled = ns["MatMulTiled"]


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


axm = alto_compile(MatMulTiled, AltoConfig(output_dir=WORK / "build", clean_output_dir=True,
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
krn, _ = read_trace(offset)
kernel_us = statistics.median(k / 1000 for k in krn)
ops = 2 * M * K * N
print(f"tiled M={M} K={K} N={N} tiles={NT}: kernel {kernel_us:.1f} us, wall {statistics.median(walls):.1f} us, "
      f"{ops / (kernel_us * 1e-6) / 1e12:.3f} TOPS (kernel)", flush=True)
