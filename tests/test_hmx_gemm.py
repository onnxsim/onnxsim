"""Phone-free check of the HMX GEMM core (``scripts/android/hmx_gemm/hmx_gemm.h``) on hexagon-sim.

``hexagon-sim -mv69 --mhmx 1`` models the Hexagon matrix unit; ``sim/gemm_sim.c`` runs
``hmx_gemm_f16`` (HVX pack/unpack, chunked K, 256 KB VTCM windows) against a double reference.
Runs only when ``HEXAGON_TOOLS`` points at a Hexagon toolchain with hexagon-sim.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ANDROID = Path(__file__).resolve().parents[1] / "scripts" / "android"
GEMM = ANDROID / "hmx_gemm"


def _harness():
    # loaded under a unique module name so no generic name leaks into sys.modules / sys.path
    spec = importlib.util.spec_from_file_location(
        "_hmx_gemm_sim_harness", ANDROID / "hexagon_sim_harness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_sim(tmp_path, src, args):
    harness = _harness()
    tools = harness.tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    elf = tmp_path / (src.stem + ".elf")
    subprocess.run(
        [
            str(tools / "bin" / "hexagon-clang"),
            "-mv69",
            "-mhmx",
            "-mhvx",
            "-O2",
            str(src),
            "-o",
            str(elf),
            "-lm",
        ],
        check=True,
    )
    env = dict(os.environ)
    harness._ensure_ncurses5(env, tools / "bin" / "hexagon-sim", tmp_path)
    out = subprocess.run(
        [
            str(tools / "bin" / "hexagon-sim"),
            "-mv69",
            "--mhmx",
            "1",
            str(elf),
            "--",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    return out.stdout + out.stderr


@pytest.mark.parametrize("shape", ["32 64 64 0", "45 576 128 1", "45 1056 128 1"])
def test_hmx_gemm_f16_on_hexagon_sim(tmp_path, shape):
    out = _run_sim(tmp_path, GEMM / "sim" / "gemm_sim.c", shape.split())
    assert "rc 0" in out and " 0 beyond fp16 rounding" in out, out[-2000:]


@pytest.mark.parametrize("ktiles", ["1", "9"])
def test_hmx_block_bit_exact_on_hexagon_sim(tmp_path, ktiles):
    # fp16 (rne of the exact sum + bias), int8 -> u16 and int8 -> u8 stores, bit-exact; 9 K-tiles also
    # exercise the split of an int8 K into two load pairs (8 + 1)
    out = _run_sim(tmp_path, GEMM / "sim" / "block_ref.c", [ktiles])
    assert "\nPASS" in out, out[-2000:]


@pytest.mark.parametrize("shape", ["64 128 128", "130 192 64"])
def test_hmx_gemm_u8_cm_on_hexagon_sim(tmp_path, shape):
    # int8 "cm" path (64 rows x 32 K x 64 columns per instruction, what QNN's int8 convs use): power-of-two
    # scales, exact vs the reference; 130x192x64 takes the scalar pack/unpack paths and a partial row block
    out = _run_sim(tmp_path, GEMM / "sim" / "gemm_u8_sim.c", [*shape.split(), "1"])
    assert " 0 mismatches" in out and "PASS" in out, out[-2000:]


def test_hmx_layers_u8_on_hexagon_sim(tmp_path):
    # three chained int8 layers with activations kept in crouton form (an output tile is the next layer's
    # activation crouton), exact vs the reference
    out = _run_sim(tmp_path, GEMM / "sim" / "layers_u8_sim.c", ["100", "128", "3"])
    assert " 0 mismatches" in out and "PASS" in out, out[-2000:]


def test_hmx_qconv_qdq_exact_on_hexagon_sim(tmp_path):
    # QDQ 1x1 conv (uint8 zp 128, per-channel int8 weights, int32 bias) vs ORT CPU's own output: QC_EXACT must be
    # bit-exact; QC_FAST (HMX-native requant) must stay in the off-by-one class (QNN's HTP: ~7% on such layers)
    pytest.importorskip("onnxruntime")
    tools = _harness().tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    import sys

    layer, case = tmp_path / "layer", tmp_path / "case"
    for script, args in (
        ("qdq_layer.py", [layer, 64, 64, 8, 16, 1, 1, 0, 7]),
        ("export_case.py", [layer, case]),
    ):
        subprocess.run(
            [sys.executable, str(GEMM / "qnn_parity" / script), *map(str, args)],
            check=True,
        )
    out = _run_sim(tmp_path, GEMM / "sim" / "qconv_sim.c", [str(case)])
    exact = next(line for line in out.splitlines() if line.startswith("exact"))
    fast = next(line for line in out.splitlines() if line.startswith("fast"))
    assert " 0 mismatches" in exact and "PASS" in exact, out[-2000:]
    assert int(fast.split(":")[1].split()[0]) < 0.1 * 128 * 64, fast


@pytest.mark.parametrize(
    "sim,shape",
    [
        (
            "qconv3_sim.c",
            [64, 64, 7, 13, 3, 1, 1, 12],
        ),  # stride 1, odd sizes, fused Relu
        (
            "qconv3_sim.c",
            [64, 64, 9, 11, 3, 2, 0, 14],
        ),  # stride 2 (phase split), odd sizes
        (
            "qconv3_stitch_sim.c",
            [64, 64, 8, 12, 3, 1, 0, 11],
        ),  # every offset window through a side crouton
    ],
)
def test_hmx_qconv3x3_qdq_exact_on_hexagon_sim(tmp_path, sim, shape):
    # 3x3 QDQ conv (pad 1) via :single row-offset windows + one-pixel-shifted copies, vs ORT CPU's output
    pytest.importorskip("onnxruntime")
    if _harness().tools_dir() is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    import sys

    layer, case = tmp_path / "layer", tmp_path / "case"
    for script, args in (
        ("qdq_layer.py", [layer, *shape]),
        ("export_case.py", [layer, case]),
    ):
        subprocess.run(
            [sys.executable, str(GEMM / "qnn_parity" / script), *map(str, args)],
            check=True,
        )
    out = _run_sim(tmp_path, GEMM / "sim" / sim, [str(case)])
    exact = next(line for line in out.splitlines() if line.startswith("exact"))
    assert " 0 mismatches" in exact and "PASS" in exact, out[-2000:]


def test_hmx_graph_runner_bit_exact_on_hexagon_sim(tmp_path):
    # the whole graph runner (runner/: qdq_graph.py lowering, rn_load.h planning, rn_exec.h execution) on a tiny
    # ResNet-shaped full_qdq graph (7x7 s2 stem on 3 channels, MaxPool, 3x3 convs, residual Adds, 3x3 s2 + 1x1 s2
    # downsample): the QC_EXACT run must match ORT CPU's output bit for bit
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxsim.full_qdq")
    if _harness().tools_dir() is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    import sys

    runner = GEMM / "runner"
    subprocess.run(
        [sys.executable, str(runner / "make_tiny.py"), str(tmp_path)], check=True
    )
    subprocess.run(
        [
            sys.executable,
            str(runner / "qdq_graph.py"),
            str(tmp_path / "model.onnx"),
            str(tmp_path / "prog"),
            str(tmp_path / "input.bin"),
            str(tmp_path / "ref.bin"),
        ],
        check=True,
    )
    out = _run_sim(
        tmp_path,
        GEMM / "sim" / "runner_sim.c",
        [
            str(tmp_path / "prog"),
            str(tmp_path / "input.bin"),
            str(tmp_path / "ref.bin"),
        ],
    )
    exact = next(line for line in out.splitlines() if line.startswith("exact"))
    assert " 0 mismatches" in exact and "PASS" in exact, out[-2000:]
