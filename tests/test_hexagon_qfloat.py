"""Tests for the Hexagon HVX qfloat TIR pass (``scripts/android/hexagon_qfloat.py``).

The pass turns ordinary vectorized multiply-accumulate chains into chained
``vmpy.qf32/qf16`` + ``vadd`` intrinsics (Hexagon's LLVM backend otherwise converts qfloat <->
IEEE around every fp op). Two layers of tests:

* TIR-level checks (need only Apache TVM **0.17** built with LLVM's Hexagon target): lower a plain
  TE schedule with the pass registered and inspect the intrinsics it emits.
* Numeric checks on the Hexagon instruction-set simulator (also need the Hexagon toolchain's
  ``hexagon-clang``/``hexagon-sim``, located through ``HEXAGON_TOOLS``): compile the same kernels
  to a Hexagon object, link a C harness, run on ``hexagon-sim`` and compare with NumPy. A stock-TVM
  control run validates the harness itself.

The whole module is skipped unless a matching TVM is importable; it runs in the dedicated
``hexagon-qfloat`` CI workflow, which builds TVM v0.17.0 and installs the open-access toolchain.
The TVM 0.17 legacy TE schedule API (``te.create_schedule``) is required, so newer TVM wheels
skip too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")
if not hasattr(tvm.te, "create_schedule"):
    pytest.skip("needs TVM 0.17's TE schedule API", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "android"))

import hexagon_conv_transpose_te as kernels  # noqa: E402
import hexagon_qfloat  # noqa: E402
import hexagon_sim_harness as sim  # noqa: E402
from tvm import te  # noqa: E402

# LLVM must have the Hexagon target, or the qfloat intrinsic ids cannot be resolved (0 = unknown).
if not tvm.target.codegen.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vmpy.qf32.sf.128B"):
    pytest.skip("TVM's LLVM has no Hexagon intrinsics", allow_module_level=True)


def _lower_with_pass(schedule, args):
    """Lowered TIR text with the qfloat pass registered at phase 2."""
    config = {"tir.add_lower_pass": [(2, hexagon_qfloat.qfloat_accumulate_pass())]}
    with tvm.transform.PassContext(config=config):
        return str(tvm.lower(schedule, args))


def _matmul(dtype, accumulate, lanes, rows=4, k=32, columns=64, combiner="sum"):
    x = te.placeholder((rows, k), name="x", dtype=dtype)
    w = te.placeholder((k, columns), name="w", dtype=dtype)
    r = te.reduce_axis((0, k), name="r")
    if combiner == "max":
        acc = te.compute(
            (rows, columns), lambda n, j: te.max(x[n, r] * w[r, j], axis=r), name="acc"
        )
    else:
        acc = te.compute(
            (rows, columns),
            lambda n, j: te.sum(
                x[n, r].astype(accumulate) * w[r, j].astype(accumulate), axis=r
            ),
            name="acc",
        )
    out = te.compute((rows, columns), lambda n, j: acc[n, j].astype(dtype), name="out")
    schedule = te.create_schedule(out.op)
    n, j = schedule[out].op.axis
    jo, ji = schedule[out].split(j, factor=lanes)
    schedule[out].reorder(jo, n, ji)
    schedule[out].vectorize(ji)
    schedule[acc].compute_at(schedule[out], jo)
    an, aj = schedule[acc].op.axis
    (ar,) = schedule[acc].op.reduce_axis
    schedule[acc].reorder(ar, an, aj)
    schedule[acc].unroll(an)
    schedule[acc].vectorize(aj)
    return schedule, [x, w, out]


def _count(text, needle):
    return text.count(needle)


def test_fp32_accumulators_become_qf32_chains():
    text = _lower_with_pass(*_matmul("float32", "float32", 32))
    # One multiply + one add per unrolled row; the tile is converted back where it is read.
    assert _count(text, "vmpy.qf32.sf") == 4
    assert _count(text, "vadd.qf32") == 4
    assert _count(text, "vconv.sf.qf32") >= 1
    assert "vmpy.qf32.hf" not in text


def test_fp16_inputs_with_fp32_accumulation_use_widening_multiply():
    text = _lower_with_pass(*_matmul("float16", "float32", 64))
    assert _count(text, "vmpy.qf32.hf") == 4  # bound once per update, not once per half
    assert _count(text, "vadd.qf32") == 8  # low and high half per accumulator
    assert _count(text, "vconv.sf.qf32") >= 2  # both halves converted on read
    assert "vmpy.qf32.sf" not in text


def test_fp16_accumulators_become_qf16_chains():
    text = _lower_with_pass(*_matmul("float16", "float16", 64))
    assert _count(text, "vmpy.qf16.hf") == 4
    assert _count(text, "vadd.qf16") == 4
    assert _count(text, "vconv.hf.qf16") >= 1


def test_non_accumulate_reductions_are_left_alone():
    text = _lower_with_pass(*_matmul("float32", "float32", 32, combiner="max"))
    assert "llvm.hexagon" not in text


def test_pass_is_disabled_by_default_in_plain_builds():
    schedule, args = _matmul("float32", "float32", 32)
    assert "llvm.hexagon" not in str(tvm.lower(schedule, args))


def test_build_declares_offset_free_aligned_buffers():
    schedule, args = _matmul("float32", "float32", 32)
    with tvm.transform.PassContext(opt_level=3):
        binds = {
            t: tvm.tir.decl_buffer(
                t.shape, t.dtype, name=t.op.name, data_alignment=128, offset_factor=0
            )
            for t in args
        }
        text = str(tvm.lower(schedule, args, binds=binds))
    assert "elem_offset" not in text and "align=128" in text


# ---- numeric checks on the Hexagon simulator ---------------------------------------------------

needs_sim = pytest.mark.skipif(
    sim.tools_dir() is None,
    reason="set HEXAGON_TOOLS to a Hexagon toolchain with hexagon-clang and hexagon-sim",
)


def _reference(shape_info, dtype, seed):
    n, ic, size, _, oc = shape_info
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.1, (n, size, size, ic)).astype(dtype)
    weight = rng.normal(0, 0.05, (2, 2, ic, oc)).astype(dtype)
    bias = rng.normal(0, 0.01, (oc,)).astype(dtype)
    d, w, b = (a.astype("float32") for a in (data, weight, bias))
    expected = np.zeros((n, size * 2, size * 2, oc), dtype="float32")
    for py in range(2):
        for px in range(2):
            expected[:, py::2, px::2, :] = np.einsum("nhwc,co->nhwo", d, w[py, px]) + b
    return [data, weight, bias], expected


@needs_sim
@pytest.mark.parametrize(
    ("mode", "use_pass", "tolerance"),
    [
        ("f32", True, 1e-5),
        ("f32", False, 1e-5),  # control: stock TVM through the same harness
        ("f16w", True, 1e-3),
        ("f16k", True, 2e-3),
    ],
)
def test_conv_transpose_matches_numpy_on_hexagon_simulator(mode, use_pass, tolerance):
    fp16 = mode != "f32"
    vectors, pixel_block = 2, 2
    oc = (64 if fp16 else 32) * vectors
    shape_info = (1, 32, 4, 4, oc)
    target = tvm.target.Target(sim.target())
    module, shape = kernels.conv_transpose_module(
        mode,
        shape_info,
        target,
        vectors,
        pixel_block,
        4,
        8,
        use_pass,
        name="kernel",
        parallel=False,
    )
    inputs, expected = _reference(shape_info, "float16" if fp16 else "float32", seed=7)
    output = sim.run_kernel(
        module, inputs, shape, "float16" if fp16 else "float32", expected, tolerance
    )
    assert "max_abs_err" in output


@needs_sim
@pytest.mark.parametrize(
    ("mode", "min_speedup"), [("f32", 2.0), ("f16w", 3.5), ("f16k", 1.8)]
)
def test_pass_cuts_simulated_kernel_cycles(mode, min_speedup):
    """hexagon-sim is deterministic, so kernel cycle counts are a stable performance guard.

    Measured on a phone the same schedules gain 1.8x (fp32), 5.1x (fp16 widening) and 2.0x
    (chunked qf16) over stock TVM; the simulator shows 2.4x / 4.5x / 2.5x.
    """
    fp16 = mode != "f32"
    vectors, pixel_block = 2, 2
    shape_info = (1, 64, 4, 4, (64 if fp16 else 32) * vectors)
    target = tvm.target.Target(sim.target())
    inputs, expected = _reference(shape_info, "float16" if fp16 else "float32", seed=11)
    cycles = {}
    for use_pass in (False, True):
        module, shape = kernels.conv_transpose_module(
            mode, shape_info, target, vectors, pixel_block, 4, 8, use_pass,
            name="kernel", parallel=False,
        )  # fmt: skip
        output = sim.run_kernel(
            module, inputs, shape, "float16" if fp16 else "float32", expected, 2e-3,
            measure_cycles=True,
        )  # fmt: skip
        cycles[use_pass] = sim.kernel_pcycles(output)
    assert cycles[False] >= min_speedup * cycles[True], cycles
