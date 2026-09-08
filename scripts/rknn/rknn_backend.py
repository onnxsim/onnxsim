#!/usr/bin/env python3
"""Thin wrapper around Rockchip's ``rknn-toolkit2`` ONNX -> RKNN converter.

Unlike the OpenVINO/QNN checks (``scripts/intel``, ``scripts/qualcomm``),
RKNN is **not** an ONNX Runtime execution provider -- there is no
``RknnExecutionProvider`` to register. `rknn-toolkit2` is Rockchip's own
standalone Python package (``pip install rknn-toolkit2``, real x86-64 Linux
wheels for cp310/cp311/cp312 on PyPI) built around ``rknn.api.RKNN``:
``load_onnx()`` parses the graph, ``build()`` quantizes/compiles it into
Rockchip's internal IR, and ``init_runtime(target=None)`` switches on a
**PC simulator** that runs that IR on the host CPU with no RK35xx/RV1106
device attached -- Rockchip's documented no-hardware development path (real
on-device inference is a separate step via ``rknn-toolkit-lite2`` on the
board itself, or ``init_runtime(target="rk3588", ...)`` over ADB/NPU-transfer
to a *connected* device, neither exercised here).

So, like the QNN/OpenVINO checks, this harness runs the **real** Rockchip
converter -- not a static op-list heuristic like ``scripts/axera`` (Pulsar2
has no pip package at all) -- but it only validates *PC-simulator* fidelity;
see this module's docstring below and ``README.md`` for what that does and
does not cover.

## A real, verified compatibility bug: ``onnx.mapping``

`rknn-toolkit2` 2.3.2 (the latest release on PyPI as of this writing) declares
``onnx>=1.16.1`` as a dependency, but its own C-extension code
(``rknn/api/base_utils.py``'s ``to_np_type``/``to_tensor_type``, called from
``ir_graph.py`` and ``ir_utils.py``) still does ``import onnx.mapping`` and
reads ``onnx.mapping.TENSOR_TYPE_TO_NP_TYPE`` /
``onnx.mapping.NP_TYPE_TO_TENSOR_TYPE``. That submodule was removed from the
``onnx`` pip package well before 1.22 (the version this was verified
against, installed alongside a stock ``pip install rknn-toolkit2``) in favor
of ``onnx.helper.tensor_dtype_to_np_dtype``/``np_dtype_to_tensor_dtype``.
Confirmed by direct reproduction: a plain ``rknn.load_onnx()`` call raises
``AttributeError: module 'onnx' has no attribute 'mapping'`` from inside
``IRGraph.rebuild`` before ever reaching onnxsim's own code, and even after
statically working around that, ``rknn.build()`` fails the same way one call
later (``NP_TYPE_TO_TENSOR_TYPE`` is missing too) -- so this is not
reachable through any public onnxsim/rknn-toolkit2 parameter, only around it.

:func:`_ensure_onnx_mapping_shim` patches in a tiny ``onnx.mapping`` module
(built from the still-current ``onnx.helper`` dtype tables) before
``rknn.api`` is imported, so this harness -- and any pipeline that loads
onnxsim's output straight into `rknn-toolkit2` on a modern `onnx` install --
does not need an old, pinned ``onnx`` version just to call ``load_onnx()``.

## A real, verified layout quirk: NCHW vs NHWC at ``inference()``

`rknn.api.RKNN.inference()` defaults ``data_format`` to ``"nhwc"`` regardless
of the ONNX graph's own input layout, and raises if a 4-D ndarray's last
dimension doesn't look like a channel count (``ValueError: The input(ndarray)
shape (1, 3, 16, 16) is wrong, expect 'nhwc' like (1, 16, 16, 3)!`` for an
ordinary NCHW ONNX input). Every onnx-native CV model onnxsim ever sees is
NCHW, so :func:`run` always passes ``data_format="nchw"`` explicitly for 4-D
inputs -- silently getting this wrong would not fail loudly elsewhere, it
would just feed the model transposed data.

## Fidelity tier: PC simulator only

``init_runtime(target=None)`` is documented by Rockchip as a *functional*
simulator (validates that the graph converts and runs, and gives an
approximate numeric result) -- not a bit-exact NPU model. Verified directly:
running a small Conv+Bias+Relu graph through ``build(do_quantization=False)``
+ the PC simulator and comparing against the ONNX Runtime CPU reference gives
a small but nonzero difference (~4e-3 max-abs on values of order 1-8), even
with quantization disabled. That is consistent with Rockchip's own docs (the
simulator's float compute path does not claim bit-exact CPU-reference
parity) and is why this harness compares **original-vs-simplified through
the same RKNN simulator build** (so that fixed simulator/backend numeric
slack cancels out) rather than asserting RKNN output matches the ONNX
Runtime reference tightly -- exactly the pattern
``scripts/qualcomm/qnn_backend.py`` already uses for HTP x86 emulation. No
real RK35xx/RV1106 device was used or claimed anywhere in this file.

This module degrades gracefully: whenever ``rknn-toolkit2`` cannot be
imported, ``RKNN_AVAILABLE`` is False and :func:`unavailable_reason` explains
why, so callers can skip rather than error.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Dict, List

import numpy as np
import onnx

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# See scripts/amd/migraphx_backend.py's comment on why scripts/ is only kept
# on sys.path for the duration of this import (bare `scripts/rfdetr`-style
# namespace-package shadowing otherwise).
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.ep_numerics import compare, random_feeds  # noqa: E402,F401
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)

# The target platform used for every build in this harness. Purely a
# compile-time parameter for the simulator build below -- it is never used to
# select or contact a real device (``init_runtime(target=None)`` always runs
# the host-CPU simulator regardless of this value). rk3588 is the flagship
# SoC RKNN Model Zoo (the downstream project already using onnxsim, see the
# top-level README) targets most of its examples at.
TARGET_PLATFORM = os.environ.get("RKNN_TARGET_PLATFORM", "rk3588")

RKNN_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None


def _ensure_onnx_mapping_shim() -> None:
    """Patch in ``onnx.mapping`` for rknn-toolkit2 2.3.2 on a modern `onnx`
    install that no longer ships it -- see this module's docstring."""
    if hasattr(onnx, "mapping"):
        return
    mapping_mod = types.ModuleType("onnx.mapping")
    tensor_to_np = {
        dtype: onnx.helper.tensor_dtype_to_np_dtype(dtype)
        for dtype in onnx.TensorProto.DataType.values()
        if dtype != onnx.TensorProto.UNDEFINED
    }
    mapping_mod.TENSOR_TYPE_TO_NP_TYPE = tensor_to_np
    mapping_mod.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in tensor_to_np.items()}
    onnx.mapping = mapping_mod
    sys.modules["onnx.mapping"] = mapping_mod


try:
    _ensure_onnx_mapping_shim()
    from rknn.api import RKNN  # noqa: E402

    RKNN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without the SDK
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> str:
    return _UNAVAILABLE_REASON or "unknown"


def _is_image_like(model: onnx.ModelProto) -> bool:
    """True when the (single) graph input is rank-4 -- the NCHW/NHWC ambiguity
    :func:`run` needs to resolve only applies to that shape."""
    inp = model.graph.input[0]
    return len(inp.type.tensor_type.shape.dim) == 4


def run(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Load, build (float, no quantization) and run ``model`` through the RKNN
    PC simulator. Raises on any failure (unsupported op, build error, ...) --
    callers decide what a raised exception means for their status."""
    if not RKNN_AVAILABLE:
        raise RuntimeError(unavailable_reason())

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        onnx.save(model, onnx_path)

        rknn = RKNN(verbose=False)
        try:
            ret = rknn.config(target_platform=TARGET_PLATFORM)
            if ret != 0:
                raise RuntimeError(f"rknn.config failed (ret={ret})")
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                raise RuntimeError(f"rknn.load_onnx failed (ret={ret})")
            # do_quantization=False: this harness checks graph-compile and
            # float-numerics fidelity, not INT8 quantization accuracy (a
            # separate, dataset-dependent concern already covered for other
            # backends by onnxsim's own calibration/quantization tests).
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"rknn.build failed (ret={ret})")
            ret = rknn.init_runtime(target=None)  # None => PC simulator, no device
            if ret != 0:
                raise RuntimeError(f"rknn.init_runtime failed (ret={ret})")

            inputs = [feeds[inp.name] for inp in model.graph.input if inp.name in feeds]
            kwargs = {"data_format": "nchw"} if _is_image_like(model) else {}
            outputs = rknn.inference(inputs=inputs, **kwargs)
            if outputs is None:
                raise RuntimeError("rknn.inference returned None")
            return list(outputs)
        finally:
            rknn.release()


def run_with_cpu(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """ONNX Runtime CPU reference, for informational comparison only (the RKNN
    PC simulator is not expected to match this tightly -- see module docstring)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in sess.get_outputs()]
    return sess.run(output_names, feeds)
