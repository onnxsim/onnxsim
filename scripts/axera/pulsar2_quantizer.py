#!/usr/bin/env python3
"""A local, Docker-free approximation of Pulsar2's INT8 PTQ, via onnxsim's own quantizer.

Pulsar2's own quantizer is proprietary: inspecting the real
`output/*/quant/quant_axmodel.onnx` that `pulsar2 build` writes (from the
real `resnet18d_Opset18` conversion -- see `pulsar2_ops.py`'s docstring)
shows it rewrites the graph into its own custom ops in the plain default
domain (`AxQuantizedConv`, `AxQuantizedAdd`, `AxQuantizeLinear`, ...) -- not
standard ONNX `QuantizeLinear`/`DequantizeLinear`, so it cannot be executed
by onnxruntime directly and there is no way to reproduce it bit-for-bit here.

What *is* confirmed from that same real file, read off its
`AxQuantizedConv` node attributes directly:

* activations: **U8** (uint8), a single scale/zero-point per tensor (e.g.
  `input_scales=[0.0187]`, `input_zeropoints=[114]`) -- i.e. per-tensor,
  **asymmetric**.
* weights: **S8** (int8), one scale *per output channel* (e.g.
  `weight_scales` has length 32 for a 32-out-channel conv) -- i.e.
  **per-channel, symmetric** (no zero-point attribute on the weight side).
* `quant_method = 0`, matching the `"calibration_method": "MinMax"` in the
  build config (see `pulsar2_ops.py`'s docstring for the real build log).

**onnxsim already has a quantizer with exactly this numeric convention**:
`onnxsim.quantize_static` (`onnxsim/calibration.py`) does calibration-based
QDQ quantization with `method="minmax"` as its default, an "asymmetric
uint8 affine quantization" for activations
(`onnxsim/passes/static_quantize_matmul.h`'s own comment, byte-for-byte the
scheme above), and per-output-channel symmetric INT8 weights. This module
used to hand-roll the same thing via `onnxruntime.quantization.
quantize_static` with matching `QuantType`/`CalibrationMethod` options --
now it just calls onnxsim's own, which is more authoritative (it's this
project's own quantizer, not a reimplementation of its intent in a
different library) and needs no extra dependency beyond what
`pulsar2_simulator.py` already required (onnxruntime, used internally by
`onnxsim.calibrate()` to run the float model and record activation ranges).

Still **not** a reproduction of Pulsar2's internal IR, exact per-op
selection, or rounding/fusion behavior -- `onnxsim.quantize_static` quantizes
Conv/MatMul/"vanilla" Gemm nodes with a constant float weight (see its own
docstring); Pulsar2 quantizes essentially the whole graph. Treat the result
as **compatible in calibration methodology and numeric precision**, not a
faithful emulation. Always confirm on real hardware before trusting it for a
deployment decision -- see `pulsar2_simulator.py` for how this is used
against real-device output.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _local_import import ensure_repo_onnxsim  # noqa: E402

# See ensure_repo_onnxsim's own docstring: without this, `import onnxsim`
# below silently resolves to whatever checkout is editable-installed
# globally (the main checkout), not this file's own, when run from an
# isolated worktree.
ensure_repo_onnxsim()

PULSAR2_QUANTIZER_AVAILABLE = False
_UNAVAILABLE_REASON: Optional[str] = None

try:
    # `onnxsim` itself is imported here too, not just `onnxruntime`: a
    # checkout that hasn't built onnxsim's compiled extension yet (e.g. this
    # repo before `pip install .`/`setup.py build_ext`) fails `import
    # onnxsim` outright, same failure mode as a missing `onnxruntime` --
    # both must be caught here so callers only relying on
    # `PULSAR2_QUANTIZER_AVAILABLE` (and, transitively, `pulsar2_simulator`'s
    # `partition()`/`coverage()`, which are documented to need only `onnx`)
    # degrade gracefully instead of failing to import at all.
    import onnxruntime  # noqa: F401

    import onnxsim

    PULSAR2_QUANTIZER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the host
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> Optional[str]:
    """Why the quantizer is unusable on this host, or None if it is usable."""
    return _UNAVAILABLE_REASON


def quantize_like_pulsar2(
    model: onnx.ModelProto,
    calibration_feeds: Iterable[Dict],
) -> onnx.ModelProto:
    """Return an INT8 QDQ ONNX model calibrated the way Pulsar2 does (MinMax).

    Thin wrapper over `onnxsim.quantize_static(model, calibration_data=...,
    method="minmax")` -- see this module's docstring for why that already
    matches Pulsar2's confirmed numeric convention with no reimplementation
    needed. `calibration_feeds` should be several representative input dicts
    (8-32, matching typical PTQ calibration set sizes -- the real build used
    16). Raises if the quantizer is unavailable; check
    `PULSAR2_QUANTIZER_AVAILABLE` first.
    """
    if not PULSAR2_QUANTIZER_AVAILABLE:
        raise RuntimeError(f"pulsar2_quantizer unavailable: {_UNAVAILABLE_REASON}")

    return onnxsim.quantize_static(
        model, calibration_data=list(calibration_feeds), method="minmax"
    )
