"""Compatibility shims for ONNX features that older ``onnx`` releases lack.

onnxsim's own pipeline targets a current ``onnx``, but the package still has to
*import* against an older one: downstream integrations pin old versions and
install onnxsim next to them (X2Paddle 1.6.0 needs ``onnx.mapping``, removed in
onnx 1.16, so its regression harness installs ``onnx<1.16``). A module-level
``onnx.TensorProto.UINT4`` turns that into an ``AttributeError`` at
``import onnxsim``, long before any 4-bit code could run.

The tensor element type numbers are fixed by the ONNX spec and never reused, so
the fallbacks below are the real wire values rather than sentinels: a model that
does carry a 4-bit initializer is still recognised by dtype even when the
installed ``onnx`` has no name for it. Only *creating* such tensors needs the
newer onnx, and that fails where it is attempted instead of at import time.
"""

import onnx

__all__ = ["INT4", "UINT4"]

# Added in onnx 1.16 (opset 21).
UINT4 = getattr(onnx.TensorProto, "UINT4", 21)
INT4 = getattr(onnx.TensorProto, "INT4", 22)
