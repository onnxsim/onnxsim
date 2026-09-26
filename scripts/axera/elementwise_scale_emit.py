"""Retarget a compiled standalone AX650 elementwise op to new quantization
scales without Pulsar2.

``docs/axera-teng2-register-decode.md`` (PR #1831) located the calibration
floats in a standalone ``Relu``'s TENG (``teng2``) segment. Controlled
calibration sweeps (``elementwise_scale_sweep.py``: every sample spans exactly
``[lo, hi]``, so Pulsar2's MinMax scales and zero points are known per build)
show that, **at fixed zero points**, the whole compiled model depends on the
scales only through float32 values in segment 2, each written as four per-lane
copies one register-write record apart:

* ``Relu`` (input and output share one scale ``s``): ``f32(1 / s)`` then
  ``f32(s)``.
* ``Sqrt`` (input scale ``sx``, output scale ``sy``): ``f32(1 / sx)``,
  ``f32(sx)``, then ``f32(sy)``. When ``sx == sy`` the compiler drops the
  third group, so a template with distinct scales only serves targets with
  distinct scales.

The scales are Pulsar2's own float32 values (``quant_axmodel.json``). Every
formula reproduced every build bit for bit, including scales whose mantissa
bytes differ. ``npu_params``, the node attributes and every other MCode byte
stay the same.

**Zero points are not retargetable.** They are written through the stream's
LZ-style compressed register writes: a value is stored as a literal byte, or,
when an earlier record already holds the same value, as a back-reference to
it. Changing a zero point can therefore change lengths and shift every later
back-reference distance (``docs/axera-elementwise-scale-emit.md``). A template
serves exactly the zero points it was built with -- except through
``retarget_relu_records``, which works on the decompressed records
(``short_unit_codec.py``) and moves a Relu to any nonzero zero point.

**Binary ops (Add/Sub/Mul/Div) are refused.** Held-out builds whose input and
output scale ratios differ from the template's change 17-53 MCode regions,
not just float literals: back-references, distances, header bytes and (for Add
and Sub) ``npu_params``.

Everything outside the measured scope raises ``ValueError``: an op, shape or
zero point without a validated template, a target whose float equalities differ
from the template's, or a template whose floats do not match its recorded
calibration.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

TEMPLATE_DIR = os.path.join(_HERE, "fixtures", "elementwise_scale_emit")
OPS = ("Relu", "Sqrt")


@lru_cache(maxsize=1)
def _template_index() -> dict:
    with open(os.path.join(TEMPLATE_DIR, "index.json")) as f:
        return json.load(f)


def _load_index(template_dir: str) -> dict:
    if template_dir == TEMPLATE_DIR:
        return _template_index()
    with open(os.path.join(template_dir, "index.json")) as f:
        return json.load(f)


def _f32(x: float) -> float:
    return float(np.float32(x))


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(x)))[0]


def _f32_bytes(x: float) -> bytes:
    return struct.pack("<I", _f32_bits(x))


def _scale(scales: Mapping[str, float], name: str) -> float:
    if name not in scales:
        raise ValueError(f"missing scale for tensor {name!r}")
    s = _f32(scales[name])
    if not (s > 0 and np.isfinite(s)):
        raise ValueError(f"scale {name!r} must be a positive finite float")
    return s


def op_floats(op: str, scales: Mapping[str, float]) -> list[bytes]:
    """The float32 values (little-endian bytes) ``op``'s TENG segment carries
    for tensor scales ``scales`` (keys ``"x"`` and, for Sqrt, ``"y"``), in a
    fixed per-op order."""
    if op == "Relu":
        s = _scale(scales, "x")
        if "y" in scales and _f32(scales["y"]) != s:
            raise ValueError("Relu's input and output share one scale")
        return [_f32_bytes(1.0 / s), _f32_bytes(s)]
    if op == "Sqrt":
        sx, sy = _scale(scales, "x"), _scale(scales, "y")
        return [_f32_bytes(1.0 / sx), _f32_bytes(sx), _f32_bytes(sy)]
    raise ValueError(f"op {op!r} is not supported; validated ops: {OPS}")


def scale_floats(scale: float) -> tuple[bytes, bytes]:
    """Relu's ``(quant, dequant)`` float32 bytes, ``f32(1/s)`` and ``f32(s)``."""
    q, d = op_floats("Relu", {"x": scale})
    return q, d


def minmax_params(lo: float, hi: float) -> tuple[float, int]:
    """Pulsar2's MinMax U8 asymmetric ``(scale, zero_point)`` for a calibration
    range whose realised float32 min/max are ``lo``/``hi``."""
    lo64 = _f32(min(lo, 0.0))
    hi64 = _f32(max(hi, 0.0))
    span = hi64 - lo64
    if not span > 0:
        raise ValueError(f"empty calibration range [{lo}, {hi}]")
    scale = _f32(span / 255.0)
    # zero point from the unrounded range, round-half-to-even (127.5 -> 128)
    zp = int(np.clip(np.round(-lo64 * 255.0 / span), 0, 255))
    return scale, zp


def _mcode_initializer(model: onnx.ModelProto):
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(f"expected one *_neu MCode initializer, found {len(matches)}")
    return matches[0]


def _groups_of_four(offsets: list[int]) -> bool:
    """Whether ``offsets`` splits into consecutive groups of four lane copies,
    each one register-write record apart (7 or 8 bytes: the stream's compact
    and full record forms)."""
    if not offsets or len(offsets) % 4:
        return False
    for g in range(0, len(offsets), 4):
        grp = offsets[g : g + 4]
        if any(b - a not in (7, 8) for a, b in zip(grp, grp[1:])):
            return False
    return True


def float_slots(mc: bytes, values: list[bytes]) -> list[list[int]]:
    """Byte offsets (into the whole MCode blob) of every copy of each float in
    ``values`` inside segment 2.

    Copies are located by value, because the stream's variable-length forms
    tokenize the same four-lane group differently at different shapes (a
    batch-1 Relu writes one group as full ``V`` records, a batch-16 one as
    compact 7-byte records, and a tiled program repeats a group per tile). The
    structure is still checked: the values must be distinct, each must occur
    in whole groups of four lane copies one record apart, and no two copies may
    overlap. Raises ``ValueError`` otherwise.
    """
    if len(set(values)) != len(values):
        raise ValueError("template floats coincide; cannot tell their copies apart")
    try:
        _, segs = mcode.segments(mc)
    except (AssertionError, IndexError, ValueError, struct.error) as exc:
        raise ValueError(f"not a decodable MCode stream: {exc}") from exc
    pos, length, _ = segs[2]
    found = []
    for k, pat in enumerate(values):
        offs = [o for o in range(pos, pos + length - 3) if mc[o : o + 4] == pat]
        if not _groups_of_four(offs):
            raise ValueError(
                f"segment 2 lacks the measured structure for float #{k} "
                f"({pat.hex()}): copies at {[o - pos for o in offs]} "
                "(segment-relative) are not whole groups of four lane records"
            )
        found.append(offs)
    spans = sorted((o, o + 4) for offs in found for o in offs)
    if any(a_end > b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])):
        raise ValueError("float copies overlap")
    return found


def retarget(mc: bytes, op: str, old_scales, new_scales) -> bytes:
    """``mc`` with ``op``'s float copies rewritten from ``old_scales`` to
    ``new_scales``. Refuses a target whose floats coincide (the compiler emits
    a different program when two of them are equal)."""
    old = op_floats(op, old_scales)
    new = op_floats(op, new_scales)
    if len(set(new)) != len(new):
        raise ValueError(
            "target floats coincide (e.g. Sqrt with equal input/output scales); "
            "the compiler emits a different program for that case"
        )
    slots = float_slots(mc, old)
    out = bytearray(mc)
    for offs, value in zip(slots, new):
        for off in offs:
            out[off : off + 4] = value
    return bytes(out)


def retarget_scale(mc: bytes, old_scale: float, new_scale: float) -> bytes:
    """Relu shorthand for ``retarget``."""
    return retarget(mc, "Relu", {"x": old_scale}, {"x": new_scale})


def retarget_relu_records(mc: bytes, scale: float, zero_point: int) -> bytes:
    """A Relu template moved to ``(scale, zero_point)`` at the record level.

    ``retarget`` above predates the MCode codec (``short_unit_codec.py``) and
    patches float literals in the compressed stream, which is why it could not
    move zero points. Decompressed, a standalone Relu writes its one shared
    zero point as whole words to 0x1b10/0x1eb0/0x1a90 and its scale as
    ``1/s``, ``s`` lanes, exactly like the Relu of a non-fused Reshape -> Relu
    (``reshape_record_emit.retarget_scale``). Moving the ``x128,y128``
    template to any nonzero zero point reproduces native builds record for
    record (``docs/axera-step-real-calibration.md``); a zero point of 0 is a
    different program (``x0,y0`` templates)."""
    import reshape_record_emit as rre

    if not 0 < int(zero_point) < 256:
        raise ValueError(f"zero point {zero_point}: only nonzero uint8 retargets")
    return rre.retarget_scale(mc, scale, int(zero_point))


def emit_relu_at(shape, scale: float, zero_point: int, out_path: str) -> str:
    """A standalone ``Relu(x[shape])`` at any nonzero shared zero point, from
    the committed ``x128,y128`` template."""
    model, _ = load_template("Relu", shape, {"x": 128, "y": 128})
    mc = retarget_relu_records(
        bytes(_mcode_initializer(model).raw_data), scale, zero_point
    )
    return _write(model, mc, out_path)


def _key(op: str, shape, zero_points: Mapping[str, int]) -> str:
    zps = ",".join(f"{k}{int(v)}" for k, v in sorted(zero_points.items()))
    return f"{op}:{'x'.join(str(int(d)) for d in shape)}:{zps}"


def load_template(
    op: str, shape, zero_points: Mapping[str, int], template_dir: str = TEMPLATE_DIR
):
    """``(model, meta)`` for a committed template, or ``ValueError`` if no
    validated template covers ``(op, shape, zero_points)``. ``zero_points``
    names every quantized tensor (``x``, ``y``)."""
    index = _load_index(template_dir)
    key = _key(op, shape, zero_points)
    if key not in index:
        raise ValueError(f"no validated template for {key}; have {sorted(index)}")
    meta = index[key]
    with gzip.open(os.path.join(template_dir, meta["file"]), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return model, meta


def _write(model: onnx.ModelProto, mc: bytes, out_path: str) -> str:
    import step_recalibrate

    model = step_recalibrate.with_mcode(model, mc)  # keeps the dims in step
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def emit(
    op: str,
    shape,
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    out_path: str,
    template_dir: str = TEMPLATE_DIR,
) -> str:
    """Write a compiled standalone ``op(x[shape])`` model for Pulsar2
    quantization ``scales``/``zero_points`` (per tensor: ``x``, ``y``) to
    ``out_path``, from a committed template."""
    if op not in OPS:
        raise ValueError(f"op {op!r} is not supported; validated ops: {OPS}")
    for name, zp in zero_points.items():
        if not 0 <= int(zp) <= 255:
            raise ValueError(f"zero point {name!r} must be in [0, 255], got {zp}")
    model, meta = load_template(op, shape, zero_points, template_dir)
    mc = retarget(bytes(_mcode_initializer(model).raw_data), op, meta["scales"], scales)
    return _write(model, mc, out_path)


def emit_from_reference(
    reference_path: str,
    op: str,
    reference_scales: Mapping[str, float],
    reference_zero_points: Mapping[str, int],
    scales: Mapping[str, float],
    zero_points: Mapping[str, int],
    out_path: str,
) -> str:
    """Like ``emit`` but from any compiled standalone ``op`` whose own
    calibration is known (e.g. from its ``quant_axmodel.json``). Zero points
    must not change."""
    if {k: int(v) for k, v in zero_points.items()} != {
        k: int(v) for k, v in reference_zero_points.items()
    }:
        raise ValueError(
            f"zero point change {dict(reference_zero_points)} -> {dict(zero_points)} "
            "is not supported: zero points live in the stream's variable-length "
            "compressed register writes"
        )
    model = onnx.load(reference_path, load_external_data=False)
    mc = retarget(
        bytes(_mcode_initializer(model).raw_data), op, reference_scales, scales
    )
    return _write(model, mc, out_path)
