"""Template-plus-patch AX650 backend skeleton for tinygrad (architecture B).

First code for ``docs/axera-tinygrad-emitter-plan.md`` (#1834). It wires the
emitters this repository has already validated into the interfaces the plan
names, and exposes tinygrad-facing classes built on the user's tinygrad fork
``onnxsim/tinygrad`` (pinned below). It compiles nothing (templates are Pulsar2-built fixtures); only
``AXProgram`` touches the device:

* ``TemplateCache`` resolves a ``TemplateKey`` to a committed, Pulsar2-built and
  checked fixture. A key with no fixture raises ``ValueError``; building a new
  template (a Pulsar2 run) is deliberately out of scope here.
* ``Edit`` implementations change values inside a template without changing its
  structure, each by delegating to an existing emitter:
  ``GatherIndexEdit`` (``memory_emit.py``), ``TemplateOnly`` (Transpose,
  ``transpose_real_shapes.py``), ``ElementwiseScaleEdit`` (Relu/Sqrt,
  ``elementwise_scale_emit.py``; same-shape Add/Sub/Mul/Div,
  ``binary_op_scale_emit.py``), ``ConvWeightEdit`` (``conv_weight_learn.py``,
  ``conv_bias_requant.py``). ``predicted_npu_params`` exposes the tile-table
  predictors (``dma_tile_predict.py``, ``add_tile_predict.py``,
  ``elementwise_two_input_tile_predict.py``) as a cross-check.
* ``QuantPolicy`` selects each constant weight's storage type from tinygrad
  dtypes (``dtypes.int8`` -> ``s8``, ``dtypes.bfloat16`` -> ``bf16``, ...,
  ``"s4"``), per op type or node, or ``"auto"`` under an error budget
  (``weight_dtype_costs`` / ``choose_weight_dtype``). Only decoded
  (path, op, dtype, shape) combinations pass ``validate_weight_choice``;
  ``encode_weight`` gives the stored bytes (``llm_build_dtype_analysis.py``
  for llm_build, ``emitter.codes_of`` for S8 Conv).
* ``plan_node`` / ``coverage_report`` map a real graph's nodes onto what the
  backend can produce today and say why the rest is refused.
* ``tinygrad_classes()`` returns ``AXCompiler`` (a tinygrad ``Compiler`` whose
  "source" is a JSON template request and whose output is patched ``.axmodel``
  bytes), ``AXAllocator`` (host-staged buffers) and ``AXProgram`` (loads the
  bytes into a persistent AXCL session, ``axcl_session.py``, and runs them on
  the card). ``register_ax_device()`` makes ``"AX"`` a tinygrad device.

Every unvalidated op, shape, calibration class or edit raises ``ValueError``;
nothing here extrapolates.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import binary_op_scale_emit as bse  # noqa: E402
import elementwise_scale_emit as ew  # noqa: E402
import llm_build_dtype_analysis as lbd  # noqa: E402
import matmul_record_emit as mre  # noqa: E402
import memory_emit  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import reshape_record_emit as rre  # noqa: E402
import transpose_real_shapes  # noqa: E402

TOOLCHAIN = "pulsar2:7.0-lite"
TINYGRAD_FORK = "onnxsim/tinygrad"
TINYGRAD_FORK_SHA = "ba00cbb3c6b8a6bcbee04c22cb4a0b0b9cee216e"
"""The ``onnxsim/tinygrad`` master commit this skeleton was written against.
Install with ``uv run --with "tinygrad@https://github.com/onnxsim/tinygrad/
archive/<sha>.tar.gz"``."""

_FIXTURES = os.path.join(_HERE, "fixtures")

# (x shape, w shape, strides, pads) -> template for the Conv weight edit.
# Only shapes whose full npu_params pipeline was validated against a native
# held-out build are listed.  The stem uses its own mcode offsets and therefore
# has a dedicated emitter, even though its weight table is the same general
# bit-permutation family as the other Conv templates.
_CONV_TEMPLATES: dict[tuple, dict[str, Any]] = {
    ((16, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1)): {
        "dir": "conv_weight_learn",
        "model": "reference.axmodel.gz",
        "map": "stage1_map.npz",
        "kind": "biased",
        "block_at": 36864,
    },
    ((16, 64, 56, 56), (128, 64, 1, 1), (2, 2), (0, 0, 0, 0)): {
        "dir": "conv_learn_downsample",
        "model": "s1_reference.axmodel.gz",
        "map": "s1_map.npz",
        "kind": "contiguous",
        "block_at": 9216,
        "block_len": 2 * 4 * 128,
    },
    ((16, 3, 224, 224), (64, 3, 7, 7), (2, 2), (3, 3, 3, 3)): {
        "dir": "conv_learn_stem",
        "model": "reference.axmodel.gz",
        "map": "stem_map.npz",
        "kind": "stem",
    },
}


def _tup(x) -> tuple:
    return tuple(int(v) for v in x)


@dataclasses.dataclass(frozen=True)
class TemplateKey:
    """What a cached compiled template is keyed by (plan section 6).

    ``shapes`` are the data inputs' shapes. ``attrs`` carries what changes the
    compiled program: ``perm`` (Transpose), ``indices`` = index count (Gather),
    ``w``/``strides``/``pads`` (Conv). ``calibration_class`` names the part of
    the calibration a value edit cannot change -- for Relu/Sqrt the zero points,
    e.g. ``"x0,y0"``. ``dtypes`` are the data inputs' (activation) dtypes;
    ``weight_dtype`` is how a constant weight is stored (``WEIGHT_DTYPES``
    names, see ``axera_weight_dtype``). It is ``""`` for ops without a
    constant weight, and for a Conv it means the template's own (``s8``).
    """

    op: str
    shapes: tuple[tuple[int, ...], ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    dtypes: tuple[str, ...] = ("float32",)
    calibration_class: str = ""
    toolchain: str = TOOLCHAIN
    weight_dtype: str = ""

    def attr(self, name: str, default=None):
        return dict(self.attrs).get(name, default)

    def to_json(self) -> dict:
        return {
            "op": self.op,
            "shapes": [list(s) for s in self.shapes],
            "attrs": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in self.attrs
            },
            "dtypes": list(self.dtypes),
            "calibration_class": self.calibration_class,
            "toolchain": self.toolchain,
            "weight_dtype": self.weight_dtype,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> TemplateKey:
        attrs = tuple(
            sorted(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in d.get("attrs", {}).items()
            )
        )
        return cls(
            op=d["op"],
            shapes=tuple(_tup(s) for s in d["shapes"]),
            attrs=attrs,
            dtypes=tuple(d.get("dtypes", ("float32",))),
            calibration_class=d.get("calibration_class", ""),
            toolchain=d.get("toolchain", TOOLCHAIN),
            weight_dtype=d.get("weight_dtype", ""),
        )


# --------------------------------------------------------------------------
# Weight dtype selection (quantization choices).
#
# A caller picks how each constant weight is stored with tinygrad dtypes (or
# the Axera names), per op type or per node, through ``QuantPolicy``. What the
# two Pulsar2 paths offer, and what this repository has decoded of it:
#
# * ``build`` (``pulsar2 build``, the CNN / training-step path): the weight
#   type is S8 or FP32 only. The committed Conv templates are S8, with the
#   per-output-channel quantizer ``emitter.codes_of`` (step ``max|w|/127.5``).
#   FP32 Conv weights have no template (a build is needed).
# * ``llm_build`` (``pulsar2 llm_build``): s4/s8/fp16/bf16/fp8_e4m3/fp8_e5m2
#   (and fp32) for Linear weights. ``llm_build_dtype_analysis.py`` (#1866)
#   reproduces the stored bytes of the first six from our own builds (fp32
#   was not built). There is no committed llm_build engine template, so this
#   path yields weight bytes (``encode_weight``), not an ``.axmodel``.
#
# In the ResNet18 training step no Conv/Gemm/MatMul weight is a constant:
# the 20 Conv, the Gemm and one MatMul take theirs from graph inputs
# (trainable state) and the other 40 MatMuls multiply computed tensors. Those are runtime tensors,
# quantized with the activation calibration, so a weight dtype choice changes
# none of them; it applies to frozen-weight deployment (build Conv) and to
# llm_build Linear layers.
# --------------------------------------------------------------------------

WEIGHT_DTYPES = ("s4", "s8", "fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "fp32")
WEIGHT_PATHS = {
    "build": ("s8", "fp32"),
    "llm_build": ("s4", "s8", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp32"),
}
"""The weight types each Pulsar2 path offers at all (decoded or not)."""

# (path, op) -> dtypes whose stored bytes are reproduced from our own builds.
_VALIDATED_WEIGHT_DTYPES: dict[tuple[str, str], tuple[str, ...]] = {
    ("build", "Conv"): ("s8",),
    ("llm_build", "Linear"): ("s4", "s8", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"),
}

# In-feature counts each llm_build type was checked at (256-hidden layer:
# cin 256 and 512; the intermediate=2048 build: s8, s4 and bf16 only).
_LLM_VALIDATED_CIN = {
    "s8": (256, 512, 2048),
    "s4": (256, 512, 2048),
    "bf16": (256, 512, 2048),
    "fp16": (256, 512),
    "fp8_e4m3": (256, 512),
    "fp8_e5m2": (256, 512),
}

# tinygrad ``DType.name`` (onnxsim/tinygrad @ TINYGRAD_FORK_SHA) -> Axera name.
# The fork has no 4-bit integer dtype, so s4 is chosen by name ("s4"/"int4").
_TINYGRAD_NAMES = {
    "float": "fp32",
    "half": "fp16",
    "__bf16": "bf16",
    "signed char": "s8",
    "float8_e4m3": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
}
_DTYPE_ALIASES = {
    **{d: d for d in WEIGHT_DTYPES},
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
    "int8": "s8",
    "int4": "s4",
    "fp8e4m3": "fp8_e4m3",
    "fp8e5m2": "fp8_e5m2",
}
_WEIGHT_OPS = ("Conv", "Gemm", "MatMul")


def axera_weight_dtype(dtype) -> str:
    """The Axera weight-type name for a tinygrad ``DType`` or a name.

    Accepts ``tinygrad.dtypes.{float32, float16, bfloat16, int8, fp8e4m3,
    fp8e5m2}``, their attribute names, the Axera names in ``WEIGHT_DTYPES``,
    and ``"int4"``/``"s4"``. Refuses everything else, including the fnuz fp8
    variants (a different encoding from llm_build's OCP-style e4m3/e5m2) and
    unsigned types (both paths store signed symmetric codes)."""
    if isinstance(dtype, str):
        out = _DTYPE_ALIASES.get(dtype)
        if out is None:
            raise ValueError(f"no Axera weight type for {dtype!r}")
        return out
    name = getattr(dtype, "name", None)
    if name is None or getattr(dtype, "count", 1) != 1:
        raise ValueError(f"not a scalar tinygrad dtype: {dtype!r}")
    out = _TINYGRAD_NAMES.get(name)
    if out is None:
        raise ValueError(f"no Axera weight type for tinygrad dtype {name!r}")
    return out


def validate_weight_choice(
    path: str, op: str, dtype, w_shape: Sequence[int] | None = None
) -> str:
    """Refuse a (path, op, dtype, weight shape) this repository has not
    decoded; return the Axera dtype name otherwise. ``op`` is ``"Conv"`` on
    the build path and ``"Linear"`` (weight ``[out_features, in_features]``)
    on llm_build."""
    dt = axera_weight_dtype(dtype)
    offered = WEIGHT_PATHS.get(path)
    if offered is None:
        raise ValueError(f"unknown Pulsar2 path {path!r}; one of {list(WEIGHT_PATHS)}")
    if dt not in offered:
        raise ValueError(f"pulsar2 {path} does not offer {dt} weights, only {offered}")
    validated = _VALIDATED_WEIGHT_DTYPES.get((path, op), ())
    if dt not in validated:
        raise ValueError(
            f"{dt} {op} weights on pulsar2 {path} are not decoded "
            f"(validated: {validated or 'none'}); that needs a Pulsar2 build"
        )
    if w_shape is None:
        return dt
    shape = _tup(w_shape)
    if path == "build":
        known = {_tup(k[1]) for k in _CONV_TEMPLATES}
        if shape not in known:
            raise ValueError(f"no validated {dt} Conv template for weight {shape}")
    else:
        if len(shape) != 2 or shape[0] % lbd.ROW_BLOCK:
            raise ValueError(
                f"Linear weight must be [out, in] with out a multiple of "
                f"{lbd.ROW_BLOCK}, got {shape}"
            )
        if shape[1] not in _LLM_VALIDATED_CIN[dt]:
            raise ValueError(
                f"{dt} was checked at in_features {_LLM_VALIDATED_CIN[dt]}, "
                f"not {shape[1]}"
            )
    return dt


def _dequantized(w: np.ndarray, path: str, dt: str) -> np.ndarray:
    import emitter

    if path == "build":  # s8, the only validated build type
        codes = emitter.codes_of(w).astype(np.float32) - 128.0
        step = emitter.weight_scales(w).reshape((-1,) + (1,) * (w.ndim - 1))
        return codes * step
    if dt in lbd.FLOAT_TYPES:
        return lbd.round_float(w, dt)
    q, scale = lbd.quantize_int(w, 8 if dt == "s8" else 4)
    return (q * scale[:, None]).astype(np.float32)


def encode_weight(w: np.ndarray, dtype, path: str = "llm_build") -> list[bytes]:
    """The stored weight bytes for ``w`` at the chosen dtype.

    llm_build: the 32-row blocks as llm_build stores them (identical blocks
    dropped, ``lbd.dedup_blocks``). build (S8 Conv): one element, the
    per-output-channel uint8 codes in ``w``'s order; their placement in
    ``npu_params`` and the requant block go through ``ConvWeightEdit``."""
    w = np.asarray(w, np.float32)
    op = "Conv" if path == "build" else "Linear"
    dt = validate_weight_choice(path, op, dtype, w.shape)
    if path == "build":
        import emitter

        return [emitter.codes_of(w).tobytes()]
    return lbd.dedup_blocks(lbd.encode_matrix(w, dt))


def weight_dtype_costs(
    w: np.ndarray, path: str = "llm_build", dtypes: Sequence | None = None
) -> list[dict]:
    """Per-dtype storage and error for ``w`` on ``path``, to choose from.

    ``bytes_per_param`` is the stored weight bytes (llm_build: whole 32-row
    blocks including the scale/row-sum tails, before deduplication; build:
    the uint8 codes, excluding the per-channel requant block).
    ``rel_rms_error`` is ``rms(deq - w) / rms(w)``. Only validated dtypes
    are listed; ``dtypes`` narrows them (and refuses unvalidated ones)."""
    w = np.asarray(w, np.float32)
    op = "Conv" if path == "build" else "Linear"
    if dtypes is None:
        names = _VALIDATED_WEIGHT_DTYPES.get((path, op), ())
    else:
        names = tuple(dict.fromkeys(axera_weight_dtype(d) for d in dtypes))
    rms = float(np.sqrt(np.mean(np.square(w, dtype=np.float64))))
    out = []
    for dt in names:
        validate_weight_choice(path, op, dt, w.shape)
        if path == "build":
            nbytes = w.size
        else:
            nbytes = lbd.block_bytes(dt, w.shape[1]) * (w.shape[0] // lbd.ROW_BLOCK)
        err = _dequantized(w, path, dt).astype(np.float64) - w
        rel = float(np.sqrt(np.mean(err * err))) / rms if rms else 0.0
        out.append(
            {
                "dtype": dt,
                "bytes_per_param": nbytes / w.size,
                "rel_rms_error": rel,
                "max_abs_error": float(np.abs(err).max()),
                "sqnr_db": float(-20 * np.log10(rel)) if rel else float("inf"),
            }
        )
    return out


def choose_weight_dtype(
    w: np.ndarray,
    error_budget: float,
    path: str = "llm_build",
    dtypes: Sequence | None = None,
) -> str:
    """The validated dtype with the fewest stored bytes whose
    ``rel_rms_error`` is within ``error_budget``; ties go to the smaller
    error. On llm_build every float type stores 4 B/param, so fp8 never
    wins over fp16/bf16 here: it saves no space and loses precision."""
    costs = weight_dtype_costs(w, path, dtypes)
    ok = [c for c in costs if c["rel_rms_error"] <= error_budget]
    if not ok:
        best = min(costs, key=lambda c: c["rel_rms_error"])
        raise ValueError(
            f"no validated dtype within rel_rms_error {error_budget}; "
            f"best is {best['dtype']} at {best['rel_rms_error']:.3g}"
        )
    return min(ok, key=lambda c: (c["bytes_per_param"], c["rel_rms_error"]))["dtype"]


@dataclasses.dataclass(frozen=True)
class QuantPolicy:
    """Which weight dtype each weight-bearing node gets.

    ``default`` and the values of ``overrides`` are tinygrad dtypes, Axera
    names, or ``"auto"`` (``choose_weight_dtype`` under ``error_budget``,
    which needs the weights). ``overrides`` keys are node names first, then
    op types. ``path`` is ``"build"`` or ``"llm_build"``."""

    default: Any = "s8"
    path: str = "build"
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    error_budget: float | None = None

    def __post_init__(self):
        if self.path not in WEIGHT_PATHS:
            raise ValueError(f"unknown Pulsar2 path {self.path!r}")
        norm = {k: self._norm(v) for k, v in dict(self.overrides).items()}
        object.__setattr__(self, "overrides", norm)
        object.__setattr__(self, "default", self._norm(self.default))

    def _norm(self, dtype) -> str:
        if dtype == "auto":
            if self.error_budget is None:
                raise ValueError('"auto" needs an error_budget')
            return "auto"
        dt = axera_weight_dtype(dtype)
        if dt not in WEIGHT_PATHS[self.path]:
            raise ValueError(
                f"pulsar2 {self.path} does not offer {dt} weights, "
                f"only {WEIGHT_PATHS[self.path]}"
            )
        return dt

    def choice_for(self, op: str, node: str | None = None) -> str:
        """The configured choice (a dtype name or ``"auto"``), unvalidated."""
        if node is not None and node in self.overrides:
            return self.overrides[node]
        return self.overrides.get(op, self.default)

    def resolve(
        self, op: str, w: np.ndarray | None = None, node: str | None = None
    ) -> str:
        """The concrete, validated dtype for one node."""
        path_op = op if self.path == "build" else "Linear"
        choice = self.choice_for(op, node)
        shape = None if w is None else np.shape(w)
        if choice == "auto":
            if w is None:
                raise ValueError('"auto" needs the weights to choose from')
            return choose_weight_dtype(w, self.error_budget, self.path)
        return validate_weight_choice(self.path, path_op, choice, shape)

    def to_json(self) -> dict:
        return {
            "default": self.default,
            "path": self.path,
            "overrides": dict(self.overrides),
            "error_budget": self.error_budget,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> QuantPolicy:
        return cls(
            default=d.get("default", "s8"),
            path=d.get("path", "build"),
            overrides=dict(d.get("overrides", {})),
            error_budget=d.get("error_budget"),
        )


def weight_dtype_for_record(rec: Mapping, policy: QuantPolicy) -> dict:
    """What ``policy`` does to one graph node's weight, for ``coverage_report``:
    ``{"weight_dtype": name or None, "note": why}``."""
    op = rec["op"]
    if op not in _WEIGHT_OPS:
        return {"weight_dtype": None, "note": "no weight operand"}
    attrs = rec.get("attrs", {})
    source = attrs.get("weight_source")
    if source is None and attrs.get("weight_is_graph_input"):
        source = "graph_input"
    if source is None:
        return {
            "weight_dtype": None,
            "note": "record has no weight_source; re-run extract_step_ops",
        }
    if source != "initializer":
        return {
            "weight_dtype": None,
            "note": f"weight is {source.replace('_', ' ')}: a runtime tensor, "
            "quantized with the activation calibration; no weight dtype applies",
        }
    try:
        if policy.path == "llm_build" and op == "Conv":
            raise ValueError("llm_build has no Conv; use the build path")
        path_op = op if policy.path == "build" else "Linear"
        if path_op == "Linear" and op != "MatMul":
            raise ValueError(
                f"llm_build Linear weights were decoded for MatMul, not {op}"
            )
        choice = policy.choice_for(op, rec.get("name"))
        if choice == "auto":
            return {"weight_dtype": "auto", "note": "chosen when weights are given"}
        w_shape = attrs.get("w")
        if path_op == "Linear" and w_shape:
            w_shape = list(w_shape)[::-1]  # ONNX MatMul B is [in, out]
        dt = validate_weight_choice(policy.path, path_op, choice, w_shape)
        return {"weight_dtype": dt, "note": "validated"}
    except ValueError as exc:
        return {"weight_dtype": None, "note": f"refused: {exc}"}


def _zero_points_from_class(cls: str) -> dict[str, int]:
    out = {}
    for part in filter(None, cls.split(",")):
        out[part[0]] = int(part[1:])
    return out


def _load_gz_model(path: str) -> onnx.ModelProto:
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


@dataclasses.dataclass
class TemplateEntry:
    kind: str
    path: str
    meta: dict


class TemplateCache:
    """Resolves measured templates and performs safe Pulsar-free graph reuse.

    Op-level misses remain explicit errors because synthesizing a new fused
    program still requires a compiler.  A complete source graph can however
    reuse a validated AX model when its topology is identical; that path is
    delegated to :mod:`template_model_generator` and never invokes Pulsar2.
    """

    def __init__(self):
        self._entries: dict[TemplateKey, TemplateEntry] = {}

    def lookup(self, key: TemplateKey) -> TemplateEntry:
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        entry = self._lookup(key)
        self._entries[key] = entry
        return entry

    def _lookup(self, key: TemplateKey) -> TemplateEntry:
        if key.toolchain != TOOLCHAIN:
            raise ValueError(f"no templates for toolchain {key.toolchain!r}")
        if key.dtypes != ("float32",):
            raise ValueError(
                f"only float32-activation templates exist, got {key.dtypes} "
                "(a weight's storage type is weight_dtype)"
            )
        op = key.op
        if op == "Conv":
            validate_weight_choice("build", "Conv", key.weight_dtype or "s8")
        elif key.weight_dtype:
            raise ValueError(f"{op} has no constant weight; weight_dtype must be ''")
        if op == "Gather":
            (shape,) = key.shapes
            pair = (shape, key.attr("indices"))
            name = memory_emit._GATHER_LAST_AXIS_TEMPLATES.get(pair)
            if name is None or key.attr("axis", len(shape) - 1) != len(shape) - 1:
                raise ValueError(f"no last-axis Gather template for {pair}")
            return TemplateEntry(
                "gather", os.path.join(_FIXTURES, name), {"pair": pair}
            )
        if op == "Transpose":
            (shape,) = key.shapes
            path = transpose_real_shapes.template_path(shape, key.attr("perm"))
            return TemplateEntry("transpose", path, {})
        if op in ew.OPS:
            (shape,) = key.shapes
            zps = _zero_points_from_class(key.calibration_class)
            _, meta = ew.load_template(op, shape, zps)
            return TemplateEntry(
                "elementwise", os.path.join(ew.TEMPLATE_DIR, meta["file"]), meta
            )
        if op in bse.OPS:
            (shape,) = key.shapes
            zps = _zero_points_from_class(key.calibration_class)
            _, meta = bse.load_template(op, shape, zps)
            return TemplateEntry(
                "binary", os.path.join(bse.TEMPLATE_DIR, meta["file"]), meta
            )
        if op == "Conv":
            (shape,) = key.shapes
            ck = (
                shape,
                _tup(key.attr("w", ())),
                _tup(key.attr("strides", (1, 1))),
                _tup(key.attr("pads", (0, 0, 0, 0))),
            )
            meta = _CONV_TEMPLATES.get(ck)
            if meta is None:
                raise ValueError(f"no validated Conv template for {ck}")
            return TemplateEntry(
                "conv", os.path.join(_FIXTURES, meta["dir"], meta["model"]), meta
            )
        raise ValueError(f"no templates for op {op!r}")

    def load(self, key: TemplateKey) -> onnx.ModelProto:
        return _load_gz_model(self.lookup(key).path)

    def generate_graph_template(
        self,
        source_path: str,
        template_source_path: str,
        template_axmodel_path: str,
        output_path: str,
    ) -> str:
        """Copy a validated same-topology AX template without Pulsar2.

        The source graph is checked against the graph used to validate the
        compiled template.  This deliberately does not patch weights or
        constants: those changes need an emitter with a corresponding
        validation record.  Returning the output path makes this suitable for
        the tinygrad compiler/cache seam while keeping the generic generator
        independently usable from the command line.
        """
        from template_model_generator import generate

        generate(
            source_path,
            template_source_path,
            template_axmodel_path,
            output_path,
        )
        return output_path

    def generate_graph_template_bytes(
        self,
        source_path: str,
        template_source_path: str,
        template_axmodel_path: str,
    ) -> bytes:
        """Return a validated same-topology AX model without disk IO."""
        from template_model_generator import generate_model

        return generate_model(
            source_path,
            template_source_path,
            template_axmodel_path,
        ).SerializeToString()

    def get_or_build(self, key: TemplateKey, onnx_bytes: bytes | None = None):
        """Plan section 6: build on a miss. Builds are Pulsar2 runs, out of scope
        for this skeleton, so a miss is reported instead of built."""
        try:
            return self.lookup(key).path
        except ValueError as exc:
            raise NotImplementedError(
                f"template miss needs a Pulsar2 build (not in this skeleton): {exc}"
            ) from exc


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise ValueError(f"model has no initializer {name!r}")


def _mcode_initializer(model: onnx.ModelProto) -> onnx.TensorProto:
    found = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(found) != 1:
        raise ValueError(f"expected one *_neu MCode initializer, found {len(found)}")
    return found[0]


class Edit(Protocol):
    """A value change inside a template (plan section 6). ``validate`` refuses
    anything not decoded for that exact template; ``apply`` returns the edited
    model."""

    def validate(self, key: TemplateKey, entry: TemplateEntry) -> None: ...

    def apply(
        self, key: TemplateKey, entry: TemplateEntry, model: onnx.ModelProto
    ) -> onnx.ModelProto: ...

    def to_json(self) -> dict: ...


@dataclasses.dataclass
class TemplateOnly:
    """The template is already the complete artifact (Transpose: no data)."""

    def validate(self, key, entry):
        if entry.kind != "transpose":
            raise ValueError(f"{key.op} template needs an edit, not TemplateOnly")

    def apply(self, key, entry, model):
        return model

    def to_json(self):
        return {"type": "template_only"}


@dataclasses.dataclass
class GatherIndexEdit:
    """Retarget a last-axis Gather's constant indices (``memory_emit.py``)."""

    indices: Sequence[int]

    def validate(self, key, entry):
        if entry.kind != "gather":
            raise ValueError("GatherIndexEdit applies only to Gather templates")
        width = key.shapes[0][-1]
        if len(self.indices) != key.attr("indices"):
            raise ValueError("index count must match the template")
        if any(not 0 <= int(i) < width for i in self.indices):
            raise ValueError(f"indices must lie in [0, {width})")

    def apply(self, key, entry, model):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.axmodel")
            out = os.path.join(tmp, "out.axmodel")
            onnx.save(model, ref)
            memory_emit.emit_gather_last_axis_axmodel(
                ref, out, indices=[int(i) for i in self.indices]
            )
            return onnx.load(out, load_external_data=False)

    def to_json(self):
        return {"type": "gather_indices", "indices": [int(i) for i in self.indices]}


@dataclasses.dataclass
class ElementwiseScaleEdit:
    """New per-tensor scales at the template's zero points: Relu/Sqrt
    (``elementwise_scale_emit.py``, #1840) or same-shape Add/Sub/Mul/Div
    (``binary_op_scale_emit.py``)."""

    scales: Mapping[str, float]

    def validate(self, key, entry):
        if entry.kind == "binary":
            bse.op_values(key.op, self.scales)  # raises on missing/invalid scales
            return
        if entry.kind != "elementwise":
            raise ValueError(
                "ElementwiseScaleEdit applies only to Relu/Sqrt/Add/Sub/Mul/Div"
            )
        ew.op_floats(key.op, self.scales)  # raises on missing/invalid scales

    def apply(self, key, entry, model):
        if entry.kind == "binary":
            return bse.emit_model(
                model,
                key.op,
                entry.meta["scales"],
                dict(self.scales),
                entry.meta["zero_points"],
            )
        import step_recalibrate

        mc = ew.retarget(
            bytes(_mcode_initializer(model).raw_data),
            key.op,
            entry.meta["scales"],
            dict(self.scales),
        )
        return step_recalibrate.with_mcode(model, mc)

    def to_json(self):
        return {"type": "elementwise_scales", "scales": dict(self.scales)}


@dataclasses.dataclass
class ConvWeightEdit:
    """New frozen weights for a Conv template (``npu_params`` weight codes plus
    the bias-aware requant block). The MCode is the template's, unedited, so the
    activation calibration must be the template's own; this is the frozen-weight
    refresh path, not the training path (trainable weights are graph inputs)."""

    w: np.ndarray
    b: np.ndarray
    x_scale: float
    x_zero: float
    y_scale: float
    y_zero: float

    def validate(self, key, entry):
        if entry.kind != "conv":
            raise ValueError("ConvWeightEdit applies only to Conv templates")
        if tuple(self.w.shape) != _tup(key.attr("w", ())):
            raise ValueError(f"weight shape {self.w.shape} != template {key.attr('w')}")
        if tuple(self.b.shape) != (self.w.shape[0],):
            raise ValueError("bias must have Cout elements")

    def apply(self, key, entry, model):
        import conv_weight_learn
        import step_recalibrate

        meta = entry.meta
        origin = np.load(os.path.join(_FIXTURES, meta["dir"], meta["map"]))["origin"]
        table_init = _initializer(model, "npu_params")
        ref = np.frombuffer(bytes(table_init.raw_data), np.uint8)
        args = (self.x_scale, self.x_zero, self.y_scale, self.y_zero)
        if meta["kind"] == "stem":
            # The stem has a different fixed-width mcode layout from the
            # stage-1 and downsample Conv families.  Keep that knowledge in
            # conv_weight_learn_stem instead of applying the generic offsets.
            import conv_weight_learn_stem

            table, mc = conv_weight_learn_stem.emit_stem_conv(
                ref,
                bytes(_mcode_initializer(model).raw_data),
                origin,
                self.w,
                self.b,
                *args,
            )
            table_init.raw_data = np.asarray(table, np.uint8).tobytes()
            return step_recalibrate.with_mcode(model, mc)
        if meta["kind"] == "biased":
            table = conv_weight_learn.emit_biased(
                ref, origin, self.w, self.b, *args, meta["block_at"]
            )
        else:
            import conv_bias_requant

            table = conv_bias_requant.emit_conv_table(
                ref,
                origin,
                self.w,
                self.b,
                *args,
                block_at=meta["block_at"],
                block_len=meta["block_len"],
            )
        table_init.raw_data = np.asarray(table, np.uint8).tobytes()
        return model

    def to_json(self):
        raise ValueError("ConvWeightEdit carries arrays; pass it in-process")


def edit_from_json(d: Mapping) -> Edit:
    kind = d.get("type")
    if kind == "template_only":
        return TemplateOnly()
    if kind == "gather_indices":
        return GatherIndexEdit(list(d["indices"]))
    if kind == "elementwise_scales":
        return ElementwiseScaleEdit(dict(d["scales"]))
    raise ValueError(f"unknown or non-serializable edit {kind!r}")


@dataclasses.dataclass
class EditSet:
    edits: list

    def build(self, key: TemplateKey, cache: TemplateCache | None = None):
        """Load the template for ``key``, validate every edit against it, then
        apply them in order. Nothing is applied if any edit is refused."""
        cache = cache or TemplateCache()
        entry = cache.lookup(key)
        if not self.edits:
            raise ValueError("an EditSet needs at least one edit (TemplateOnly ok)")
        for edit in self.edits:
            edit.validate(key, entry)
        model = _load_gz_model(entry.path)
        for edit in self.edits:
            model = edit.apply(key, entry, model)
        return model


def predicted_npu_params(
    op: str, shape: Sequence[int], scales: Mapping[str, float] | None = None
) -> bytes:
    """The tile-table predictors behind one call. Raises outside their domains."""
    import dma_tile_predict

    shape = [int(d) for d in shape]
    if op in ("Relu", "Sqrt"):
        return dma_tile_predict.predict_params(shape)
    if scales is None:
        raise ValueError(f"{op} tile prediction needs x/z/y scales")
    if op == "Add":
        import add_tile_predict

        return add_tile_predict.predict_params(
            shape, scales["x"], scales["z"], scales["y"]
        )
    if op in ("Sub", "Mul", "Div"):
        import elementwise_two_input_tile_predict as two

        return two.predict_params(op, shape, scales["x"], scales["z"], scales["y"])
    raise ValueError(f"no tile predictor for {op!r}")


# --------------------------------------------------------------------------
# Graph side: real graph nodes -> keys -> which edit, or why not.
# --------------------------------------------------------------------------

_ELEMENTWISE_ZP_CLASSES = ("x0,y0", "x128,y128")
# Div's denominators in the step are positive (sqrt(v) + eps), hence z0.
_BINARY_ZP_CLASSES = {
    "Add": ("x0,y0,z0", "x128,y128,z128"),
    "Sub": ("x0,y0,z0", "x128,y128,z128"),
    "Mul": ("x0,y0,z0", "x128,y128,z128"),
    "Div": ("x0,y0,z0", "x128,y128,z0"),
}


def extract_step_ops(onnx_path: str) -> list[dict]:
    """Per-node records of a real graph for ``coverage_report``: op type, data
    input shapes, and the attributes that pick a template."""
    from onnx import shape_inference

    model = shape_inference.infer_shapes(onnx.load(onnx_path, load_external_data=False))
    shapes: dict[str, list[int]] = {}
    for v in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    inits = {i.name: i for i in model.graph.initializer}
    consts = {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    graph_inputs = {v.name for v in model.graph.input}
    records = []
    producers = {o for n in model.graph.node for o in n.output}
    for node in model.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        rec: dict[str, Any] = {
            "op": node.op_type,
            "name": node.name,
            "attrs": {},
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        if node.op_type in _WEIGHT_OPS and len(node.input) > 1:
            wname = node.input[1]
            if wname in inits:
                rec["attrs"]["weight_source"] = "initializer"
                rec["attrs"].setdefault("w", list(inits[wname].dims))
            elif wname in graph_inputs:
                rec["attrs"]["weight_source"] = "graph_input"
            elif wname in producers:
                rec["attrs"]["weight_source"] = "computed"
        data_inputs = [i for i in node.input if i and i not in inits]
        rec["shapes"] = [shapes.get(i, []) for i in data_inputs[:1]]
        if node.op_type == "Transpose":
            rec["attrs"]["perm"] = list(attrs.get("perm", []))
        elif node.op_type == "Gather":
            rank = len(rec["shapes"][0]) if rec["shapes"] else 0
            rec["attrs"]["axis"] = int(attrs.get("axis", 0)) % max(rank, 1)
            idx = inits.get(node.input[1])
            rec["attrs"]["indices"] = int(np.prod(idx.dims)) if idx is not None else -1
        elif node.op_type == "Conv":
            rec["attrs"]["w"] = shapes.get(node.input[1]) or list(
                inits[node.input[1]].dims
            )
            rec["attrs"]["strides"] = list(attrs.get("strides", [1, 1]))
            rec["attrs"]["pads"] = list(attrs.get("pads", [0, 0, 0, 0]))
            rec["attrs"]["weight_is_graph_input"] = node.input[1] in graph_inputs
        elif node.op_type in ("Reshape", "Squeeze"):
            rec["attrs"]["out"] = shapes.get(node.output[0], [])
        elif node.op_type in bse.OPS:
            a, b = node.input[:2]
            if {a, b} & (set(inits) | consts):
                rec["attrs"]["form"] = "const"
            elif shapes.get(a) != shapes.get(b):
                rec["attrs"]["form"] = "broadcast"
            else:
                rec["attrs"]["form"] = "same_shape"
        records.append(rec)
    # misc_op_record_emit template keys (misc.STEP_OPS), in node order
    misc_keys = iter(misc.step_node_keys(onnx_path))
    for rec in records:
        if rec["op"] in _MISC_OPS:
            rec["attrs"]["misc_key"] = next(misc_keys)[1]
    # A bias-flatten Reshape compiles fused with its producing ReduceSum: key
    # the pair's chain template (misc_op_record_emit, ``...:reshape<C>``).
    by_output = {o: r for r in records for o in r["outputs"]}
    for rec in records:
        src = by_output.get(rec["inputs"][0]) if rec["op"] == "Reshape" else None
        if (
            src is not None
            and src["op"] == "ReduceSum"
            and len(rec["attrs"]["out"]) == 1
        ):
            rs_key = src["attrs"]["misc_key"]
            rec["attrs"]["fused_key"] = f"{rs_key}:reshape{rec['attrs']['out'][0]}"
            rec["attrs"]["fused_producer_key"] = rs_key
            rec["attrs"]["fused_input"] = src["inputs"][0]
    return records


_MISC_OPS = misc.STEP_OPS


def _fused_reducesum_chain(rec: Mapping) -> str | None:
    """The validated template computing a bias-flatten Reshape together with
    its producing ReduceSum: the ``ReduceSum -> Reshape`` chain build, or the
    producer's same-bytes equivalent, which already writes the flattened
    ``[C]`` output (ReduceSum_474 -> Reshape_475)."""
    attrs = rec.get("attrs", {})
    key = attrs.get("fused_key")
    if key is None:
        return None
    if key in misc.load_index():
        return key
    alt = misc.equivalent_key(attrs["fused_producer_key"])
    if (
        alt is not None
        and misc.load_index()[alt]["shape"]
        and [
            d
            for i, d in enumerate(misc.load_index()[alt]["shape"])
            if i not in misc.load_index()[alt]["axes"]
        ]
        == list(attrs.get("out", []))
    ):
        return alt
    return None


def _plan_misc(rec: Mapping) -> tuple[str, str] | None:
    """``misc_op_record_emit`` templates (step-shape ReduceSum, Greater/Less
    -> Cast, Sqrt ``[512,512,3,3]``, Softmax, Log, MaxPool, ReduceMean), or
    ``None`` without one."""
    key = rec.get("attrs", {}).get("misc_key")
    meta = misc.load_index().get(key) if key else None
    alt = misc.equivalent_key(key) if key else None
    if meta is None and alt:
        return (
            "conditional",
            f"same-bytes equivalent template {alt} (Pulsar2 cannot tile the node "
            "as written) retargeted if zp_x != 0",
        )
    if meta is None:
        return None
    if meta["op"] in misc.CALIBRATION_FREE:
        return ("covered", "TemplateOnly (not quantized; misc_op_record_emit)")
    if meta["op"] == "Neg":
        return (
            "conditional",
            "misc_op_record_emit retarget (any zp_x, zp_y = 255 - zp_x; the "
            "scale picks one of two program templates, split at s = 1/64) if "
            "s_y = s_x",
        )
    fixed = {
        "ReduceSum": "",
        "Softmax": f" and zp_y = {meta['zero_points']['y']}",
    }.get(meta["op"], f" and zero points = {meta['zero_points']}")
    return (
        "conditional",
        f"misc_op_record_emit retarget if zp_x != 0{fixed} and the scale "
        "formulas stay distinct",
    )


def key_for_record(
    rec: Mapping, calibration_class: str = "", weight_dtype: str = ""
) -> TemplateKey:
    attrs = rec.get("attrs", {})
    shapes = rec["shapes"]
    if rec["op"] in bse.OPS and attrs.get("output_shape"):
        shapes = [attrs["output_shape"]]
    keep: dict[str, Any] = {}
    if rec["op"] == "Transpose":
        keep["perm"] = tuple(attrs["perm"])
    elif rec["op"] == "Gather":
        keep["axis"] = attrs["axis"]
        keep["indices"] = attrs["indices"]
    elif rec["op"] == "Conv":
        keep["w"] = tuple(attrs["w"])
        keep["strides"] = tuple(attrs["strides"])
        keep["pads"] = tuple(attrs["pads"])
    return TemplateKey(
        op=rec["op"],
        shapes=tuple(_tup(s) for s in shapes),
        attrs=tuple(sorted(keep.items())),
        calibration_class=calibration_class,
        weight_dtype=weight_dtype,
    )


def plan_node(rec: Mapping, cache: TemplateCache | None = None) -> tuple[str, str]:
    """``(status, detail)`` for one graph node. ``status`` is ``"covered"`` (a
    template plus an edit reproduce it), ``"conditional"`` (covered only if a
    calibration property holds that the graph alone does not show), or
    ``"refused"``."""
    cache = cache or TemplateCache()
    op = rec["op"]
    attrs = rec.get("attrs", {})
    try:
        live = mre.step_manifest()["nodes"].get(rec.get("name", ""))
        if live is not None and (
            op in ("MatMul", "Gemm")
            or (op == "Conv" and attrs.get("weight_is_graph_input", True))
        ):
            # Live operands: no weight table to edit, only calibration records
            # (matmul_record_emit.py). A live-weight Conv is served in its
            # act_weight_conv_to_matmul form, the Gemm in its gemm_to_matmul one.
            return (
                "conditional",
                f"MatMul recalibration from scales ({live['template']}) if no "
                "zero point crosses between zero and nonzero vs the template",
            )
        if op == "Conv":
            key = key_for_record(rec)
            cache.lookup(key)
            if attrs.get("weight_is_graph_input"):
                return (
                    "refused",
                    "trainable weight is a graph input (runtime state); a template "
                    "exists for frozen-weight deployment only",
                )
            return ("covered", "ConvWeightEdit")
        if op in (
            "ReduceSum",
            "Greater",
            "Less",
            "Cast",
            "Softmax",
            "Log",
            "MaxPool",
            "ReduceMean",
            "Neg",
        ):
            return _plan_misc(rec) or (
                "refused",
                f"no misc_op_record_emit template for {op}",
            )
        if op in ew.OPS:
            hits = []
            for zp in _ELEMENTWISE_ZP_CLASSES:
                try:
                    cache.lookup(key_for_record(rec, zp))
                    hits.append(zp)
                except ValueError:
                    pass
            if not hits:
                planned = _plan_misc(rec)
                if planned:
                    return planned
                raise ValueError("no template at this shape")
            return (
                "conditional",
                f"ElementwiseScaleEdit if zero points are one of {hits}",
            )
        if op in bse.OPS:
            form = attrs.get("form")
            if form != "same_shape":
                return (
                    "refused",
                    f"{op} with a {form} operand compiles to a different program; "
                    "templates exist for two live same-shape inputs only",
                )
            hits = []
            for zp in _BINARY_ZP_CLASSES[op]:
                try:
                    cache.lookup(key_for_record(rec, zp))
                    hits.append(zp)
                except ValueError:
                    pass
            if not hits:
                raise ValueError("no template at this shape")
            return (
                "conditional",
                f"ElementwiseScaleEdit if zero points are one of {hits}",
            )
        if op == "Gather":
            cache.lookup(key_for_record(rec))
            if attrs.get("indices", -1) < 0:
                raise ValueError("indices are not a constant initializer")
            return ("covered", "GatherIndexEdit")
        if op == "Transpose":
            cache.lookup(key_for_record(rec))
            return ("covered", "TemplateOnly")
        if op == "Squeeze":
            # a Squeeze is the Reshape to its output shape; it takes the same
            # step template (a standalone Squeeze trips Pulsar2's scheduler)
            shape = rec["shapes"][0] if rec["shapes"] else []
            try:
                rre.step_template(shape, attrs.get("out", []))
            except ValueError:
                return ("refused", "Squeeze: no validated Reshape step template")
            return (
                "conditional",
                "Reshape step template (Squeeze as Reshape) retargeted to the "
                "calibration if its zero point is nonzero",
            )
        if op == "Reshape":
            shape = rec["shapes"][0] if rec["shapes"] else []
            out = attrs.get("out", [])
            if len(shape) == 2 and shape[0] == 1 and out == shape[1:]:
                chain = _fused_reducesum_chain(rec)
                if chain is not None:
                    return (
                        "conditional",
                        f"bias flatten fused with its ReduceSum: chain template "
                        f"{chain} retargeted (misc_op_record_emit) if zp_x != 0",
                    )
                return (
                    "conditional",
                    "bias flatten fuses into its neighbour (reshape_emit.py, "
                    "Relu-neighbour pairs only)",
                )
            forms = []
            for form, find in (
                ("Identity", rre.step_template_identity),
                ("zero-point-0", rre.step_template_zp0),
                ("Relu", rre.step_template),
            ):
                try:
                    find(shape, out)
                    forms.append(form)
                except ValueError:
                    pass
            if not forms:
                return ("refused", "non-fused Reshape: no validated step template")
            # reshape_record_emit.retarget_scale: scale lanes and zero point;
            # a zero point of 0 compiles to a different program, and a nonzero
            # one (a signed input) needs the Identity form: the Relu form clips
            return (
                "conditional",
                f"Reshape step templates ({', '.join(forms)}): the Identity one "
                "retargeted if the zero point is nonzero, the zero-point-0 one "
                "if it is 0",
            )
        return ("refused", f"no template or edit for {op}")
    except ValueError as exc:
        return ("refused", str(exc))


# --------------------------------------------------------------------------
# A real calibration (step_calibration.py) settles "conditional" nodes.
# --------------------------------------------------------------------------


class _NotAtCalibration(ValueError):
    pass


def _tensor_q(calib: Mapping, name: str) -> Mapping:
    q = calib["tensors"].get(name)
    if q is None:
        raise _NotAtCalibration(f"no calibration for tensor {name!r}")
    return q


def _u8_zp(calib: Mapping, name: str) -> int:
    q = _tensor_q(calib, name)
    if q["signed"]:
        raise _NotAtCalibration(
            f"{name} is symmetric int8 (it only feeds MatMuls); templates are uint8"
        )
    return int(q["zero_point"])


def _class_hits(
    rec: Mapping, classes: Sequence[str], cache: TemplateCache
) -> list[str]:
    hits = []
    for zp in classes:
        try:
            cache.lookup(key_for_record(rec, zp))
            hits.append(zp)
        except ValueError:
            pass
    return hits


def _misc_accepts(meta: Mapping, zx: int, zy: int) -> str | None:
    """Why template ``meta`` cannot take zero points ``(zx, zy)``, or None.
    What each op's ``misc_op_record_emit.retarget`` can move is measured:
    ReduceSum both zero points (while zp_x stays nonzero; a zero-point-0
    template keeps its own), Softmax zp_x, MaxPool its one shared nonzero
    zero point, Neg any zp_x with zp_y = 255 - zp_x; every other zero point is
    fixed by the template."""
    op, fixed = meta["op"], dict(meta.get("zero_points") or {})
    if op == "ReduceSum" and fixed.get("x", 1) != 0:
        return None if zx != 0 else "zp_x = 0 (a different program)"
    if op == "Softmax":
        if zx == 0:
            return "zp_x = 0 (unmeasured)"
        return None if zy == fixed["y"] else f"zp_y {zy} != template {fixed['y']}"
    if op == "MaxPool" and fixed.get("x", 0) != 0:
        return None if zx == zy != 0 else f"x{zx},y{zy} is not one nonzero zero point"
    if op == "Neg":
        return None if zy == 255 - zx else f"zp_y {zy} != 255 - zp_x {zx}"
    if {"x": zx, "y": zy} != fixed:
        return f"zero points x{zx},y{zy} are fixed by the template at {fixed}"
    return None


def _at_calibration_misc(rec: Mapping, calib: Mapping) -> str:
    index = misc.load_index()
    base = rec["attrs"]["misc_key"]
    if base not in index:
        # ReduceSum_474: served by a same-bytes equivalent template
        base = misc.equivalent_key(base)
    zx = _u8_zp(calib, rec["inputs"][0])
    zy = _u8_zp(calib, rec["outputs"][0])
    keys = [base] + sorted(k for k, v in index.items() if v.get("variant_of") == base)
    if rec["op"] == "Neg":
        # The input scale picks one of Neg's two program templates.
        want = misc.neg_program(float(_tensor_q(calib, rec["inputs"][0])["scale"]))
        keys = [
            k
            for k in [base, *index[base].get("programs", {}).values()]
            if index[k].get("program") == want
        ]
    why = []
    for key in keys:
        reason = _misc_accepts(index[key], zx, zy)
        if reason is None:
            return f"misc_op_record_emit retarget of {key} (zero points x{zx},y{zy})"
        why.append(f"{key}: {reason}")
    raise _NotAtCalibration("; ".join(why))


def _at_calibration_matmul(rec: Mapping, calib: Mapping) -> str:
    """Actually recalibrate the node's template onto the predicted scales."""
    entry = mre.step_template(rec["name"])
    old = mre.load_scales(entry["quant"])
    real: dict[str, tuple[float, float]] = {}
    for step_name, name in entry["names"].items():
        q = _tensor_q(calib, step_name)
        if "consumer_int8_scale" in q and name in old and old[name][1] == 0:
            # the MatMul's own symmetric requantization of a mixed-use tensor
            real[step_name] = (q["consumer_int8_scale"], 0.0)
        else:
            real[step_name] = (q["scale"], float(q["zero_point"]))
        if "consumer_int8_scale" in q and name + mre.I8 in old:
            # a uint8 tensor the MatMul requantizes: its int8 view
            real[step_name + mre.I8] = (q["consumer_int8_scale"], 0.0)
    try:
        new = mre.step_node_scales(entry, old, real)
        mre.recalibrate(mre.load_model(entry["axmodel"]), old, new)
    except mre.CalibrationError as exc:
        raise _NotAtCalibration(str(exc)) from exc
    return f"matmul_record_emit.recalibrate onto the scales ({entry['template']})"


def plan_at_calibration(
    rec: Mapping,
    calib: Mapping,
    cache: TemplateCache | None = None,
    *,
    validate_live: bool = True,
) -> tuple[str, str]:
    """``plan_node`` with ``"conditional"`` settled against a real calibration
    (``step_calibration.calibrate``): ``"covered"`` when the node's predicted
    zero points fall in its template's class (for live-operand MatMuls, when
    ``recalibrate`` succeeds on the predicted scales), else ``"refused"``."""
    cache = cache or TemplateCache()
    status, detail = plan_node(rec, cache)
    if (
        status == "refused"
        and rec.get("attrs", {}).get("form") == "broadcast"
        and all(name in calib.get("tensors", {}) for name in rec.get("inputs", ())[:2])
    ):
        # A broadcast of live tensors can use the measured full-shape binary
        # program after the smaller operand is expanded at the segment edge.
        expanded = dict(rec)
        expanded["shapes"] = [list(rec["shapes"][0])]
        expanded["attrs"] = dict(rec.get("attrs", {}))
        expanded["attrs"].update(
            {"form": "same_shape", "output_shape": list(rec["shapes"][0])}
        )
        status, detail = plan_node(expanded, cache)
        if status == "conditional":
            rec = expanded
    if status != "conditional":
        return status, detail
    op, attrs = rec["op"], rec.get("attrs", {})
    try:
        live = mre.step_manifest()["nodes"].get(rec.get("name", ""))
        if live is not None and op in ("MatMul", "Gemm", "Conv"):
            if not validate_live:
                return "covered", (
                    "matmul_record_emit.recalibrate deferred to segment emit "
                    f"({live['template']})"
                )
            return "covered", _at_calibration_matmul(rec, calib)
        key = attrs.get("misc_key")
        if key and (misc.load_index().get(key) or misc.equivalent_key(key)):
            if op not in ew.OPS or not _class_hits(rec, _ELEMENTWISE_ZP_CLASSES, cache):
                return "covered", _at_calibration_misc(rec, calib)
        if op in ew.OPS:
            zx = _u8_zp(calib, rec["inputs"][0])
            zy = _u8_zp(calib, rec["outputs"][0])
            cls = f"x{zx},y{zy}"
            hits = _class_hits(rec, _ELEMENTWISE_ZP_CLASSES, cache)
            if op == "Relu" and cls not in hits and zx == zy != 0:
                if "x128,y128" in hits:
                    # ew.retarget_relu_records: the zero point is a whole word
                    return "covered", f"Relu record retarget from x128,y128 ({cls})"
                raise _NotAtCalibration(
                    f"zero points {cls} are not a template class {hits}"
                )
            return "covered", f"ElementwiseScaleEdit ({cls})"
        if op in bse.OPS:
            zx = _u8_zp(calib, rec["inputs"][0])
            zz = _u8_zp(calib, rec["inputs"][1])
            zy = _u8_zp(calib, rec["outputs"][0])
            cls = f"x{zx},y{zy},z{zz}"
            hits = _class_hits(rec, _BINARY_ZP_CLASSES[op], cache)
            if cls not in hits:
                raise _NotAtCalibration(
                    f"zero points {cls} are not a template class {hits}"
                )
            return "covered", f"ElementwiseScaleEdit ({cls})"
        chain = _fused_reducesum_chain(rec) if op == "Reshape" else None
        if chain is not None:
            # the chain's input is the ReduceSum's; its output this Reshape's
            # (a Reshape shares its input's quantization)
            zx = _u8_zp(calib, rec["attrs"]["fused_input"])
            zy = _u8_zp(calib, rec["outputs"][0])
            reason = _misc_accepts(misc.load_index()[chain], zx, zy)
            if reason is not None:
                raise _NotAtCalibration(f"{chain}: {reason}")
            return "covered", (
                f"misc_op_record_emit retarget of the fused chain {chain} "
                f"(zero points x{zx},y{zy})"
            )
        if op in ("Reshape", "Squeeze") and "bias flatten" not in detail:
            zp = _u8_zp(calib, rec["inputs"][0])
            if zp != 0:
                # a nonzero zero point is a signed input: only the Identity
                # form keeps it (a Reshape -> Relu template clips it)
                try:
                    rre.step_template_identity(rec["shapes"][0], attrs.get("out", []))
                except ValueError as exc:
                    raise _NotAtCalibration(
                        f"signed input (zp {zp}) and no Reshape -> Identity "
                        "template serves this shape"
                    ) from exc
            if zp == 0:
                try:
                    rre.step_template_zp0(rec["shapes"][0], attrs.get("out", []))
                except ValueError as exc:
                    raise _NotAtCalibration(
                        "Reshape zero point is 0 and no zero-point-0 template "
                        "serves this shape"
                    ) from exc
                return "covered", "reshape_record_emit zero-point-0 template"
            return "covered", (
                f"reshape_record_emit.retarget_scale of the Identity template (zp {zp})"
            )
    except _NotAtCalibration as exc:
        return "refused", f"at calibration: {exc}"
    return status, detail


def chain_internal_nodes() -> dict[str, set[str]]:
    """``{step tensor: {MatMul node}}``: tensors a live-operand MatMul step
    template computes inside its own chain (its Gather/Mul/Reshape/Transpose
    ops), i.e. every chain tensor that is not a template graph input."""
    out: dict[str, set[str]] = {}
    for node in mre.step_manifest()["nodes"]:
        entry = mre.step_template(node)
        inputs = {i.name for i in mre.load_model(entry["axmodel"]).graph.input}
        for step_name, name in entry["names"].items():
            if name not in inputs:
                out.setdefault(step_name, set()).add(node)
    return out


def _absorb_into_chains(
    records: Sequence[Mapping], plans: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Nodes computed inside a MatMul chain template that is itself covered
    are covered by that template (their own template, if any, is moot)."""
    by_name = {r.get("name"): p for r, p in zip(records, plans)}
    internal = chain_internal_nodes()
    out = []
    for rec, (status, detail) in zip(records, plans):
        if status != "covered" and rec["op"] not in ("MatMul", "Gemm", "Conv"):
            chains = sorted(
                c
                for t in rec.get("outputs", [])
                for c in internal.get(t, ())
                if by_name.get(c, ("",))[0] == "covered"
            )
            if chains:
                status, detail = (
                    "covered",
                    f"computed inside the {chains[0]} chain template",
                )
        out.append((status, detail))
    return out


def load_calibration(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def coverage_report(
    records: Sequence[Mapping],
    policy: QuantPolicy | None = None,
    calibration: Mapping | None = None,
) -> dict:
    """Per-op counts of covered / conditional / refused nodes, with reasons.
    With a ``policy``, also each node's weight dtype (``per_node``) and a
    count of those outcomes per op (``weight_dtypes``). With a
    ``calibration`` (``step_calibration.calibrate``), conditional nodes are
    settled against it (``plan_at_calibration``); records then need their
    ``inputs``/``outputs`` tensor names (``extract_step_ops``)."""
    cache = TemplateCache()
    per_op: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    per_node = []
    wd: dict[str, Counter] = defaultdict(Counter)
    plans = []
    for rec in records:
        if calibration is not None:
            plans.append(plan_at_calibration(rec, calibration, cache))
        else:
            plans.append(plan_node(rec, cache))
    if calibration is not None:
        plans = _absorb_into_chains(records, plans)
    for i, rec in enumerate(records):
        status, detail = plans[i]
        per_op[rec["op"]][status] += 1
        reasons[rec["op"]][f"{status}: {detail}"] += 1
        if policy is not None:
            choice = weight_dtype_for_record(rec, policy)
            per_node.append(
                {
                    "index": i,
                    "op": rec["op"],
                    "name": rec.get("name", ""),
                    "status": status,
                    **choice,
                }
            )
            if rec["op"] in _WEIGHT_OPS:
                wd[rec["op"]][f"{choice['weight_dtype']}: {choice['note']}"] += 1
    totals = Counter()
    for c in per_op.values():
        totals.update(c)
    report = {
        "nodes": len(records),
        "totals": dict(totals),
        "per_op": {op: dict(c) for op, c in sorted(per_op.items())},
        "reasons": {op: dict(c) for op, c in sorted(reasons.items())},
    }
    if policy is not None:
        report["policy"] = policy.to_json()
        report["weight_dtypes"] = {op: dict(c) for op, c in sorted(wd.items())}
        report["per_node"] = per_node
    return report


# --------------------------------------------------------------------------
# tinygrad seam (fork pinned above). Imported lazily so the rest of this
# module works without tinygrad installed.
# --------------------------------------------------------------------------


def build_request(
    key: TemplateKey, edits: Sequence[Edit], node: str | None = None
) -> str:
    """The "source" ``AXCompiler`` compiles: a JSON template request. This is
    what a graph/JIT-level hook would emit for one fused subgraph (plan section
    5: the compile unit is the fused subgraph, not a per-kernel renderer)."""
    req = {"key": key.to_json(), "edits": [e.to_json() for e in edits]}
    if node is not None:
        req["node"] = node
    return json.dumps(req, sort_keys=True)


def build_graph_template_request(
    source_path: str,
    template_source_path: str,
    template_axmodel_path: str,
    output_path: str | None = None,
) -> str:
    """Build the JSON source accepted by ``AXCompiler`` for graph reuse."""
    req = {
        "kind": "graph_template",
        "source": source_path,
        "template_source": template_source_path,
        "template_axmodel": template_axmodel_path,
    }
    if output_path is not None:
        req["output"] = output_path
    return json.dumps(req, sort_keys=True)


def build_generated_graph_request(
    source_path: str,
    output_path: str | None = None,
    schedule_path: str | None = None,
) -> str:
    """Build the JSON request for a measured graph generator family.

    Unlike ``build_graph_template_request``, this request does not need a
    source/template AX pair: ``graph_generator`` selects a validated fused
    family and emits its model plus optional scheduler sidecar directly.
    """
    req = {"kind": "generated_graph", "source": source_path}
    if output_path is not None:
        req["output"] = output_path
    if schedule_path is not None:
        req["schedule"] = schedule_path
    return json.dumps(req, sort_keys=True)


def build_onnx_uop_request(
    source_path: str,
    output_path: str | None = None,
    schedule_path: str | None = None,
    calibration: Mapping[str, Mapping[str, float | int]] | None = None,
) -> str:
    """Build an ``AXCompiler`` request for ONNX -> tinygrad UOp -> mcode."""
    req = {"kind": "onnx_uop", "source": source_path}
    if output_path is not None:
        req["output"] = output_path
    if schedule_path is not None:
        req["schedule"] = schedule_path
    if calibration is not None:
        req["calibration"] = calibration
    return json.dumps(req, sort_keys=True)


def lower_uop_to_onnx(root) -> onnx.ModelProto:
    """Lower one statically shaped tinygrad UOp pattern to ordinary ONNX.

    The supported patterns are tinygrad's canonical Relu lowering,
    ``WHERE(CMPLT(0, x), x, 0)``, fused with either ``Relu(Reshape(x))`` or
    ``Reshape(Relu(x))``. Both orderings are measured by the Pulsar-free
    generator. Arbitrary movement/reduction UOps still need dedicated AX
    templates or MCode semantics.
    """
    from tinygrad.uop.ops import Ops

    def alloc_backed_view(node):
        """Whether a reshape/permute view ultimately reads one allocation."""
        if node.op is Ops.ALLOC:
            return True
        return bool(
            node.src
            and node.op in (Ops.RESHAPE, Ops.PERMUTE)
            and alloc_backed_view(node.src[0])
        )

    def lower_matmul(node):
        bias = None
        if node.op is Ops.ADD and len(node.src) == 2:
            candidate, other = node.src
            if candidate.op is Ops.REDUCE and other.op is Ops.ALLOC:
                node, bias = candidate, other
            elif other.op is Ops.REDUCE and candidate.op is Ops.ALLOC:
                node, bias = other, candidate
        if node.op is not Ops.REDUCE or node.arg[0] is not Ops.ADD or node.arg[1] != 1:
            return None
        if not node.src or node.src[0].op is not Ops.PERMUTE:
            return None
        output_permute = node.src[0]
        if not output_permute.src:
            return None
        product = output_permute.src[0]
        if product.op is not Ops.MUL or len(product.src) != 2:
            return None
        left, right = product.src
        if (
            left.op is not Ops.RESHAPE
            or not left.src
            or not alloc_backed_view(left.src[0])
        ):
            return None
        if (
            right.op is not Ops.PERMUTE
            or not right.src
            or right.src[0].op is not Ops.RESHAPE
            or not right.src[0].src
            or not alloc_backed_view(right.src[0].src[0])
        ):
            return None
        a_shape = tuple(int(dim) for dim in left.src[0].shape)
        b_shape = tuple(int(dim) for dim in right.src[0].src[0].shape)
        output_shape = tuple(int(dim) for dim in node.shape)
        if len(a_shape) < 2 or len(b_shape) < 2:
            return None
        batch_shape = a_shape[:-2]
        if b_shape[:-2] not in ((), batch_shape):
            return None
        if b_shape[-2] != a_shape[-1] or output_shape != batch_shape + (
            a_shape[-2],
            b_shape[-1],
        ):
            return None
        expected_left = batch_shape + (a_shape[-2], 1, a_shape[-1])
        right_batch = b_shape[:-2]
        expected_right = right_batch + (1, b_shape[-2], b_shape[-1])
        expected_output_permute = (len(batch_shape) + 2,) + tuple(
            range(len(batch_shape) + 2)
        )
        expected_right_permute = tuple(range(len(right_batch) + 1)) + (
            len(right_batch) + 2,
            len(right_batch) + 1,
        )
        if (
            tuple(int(dim) for dim in left.shape) != expected_left
            or tuple(int(axis) for axis in right.arg) != expected_right_permute
            or tuple(int(dim) for dim in right.src[0].shape) != expected_right
            or tuple(int(axis) for axis in output_permute.arg)
            != expected_output_permute
            or str(node.dtype).split(".")[-1] != "float"
        ):
            return None
        op_type = (
            "Gemm"
            if bias is not None and len(a_shape) == len(b_shape) == 2
            else "MatMul"
        )
        graph_inputs = [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, a_shape),
            onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, b_shape),
        ]
        inputs = ["x", "z"]
        if bias is not None:
            if tuple(int(dim) for dim in bias.shape) != (b_shape[-1],):
                return None
            graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "b", onnx.TensorProto.FLOAT, (b_shape[-1],)
                )
            )
            inputs.append("b")
        nodes = [
            onnx.helper.make_node(
                op_type,
                inputs,
                ["matmul_y"] if op_type == "MatMul" and bias is not None else ["y"],
            )
        ]
        if op_type == "MatMul" and bias is not None:
            nodes.append(onnx.helper.make_node("Add", ["matmul_y", "b"], ["y"]))
        graph = onnx.helper.make_graph(
            nodes,
            "tinygrad_uop_matmul_ax",
            graph_inputs,
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        )
        return onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
        )

    matmul_model = lower_matmul(root)
    if matmul_model is not None:
        return matmul_model

    def walk(node):
        yield node
        for child in node.src:
            yield from walk(child)

    reductions = [node for node in walk(root) if node.op is Ops.REDUCE]
    if len(reductions) == 1 and reductions[0].arg[0] is Ops.ADD:
        reduction = reductions[0]
        output_shape = tuple(int(dim) for dim in root.shape)
        reduction_shape = tuple(int(dim) for dim in reduction.shape)
        if len(output_shape) == 4 and len(reduction_shape) >= 5:
            alloc_reshapes = [
                node
                for node in walk(reduction)
                if node.op is Ops.RESHAPE
                and node.src
                and node.src[0].op is Ops.ALLOC
                and len(node.shape) == 4
            ]
            batch = output_shape[0]
            input_candidates = [
                tuple(int(dim) for dim in node.shape)
                for node in alloc_reshapes
                if int(node.shape[0]) == batch
            ]
            weight_candidates = [
                tuple(int(dim) for dim in node.shape)
                for node in alloc_reshapes
                if int(node.shape[0]) == output_shape[1]
            ]
            if len(input_candidates) == 1 and len(weight_candidates) == 1:
                input_shape = input_candidates[0]
                weight_shape = weight_candidates[0]
                if (
                    input_shape[1] % weight_shape[1] != 0
                    or output_shape[1] % (input_shape[1] // weight_shape[1]) != 0
                ):
                    return None
                groups = input_shape[1] // weight_shape[1]
                _, _, kernel_h, kernel_w = weight_shape
                stride = None
                pad_h = pad_w = None
                dilation = None
                pad_shapes = [
                    tuple(int(dim) for dim in node.shape)
                    for node in walk(reduction)
                    if node.op is Ops.PAD and len(node.shape) == 4
                ]
                fixed_pads = None
                for shape in pad_shapes:
                    if (
                        shape[:2] == input_shape[:2]
                        and shape[2] >= input_shape[2]
                        and shape[3] >= input_shape[3]
                    ):
                        fixed_pads = (
                            (shape[2] - input_shape[2]) // 2,
                            (shape[3] - input_shape[3]) // 2,
                        )
                        break
                for candidate_dilation in range(1, max(weight_shape[2:]) + 1):
                    candidate_pads = [fixed_pads] if fixed_pads is not None else [None]
                    if fixed_pads is None:
                        candidate_pads = [
                            (ph, pw)
                            for ph in range(input_shape[2] + kernel_h)
                            for pw in range(input_shape[3] + kernel_w)
                        ]
                    for candidate_pad in candidate_pads:
                        ph, pw = candidate_pad or (0, 0)
                        if fixed_pads is None:
                            if (
                                input_shape[2]
                                + 2 * ph
                                - candidate_dilation * (kernel_h - 1)
                                - 1
                            ) < 0 or (
                                input_shape[3]
                                + 2 * pw
                                - candidate_dilation * (kernel_w - 1)
                                - 1
                            ) < 0:
                                continue
                        for candidate_stride in range(1, max(input_shape[2:]) + 1):
                            if (
                                input_shape[2]
                                + 2 * ph
                                - candidate_dilation * (kernel_h - 1)
                                - 1
                            ) // candidate_stride + 1 == output_shape[2] and (
                                input_shape[3]
                                + 2 * pw
                                - candidate_dilation * (kernel_w - 1)
                                - 1
                            ) // candidate_stride + 1 == output_shape[3]:
                                stride, pad_h, pad_w, dilation = (
                                    candidate_stride,
                                    ph,
                                    pw,
                                    candidate_dilation,
                                )
                                break
                        if stride is not None:
                            break
                    if stride is not None:
                        break
                if stride is not None and str(root.dtype).split(".")[-1] == "float":
                    graph = onnx.helper.make_graph(
                        [
                            onnx.helper.make_node(
                                "Conv",
                                ["x", "w"],
                                ["y"],
                                strides=[stride, stride],
                                pads=[pad_h, pad_w, pad_h, pad_w],
                                **(
                                    {"dilations": [dilation, dilation]}
                                    if dilation != 1
                                    else {}
                                ),
                                **({"group": groups} if groups != 1 else {}),
                            )
                        ],
                        "tinygrad_uop_conv_ax",
                        [
                            onnx.helper.make_tensor_value_info(
                                "x", onnx.TensorProto.FLOAT, input_shape
                            ),
                            onnx.helper.make_tensor_value_info(
                                "w", onnx.TensorProto.FLOAT, weight_shape
                            ),
                        ],
                        [
                            onnx.helper.make_tensor_value_info(
                                "y", onnx.TensorProto.FLOAT, output_shape
                            )
                        ],
                    )
                    return onnx.helper.make_model(
                        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                    )

    if root.op is Ops.PERMUTE and root.src:
        source = root.src[0] if root.src else None
        shape = tuple(int(dim) for dim in root.shape)
        if (
            source is not None
            and source.op is Ops.RESHAPE
            and source.src
            and source.src[0].op is Ops.ALLOC
        ):
            input_shape = tuple(int(dim) for dim in source.shape)
            perm = tuple(int(axis) for axis in root.arg)
            try:
                transpose_real_shapes.template_path(input_shape, perm)
            except ValueError:
                pass
            else:
                if shape != tuple(input_shape[axis] for axis in perm):
                    raise ValueError(
                        "AX UOp Transpose output shape does not match its permutation"
                    )
                if str(root.dtype).split(".")[-1] != "float":
                    raise ValueError(
                        "AX UOp Transpose lowering currently supports float32 data only"
                    )
                graph = onnx.helper.make_graph(
                    [onnx.helper.make_node("Transpose", ["x"], ["y"], perm=list(perm))],
                    "tinygrad_uop_transpose_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, input_shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, shape
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.RESHAPE and root.src:
        reduction = root.src[0]
        if (
            reduction.op is Ops.REDUCE
            and reduction.arg[0] is Ops.ADD
            and reduction.arg[1] == 1
            and len(reduction.src) == 1
        ):
            reduced = reduction.src[0]
            if (
                reduced.op is Ops.RESHAPE
                and reduced.src
                and reduced.src[0].op is Ops.ALLOC
            ):
                axis, input_shape = 0, tuple(int(dim) for dim in reduced.shape)
            elif (
                reduced.op is Ops.PERMUTE
                and tuple(int(axis) for axis in reduced.arg) == (1, 0)
                and reduced.src
                and reduced.src[0].op is Ops.RESHAPE
                and reduced.src[0].src
                and reduced.src[0].src[0].op is Ops.ALLOC
            ):
                axis, input_shape = 1, tuple(int(dim) for dim in reduced.src[0].shape)
            else:
                axis, input_shape = None, ()
            output_shape = tuple(int(dim) for dim in root.shape)
            if (
                axis is not None
                and input_shape == (16, 1000)
                and output_shape == ((1, 1000) if axis == 0 else (16, 1))
            ):
                if str(root.dtype).split(".")[-1] != "float":
                    raise ValueError(
                        "AX UOp ReduceSum lowering currently supports float32 data only"
                    )
                graph = onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "ReduceSum", ["x"], ["y"], axes=[axis], keepdims=1
                        )
                    ],
                    "tinygrad_uop_reducesum_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, input_shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, output_shape
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.CAST and len(root.src) == 1:
        comparison = root.src[0]
        if comparison.op is Ops.CMPLT and len(comparison.src) == 2:
            left, right = comparison.src
            if left.op is Ops.CONST and float(left.arg) == 0.0:
                comparison_op, data = "Greater", right
            elif right.op is Ops.CONST and float(right.arg) == 0.0:
                comparison_op, data = "Less", left
            else:
                comparison_op = None
                data = None
            if comparison_op is not None:
                shape = tuple(int(dim) for dim in root.shape)
                if (
                    data.op is not Ops.RESHAPE
                    or not data.src
                    or data.src[0].op is not Ops.ALLOC
                    or shape not in ((16, 64, 112, 112), (1024, 9, 3136))
                ):
                    raise ValueError(
                        "AX UOp comparison lowering is measured only for "
                        "one of the measured ALLOC-backed input shapes"
                    )
                if str(root.dtype).split(".")[-1] != "float":
                    raise ValueError(
                        "AX UOp comparison lowering currently supports float32 casts only"
                    )
                graph = onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            comparison_op, ["x", "zero"], ["comparison"]
                        ),
                        onnx.helper.make_node("Cast", ["comparison"], ["y"], to=1),
                    ],
                    f"tinygrad_uop_{comparison_op.lower()}cast_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, shape
                        )
                    ],
                    [
                        onnx.numpy_helper.from_array(
                            np.asarray(0.0, dtype=np.float32), "zero"
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.REDUCE and root.arg[0] is Ops.ADD and len(root.src) == 1:
        permuted = root.src[0]
        if (
            permuted.op is Ops.PERMUTE
            and tuple(int(axis) for axis in permuted.arg) == (0, 2, 3, 1)
            and len(permuted.src) == 1
            and permuted.src[0].op is Ops.RESHAPE
            and permuted.src[0].src
            and permuted.src[0].src[0].op is Ops.ALLOC
        ):
            input_shape = tuple(int(dim) for dim in permuted.src[0].shape)
            output_shape = tuple(int(dim) for dim in root.shape)
            if input_shape != (16, 64, 112, 112) or output_shape != (64,):
                raise ValueError(
                    "AX UOp ReduceSum lowering is measured only for "
                    "[16,64,112,112] -> [64]"
                )
            if str(root.dtype).split(".")[-1] != "float":
                raise ValueError(
                    "AX UOp ReduceSum lowering currently supports float32 data only"
                )
            graph = onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "ReduceSum", ["x"], ["y"], axes=[0, 2, 3], keepdims=0
                    )
                ],
                "tinygrad_uop_reducesum_ax",
                [
                    onnx.helper.make_tensor_value_info(
                        "x", onnx.TensorProto.FLOAT, input_shape
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "y", onnx.TensorProto.FLOAT, output_shape
                    )
                ],
            )
            return onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
            )

    if root.op is Ops.REDUCE and root.arg[0] is Ops.ADD and len(root.src) == 1:
        permuted = root.src[0]
        if (
            permuted.op is Ops.PERMUTE
            and tuple(int(axis) for axis in permuted.arg) == (0, 3, 1, 2)
            and len(permuted.src) == 1
            and permuted.src[0].op is Ops.RESHAPE
            and permuted.src[0].src
            and permuted.src[0].src[0].op is Ops.ALLOC
        ):
            input_shape = tuple(int(dim) for dim in permuted.src[0].shape)
            output_shape = tuple(int(dim) for dim in root.shape)
            if input_shape not in (
                (16, 1, 64, 3136),
                (16, 1, 512, 49),
            ) or output_shape != (
                (1, 64) if input_shape == (16, 1, 64, 3136) else (1, 512)
            ):
                raise ValueError(
                    "AX UOp ReduceSum lowering requires a measured "
                    "[16,1,C,S] -> [1,C] form"
                )
            if str(root.dtype).split(".")[-1] != "float":
                raise ValueError(
                    "AX UOp ReduceSum lowering currently supports float32 data only"
                )
            graph = onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "ReduceSum", ["x"], ["y"], axes=[0, 3], keepdims=0
                    )
                ],
                "tinygrad_uop_reducesum_ax",
                [
                    onnx.helper.make_tensor_value_info(
                        "x", onnx.TensorProto.FLOAT, input_shape
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "y", onnx.TensorProto.FLOAT, output_shape
                    )
                ],
            )
            return onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
            )

    # Generic canonical tinygrad reduction lowering.  tinygrad moves reduced
    # dimensions to the front of a permuted view, so the first ``count``
    # entries of the permutation are the original ONNX axes.  A plain reshape
    # is the fast path for a leading-axis reduction.
    reduction_root = root
    keepdims = 0
    if root.op is Ops.RESHAPE and root.src and root.src[0].op is Ops.REDUCE:
        reduction_root = root.src[0]
        keepdims = 1
    if reduction_root.op is Ops.REDUCE and reduction_root.arg[0] is Ops.ADD:
        source = reduction_root.src[0] if reduction_root.src else None
        count = int(reduction_root.arg[1])
        axes = None
        input_shape = ()
        if source is not None and source.op is Ops.PERMUTE and source.src:
            base = source.src[0]
            if base.op is Ops.RESHAPE and base.src and base.src[0].op is Ops.ALLOC:
                axes = tuple(sorted(int(axis) for axis in source.arg[:count]))
                input_shape = tuple(int(dim) for dim in base.shape)
        elif source is not None and source.op is Ops.RESHAPE and source.src:
            if source.src[0].op is Ops.ALLOC:
                axes = tuple(range(count))
                input_shape = tuple(int(dim) for dim in source.shape)
        output_shape = tuple(int(dim) for dim in root.shape)
        if (
            axes is not None
            and input_shape
            and str(root.dtype).split(".")[-1] == "float"
        ):
            expected_output = (
                tuple(
                    1 if axis in axes else dim for axis, dim in enumerate(input_shape)
                )
                if keepdims
                else tuple(
                    dim for axis, dim in enumerate(input_shape) if axis not in axes
                )
            )
            if output_shape == expected_output:
                graph = onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "ReduceSum",
                            ["x"],
                            ["y"],
                            axes=list(axes),
                            keepdims=keepdims,
                        )
                    ],
                    "tinygrad_uop_reducesum_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, input_shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, output_shape
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.REDUCE and root.arg[0] is Ops.MAX and len(root.src) == 1:
        outer_permute = root.src[0]
        if (
            outer_permute.op is Ops.PERMUTE
            and tuple(int(axis) for axis in outer_permute.arg) == (4, 5, 0, 1, 2, 3)
            and len(outer_permute.src) == 1
        ):
            inner_permute = outer_permute.src[0]
            if (
                inner_permute.op is Ops.PERMUTE
                and tuple(int(axis) for axis in inner_permute.arg) == (0, 1, 3, 5, 2, 4)
                and len(inner_permute.src) == 1
                and inner_permute.src[0].op is Ops.RESHAPE
            ):
                windowed = inner_permute.src[0]
                input_shape = (16, 64, 112, 112)
                output_shape = tuple(int(dim) for dim in root.shape)
                if tuple(int(dim) for dim in windowed.shape) != (
                    16,
                    64,
                    3,
                    56,
                    3,
                    56,
                ) or output_shape != (16, 64, 56, 56):
                    raise ValueError(
                        "AX UOp MaxPool lowering is measured only for "
                        "[16,64,112,112] -> [16,64,56,56]"
                    )
                if str(root.dtype).split(".")[-1] != "float":
                    raise ValueError(
                        "AX UOp MaxPool lowering currently supports float32 data only"
                    )
                graph = onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "MaxPool",
                            ["x"],
                            ["y"],
                            kernel_shape=[3, 3],
                            strides=[2, 2],
                            pads=[1, 1, 1, 1],
                        )
                    ],
                    "tinygrad_uop_maxpool_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, input_shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, output_shape
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.MUL and len(root.src) == 2:
        reduced, reciprocal = root.src
        if (
            reduced.op is Ops.RESHAPE
            and reciprocal.op is Ops.RECIPROCAL
            and len(reciprocal.src) == 1
            and reciprocal.src[0].op is Ops.CONST
            and float(reciprocal.src[0].arg) == 49.0
            and reduced.src
            and reduced.src[0].op is Ops.REDUCE
        ):
            reduction = reduced.src[0]
            if (
                reduction.arg[0] is Ops.ADD
                and reduction.src
                and reduction.src[0].op is Ops.PERMUTE
                and reduction.src[0].src
                and reduction.src[0].src[0].op is Ops.RESHAPE
                and reduction.src[0].src[0].src
                and reduction.src[0].src[0].src[0].op is Ops.ALLOC
            ):
                input_shape = tuple(int(dim) for dim in reduction.src[0].src[0].shape)
                output_shape = tuple(int(dim) for dim in root.shape)
                if input_shape != (16, 512, 7, 7) or output_shape != (16, 512, 1, 1):
                    raise ValueError(
                        "AX UOp ReduceMean lowering is measured only for [16,512,7,7]"
                    )
                if str(root.dtype).split(".")[-1] != "float":
                    raise ValueError(
                        "AX UOp ReduceMean lowering currently supports float32 data only"
                    )
                graph = onnx.helper.make_graph(
                    [
                        onnx.helper.make_node(
                            "ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=1
                        )
                    ],
                    "tinygrad_uop_reducemean_ax",
                    [
                        onnx.helper.make_tensor_value_info(
                            "x", onnx.TensorProto.FLOAT, input_shape
                        )
                    ],
                    [
                        onnx.helper.make_tensor_value_info(
                            "y", onnx.TensorProto.FLOAT, output_shape
                        )
                    ],
                )
                return onnx.helper.make_model(
                    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
                )

    if root.op is Ops.MUL and len(root.src) == 2:
        exp2_node, reciprocal = root.src
        if (
            exp2_node.op is Ops.EXP2
            and len(exp2_node.src) == 1
            and reciprocal.op is Ops.RECIPROCAL
            and len(reciprocal.src) == 1
        ):
            exp2_mul = exp2_node.src[0]
            if exp2_mul.op is Ops.MUL and len(exp2_mul.src) == 2:
                centered, log2e = exp2_mul.src
                if (
                    log2e.op is Ops.CONST
                    and float(log2e.arg) == 1.4426950408889634
                    and centered.op is Ops.ADD
                    and len(centered.src) == 2
                ):
                    data, max_term = centered.src
                    if (
                        data.op is Ops.RESHAPE
                        and data.src
                        and data.src[0].op is Ops.ALLOC
                        and max_term.op is Ops.MUL
                        and len(max_term.src) == 2
                        and max_term.src[1].op is Ops.CONST
                        and float(max_term.src[1].arg) == -1.0
                    ):
                        max_detach = max_term.src[0]
                        if max_detach.op is Ops.DETACH and len(max_detach.src) == 1:
                            max_reshape = max_detach.src[0]
                            if max_reshape.op is Ops.RESHAPE and max_reshape.src:
                                max_reduce = max_reshape.src[0]
                                sum_reshape = reciprocal.src[0]
                                if sum_reshape.op is Ops.RESHAPE and sum_reshape.src:
                                    sum_reduce = sum_reshape.src[0]
                                    if (
                                        max_reduce.op is Ops.REDUCE
                                        and max_reduce.arg[0] is Ops.MAX
                                        and max_reduce.src
                                        and max_reduce.src[0].op is Ops.PERMUTE
                                        and max_reduce.src[0].src[0] is data
                                        and sum_reduce.op is Ops.REDUCE
                                        and sum_reduce.arg[0] is Ops.ADD
                                        and sum_reduce.src
                                        and sum_reduce.src[0].op is Ops.PERMUTE
                                        and sum_reduce.src[0].src[0] is exp2_node
                                    ):
                                        shape = tuple(int(dim) for dim in root.shape)
                                        if shape != (16, 1000):
                                            raise ValueError(
                                                "AX UOp Softmax lowering is measured only for shape (16, 1000)"
                                            )
                                        if str(root.dtype).split(".")[-1] != "float":
                                            raise ValueError(
                                                "AX UOp Softmax lowering currently supports float32 data only"
                                            )
                                        graph = onnx.helper.make_graph(
                                            [
                                                onnx.helper.make_node(
                                                    "Softmax", ["x"], ["y"], axis=1
                                                )
                                            ],
                                            "tinygrad_uop_softmax_ax",
                                            [
                                                onnx.helper.make_tensor_value_info(
                                                    "x", onnx.TensorProto.FLOAT, shape
                                                )
                                            ],
                                            [
                                                onnx.helper.make_tensor_value_info(
                                                    "y", onnx.TensorProto.FLOAT, shape
                                                )
                                            ],
                                        )
                                        return onnx.helper.make_model(
                                            graph,
                                            opset_imports=[
                                                onnx.helper.make_opsetid("", 13)
                                            ],
                                        )

    misc_op = None
    misc_data = None
    if root.op is Ops.SQRT and len(root.src) == 1:
        misc_op, misc_data = "Sqrt", root.src[0]
    elif root.op is Ops.MUL and len(root.src) == 2:
        log2_node, constant = root.src
        if (
            log2_node.op is Ops.LOG2
            and len(log2_node.src) == 1
            and constant.op is Ops.CONST
            and float(constant.arg) == 0.6931471805599453
        ):
            misc_op, misc_data = "Log", log2_node.src[0]
    if misc_op is not None:
        if (
            misc_data is None
            or misc_data.op is not Ops.RESHAPE
            or not misc_data.src
            or misc_data.src[0].op is not Ops.ALLOC
        ):
            raise ValueError(
                f"AX UOp {misc_op} lowering requires an ALLOC-backed input"
            )
        if str(root.dtype).split(".")[-1] != "float":
            raise ValueError(
                f"AX UOp {misc_op} lowering currently supports float32 data only"
            )
        shape = tuple(int(dim) for dim in root.shape)
        measured_shapes = {
            "Sqrt": (512, 512, 3, 3),
            "Log": (16, 1000),
        }
        if shape != measured_shapes[misc_op]:
            raise ValueError(
                f"AX UOp {misc_op} lowering is measured only for shape {measured_shapes[misc_op]}"
            )
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(misc_op, ["x"], ["y"])],
            f"tinygrad_uop_{misc_op.lower()}_ax",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        )
        return onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
        )

    if root.op is Ops.MUL and len(root.src) == 2:
        data, constant = root.src
        if constant.op is Ops.CONST and float(constant.arg) == -1.0:
            if (
                data.op is not Ops.RESHAPE
                or not data.src
                or data.src[0].op is not Ops.ALLOC
            ):
                raise ValueError("AX UOp Neg lowering requires an ALLOC-backed input")
            if str(root.dtype).split(".")[-1] != "float":
                raise ValueError(
                    "AX UOp Neg lowering currently supports float32 data only"
                )
            shape = tuple(int(dim) for dim in root.shape)
            if shape != (1, 1):
                raise ValueError("AX UOp Neg lowering is measured only for shape [1,1]")
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Neg", ["x"], ["y"])],
                "tinygrad_uop_neg_ax",
                [
                    onnx.helper.make_tensor_value_info(
                        "x", onnx.TensorProto.FLOAT, shape
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "y", onnx.TensorProto.FLOAT, shape
                    )
                ],
            )
            return onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
            )

    binary_op = None
    left = right = None
    if root.op is Ops.ADD and len(root.src) == 2:
        left, right = root.src
        if (
            right.op is Ops.MUL
            and len(right.src) == 2
            and right.src[1].op is Ops.CONST
            and float(right.src[1].arg) == -1.0
        ):
            binary_op = "Sub"
            right = right.src[0]
        else:
            binary_op = "Add"
    elif root.op is Ops.MUL and len(root.src) == 2:
        left, right = root.src
        if right.op is Ops.RECIPROCAL and len(right.src) == 1:
            binary_op = "Div"
            right = right.src[0]
        else:
            binary_op = "Mul"
    if binary_op is not None:
        if len(root.src) != 2:
            raise ValueError(f"AX UOp {binary_op} lowering requires two operands")
        assert left is not None and right is not None
        left_shape = tuple(int(dim) for dim in left.shape)
        right_shape = tuple(int(dim) for dim in right.shape)
        shape = tuple(int(dim) for dim in root.shape)
        if not left_shape or not right_shape:
            raise ValueError(
                f"AX UOp {binary_op} lowering requires non-empty static operand shapes"
            )
        try:
            broadcast_shape = tuple(np.broadcast_shapes(left_shape, right_shape))
        except ValueError as exc:
            raise ValueError(
                f"AX UOp {binary_op} lowering requires broadcast-compatible shapes"
            ) from exc
        if broadcast_shape != shape:
            raise ValueError(
                f"AX UOp {binary_op} output shape {shape} does not match broadcast shape {broadcast_shape}"
            )
        if not alloc_backed_view(left) or not alloc_backed_view(right):
            raise ValueError(
                f"AX UOp {binary_op} lowering requires ALLOC-backed inputs"
            )
        if str(root.dtype).split(".")[-1] != "float":
            raise ValueError(
                f"AX UOp {binary_op} lowering currently supports float32 data only"
            )
        if any(dim <= 0 for dim in shape):
            raise ValueError(
                f"AX UOp {binary_op} lowering requires positive static shapes"
            )
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(binary_op, ["x", "z"], ["y"])],
            f"tinygrad_uop_{binary_op.lower()}_ax",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, left_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, right_shape
                ),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        )
        return onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
        )

    position = "before"
    relu = root
    if root.op is Ops.RESHAPE:
        if len(root.src) < 1 or root.src[0].op is not Ops.WHERE:
            raise ValueError(
                "AX UOp lowering requires a Relu-shaped WHERE root or outer Reshape"
            )
        position = "after"
        relu = root.src[0]
    if relu.op is not Ops.WHERE or len(relu.src) != 3:
        raise ValueError("AX UOp lowering currently requires a Relu-shaped WHERE root")
    condition, true_value, false_value = relu.src
    if condition.op is not Ops.CMPLT or len(condition.src) != 2:
        raise ValueError("AX UOp lowering requires CMPLT(0, x) as the Relu condition")
    zero, data = condition.src
    if zero.op is not Ops.CONST or float(zero.val) != 0.0:
        raise ValueError("AX UOp Relu condition must compare against zero")
    if (
        true_value is not data
        or false_value.op is not Ops.CONST
        or float(false_value.val) != 0.0
    ):
        raise ValueError("AX UOp Relu branches are not in the supported canonical form")
    standalone = (
        position == "before"
        and data.op is Ops.RESHAPE
        and len(data.src) >= 1
        and data.src[0].op is Ops.ALLOC
    )
    if standalone:
        source = data
        source_shape = tuple(int(dim) for dim in data.shape)
        target_shape = source_shape
    elif position == "before":
        if (
            data.op is not Ops.RESHAPE
            or len(data.src) < 1
            or data.src[0].op is not Ops.RESHAPE
        ):
            raise ValueError("AX UOp lowering requires one source reshape before Relu")
        source = data.src[0]
        source_shape = tuple(int(dim) for dim in source.shape)
        target_shape = tuple(int(dim) for dim in data.shape)
    else:
        if data.op is not Ops.RESHAPE or len(data.src) < 1:
            raise ValueError("AX UOp lowering requires an ALLOC-backed input to Relu")
        source = data
        source_shape = tuple(int(dim) for dim in data.shape)
        target_shape = tuple(int(dim) for dim in root.shape)
    if (
        not source_shape
        or not target_shape
        or any(dim <= 0 for dim in (*source_shape, *target_shape))
    ):
        raise ValueError("AX UOp lowering requires positive static shapes")
    if source.src[0].op is not Ops.ALLOC:
        raise ValueError("AX UOp lowering requires an ALLOC-backed input")
    if relu.dtype != data.dtype or root.dtype != relu.dtype:
        raise ValueError("AX UOp Relu output dtype differs from its data input")
    if str(root.dtype).split(".")[-1] != "float":
        raise ValueError("AX UOp lowering currently supports float32 data only")

    shape_name = "uop_reshape_shape"
    nodes = (
        [onnx.helper.make_node("Relu", ["x"], ["y"])]
        if standalone
        else [
            onnx.helper.make_node("Reshape", ["x", shape_name], ["reshaped"]),
            onnx.helper.make_node("Relu", ["reshaped"], ["y"]),
        ]
        if position == "before"
        else [
            onnx.helper.make_node("Relu", ["x"], ["relu"]),
            onnx.helper.make_node("Reshape", ["relu", shape_name], ["y"]),
        ]
    )
    initializers = (
        []
        if standalone
        else [
            onnx.numpy_helper.from_array(
                np.asarray(target_shape, dtype=np.int64), shape_name
            )
        ]
    )
    graph = onnx.helper.make_graph(
        nodes,
        "tinygrad_uop_ax",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, source_shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, target_shape)],
        initializers,
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )


def compile_uop(
    root,
    schedule_path: str | None = None,
    calibration: Mapping[str, Mapping[str, float | int]] | None = None,
) -> bytes:
    """Lower a supported tinygrad UOp and emit an AX model without Pulsar2.

    Standalone binary operations require explicit
    ``calibration={"scales": {"x", "z", "y"}, "zero_points": {"x", "z",
    "y"}}`` because their measured AX programs depend on quantization, not
    just the UOp shape.
    """
    import graph_generator

    model = lower_uop_to_onnx(root)
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "uop.onnx")
        output = os.path.join(directory, "uop.axmodel")
        onnx.save(model, source)
        graph_generator.generate(
            source, output, schedule_path=schedule_path, calibration=calibration
        )
        with open(output, "rb") as stream:
            return stream.read()


def onnx_to_uop(model_or_path):
    """Import one static-shape ONNX graph through tinygrad and return its UOp.

    This is deliberately a small bridge rather than a second ONNX importer:
    tinygrad's ``OnnxRunner`` is the importer, so its normal lowering rules
    (including the fork's view and broadcast lowering) are the source of
    truth.  Inputs are symbolic ``Tensor.empty`` values; no input data is
    realized.  The returned UOp therefore describes the graph and is ready
    for :func:`compile_uop`.

    The AX mcode path currently emits one output at a time.  Graphs with more
    than one output, dynamic dimensions, non-float inputs, or missing output
    names are rejected before tinygrad can silently choose a different
    program family.
    """
    try:
        from tinygrad import Tensor
        from tinygrad.nn.onnx import OnnxRunner
    except ImportError as exc:
        raise ImportError(
            "ONNX-to-UOp compilation needs the pinned tinygrad fork"
        ) from exc

    cleanup = False
    if isinstance(model_or_path, onnx.ModelProto):
        model = model_or_path
        handle = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        try:
            handle.write(model.SerializeToString())
            handle.close()
        except Exception:
            handle.close()
            os.unlink(handle.name)
            raise
        model_path = handle.name
        cleanup = True
    elif isinstance(model_or_path, (bytes, bytearray, memoryview)):
        model = onnx.load_model_from_string(bytes(model_or_path))
        handle = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        handle.write(model.SerializeToString())
        handle.close()
        model_path = handle.name
        cleanup = True
    else:
        model_path = os.fspath(model_or_path)
        model = onnx.load(model_path, load_external_data=False)

    try:
        initializer_names = {item.name for item in model.graph.initializer}
        feeds = {}
        for value in model.graph.input:
            if value.name in initializer_names:
                continue
            tensor_type = value.type.tensor_type
            if tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise ValueError(
                    f"ONNX-to-UOp currently supports float32 inputs only: {value.name}"
                )
            shape = []
            for dim in tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    raise ValueError(
                        f"ONNX-to-UOp requires static positive shape for {value.name}"
                    )
                shape.append(int(dim.dim_value))
            feeds[value.name] = Tensor.empty(*shape)
        if len(model.graph.output) != 1:
            raise ValueError(
                f"ONNX-to-UOp currently emits one output, got {len(model.graph.output)}"
            )
        output_name = model.graph.output[0].name
        runner = OnnxRunner(model_path)
        outputs = runner(feeds)
        try:
            output = outputs[output_name]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(
                f"tinygrad OnnxRunner did not produce ONNX output {output_name!r}"
            ) from exc
        if not hasattr(output, "uop"):
            raise ValueError("tinygrad OnnxRunner output is not a Tensor")
        return output.uop
    finally:
        if cleanup:
            os.unlink(model_path)


def compile_onnx(
    model_or_path,
    schedule_path: str | None = None,
    calibration: Mapping[str, Mapping[str, float | int]] | None = None,
) -> bytes:
    """Compile ONNX through the complete ``ONNX -> tinygrad UOp -> mcode`` path.

    ``model_or_path`` may be an ONNX ``ModelProto``, serialized model bytes,
    or a filesystem path.  The UOp lowering remains intentionally measured:
    unsupported shapes and graph forms raise from :func:`compile_uop` rather
    than falling back to Pulsar2.
    """
    return compile_uop(
        onnx_to_uop(model_or_path),
        schedule_path=schedule_path,
        calibration=calibration,
    )


def apply_policy(
    key: TemplateKey, policy: QuantPolicy, node: str | None = None
) -> TemplateKey:
    """``key`` with its weight dtype set by ``policy``. A key that already
    names a weight dtype must agree with the policy. Templates exist only
    for the build path, so an llm_build policy is refused here (use
    ``encode_weight`` for llm_build weight bytes)."""
    if key.op not in _WEIGHT_OPS:
        if key.weight_dtype:
            raise ValueError(f"{key.op} has no constant weight")
        return key
    if policy.path != "build":
        raise ValueError(
            "no llm_build engine templates are committed; encode_weight() "
            "gives llm_build weight bytes"
        )
    choice = policy.choice_for(key.op, node)
    if choice == "auto":
        raise ValueError('"auto" is resolved from weights; call policy.resolve()')
    dt = validate_weight_choice("build", key.op, choice, key.attr("w"))
    if key.weight_dtype and key.weight_dtype != dt:
        raise ValueError(
            f"request asks for {key.weight_dtype} weights, policy says {dt}"
        )
    return dataclasses.replace(key, weight_dtype=dt)


def compile_request(
    src: str, cache: TemplateCache | None = None, policy: QuantPolicy | None = None
) -> bytes:
    req = json.loads(src)
    if req.get("kind") == "graph_template":
        required = ("source", "template_source", "template_axmodel")
        missing = [name for name in required if not isinstance(req.get(name), str)]
        if missing:
            raise ValueError(f"graph_template request missing paths: {missing}")
        cache = cache or TemplateCache()
        output = req.get("output")
        if output is None:
            return cache.generate_graph_template_bytes(
                req["source"],
                req["template_source"],
                req["template_axmodel"],
            )
        cache.generate_graph_template(
            req["source"],
            req["template_source"],
            req["template_axmodel"],
            output,
        )
        with open(output, "rb") as f:
            return f.read()
    if req.get("kind") == "generated_graph":
        source = req.get("source")
        if not isinstance(source, str):
            raise ValueError("generated_graph request missing source path")
        import graph_generator

        output = req.get("output")
        schedule = req.get("schedule")
        if output is not None and not isinstance(output, str):
            raise ValueError("generated_graph output must be a path")
        if schedule is not None and not isinstance(schedule, str):
            raise ValueError("generated_graph schedule must be a path")
        if output is None:
            with tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, "generated.axmodel")
                graph_generator.generate(source, output, schedule_path=schedule)
                with open(output, "rb") as f:
                    return f.read()
        graph_generator.generate(source, output, schedule_path=schedule)
        with open(output, "rb") as f:
            return f.read()
    if req.get("kind") == "onnx_uop":
        source = req.get("source")
        if not isinstance(source, str):
            raise ValueError("onnx_uop request missing source path")
        output = req.get("output")
        schedule = req.get("schedule")
        if output is not None and not isinstance(output, str):
            raise ValueError("onnx_uop output must be a path")
        if schedule is not None and not isinstance(schedule, str):
            raise ValueError("onnx_uop schedule must be a path")
        calibration = req.get("calibration")
        model_bytes = compile_onnx(
            source, schedule_path=schedule, calibration=calibration
        )
        if output is not None:
            with open(output, "wb") as f:
                f.write(model_bytes)
        return model_bytes
    key = TemplateKey.from_json(req["key"])
    if policy is not None:
        key = apply_policy(key, policy, req.get("node"))
    model = EditSet([edit_from_json(e) for e in req["edits"]]).build(key, cache)
    return model.SerializeToString()


def tinygrad_classes() -> dict[str, type]:
    """``AXCompiler`` / ``AXAllocator`` / ``AXProgram`` subclassing the fork's
    ``tinygrad.device`` bases. Only the compiler does anything."""
    from tinygrad.device import Allocator, Compiler, Program

    class AXCompiler(Compiler):
        """Architecture B's compiler: template lookup + validated edits. The
        cache key disables tinygrad's disk cache; templates are already cached
        as fixtures. ``weight_dtype`` (a tinygrad dtype, e.g. ``dtypes.int8``)
        or a full ``policy`` selects how constant weights are stored."""

        def __init__(
            self,
            cache: TemplateCache | None = None,
            policy: QuantPolicy | None = None,
            weight_dtype=None,
        ):
            super().__init__(cachekey=None)
            self.cache = cache or TemplateCache()
            if policy is not None and weight_dtype is not None:
                raise ValueError("pass policy or weight_dtype, not both")
            if weight_dtype is not None:
                policy = QuantPolicy(default=weight_dtype)
            self.policy = policy

        def compile(self, src: str) -> bytes:
            return compile_request(src, self.cache, self.policy)

    class AXAllocator(Allocator):
        """Host-staged AXCL buffers. AXCL binds device memory to one loaded
        model's IO (``axclrtEngineSetInputBufferByIndex``), so a tinygrad
        buffer lives in host memory and ``AXProgram`` stages it into that
        model's IO on each call (``axcl_session.AXSession``). The storage is
        host-visible, so ``Buffer.numpy()``/``initial_value`` need no copy
        program."""

        def __init__(self, dev):
            super().__init__(
                dev, supports_copy_from_disk=False, supports_transfer=False
            )

        def _alloc(self, size, options):
            from tinygrad.device import BufferStorage
            from tinygrad.runtime.support.memory import MMIOInterface

            arr = np.zeros(size, np.uint8)
            return BufferStorage(
                arr, None, MMIOInterface(arr.ctypes.data, size, fmt="B")
            )

        def _free(self, storage, options):
            pass  # numpy owns the memory

        def _copyin(self, dest, src: memoryview):
            dest[:] = np.frombuffer(src.cast("B"), np.uint8)

        def _copyout(self, dest: memoryview, src):
            dest.cast("B")[:] = src.tobytes()

        def _as_buffer(self, src) -> memoryview:
            return memoryview(src)

    class AXProgram(Program):
        """A compiled ``.axmodel`` (``AXCompiler`` output) loaded into the
        process-wide AXCL session (``ax_session``). tinygrad's convention: the
        call's buffers are the model's outputs first, then its inputs, each
        the raw bytes of the model's IO tensor."""

        def __init__(self, dev, obj):
            self.dev, self.obj = dev, bytes(obj)
            self.session = ax_session()
            self.model = self.session.load(self.obj)

        @staticmethod
        def _stage_input(buffer, spec):
            """Shape or broadcast a tinygrad input to the emitted model IO.

            Binary templates are emitted at the full output shape because the
            AX mcode has no separate broadcast instruction.  A scalar (or
            another smaller broadcastable input) can therefore arrive in a
            tinygrad buffer with fewer elements than the template expects.
            """
            raw = np.frombuffer(
                buffer, np.uint8, count=memoryview(buffer).nbytes
            ).view(spec.dtype)
            expected = int(np.prod(spec.shape, dtype=np.int64))
            if raw.size == expected:
                return raw.reshape(spec.shape)
            try:
                return np.broadcast_to(raw, spec.shape).copy()
            except ValueError as exc:
                raise ValueError(
                    f"input has {raw.size} elements, cannot broadcast to {spec.shape}"
                ) from exc

        def __call__(
            self,
            *bufs,
            global_size=(1, 1, 1),
            local_size=(1, 1, 1),
            vals=(),
            wait=False,
        ):
            m = self.model
            n_out = len(m.outputs)
            if len(bufs) != n_out + len(m.inputs):
                raise ValueError(
                    f"model has {n_out} outputs + {len(m.inputs)} inputs, got {len(bufs)} buffers"
                )
            ins = [self._stage_input(b, spec) for b, spec in zip(bufs[n_out:], m.inputs)]
            before = self.session.exec_us
            outs = self.session.run(m, ins)
            for b, y in zip(bufs[:n_out], outs):
                b[: y.nbytes] = np.frombuffer(y.tobytes(), np.uint8)
            return (self.session.exec_us - before) / 1e6 if wait else None

        def __del__(self):
            try:
                if self.session._proc is not None:
                    self.session.unload(self.model)
            except Exception:
                pass

    return {
        "AXCompiler": AXCompiler,
        "AXAllocator": AXAllocator,
        "AXProgram": AXProgram,
    }


_AX_SESSION = None


def ax_session():
    """The process-wide ``axcl_session.AXSession`` the tinygrad ``AX`` device
    runs on, opened on first use (it holds ``/tmp/axcl-device.lock`` until
    ``close_ax_session``)."""
    global _AX_SESSION
    if _AX_SESSION is None:
        import axcl_session

        _AX_SESSION = axcl_session.AXSession().__enter__()
    return _AX_SESSION


def close_ax_session() -> None:
    global _AX_SESSION
    if _AX_SESSION is not None:
        _AX_SESSION.close()
        _AX_SESSION = None


def register_ax_device() -> str:
    """Make ``"AX"`` a tinygrad device (``Device["AX"]``, ``Buffer("AX", ...)``)
    backed by ``AXAllocator``/``AXProgram``. The fork discovers devices as
    ``tinygrad.runtime.ops_<name>`` modules, so this installs one in
    ``sys.modules``. tinygrad's own scheduler cannot target it (there is no
    renderer from UOps to templates); programs come from ``AXCompiler``."""
    import types

    from tinygrad.device import Compiled

    name = "tinygrad.runtime.ops_ax"
    if name not in sys.modules:
        classes = tinygrad_classes()

        class AXDevice(Compiled):
            def __init__(self, device: str):
                super().__init__(
                    device, classes["AXAllocator"](self), [], classes["AXProgram"]
                )
                self.compiler_ = classes["AXCompiler"]()

            def synchronize(self, timeout=None):
                pass  # every AXProgram call is synchronous

            def finalize(self):
                close_ax_session()

        mod = types.ModuleType(name)
        mod.AXDevice = AXDevice
        sys.modules[name] = mod
    return "AX"


def npu_params_of(model_bytes: bytes) -> bytes:
    model = onnx.load_model_from_string(model_bytes)
    return bytes(_initializer(model, "npu_params").raw_data)


def mcode_of(model_bytes: bytes) -> bytes:
    return bytes(_mcode_initializer(onnx.load_model_from_string(model_bytes)).raw_data)


def gather_indices_of(model_bytes: bytes, count: int) -> list[int]:
    table = npu_params_of(model_bytes)
    return list(struct.unpack(f"<{count}I", table[: 4 * count]))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="write a graph's op records as gzipped JSON")
    ex.add_argument("onnx")
    ex.add_argument("out")
    cov = sub.add_parser("coverage", help="coverage report from op records")
    cov.add_argument("records")
    cov.add_argument("--weight-dtype", help="per-node weight dtype for this choice")
    cov.add_argument("--path", default="build", choices=sorted(WEIGHT_PATHS))
    cov.add_argument(
        "--calibration",
        help="step_calibration.py JSON: settle conditional nodes against it",
    )
    args = p.parse_args(argv)
    if args.cmd == "extract":
        with gzip.open(args.out, "wt") as f:
            json.dump(extract_step_ops(args.onnx), f)
        return 0
    with gzip.open(args.records, "rt") as f:
        records = json.load(f)
    policy = None
    if args.weight_dtype:
        policy = QuantPolicy(default=args.weight_dtype, path=args.path)
    calib = load_calibration(args.calibration) if args.calibration else None
    report = coverage_report(records, policy, calib)
    report.pop("per_node", None)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
