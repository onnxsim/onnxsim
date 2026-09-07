"""Builds a runnable ONNX graph *and* its weights directly from
Qwen-Drive-1.0's BEV perception head (``qwen_drive_perception``, released
separately from the VLM as ``Qwen/Qwen-Drive-1.0-4B``'s ``perception``
checkpoint) -- the BEVFormer-style spine (image features -> ego-anchored
BEV features) plus its three task heads (3D detection, occupancy, BEV map
segmentation).

Same "known architecture template, hydrate with the checkpoint's own
tensors" approach as :mod:`onnxsim.hf_reconstruct`/
:mod:`onnxsim.gguf_reconstruct`/:mod:`onnxsim.qwen3_5_reconstruct`, reusing
that family's safetensors reader and ``_Builder``/``_linear``/``_unsqueeze``/
``_slice_last_dim`` helpers. Every architectural detail below was confirmed
against the real source in ``QwenLM/Qwen-Drive-1.0``
(``src/qwen_drive_perception/*.py`` -- ``configuration_perception.py``,
``modeling_perception.py``, ``fpn.py``, ``layers.py``, ``view_transform.py``,
``geometry.py``, ``bev_encoder.py``, ``perception_transformer.py``,
``attention.py``, ``heads.py``, ``occ_refiner.py``, ``map_seg.py``) and the
CUDA kernel sources under ``src/qwen_drive_perception/ops/*/src/*.cu`` --
not guessed from the BEVFormer/lift-splat papers alone (though the
architecture does follow them closely, and this module cites them where the
real code's own docstrings do).

**Three design decisions make this a static-shape ONNX graph** despite the
reference implementation leaning on two custom CUDA kernels and one
genuinely data-dependent shape:

1. **Camera calibration is a build-time constant, not a graph input.**
   The reference code takes ``lidar2img``/``lidar2ego`` through
   ``torch.inverse``/``torch.linalg.inv`` in three places
   (``Uni3DVoxelPoolDepth.coord_preparing``, ``BEVFormerEncoder.
   point_sampling``, ``geometry.ego_to_lidar_boxes``) -- but a deployed
   vehicle's camera rig calibration does not change frame to frame, only
   the *image content* does. So every one of those matrix products, and
   every geometric quantity derived purely from calibration + fixed config
   (the frustum's unprojected ego-frame voxel index and validity per
   pixel/depth bin, the BEV encoder's per-query reference points projected
   into each camera and their per-camera visibility/count) is precomputed
   in plain numpy at graph-*build* time from caller-supplied calibration,
   and baked in as constant initializers -- not a single matrix inverse or
   camera-projection op appears in the built graph. This is the same
   "push what's genuinely static to build time" choice
   :mod:`onnxsim.qwen3_5_reconstruct` already makes for the vision
   encoder's per-image ``grid_thw``-derived position tables, taken one
   step further because *every* geometric quantity here turns out to be
   calibration-derived rather than image-content-derived.
2. **``SpatialCrossAttention``'s data-dependent rebatch is replaced by
   dense masked evaluation.** The real module's own forward (confirmed
   from source, ``attention.py`` lines ~251-285) computes
   ``max_len = int(valid_counts.max().item())`` and gathers only the
   queries each camera actually sees before running deformable attention,
   purely as a compute optimization -- masked-out queries contribute
   exactly zero to the final sum either way. This builder instead runs
   the deformable attention for *every* BEV query against *every* camera
   (a static ``[num_cams, num_query, ...]`` shape), multiplies by the same
   build-time visibility mask, and sums over the (fixed-size) camera axis
   -- provably the same result, more FLOPs, no dynamic shape.
3. **Deformable-attention sampling uses ONNX's native ``GridSample``.**
   The real CUDA kernel (``ms_deform_attn_bf16_cuda.cu``) does a per-point
   4-tap bilinear gather-and-accumulate -- confirmed to be exactly what
   ``F.grid_sample(mode="bilinear", padding_mode="zeros",
   align_corners=False)`` computes (the repo's own ``layers.py::
   multi_scale_deformable_attn_pytorch`` is its CPU fallback, built on
   exactly that call), which is ONNX's ``GridSample`` op (opset 16+)
   verbatim.

Two more narrower simplifications, each with the same "known template"
justification as the rest of this module family:

* **``voxel_pool_depth`` (the other custom CUDA kernel) is a depth-weighted
  scatter-add.** Confirmed from ``voxel_pool_cuda.cu``: for every valid
  frustum cell it adds ``img_feat[cam, :, h, w] * img_depth[cam, d, h, w]``
  into its unprojected ego voxel bin; the CUDA kernel's rank-sorted
  interval grouping is a parallelism trick for the same sum, not a
  different result. Built here as ``Gather`` (per-point feature and depth)
  + ``Mul`` + ``ScatterND`` with ``reduction="add"`` over build-time
  per-point voxel indices (out-of-range points get their index clamped
  in-bounds and their contribution zeroed by a build-time mask, so they
  add exactly nothing -- see point 1 above for why the indices themselves
  are build-time).
* **Every ``F.grid_sample`` call whose sampling grid is a pure function of
  fixed config** (``PerceptionTransformer._normalized_occ_grid`` for the
  occupancy crop/resample, ``heads.BevFeatureSlicer`` for the map-seg BEV
  crop) **has that grid baked in as a constant**, leaving only ONNX's
  ``GridSample`` on the runtime feature volume.

Scope, narrower than the full ``QwenDrivePerception.infer()`` pipeline:

* **Single call, single sample, no temporal fusion** -- matches how
  ``infer()`` itself always calls the model (``img_metas`` is a length-1
  list, ``prev_bev=None`` always). ``TemporalSelfAttention`` therefore
  always runs its single-frame degenerate path (``value = stack([query,
  query])``, confirmed from source), never real cross-frame attention.
* **A caller-chosen, fixed camera count and BEV/frustum/occupancy grid
  sizes** -- all static build parameters (defaults match the released
  config's real values from ``configuration_perception.py``), like every
  other static dimension in this module family.
* **Raw (ego-frame, un-postprocessed) outputs.** ``sigmoid``/``softmax``/
  ``argmax`` and the final top-``k`` *validity* filtering (``NMSFreeCoder``
  drops boxes outside ``post_center_range`` -- a genuinely variable-length
  result) are left to the caller: the detection head returns exactly
  ``max_num`` (300 by default) candidate boxes/scores/labels plus a
  ``valid`` boolean mask, rather than a dynamically-shaped filtered list.
  The final ego-to-lidar box-frame conversion (``geometry.
  ego_to_lidar_boxes``, itself another rigid-transform inverse) is
  likewise the caller's job -- one more instance of point 1 above.
* **No BF16 CUDA path.** Every op here is plain float32; the checkpoint's
  own BF16 storage is upcast on read exactly like
  :mod:`onnxsim.hf_reconstruct` does (see that module's own docstring for
  why upcasting the initializer bytes, not a graph-level ``Cast``, is the
  correct choice here too).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnx.helper

from onnxsim.gguf_reconstruct import (
    _IR_VERSION,
    _OPSET,
    UnsupportedArchitectureError,
    _Builder,
    _linear,
    _unsqueeze,
)
from onnxsim.hf_reconstruct import (
    _index_safetensors_checkpoint,
    _read_tensor,
    read_hf_config,
)

_SUPPORTED_MODEL_TYPE = "qwen_drive_perception"

# ---------------------------------------------------------------------------
# Fixed hyperparameters the real modeling code hardcodes directly in class
# constructors rather than reading from ``QwenDrivePerceptionConfig`` --
# confirmed from ``configuration_perception.py``'s own module-level constants
# and ``bev_encoder.py``/``perception_transformer.py``'s literal constructor
# calls (e.g. ``BEVFormerLayer.__init__``'s ``TemporalSelfAttention(...,
# num_heads=8, num_levels=1)``). Not part of the HF ``config.json`` a real
# checkpoint ships (only the ``self.xxx`` attributes ``QwenDrivePerceptionConfig.
# __init__`` assigns are), so this builder hardcodes them the same way the
# real model does, with the same values.
NUM_HEADS = 8
FFN_CHANNELS = 512
NUM_FEATURE_LEVELS = 4
POINTS_IN_PILLAR = 4
SCA_NUM_POINTS = 8
TSA_NUM_POINTS = 4
DECODER_NUM_POINTS = 4
DEFAULT_MAX_NUM_BOXES = 300
ASPP_MID_CHANNELS = (
    96  # DepthNet's own hardcoded `aspp_mid_channels=96` (modeling_perception.py)
)
DEFAULT_POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
FPN_ADAPTOR_SCALES = (4.0, 2.0, 1.0, 0.5)


# ---------------------------------------------------------------------------
# Generic op-building helpers (this module's own opset/ReduceMean-as-input
# needs differ enough from gguf_reconstruct's RoPE-oriented helpers that it
# is not worth sharing beyond _Builder/_linear/_unsqueeze themselves).


def _reduce(
    b: _Builder, op_type: str, x: str, axes: List[int], keepdims: int, prefix: str
) -> str:
    """``ReduceSum``'s reduction axes moved from an attribute to an input at
    opset 13; every other ``Reduce*`` op's (``ReduceMean``, ``ReduceMin``,
    ...) stayed an attribute until opset 18 (see
    :mod:`onnxsim.gguf_reconstruct`'s own opset comment -- this whole
    module family targets opset 17, in between the two), so ``ReduceSum``
    needs a different node shape here than every other reduction."""
    if op_type != "ReduceSum":
        return b.op(op_type, [x], prefix, axes=axes, keepdims=keepdims)
    axes_c = b.const(np.array(axes, dtype=np.int64), prefix="reduce_axes")
    return b.op(op_type, [x, axes_c], prefix, keepdims=keepdims)


def _reduce_sum(
    b: _Builder, x: str, axes: List[int], keepdims: int, prefix: str
) -> str:
    return _reduce(b, "ReduceSum", x, axes, keepdims, prefix)


def _reduce_mean(
    b: _Builder, x: str, axes: List[int], keepdims: int, prefix: str
) -> str:
    return _reduce(b, "ReduceMean", x, axes, keepdims, prefix)


def _relu(b: _Builder, x: str, prefix: str) -> str:
    return b.op("Relu", [x], prefix)


def _softplus(b: _Builder, x: str, prefix: str) -> str:
    return b.op("Softplus", [x], prefix)


def _sigmoid(b: _Builder, x: str, prefix: str) -> str:
    return b.op("Sigmoid", [x], prefix)


def _softmax(b: _Builder, x: str, axis: int, prefix: str) -> str:
    return b.op("Softmax", [x], prefix, axis=axis)


def _conv(
    b: _Builder,
    x: str,
    weight: str,
    bias: Optional[str],
    prefix: str,
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Optional[Sequence[int]] = None,
) -> str:
    """``Conv`` (2D or 3D, inferred from ``len(strides)``). ``pads`` is the
    symmetric per-spatial-dim padding (mirrored to ONNX's begin+end form)."""
    kwargs = dict(strides=list(strides), pads=list(pads) * 2)
    if dilations is not None:
        kwargs["dilations"] = list(dilations)
    inputs = [x, weight] + ([bias] if bias is not None else [])
    return b.op("Conv", inputs, prefix, **kwargs)


def _conv_transpose(
    b: _Builder, x: str, weight: str, bias: Optional[str], prefix: str, stride: int
) -> str:
    inputs = [x, weight] + ([bias] if bias is not None else [])
    return b.op(
        "ConvTranspose",
        inputs,
        prefix,
        strides=[stride, stride],
        kernel_shape=[stride, stride],
    )


def _maxpool2d(b: _Builder, x: str, prefix: str, kernel: int, stride: int) -> str:
    return b.op(
        "MaxPool", [x], prefix, kernel_shape=[kernel, kernel], strides=[stride, stride]
    )


def _batchnorm(
    b: _Builder,
    x: str,
    weight: str,
    bias: str,
    mean: str,
    var: str,
    eps: float,
    prefix: str,
) -> str:
    return b.op(
        "BatchNormalization",
        [x, weight, bias, mean, var],
        prefix,
        epsilon=eps,
        momentum=0.9,
    )


def _layernorm_channels_first(
    b: _Builder, x: str, weight: str, bias: str, eps: float, prefix: str
) -> str:
    """``LayerNorm2d``: per-position mean/var over the channel axis (axis=1)
    of an NCHW tensor -- confirmed from ``fpn.py``'s own hand-written
    forward (mean/var over ``dim=1``), *not* the same thing as ONNX's
    ``LayerNormalization`` with ``axis=1`` (which would normalize over axes
    1..end, i.e. channel *and* spatial together) -- so built from scratch
    rather than the native op."""
    mean = _reduce_mean(b, x, [1], 1, f"{prefix}.mean")
    centered = b.op("Sub", [x, mean], f"{prefix}.centered")
    var = _reduce_mean(
        b, b.op("Mul", [centered, centered], f"{prefix}.sq"), [1], 1, f"{prefix}.var"
    )
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="ln2d_eps")
    normed = b.op(
        "Div",
        [
            centered,
            b.op(
                "Sqrt",
                [b.op("Add", [var, eps_c], f"{prefix}.var_eps")],
                f"{prefix}.std",
            ),
        ],
        f"{prefix}.normed",
    )
    weight_r = b.op("Reshape", [weight, b.shape_const([1, -1, 1, 1])], f"{prefix}.w_r")
    bias_r = b.op("Reshape", [bias, b.shape_const([1, -1, 1, 1])], f"{prefix}.b_r")
    return b.op(
        "Add", [b.op("Mul", [normed, weight_r], f"{prefix}.scaled"), bias_r], prefix
    )


def _groupnorm(
    b: _Builder,
    x: str,
    weight: str,
    bias: str,
    num_groups: int,
    num_channels: int,
    eps: float,
    prefix: str,
    spatial_dims: int,
) -> str:
    """``nn.GroupNorm`` over an ``(N, C, *spatial)`` tensor, built from
    ``ReduceMean`` (no native ``GroupNormalization`` op at this module's
    target opset -- that op only exists from opset 18). A ``0`` entry in an
    ONNX ``Reshape`` target shape means "copy this dim from the input", so
    the batch dim never needs to be read out explicitly."""
    # (N, C, *S) -> (N, num_groups, C//num_groups * prod(S)) to normalize
    # over each group's channels+spatial jointly, then back. The "back"
    # reshape needs the real (possibly multi-axis) spatial shape, which a
    # `0`/`-1`-only shape can't express once spatial_dims > 1 (Reshape
    # allows only one -1) -- so it reuses `x`'s own runtime Shape instead.
    orig_shape = b.op("Shape", [x], f"{prefix}.orig_shape")
    reshaped = b.op(
        "Reshape", [x, b.shape_const([0, num_groups, -1])], f"{prefix}.grouped"
    )
    mean = _reduce_mean(b, reshaped, [2], 1, f"{prefix}.mean")
    centered = b.op("Sub", [reshaped, mean], f"{prefix}.centered")
    var = _reduce_mean(
        b, b.op("Mul", [centered, centered], f"{prefix}.sq"), [2], 1, f"{prefix}.var"
    )
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="gn_eps")
    normed = b.op(
        "Div",
        [
            centered,
            b.op(
                "Sqrt",
                [b.op("Add", [var, eps_c], f"{prefix}.var_eps")],
                f"{prefix}.std",
            ),
        ],
        f"{prefix}.normed",
    )
    back = b.op("Reshape", [normed, orig_shape], f"{prefix}.back")
    weight_r = b.op(
        "Reshape",
        [weight, b.shape_const([1, num_channels] + [1] * spatial_dims)],
        f"{prefix}.w_r",
    )
    bias_r = b.op(
        "Reshape",
        [bias, b.shape_const([1, num_channels] + [1] * spatial_dims)],
        f"{prefix}.b_r",
    )
    return b.op(
        "Add", [b.op("Mul", [back, weight_r], f"{prefix}.scaled"), bias_r], prefix
    )


def _layernorm_last_dim(
    b: _Builder, x: str, weight: str, bias: str, eps: float, prefix: str
) -> str:
    return b.op("LayerNormalization", [x, weight, bias], prefix, axis=-1, epsilon=eps)


def _upsample_bilinear(
    b: _Builder, x: str, scale: float, align_corners: bool, prefix: str, dims: int = 2
) -> str:
    """``nn.Upsample``/``F.interpolate`` (2D bilinear or 3D trilinear, per
    ``dims``) via ONNX ``Resize``. ``align_corners`` maps to ``Resize``'s
    ``coordinate_transformation_mode`` (``"align_corners"`` vs.
    ``"pytorch_half_pixel"``, which matches ``align_corners=False`` exactly,
    unlike plain ``"half_pixel"``, for a size-1 output dim edge case)."""
    mode = "linear"
    coord_mode = "align_corners" if align_corners else "pytorch_half_pixel"
    roi = b.const(np.array([], dtype=np.float32), prefix="resize_roi")
    scales = b.const(
        np.array([1.0, 1.0] + [scale] * dims, dtype=np.float32), prefix="resize_scales"
    )
    return b.op(
        "Resize",
        [x, roi, scales],
        prefix,
        mode=mode,
        coordinate_transformation_mode=coord_mode,
    )


def _resize_to_shape(
    b: _Builder, x: str, target_shape: Sequence[int], align_corners: bool, prefix: str
) -> str:
    """``F.interpolate(size=..., mode="trilinear"/"bilinear")`` -- resize to
    an explicit target shape rather than a scale factor. ``target_shape`` is
    the *full* output shape (batch and channel dims included, unchanged from
    ``x``'s own) -- every dim in this whole module is a static, build-time
    Python int (see the module docstring), so this is always a plain
    constant, never a runtime ``Shape`` op."""
    coord_mode = "align_corners" if align_corners else "pytorch_half_pixel"
    roi = b.const(np.array([], dtype=np.float32), prefix="resize_roi")
    empty_scales = b.const(np.array([], dtype=np.float32), prefix="resize_empty_scales")
    target_shape_c = b.const(
        np.array(target_shape, dtype=np.int64), prefix="resize_target_shape"
    )
    return b.op(
        "Resize",
        [x, roi, empty_scales, target_shape_c],
        prefix,
        mode="linear",
        coordinate_transformation_mode=coord_mode,
    )


def _slice_axis(
    b: _Builder, x: str, axis: int, start: int, end: int, prefix: str
) -> str:
    starts = b.const(np.array([start], dtype=np.int64), prefix="slice_start")
    ends = b.const(np.array([end], dtype=np.int64), prefix="slice_end")
    axes = b.const(np.array([axis], dtype=np.int64), prefix="slice_axis")
    return b.op("Slice", [x, starts, ends, axes], prefix)


def _squeeze(b: _Builder, x: str, axes: List[int], prefix: str) -> str:
    return b.op(
        "Squeeze",
        [x, b.const(np.array(axes, dtype=np.int64), prefix="squeeze_axes")],
        prefix,
    )


def _grid_sample(
    b: _Builder,
    x: str,
    grid: str,
    prefix: str,
    align_corners: bool,
    padding_mode: str = "zeros",
) -> str:
    return b.op(
        "GridSample",
        [x, grid],
        prefix,
        mode="linear",
        padding_mode=padding_mode,
        align_corners=1 if align_corners else 0,
    )


def _gelu(b: _Builder, x: str, prefix: str) -> str:
    c0 = b.const(np.array(0.5, dtype=np.float32), prefix="gelu_half")
    c1 = b.const(np.array(1.0, dtype=np.float32), prefix="gelu_one")
    inv_sqrt2 = b.const(
        np.array(1.0 / np.sqrt(2.0), dtype=np.float32), prefix="gelu_inv_sqrt2"
    )
    erf = b.op(
        "Erf", [b.op("Mul", [x, inv_sqrt2], f"{prefix}.scaled")], f"{prefix}.erf"
    )
    return b.op(
        "Mul",
        [
            b.op("Mul", [x, c0], f"{prefix}.half_x"),
            b.op("Add", [erf, c1], f"{prefix}.erf_p1"),
        ],
        prefix,
    )


# ---------------------------------------------------------------------------
# SimpleFPN (fpn.py), ConvModule/BasicBlock (layers.py), ASPP/DepthNet
# (view_transform.py) -- plain conv/norm stacks, confirmed 1:1 against source.


def _simple_fpn(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    dim: int,
    out_channels: int,
    scale_factors: Sequence[float],
) -> List[str]:
    outputs = []
    for i, scale in enumerate(scale_factors):
        p = f"{prefix}.stages.{i}"
        h = x
        out_dim = dim
        layer_idx = 0
        if scale == 4.0:
            h = _conv_transpose(
                b,
                h,
                declare(f"{p}.{layer_idx}.weight"),
                declare(f"{p}.{layer_idx}.bias"),
                f"{p}.deconv1",
                stride=2,
            )
            layer_idx += 1
            h = _layernorm_channels_first(
                b,
                h,
                declare(f"{p}.{layer_idx}.weight"),
                declare(f"{p}.{layer_idx}.bias"),
                1e-6,
                f"{p}.ln1",
            )
            layer_idx += 1
            h = _gelu(b, h, f"{p}.gelu")
            layer_idx += 1
            h = _conv_transpose(
                b,
                h,
                declare(f"{p}.{layer_idx}.weight"),
                declare(f"{p}.{layer_idx}.bias"),
                f"{p}.deconv2",
                stride=2,
            )
            layer_idx += 1
            out_dim = dim // 4
        elif scale == 2.0:
            h = _conv_transpose(
                b,
                h,
                declare(f"{p}.{layer_idx}.weight"),
                declare(f"{p}.{layer_idx}.bias"),
                f"{p}.deconv",
                stride=2,
            )
            layer_idx += 1
            out_dim = dim // 2
        elif scale == 1.0:
            pass
        elif scale == 0.5:
            h = _maxpool2d(b, h, f"{p}.maxpool", kernel=2, stride=2)
            layer_idx += 1
        else:
            raise UnsupportedArchitectureError(
                f"SimpleFPN scale_factor={scale!r} is not supported"
            )

        h = _conv(
            b,
            h,
            declare(f"{p}.{layer_idx}.weight"),
            None,
            f"{p}.proj1",
            strides=[1, 1],
            pads=[0, 0],
        )
        layer_idx += 1
        h = _layernorm_channels_first(
            b,
            h,
            declare(f"{p}.{layer_idx}.weight"),
            declare(f"{p}.{layer_idx}.bias"),
            1e-6,
            f"{p}.ln2",
        )
        layer_idx += 1
        h = _conv(
            b,
            h,
            declare(f"{p}.{layer_idx}.weight"),
            None,
            f"{p}.proj2",
            strides=[1, 1],
            pads=[1, 1],
        )
        layer_idx += 1
        h = _layernorm_channels_first(
            b,
            h,
            declare(f"{p}.{layer_idx}.weight"),
            declare(f"{p}.{layer_idx}.bias"),
            1e-6,
            f"{p}.ln3",
        )
        outputs.append(h)
        del out_dim
    return outputs


def _conv_module_2d(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    norm: Optional[str],
    num_groups: int,
    num_channels: int,
    act: bool,
    strides=(1, 1),
    pads=(0, 0),
    bias: bool = True,
) -> str:
    conv_bias = declare(f"{prefix}.conv.bias") if bias else None
    h = _conv(
        b,
        x,
        declare(f"{prefix}.conv.weight"),
        conv_bias,
        f"{prefix}.conv",
        strides=list(strides),
        pads=list(pads),
    )
    if norm == "gn":
        h = _groupnorm(
            b,
            h,
            declare(f"{prefix}.gn.weight"),
            declare(f"{prefix}.gn.bias"),
            num_groups,
            num_channels,
            1e-5,
            f"{prefix}.gn",
            spatial_dims=2,
        )
    elif norm == "bn":
        h = _batchnorm(
            b,
            h,
            declare(f"{prefix}.bn.weight"),
            declare(f"{prefix}.bn.bias"),
            declare(f"{prefix}.bn.running_mean"),
            declare(f"{prefix}.bn.running_var"),
            1e-5,
            f"{prefix}.bn",
        )
    if act:
        h = _relu(b, h, f"{prefix}.act")
    return h


def _conv_module_3d(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    norm: Optional[str],
    num_groups: int,
    num_channels: int,
    act: bool,
    bias: bool = False,
) -> str:
    conv_bias = declare(f"{prefix}.conv.bias") if bias else None
    h = _conv(
        b,
        x,
        declare(f"{prefix}.conv.weight"),
        conv_bias,
        f"{prefix}.conv",
        strides=[1, 1, 1],
        pads=[0, 0, 0],
    )
    if norm == "bn":
        h = _batchnorm(
            b,
            h,
            declare(f"{prefix}.bn.weight"),
            declare(f"{prefix}.bn.bias"),
            declare(f"{prefix}.bn.running_mean"),
            declare(f"{prefix}.bn.running_var"),
            1e-5,
            f"{prefix}.bn",
        )
    if act:
        h = _relu(b, h, f"{prefix}.act")
    return h


def _basic_block_2d_gn(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    planes: int,
    num_groups: int,
    stride: int,
    has_downsample: bool,
) -> str:
    """``BasicBlock`` (``layers.py``) with GroupNorm -- used by ``DepthNet``.
    Attribute names ``gn1``/``gn2`` (vs. ``map_seg.py``'s ``_GNBasicBlock``,
    which keeps its GroupNorms under the stale ``bn1``/``bn2`` names -- see
    :func:`_basic_block_2d_gn_resnet18`)."""
    return _basic_block_2d_named_gn(
        b,
        x,
        declare,
        prefix,
        planes,
        num_groups,
        stride,
        has_downsample,
        norm_prefix="gn",
    )


def _basic_block_2d_gn_resnet18(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    planes: int,
    stride: int,
    has_downsample: bool,
) -> str:
    """``_GNBasicBlock`` (``map_seg.py``) -- ``bn1``/``bn2`` are GroupNorm
    (``min(32, planes)`` groups), confirmed from source's own ``_gn`` helper."""
    return _basic_block_2d_named_gn(
        b,
        x,
        declare,
        prefix,
        planes,
        min(32, planes),
        stride,
        has_downsample,
        norm_prefix="bn",
    )


def _basic_block_2d_named_gn(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    planes: int,
    num_groups: int,
    stride: int,
    has_downsample: bool,
    norm_prefix: str,
) -> str:
    identity = x
    h = _conv(
        b,
        x,
        declare(f"{prefix}.conv1.weight"),
        None,
        f"{prefix}.conv1",
        strides=[stride, stride],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.{norm_prefix}1.weight"),
        declare(f"{prefix}.{norm_prefix}1.bias"),
        num_groups,
        planes,
        1e-5,
        f"{prefix}.{norm_prefix}1",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.relu1")
    h = _conv(
        b,
        h,
        declare(f"{prefix}.conv2.weight"),
        None,
        f"{prefix}.conv2",
        strides=[1, 1],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.{norm_prefix}2.weight"),
        declare(f"{prefix}.{norm_prefix}2.bias"),
        num_groups,
        planes,
        1e-5,
        f"{prefix}.{norm_prefix}2",
        spatial_dims=2,
    )
    if has_downsample:
        identity = _conv(
            b,
            x,
            declare(f"{prefix}.downsample.0.weight"),
            None,
            f"{prefix}.down_conv",
            strides=[stride, stride],
            pads=[0, 0],
        )
        identity = _groupnorm(
            b,
            identity,
            declare(f"{prefix}.downsample.1.weight"),
            declare(f"{prefix}.downsample.1.bias"),
            num_groups,
            planes,
            1e-5,
            f"{prefix}.down_gn",
            spatial_dims=2,
        )
    return _relu(b, b.op("Add", [h, identity], f"{prefix}.resid"), f"{prefix}.relu2")


def _aspp(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    inplanes: int,
    aspp_mid_channels: int,
    batch: int,
    feat_h: int,
    feat_w: int,
) -> str:
    """``ASPP`` (``view_transform.py``): 4 atrous branches (dilations 1/6/12/18,
    32-group GroupNorm) + a global-average-pool branch resized back up, concat,
    project back to ``inplanes`` channels. ``(batch, feat_h, feat_w)`` is
    ``x4``'s (every branch's) static shape, target for the global-pool
    branch's ``Resize`` (``F.interpolate(size=x4.shape[2:],
    align_corners=True)``)."""

    def branch(idx: int, kernel: int, dilation: int) -> str:
        p = f"{prefix}.aspp{idx}"
        pad = dilation if kernel == 3 else 0
        h = _conv(
            b,
            x,
            declare(f"{p}.atrous_conv.weight"),
            None,
            f"{p}.conv",
            strides=[1, 1],
            pads=[pad, pad],
            dilations=[dilation, dilation],
        )
        h = _groupnorm(
            b,
            h,
            declare(f"{p}.bn.weight"),
            declare(f"{p}.bn.bias"),
            32,
            aspp_mid_channels,
            1e-5,
            f"{p}.gn",
            spatial_dims=2,
        )
        return _relu(b, h, f"{p}.relu")

    x1 = branch(1, 1, 1)
    x2 = branch(2, 3, 6)
    x3 = branch(3, 3, 12)
    x4 = branch(4, 3, 18)

    gp = _reduce_mean(b, x, [2, 3], 1, f"{prefix}.gap")
    gp = _conv(
        b,
        gp,
        declare(f"{prefix}.global_avg_pool.1.weight"),
        None,
        f"{prefix}.gap_conv",
        strides=[1, 1],
        pads=[0, 0],
    )
    gp = _groupnorm(
        b,
        gp,
        declare(f"{prefix}.global_avg_pool.2.weight"),
        declare(f"{prefix}.global_avg_pool.2.bias"),
        32,
        aspp_mid_channels,
        1e-5,
        f"{prefix}.gap_gn",
        spatial_dims=2,
    )
    gp = _relu(b, gp, f"{prefix}.gap_relu")
    x5 = _resize_to_shape(
        b,
        gp,
        [batch, aspp_mid_channels, feat_h, feat_w],
        align_corners=True,
        prefix=f"{prefix}.gap_resize",
    )

    cat = b.op("Concat", [x1, x2, x3, x4, x5], f"{prefix}.cat", axis=1)
    h = _conv(
        b,
        cat,
        declare(f"{prefix}.conv1.weight"),
        None,
        f"{prefix}.proj",
        strides=[1, 1],
        pads=[0, 0],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.bn1.weight"),
        declare(f"{prefix}.bn1.bias"),
        32,
        inplanes,
        1e-5,
        f"{prefix}.proj_gn",
        spatial_dims=2,
    )
    return _relu(b, h, f"{prefix}.proj_relu")


def _depth_net(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    mid_channels: int,
    depth_channels: int,
    aspp_mid_channels: int,
    batch: int,
    feat_h: int,
    feat_w: int,
) -> str:
    h = _conv(
        b,
        x,
        declare(f"{prefix}.reduce_conv.0.weight"),
        declare(f"{prefix}.reduce_conv.0.bias"),
        f"{prefix}.reduce_conv0",
        strides=[1, 1],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.reduce_conv.1.weight"),
        declare(f"{prefix}.reduce_conv.1.bias"),
        32,
        mid_channels,
        1e-5,
        f"{prefix}.reduce_gn",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.reduce_relu")

    h = _basic_block_2d_gn(
        b,
        h,
        declare,
        f"{prefix}.depth_conv.0",
        mid_channels,
        32,
        stride=1,
        has_downsample=False,
    )
    h = _basic_block_2d_gn(
        b,
        h,
        declare,
        f"{prefix}.depth_conv.1",
        mid_channels,
        32,
        stride=1,
        has_downsample=False,
    )
    h = _basic_block_2d_gn(
        b,
        h,
        declare,
        f"{prefix}.depth_conv.2",
        mid_channels,
        32,
        stride=1,
        has_downsample=False,
    )
    h = _aspp(
        b,
        h,
        declare,
        f"{prefix}.depth_conv.3",
        mid_channels,
        aspp_mid_channels,
        batch,
        feat_h,
        feat_w,
    )
    h = _conv(
        b,
        h,
        declare(f"{prefix}.depth_conv.4.weight"),
        declare(f"{prefix}.depth_conv.4.bias"),
        f"{prefix}.depth_out",
        strides=[1, 1],
        pads=[0, 0],
    )
    return h


# ---------------------------------------------------------------------------
# Build-time geometry precompute (see this module's own docstring, point 1):
# every quantity below depends only on fixed config and caller-supplied,
# deployment-static camera calibration -- never on the image content -- so
# it is computed once in plain numpy and baked into the graph as constants.


def _build_frustum_grid(
    frustum_range: Sequence[float], frustum_size: Sequence[float]
) -> Tuple[np.ndarray, int, int, int]:
    """The ``(W, H, D, 3)`` frustum grid ``view_transform.py``'s own
    ``Uni3DVoxelPoolDepth.frustum`` property builds via
    ``torch.meshgrid(..., indexing="ij")`` -- ``W``/``H`` are image-plane
    pixel bins, ``D`` is depth bins."""
    w_vals = np.arange(
        frustum_range[0], frustum_range[3], frustum_size[0], dtype=np.float64
    )
    h_vals = np.arange(
        frustum_range[1], frustum_range[4], frustum_size[1], dtype=np.float64
    )
    d_vals = np.arange(
        frustum_range[2], frustum_range[5], frustum_size[2], dtype=np.float64
    )
    w, h, d = len(w_vals), len(h_vals), len(d_vals)
    grid = np.stack(np.meshgrid(w_vals, h_vals, d_vals, indexing="ij"), axis=-1)
    return grid, w, h, d


def _precompute_voxel_pool_indices(
    frustum_range: Sequence[float],
    frustum_size: Sequence[float],
    img2ego: np.ndarray,
    pc_range: Sequence[float],
    voxel_size: Sequence[float],
    voxel_shape: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Reproduces ``Uni3DVoxelPoolDepth.coord_preparing`` (frustum
    unprojection into ego-frame voxel indices) followed by the indexing
    ``voxel_pool_depth`` (``ops/voxel_pool/src/voxel_pool.cpp``) itself does
    to turn ``(point_indices, coords)`` into a scatter -- but entirely in
    numpy, since every input here is build-time-static (see this section's
    own header comment). ``img2ego`` is ``lidar2ego @ inv(lidar2img)`` per
    camera, i.e. the *caller's* job to invert (a deployment-fixed 4x4, not
    a per-call one -- see this module's docstring, point 1).

    Returns ``(scatter_indices[N,3], valid_mask[N], feat_gather_index[N],
    depth_gather_index[N], W, H, D)`` where ``N = num_cams * W * H * D``:
    for point ``i``, the pooled voxel gets
    ``feat[feat_gather_index[i]] * depth[depth_gather_index[i]] *
    valid_mask[i]`` added at ``scatter_indices[i]`` (``feat``/``depth``
    flattened as this module's own view-transform builder does).
    """
    grid, w, h, d = _build_frustum_grid(frustum_range, frustum_size)
    num_cams = img2ego.shape[0]
    u, v, dep = grid[..., 0], grid[..., 1], grid[..., 2]
    homog = np.stack([u * dep, v * dep, dep, np.ones_like(dep)], axis=-1).reshape(
        -1, 4
    )  # (W*H*D, 4)
    pts = np.einsum("cij,pj->cpi", img2ego.astype(np.float64), homog)[
        ..., :3
    ]  # (num_cams, W*H*D, 3)

    voxel_f = (pts - np.asarray(pc_range[:3], dtype=np.float64)) / np.asarray(
        voxel_size, dtype=np.float64
    )
    voxel_idx = voxel_f.astype(
        np.int64
    )  # truncation toward zero -- matches torch's `.int()`
    x, y, z = voxel_shape
    valid = (
        (voxel_idx[..., 0] >= 0)
        & (voxel_idx[..., 0] < x)
        & (voxel_idx[..., 1] >= 0)
        & (voxel_idx[..., 1] < y)
        & (voxel_idx[..., 2] >= 0)
        & (voxel_idx[..., 2] < z)
    )
    clipped = np.clip(voxel_idx, [0, 0, 0], [x - 1, y - 1, z - 1])

    iw, ih, idd = (
        a.reshape(-1)
        for a in np.meshgrid(np.arange(w), np.arange(h), np.arange(d), indexing="ij")
    )
    cam_idx = np.repeat(np.arange(num_cams), w * h * d)
    feat_gather_index = (cam_idx * h * w + np.tile(ih * w + iw, num_cams)).astype(
        np.int64
    )
    depth_gather_index = (
        cam_idx * d * h * w + np.tile(idd * h * w + ih * w + iw, num_cams)
    ).astype(np.int64)
    scatter_indices = clipped.reshape(-1, 3).astype(np.int64)
    valid_mask = valid.reshape(-1).astype(np.float32)
    return scatter_indices, valid_mask, feat_gather_index, depth_gather_index, w, h, d


def _precompute_point_sampling(
    bev_h: int,
    bev_w: int,
    pc_range: Sequence[float],
    num_points_in_pillar: int,
    ego2img: np.ndarray,
    img_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces ``BEVFormerEncoder.get_reference_points(dim="3d")`` +
    ``point_sampling`` entirely in numpy (again, every input is
    build-time-static -- see this section's own header comment).
    ``ego2img`` is ``lidar2img @ inv(lidar2ego)`` per camera (the opposite
    composition from :func:`_precompute_voxel_pool_indices`'s ``img2ego``:
    this one projects ego points *into* each camera, that one unprojects
    camera frustum cells *into* ego).

    Returns ``(reference_points_cam[num_cams, bev_h*bev_w,
    num_points_in_pillar, 2], mask_any[num_cams, bev_h*bev_w], count[bev_h*
    bev_w])`` -- ``reference_points_cam`` in ``[0, 1]`` normalized pixel
    coordinates (garbage where invalid, always paired with ``mask_any``,
    exactly like the real ``bev_mask``), ``count`` already
    ``clamp(min=1)``-ed as ``SpatialCrossAttention.forward`` does.
    """
    height, width = img_shape
    z_len = pc_range[5] - pc_range[2]
    zs = np.linspace(0.5, z_len - 0.5, num_points_in_pillar, dtype=np.float64) / z_len
    xs = (np.arange(bev_w, dtype=np.float64) + 0.5) / bev_w
    ys = (np.arange(bev_h, dtype=np.float64) + 0.5) / bev_h

    grid_x, grid_y = np.meshgrid(
        xs, ys
    )  # (bev_h, bev_w) each, grid_x[i,j]=xs[j], grid_y[i,j]=ys[i]
    ref_hw = np.stack([grid_x, grid_y], axis=-1).reshape(bev_h * bev_w, 2)
    num_query = bev_h * bev_w
    ref = np.empty((num_points_in_pillar, num_query, 3), dtype=np.float64)
    ref[:, :, 0:2] = ref_hw[None, :, :]
    ref[:, :, 2] = zs[:, None]

    ref[..., 0] = ref[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref[..., 1] = ref[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref[..., 2] = ref[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    homog = np.concatenate([ref, np.ones_like(ref[..., :1])], axis=-1)  # (D, Q, 4)

    cam_pts = np.einsum(
        "cij,dqj->cdqi", ego2img.astype(np.float64), homog
    )  # (num_cams, D, Q, 4)
    depth = cam_pts[..., 2]
    eps = 1e-5
    bev_mask = depth > eps
    uv = cam_pts[..., :2] / np.maximum(depth[..., None], eps)
    uv_x = uv[..., 0] / width
    uv_y = uv[..., 1] / height
    bev_mask = bev_mask & (uv_y > 0.0) & (uv_y < 1.0) & (uv_x > 0.0) & (uv_x < 1.0)

    reference_points_cam = np.stack([uv_x, uv_y], axis=-1).transpose(
        0, 2, 1, 3
    )  # (num_cams, Q, D, 2)
    bev_mask = bev_mask.transpose(0, 2, 1)  # (num_cams, Q, D)
    mask_any = bev_mask.any(axis=-1)  # (num_cams, Q)
    count = np.clip(mask_any.sum(axis=0).astype(np.float32), 1.0, None)  # (Q,)
    return reference_points_cam.astype(np.float32), mask_any, count


# ---------------------------------------------------------------------------
# Uni3DVoxelPoolDepth (view_transform.py): frustum unprojection (precomputed,
# see above) + depth-weighted ScatterND pooling + a 3-layer Conv3d stack.


def _view_transform(
    b: _Builder,
    img_feat: str,
    img_depth: str,
    declare,
    prefix: str,
    embed_dim: int,
    num_cams: int,
    feat_h: int,
    feat_w: int,
    depth_dim: int,
    frustum_range: Sequence[float],
    frustum_size: Sequence[float],
    pc_range: Sequence[float],
    voxel_size: Sequence[float],
    voxel_shape: Sequence[int],
    img2ego: np.ndarray,
    num_conv_layers: int = 3,
) -> str:
    """``img_feat``: ``[num_cams, embed_dim, feat_h, feat_w]``. ``img_depth``:
    ``[num_cams, depth_dim, feat_h, feat_w]`` (already softmax-ed over
    ``depth_dim``). Returns the pooled+refined ego voxel volume,
    ``[embed_dim, Z, Y, X]`` (batch dim dropped -- this whole module family
    is single-sample; matches ``feat_encoding``'s own ``[B, C, D, H, W]``
    layout with ``B`` squeezed away)."""
    scatter_idx, valid_mask, feat_gather_idx, depth_gather_idx, w, h, d = (
        _precompute_voxel_pool_indices(
            frustum_range, frustum_size, img2ego, pc_range, voxel_size, voxel_shape
        )
    )
    if (w, h, d) != (feat_w, feat_h, depth_dim):
        raise UnsupportedArchitectureError(
            f"frustum grid (W={w}, H={h}, D={d}) does not match the feature/depth "
            f"map (feat_w={feat_w}, feat_h={feat_h}, depth_dim={depth_dim}) -- "
            "the ViT neck's spatial resolution must exactly match frustum_size's bins"
        )
    x_dim, y_dim, z_dim = voxel_shape

    feat_t = b.op("Transpose", [img_feat], f"{prefix}.feat_t", perm=[0, 2, 3, 1])
    feat_flat = b.op(
        "Reshape",
        [feat_t, b.shape_const([num_cams * h * w, embed_dim])],
        f"{prefix}.feat_flat",
    )
    depth_flat = b.op(
        "Reshape", [img_depth, b.shape_const([-1])], f"{prefix}.depth_flat"
    )

    feat_idx_c = b.const(feat_gather_idx, prefix="vp_feat_idx")
    depth_idx_c = b.const(depth_gather_idx, prefix="vp_depth_idx")
    gathered_feat = b.op(
        "Gather", [feat_flat, feat_idx_c], f"{prefix}.gathered_feat", axis=0
    )
    gathered_depth = _unsqueeze(
        b,
        b.op("Gather", [depth_flat, depth_idx_c], f"{prefix}.gathered_depth", axis=0),
        [-1],
        f"{prefix}.gathered_depth_u",
    )
    mask_c = b.const(valid_mask.reshape(-1, 1), prefix="vp_mask")
    weighted = b.op(
        "Mul",
        [b.op("Mul", [gathered_feat, gathered_depth], f"{prefix}.fd"), mask_c],
        f"{prefix}.weighted",
    )

    scatter_idx_c = b.const(scatter_idx, prefix="vp_scatter_idx")
    zeros_grid = b.const(
        np.zeros((x_dim, y_dim, z_dim, embed_dim), dtype=np.float32), prefix="vp_zeros"
    )
    voxel_space = b.op(
        "ScatterND",
        [zeros_grid, scatter_idx_c, weighted],
        f"{prefix}.scattered",
        reduction="add",
    )

    # (X, Y, Z, C) -> (1, C, Z, Y, X), matching Conv3d's NCDHW convention
    # with D=Z, H=Y, W=X (the real kernel's own final permute target).
    voxel_space = b.op(
        "Transpose", [voxel_space], f"{prefix}.chw_first", perm=[3, 2, 1, 0]
    )
    voxel_space = _unsqueeze(b, voxel_space, [0], f"{prefix}.batched")

    for i in range(num_conv_layers):
        p = f"{prefix}.conv_layer.{i}.0"
        conv_bias = declare(f"{p}.bias")
        voxel_space = _conv(
            b,
            voxel_space,
            declare(f"{p}.weight"),
            conv_bias,
            f"{prefix}.conv{i}",
            strides=[1, 1, 1],
            pads=[1, 1, 1],
        )
        voxel_space = _batchnorm(
            b,
            voxel_space,
            declare(f"{prefix}.conv_layer.{i}.1.weight"),
            declare(f"{prefix}.conv_layer.{i}.1.bias"),
            declare(f"{prefix}.conv_layer.{i}.1.running_mean"),
            declare(f"{prefix}.conv_layer.{i}.1.running_var"),
            1e-5,
            f"{prefix}.bn{i}",
        )
        voxel_space = _relu(b, voxel_space, f"{prefix}.relu{i}")

    return b.op(
        "Squeeze",
        [voxel_space, b.const(np.array([0], dtype=np.int64), prefix="squeeze0")],
        f"{prefix}.unbatched",
    )


# ---------------------------------------------------------------------------
# Multi-scale deformable attention (attention.py / layers.py), built on ONNX's
# native GridSample -- confirmed exactly equivalent to the real CUDA kernel's
# per-point bilinear gather-and-accumulate (see this module's own docstring).


def _ms_deform_attn_sample(
    b: _Builder,
    value: str,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_locations: str,
    attention_weights: str,
    batch: int,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    num_levels: int,
    num_points: int,
    prefix: str,
) -> str:
    """A 1:1 ONNX translation of ``layers.py::multi_scale_deformable_attn_pytorch``.

    ``value``: ``[batch, sum(Hl*Wl), num_heads, head_dim]``.
    ``sampling_locations``: ``[batch, num_queries, num_heads, num_levels,
    num_points, 2]``, in ``[0, 1]`` (this function applies the ``2x - 1``
    conversion to ``GridSample``'s ``[-1, 1]`` convention itself, matching
    the reference's own ``sampling_grids = 2 * sampling_locations - 1``).
    ``attention_weights``: ``[batch, num_queries, num_heads, num_levels,
    num_points]``. ``spatial_shapes`` is a plain Python list of ``(Hl, Wl)``
    -- static, unlike the real function's own runtime tensor, since every
    level's feature map size is a build-time constant in this whole module.
    Returns ``[batch, num_queries, num_heads * head_dim]``.
    """
    two = b.const(np.array(2.0, dtype=np.float32), prefix="msda_two")
    one = b.const(np.array(1.0, dtype=np.float32), prefix="msda_one")
    sampling_grids = b.op(
        "Sub",
        [b.op("Mul", [sampling_locations, two], f"{prefix}.grids2"), one],
        f"{prefix}.grids",
    )

    level_outputs = []
    value_offset = 0
    for level, (h_l, w_l) in enumerate(spatial_shapes):
        hw_l = h_l * w_l
        value_l = _slice_axis(
            b, value, 1, value_offset, value_offset + hw_l, f"{prefix}.vsl{level}"
        )
        value_offset += hw_l
        v = b.op(
            "Reshape",
            [value_l, b.shape_const([batch, hw_l, num_heads * head_dim])],
            f"{prefix}.v_merge{level}",
        )
        v = b.op("Transpose", [v], f"{prefix}.v_t{level}", perm=[0, 2, 1])
        v = b.op(
            "Reshape",
            [v, b.shape_const([batch * num_heads, head_dim, h_l, w_l])],
            f"{prefix}.v_r{level}",
        )

        g = _slice_axis(b, sampling_grids, 3, level, level + 1, f"{prefix}.gsl{level}")
        g = _squeeze(b, g, [3], f"{prefix}.gsq{level}")
        g = b.op("Transpose", [g], f"{prefix}.g_t{level}", perm=[0, 2, 1, 3, 4])
        g = b.op(
            "Reshape",
            [g, b.shape_const([batch * num_heads, num_queries, num_points, 2])],
            f"{prefix}.g_r{level}",
        )

        sampled = _grid_sample(
            b,
            v,
            g,
            f"{prefix}.sample{level}",
            align_corners=False,
            padding_mode="zeros",
        )
        level_outputs.append(_unsqueeze(b, sampled, [3], f"{prefix}.u{level}"))

    stacked = b.op("Concat", level_outputs, f"{prefix}.stacked", axis=3)
    stacked = b.op(
        "Reshape",
        [
            stacked,
            b.shape_const(
                [batch * num_heads, head_dim, num_queries, num_levels * num_points]
            ),
        ],
        f"{prefix}.stacked_flat",
    )

    aw = b.op("Transpose", [attention_weights], f"{prefix}.aw_t", perm=[0, 2, 1, 3, 4])
    aw = b.op(
        "Reshape",
        [
            aw,
            b.shape_const([batch * num_heads, 1, num_queries, num_levels * num_points]),
        ],
        f"{prefix}.aw_r",
    )

    out = _reduce_sum(
        b, b.op("Mul", [stacked, aw], f"{prefix}.weighted"), [-1], 0, f"{prefix}.summed"
    )
    out = b.op(
        "Reshape",
        [out, b.shape_const([batch, num_heads * head_dim, num_queries])],
        f"{prefix}.merged",
    )
    return b.op("Transpose", [out], prefix, perm=[0, 2, 1])


# ---------------------------------------------------------------------------
# BEVFormer encoder attentions (attention.py). Every builder below is scoped
# to batch=1 (this whole module family's scope -- see the module docstring),
# so "batch" arguments to _ms_deform_attn_sample below are really "how many
# independent attention instances share this query/value" (2 BEV-queue slots
# for TSA, num_cams for the dense SCA, 1 for the decoder's cross-attention),
# not a real batch dimension.


def _bev_positional_encoding(
    b: _Builder, declare, prefix: str, bev_h: int, bev_w: int, num_feats: int
) -> str:
    """``LearnedPositionalEncoding`` (``layers.py``) with a fixed, all-zero
    ``mask`` (confirmed from ``heads.py::BEVFormerHead.forward``:
    ``bev_mask = torch.zeros(...)``) -- so every row/col embedding index is
    simply ``arange``, and the whole embedding table is used unchanged
    (no ``Gather`` needed). Returns ``[bev_h*bev_w, 2*num_feats]``."""
    row_w = declare(f"{prefix}.row_embed.weight")  # (bev_h, num_feats)
    col_w = declare(f"{prefix}.col_embed.weight")  # (bev_w, num_feats)
    x_embed = b.op(
        "Tile",
        [
            _unsqueeze(b, col_w, [0], f"{prefix}.col_u"),
            b.const(np.array([bev_h, 1, 1], dtype=np.int64), prefix="tile_reps"),
        ],
        f"{prefix}.x_embed",
    )
    y_embed = b.op(
        "Tile",
        [
            _unsqueeze(b, row_w, [1], f"{prefix}.row_u"),
            b.const(np.array([1, bev_w, 1], dtype=np.int64), prefix="tile_reps"),
        ],
        f"{prefix}.y_embed",
    )
    pos = b.op(
        "Concat", [x_embed, y_embed], f"{prefix}.pos", axis=-1
    )  # (bev_h, bev_w, 2*num_feats)
    return b.op("Reshape", [pos, b.shape_const([bev_h * bev_w, 2 * num_feats])], prefix)


def _temporal_self_attention(
    b: _Builder,
    query: str,
    query_pos: str,
    declare,
    prefix: str,
    num_query: int,
    embed_dims: int,
    num_heads: int,
    num_points: int,
    bev_h: int,
    bev_w: int,
) -> str:
    """``TemporalSelfAttention`` (``attention.py``), single-frame degenerate
    path (``value = stack([query, query])``, confirmed from source -- this
    module's scope is always a single frame, see the module docstring)."""
    head_dim = embed_dims // num_heads
    identity = query
    query_with_pos = b.op("Add", [query, query_pos], f"{prefix}.query_pos")

    value_proj = _linear(
        b,
        query,
        declare(f"{prefix}.value_proj.weight"),
        declare(f"{prefix}.value_proj.bias"),
        f"{prefix}.value_proj",
    )
    value_dup = b.op(
        "Concat", [value_proj, value_proj], f"{prefix}.value_dup", axis=0
    )  # (2, num_query, embed_dims)
    value_dup = b.op(
        "Reshape",
        [value_dup, b.shape_const([2, num_query, num_heads, head_dim])],
        f"{prefix}.value_r",
    )

    query_cat = b.op(
        "Concat", [query, query_with_pos], f"{prefix}.query_cat", axis=-1
    )  # (num_query, 2*embed_dims)
    off_raw = _linear(
        b,
        query_cat,
        declare(f"{prefix}.sampling_offsets.weight"),
        declare(f"{prefix}.sampling_offsets.bias"),
        f"{prefix}.offsets",
    )
    aw_raw = _linear(
        b,
        query_cat,
        declare(f"{prefix}.attention_weights.weight"),
        declare(f"{prefix}.attention_weights.bias"),
        f"{prefix}.aw",
    )

    off = b.op(
        "Reshape",
        [off_raw, b.shape_const([num_query, num_heads, 2, 1, num_points, 2])],
        f"{prefix}.off_r",
    )
    aw = b.op(
        "Reshape",
        [aw_raw, b.shape_const([num_query, num_heads, 2, 1 * num_points])],
        f"{prefix}.aw_r0",
    )
    aw = _softmax(b, aw, -1, f"{prefix}.aw_softmax")
    aw = b.op(
        "Reshape",
        [aw, b.shape_const([num_query, num_heads, 2, 1, num_points])],
        f"{prefix}.aw_r1",
    )

    def slot(t: str, idx: int, name: str) -> str:
        return _squeeze(
            b,
            _slice_axis(b, t, 2, idx, idx + 1, f"{prefix}.{name}{idx}"),
            [2],
            f"{prefix}.{name}{idx}sq",
        )

    off0, off1 = slot(off, 0, "off_slot"), slot(off, 1, "off_slot")
    aw0, aw1 = slot(aw, 0, "aw_slot"), slot(aw, 1, "aw_slot")
    # (num_query, heads, levels, points[, 2]) -> (1, num_query, heads, levels, points[, 2]) per slot, concat -> batch=2
    off_cat = b.op(
        "Concat",
        [
            _unsqueeze(b, off0, [0], f"{prefix}.off0u"),
            _unsqueeze(b, off1, [0], f"{prefix}.off1u"),
        ],
        f"{prefix}.off_cat",
        axis=0,
    )
    aw_cat = b.op(
        "Concat",
        [
            _unsqueeze(b, aw0, [0], f"{prefix}.aw0u"),
            _unsqueeze(b, aw1, [0], f"{prefix}.aw1u"),
        ],
        f"{prefix}.aw_cat",
        axis=0,
    )

    ref_2d = _build_time_bev_ref_2d_const(
        b, bev_h, bev_w, prefix
    )  # (1, num_query, 1, 1, 1, 2), see below
    offset_normalizer = b.const(
        np.array([bev_w, bev_h], dtype=np.float32).reshape(1, 1, 1, 1, 1, 2),
        prefix="tsa_offset_norm",
    )
    sampling_locations = b.op(
        "Add",
        [ref_2d, b.op("Div", [off_cat, offset_normalizer], f"{prefix}.off_norm")],
        f"{prefix}.locs",
    )

    out = _ms_deform_attn_sample(
        b,
        value_dup,
        [(bev_h, bev_w)],
        sampling_locations,
        aw_cat,
        batch=2,
        num_queries=num_query,
        num_heads=num_heads,
        head_dim=head_dim,
        num_levels=1,
        num_points=num_points,
        prefix=f"{prefix}.msda",
    )  # (2, num_query, embed_dims)
    out0 = _squeeze(
        b, _slice_axis(b, out, 0, 0, 1, f"{prefix}.out0"), [0], f"{prefix}.out0sq"
    )
    out1 = _squeeze(
        b, _slice_axis(b, out, 0, 1, 2, f"{prefix}.out1"), [0], f"{prefix}.out1sq"
    )
    half = b.const(np.array(0.5, dtype=np.float32), prefix="half")
    averaged = b.op(
        "Mul", [b.op("Add", [out0, out1], f"{prefix}.sum2"), half], f"{prefix}.avg"
    )

    projected = _linear(
        b,
        averaged,
        declare(f"{prefix}.output_proj.weight"),
        declare(f"{prefix}.output_proj.bias"),
        f"{prefix}.output_proj",
    )
    return b.op("Add", [projected, identity], prefix)


_BEV_REF_2D_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _build_time_bev_ref_2d_const(
    b: _Builder, bev_h: int, bev_w: int, prefix: str
) -> str:
    """``BEVFormerEncoder.get_reference_points(dim="2d")`` -- purely a
    function of ``(bev_h, bev_w)``, so a build-time constant. Shape
    ``[1, bev_h*bev_w, 1, 1, 1, 2]`` (pre-broadcast for
    :func:`_temporal_self_attention`'s ``sampling_locations`` add -- both
    BEV-queue slots use the identical reference points, confirmed from
    source's own ``hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1)``)."""
    key = (bev_h, bev_w)
    if key not in _BEV_REF_2D_CACHE:
        xs = (np.arange(bev_w, dtype=np.float32) + 0.5) / bev_w
        ys = (np.arange(bev_h, dtype=np.float32) + 0.5) / bev_h
        grid_x, grid_y = np.meshgrid(xs, ys)
        ref = np.stack([grid_x, grid_y], axis=-1).reshape(1, bev_h * bev_w, 1, 1, 1, 2)
        _BEV_REF_2D_CACHE[key] = ref
    return b.const(_BEV_REF_2D_CACHE[key], prefix=f"{prefix}.ref_2d")


def _spatial_cross_attention(
    b: _Builder,
    query: str,
    query_pos: str,
    key: str,
    value: str,
    declare,
    prefix: str,
    num_query: int,
    num_cams: int,
    embed_dims: int,
    spatial_shapes: Sequence[Tuple[int, int]],
    reference_points_cam: np.ndarray,
    mask_any: np.ndarray,
    count: np.ndarray,
) -> str:
    """``SpatialCrossAttention`` (``attention.py``) + its inner
    ``MSDeformableAttention3D``, as **dense masked evaluation** rather than
    the real module's data-dependent rebatch -- see this module's own
    docstring, point 2, for why this is exact, not an approximation.

    ``key``/``value``: ``[num_cams, sum(Hl*Wl), embed_dims]`` (bs=1 folded
    away). ``reference_points_cam``/``mask_any``/``count``: build-time numpy
    arrays from :func:`_precompute_point_sampling`.
    """
    num_heads, num_levels, num_points = NUM_HEADS, NUM_FEATURE_LEVELS, SCA_NUM_POINTS
    head_dim = embed_dims // num_heads
    num_z = reference_points_cam.shape[2]  # points_in_pillar
    inp_residual = query
    query = b.op("Add", [query, query_pos], f"{prefix}.query_pos")

    value_proj = _linear(
        b,
        value,
        declare(f"{prefix}.deformable_attention.value_proj.weight"),
        declare(f"{prefix}.deformable_attention.value_proj.bias"),
        f"{prefix}.value_proj",
    )
    sum_hw = sum(h * w for h, w in spatial_shapes)
    value_proj = b.op(
        "Reshape",
        [value_proj, b.shape_const([num_cams, sum_hw, num_heads, head_dim])],
        f"{prefix}.value_r",
    )

    off_raw = _linear(
        b,
        query,
        declare(f"{prefix}.deformable_attention.sampling_offsets.weight"),
        declare(f"{prefix}.deformable_attention.sampling_offsets.bias"),
        f"{prefix}.offsets",
    )
    aw_raw = _linear(
        b,
        query,
        declare(f"{prefix}.deformable_attention.attention_weights.weight"),
        declare(f"{prefix}.deformable_attention.attention_weights.bias"),
        f"{prefix}.aw",
    )

    off = b.op(
        "Reshape",
        [off_raw, b.shape_const([num_query, num_heads, num_levels, num_points, 2])],
        f"{prefix}.off_r",
    )
    offset_normalizer = b.const(
        np.array([[w, h] for h, w in spatial_shapes], dtype=np.float32).reshape(
            1, 1, num_levels, 1, 2
        ),
        prefix="sca_offset_norm",
    )
    off = b.op("Div", [off, offset_normalizer], f"{prefix}.off_norm")
    off = b.op(
        "Reshape",
        [
            off,
            b.shape_const(
                [num_query, num_heads, num_levels, num_points // num_z, num_z, 2]
            ),
        ],
        f"{prefix}.off_zsplit",
    )
    off = _unsqueeze(
        b, off, [0], f"{prefix}.off_u"
    )  # (1, num_query, heads, levels, points/D, D, 2)

    aw = b.op(
        "Reshape",
        [aw_raw, b.shape_const([num_query, num_heads, num_levels * num_points])],
        f"{prefix}.aw_r0",
    )
    aw = _softmax(b, aw, -1, f"{prefix}.aw_softmax")
    aw = b.op(
        "Reshape",
        [aw, b.shape_const([num_query, num_heads, num_levels, num_points])],
        f"{prefix}.aw_r1",
    )
    aw_expanded = b.op(
        "Expand",
        [
            _unsqueeze(b, aw, [0], f"{prefix}.aw_u"),
            b.shape_const([num_cams, num_query, num_heads, num_levels, num_points]),
        ],
        f"{prefix}.aw_expand",
    )

    ref_const = b.const(
        reference_points_cam.reshape(num_cams, num_query, 1, 1, 1, num_z, 2).astype(
            np.float32
        ),
        prefix="sca_ref_cam",
    )
    sampling_locations = b.op("Add", [ref_const, off], f"{prefix}.locs_zsplit")
    sampling_locations = b.op(
        "Reshape",
        [
            sampling_locations,
            b.shape_const([num_cams, num_query, num_heads, num_levels, num_points, 2]),
        ],
        f"{prefix}.locs",
    )

    sampled = _ms_deform_attn_sample(
        b,
        value_proj,
        spatial_shapes,
        sampling_locations,
        aw_expanded,
        batch=num_cams,
        num_queries=num_query,
        num_heads=num_heads,
        head_dim=head_dim,
        num_levels=num_levels,
        num_points=num_points,
        prefix=f"{prefix}.msda",
    )  # (num_cams, num_query, embed_dims)

    mask_const = b.const(
        mask_any.astype(np.float32).reshape(num_cams, num_query, 1),
        prefix="sca_mask_any",
    )
    masked = b.op("Mul", [sampled, mask_const], f"{prefix}.masked")
    summed = _reduce_sum(
        b, masked, [0], 0, f"{prefix}.summed_over_cams"
    )  # (num_query, embed_dims)
    count_const = b.const(count.reshape(num_query, 1), prefix="sca_count")
    slots = b.op("Div", [summed, count_const], f"{prefix}.slots")

    projected = _linear(
        b,
        slots,
        declare(f"{prefix}.output_proj.weight"),
        declare(f"{prefix}.output_proj.bias"),
        f"{prefix}.output_proj",
    )
    return b.op("Add", [projected, inp_residual], prefix)


def _custom_ms_deformable_attention(
    b: _Builder,
    query: str,
    query_pos: str,
    value: str,
    declare,
    prefix: str,
    num_query: int,
    num_value: int,
    embed_dims: int,
    spatial_shapes: Sequence[Tuple[int, int]],
    reference_points: str,
) -> str:
    """``CustomMSDeformableAttention`` (``attention.py``) -- the detection
    decoder's deformable cross-attention into the BEV embedding. Unlike
    ``SpatialCrossAttention``, ``reference_points`` here is a genuine
    **runtime** tensor (iteratively refined per decoder layer from the
    previous layer's box predictions), not a build-time constant."""
    num_heads, num_levels, num_points = NUM_HEADS, 1, DECODER_NUM_POINTS
    head_dim = embed_dims // num_heads
    identity = query
    query = b.op("Add", [query, query_pos], f"{prefix}.query_pos")

    value_proj = _linear(
        b,
        value,
        declare(f"{prefix}.value_proj.weight"),
        declare(f"{prefix}.value_proj.bias"),
        f"{prefix}.value_proj",
    )
    value_proj = b.op(
        "Reshape",
        [value_proj, b.shape_const([1, num_value, num_heads, head_dim])],
        f"{prefix}.value_r",
    )

    off_raw = _linear(
        b,
        query,
        declare(f"{prefix}.sampling_offsets.weight"),
        declare(f"{prefix}.sampling_offsets.bias"),
        f"{prefix}.offsets",
    )
    aw_raw = _linear(
        b,
        query,
        declare(f"{prefix}.attention_weights.weight"),
        declare(f"{prefix}.attention_weights.bias"),
        f"{prefix}.aw",
    )
    off = b.op(
        "Reshape",
        [off_raw, b.shape_const([1, num_query, num_heads, num_levels, num_points, 2])],
        f"{prefix}.off_r",
    )
    aw = b.op(
        "Reshape",
        [aw_raw, b.shape_const([1, num_query, num_heads, num_levels * num_points])],
        f"{prefix}.aw_r0",
    )
    aw = _softmax(b, aw, -1, f"{prefix}.aw_softmax")
    aw = b.op(
        "Reshape",
        [aw, b.shape_const([1, num_query, num_heads, num_levels, num_points])],
        f"{prefix}.aw_r1",
    )

    offset_normalizer = b.const(
        np.array([[w, h] for h, w in spatial_shapes], dtype=np.float32).reshape(
            1, 1, 1, num_levels, 1, 2
        ),
        prefix="dec_offset_norm",
    )
    ref = b.op(
        "Reshape",
        [reference_points, b.shape_const([1, num_query, 1, num_levels, 1, 2])],
        f"{prefix}.ref_r",
    )
    sampling_locations = b.op(
        "Add",
        [ref, b.op("Div", [off, offset_normalizer], f"{prefix}.off_norm")],
        f"{prefix}.locs",
    )

    sampled = _ms_deform_attn_sample(
        b,
        value_proj,
        spatial_shapes,
        sampling_locations,
        aw,
        batch=1,
        num_queries=num_query,
        num_heads=num_heads,
        head_dim=head_dim,
        num_levels=num_levels,
        num_points=num_points,
        prefix=f"{prefix}.msda",
    )
    sampled = _squeeze(b, sampled, [0], f"{prefix}.sampled_sq")
    projected = _linear(
        b,
        sampled,
        declare(f"{prefix}.output_proj.weight"),
        declare(f"{prefix}.output_proj.bias"),
        f"{prefix}.output_proj",
    )
    return b.op("Add", [projected, identity], prefix)


def _read_array(entries, name: str) -> np.ndarray:
    return onnx.numpy_helper.to_array(_read_tensor(entries[name], name)).astype(
        np.float32
    )


def _multihead_self_attention(
    b: _Builder,
    query: str,
    query_pos: str,
    entries,
    prefix: str,
    num_query: int,
    embed_dims: int,
    num_heads: int,
) -> str:
    """``layers.py::MultiheadAttention`` wrapping ``torch.nn.MultiheadAttention``
    for plain (non-deformable) self-attention -- confirmed call convention
    (``BaseTransformerLayer``'s ``self_attn`` branch): ``query=key=(original
    query + query_pos)``, ``value=original query`` (no positional embedding
    on the value stream, standard DETR-style). ``in_proj_weight``/``bias``
    (PyTorch's own combined QKV parameter) are split into separate Q/K/V
    weights **in numpy at declare time** (real values, known once read from
    the checkpoint -- no runtime ``Slice`` needed for a checkpoint tensor)."""
    head_dim = embed_dims // num_heads
    in_proj_w = _read_array(entries, f"{prefix}.in_proj_weight")
    in_proj_b = _read_array(entries, f"{prefix}.in_proj_bias")
    wq, wk, wv = np.split(in_proj_w, 3, axis=0)
    bq, bk, bv = np.split(in_proj_b, 3, axis=0)

    identity = query
    qk_input = b.op("Add", [query, query_pos], f"{prefix}.qk_in")
    q = _linear(
        b,
        qk_input,
        b.const(wq, prefix="mha_wq"),
        b.const(bq, prefix="mha_bq"),
        f"{prefix}.q",
    )
    k = _linear(
        b,
        qk_input,
        b.const(wk, prefix="mha_wk"),
        b.const(bk, prefix="mha_bk"),
        f"{prefix}.k",
    )
    v = _linear(
        b,
        query,
        b.const(wv, prefix="mha_wv"),
        b.const(bv, prefix="mha_bv"),
        f"{prefix}.v",
    )

    def split_heads(t: str, name: str) -> str:
        t = b.op(
            "Reshape",
            [t, b.shape_const([num_query, num_heads, head_dim])],
            f"{prefix}.{name}_r",
        )
        return b.op("Transpose", [t], f"{prefix}.{name}_t", perm=[1, 0, 2])

    qh, kh, vh = split_heads(q, "q"), split_heads(k, "k"), split_heads(v, "v")
    inv_sqrt_d = b.const(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), prefix="mha_scale"
    )
    scores = b.op(
        "Mul",
        [
            b.op(
                "MatMul",
                [qh, b.op("Transpose", [kh], f"{prefix}.k_t2", perm=[0, 2, 1])],
                f"{prefix}.scores",
            ),
            inv_sqrt_d,
        ],
        f"{prefix}.scores_scaled",
    )
    attn = _softmax(b, scores, -1, f"{prefix}.softmax")
    out = b.op("MatMul", [attn, vh], f"{prefix}.attn_out")
    out = b.op("Transpose", [out], f"{prefix}.out_t", perm=[1, 0, 2])
    out = b.op(
        "Reshape", [out, b.shape_const([num_query, embed_dims])], f"{prefix}.out_r"
    )
    out_proj_w = _read_array(entries, f"{prefix}.out_proj.weight")
    out_proj_b = _read_array(entries, f"{prefix}.out_proj.bias")
    out = _linear(
        b,
        out,
        b.const(out_proj_w, prefix="mha_out_w"),
        b.const(out_proj_b, prefix="mha_out_b"),
        f"{prefix}.out_proj",
    )
    return b.op("Add", [out, identity], prefix)


def _ffn(b: _Builder, x: str, declare, prefix: str) -> str:
    """``FFN`` (``layers.py``): two linear layers, ReLU between, residual."""
    identity = x
    h = _linear(
        b,
        x,
        declare(f"{prefix}.layers.0.0.weight"),
        declare(f"{prefix}.layers.0.0.bias"),
        f"{prefix}.fc1",
    )
    h = _relu(b, h, f"{prefix}.relu")
    h = _linear(
        b,
        h,
        declare(f"{prefix}.layers.1.weight"),
        declare(f"{prefix}.layers.1.bias"),
        f"{prefix}.fc2",
    )
    return b.op("Add", [h, identity], prefix)


# ---------------------------------------------------------------------------
# BEVFormerEncoder / DetectionTransformerDecoder (bev_encoder.py): stack the
# attentions above into post-norm transformer layers, matching
# BaseTransformerLayer's fixed ("self_attn","norm","cross_attn","norm","ffn",
# "norm") operation order exactly.


def _bev_former_layer(
    b: _Builder,
    query: str,
    key: str,
    value: str,
    bev_pos: str,
    declare,
    entries,
    layer_prefix: str,
    num_query: int,
    embed_dims: int,
    bev_h: int,
    bev_w: int,
    spatial_shapes: Sequence[Tuple[int, int]],
    reference_points_cam: np.ndarray,
    mask_any: np.ndarray,
    count: np.ndarray,
) -> str:
    q = _temporal_self_attention(
        b,
        query,
        bev_pos,
        declare,
        f"{layer_prefix}.attentions.0",
        num_query,
        embed_dims,
        NUM_HEADS,
        TSA_NUM_POINTS,
        bev_h,
        bev_w,
    )
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.0.weight"),
        declare(f"{layer_prefix}.norms.0.bias"),
        1e-5,
        f"{layer_prefix}.norm0",
    )
    q = _spatial_cross_attention(
        b,
        q,
        bev_pos,
        key,
        value,
        declare,
        f"{layer_prefix}.attentions.1",
        num_query,
        len(reference_points_cam),
        embed_dims,
        spatial_shapes,
        reference_points_cam,
        mask_any,
        count,
    )
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.1.weight"),
        declare(f"{layer_prefix}.norms.1.bias"),
        1e-5,
        f"{layer_prefix}.norm1",
    )
    q = _ffn(b, q, declare, f"{layer_prefix}.ffns.0")
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.2.weight"),
        declare(f"{layer_prefix}.norms.2.bias"),
        1e-5,
        f"{layer_prefix}.norm2",
    )
    return q


def _bev_former_encoder(
    b: _Builder,
    bev_query: str,
    bev_pos: str,
    key: str,
    value: str,
    declare,
    entries,
    prefix: str,
    num_layers: int,
    num_query: int,
    embed_dims: int,
    bev_h: int,
    bev_w: int,
    spatial_shapes: Sequence[Tuple[int, int]],
    reference_points_cam: np.ndarray,
    mask_any: np.ndarray,
    count: np.ndarray,
) -> str:
    q = bev_query
    for i in range(num_layers):
        q = _bev_former_layer(
            b,
            q,
            key,
            value,
            bev_pos,
            declare,
            entries,
            f"{prefix}.layers.{i}",
            num_query,
            embed_dims,
            bev_h,
            bev_w,
            spatial_shapes,
            reference_points_cam,
            mask_any,
            count,
        )
    return q


def _detr_decoder_layer(
    b: _Builder,
    query: str,
    query_pos: str,
    value: str,
    declare,
    entries,
    layer_prefix: str,
    num_query: int,
    num_value: int,
    embed_dims: int,
    spatial_shapes: Sequence[Tuple[int, int]],
    reference_points: str,
) -> str:
    q = _multihead_self_attention(
        b,
        query,
        query_pos,
        entries,
        f"{layer_prefix}.attentions.0.attn",
        num_query,
        embed_dims,
        NUM_HEADS,
    )
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.0.weight"),
        declare(f"{layer_prefix}.norms.0.bias"),
        1e-5,
        f"{layer_prefix}.norm0",
    )
    q = _custom_ms_deformable_attention(
        b,
        q,
        query_pos,
        value,
        declare,
        f"{layer_prefix}.attentions.1",
        num_query,
        num_value,
        embed_dims,
        spatial_shapes,
        reference_points,
    )
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.1.weight"),
        declare(f"{layer_prefix}.norms.1.bias"),
        1e-5,
        f"{layer_prefix}.norm1",
    )
    q = _ffn(b, q, declare, f"{layer_prefix}.ffns.0")
    q = _layernorm_last_dim(
        b,
        q,
        declare(f"{layer_prefix}.norms.2.weight"),
        declare(f"{layer_prefix}.norms.2.bias"),
        1e-5,
        f"{layer_prefix}.norm2",
    )
    return q


def _inverse_sigmoid(b: _Builder, x: str, prefix: str, eps: float = 1e-5) -> str:
    zero = b.const(np.array(0.0, dtype=np.float32), prefix="isig_zero")
    one = b.const(np.array(1.0, dtype=np.float32), prefix="isig_one")
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="isig_eps")
    x = b.op("Clip", [x, zero, one], f"{prefix}.clip01")
    x1 = b.op("Max", [x, eps_c], f"{prefix}.x1")
    x2 = b.op("Max", [b.op("Sub", [one, x], f"{prefix}.1mx"), eps_c], f"{prefix}.x2")
    return b.op("Log", [b.op("Div", [x1, x2], f"{prefix}.ratio")], prefix)


def _detection_transformer_decoder(
    b: _Builder,
    query: str,
    query_pos: str,
    value: str,
    init_reference_points: str,
    declare,
    entries,
    prefix: str,
    num_layers: int,
    num_query: int,
    num_value: int,
    embed_dims: int,
    code_size: int,
    spatial_shapes: Sequence[Tuple[int, int]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """``DetectionTransformerDecoder``: iterative box-reference refinement
    per layer, using this layer's own ``reg_branches`` output (confirmed
    ``reg_branches`` is passed once, shared across layers by index -- one
    independent MLP per layer, ``num_pred = num_decoder_layers`` copies).

    Real source calls ``reg_branches[lid](output)`` *twice* on the same
    ``output`` with the same weights -- once here (to refine the next
    layer's reference point) and again in ``BEVFormerHead.forward`` (to
    produce that layer's final box, additionally pc_range-denormalized).
    Since both calls are provably identical, this builder computes it once
    and returns it (``reg_out``) for the caller to reuse for the final
    per-layer box output, alongside ``used_ref`` -- the reference points
    *this* layer's cross-attention actually used (``init_reference`` for
    layer 0, else the previous layer's updated reference -- confirmed from
    ``BEVFormerHead.forward``'s own ``lvl==0`` special case) -- rather than
    re-deriving that indexing at the call site.
    """
    output = query
    reference_points = init_reference_points
    hs, used_ref, reg_out, new_ref_list = [], [], [], []
    for lid in range(num_layers):
        used_ref.append(reference_points)
        ref_input = _slice_last_dim(b, reference_points, 0, 2, f"{prefix}.ref_xy.{lid}")
        output = _detr_decoder_layer(
            b,
            output,
            query_pos,
            value,
            declare,
            entries,
            f"{prefix}.layers.{lid}",
            num_query,
            num_value,
            embed_dims,
            spatial_shapes,
            ref_input,
        )
        tmp = _reg_branch(
            b, output, declare, f"{prefix}.reg_branches.{lid}", embed_dims, code_size
        )
        ref_sig_inv = _inverse_sigmoid(b, reference_points, f"{prefix}.ref_isig.{lid}")
        new_xy = b.op(
            "Add",
            [
                _slice_last_dim(b, tmp, 0, 2, f"{prefix}.tmp_xy.{lid}"),
                _slice_last_dim(b, ref_sig_inv, 0, 2, f"{prefix}.ref_isig_xy.{lid}"),
            ],
            f"{prefix}.new_xy.{lid}",
        )
        new_z = b.op(
            "Add",
            [
                _slice_last_dim(b, tmp, 4, 5, f"{prefix}.tmp_z.{lid}"),
                _slice_last_dim(b, ref_sig_inv, 2, 3, f"{prefix}.ref_isig_z.{lid}"),
            ],
            f"{prefix}.new_z.{lid}",
        )
        new_ref = b.op(
            "Concat", [new_xy, new_z], f"{prefix}.new_ref_cat.{lid}", axis=-1
        )
        reference_points = _sigmoid(b, new_ref, f"{prefix}.new_ref_sig.{lid}")
        hs.append(output)
        reg_out.append(tmp)
        new_ref_list.append(reference_points)
    return hs, reg_out, new_ref_list, used_ref


def _slice_last_dim(b: _Builder, x: str, start: int, end: int, prefix: str) -> str:
    starts = b.const(np.array([start], dtype=np.int64), prefix="slice_start")
    ends = b.const(np.array([end], dtype=np.int64), prefix="slice_end")
    axes = b.const(np.array([-1], dtype=np.int64), prefix="slice_axis")
    return b.op("Slice", [x, starts, ends, axes], prefix)


def _reg_branch(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    embed_dims: int,
    code_size: int,
    num_reg_fcs: int = 2,
) -> str:
    h = x
    for i in range(num_reg_fcs):
        h = _linear(
            b,
            h,
            declare(f"{prefix}.{2 * i}.weight"),
            declare(f"{prefix}.{2 * i}.bias"),
            f"{prefix}.fc{i}",
        )
        h = _relu(b, h, f"{prefix}.relu{i}")
    return _linear(
        b,
        h,
        declare(f"{prefix}.{2 * num_reg_fcs}.weight"),
        declare(f"{prefix}.{2 * num_reg_fcs}.bias"),
        f"{prefix}.fc_out",
    )


def _cls_branch(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    embed_dims: int,
    num_classes: int,
    num_reg_fcs: int = 2,
) -> str:
    h = x
    for i in range(num_reg_fcs):
        h = _linear(
            b,
            h,
            declare(f"{prefix}.{3 * i}.weight"),
            declare(f"{prefix}.{3 * i}.bias"),
            f"{prefix}.fc{i}",
        )
        h = _layernorm_last_dim(
            b,
            h,
            declare(f"{prefix}.{3 * i + 1}.weight"),
            declare(f"{prefix}.{3 * i + 1}.bias"),
            1e-5,
            f"{prefix}.ln{i}",
        )
        h = _relu(b, h, f"{prefix}.relu{i}")
    return _linear(
        b,
        h,
        declare(f"{prefix}.{3 * num_reg_fcs}.weight"),
        declare(f"{prefix}.{3 * num_reg_fcs}.bias"),
        f"{prefix}.fc_out",
    )


def _detection_head_outputs(
    b: _Builder,
    hs_list: List[str],
    reg_out_list: List[str],
    new_ref_list: List[str],
    declare,
    prefix: str,
    embed_dims: int,
    num_classes: int,
    pc_range: Sequence[float],
) -> Tuple[List[str], List[str]]:
    """Per-layer ``cls_branches``/final (pc_range-denormalized) box output --
    see :func:`_detection_transformer_decoder`'s own docstring for why this
    reuses ``reg_out``/``new_ref`` rather than recomputing ``reg_branches``."""
    scale = [
        pc_range[3] - pc_range[0],
        pc_range[4] - pc_range[1],
        pc_range[5] - pc_range[2],
    ]
    offset = [pc_range[0], pc_range[1], pc_range[2]]
    all_cls, all_bbox = [], []
    for lvl, (hs, tmp, xyz) in enumerate(zip(hs_list, reg_out_list, new_ref_list)):
        cls = _cls_branch(
            b, hs, declare, f"{prefix}.cls_branches.{lvl}", embed_dims, num_classes
        )
        scaled = []
        for i in range(3):
            v = _slice_last_dim(b, xyz, i, i + 1, f"{prefix}.xyz{i}.{lvl}")
            scale_c = b.const(np.array(scale[i], dtype=np.float32), prefix="det_scale")
            offset_c = b.const(
                np.array(offset[i], dtype=np.float32), prefix="det_offset"
            )
            scaled.append(
                b.op(
                    "Add",
                    [b.op("Mul", [v, scale_c], f"{prefix}.xyz{i}_s.{lvl}"), offset_c],
                    f"{prefix}.xyz{i}_o.{lvl}",
                )
            )
        mid = _slice_last_dim(b, tmp, 2, 4, f"{prefix}.mid.{lvl}")
        tail = _slice_last_dim(b, tmp, 5, 10, f"{prefix}.tail.{lvl}")
        box = b.op(
            "Concat",
            [scaled[0], scaled[1], mid, scaled[2], tail],
            f"{prefix}.box.{lvl}",
            axis=-1,
        )
        all_cls.append(cls)
        all_bbox.append(box)
    return all_cls, all_bbox


def _atan2(b: _Builder, y: str, x: str, prefix: str) -> str:
    """ONNX has no native ``Atan2``. Standard half-angle workaround:
    ``atan2(y, x) = 2*atan(y / (sqrt(x^2+y^2) + x))``, exact everywhere
    except the ``x <= 0, y == 0`` ray, guarded here with a small epsilon on
    the denominator (this feeds a box heading angle regressed by a trained
    network, not a hand-crafted ``x=0`` input, so that ray is measure-zero
    in practice) -- the same workaround ONNX export tools for atan2 commonly
    use."""
    eps = b.const(np.array(1e-7, dtype=np.float32), prefix="atan2_eps")
    two = b.const(np.array(2.0, dtype=np.float32), prefix="atan2_two")
    r = b.op(
        "Sqrt",
        [
            b.op(
                "Add",
                [
                    b.op("Mul", [x, x], f"{prefix}.x2"),
                    b.op("Mul", [y, y], f"{prefix}.y2"),
                ],
                f"{prefix}.r2",
            )
        ],
        f"{prefix}.r",
    )
    denom = b.op("Add", [b.op("Add", [r, x], f"{prefix}.rpx"), eps], f"{prefix}.denom")
    return b.op(
        "Mul",
        [
            b.op(
                "Atan", [b.op("Div", [y, denom], f"{prefix}.ratio")], f"{prefix}.atan"
            ),
            two,
        ],
        prefix,
    )


def _denormalize_bbox(b: _Builder, boxes: str, prefix: str) -> str:
    """``heads.py::denormalize_bbox``: 10-dim normalized box -> 9-dim
    ``[cx, cy, cz, w, l, h, yaw, vx, vy]``."""
    rot_sine = _slice_last_dim(b, boxes, 6, 7, f"{prefix}.rot_sin")
    rot_cosine = _slice_last_dim(b, boxes, 7, 8, f"{prefix}.rot_cos")
    rot = _atan2(b, rot_sine, rot_cosine, f"{prefix}.atan2")
    cx = _slice_last_dim(b, boxes, 0, 1, f"{prefix}.cx")
    cy = _slice_last_dim(b, boxes, 1, 2, f"{prefix}.cy")
    cz = _slice_last_dim(b, boxes, 4, 5, f"{prefix}.cz")
    w = b.op("Exp", [_slice_last_dim(b, boxes, 2, 3, f"{prefix}.w_log")], f"{prefix}.w")
    length = b.op(
        "Exp", [_slice_last_dim(b, boxes, 3, 4, f"{prefix}.l_log")], f"{prefix}.l"
    )
    h = b.op("Exp", [_slice_last_dim(b, boxes, 5, 6, f"{prefix}.h_log")], f"{prefix}.h")
    vx = _slice_last_dim(b, boxes, 8, 9, f"{prefix}.vx")
    vy = _slice_last_dim(b, boxes, 9, 10, f"{prefix}.vy")
    return b.op("Concat", [cx, cy, cz, w, length, h, rot, vx, vy], prefix, axis=-1)


def _nms_free_decode(
    b: _Builder,
    cls_scores: str,
    bbox_preds: str,
    prefix: str,
    num_classes: int,
    max_num: int,
    post_center_range: Sequence[float],
) -> Tuple[str, str, str, str]:
    """``NMSFreeCoder.decode_single`` -- top-``k`` over the flattened
    ``(num_query * num_classes)`` sigmoid score grid, no NMS. Returns
    ``(scores, labels, boxes, valid)`` all length ``max_num`` -- ``valid``
    (the post-center-range mask) is a companion boolean output rather than
    a dynamic-length filter (see this module's own docstring's scope note:
    a genuinely variable-length result is the caller's job). ``TopK`` has
    two outputs, so this builds that one node directly rather than through
    ``_Builder.op`` (which only names a single output)."""
    scores_sig = _sigmoid(b, cls_scores, f"{prefix}.sig")
    flat = b.op("Reshape", [scores_sig, b.shape_const([-1])], f"{prefix}.flat")

    values_name = b._name(f"{prefix}.topk_values")
    indices_name = b._name(f"{prefix}.topk_indices")
    b.nodes.append(
        onnx.helper.make_node(
            "TopK",
            [flat, b.const(np.array([max_num], dtype=np.int64), prefix="topk_k")],
            [values_name, indices_name],
            axis=-1,
            largest=1,
            sorted=1,
        )
    )
    num_classes_c = b.const(
        np.array(num_classes, dtype=np.int64), prefix="nfc_num_classes"
    )
    labels = b.op("Mod", [indices_name, num_classes_c], f"{prefix}.labels")
    bbox_index = b.op("Div", [indices_name, num_classes_c], f"{prefix}.bbox_index")
    gathered_boxes = b.op(
        "Gather", [bbox_preds, bbox_index], f"{prefix}.gathered_boxes", axis=0
    )
    final_boxes = _denormalize_bbox(b, gathered_boxes, f"{prefix}.denorm")

    pcr_lo = b.const(np.array(post_center_range[:3], dtype=np.float32), prefix="pcr_lo")
    pcr_hi = b.const(np.array(post_center_range[3:], dtype=np.float32), prefix="pcr_hi")
    centers = _slice_last_dim(b, final_boxes, 0, 3, f"{prefix}.centers")
    ge_lo = b.op("GreaterOrEqual", [centers, pcr_lo], f"{prefix}.ge_lo")
    le_hi = b.op("LessOrEqual", [centers, pcr_hi], f"{prefix}.le_hi")
    valid_per_axis = b.op("And", [ge_lo, le_hi], f"{prefix}.valid_axis")
    valid = _squeeze(
        b,
        _reduce(
            b,
            "ReduceMin",
            b.op(
                "Cast",
                [valid_per_axis],
                f"{prefix}.valid_axis_i",
                to=onnx.TensorProto.INT32,
            ),
            [-1],
            1,
            f"{prefix}.valid_min",
        ),
        [-1],
        f"{prefix}.valid",
    )
    return values_name, labels, final_boxes, valid


# ---------------------------------------------------------------------------
# Occupancy branch (perception_transformer.py's ``_adapt_bev_for_occ``/
# ``_adapt_volume_for_occ``/``_fuse_uvtr_occ_feat`` + occ_refiner.py's
# ``OccVoxelUNetRefiner``). The two ``F.grid_sample`` crops here have a
# purely config-derived sampling grid (``_normalized_occ_grid``), so both
# grids are precomputed constants, same as the view transform's frustum
# indices -- see this module's own docstring, point 1.


def _axis_center_coords_np(
    target_min: float,
    target_max: float,
    source_min: float,
    source_max: float,
    source_size: int,
    target_size: int,
) -> np.ndarray:
    source_size, target_size = max(int(source_size), 1), max(int(target_size), 1)
    source_step = (source_max - source_min) / max(float(source_size), 1e-6)
    source_center0 = source_min + 0.5 * source_step
    target_step = (target_max - target_min) / max(float(target_size), 1e-6)
    target_centers = (
        target_min + (np.arange(target_size, dtype=np.float64) + 0.5) * target_step
    )
    source_index = (target_centers - source_center0) / max(source_step, 1e-6)
    source_norm = (
        2.0 * source_index / float(source_size - 1) - 1.0
        if source_size > 1
        else np.zeros_like(source_index)
    )
    return np.clip(source_norm, -1.0, 1.0)


def _precompute_occ_grid(
    det_pc_range: Sequence[float],
    occ_pc_range: Sequence[float],
    occ_voxel_size: Sequence[float],
    bev_h: int,
    bev_w: int,
    occ_pillar_h: int,
) -> Tuple[np.ndarray, int, int, int]:
    """Reproduces ``PerceptionTransformer._normalized_occ_grid`` in numpy.
    Returns ``(grid[1, target_d, target_h, target_w, 3], target_h,
    target_w, target_d)`` -- ``grid`` is ready for ONNX ``GridSample`` on a
    5D ``(N, C, D, H, W)`` volume."""
    sx0, sy0, sz0, sx1, sy1, sz1 = det_pc_range
    tx0, ty0, tz0, tx1, ty1, tz1 = occ_pc_range
    target_w = max(int(round((tx1 - tx0) / max(float(occ_voxel_size[0]), 1e-6))), 1)
    target_h = max(int(round((ty1 - ty0) / max(float(occ_voxel_size[1]), 1e-6))), 1)
    target_d = occ_pillar_h
    x = _axis_center_coords_np(tx0, tx1, sx0, sx1, bev_w, target_w)
    y = _axis_center_coords_np(ty0, ty1, sy0, sy1, bev_h, target_h)
    z = _axis_center_coords_np(tz0, tz1, sz0, sz1, occ_pillar_h, target_d)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")
    grid = np.stack([grid_x, grid_y, grid_z], axis=-1)[None]
    return grid.astype(np.float32), target_h, target_w, target_d


def _residual_stage_3d(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    in_channels: int,
    out_channels: int,
    stride: Tuple[int, int, int],
    num_blocks: int,
) -> str:
    """``ResidualStage3D`` (``occ_refiner.py``): a ``BasicBlock3D`` stack."""
    for i in range(num_blocks):
        p = f"{prefix}.blocks.{i}"
        block_in = in_channels if i == 0 else out_channels
        block_stride = stride if i == 0 else (1, 1, 1)
        identity = x
        h = _conv(
            b,
            x,
            declare(f"{p}.conv1.weight"),
            None,
            f"{p}.conv1",
            strides=list(block_stride),
            pads=[1, 1, 1],
        )
        h = _batchnorm(
            b,
            h,
            declare(f"{p}.norm1.weight"),
            declare(f"{p}.norm1.bias"),
            declare(f"{p}.norm1.running_mean"),
            declare(f"{p}.norm1.running_var"),
            1e-5,
            f"{p}.norm1",
        )
        h = _relu(b, h, f"{p}.relu1")
        h = _conv(
            b,
            h,
            declare(f"{p}.conv2.weight"),
            None,
            f"{p}.conv2",
            strides=[1, 1, 1],
            pads=[1, 1, 1],
        )
        h = _batchnorm(
            b,
            h,
            declare(f"{p}.norm2.weight"),
            declare(f"{p}.norm2.bias"),
            declare(f"{p}.norm2.running_mean"),
            declare(f"{p}.norm2.running_var"),
            1e-5,
            f"{p}.norm2",
        )
        if block_stride != (1, 1, 1) or block_in != out_channels:
            identity = _conv(
                b,
                x,
                declare(f"{p}.downsample.0.weight"),
                None,
                f"{p}.down_conv",
                strides=list(block_stride),
                pads=[0, 0, 0],
            )
            identity = _batchnorm(
                b,
                identity,
                declare(f"{p}.downsample.1.weight"),
                declare(f"{p}.downsample.1.bias"),
                declare(f"{p}.downsample.1.running_mean"),
                declare(f"{p}.downsample.1.running_var"),
                1e-5,
                f"{p}.down_bn",
            )
        x = _relu(b, b.op("Add", [h, identity], f"{p}.resid"), f"{p}.relu2")
    return x


def _occ_voxel_unet_refiner(
    b: _Builder,
    x: str,
    declare,
    prefix: str,
    in_channels: int,
    out_channels: int,
    batch: int,
    in_d: int,
    in_h: int,
    in_w: int,
) -> str:
    """``OccVoxelUNetRefiner`` (``occ_refiner.py``): a residual 3D U-Net,
    downsampling only in (H, W) (``stride=(1,2,2)``, halving on each real,
    even-sized encoder stage), trilinear upsampling back to each skip
    connection's exact (static, so plain Python arithmetic, not a runtime
    ``Shape`` op) spatial shape."""
    c0 = max(64, out_channels * 2)
    c1, c2, c3 = c0 * 2, c0 * 4, c0 * 6
    h1, w1 = in_h // 2, in_w // 2
    h2, w2 = h1 // 2, w1 // 2

    skip0 = _residual_stage_3d(
        b, x, declare, f"{prefix}.input_proj", in_channels, c0, (1, 1, 1), 2
    )
    skip1 = _residual_stage_3d(
        b, skip0, declare, f"{prefix}.enc1", c0, c1, (1, 2, 2), 2
    )
    skip2 = _residual_stage_3d(
        b, skip1, declare, f"{prefix}.enc2", c1, c2, (1, 2, 2), 2
    )
    h = _residual_stage_3d(b, skip2, declare, f"{prefix}.enc3", c2, c3, (1, 2, 2), 2)
    h = _residual_stage_3d(b, h, declare, f"{prefix}.bottleneck", c3, c3, (1, 1, 1), 2)

    def upsample_to(t: str, channels: int, d: int, hh: int, ww: int, name: str) -> str:
        return _resize_to_shape(
            b,
            t,
            [batch, channels, d, hh, ww],
            align_corners=False,
            prefix=f"{prefix}.{name}",
        )

    h = upsample_to(h, c3, in_d, h2, w2, "up2")
    h = _residual_stage_3d(
        b,
        b.op("Concat", [h, skip2], f"{prefix}.cat2", axis=1),
        declare,
        f"{prefix}.dec2",
        c3 + c2,
        c2,
        (1, 1, 1),
        2,
    )

    h = upsample_to(h, c2, in_d, h1, w1, "up1")
    h = _residual_stage_3d(
        b,
        b.op("Concat", [h, skip1], f"{prefix}.cat1", axis=1),
        declare,
        f"{prefix}.dec1",
        c2 + c1,
        c1,
        (1, 1, 1),
        2,
    )

    h = upsample_to(h, c1, in_d, in_h, in_w, "up0")
    h = _residual_stage_3d(
        b,
        b.op("Concat", [h, skip0], f"{prefix}.cat0", axis=1),
        declare,
        f"{prefix}.dec0",
        c1 + c0,
        c0,
        (1, 1, 1),
        2,
    )

    h = _residual_stage_3d(b, h, declare, f"{prefix}.out_block", c0, c0, (1, 1, 1), 2)
    h = _conv(
        b,
        h,
        declare(f"{prefix}.out_proj.0.weight"),
        None,
        f"{prefix}.out_conv",
        strides=[1, 1, 1],
        pads=[1, 1, 1],
    )
    h = _batchnorm(
        b,
        h,
        declare(f"{prefix}.out_proj.1.weight"),
        declare(f"{prefix}.out_proj.1.bias"),
        declare(f"{prefix}.out_proj.1.running_mean"),
        declare(f"{prefix}.out_proj.1.running_var"),
        1e-5,
        f"{prefix}.out_bn",
    )
    return _relu(b, h, f"{prefix}.out_relu")


def _occupancy_branch(
    b: _Builder,
    bev_feat_ego: str,
    uvtr_occ_feat: str,
    declare,
    prefix: str,
    embed_dims: int,
    bev_h: int,
    bev_w: int,
    occ_pillar_h: int,
    occ_dim: int,
    occ_num_classes: int,
    det_pc_range: Sequence[float],
    occ_pc_range: Sequence[float],
    occ_voxel_size: Sequence[float],
) -> str:
    """``bev_feat_ego``: ``[embed_dims, bev_h, bev_w]``. ``uvtr_occ_feat``:
    ``[embed_dims, occ_pillar_h, bev_h, bev_w]`` (the view transform's own
    output -- already exactly this shape). Returns occupancy logits
    ``[target_w, target_h, target_d, occ_num_classes]`` (matches
    ``PerceptionTransformer.forward``'s own final ``permute(0,4,3,2,1)``,
    batch dim dropped)."""
    middle_dims = embed_dims // occ_pillar_h
    grid_np, target_h, target_w, target_d = _precompute_occ_grid(
        det_pc_range, occ_pc_range, occ_voxel_size, bev_h, bev_w, occ_pillar_h
    )
    grid_c = b.const(grid_np, prefix="occ_grid")

    bev_vol = b.op(
        "Reshape",
        [bev_feat_ego, b.shape_const([1, middle_dims, occ_pillar_h, bev_h, bev_w])],
        f"{prefix}.bev_vol",
    )
    bev_feat = _grid_sample(
        b,
        bev_vol,
        grid_c,
        f"{prefix}.bev_sample",
        align_corners=True,
        padding_mode="zeros",
    )

    uvtr_vol = _unsqueeze(b, uvtr_occ_feat, [0], f"{prefix}.uvtr_vol")
    uvtr_feat = _grid_sample(
        b,
        uvtr_vol,
        grid_c,
        f"{prefix}.uvtr_sample",
        align_corners=True,
        padding_mode="zeros",
    )
    uvtr_proj = _conv(
        b,
        uvtr_feat,
        declare(f"{prefix}.uvtr_occ_proj.weight"),
        declare(f"{prefix}.uvtr_occ_proj.bias"),
        f"{prefix}.uvtr_proj",
        strides=[1, 1, 1],
        pads=[0, 0, 0],
    )
    fused = _conv_module_3d(
        b,
        b.op("Concat", [bev_feat, uvtr_proj], f"{prefix}.fuse_cat", axis=1),
        declare,
        f"{prefix}.uvtr_occ_fuse",
        norm="bn",
        num_groups=0,
        num_channels=middle_dims,
        act=True,
        bias=False,
    )
    bev_feat = b.op("Add", [bev_feat, fused], f"{prefix}.fused")

    occ_feat = _occ_voxel_unet_refiner(
        b,
        bev_feat,
        declare,
        f"{prefix}.occ_decoder",
        middle_dims,
        occ_dim,
        1,
        target_d,
        target_h,
        target_w,
    )
    occ_feat = _squeeze(
        b, occ_feat, [0], f"{prefix}.occ_feat_unbatched"
    )  # (occ_dim, D, H, W)
    occ_feat = b.op(
        "Transpose", [occ_feat], f"{prefix}.occ_feat_t", perm=[3, 2, 1, 0]
    )  # (W, H, D, occ_dim)

    h1 = _linear(
        b,
        occ_feat,
        declare(f"{prefix}.occ_pred_head.0.weight"),
        declare(f"{prefix}.occ_pred_head.0.bias"),
        f"{prefix}.pred_fc1",
    )
    h1 = _softplus(b, h1, f"{prefix}.pred_softplus")
    return _linear(
        b,
        h1,
        declare(f"{prefix}.occ_pred_head.2.weight"),
        declare(f"{prefix}.occ_pred_head.2.bias"),
        f"{prefix}.pred_fc2",
    )


# ---------------------------------------------------------------------------
# Map segmentation branch (heads.py's BevFeatureSlicer + map_seg.py's
# MapSegEncode). BevFeatureSlicer's sampling grid is, like the occupancy
# grids above, a pure function of fixed config -- precomputed as a constant.
# Its ``bev_start_position`` registered buffer is loaded by the real model
# but never actually read by ``forward()``/``_grid()`` (confirmed from
# source: ``_map_params`` are computed directly from ``grid_conf``, not from
# that buffer) -- vestigial checkpoint state this builder simply never
# declares, since it cannot affect any output.


def _precompute_bev_feature_slicer_grid(
    det_grid_conf: dict, map_grid_conf: dict
) -> tuple[np.ndarray, int, int]:
    map_res_x, map_res_y = map_grid_conf["xbound"][2], map_grid_conf["ybound"][2]
    map_start_x = map_grid_conf["xbound"][0] + map_res_x / 2.0
    map_start_y = map_grid_conf["ybound"][0] + map_res_y / 2.0
    nx = -(det_grid_conf["xbound"][0] + det_grid_conf["xbound"][2] / 2.0)
    ny = -(det_grid_conf["ybound"][0] + det_grid_conf["ybound"][2] / 2.0)
    norm_x = (
        np.arange(map_start_x, map_grid_conf["xbound"][1], map_res_x, dtype=np.float64)
        / nx
    )
    norm_y = (
        np.arange(map_start_y, map_grid_conf["ybound"][1], map_res_y, dtype=np.float64)
        / ny
    )
    grid_y, grid_x = np.meshgrid(norm_y, norm_x, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=2)[None]  # (1, map_h, map_w, 2)
    return grid.astype(np.float32), len(norm_y), len(norm_x)


def _map_seg_encode(
    b: _Builder, x: str, declare, prefix: str, in_c: int, out_c: int
) -> str:
    """``MapSegEncode`` (``map_seg.py``): a resnet18-style trunk (GroupNorm,
    ``bn1``/``bn2`` attribute names kept from the pre-swap BatchNorm layers)
    + a 2-scale upsampling decoder."""

    def make_layer(
        x: str, in_planes: int, planes: int, blocks: int, stride: int, name: str
    ) -> str:
        has_down = stride != 1 or in_planes != planes
        h = _basic_block_2d_gn_resnet18(
            b, x, declare, f"{prefix}.{name}.0", planes, stride, has_down
        )
        for i in range(1, blocks):
            h = _basic_block_2d_gn_resnet18(
                b, h, declare, f"{prefix}.{name}.{i}", planes, 1, False
            )
        return h

    h = _conv(
        b,
        x,
        declare(f"{prefix}.conv1.weight"),
        None,
        f"{prefix}.conv1",
        strides=[2, 2],
        pads=[3, 3],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.bn1.weight"),
        declare(f"{prefix}.bn1.bias"),
        min(32, 64),
        64,
        1e-5,
        f"{prefix}.bn1",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.relu")
    x1 = make_layer(h, 64, 64, 2, 1, "layer1")
    h = make_layer(x1, 64, 128, 2, 2, "layer2")
    x2 = make_layer(h, 128, 256, 2, 2, "layer3")

    # _Up(64+256, 256, scale_factor=4): upsample x2 4x (bilinear), concat
    # with x1, then a 2x(conv+gn+relu).
    up = _upsample_bilinear(b, x2, 4.0, align_corners=True, prefix=f"{prefix}.up1.up")
    cat = b.op("Concat", [x1, up], f"{prefix}.up1.cat", axis=1)
    h = _conv(
        b,
        cat,
        declare(f"{prefix}.up1.conv.0.weight"),
        None,
        f"{prefix}.up1.conv0",
        strides=[1, 1],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.up1.conv.1.weight"),
        declare(f"{prefix}.up1.conv.1.bias"),
        min(32, 256),
        256,
        1e-5,
        f"{prefix}.up1.gn0",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.up1.relu0")
    h = _conv(
        b,
        h,
        declare(f"{prefix}.up1.conv.3.weight"),
        None,
        f"{prefix}.up1.conv3",
        strides=[1, 1],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.up1.conv.4.weight"),
        declare(f"{prefix}.up1.conv.4.bias"),
        min(32, 256),
        256,
        1e-5,
        f"{prefix}.up1.gn1",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.up1.relu1")

    h = _upsample_bilinear(b, h, 2.0, align_corners=True, prefix=f"{prefix}.up2.up")
    h = _conv(
        b,
        h,
        declare(f"{prefix}.up2.1.weight"),
        None,
        f"{prefix}.up2.conv0",
        strides=[1, 1],
        pads=[1, 1],
    )
    h = _groupnorm(
        b,
        h,
        declare(f"{prefix}.up2.2.weight"),
        declare(f"{prefix}.up2.2.bias"),
        min(32, 128),
        128,
        1e-5,
        f"{prefix}.up2.gn0",
        spatial_dims=2,
    )
    h = _relu(b, h, f"{prefix}.up2.relu0")
    return _conv(
        b,
        h,
        declare(f"{prefix}.up2.4.weight"),
        declare(f"{prefix}.up2.4.bias"),
        f"{prefix}.up2.conv_out",
        strides=[1, 1],
        pads=[0, 0],
    )


def _map_segmentation_branch(
    b: _Builder,
    bev_feat_ego: str,
    declare,
    prefix: str,
    embed_dims: int,
    det_grid_conf: dict,
    map_grid_conf: dict,
    map_num_classes: int,
) -> str:
    """``bev_feat_ego``: ``[embed_dims, bev_h, bev_w]``. Returns map logits
    ``[map_num_classes, map_h, map_w]``."""
    grid_np, map_h, map_w = _precompute_bev_feature_slicer_grid(
        det_grid_conf, map_grid_conf
    )
    grid_c = b.const(grid_np, prefix="map_slicer_grid")
    cropped = _grid_sample(
        b,
        _unsqueeze(b, bev_feat_ego, [0], f"{prefix}.batched"),
        grid_c,
        f"{prefix}.crop",
        align_corners=True,
        padding_mode="zeros",
    )
    logits = _map_seg_encode(
        b, cropped, declare, f"{prefix}.seg_decoder", embed_dims, map_num_classes
    )
    return _squeeze(b, logits, [0], f"{prefix}.unbatched")


# ---------------------------------------------------------------------------
# Top-level composition (modeling_perception.py::BEVFormerModelV2 +
# heads.py::BEVFormerHead).


def _uvtr_voxel_to_bev_tokens(
    b: _Builder,
    voxel_space: str,
    declare,
    prefix: str,
    embed_dims: int,
    occ_pillar_h: int,
    bev_h: int,
    bev_w: int,
) -> str:
    """``BEVFormerModelV2._uvtr_voxel_to_bev_tokens`` -- the ``B > 1: mean``
    branch never fires in this module's single-sample scope."""
    merged = b.op(
        "Reshape",
        [voxel_space, b.shape_const([1, embed_dims * occ_pillar_h, bev_h, bev_w])],
        f"{prefix}.merged",
    )
    proj = _conv(
        b,
        merged,
        declare(f"{prefix}.weight"),
        declare(f"{prefix}.bias"),
        f"{prefix}.proj",
        strides=[1, 1],
        pads=[0, 0],
    )
    proj = _squeeze(b, proj, [0], f"{prefix}.unbatched")  # (embed_dims, bev_h, bev_w)
    proj = b.op(
        "Transpose", [proj], f"{prefix}.hwc", perm=[1, 2, 0]
    )  # (bev_h, bev_w, embed_dims)
    return b.op("Reshape", [proj, b.shape_const([bev_h * bev_w, embed_dims])], prefix)


def reconstruct_qwen_drive_perception(
    hf_dir: str,
    num_cams: int,
    lidar2img: np.ndarray,
    lidar2ego: np.ndarray,
    img_shape: Tuple[int, int],
    llm_feat_hw: Tuple[int, int],
    vit_feat_hw: Tuple[int, int],
) -> onnx.ModelProto:
    """Builds the full BEV perception graph -- spine (FPNs, depth net, view
    transform, BEVFormer encoder) + all three heads (detection, occupancy,
    map segmentation) -- from a ``qwen_drive_perception`` HuggingFace
    checkpoint directory.

    :param hf_dir: checkpoint directory (``config.json`` + safetensors)
    :param num_cams: static camera count for this build
    :param lidar2img: ``[num_cams, 4, 4]`` per-camera lidar-to-image
            calibration (real values, e.g. from ``geometry.build_lidar2img``)
            -- a deployment-fixed constant, not a per-call input; see this
            module's own docstring, point 1, for why. Inverted here in numpy
            (``np.linalg.inv``) at graph-*build* time -- never inside the
            graph.
    :param lidar2ego: ``[4, 4]`` lidar-to-ego calibration, likewise
            build-time-static.
    :param img_shape: ``(height, width)`` of the camera images the
            calibration above was computed for.
    :param llm_feat_hw: ``(H, W)`` of ``img_llm_feats`` (the post-merge LLM
            token grid per camera).
    :param vit_feat_hw: ``(H, W)`` of ``img_vit_feats`` (the pre-merge ViT
            patch grid per camera) -- must be exactly ``2x`` ``llm_feat_hw``
            (the merger's own spatial ratio) *and* match the frustum's own
            pixel-bin grid (``frustum_range``/``frustum_size`` in the
            checkpoint's config) exactly -- see :func:`_view_transform`.
    :returns: the constructed, hydrated model. Inputs ``img_llm_feats``
            (``float32[num_cams, *llm_feat_hw, llm_dim]``), ``img_vit_feats``
            (``float32[num_cams, *vit_feat_hw, vit_dim]``). Outputs
            ``det_scores``/``det_labels``/``det_boxes``/``det_valid``
            (``max_num`` candidates, ego frame, un-filtered -- see this
            module's docstring's scope note), ``occ_logits``
            (``[X, Y, Z, occ_num_classes]``), ``map_logits``
            (``[map_num_classes, map_h, map_w]``).
    """
    config = read_hf_config(hf_dir)
    if config.get("model_type") != _SUPPORTED_MODEL_TYPE:
        raise UnsupportedArchitectureError(
            f"model_type={config.get('model_type')!r}, expected {_SUPPORTED_MODEL_TYPE!r}"
        )
    entries = _index_safetensors_checkpoint(hf_dir)
    b = _Builder()

    def declare(name: str) -> str:
        entry = entries.get(name)
        if entry is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor {name!r}"
            )
        b.initializers.append(_read_tensor(entry, name))
        if entry.dtype in ("F32", "BF16"):
            return name
        return b.op("Cast", [name], f"{name}.f32", to=onnx.TensorProto.FLOAT)

    llm_dim, vit_dim, embed_dim = (
        config["llm_dim"],
        config["vit_dim"],
        config["embed_dim"],
    )
    bev_h, bev_w = config["bev_h"], config["bev_w"]
    occ_pillar_h, occ_dim = config["occ_pillar_h"], config["occ_dim"]
    num_query, code_size = config["num_query"], config["code_size"]
    det_num_classes, occ_num_classes, map_num_classes = (
        config["det_num_classes"],
        config["occ_num_classes"],
        config["map_num_classes"],
    )
    num_encoder_layers, num_decoder_layers = (
        config["num_encoder_layers"],
        config["num_decoder_layers"],
    )
    det_pc_range, det_voxel_size = config["det_pc_range"], config["det_voxel_size"]
    nuscenes_occ_pc_range, nuscenes_occ_voxel_size = (
        config["nuscenes_occ_pc_range"],
        config["nuscenes_occ_voxel_size"],
    )
    frustum_range, frustum_size = config["frustum_range"], config["frustum_size"]
    map_xbound, map_ybound = config["map_xbound"], config["map_ybound"]
    post_center_range = tuple(
        config.get("post_center_range", DEFAULT_POST_CENTER_RANGE)
    )
    max_num_boxes = int(config.get("max_num_boxes", DEFAULT_MAX_NUM_BOXES))

    llm_h, llm_w = llm_feat_hw
    vit_h, vit_w = vit_feat_hw

    ego2img = np.stack(
        [lidar2img[c] @ np.linalg.inv(lidar2ego) for c in range(num_cams)], axis=0
    )
    img2ego = np.stack(
        [lidar2ego @ np.linalg.inv(lidar2img[c]) for c in range(num_cams)], axis=0
    )

    img_llm_feats = "img_llm_feats"
    img_vit_feats = "img_vit_feats"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            img_llm_feats, onnx.TensorProto.FLOAT, [num_cams, llm_h, llm_w, llm_dim]
        ),
        onnx.helper.make_tensor_value_info(
            img_vit_feats, onnx.TensorProto.FLOAT, [num_cams, vit_h, vit_w, vit_dim]
        ),
    ]

    # --- adaptor (main/LLM stream, 4 levels) ---------------------------
    feat_main = b.op(
        "Transpose", [img_llm_feats], "feat_main", perm=[0, 3, 1, 2]
    )  # (num_cams, llm_dim, H, W)
    mlvl_feats = _simple_fpn(
        b,
        feat_main,
        declare,
        "bev_modeling.adaptor",
        llm_dim,
        embed_dim,
        FPN_ADAPTOR_SCALES,
    )
    spatial_shapes: List[Tuple[int, int]] = []
    for scale in FPN_ADAPTOR_SCALES:
        spatial_shapes.append((int(llm_h * scale), int(llm_w * scale)))

    # --- vit_neck (UVTR stream, 1 level) + depth net -------------------
    feat_vit = b.op(
        "Transpose", [img_vit_feats], "feat_vit", perm=[0, 3, 1, 2]
    )  # (num_cams, vit_dim, Hv, Wv)
    (vit_feat,) = _simple_fpn(
        b, feat_vit, declare, "bev_modeling.vit_neck", vit_dim, embed_dim, (1.0,)
    )

    depth_dim = int((frustum_range[5] - frustum_range[2]) / frustum_size[2])
    depth_logits = _depth_net(
        b,
        vit_feat,
        declare,
        "bev_modeling.depth_net",
        embed_dim,
        depth_dim,
        ASPP_MID_CHANNELS,
        num_cams,
        vit_h,
        vit_w,
    )
    img_depth = _softmax(b, depth_logits, 1, "img_depth")

    # --- view transform: image features -> ego voxel volume -----------
    uvtr_voxel_space = _view_transform(
        b,
        vit_feat,
        img_depth,
        declare,
        "bev_modeling.view_trans",
        embed_dim,
        num_cams,
        vit_h,
        vit_w,
        depth_dim,
        frustum_range,
        frustum_size,
        det_pc_range,
        [
            det_voxel_size[0],
            det_voxel_size[1],
            (det_pc_range[5] - det_pc_range[2]) / occ_pillar_h,
        ],
        [bev_w, bev_h, occ_pillar_h],
        img2ego,
    )
    uvtr_bev_tokens = _uvtr_voxel_to_bev_tokens(
        b,
        uvtr_voxel_space,
        declare,
        "bev_modeling.uvtr_query_proj",
        embed_dim,
        occ_pillar_h,
        bev_h,
        bev_w,
    )

    # --- BEV queries + positional encoding -----------------------------
    bev_embedding_weight = declare("bev_modeling.head.bev_embedding.weight")
    bev_query = b.op("Add", [bev_embedding_weight, uvtr_bev_tokens], "bev_query")
    bev_pos = _bev_positional_encoding(
        b,
        declare,
        "bev_modeling.head.positional_encoding",
        bev_h,
        bev_w,
        embed_dim // 2,
    )

    # --- flatten multi-level, multi-camera adaptor features for SCA ----
    level_embeds = declare("bev_modeling.head.transformer.level_embeds")
    feat_flat_levels = []
    for lvl, (feat, (h_l, w_l)) in enumerate(zip(mlvl_feats, spatial_shapes)):
        f = b.op("Transpose", [feat], f"feat_flat.{lvl}.t", perm=[0, 2, 3, 1])
        f = b.op(
            "Reshape",
            [f, b.shape_const([num_cams, h_l * w_l, embed_dim])],
            f"feat_flat.{lvl}.r",
        )
        level_embed_lvl = _unsqueeze(
            b,
            _slice_axis(b, level_embeds, 0, lvl, lvl + 1, f"level_embed.{lvl}"),
            [0],
            f"level_embed.{lvl}.u",
        )
        f = b.op("Add", [f, level_embed_lvl], f"feat_flat.{lvl}.pe")
        feat_flat_levels.append(f)
    feat_flatten = b.op("Concat", feat_flat_levels, "feat_flatten", axis=1)

    reference_points_cam, mask_any, count = _precompute_point_sampling(
        bev_h, bev_w, det_pc_range, POINTS_IN_PILLAR, ego2img, img_shape
    )

    bev_embed = _bev_former_encoder(
        b,
        bev_query,
        bev_pos,
        feat_flatten,
        feat_flatten,
        declare,
        entries,
        "bev_modeling.head.transformer.encoder",
        num_encoder_layers,
        bev_h * bev_w,
        embed_dim,
        bev_h,
        bev_w,
        spatial_shapes,
        reference_points_cam,
        mask_any,
        count,
    )
    bev_feat_ego = b.op(
        "Transpose",
        [
            b.op(
                "Reshape",
                [bev_embed, b.shape_const([bev_h, bev_w, embed_dim])],
                "bev_feat_r",
            )
        ],
        "bev_feat_ego",
        perm=[2, 0, 1],
    )

    # --- detection decoder ----------------------------------------------
    query_embedding_weight = declare(
        "bev_modeling.head.query_embedding.weight"
    )  # (num_query, 2*embed_dim)
    query_pos0 = _slice_last_dim(b, query_embedding_weight, 0, embed_dim, "query_pos0")
    query0 = _slice_last_dim(
        b, query_embedding_weight, embed_dim, 2 * embed_dim, "query0"
    )
    init_reference = _sigmoid(
        b,
        _linear(
            b,
            query_pos0,
            declare("bev_modeling.head.transformer.reference_points.weight"),
            declare("bev_modeling.head.transformer.reference_points.bias"),
            "init_ref_lin",
        ),
        "init_reference",
    )

    hs_list, reg_out_list, new_ref_list, _used_ref = _detection_transformer_decoder(
        b,
        query0,
        query_pos0,
        bev_embed,
        init_reference,
        declare,
        entries,
        "bev_modeling.head.transformer.decoder",
        num_decoder_layers,
        num_query,
        bev_h * bev_w,
        embed_dim,
        code_size,
        [(bev_h, bev_w)],
    )
    all_cls, all_bbox = _detection_head_outputs(
        b,
        hs_list,
        reg_out_list,
        new_ref_list,
        declare,
        "bev_modeling.head",
        embed_dim,
        det_num_classes,
        det_pc_range,
    )
    det_scores, det_labels, det_boxes, det_valid = _nms_free_decode(
        b,
        all_cls[-1],
        all_bbox[-1],
        "bev_modeling.head.bbox_coder",
        det_num_classes,
        max_num_boxes,
        post_center_range,
    )

    # --- occupancy branch -------------------------------------------------
    occ_logits = _occupancy_branch(
        b,
        bev_feat_ego,
        uvtr_voxel_space,
        declare,
        "bev_modeling.head.transformer",
        embed_dim,
        bev_h,
        bev_w,
        occ_pillar_h,
        occ_dim,
        occ_num_classes,
        det_pc_range,
        nuscenes_occ_pc_range,
        nuscenes_occ_voxel_size,
    )
    _, occ_target_h, occ_target_w, occ_target_d = _precompute_occ_grid(
        det_pc_range,
        nuscenes_occ_pc_range,
        nuscenes_occ_voxel_size,
        bev_h,
        bev_w,
        occ_pillar_h,
    )

    # --- map segmentation branch -------------------------------------------
    det_grid_conf = {
        "xbound": [det_pc_range[0], det_pc_range[3], det_voxel_size[0]],
        "ybound": [det_pc_range[1], det_pc_range[4], det_voxel_size[1]],
        "zbound": [
            det_pc_range[2],
            det_pc_range[5],
            (det_pc_range[5] - det_pc_range[2]) / occ_pillar_h,
        ],
    }
    map_grid_conf = {
        "xbound": list(map_xbound),
        "ybound": list(map_ybound),
        "zbound": [-10.0, 10.0, 20.0],
    }
    map_logits = _map_segmentation_branch(
        b,
        bev_feat_ego,
        declare,
        "bev_modeling.head",
        embed_dim,
        det_grid_conf,
        map_grid_conf,
        map_num_classes,
    )

    _, map_h, map_w = _precompute_bev_feature_slicer_grid(det_grid_conf, map_grid_conf)

    graph_outputs = [
        onnx.helper.make_tensor_value_info(
            det_scores, onnx.TensorProto.FLOAT, [max_num_boxes]
        ),
        onnx.helper.make_tensor_value_info(
            det_labels, onnx.TensorProto.INT64, [max_num_boxes]
        ),
        onnx.helper.make_tensor_value_info(
            det_boxes, onnx.TensorProto.FLOAT, [max_num_boxes, 9]
        ),
        onnx.helper.make_tensor_value_info(
            det_valid, onnx.TensorProto.INT32, [max_num_boxes]
        ),
        onnx.helper.make_tensor_value_info(
            occ_logits,
            onnx.TensorProto.FLOAT,
            [occ_target_w, occ_target_h, occ_target_d, occ_num_classes],
        ),
        onnx.helper.make_tensor_value_info(
            map_logits, onnx.TensorProto.FLOAT, [map_num_classes, map_h, map_w]
        ),
    ]

    graph = onnx.helper.make_graph(
        b.nodes,
        "qwen_drive_perception",
        graph_inputs,
        graph_outputs,
        initializer=b.initializers,
    )
    return onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
