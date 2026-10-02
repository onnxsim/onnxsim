"""AMD Quark's ``FastFinetune`` (AdaRound / AdaQuant) for QDQ models, in numpy.

This is a port of ``quark.onnx.algorithm.finetuning`` (read from the
``amd-quark`` 0.13 wheel; no code copied): the same *blocks*, the same
*training loop*, the same *loss* and the same option names / defaults, so that
``QConfig`` algo configs (``AdaRoundConfig`` / ``AdaQuantConfig``) mean the
same thing here as there. Quark runs it on torch; here every gradient is
hand-derived numpy (float64; float32 in torch's order for AdaQuant), so results agree with Quark's *statistically*
(its mini-batches come from ``torch.randperm``), and **exactly** when given the
same mini-batch indices (``tests/test_quark_finetune_parity.py`` feeds torch's
own ``randperm`` stream through ``perm_fn``).

What a "block" is (Quark's ``Subgraph``): for every ``Conv`` / ``ConvTranspose``
/ ``Gemm`` / ``MatMul`` (constant weight) / ``InstanceNormalization`` /
``LayerNormalization`` in the quantized model, the sub-model from the float
tensor in front of the layer's *input* ``QuantizeLinear`` to the layer's
output: input Q/DQ, weight Q/DQ, the op, the (quantized) bias, an optional
following activation (``Relu``, ``PRelu``, ``LeakyRelu``, ``Clip``,
``Sigmoid``, ``Tanh``, ``Gelu``, ``Softmax``) and -- with ``output_qdq`` -- the output Q/DQ.
A ``Relu`` that was folded into the output quantizer (its range starts at 0, as
in Quark's ``INT8_CNN_DEFAULT`` on a ``MatMul`` / ``Gemm`` / ``Conv`` model) is
*not* a node any more, so the block ends at the op's own output and, as in
Quark, its target is the float model's *pre*-Relu tensor.
The training target is the *float* model's tensor at the same block output,
the input is the quantized model's pre-quantization tensor (``drop_ratio`` mixes
it element-wise with the float model's) and the loss is Quark's
``(||quant - float||_F over dim 1)^2`` averaged over the rest.

Per layer, in graph order, **sequentially**: the quantized model's input
activation is re-captured after every layer's update (``parallel=True``
captures them all once up front, like Quark's ``Parallel``).

* AdaRound learns one rectified-sigmoid ``alpha`` per weight against that loss
  plus the annealed rounding regularizer (cosine ``beta``, ``reg_param``,
  ``warm_start``); the new codes are ``floor(w / s) + (alpha >= 0)``.
* AdaQuant instead trains the float weight (and, with ``update_bias``, the
  quantized bias) directly through a straight-through quantizer with Adam
  (``learning_rate`` defaults to ``1e-5``, as in Quark) and rewrites the codes
  as the quantization of the trained value.

Mini-batches are ``batch_size`` samples drawn without replacement from all
calibration samples (the rows of every calibration batch's leading axis) each
iteration, ``num_iterations`` times; ``early_stop`` is Quark's rule verbatim,
including its quirks (the window is ``num_batches`` iterations when
``num_batches > 1`` else ``num_iterations / 10``; the window's last
iteration's loss is *not* accumulated; AdaRound compares the mean *rounding*
loss, AdaQuant the reconstruction loss; the break happens before that
iteration's optimizer step).

Layers Quark's own torch modules cannot handle are skipped exactly where
Quark skips them (its driver logs the failure and goes on; probed against
``amd-quark`` 0.13, ``tests/test_quark_finetune_coverage_parity.py``):

* ``auto_pad`` other than ``NOTSET``, any ``ConvTranspose`` that has an
  ``output_padding`` or ``output_shape`` attribute (even an all-zero one),
  ``LayerNormalization`` over an axis other than the last: its converter
  raises. (``SelectMaxMemLayer`` converts every layer up front without a
  guard, so there Quark aborts the whole run -- and so does this port.)
* ``ConvTranspose`` with asymmetric ``pads``: Quark puts a ``ConstantPad`` in
  front of a transposed convolution that has no padding, so its output no
  longer has the float layer's shape and the first forward fails.
* a layer whose first forward does not fit its target (``Gemm`` with ``transA``
  only fits when there is one calibration batch whose two axes and the
  mini-batch are all of the same size -- there it *trains*, the "samples" being
  the rows of ``A``, and so does this port). Torch *broadcasts* ``quant - float``,
  so an output that differs from its target only along size-1 axes trains too
  (``transA`` on a single one-row sample: the ``[M, N]`` output minus the
  ``[1, N]`` target row; the loss sums over the broadcast rows) and so does this
  port; a target with fewer rows than there are samples (``transA`` with
  ``M == 1``) fails Quark's ``torch.cat`` and the layer is skipped.

What Quark *does* train, and this port reproduces bit for bit given its random
stream: 1-D / 2-D / 3-D ``Conv`` and ``ConvTranspose`` (``ConvTranspose`` also
with ``group > 1``), ``MatMul`` / ``Gemm`` on activations of any rank, ``PRelu`` blocks (Quark builds ``torch.nn.PReLU()`` whatever
the node's slope is: the layer is trained with the slope 0.25), ``Clip`` whose
bounds are not both initializers (``torch.nn.ReLU6``) or that has fewer than two
bounds as attributes (the identity), bias-less ``Gemm`` (``torch.nn.Linear``
owns a randomly initialized bias that is added to every forward), and 16-bit
or ``uint8`` / asymmetric weight grids. Two Quark quirks come along: its
asymmetric 3-D ``pads`` are scrambled across the three axes (its "swap H and W"
step is only right for 2-D), and a >= 3-D ``Gemm`` output gets its bias along
axis 1 whenever that axis has the bias' length. Activation fake-quantization is
done in float32 like torch's, which keeps its rounding decisions for 16-bit
codes (float64 would not).

``AdaQuant`` runs its whole loop in float32 in torch's operation order
(``FinetuneOptions.float32``, on by default for AdaQuant): the straight-through
quantizer (``round`` / ``clamp`` / ``(q - zp) * scale``, divided by the scale
again on the way back), the ``(norm(err, dim=1) ** 2).mean()`` autograd chain
(``(err / norm) * (2 * norm / N)``) and ``torch.optim.Adam`` (``lerp`` /
``addcmul`` / ``addcdiv``). Probed against Quark, that is what the chaos of 16-bit
AdaQuant (a float32 ULP in the forward pass flips ~0.4 % of the codes per
layer) needs: ``MatMul`` / ``Gemm`` layers come out *bit-identical* to Quark's
(10 to 300 iterations on int16 and int8 weights), where float64 arithmetic is off by 2-15 %
of the codes after 10 iterations. What cannot be exact: ``Conv`` / ``ConvTranspose``
(torch's oneDNN accumulates in an order numpy's im2col product does not
reproduce), the norm layers and ``Gelu`` / ``Tanh`` / ``Sigmoid`` (their float32
kernels differ in the last bit), a numpy whose BLAS rounds differently from torch's
for a given product shape, and torch's vectorized ``sqrt`` (Sleef's 0.5001-ULP one
differs from IEEE's in ~0.7 % of the values, 1 ULP, inside Adam's denominator);
there AdaQuant stays chaotic, compare it statistically. AdaRound agrees in float64
already, so it keeps it.

``MemOptLevel=2`` is a different training loop in Quark (its ``DataLoader``
path): samples are whole calibration *batches* (``np.load(f).squeeze(0)``),
epochs of ``len // batch_size`` shuffled mini-batches of which the last is never
used, an early stop with patience two checked per epoch, no ``LRAdjust``, no
``parallel`` capture, and ``NumWorkers=0`` with ``batch_size > 1`` fails in
every layer (``DataLoader`` options); all of it is mirrored. ``DynamicBatch``
only works in Quark for readers that yield one sample per batch (otherwise ONNX
Runtime rejects the input for every layer) and is a no-op then.

``SelectiveUpdate`` is checked twice, as in Quark: per module (when the error
after training is worse than the *initial* one of the hard-rounded float weight,
the module's weight and bias are dropped; its ``DataLoader`` loop has no such
check) and after every layer on the whole model's average L2 distance to the
float one.

``SaveAndRestore`` (:func:`load_saved_layers`, :func:`save_checkpoint`): before
each layer Quark writes ``model_to_finetune`` (the model so far, to ``<json
path>.onnx``) and ``layers_to_finetune`` (this layer to the last) into the JSON
file, and when the file already exists it trains only the layers it lists --
on top of the *original* quantized model: the model it loads from the file is
dropped (its ``Subgraph`` was built before), so a resume does not continue from the
saved weights. ``SelectMaxMemLayer`` builds every layer's torch module up front
(it draws torch's random numbers; the restored list overrides its choice).

Not replicated (no effect on the numbers, or out of scope): ``optim_device`` /
``infer_device`` / ``pin_memory`` / ``use_gds`` / ``log_period`` / ``cache_dir``
and ``mem_opt_level`` 0 vs 1 (only choose where tensors are cached and which
device runs them; probed: identical codes). At ``MemOptLevel=2`` Quark does not
check that the quantized and float inputs have the same shape when
``DropRatio`` is 1 (the float input is unused then); this port skips the
layer. Calibration batches with a leading axis > 1 at ``MemOptLevel=2`` are
mirrored only as far as the numpy ops reach (``Conv`` fails in Quark and is
skipped here; ``MatMul`` / ``Gemm`` / norms train on the stacked batches). A
bias-less ``Gemm`` draws its Linear bias from numpy's generator unless a test
hands in torch's (``block_hook``). The one deliberate addition is ``guard``
(default on): a layer's new codes are kept only if the block's reconstruction
error on all samples did not get worse, which Quark has no equivalent of.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, numpy_helper

from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.quark_weight_rounding import (
    LayerReport,
    _attr,
    _avg_l2,
    _copy_tensor,
)

TARGET_OPS = (
    "Conv",
    "ConvTranspose",
    "InstanceNormalization",
    "LayerNormalization",
    "Gemm",
    "MatMul",
)
_ACT_OPS = (
    "Relu",
    "PRelu",
    "LeakyRelu",
    "Gelu",
    "Tanh",
    "Clip",
    "Sigmoid",
    "Softmax",
)
_GAMMA, _ZETA = -0.1, 1.1  # AdaRound's rectified-sigmoid stretch

_RANGES = {
    TensorProto.INT8: (-128.0, 127.0),
    TensorProto.UINT8: (0.0, 255.0),
    TensorProto.INT16: (-32768.0, 32767.0),
    TensorProto.UINT16: (0.0, 65535.0),
    TensorProto.INT32: (-(2.0**31), 2.0**31 - 1),
}


@dataclass
class FinetuneOptions:
    """Quark's ``FastFinetune`` options (``AdaRoundConfig`` / ``AdaQuantConfig``
    names in snake case). ``learning_rate=None`` picks Quark's per-algorithm
    default (``0.1`` AdaRound, ``1e-5`` AdaQuant)."""

    algorithm: str = "adaround"
    num_iterations: int = 1000
    learning_rate: Optional[float] = None
    batch_size: int = 1
    num_batches: int = 1
    early_stop: bool = False
    reg_param: float = 0.01
    beta_range: Tuple[float, float] = (20.0, 2.0)
    warm_start: float = 0.2
    drop_ratio: float = 1.0
    lr_adjust: Optional[Tuple[float, float]] = None
    selective_update: bool = False
    update_bias: bool = False
    output_qdq: bool = False
    parallel: bool = False
    mem_opt_level: int = 1
    num_workers: int = 1
    dynamic_batch: bool = False
    output_index: Optional[int] = None
    select_max_mem_layer: bool = False
    target_ops: Sequence[str] = TARGET_OPS
    seed: int = 1705472343
    guard: bool = True
    #: run AdaQuant's training loop in float32 in torch's operation order
    #: (``None``: on for AdaQuant, which is chaotic enough that float64 noise
    #: flips codes; AdaRound agrees with Quark in float64 already)
    float32: Optional[bool] = None

    def lr(self) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return 1e-5 if self.algorithm == "adaquant" else 0.1

    def use_float32(self) -> bool:
        if self.float32 is not None:
            return bool(self.float32)
        return self.algorithm == "adaquant"


# -- quantized constants & activation quantizers -----------------------------------------


def _clamp_grad(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The gradient factor of ``torch.clamp(q, min_q, max_q)`` with *tensor*
    bounds, which is what Quark's integer quantizers call: 1 inside, 0 outside
    and 0.5 *at* a bound (torch splits the gradient of a tie between the input
    and the bound, unlike the scalar-bound ``clamp`` that gives 0 there). The
    largest weight of a per-tensor symmetric grid sits exactly on ``hi`` --
    one halved gradient is enough to move its code (and, by Adam's per-element
    normalization, the codes around it) away from a run that treats the tie as
    inside."""
    inside = (q > lo) & (q < hi)
    tie = (q == lo) | (q == hi)
    return np.where(inside, 1.0, np.where(tie, 0.5, 0.0)).astype(q.dtype, copy=False)


@dataclass
class _QConst:
    """A weight / bias quantized as ``DequantizeLinear(int codes, scale, zp)``."""

    name: str  # the integer initializer
    codes: np.ndarray
    scale: np.ndarray  # broadcast to codes.shape
    zp: np.ndarray
    lo: float
    hi: float

    def dequant(self, codes: Optional[np.ndarray] = None) -> np.ndarray:
        c = self.codes if codes is None else codes
        return (c.astype(np.float64) - self.zp) * self.scale

    def ste(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize-dequantize with Quark's straight-through gradient mask."""
        q = np.round(w / self.scale) + self.zp
        mask = _clamp_grad(q, self.lo, self.hi)
        return (np.clip(q, self.lo, self.hi) - self.zp) * self.scale, mask

    def encode(self, w: np.ndarray) -> np.ndarray:
        return np.clip(np.round(w / self.scale) + self.zp, self.lo, self.hi)

    def ste32(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """:meth:`ste` as torch's float32 quantizer computes it (``round(w /
        scale) + zp``, ``clamp``, ``(q - zp) * scale``, every op in float32)."""
        f32 = np.float32
        s, z = self.scale.astype(f32), self.zp.astype(f32)
        q = np.round(w.astype(f32) / s) + z
        mask = _clamp_grad(q, f32(self.lo), f32(self.hi))
        return (np.clip(q, f32(self.lo), f32(self.hi)) - z) * s, mask

    def encode32(self, w: np.ndarray) -> np.ndarray:
        f32 = np.float32
        q = np.round(w.astype(f32) / self.scale.astype(f32)) + self.zp.astype(f32)
        return np.clip(q, f32(self.lo), f32(self.hi)).astype(np.float64)


@dataclass
class _ActQ:
    """An activation Q/DQ pair. Torch's quantizer works in float32 (``x / scale``,
    ``round`` half to even, ``+ zp``, ``clamp``, ``- zp``, ``* scale``); doing
    the same keeps its rounding decisions -- they matter for 16-bit codes, where
    float32 cannot tell ``x / scale`` from a half-way tie within ~0.004 of a
    code."""

    scale: float
    zp: float
    lo: float
    hi: float
    pre: str = ""  # float tensor feeding the QuantizeLinear (input quantizers)

    def _q(self, x: np.ndarray) -> np.ndarray:
        f32 = np.float32
        return np.round(x.astype(f32) / f32(self.scale)) + f32(self.zp)

    def fq(self, x: np.ndarray) -> np.ndarray:
        f32 = np.float32
        q = np.clip(self._q(x), f32(self.lo), f32(self.hi))
        y = (q - f32(self.zp)) * f32(self.scale)
        return y if x.dtype == f32 else y.astype(np.float64)

    def fq_mask(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f32 = np.float32
        q = self._q(x)
        y = (np.clip(q, f32(self.lo), f32(self.hi)) - f32(self.zp)) * f32(self.scale)
        return (
            y if x.dtype == f32 else y.astype(np.float64),
            _clamp_grad(q, f32(self.lo), f32(self.hi)),
        )


def _broadcast(
    values: np.ndarray, shape: Tuple[int, ...], axis: int
) -> Optional[np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 1:
        return np.full(shape, float(v.reshape(-1)[0]))
    if v.ndim == 1 and shape and v.size == shape[axis % len(shape)]:
        s = [1] * len(shape)
        s[axis % len(shape)] = -1
        return np.broadcast_to(v.reshape(s), shape).copy()
    return None


def _qconst(
    dq: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[_QConst]:
    """The folded-Q ``DequantizeLinear`` over an integer initializer, if ``dq``
    is one (per-tensor / per-axis scales only)."""
    if dq.op_type != "DequantizeLinear" or len(dq.input) < 3 or not dq.input[2]:
        return None
    codes_t, scale_t, zp_t = (inits.get(i) for i in dq.input[:3])
    if (
        codes_t is None
        or scale_t is None
        or zp_t is None
        or codes_t.data_type not in _RANGES
        or zp_t.data_type != codes_t.data_type
    ):
        return None
    codes = numpy_helper.to_array(codes_t)
    axis = int(_attr(dq, "axis", 1))
    scale = _broadcast(numpy_helper.to_array(scale_t), codes.shape, axis)
    zp = _broadcast(numpy_helper.to_array(zp_t), codes.shape, axis)
    if scale is None or zp is None or _attr(dq, "block_size", 0):
        return None
    lo, hi = _RANGES[codes_t.data_type]
    return _QConst(codes_t.name, codes, scale, zp, lo, hi)


def _act_quant(
    dq: onnx.NodeProto, inits: Dict[str, onnx.TensorProto], pre: str = ""
) -> Optional[_ActQ]:
    if dq.op_type != "DequantizeLinear" or len(dq.input) < 3 or not dq.input[2]:
        return None
    s, z = inits.get(dq.input[1]), inits.get(dq.input[2])
    if s is None or z is None or z.data_type not in _RANGES:
        return None
    sv, zv = numpy_helper.to_array(s), numpy_helper.to_array(z)
    if sv.size != 1 or zv.size != 1:
        return None
    lo, hi = _RANGES[z.data_type]
    return _ActQ(float(sv.reshape(-1)[0]), float(zv.reshape(-1)[0]), lo, hi, pre=pre)


# -- the compute ops, in natural layout, with hand-derived weight gradients -----------------


class _Op:
    #: axis of the output the bias runs along (None: the last axis)
    bias_axis: Optional[int] = None

    def forward(self, x: np.ndarray, w: np.ndarray):  # -> (y, ctx)
        raise NotImplementedError

    def backward(self, ctx, dy: np.ndarray) -> np.ndarray:  # -> dw
        raise NotImplementedError

    def add_bias(self, y: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.bias_axis is None:
            return y + b
        shape = [1] * y.ndim
        shape[self.bias_axis] = -1
        return y + b.reshape(shape)

    def bias_grad(self, dy: np.ndarray) -> np.ndarray:
        if self.bias_axis is None:
            return dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        axes = tuple(i for i in range(dy.ndim) if i != self.bias_axis)
        return dy.sum(axis=axes)


class _MatMulOp(_Op):
    def __init__(self, transposed: bool, trans_a: bool = False) -> None:
        self.t = transposed  # weight stored [N, K]
        self.ta = trans_a  # Gemm ``transA``: the input is used as ``x^T``

    def forward(self, x, w):
        wk = w.T if self.t else w
        xa = np.swapaxes(x, -1, -2) if self.ta else x
        return xa @ wk, xa

    def backward(self, ctx, dy):
        x2 = ctx.reshape(-1, ctx.shape[-1])
        dw = x2.T @ dy.reshape(-1, dy.shape[-1])
        return dw.T if self.t else dw

    # Quark's wrapper broadcasts a bias along axis 1 whenever the output's
    # axis 1 happens to have the bias' length (``bias.view(1, -1, 1, ...)``),
    # which for a >= 3-D Gemm output is not the last axis
    @staticmethod
    def _bias_axis(shape: Sequence[int], n: int) -> Optional[int]:
        return 1 if len(shape) >= 3 and shape[1] == n else None

    def add_bias(self, y, b):
        axis = self._bias_axis(y.shape, b.shape[0])
        if axis is None:
            return y + b
        shape = [1] * y.ndim
        shape[axis] = -1
        return y + b.reshape(shape)

    def bias_grad(self, dy):
        axis = self._bias_axis(dy.shape, dy.shape[-1]) if dy.ndim >= 3 else None
        if axis is None:
            return dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        return dy.sum(axis=tuple(i for i in range(dy.ndim) if i != axis))


def _quark_is_symmetric(pads: Sequence[int]) -> bool:
    idx = len(pads) // 2
    return all(pads[i] == pads[idx + i] for i in range(idx))


def _quark_pad_list(pads: Sequence[int]) -> List[int]:
    """Quark's ``extract_padding_params``: the ONNX ``pads`` of an asymmetric
    convolution as the argument list of ``torch.nn.ConstantPad{n}d`` (last
    dimension first). Its "swap H and W" step is only right for 2-D pads: for
    3-D ones it scrambles the three axes (and drops them to the last axis'
    pair when the first four entries are zero), which is reproduced here."""
    n = len(pads) // 2
    if n == 0:
        return []
    p = np.array(pads).reshape(-1, n)
    if n > 1:
        p[:, [-2, -1]] = p[:, [-1, -2]]
    p = p.T.flatten()
    if n > 2 and (p[:4] == 0).all():
        p = p[4:]
    return [int(v) for v in p]


def _pad_list_widths(pad_list: Sequence[int], nd: int) -> List[Tuple[int, int]]:
    """``torch.nn.functional.pad`` argument list -> per-spatial-axis
    ``(before, after)`` (the list starts at the last axis)."""
    widths = [(0, 0)] * nd
    for j in range(len(pad_list) // 2):
        widths[nd - 1 - j] = (int(pad_list[2 * j]), int(pad_list[2 * j + 1]))
    return widths


def _conv_geometry(node: onnx.NodeProto, nd: int):
    strides = tuple(int(v) for v in _attr(node, "strides", [1] * nd))
    dil = tuple(int(v) for v in _attr(node, "dilations", [1] * nd))
    pads = [int(v) for v in _attr(node, "pads", [0] * (2 * nd))]
    return strides, dil, pads


class _ConvOp(_Op):
    """N-d ``Conv`` (any number of groups). With ``quark_pads`` the padding is
    what Quark's torch module does: asymmetric pads become a separate
    ``ConstantPad`` layer in front, built by :func:`_quark_pad_list`."""

    bias_axis = 1

    def __init__(
        self,
        node: onnx.NodeProto,
        w_shape: Tuple[int, ...],
        quark_pads: bool = False,
    ) -> None:
        nd = len(w_shape) - 2
        self.nd = nd
        self.strides, self.dil, pads = _conv_geometry(node, nd)
        self.group = int(_attr(node, "group", 1))
        self.pad_layer: Optional[List[int]] = None  # Quark's ConstantPad argument
        if len(pads) != 2 * nd or _quark_is_symmetric(pads):
            self.widths = [(pads[i], pads[i]) for i in range(min(nd, len(pads) // 2))]
            self.widths += [(0, 0)] * (nd - len(self.widths))
        else:
            self.pad_layer = _quark_pad_list(pads)
            self.widths = (
                _pad_list_widths(self.pad_layer, nd)
                if quark_pads
                else [(pads[i], pads[nd + i]) for i in range(nd)]
            )

    def pad_layer_shape(self, x_shape: Sequence[int]) -> Optional[List[int]]:
        """The input shape after Quark's pad layer (``None``: it has none)."""
        if self.pad_layer is None:
            return None
        shape = list(x_shape)
        for i, (b, a) in enumerate(self.widths):
            shape[2 + i] += b + a
        return shape

    def forward(self, x, w):
        nd, g = self.nd, self.group
        o, ig = w.shape[0], w.shape[1]
        ksize = w.shape[2:]
        og = o // g
        xp = np.pad(x, [(0, 0), (0, 0)] + list(self.widths))
        eff = [(ksize[i] - 1) * self.dil[i] + 1 for i in range(nd)]
        win = sliding_window_view(xp, eff, axis=tuple(range(2, 2 + nd)))
        win = win[
            (slice(None), slice(None))
            + tuple(slice(None, None, s) for s in self.strides)
            + tuple(slice(None, None, d) for d in self.dil)
        ]  # [b, c, *out, *k]
        b = x.shape[0]
        out = win.shape[2 : 2 + nd]
        perm = (0, *range(2, 2 + nd), 1, *range(2 + nd, 2 + 2 * nd))
        cols, ys = [], []
        for gi in range(g):
            c = (
                win[:, gi * ig : (gi + 1) * ig]
                .transpose(perm)
                .reshape(b * int(np.prod(out)), -1)
            )
            cols.append(c)
            ys.append(c @ w[gi * og : (gi + 1) * og].reshape(og, -1).T)
        y = np.concatenate(ys, axis=1).reshape(b, *out, o)
        return np.moveaxis(y, -1, 1), (cols, w.shape)

    def backward(self, ctx, dy):
        cols, wshape = ctx
        o, ig = wshape[0], wshape[1]
        og = o // self.group
        dyr = np.moveaxis(dy, 1, -1).reshape(-1, o)
        parts = [
            (dyr[:, gi * og : (gi + 1) * og].T @ cols[gi]).reshape(og, ig, *wshape[2:])
            for gi in range(self.group)
        ]
        return np.concatenate(parts, axis=0)


class _ConvTransposeOp(_Op):
    """N-d ``ConvTranspose`` (any number of groups); ``pads`` crop the full
    output. (Quark cannot train it with asymmetric pads, see ``_make_block``.)"""

    bias_axis = 1

    def __init__(self, node: onnx.NodeProto, w_shape: Tuple[int, ...]) -> None:
        nd = len(w_shape) - 2
        self.nd = nd
        self.strides, self.dil, pads = _conv_geometry(node, nd)
        self.group = int(_attr(node, "group", 1))
        self.crop = [(pads[i], pads[nd + i]) for i in range(nd)]
        self.pad_layer: Optional[List[int]] = (
            None if _quark_is_symmetric(pads) else _quark_pad_list(pads)
        )

    def pad_layer_shape(self, x_shape: Sequence[int]) -> Optional[List[int]]:
        if self.pad_layer is None:
            return None
        shape = list(x_shape)
        for i, (b, a) in enumerate(_pad_list_widths(self.pad_layer, self.nd)):
            shape[2 + i] += b + a
        return shape

    def _slices(self, sp, ksize, kidx):
        return tuple(
            slice(
                kidx[i] * self.dil[i],
                kidx[i] * self.dil[i] + self.strides[i] * sp[i],
                self.strides[i],
            )
            for i in range(self.nd)
        )

    def _full(self, sp, ksize):
        return [
            self.strides[i] * (sp[i] - 1) + self.dil[i] * (ksize[i] - 1) + 1
            for i in range(self.nd)
        ]

    def forward(self, x, w):
        nd, g = self.nd, self.group
        b, c = x.shape[:2]
        sp, ksize = x.shape[2:], w.shape[2:]
        cg, og = c // g, w.shape[1]
        full = self._full(sp, ksize)
        buf = np.zeros((b, g * og, *full), dtype=x.dtype)
        for gi in range(g):
            cols = np.tensordot(
                x[:, gi * cg : (gi + 1) * cg],
                w[gi * cg : (gi + 1) * cg],
                axes=([1], [0]),
            )  # [b, *sp, og, *k]
            view = buf[:, gi * og : (gi + 1) * og]
            for kidx in itertools.product(*[range(k) for k in ksize]):
                piece = cols[(slice(None),) * (nd + 2) + kidx]  # [b, *sp, og]
                view[(slice(None), slice(None)) + self._slices(sp, ksize, kidx)] += (
                    np.moveaxis(piece, -1, 1)
                )
        crop = tuple(slice(p0, full[i] - p1) for i, (p0, p1) in enumerate(self.crop))
        return buf[(slice(None), slice(None)) + crop], (x, w.shape, full)

    def backward(self, ctx, dy):
        x, wshape, full = ctx
        nd, g = self.nd, self.group
        b, c = x.shape[:2]
        sp, ksize = x.shape[2:], wshape[2:]
        cg, og = c // g, wshape[1]
        dfull = np.zeros((b, g * og, *full), dtype=dy.dtype)
        crop = tuple(slice(p0, full[i] - p1) for i, (p0, p1) in enumerate(self.crop))
        dfull[(slice(None), slice(None)) + crop] = dy
        axes = [0, *range(2, 2 + nd)]
        dw = np.empty((c, og, *ksize), dtype=dy.dtype)
        for gi in range(g):
            xg = x[:, gi * cg : (gi + 1) * cg]
            dview = dfull[:, gi * og : (gi + 1) * og]
            for kidx in itertools.product(*[range(k) for k in ksize]):
                piece = dview[
                    (slice(None), slice(None)) + self._slices(sp, ksize, kidx)
                ]
                dw[(slice(gi * cg, (gi + 1) * cg), slice(None)) + kidx] = np.tensordot(
                    xg, piece, axes=(axes, axes)
                )
        return dw


class _LayerNormOp(_Op):
    def __init__(self, eps: float) -> None:
        self.eps = eps

    def forward(self, x, w):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        return xhat * w, xhat

    def backward(self, ctx, dy):
        return (dy * ctx).reshape(-1, dy.shape[-1]).sum(axis=0)


class _InstanceNormOp(_Op):
    bias_axis = 1

    def __init__(self, eps: float) -> None:
        self.eps = eps

    def forward(self, x, w):
        axes = tuple(range(2, x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        shape = [1] * x.ndim
        shape[1] = -1
        return xhat * w.reshape(shape), xhat

    def backward(self, ctx, dy):
        axes = tuple(i for i in range(dy.ndim) if i != 1)
        return (dy * ctx).sum(axis=axes)


# -- activations after the op ----------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(x).astype(x.dtype)


class _Act:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, z: np.ndarray, a: np.ndarray, da: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Relu(_Act):
    def forward(self, z):
        return np.maximum(z, 0.0)

    def backward(self, z, a, da):
        return da * (z > 0)


class _LeakyRelu(_Act):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z, a, da):
        return da * np.where(z > 0, 1.0, self.alpha)


class _Clip(_Act):
    """``torch.clamp(z, lo, hi)`` (gradient where ``lo <= z <= hi``); with
    ``strict`` it is ``torch.nn.ReLU6`` (``hardtanh``: ``lo < z < hi``)."""

    def __init__(self, lo: float, hi: float, strict: bool = False) -> None:
        self.lo, self.hi, self.strict = lo, hi, strict

    def forward(self, z):
        return np.clip(z, self.lo, self.hi)

    def backward(self, z, a, da):
        if self.strict:
            return da * ((z > self.lo) & (z < self.hi))
        return da * ((z >= self.lo) & (z <= self.hi))


class _Identity(_Act):
    def forward(self, z):
        return z

    def backward(self, z, a, da):
        return da


class _PRelu(_Act):
    """Quark builds ``torch.nn.PReLU()`` whatever the node's slope input is: one
    shared parameter at its initial value 0.25 that its optimizer never
    touches, so that is the slope the layer is trained with."""

    SLOPE = 0.25

    def forward(self, z):
        return np.where(z > 0, z, self.SLOPE * z)

    def backward(self, z, a, da):
        return da * np.where(z > 0, 1.0, self.SLOPE)


class _Sigmoid(_Act):
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z, a, da):
        return da * a * (1.0 - a)


class _Tanh(_Act):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a, da):
        return da * (1.0 - a * a)


class _Gelu(_Act):
    # Quark's module is torch.nn.GELU() whatever the node's ``approximate``
    def forward(self, z):
        return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))

    def backward(self, z, a, da):
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return da * (0.5 * (1.0 + _erf(z / math.sqrt(2.0))) + z * pdf)


class _Softmax(_Act):
    def __init__(self, axis: int) -> None:
        self.axis = axis

    def forward(self, z):
        e = np.exp(z - z.max(axis=self.axis, keepdims=True))
        return e / e.sum(axis=self.axis, keepdims=True)

    def backward(self, z, a, da):
        return a * (da - (da * a).sum(axis=self.axis, keepdims=True))


def _make_act(
    node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[_Act]:
    """The torch module Quark's ``convert_act`` builds for an activation node
    (``None``: it cannot convert it, the layer is skipped)."""
    t = node.op_type
    if t == "Relu":
        return _Relu()
    if t == "PRelu":
        return _PRelu()
    if t == "LeakyRelu":
        return _LeakyRelu(float(_attr(node, "alpha", 0.01)))
    if t == "Sigmoid":
        return _Sigmoid()
    if t == "Tanh":
        return _Tanh()
    if t == "Gelu":
        return _Gelu()
    if t == "Softmax":
        return _Softmax(int(_attr(node, "axis", -1)))
    if t == "Clip":
        if len(node.input) == 1:
            lo, hi = _attr(node, "min", None), _attr(node, "max", None)
            if lo is None or hi is None:
                return _Identity()  # Quark's Clip module without both bounds
            return _Clip(float(lo), float(hi))
        if len(node.input) == 3 and node.input[1] in inits and node.input[2] in inits:
            lo_a = numpy_helper.to_array(inits[node.input[1]])
            hi_a = numpy_helper.to_array(inits[node.input[2]])
            if lo_a.size != 1 or hi_a.size != 1:
                return None  # ``.item()`` fails: Quark cannot convert it
            return _Clip(float(lo_a.reshape(-1)[0]), float(hi_a.reshape(-1)[0]))
        # bounds that are not both initializers (empty input, a Constant node,
        # only a min): Quark falls back to its ``Clip -> ReLU6`` table entry
        return _Clip(0.0, 6.0, strict=True)
    return None


# -- finding the blocks ------------------------------------------------------------------------


@dataclass
class _Block:
    name: str
    op_type: str
    op: _Op
    w_float: np.ndarray
    qw: _QConst
    qb: Optional[_QConst]  # quantized bias
    b_float: Optional[np.ndarray]  # the float model's bias (what AdaQuant trains)
    b_plain: Optional[np.ndarray]  # an unquantized bias, used as is
    w_alpha: float
    b_beta: float
    in_q: Optional[_ActQ]
    q_start: str  # quantized-model tensor the block starts from (pre input Q)
    f_start: str
    f_end: str
    act: Optional[_Act]
    out_q: Optional[_ActQ]
    #: bound of the random bias Quark's Gemm module owns although the node has
    #: none (``b_plain`` is filled with the draw before training)
    phantom_bias: Optional[float] = None


def _consumers(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    out: Dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for i in n.input:
            out.setdefault(i, n)  # Quark's converter takes the first consumer
    return out


class _QuarkConversionError(Exception):
    """Quark's ``convert_onnx_to_torch`` raises for this layer: its driver logs
    it and skips the layer (``SelectMaxMemLayer`` converts every layer up front
    without a guard, so there it aborts the whole run)."""


def _find_blocks(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    opt: FinetuneOptions,
    conversion_errors: Optional[List[str]] = None,
) -> List[_Block]:
    f_inits = {t.name: t for t in float_model.graph.initializer}
    q_inits = {t.name: t for t in quant_model.graph.initializer}
    q_prod = {o: n for n in quant_model.graph.node for o in n.output}
    q_cons = _consumers(quant_model)
    f_cons = _consumers(float_model)
    f_by_name: Dict[str, List[onnx.NodeProto]] = {}
    f_by_weight: Dict[Tuple[str, str], onnx.NodeProto] = {}
    for fn in float_model.graph.node:
        f_by_name.setdefault(fn.name, []).append(fn)
        if len(fn.input) > 1:
            f_by_weight[(fn.op_type, fn.input[1])] = fn

    use_count: Dict[str, int] = {}
    for n in quant_model.graph.node:
        for i in n.input:
            use_count[i] = use_count.get(i, 0) + 1

    blocks: List[_Block] = []
    for qn in quant_model.graph.node:
        if qn.op_type not in opt.target_ops or qn.op_type not in TARGET_OPS:
            continue
        if len(qn.input) < 2 or qn.output[0] not in q_cons:
            continue
        try:
            blk = _make_block(
                qn,
                q_prod,
                q_inits,
                q_cons,
                f_inits,
                f_cons,
                f_by_name,
                f_by_weight,
                use_count,
                opt,
            )
        except _QuarkConversionError as e:
            if conversion_errors is not None:
                conversion_errors.append(f"{qn.name or qn.output[0]}: {e}")
            blk = None
        if blk is not None:
            blocks.append(blk)
    seen: Dict[str, int] = {}
    for b in blocks:
        seen[b.qw.name] = seen.get(b.qw.name, 0) + 1
    return [b for b in blocks if seen[b.qw.name] == 1]


def _make_block(
    qn: onnx.NodeProto,
    q_prod: Dict[str, onnx.NodeProto],
    q_inits: Dict[str, onnx.TensorProto],
    q_cons: Dict[str, onnx.NodeProto],
    f_inits: Dict[str, onnx.TensorProto],
    f_cons: Dict[str, onnx.NodeProto],
    f_by_name: Dict[str, List[onnx.NodeProto]],
    f_by_weight: Dict[Tuple[str, str], onnx.NodeProto],
    use_count: Dict[str, int],
    opt: FinetuneOptions,
) -> Optional[_Block]:
    t = qn.op_type
    # -- the weight: DequantizeLinear over an integer initializer ---------------------
    wdq = q_prod.get(qn.input[1])
    if wdq is None:
        return None
    qw = _qconst(wdq, q_inits)
    if (
        qw is None
        or use_count.get(qw.name, 0) != 1
        or use_count.get(qn.input[1], 0) != 1
    ):
        return None
    # -- the float node: by name, else through the weight's name ------------------------
    fn = None
    cands = f_by_name.get(qn.name, []) if qn.name else []
    if len(cands) == 1 and cands[0].op_type == t:
        fn = cands[0]
    if fn is None:
        fn = f_by_weight.get((t, wdq.input[0].split("/qdq")[0].split("_quantized")[0]))
    if fn is None or len(fn.input) < 2 or fn.input[1] not in f_inits:
        return None
    w_float = numpy_helper.to_array(f_inits[fn.input[1]]).astype(np.float64)
    if (
        w_float.shape != qw.codes.shape
        or f_inits[fn.input[1]].data_type != TensorProto.FLOAT
    ):
        return None

    # -- the input: DequantizeLinear <- QuantizeLinear (float tensor "pre") -----------------
    in_dq = q_prod.get(qn.input[0])
    if in_dq is None:
        return None  # Quark: no producer, no block
    in_q: Optional[_ActQ] = None
    q_start = qn.input[0]
    if in_dq.op_type == "DequantizeLinear":
        q_node = q_prod.get(in_dq.input[0])
        if q_node is not None and q_node.op_type == "QuantizeLinear":
            in_q = _act_quant(in_dq, q_inits, pre=q_node.input[0])
            if in_q is None:
                return None
            q_start = q_node.input[0]

    # -- op, layout, attributes -----------------------------------------------------------------
    op: _Op
    w_alpha, b_beta = 1.0, 1.0
    if t == "MatMul":
        if w_float.ndim != 2:
            return None
        op = _MatMulOp(False)
    elif t == "Gemm":
        if w_float.ndim != 2:
            return None
        op = _MatMulOp(bool(_attr(qn, "transB", 0)), bool(_attr(qn, "transA", 0)))
        w_alpha, b_beta = float(_attr(qn, "alpha", 1.0)), float(_attr(qn, "beta", 1.0))
    elif t in ("Conv", "ConvTranspose"):
        if w_float.ndim not in (3, 4, 5):  # Quark has 1-D / 2-D / 3-D modules
            return None
        # what Quark's ``convert_conv`` raises on (the layer is then skipped)
        auto_pad = _attr(qn, "auto_pad", b"NOTSET")
        if auto_pad not in (b"NOTSET", "NOTSET"):
            raise _QuarkConversionError(
                f"auto_pad={auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad}"
                " functionality not implemented."
            )
        if t == "Conv":
            op = _ConvOp(qn, w_float.shape, quark_pads=True)
        else:
            # Quark raises as soon as the attribute exists, whatever its value
            for a in qn.attribute:
                if a.name in ("output_padding", "output_shape"):
                    raise _QuarkConversionError(
                        f"ConvTranspose with {a.name} not implemented."
                    )
            cop = _ConvTransposeOp(qn, w_float.shape)
            if cop.pad_layer is not None:
                # asymmetric pads: Quark puts a ConstantPad in front of a
                # ConvTranspose with no padding, whose output no longer has the
                # float layer's shape -- its first forward fails, layer skipped
                return None
            op = cop
    elif t == "LayerNormalization":
        if int(_attr(qn, "axis", -1)) != -1:
            raise _QuarkConversionError(
                "LayerNorm whose axis is not -1 is not supported."
            )
        if w_float.ndim != 1:
            return None
        op = _LayerNormOp(float(_attr(qn, "epsilon", 1e-5)))
    else:  # InstanceNormalization
        if w_float.ndim != 1:
            return None
        op = _InstanceNormOp(float(_attr(qn, "epsilon", 1e-5)))

    # -- bias ------------------------------------------------------------------------------------
    qb: Optional[_QConst] = None
    b_float: Optional[np.ndarray] = None
    b_plain: Optional[np.ndarray] = None
    if len(qn.input) > 2 and qn.input[2]:
        bdq = q_prod.get(qn.input[2])
        if bdq is not None:
            qb = _qconst(bdq, q_inits)
            if qb is None:
                return None
            if len(fn.input) < 3 or fn.input[2] not in f_inits:
                return None
            b_float = numpy_helper.to_array(f_inits[fn.input[2]]).astype(np.float64)
            if b_float.shape != qb.codes.shape:
                return None
        elif qn.input[2] in q_inits:
            b_plain = numpy_helper.to_array(q_inits[qn.input[2]]).astype(np.float64)
        else:
            return None
    elif t == "InstanceNormalization":
        return None
    phantom: Optional[float] = None
    if t == "Gemm" and qb is None and b_plain is None:
        # a bias-less Gemm still gets ``torch.nn.Linear``'s randomly initialized
        # bias in Quark's module, which nothing overwrites and which is added
        # to every forward: U(-1/sqrt(K), 1/sqrt(K)) over the N outputs
        n_out = w_float.shape[0] if getattr(op, "t", False) else w_float.shape[1]
        k_in = w_float.shape[1] if getattr(op, "t", False) else w_float.shape[0]
        phantom = 1.0 / math.sqrt(k_in)
        b_plain = np.zeros(n_out)

    # -- the end of the block ----------------------------------------------------------------------
    cons = q_cons.get(qn.output[0])
    if cons is None:
        return None
    act: Optional[_Act] = None
    out_q: Optional[_ActQ] = None
    f_end = fn.output[0]
    tail = qn.output[0]
    if cons.op_type in _ACT_OPS:
        act = _make_act(cons, q_inits)
        fcons = f_cons.get(fn.output[0])
        if act is None or fcons is None or fcons.op_type != cons.op_type:
            return None
        f_end, tail = fcons.output[0], cons.output[0]
    # A Relu folded into the output quantizer (``fold_relu``: the Q range starts
    # at 0, e.g. INT8_CNN_DEFAULT on a MatMul / Gemm model) leaves no Relu node
    # in the quantized graph: Quark's block ends at the op's own output (or the
    # output Q/DQ) and its target is the float *pre-Relu* output (the float
    # node's output tensor), so the block is trained without an activation.
    if opt.output_qdq:
        nxt = q_cons.get(tail)
        if nxt is not None and nxt.op_type == "QuantizeLinear":
            dqn = q_cons.get(nxt.output[0])
            if dqn is None or dqn.op_type != "DequantizeLinear":
                return None  # Quark: a Q without its DQ is an error, layer skipped
            out_q = _act_quant(dqn, q_inits)
            if out_q is None:
                return None
    return _Block(
        qn.name or qn.output[0],
        t,
        op,
        w_float,
        qw,
        qb,
        b_float,
        b_plain,
        w_alpha,
        b_beta,
        in_q,
        q_start,
        fn.input[0],
        f_end,
        act,
        out_q,
        phantom,
    )


# -- capturing activations -------------------------------------------------------------------------


def _capture(
    model: onnx.ModelProto,
    names: Sequence[str],
    data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
    optimize: bool,
    split: bool = False,
) -> Dict[str, Any]:
    """``{name: [samples, ...]}``: each tensor over all calibration batches,
    concatenated along the leading axis (``split``: the list of per-batch
    arrays instead). ``optimize=False`` runs ORT without graph optimizations
    (Quark does for the quantized model: ORT would otherwise fuse DQ -> op -> Q
    into integer kernels)."""
    import onnxruntime as ort

    names = sorted(set(names))
    probe = _add_probe_outputs(model, names)
    so = ort.SessionOptions()
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        probe.SerializeToString(),
        so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )
    outs = [o.name for o in sess.get_outputs()]
    acc: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    for batch in data:
        res = dict(zip(outs, sess.run(outs, batch)))
        for n in names:
            acc[n].append(np.asarray(res[n], dtype=np.float64))
    if split:
        return acc
    return {n: np.concatenate(v, axis=0) for n, v in acc.items()}


def _loader_samples(batches: Sequence[np.ndarray]) -> np.ndarray:
    """What Quark's ``TrainDataset`` hands the ``DataLoader`` at ``MemOptLevel``
    2: one sample per calibration *batch* (``torch.from_numpy(np.load(f))
    .squeeze(0)``, which only drops a leading axis of size 1), stacked on a
    new leading axis."""
    if len({b.shape for b in batches}) != 1:
        raise _SkipLayer("calibration batches of different shapes cannot be stacked")
    st = np.stack(batches)
    return st[:, 0] if st.shape[1] == 1 else st


def _model_outputs(
    model: onnx.ModelProto,
    data: Sequence[Dict[str, np.ndarray]],
    providers: Optional[Sequence[str]],
    output_index: Optional[int],
) -> List[List[np.ndarray]]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(),
        so,
        providers=list(providers or ["CPUExecutionProvider"]),
    )
    n_out = len(sess.get_outputs())
    res = [sess.run(None, b) for b in data]
    if output_index is not None and 0 <= output_index < n_out:
        return [[r[output_index]] for r in res]
    return res


# -- training ------------------------------------------------------------------------------------------


class _SkipLayer(Exception):
    """Quark's torch module of this layer fails on its first forward (the
    shapes do not fit), which its driver logs and skips."""


def _adam_step32(p, g, m, v, t, lr):
    """``torch.optim.Adam`` on a float32 CPU tensor, op for op: ``exp_avg.lerp_``,
    ``exp_avg_sq.mul_().addcmul_()``, ``denom = (sqrt(v) / sqrt(bc2)) + eps``,
    ``param.addcdiv_(exp_avg, denom, value=-lr / bc1)``."""
    f32 = np.float32
    w = f32(1.0 - 0.9)
    m[...] = m + w * (g - m)  # lerp, weight < 0.5
    v[...] = v * f32(0.999)
    v[...] = v + (f32(1.0 - 0.999) * g) * g
    step = t + 1
    bc1 = 1.0 - 0.9**step
    bc2 = 1.0 - 0.999**step
    denom = np.sqrt(v) / f32(math.sqrt(bc2)) + f32(1e-8)
    return p + (f32(-(lr / bc1)) * m) / denom


def _sigmoid32(a: np.ndarray) -> np.ndarray:
    """``torch.sigmoid`` on a float32 tensor."""
    f32 = np.float32
    return f32(1.0) / (f32(1.0) + np.exp(-a.astype(f32)))


def _adam_step(p, g, m, v, t, lr):
    m *= 0.9
    m += 0.1 * g
    v *= 0.999
    v += 0.001 * g * g
    mh = m / (1.0 - 0.9 ** (t + 1))
    vh = v / (1.0 - 0.999 ** (t + 1))
    return p - lr * mh / (np.sqrt(vh) + 1e-8)


def _beta(max_iter: int, it: int, beta_range, warm_start: float) -> float:
    start, end = beta_range
    ws = warm_start * max_iter
    rel = (it - ws) / (max_iter - ws)
    return end + 0.5 * (start - end) * (1 + math.cos(rel * math.pi))


def _block_forward(
    blk: _Block,
    x_in: np.ndarray,
    w_hat: np.ndarray,
    bias: Optional[np.ndarray],
    out_mask: bool = False,
):
    """``x_in`` is already fake-quantized. Returns ``(y, cache)``."""
    try:
        z, ctx = blk.op.forward(x_in, w_hat * blk.w_alpha)
        if bias is not None:
            z = blk.op.add_bias(z, bias * blk.b_beta)
    except ValueError as e:  # numpy's shape mismatch is torch's RuntimeError
        raise _SkipLayer(str(e)) from e
    a = z if blk.act is None else blk.act.forward(z).astype(z.dtype, copy=False)
    mask = None
    y = a
    if blk.out_q is not None:
        y, mask = blk.out_q.fq_mask(a)
    return y, (ctx, z, a, mask)


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` back to ``shape`` over the axes numpy broadcast."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


def _diff(y: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """``y - y_ref`` as torch computes it: it broadcasts, so a block output
    that differs from its target only along size-1 axes still trains (Quark
    does, e.g. a ``Gemm`` with ``transA`` on a single one-row sample); any other
    mismatch is the RuntimeError its driver logs and skips."""
    try:
        return y - y_ref
    except ValueError as e:
        raise _SkipLayer(f"output {y.shape} vs target {y_ref.shape}") from e


def _recon_grad(
    blk: _Block, cache, y: np.ndarray, y_ref: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Quark's loss ``mean(sum((y - y_ref)^2, dim=1))`` and the gradients of it
    w.r.t. the quantized-weight tensor and the bias."""
    ctx, z, a, mask = cache
    err = _diff(y, y_ref)
    if err.ndim < 2:
        raise _SkipLayer("loss needs at least two dimensions")
    denom = err.size / err.shape[1]
    loss = float(np.sum(err * err) / denom)
    dy = _unbroadcast(2.0 * err / denom, y.shape)
    if mask is not None:
        dy = dy * mask
    dz = dy if blk.act is None else blk.act.backward(z, a, dy)
    dw = blk.op.backward(ctx, dz) * blk.w_alpha
    db = blk.op.bias_grad(dz) * blk.b_beta
    return loss, dw, db


def _recon_grad32(
    blk: _Block, cache, y: np.ndarray, y_ref: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """:func:`_recon_grad` the way torch's float32 autograd evaluates
    ``(torch.norm(err, 'fro', dim=1) ** 2).mean()``: ``mean`` hands ``1 / N``
    down, ``pow`` multiplies it by ``2 * norm``, and ``norm`` returns
    ``(err / norm) * grad`` (zero where the norm is) -- the same number as
    ``2 err / N`` up to the float32 rounding of every step, which is what flips
    16-bit codes."""
    f32 = np.float32
    ctx, z, a, mask = cache
    err = _diff(y.astype(f32, copy=False), y_ref.astype(f32, copy=False))
    if err.ndim < 2:
        raise _SkipLayer("loss needs at least two dimensions")
    n_out = err.size // err.shape[1]
    norm = np.sqrt((err * err).sum(axis=1, dtype=f32)).astype(f32)
    loss = float((norm * norm).mean(dtype=f32))
    g = f32(1.0) / f32(n_out)
    nshape = list(err.shape)
    nshape[1] = 1
    norm_k = norm.reshape(nshape)
    with np.errstate(divide="ignore", invalid="ignore"):
        dy = np.where(
            norm_k == 0,
            f32(0.0),
            (err / norm_k) * ((g * (f32(2.0) * norm)).reshape(nshape)),
        ).astype(f32)
    dy = _unbroadcast(dy, y.shape)
    if mask is not None:
        dy = dy * mask
    dz = dy if blk.act is None else blk.act.backward(z, a, dy).astype(f32, copy=False)
    dw = blk.op.backward(ctx, dz) * f32(blk.w_alpha)
    db = blk.op.bias_grad(dz) * f32(blk.b_beta)
    return loss, dw, db


def _eval_error(
    blk: _Block,
    x_all: np.ndarray,
    y_all: np.ndarray,
    w_hat: np.ndarray,
    bias: Optional[np.ndarray],
) -> float:
    """Quark's ``_calc_recons_metrics``: ``F.mse_loss`` over every element, on all
    samples at once (a ``Gemm`` with ``transA`` multiplies *across* samples, so
    those are not chunked)."""
    step = x_all.shape[0] if getattr(blk.op, "ta", False) else 64
    tot, count = 0.0, 0
    for i in range(0, x_all.shape[0], step):
        y, _ = _block_forward(blk, x_all[i : i + step], w_hat, bias)
        ref = y_all if step == x_all.shape[0] else y_all[i : i + step]
        d = _diff(y, ref)
        tot += float(np.sum(d * d))
        count += d.size
    return tot / max(count, 1)


@dataclass
class _Trained:
    codes: np.ndarray
    bias_codes: Optional[np.ndarray] = None
    err_rtn: float = 0.0  # Quark's "initial" error (hard-rounded float weight)
    err_final: float = 0.0  # its error after training (hard rounding)
    iterations: int = 0


def _train_block(
    blk: _Block,
    xq: np.ndarray,
    xf: np.ndarray,
    yf: np.ndarray,
    opt: FinetuneOptions,
    perm_fn: Callable[[int], np.ndarray],
    rng: np.random.Generator,
    trace: Optional[List[Tuple[int, float, float]]] = None,
    rand_fn: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
) -> _Trained:
    qw = blk.qw
    s_total = xq.shape[0]
    adaround = opt.algorithm == "adaround"
    f32 = np.float32
    use32 = (not adaround) and opt.use_float32()
    if use32:  # torch's tensors: float32 from the data on
        xq, xf, yf = (a.astype(f32) for a in (xq, xf, yf))
    in_fq = (lambda x: x) if blk.in_q is None else blk.in_q.fq
    x_eval = in_fq(xq)
    # the bias the layer is trained with: Quark feeds the float bias through
    # the bias quantizer (identical to the model's own codes unless AdaQuant
    # updates it)
    b_float = blk.b_float
    bias_fq = blk.b_plain if blk.qb is None else blk.qb.ste(blk.b_float)[0]  # type: ignore[arg-type]
    if use32:
        bias_fq = (
            None
            if bias_fq is None
            else (bias_fq if blk.qb is None else blk.qb.ste32(blk.b_float)[0]).astype(
                f32
            )  # type: ignore[arg-type,union-attr]
        )

    num_iter = int(opt.num_iterations)
    lr = opt.lr()
    w = blk.w_float
    scale, zp, lo, hi = qw.scale, qw.zp, qw.lo, qw.hi

    # Quark computes w / scale in float32; an element sitting exactly on the
    # grid (every channel's largest weight) lands on either side of the
    # floor, and the rectified sigmoid there on either side of 0, by ULP
    # noise that decides whether it is ever allowed to move -- so mirror it
    wd32 = w.astype(f32) / scale.astype(f32)
    floor_w = np.floor(wd32).astype(np.float64)
    diff32 = wd32 - np.floor(wd32)
    # initial (hard-rounded float weight) error: drives LRAdjust
    w_rtn = (np.clip(floor_w + (diff32 >= 0.5) + zp, lo, hi) - zp) * scale
    if adaround:
        w_init_hat = w_rtn
    else:
        w_init_hat = qw.ste(w)[0]
    err0 = 0.0
    if opt.mem_opt_level != 2:  # (the DataLoader loop has no initial metrics)
        err0 = _eval_error(blk, x_eval, yf, w_init_hat, bias_fq)
    if (
        opt.mem_opt_level != 2
        and opt.lr_adjust is not None
        and len(opt.lr_adjust) == 2
        and err0 > opt.lr_adjust[0]
    ):
        lr = float(opt.lr_adjust[1])

    bs = int(opt.batch_size)
    if bs < 1 or bs > s_total:
        bs = 1

    # parameters
    if adaround:
        alpha = -np.log(f32(_ZETA - _GAMMA) / (diff32 - f32(_GAMMA)) - f32(1.0))
        params = [alpha.astype(np.float64)]
    else:
        wv = w.astype(f32) if use32 else w.copy()
        bv = (
            (b_float.astype(f32) if use32 else b_float.copy())
            if (opt.update_bias and blk.qb is not None and b_float is not None)
            else None
        )
        params = [wv] + ([bv] if bv is not None else [])
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]

    best_loss = float("inf")
    mean_loss = 0.0
    es_window = opt.num_batches if opt.num_batches > 1 else num_iter / 10
    ws_iter = num_iter * opt.warm_start
    pre_mixed = None
    if opt.drop_ratio >= 1:
        pre_mixed = x_eval
    elif opt.drop_ratio <= 0:
        pre_mixed = in_fq(xf)

    def _compute(it: int, idx: np.ndarray):
        """One iteration's mini-batch ``idx``: loss and gradients."""
        if pre_mixed is not None:
            x_in = pre_mixed[idx]
        else:
            xqb, xfb = xq[idx], xf[idx]
            u = rand_fn(xqb.shape) if rand_fn is not None else rng.random(xqb.shape)
            x_in = in_fq(np.where(u < opt.drop_ratio, xqb, xfb))
        try:
            y_ref = yf[idx]
        except IndexError as e:  # fewer target rows than samples: Quark's
            raise _SkipLayer(str(e)) from e  # torch.cat fails, layer skipped

        if adaround:
            sig32 = _sigmoid32(params[0])
            raw_h = (sig32 * f32(_ZETA - _GAMMA) + f32(_GAMMA)).astype(np.float64)
            sig = sig32.astype(np.float64)
            h = np.clip(raw_h, 0.0, 1.0)
            raw_q = floor_w + h + zp
            w_hat = (np.clip(raw_q, lo, hi) - zp) * scale
            bias = bias_fq
        elif use32:
            w_hat, wmask = qw.ste32(params[0])
            if len(params) > 1:
                bias, bmask = blk.qb.ste32(params[1])  # type: ignore[union-attr]
            else:
                bias, bmask = bias_fq, None
        else:
            w_hat, wmask = qw.ste(params[0])
            if len(params) > 1:
                bias, bmask = blk.qb.ste(params[1])  # type: ignore[union-attr]
            else:
                bias, bmask = bias_fq, None

        y, cache = _block_forward(blk, x_in, w_hat, bias)
        recons, dw_hat, db = (_recon_grad32 if use32 else _recon_grad)(
            blk, cache, y, y_ref
        )

        round_loss = 0.0
        grads: List[np.ndarray]
        if adaround:
            dq_mask = _clamp_grad(raw_q, lo, hi)
            dh = dw_hat * scale * dq_mask
            h_mask = (raw_h > 0.0) & (raw_h < 1.0)
            dh_dalpha = np.where(h_mask, sig * (1.0 - sig) * (_ZETA - _GAMMA), 0.0)
            if it >= ws_iter:
                beta = _beta(num_iter, it, opt.beta_range, opt.warm_start)
                u = 2.0 * h - 1.0
                round_loss = opt.reg_param * float(np.sum(1.0 - np.abs(u) ** beta))
                dreg = (
                    -2.0 * opt.reg_param * beta * np.sign(u) * np.abs(u) ** (beta - 1.0)
                )
                grads = [(dw_hat * scale * dq_mask + dreg) * dh_dalpha]
            else:
                grads = [dh * dh_dalpha]
        elif use32:
            # torch: out = (clamp(...) - zp) * scale, divided by scale again on
            # the way back to the weight; the clamp lets the gradient through
            # where lo <= q <= hi
            s32 = qw.scale.astype(f32)
            grads = [((dw_hat * s32) * wmask) / s32]
            if len(params) > 1:
                sb = blk.qb.scale.astype(f32)  # type: ignore[union-attr]
                grads.append(((db * sb) * bmask) / sb)  # type: ignore[operator]
        else:
            grads = [dw_hat * wmask]
            if len(params) > 1:
                grads.append(db * bmask)  # type: ignore[operator]
        if trace is not None:
            trace.append((it, recons, round_loss))
        return recons, round_loss, grads

    def _step(it: int, grads: List[np.ndarray]) -> None:
        for k, p in enumerate(params):
            step = _adam_step32 if use32 else _adam_step
            params[k] = step(p, grads[k], ms[k], vs[k], it, lr)

    done = 0
    if opt.mem_opt_level != 2:
        for it in range(num_iter):
            idx = perm_fn(s_total)[:bs]
            recons, round_loss, grads = _compute(it, idx)
            # Quark's early-stop rule, verbatim (it reuses num_batches / warm_start)
            if opt.early_stop and it >= ws_iter:
                if it % es_window == es_window - 1:
                    mean_loss = mean_loss / es_window
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                    else:
                        break
                    mean_loss = 0.0
                else:
                    mean_loss += round_loss if adaround else recons
            _step(it, grads)
            done = it + 1
    else:
        # MemOptLevel 2 is Quark's torch ``DataLoader`` loop: shuffled epochs of
        # ``len(samples) // batch_size`` mini-batches (the last one of an epoch is
        # never used), the early stop is checked per epoch with a patience of
        # two, and there is no LRAdjust
        if bs > 1 and opt.num_workers == 0:
            # DataLoader(persistent_workers=True, prefetch_factor=2) needs workers
            raise _SkipLayer("DataLoader options need NumWorkers > 0")
        num_steps = s_total // bs
        num_epochs = (num_iter + num_steps - 1) // num_steps
        it, no_improve = 0, 0
        for _epoch in range(num_epochs):
            mean_loss = 0.0
            order = perm_fn(s_total)
            for b_idx in range(num_steps):
                if it >= num_iter or b_idx >= num_steps - 1:
                    break
                recons, round_loss, grads = _compute(
                    it, order[b_idx * bs : (b_idx + 1) * bs]
                )
                _step(it, grads)
                it += 1
                mean_loss += round_loss if adaround else recons
            if it >= num_iter:
                break
            if opt.early_stop and it >= num_iter * opt.warm_start:
                mean_loss = mean_loss / num_steps
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 2:
                        break
        done = it

    if adaround:
        codes = np.clip(floor_w + (params[0] >= 0) + zp, lo, hi)
        bcodes = None
        w_final, b_final = qw.dequant(codes), bias_fq
    else:
        enc = (lambda q, v: q.encode32(v)) if use32 else (lambda q, v: q.encode(v))
        ste = (lambda q, v: q.ste32(v)[0]) if use32 else (lambda q, v: q.ste(v)[0])
        codes = enc(qw, params[0])
        bcodes = enc(blk.qb, params[1]) if len(params) > 1 else None
        w_final = qw.dequant(codes)
        b_final = ste(blk.qb, params[1]) if len(params) > 1 else bias_fq
    err_final = 0.0
    if opt.selective_update and opt.mem_opt_level != 2:
        err_final = _eval_error(blk, x_eval, yf, w_final, b_final)
    return _Trained(codes, bcodes, err0, err_final, done)


# -- Quark's ``SaveAndRestore`` checkpoint file ---------------------------------------------


def load_saved_layers(path: Any) -> Optional[List[int]]:
    """What Quark's ``fast_finetune`` restores from ``extra_options
    ["SaveAndRestore"]``: when the JSON file exists, the layer indices it lists
    under ``"layers_to_finetune"`` (``None`` if absent or empty: then every
    layer trains). Quark also loads the ``"model_to_finetune"`` ONNX file the
    JSON names -- which raises if it is missing -- but then drops the model:
    its ``Subgraph`` was built from the original quantized model, so resuming
    does *not* continue from the saved weights, it only skips the layers
    before the one the previous run had reached."""
    if not path or not os.path.exists(str(path)):
        return None
    with open(str(path)) as f:
        saved = json.load(f)
    model_path = saved.get("model_to_finetune")
    if model_path is not None:
        onnx.load(model_path)
    layers = saved.get("layers_to_finetune")
    return [int(i) for i in layers] if layers else None


def save_checkpoint(
    path: Any, index: int, n_layers: int, model: onnx.ModelProto
) -> None:
    """Quark's checkpoint before it fine-tunes layer ``index``: the JSON file
    gets ``"model_to_finetune"`` (``<path>.onnx`` for a ``.json`` path, else
    ``model_to_finetune.onnx`` in the working directory) and
    ``"layers_to_finetune"`` (``index .. n_layers - 1``) -- other keys of an
    existing file are kept -- and the current model is written to that ONNX
    file."""
    path = str(path)
    model_path = (
        path.replace(".json", ".onnx")
        if path.endswith(".json")
        else "model_to_finetune.onnx"
    )
    saved: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
    saved["model_to_finetune"] = model_path
    saved["layers_to_finetune"] = list(range(index, n_layers))
    with open(path, "w") as f:
        json.dump(saved, f, indent=2)
    onnx.save(model, model_path)


# -- the driver --------------------------------------------------------------------------------------


def _shape_ok(blk: _Block, x: np.ndarray) -> bool:
    t = blk.op_type
    if t in ("Gemm", "MatMul"):  # torch.matmul: any batch dims (a Gemm whose
        return x.ndim >= 2  # samples are whole batches at MemOptLevel 2, too)
    if t in ("Conv", "ConvTranspose"):
        return x.ndim == blk.w_float.ndim
    if t == "InstanceNormalization":
        return x.ndim >= 3
    return x.ndim >= 2  # LayerNormalization


def _estimate_memory(
    blk: _Block, y_shape: Tuple[int, ...], x_shape: Tuple[int, ...]
) -> float:
    """Quark's ``estimate_memory`` for one block, in MiB: the torch module's
    parameters (what *its* constructors create: Gemm and the norms always own a
    bias, MatMul never), the outputs of the module's direct children (the op
    after its bias, a pad layer for asymmetric conv pads, the activation, the
    output Q/DQ) for one float *calibration batch*, and Adam's state (3 x
    parameters), all float32."""
    t = blk.op_type
    w = blk.w_float
    if t == "MatMul":
        params = w.size
    elif t == "Gemm":
        params = w.size + (w.shape[0] if getattr(blk.op, "t", False) else w.shape[1])
    elif t in ("LayerNormalization", "InstanceNormalization"):
        params = 2 * w.size
    else:  # Conv / ConvTranspose: a bias only if the node has one
        has_bias = blk.b_float is not None or blk.b_plain is not None
        out_ch = (
            w.shape[1] * getattr(blk.op, "group", 1)
            if t == "ConvTranspose"
            else w.shape[0]
        )
        params = w.size + (out_ch if has_bias else 0)
    n_out = int(np.prod(y_shape))
    acts = n_out * (1 + (blk.act is not None) + (blk.out_q is not None))
    pad_layer = getattr(blk.op, "pad_layer_shape", None)
    padded = None if pad_layer is None else pad_layer(x_shape)
    if padded is not None:  # Quark's separate ConstantPad layer is a child too
        acts += int(np.prod(padded))
    mib = 1024.0**2
    return (params * 4 + acts * 4 + 3 * params * 4) / mib


def finetune(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    options: Optional[FinetuneOptions] = None,
    providers: Optional[Sequence[str]] = None,
    perm_fn: Optional[Callable[[int], np.ndarray]] = None,
    trace: Optional[List[List[Tuple[int, float, float]]]] = None,
    rand_fn: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
    block_hook: Optional[Callable[[int, str], Optional[Dict[str, np.ndarray]]]] = None,
    layers: Optional[Sequence[int]] = None,
    checkpoint: Optional[Callable[[int, int, onnx.ModelProto], None]] = None,
) -> Tuple[onnx.ModelProto, List[LayerReport]]:
    """Quark ``FastFinetune`` over a QDQ model (see the module docstring).

    ``float_model`` is the reference (its tensors are the training targets),
    ``quant_model`` the QDQ model whose integer weight (and, for AdaQuant with
    ``update_bias``, bias) codes get rewritten. Returns the new model and one
    :class:`~onnxsim.quark_weight_rounding.LayerReport` per block.

    ``perm_fn(n)`` returns a permutation of ``range(n)``; the first
    ``batch_size`` entries are an iteration's mini-batch. The default is a
    seeded numpy generator; pass torch's ``randperm`` stream to replay Quark's
    mini-batches exactly. ``trace``, if a list, receives one list of
    ``(iteration, reconstruction loss, rounding loss)`` per trained block;
    ``rand_fn(shape)`` replaces the uniform draw behind ``drop_ratio`` mixing
    and ``block_hook(i, name)`` is called before each block trains (both let
    a test replay Quark's torch random stream); it may return
    ``{"phantom_bias": array}``, the bias torch drew for a bias-less ``Gemm``
    (otherwise numpy draws one).

    ``layers`` restricts training to those block indices (Quark's restored
    ``SaveAndRestore`` list; it also overrides ``select_max_mem_layer``) and
    ``checkpoint(index, n_blocks, model)`` is called before each selected block
    with the model as it stands (see :func:`save_checkpoint`).
    """
    opt = options or FinetuneOptions()
    if opt.algorithm not in ("adaround", "adaquant"):
        raise ValueError(f"unknown algorithm {opt.algorithm!r}")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    out = onnx.ModelProto()
    out.CopyFrom(quant_model)
    if opt.dynamic_batch:
        # Quark makes every input's batch axis ``len(calibration batches)`` and
        # runs all calibration data as one batch: that only fits readers that
        # yield a single sample per batch, otherwise ONNX Runtime rejects the
        # input for every layer (and ``SelectMaxMemLayer`` aborts on it)
        n_batches = len(calibration_data)
        rows = {
            k: sum(np.asarray(b[k]).shape[0] for b in calibration_data if k in b)
            for k in calibration_data[0]
            if np.asarray(calibration_data[0][k]).ndim > 0
        }
        if any(r != n_batches for r in rows.values()):
            if opt.select_max_mem_layer:
                raise RuntimeError(
                    "DynamicBatch: the concatenated calibration data does not "
                    f"fit the model's batch axis of {n_batches} (Quark: ONNX "
                    "Runtime InvalidArgument)"
                )
            return out, []
    conversion_errors: List[str] = []
    blocks = _find_blocks(float_model, quant_model, opt, conversion_errors)
    if conversion_errors and opt.select_max_mem_layer:
        raise NotImplementedError(
            "SelectMaxMemLayer converts every layer up front and Quark cannot "
            "convert: " + "; ".join(conversion_errors)
        )
    if not blocks:
        return out, []

    if perm_fn is None:
        perm_rng = np.random.default_rng(opt.seed)
        perm_fn = perm_rng.permutation
    mix_rng = np.random.default_rng([opt.seed, 1])

    f_cache: Dict[str, np.ndarray] = {}
    all_blocks = blocks
    selected = list(range(len(all_blocks)))
    if layers:  # a restored checkpoint wins over everything else
        selected = [int(i) for i in layers]
    elif opt.select_max_mem_layer:
        # Quark estimates every block from the first float calibration batch
        # and finetunes the most memory-hungry one only
        probe = _capture(
            float_model,
            sorted({n for b in blocks for n in (b.f_start, b.f_end)}),
            calibration_data[:1],
            providers,
            True,
        )
        mems = [
            _estimate_memory(b, probe[b.f_end].shape, probe[b.f_start].shape)
            for b in blocks
        ]
        selected = [int(np.argmax(mems))]
    # Quark walks its layer list in order and skips those not selected
    todo = [(i, all_blocks[i]) for i in sorted(set(selected)) if 0 <= i < len(blocks)]
    blocks = [b for _, b in todo]
    if opt.mem_opt_level == 0:
        f_cache = _capture(
            float_model,
            [n for b in blocks for n in (b.f_start, b.f_end)],
            calibration_data,
            providers,
            True,
        )
    q_parallel: Dict[str, np.ndarray] = {}
    if opt.parallel and opt.mem_opt_level != 2:  # (the DataLoader path is sequential)
        q_parallel = _capture(
            quant_model,
            [b.q_start for b in blocks],
            calibration_data,
            providers,
            False,
        )

    inits = {t.name: t for t in out.graph.initializer}
    f_out: List[List[np.ndarray]] = []
    l2 = 0.0
    if opt.selective_update:
        f_out = _model_outputs(
            float_model, calibration_data, providers, opt.output_index
        )
        l2 = _avg_l2(
            f_out,
            _model_outputs(out, calibration_data, providers, opt.output_index),
        )

    reports: List[LayerReport] = []
    for blk_index, blk in todo:
        if checkpoint is not None:
            checkpoint(blk_index, len(all_blocks), out)
        level2 = opt.mem_opt_level == 2
        fc = f_cache or _capture(
            float_model,
            [blk.f_start, blk.f_end],
            calibration_data,
            providers,
            True,
            split=level2,
        )
        xf, yf = fc[blk.f_start], fc[blk.f_end]
        if blk.q_start in q_parallel:
            xq = q_parallel[blk.q_start]
        else:
            xq = _capture(
                out, [blk.q_start], calibration_data, providers, False, split=level2
            )[blk.q_start]
        try:
            if level2:  # Quark's samples are whole calibration batches
                xq, xf, yf = (_loader_samples(a) for a in (xq, xf, yf))
        except _SkipLayer:
            continue
        if xq.shape != xf.shape or not _shape_ok(blk, xq):
            continue
        layer_trace: Optional[List[Tuple[int, float, float]]] = None
        if trace is not None:
            layer_trace = []
            trace.append(layer_trace)
        extras = None
        if block_hook is not None:
            extras = block_hook(blk_index, blk.name)
        if blk.phantom_bias is not None:
            drawn = extras.get("phantom_bias") if isinstance(extras, dict) else None
            if drawn is None:
                bound = blk.phantom_bias
                drawn = mix_rng.uniform(-bound, bound, size=blk.b_plain.shape)  # type: ignore[union-attr]
            blk.b_plain = np.asarray(drawn, dtype=np.float64).reshape(-1)
        try:
            res = _train_block(
                blk, xq, xf, yf, opt, perm_fn, mix_rng, layer_trace, rand_fn
            )
        except _SkipLayer:  # Quark logs the failed module and moves on
            if trace is not None:
                trace.pop()
            continue

        x_eval = xq if blk.in_q is None else blk.in_q.fq(xq)
        qw, qb = blk.qw, blk.qb
        cur_bias = blk.b_plain if qb is None else qb.dequant()
        before = _eval_error(blk, x_eval, yf, qw.dequant(), cur_bias)
        if qb is not None and res.bias_codes is not None:
            new_bias = qb.dequant(res.bias_codes)
        else:
            new_bias = cur_bias
        after = _eval_error(blk, x_eval, yf, qw.dequant(res.codes), new_bias)
        accepted = after <= before or not opt.guard
        if (
            opt.selective_update
            and opt.mem_opt_level != 2
            and res.err_final - res.err_rtn > 0
        ):
            # Quark's per-module SelectiveUpdate: when the module's error got
            # worse than its initial (hard-rounded float weight) one it drops
            # the new weight and bias (its DataLoader loop has no such check)
            accepted = False
        changed = float(np.mean(res.codes != qw.codes))
        undo: List[Tuple[str, onnx.TensorProto]] = []
        if accepted:
            writes = [(qw.name, res.codes)]
            if qb is not None and res.bias_codes is not None:
                writes.append((qb.name, res.bias_codes))
            for name, codes in writes:
                undo.append((name, _copy_tensor(inits[name])))
                dtype = numpy_helper.to_array(inits[name]).dtype
                inits[name].CopyFrom(numpy_helper.from_array(codes.astype(dtype), name))
        if opt.selective_update and undo:
            new_l2 = _avg_l2(
                f_out,
                _model_outputs(out, calibration_data, providers, opt.output_index),
            )
            if new_l2 < l2:
                l2 = new_l2
            else:
                for name, prev in undo:
                    inits[name].CopyFrom(prev)
                accepted, changed = False, 0.0
        reports.append(
            LayerReport(
                blk.name,
                blk.op_type,
                tuple(qw.codes.shape),
                before,
                after if accepted else before,
                accepted,
                changed,
            )
        )
    return out, reports


__all__ = [
    "FinetuneOptions",
    "TARGET_OPS",
    "finetune",
    "load_saved_layers",
    "save_checkpoint",
]
