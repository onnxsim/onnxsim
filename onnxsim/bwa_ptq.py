"""Binary Weight-Activation PTQ (Song, Wang, Wang, Yang, Zhang, 2025,
"Achieving binary weight and activation for LLMs using Post-Training
Quantization", ACL 2025 Findings, https://arxiv.org/abs/2504.05352, code at
https://github.com/JimmyCrave/LLM-PTQ-binarization). onnxsim ports the
*algorithm*, not any framework's code, per the same rationale as
:mod:`onnxsim.billm`/:mod:`onnxsim.gptq` (the paper's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

The paper's own configuration name is ``W(1+1)A(1x4)``. This module ports
its **weight** side only -- the ``W(1+1)`` piece -- and documents why the
``A(1x4)`` activation side is deliberately out of scope, see below.

**Weight side, W(1+1): Hessian-aware two-scale binary grouping.** Read
:mod:`onnxsim.billm` first -- both modules binarize an ordinary dense
float32 MatMul/Gemm weight straight from the float model (not a
rounding-refinement lever on an already-INT4-quantized model the way
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` are), and
both use a calibration-data Hessian. Where this module differs from BiLLM
is the *shape* of the extra bit each weight gets beyond its sign:

- :mod:`onnxsim.billm` spends its extra bit *structurally*, on whole
  Hessian-salient **columns**: a small, data-dependent fraction of columns
  get a second, *additive* binarization level (``W ~= B1 + B2``, a residual
  correction), while every other column stays a single flat sign*scale.
  Most of a BiLLM-quantized layer is exactly 1 bit/element; a few columns
  cost close to 2.
- This module (and the source paper) instead spends the extra bit
  *uniformly*, on every weight, as a **selector between two candidate
  scales** for its own group (a group being ``group_size`` contiguous
  elements of the reduction axis within one output channel row, matching
  the group convention :func:`onnxsim.quantize_weight_only_int4` already
  uses) rather than as an additive correction. Concretely, for each group,
  the paper's own "EM-based quantization scheme" alternates:

    1. (M-step) each of the group's two scale candidates is set to the
       Hessian-diagonal-weighted mean of ``|w|`` over whichever elements
       are currently assigned to it (the closed-form minimizer of a
       weighted-L2 binary fit, the same argument
       :func:`onnxsim.billm._binary` uses unweighted);
    2. (E-step) every element is reassigned to whichever of the two scales
       gives it lower Hessian-weighted squared error;

  repeated to convergence (a 2-component weighted Lloyd/k-means run on
  ``|w|``, seeded from a median split). Every weight ends up encoded as
  ``sign(w) * scale[group_select]`` -- exactly 1 sign bit + 1 group-select
  bit/element, no column-level structure and no additive residual term.
  Both this module and BiLLM are honest, different simplifications of
  their own papers' more elaborate schemes (see :mod:`onnxsim.billm`'s own
  docstring for its own documented simplification) -- neither supersedes
  the other; this module exists to make PTQ's uniform-two-scale family
  available alongside BiLLM's own salient-column-residual family.

  This is also distinct from :mod:`onnxsim.pb_llm` (keeps a salient
  *column fraction* at a much higher bit-width, e.g. INT8, and only
  binarizes the rest) -- this module never raises any weight above ~1 bit,
  uniformly, everywhere.

**Activation side, A(1x4): deliberately not ported.** The paper's own
activation scheme decomposes an INT4-quantized activation code into
``4 x INT1`` "bit-planes" (``q = sum_{k=0}^{3} 2**k * bit_k``) purely so a
bit-serial accelerator can compute the matmul via four cheap binary
popcount passes instead of one INT4 pass. Algebraically this decomposition
reconstructs *exactly* the same real number an ordinary calibrated INT4
quantizer already would -- it changes how a specific accelerator computes
the matmul, not the quantized value onnxsim's own QDQ graph would ever
represent. Since onnxsim doesn't emit bit-serial popcount kernels, this
module's activations are left for onnxsim's existing calibrated INT4/INT8
activation quantizers (e.g. :func:`onnxsim.quantize_static`) to handle,
exactly as :mod:`onnxsim.billm`/:mod:`onnxsim.pb_llm` also only handle the
weight side. Also not ported: the paper's own error-aware smoothing of the
weight/activation scaling factors (a joint calibration step tying the two
sides together) -- out of scope alongside the activation side it smooths.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.quip_sharp import _match_matmul_like


def _sign(w: np.ndarray) -> np.ndarray:
    """``sign(x) = 1 if x >= 0 else -1`` -- matches :func:`onnxsim.billm._sign`
    (not ``np.sign``, which maps exactly 0 to 0 rather than +1).
    """
    return np.where(w >= 0.0, 1.0, -1.0)


def _em_two_scale_binary(
    abs_w: np.ndarray, importance: np.ndarray, max_iters: int
) -> Tuple[np.ndarray, float, float]:
    """One group's Hessian-weighted 2-component binary EM (see this
    module's own docstring): alternates a closed-form weighted-mean scale
    update (M-step) with a nearest-scale reassignment (E-step) until the
    assignment stops changing. Returns ``(assign, scale0, scale1)`` --
    ``assign`` (int8, same shape as ``abs_w``) is 0/1 per element,
    ``scale0 <= scale1`` by construction (canonicalized so the encoding is
    deterministic regardless of which component the search happens to
    settle into first).
    """
    if abs_w.size == 0:
        return np.zeros(0, dtype=np.int8), 0.0, 0.0
    median = float(np.median(abs_w))
    assign = (abs_w >= median).astype(np.int64)
    scale0 = scale1 = 0.0
    for _ in range(max_iters):
        mask1 = assign == 1
        mask0 = ~mask1
        w0, w1 = importance[mask0], importance[mask1]
        scale0 = (
            float(np.sum(w0 * abs_w[mask0]) / max(np.sum(w0), 1e-12))
            if mask0.any()
            else 0.0
        )
        scale1 = (
            float(np.sum(w1 * abs_w[mask1]) / max(np.sum(w1), 1e-12))
            if mask1.any()
            else 0.0
        )
        err0 = importance * (abs_w - scale0) ** 2
        err1 = importance * (abs_w - scale1) ** 2
        new_assign = (err1 < err0).astype(np.int64)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
    if scale0 > scale1:
        assign = 1 - assign
        scale0, scale1 = scale1, scale0
    return assign.astype(np.int8), scale0, scale1


def _bwa_quantize_weight(
    w_nk: np.ndarray, hessian_diag_k: np.ndarray, group_size: int, max_iters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs the Hessian-weighted two-scale binary EM over every
    ``group_size``-column group of ``w_nk`` ([N, K], output channel
    first). Returns ``(sign_nk, group_select_nk, scale0_g, scale1_g)``:
    two ``[N, K]`` int8 arrays and two length-``num_groups`` float64 scale
    arrays (one pair of scales shared by every element of a group,
    regardless of output row -- only the sign and the group-select bit
    vary per element).
    """
    n, k = w_nk.shape
    num_groups = (k + group_size - 1) // group_size
    sign_nk = np.empty((n, k), dtype=np.int8)
    group_select_nk = np.empty((n, k), dtype=np.int8)
    scale0_g = np.zeros(num_groups, dtype=np.float64)
    scale1_g = np.zeros(num_groups, dtype=np.float64)

    for gi, start in enumerate(range(0, k, group_size)):
        end = min(start + group_size, k)
        w_blk = w_nk[:, start:end]  # [N, gs]
        sign_blk = _sign(w_blk)
        abs_flat = np.abs(w_blk).ravel()
        importance_flat = np.broadcast_to(
            hessian_diag_k[start:end], w_blk.shape
        ).ravel()
        assign_flat, s0, s1 = _em_two_scale_binary(abs_flat, importance_flat, max_iters)
        sign_nk[:, start:end] = sign_blk.astype(np.int8)
        group_select_nk[:, start:end] = assign_flat.reshape(w_blk.shape)
        scale0_g[gi] = s0
        scale1_g[gi] = s1

    return sign_nk, group_select_nk, scale0_g, scale1_g


def apply_bwa_ptq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    group_size: int = 128,
    max_em_iters: int = 10,
    skip_names: Optional[Iterable[str]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Binarizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight to exactly 1 sign bit + 1 group-select bit/element,
    using Hessian-weighted two-scale binary EM -- see this module's own
    docstring for the technique and its documented scope (weight side
    only; the source paper's activation-side bit-plane decomposition is
    numerically equivalent to ordinary INT4 activation quantization, see
    :func:`onnxsim.quantize_static`).

    Needs real calibration activations to compute each layer's Hessian
    diagonal (the per-input-channel importance weight the EM step uses),
    the same requirement :mod:`onnxsim.billm`/:mod:`onnxsim.gptq` have.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param group_size: contiguous reduction-axis elements sharing one pair
            of candidate scales -- matches
            :func:`onnxsim.quantize_weight_only_int4`'s own group
            convention
    :param max_em_iters: upper bound on EM iterations per group (the loop
            already stops early once the assignment stops changing)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Cast(Sign), Add(Scale0, Mul(Cast(GroupSelect), Sub(Scale1,
            Scale0))))`` feeding the original MatMul/Gemm node -- ordinary
            ONNX ops only, opset 11+. Layers with a non-constant, non-2-D,
            or non-float32 weight, or whose activation input isn't a plain
            2-D tensor, are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, weight_transposed))

    if not candidates:
        return out

    probe_names = sorted({c[1] for c in candidates})
    probe_model = _add_probe_outputs(model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(result[name], dtype=np.float64))

    for node, x_name, w_name, weight_transposed in candidates:
        acts = [a for a in activations[x_name] if a.ndim == 2]
        if not acts:
            continue  # not a plain 2-D activation; skip
        x = np.concatenate(acts, axis=0)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if x.shape[1] != k:
            continue  # activation's feature dim doesn't match K; skip

        hessian_diag_k = np.sum(x**2, axis=0)
        sign_nk, group_select_nk, scale0_g, scale1_g = _bwa_quantize_weight(
            w_nk, hessian_diag_k, group_size, max_em_iters
        )

        sign_orig = sign_nk if weight_transposed else sign_nk.T
        group_select_orig = group_select_nk if weight_transposed else group_select_nk.T
        assert sign_orig.shape == (dim0, dim1)

        scale0_full_k = np.repeat(scale0_g, group_size)[:k]
        scale1_full_k = np.repeat(scale1_g, group_size)[:k]

        # scale{0,1}_full_k are indexed along K (the reduction dim). When
        # weight_transposed (W is [N, K], K last), they broadcast against
        # W as-is; otherwise (W is [K, N], K first) they need a trailing
        # size-1 axis to broadcast against axis 0 instead of axis -1.
        if weight_transposed:
            scale0_orig = scale0_full_k
            scale1_orig = scale1_full_k
        else:
            scale0_orig = scale0_full_k[:, np.newaxis]
            scale1_orig = scale1_full_k[:, np.newaxis]

        prefix = f"{w_name}_bwa"
        sign_name = _unique_name(f"{prefix}_sign", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(sign_orig.astype(np.int8), name=sign_name)
        )
        group_name = _unique_name(f"{prefix}_group_select", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                group_select_orig.astype(np.int8), name=group_name
            )
        )
        scale0_name = _unique_name(f"{prefix}_scale0", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(scale0_orig).astype(np.float32), name=scale0_name
            )
        )
        scale1_name = _unique_name(f"{prefix}_scale1", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(scale1_orig).astype(np.float32), name=scale1_name
            )
        )

        sign_f_out = _unique_name(f"{prefix}_sign_f", taken_names)
        sign_cast_node = onnx.helper.make_node(
            "Cast", [sign_name], [sign_f_out], to=onnx.TensorProto.FLOAT
        )
        group_f_out = _unique_name(f"{prefix}_group_f", taken_names)
        group_cast_node = onnx.helper.make_node(
            "Cast", [group_name], [group_f_out], to=onnx.TensorProto.FLOAT
        )
        diff_out = _unique_name(f"{prefix}_scale_diff", taken_names)
        sub_node = onnx.helper.make_node("Sub", [scale1_name, scale0_name], [diff_out])
        sel_out = _unique_name(f"{prefix}_scale_sel", taken_names)
        mul_sel_node = onnx.helper.make_node("Mul", [group_f_out, diff_out], [sel_out])
        scale_eff_out = _unique_name(f"{prefix}_scale_eff", taken_names)
        add_node = onnx.helper.make_node("Add", [scale0_name, sel_out], [scale_eff_out])
        dq_out = _unique_name(f"{prefix}_dq", taken_names)
        mul_final_node = onnx.helper.make_node(
            "Mul",
            [sign_f_out, scale_eff_out],
            [dq_out],
            name=_unique_name(f"{prefix}_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (
            sign_cast_node,
            group_cast_node,
            sub_node,
            mul_sel_node,
            add_node,
            mul_final_node,
        ):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
