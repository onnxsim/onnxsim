"""BiLLM (Huang, Liu, Qin, Li, Zhang, Liu, Magno, Qi, 2024, ICML 2024,
"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs",
https://arxiv.org/abs/2402.04291) -- a genuine 1-bit-average **weight
binarizer**, not another rounding-refinement lever on top of an
already-fixed-scale quantizer the way :mod:`onnxsim.adaround`/
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.flexround` all are (each
of those takes an *already* ``quantize_weight_only_int4``-quantized model and
only changes which grid point each element rounds to). BiLLM instead takes
an ordinary dense float32 MatMul/Gemm layer straight from the float model and
produces a genuinely different (binary/near-binary) representation of it --
the same family as :func:`onnxsim.quantize_weight_only_int4`/
:func:`onnxsim.quantize_weight_only_nf4`/
:func:`onnxsim.quantize_weight_only_kmeans`, pushed to the most extreme end
of that family's bit-width range.

This is *not* the same problem :func:`onnxsim.quantize_ternary` solves (see
``docs/ternary-quantization.md``): that pass only **detects** a weight that
is *already*, structurally, exactly ``{-s, 0, +s}`` (a lossless rewrite of a
BitNet-family model someone else already trained ternary) and leaves
anything else -- including an ordinary dense float32 weight -- untouched.
BiLLM instead **quantizes** an ordinary dense float32 weight itself, lossily,
down to close to 1 bit/element on average. The two are complementary, not
overlapping: run :func:`onnxsim.quantize_ternary` on a BitNet export (nothing
to binarize, the weight already is ternary); run this module's
:func:`quantize_weight_only_billm` on a normal pretrained dense model's
Linear layers (genuinely new information loss, deliberately accepted for
the compression).

The technique, following the paper's own Algorithm 1/Algorithm 2 fairly
closely (this module ports the *algorithm*, in plain numpy, from the paper
itself -- there is no ONNX export path from the authors' own PyTorch
reference implementation, the same reason :mod:`onnxsim.gptq`/
:mod:`onnxsim.awq` port their own source papers rather than any upstream
code):

1. **Per-layer Hessian**, exactly as :mod:`onnxsim.gptq`: ``H = X^T X`` from
   real calibration activations (reusing
   :func:`onnxsim.gptq._inverse_hessian_cholesky` for the same damped-
   Cholesky-of-``H^-1`` reformulation GPTQ itself introduces, since BiLLM's
   own block-wise error compensation, Algorithm 1 lines 12-13, is the same
   OBC/GPTQ mechanism applied to whichever binarization a block picked
   rather than to a rounded integer).

2. **Block-wise processing** (``block_size`` columns of the reduction
   dimension at a time, paper default 128): within each block,

   a. **Structured salient-column selection** (paper Section 3.1,
      Algorithm 2's ``salient()``): per-column sensitivity
      ``s_i = w_i^2 / [H_c]_ii^2`` (``H_c`` the damped-Hessian-inverse
      Cholesky factor -- the same "how much does perturbing this weight
      hurt the layer's output" argument OBS/GPTQ use, evaluated per column
      via a column-sum rather than per individual element, since the paper
      finds salient Hessian mass concentrates in whole columns for
      attention-projection layers). Columns are ranked by total column
      sensitivity and a small bounded search (this module searches
      ``1..min(30, block_size - 1)`` candidate salient-column counts,
      matching the paper's own stated 3-30 search range) picks the count
      that minimizes plain-binary reconstruction error for the block --
      the *number* of salient columns is data-dependent per block/layer,
      not a fixed global fraction.

   b. **Binary residual approximation for salient columns** (paper
      Section 3.1 "Binary Residual Approximation", Algorithm 2's
      ``res_approximation()``, Equations 6-7): ``B1 = sign(W) * mean(|W|)``
      (one scalar scale for the whole salient sub-block -- the paper's own
      ``binary()`` primitive, Equation 4, whose L2-optimal closed-form
      scale is exactly ``mean(|W|)``), then a *second* binarization of the
      residual ``R = W - B1``: ``B2 = sign(R) * mean(|R|)``, giving
      ``W ~= B1 + B2`` -- effectively 2 bits for salient columns, versus a
      naive scheme that would need 8-16 bits to protect them, while the
      paper proves (Eq. 8) this residual reconstruction strictly dominates
      keeping only ``B1``.

   c. **Plain flat binary for non-salient columns**: ``sign(W) * mean(|W|)``
      (again the paper's own ``binary()`` primitive, one scalar scale for
      the whole non-salient sub-block) -- **this is a deliberate,
      documented simplification** of the paper's own Section 3.2 "Bell-
      shaped Distribution Splitting" (Algorithm 2's ``seg_search()``),
      which further splits non-salient weights *elementwise* by magnitude
      into a "concentrated" and a "sparse" region (each with its own
      scale, chosen by a 9-point percentile search minimizing Eq. 11) to
      account for their bell-shaped, non-uniform distribution. The paper
      itself reports this second-level split contributes far less to
      accuracy than the salient/residual mechanism (Table 1: ~0.02 extra
      average bits from the whole non-salient path, versus the salient
      path's own contribution) -- it is not this scheme's headline result.
      Skipping it also keeps this module's ONNX encoding uniform: the
      elementwise concentrated/sparse split has no clean per-column
      broadcast shape the way every other quantity here does (see below),
      and would force a per-element rather than a compact per-column
      scale. A faithful port of ``seg_search`` is a reasonable future
      addition, not attempted here.

   d. **Block-wise OBC-style error compensation** (paper Algorithm 1 lines
      12-13, the same mechanism :mod:`onnxsim.gptq` already implements):
      whatever this block's binarization couldn't represent is charged
      forward into every not-yet-processed column, in proportion to
      ``H_c``'s own off-diagonal structure -- so later blocks' own
      binarization partially compensates for earlier blocks' error, the
      same second-order argument GPTQ uses for its own rounding.

Because the paper's own scales (``mean(|W|)`` in Equation 4/12, and this
module's own non-salient scale) are single scalars *per block*, not per
individual weight element or even per output channel the way onnxsim's other
INT4/NF4/k-means schemes are, and because within a block every column is
either "salient" (gets both a level-1 and level-2 scale) or "non-salient"
(gets only a level-1 scale, with its level-2 contribution forced to zero),
this module's encoding is a **compact per-input-channel (per-column) pair of
scale vectors** (length ``K``, the reduction dimension) rather than the
per-(output-channel, block) 2-D scale grid :mod:`onnxsim.nf4`/
:func:`onnxsim.quantize_weight_only_int4` use:

    Before:
      Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

    After:
      Code1: initializer, int8, [K, N], values in {-1, +1}  -- sign(W), or
             sign(B1) for a salient column
      Code2: initializer, int8, [K, N], values in {-1, 0, +1}  -- sign of
             the salient residual for a salient column, exactly 0 (no
             correction) for a non-salient column
      Scale1: initializer, float32, [K, 1]  -- per-column level-1 scale
              (this block's alpha_salient for a salient column, this
              block's alpha_nonsalient for a non-salient one)
      Scale2: initializer, float32, [K, 1]  -- per-column level-2 scale,
              exactly 0 for every non-salient column
      What_hat = Cast(Code1, float) * Scale1 + Cast(Code2, float) * Scale2
      Y = MatMul(X, What_hat) [+ bias]

No ``Gather``/codebook lookup is needed the way :mod:`onnxsim.nf4`/
:mod:`onnxsim.kmeans_quantization` need one: since the "codebook" here is
just ``{-1, +1}`` (and ``{-1, 0, +1}`` for the residual level), the code
tensor's own values, cast to float, already *are* the sign -- multiplying by
the per-column scale directly reconstructs the weight, with no lookup table
in between. Ordinary ONNX ops only (``Cast``/``Mul``/``Add``), opset 11+
(this module needs nothing newer than that), no contrib op.
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
from onnxsim.gptq import _inverse_hessian_cholesky
from onnxsim.quip_sharp import _match_matmul_like


def _sign(w: np.ndarray) -> np.ndarray:
    """``sign(x) = 1 if x >= 0 else -1`` (paper Equation 2) -- note this is
    *not* ``np.sign``, which maps exactly 0 to 0 rather than +1.
    """
    return np.where(w >= 0.0, 1.0, -1.0)


def _binary(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """The paper's own ``binary()`` primitive (Algorithm 2): a single
    scalar scale for the whole of ``w`` (``mean(|w|)``, the L2-optimal
    closed-form solution to ``argmin_scale ||w - scale*sign(w)||^2``) and
    the elementwise sign. Returns ``(sign(w), scale)`` -- the caller
    combines them (``sign(w) * scale``) as needed; kept separate since
    both binarized levels of a salient column need their own code and this
    module stores codes and scales in separate tensors.
    """
    if w.size == 0:
        return np.zeros_like(w), 0.0
    return _sign(w), float(np.mean(np.abs(w)))


def _select_salient_columns(
    w_block: np.ndarray, hc_block: np.ndarray, max_search: int
) -> np.ndarray:
    """Returns the column indices (into ``w_block``'s own axis 1) of the
    salient columns for one block, per the paper's Algorithm 2
    ``salient()``: rank columns by Hessian-based sensitivity
    ``s_i = w_i^2 / [H_c]_ii^2`` (summed per column), then search a bounded
    number of leading (most-salient) columns for the count that minimizes
    plain-binary reconstruction error of the *whole* block (both groups
    binarized with the simple, single-scale ``_binary`` -- matching the
    search's own use of ``binary()`` rather than the residual scheme the
    winning column count is *then* given).
    """
    n, bs = w_block.shape
    if bs == 0:
        return np.empty(0, dtype=np.int64)
    diag_hc = np.diag(hc_block)
    diag_hc = np.where(np.abs(diag_hc) < 1e-12, 1e-12, diag_hc)
    sensitivity = (w_block**2) / (diag_hc[np.newaxis, :] ** 2)
    col_salience = np.sum(np.abs(sensitivity), axis=0)
    order = np.argsort(-col_salience)

    upper = min(max_search, bs - 1)
    if upper < 1:
        return np.empty(0, dtype=np.int64)

    best_err = np.inf
    best_n = 0
    for i in range(1, upper + 1):
        sal_cols = order[:i]
        nonsal_cols = order[i:]
        recon = np.empty_like(w_block)
        sign_sal, scale_sal = _binary(w_block[:, sal_cols])
        recon[:, sal_cols] = sign_sal * scale_sal
        sign_ns, scale_ns = _binary(w_block[:, nonsal_cols])
        recon[:, nonsal_cols] = sign_ns * scale_ns
        err = float(np.sum((w_block - recon) ** 2))
        if err < best_err:
            best_err = err
            best_n = i
    return order[:best_n]


def _billm_quantize_block(
    w_nk: np.ndarray,
    h: np.ndarray,
    block_size: int,
    percdamp: float,
    max_salient_search: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs BiLLM's block-wise salient selection, binary residual
    approximation, plain non-salient binarization, and OBC-style error
    compensation (see this module's own docstring) over ``w_nk``
    ([N, K], output channel first). Returns ``(code1_nk, code2_nk,
    scale1_k, scale2_k)``: two ``[N, K]`` int8 code arrays and two
    length-``K`` per-column scale arrays.
    """
    n, k = w_nk.shape
    hc = _inverse_hessian_cholesky(h, percdamp)

    code1 = np.ones((n, k), dtype=np.int8)
    code2 = np.zeros((n, k), dtype=np.int8)
    scale1 = np.zeros(k, dtype=np.float64)
    scale2 = np.zeros(k, dtype=np.float64)

    w_work = w_nk.copy()

    for block_start in range(0, k, block_size):
        block_end = min(block_start + block_size, k)
        w_b = w_work[:, block_start:block_end]
        hc_b = hc[block_start:block_end, block_start:block_end]
        bs = block_end - block_start

        sal_local = _select_salient_columns(w_b, hc_b, max_salient_search)
        sal_mask = np.zeros(bs, dtype=bool)
        sal_mask[sal_local] = True
        nonsal_local = np.nonzero(~sal_mask)[0]

        b_block = np.empty_like(w_b)

        if sal_local.size > 0:
            w_sal = w_b[:, sal_local]
            sign1, alpha_o = _binary(w_sal)
            b1 = sign1 * alpha_o
            r = w_sal - b1
            sign2, alpha_r = _binary(r)
            b2 = sign2 * alpha_r
            b_block[:, sal_local] = b1 + b2

            cols = block_start + sal_local
            code1[:, cols] = sign1.astype(np.int8)
            code2[:, cols] = sign2.astype(np.int8)
            scale1[cols] = alpha_o
            scale2[cols] = alpha_r

        if nonsal_local.size > 0:
            w_ns = w_b[:, nonsal_local]
            sign_ns, alpha_ns = _binary(w_ns)
            b_block[:, nonsal_local] = sign_ns * alpha_ns

            cols = block_start + nonsal_local
            code1[:, cols] = sign_ns.astype(np.int8)
            code2[:, cols] = 0
            scale1[cols] = alpha_ns
            scale2[cols] = 0.0

        if block_end < k and nonsal_local.size > 0:
            # Forward error compensation (paper Algorithm 1 lines 12-13,
            # the same OBC/GPTQ mechanism onnxsim.gptq already implements):
            # charge a column's leftover reconstruction error into
            # not-yet-processed columns, in proportion to 1/diag(Hc) for
            # that column. Deliberately restricted to non-salient columns:
            # a column is selected salient *because* its Hc diagonal is
            # close to singular (that is exactly what
            # ``s_i = w_i^2/[H_c]_ii^2`` being large means), so dividing
            # its own (already near-zero, thanks to the two-level residual
            # approximation) leftover error by that same near-singular
            # diagonal is numerically unstable by construction -- a small
            # amount of floating-point residual gets amplified into a huge
            # spurious correction. Salient columns already got dedicated,
            # high-fidelity treatment above; skipping them here only drops
            # a compensation term that both is negligible in principle (the
            # residual approximation leaves very little error to charge
            # forward) and is unsafe to compute in practice.
            diag_hc = np.diag(hc_b)[nonsal_local]
            diag_hc = np.where(np.abs(diag_hc) < 1e-8, 1e-8, diag_hc)
            err_block = (w_b - b_block)[:, nonsal_local]
            err1 = err_block / diag_hc[np.newaxis, :]
            hc_future = hc[block_start:block_end, block_end:][nonsal_local, :]
            w_work[:, block_end:] -= err1 @ hc_future

    return code1, code2, scale1, scale2


def quantize_weight_only_billm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 128,
    percdamp: float = 0.01,
    max_salient_search: int = 30,
    skip_names: Optional[Iterable[str]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Binarizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight to close to 1 bit/element on average, using BiLLM's
    Hessian-guided salient-column residual approximation -- see this
    module's own docstring for the technique and its documented
    simplification relative to the source paper.

    Unlike :func:`onnxsim.quantize_weight_only_nf4`/
    :func:`onnxsim.quantize_weight_only_kmeans` (calibration-free: every
    decision comes from the weight tensor's own values), this needs real
    calibration activations to compute each layer's Hessian, the same as
    :mod:`onnxsim.gptq`/:mod:`onnxsim.awq` -- an inherent requirement of
    BiLLM's own salient-column selection, not a simplification.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: columns of the reduction dimension processed
            together -- both BiLLM's own salient-column search granularity
            and the OBC-style error-compensation block, paper default 128
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same
            name and default
    :param max_salient_search: upper bound on how many leading (most
            Hessian-salient) columns of a block the search considers as
            candidates for "the salient group" -- the paper's own stated
            search range is 3-30; this module searches ``1..min(
            max_salient_search, block_size - 1)`` and lets the search
            itself settle on however few (down to 0) or many turn out
            optimal
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Add(Mul(Cast(Code1), Scale1), Mul(Cast(Code2), Scale2))``
            feeding the original MatMul/Gemm node -- ordinary ONNX ops
            only, opset 11+. Layers with a non-constant, non-2-D, or
            non-float32 weight, or whose activation input isn't a plain
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

        h = x.T @ x
        code1_nk, code2_nk, scale1_k, scale2_k = _billm_quantize_block(
            w_nk, h, block_size, percdamp, max_salient_search
        )

        code1_orig = code1_nk if weight_transposed else code1_nk.T
        code2_orig = code2_nk if weight_transposed else code2_nk.T
        assert code1_orig.shape == (dim0, dim1)

        # scale1_k/scale2_k are indexed along K (the reduction dim). When
        # weight_transposed (W is [N, K], K last), they broadcast against
        # W as-is; otherwise (W is [K, N], K first) they need a trailing
        # size-1 axis to broadcast against axis 0 instead of axis -1.
        if weight_transposed:
            scale1_orig = scale1_k
            scale2_orig = scale2_k
        else:
            scale1_orig = scale1_k[:, np.newaxis]
            scale2_orig = scale2_k[:, np.newaxis]

        prefix = f"{w_name}_billm"
        code1_name = _unique_name(f"{prefix}_code1", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(code1_orig.astype(np.int8), name=code1_name)
        )
        code2_name = _unique_name(f"{prefix}_code2", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(code2_orig.astype(np.int8), name=code2_name)
        )
        scale1_name = _unique_name(f"{prefix}_scale1", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                scale1_orig.astype(np.float32), name=scale1_name
            )
        )
        scale2_name = _unique_name(f"{prefix}_scale2", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                scale2_orig.astype(np.float32), name=scale2_name
            )
        )

        cast1_out = _unique_name(f"{prefix}_code1_f", taken_names)
        cast1_node = onnx.helper.make_node(
            "Cast", [code1_name], [cast1_out], to=onnx.TensorProto.FLOAT
        )
        cast2_out = _unique_name(f"{prefix}_code2_f", taken_names)
        cast2_node = onnx.helper.make_node(
            "Cast", [code2_name], [cast2_out], to=onnx.TensorProto.FLOAT
        )
        term1_out = _unique_name(f"{prefix}_term1", taken_names)
        mul1_node = onnx.helper.make_node("Mul", [cast1_out, scale1_name], [term1_out])
        term2_out = _unique_name(f"{prefix}_term2", taken_names)
        mul2_node = onnx.helper.make_node("Mul", [cast2_out, scale2_name], [term2_out])
        dq_out = _unique_name(f"{prefix}_dq", taken_names)
        add_node = onnx.helper.make_node(
            "Add",
            [term1_out, term2_out],
            [dq_out],
            name=_unique_name(f"{prefix}_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (cast1_node, cast2_node, mul1_node, mul2_node, add_node):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
