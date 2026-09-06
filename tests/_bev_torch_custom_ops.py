"""Shared ``torch.autograd.Function`` definitions for the BEV torch E2E tests.

This module is deliberately named without a ``test_`` prefix so pytest's
default ``test_*.py`` collection glob (see ``tests/conftest.py``'s own
docstring on the ``tests/`` import-path convention; no ``python_files``
override is set in ``pyproject.toml``, so the default applies) does not try
to collect it as a test module -- it is a plain importable helper, used by
``tests/test_bev_models_torch_e2e.py``.

Each class below pairs:

  - a real, numerically-correct ``forward()`` -- a port of the exact
    reference algorithm already validated (independently, from scratch) in
    this repo's per-op rewrite-pass test files
    (``tests/test_msdeformattn_to_gridsample.py``,
    ``tests/test_deform_conv_to_gather.py``, ``tests/test_trt_batched_nms.py``),
    reimplemented here with plain PyTorch tensor ops (or, for
    ``TRTBatchedNMSFunction``, a direct NumPy port of that same test file's
    ``reference_trt_batched_nms`` -- fine since these are inference-only
    tests with no gradient needed, only correct forward values);
  - a ``symbolic()`` emitting the exact mmdeploy/mmcv custom op each of
    ``onnxsim/passes/rewrite_msdeformattn_to_gridsample.h``,
    ``rewrite_deform_conv_to_gather.h``, and ``rewrite_trt_batched_nms.h``
    matches -- domain ``"mmdeploy"``, exact op_type, input order/count, and
    attribute names, read from each pass's own ``patternMatchPredicate``/
    ``TryMatch`` code (not just its header comment) to get the contract
    exactly right.

Exporting through ``torch.onnx.export(..., dynamo=False)`` (the legacy
TorchScript-tracing exporter -- the new dynamo/torch.export-based default in
torch>=2 does not support this ``symbolic()`` mechanism) calls ``forward()``
to trace concrete values through the rest of the graph, and substitutes the
node ``symbolic()`` describes in the exported graph in place of whatever
``forward()`` internally did. So ``forward()`` doubles as both "the ground
truth eager output" and "what the traced downstream ops see" -- the same
function that defines correctness also defines what's exported.
"""

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. MMCVMultiScaleDeformableAttention (BEVFormer/Deformable-DETR deformable
#    attention) -- see onnxsim/passes/rewrite_msdeformattn_to_gridsample.h.
#
# Op spec matched (read off ``ExtractDims``/``patternMatchPredicate`` there):
#   5 inputs, in this fixed order: value, spatial_shapes, level_start_index,
#   sampling_locations, attention_weights. 1 output. domain "" or "mmdeploy".
#   im2col_step is cosmetic and never read by the pass -- included here only
#   because real mmdeploy exports always carry it.
# --------------------------------------------------------------------------- #


class MSDeformAttnFunction(torch.autograd.Function):
    """Forward is a direct torch-tensor-op port of mmcv's own pure-PyTorch
    fallback, ``multi_scale_deformable_attn_pytorch`` -- the same reference
    algorithm ``rewrite_msdeformattn_to_gridsample.h``'s derivation comment
    and ``tests/test_msdeformattn_to_gridsample.py``'s NumPy reference are
    built from, just using ``torch``'s own ``F.grid_sample`` instead of a
    hand-rolled bilinear sampler or a second NumPy port."""

    @staticmethod
    def forward(
        ctx,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
    ):
        bs, _, M, D = value.shape
        _, nq, _, L, P, _ = sampling_locations.shape

        spatial_shapes_list = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        value_list = value.split([h * w for h, w in spatial_shapes_list], dim=1)
        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []
        for level, (H_, W_) in enumerate(spatial_shapes_list):
            # (bs, H_*W_, M, D) -> (bs, H_*W_, M*D) -> (bs, M*D, H_*W_)
            # -> (bs*M, D, H_, W_).
            value_l_ = (
                value_list[level].flatten(2).transpose(1, 2).reshape(bs * M, D, H_, W_)
            )
            # (bs, nq, M, P, 2) -> (bs, M, nq, P, 2) -> (bs*M, nq, P, 2).
            sampling_grid_l_ = (
                sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)  # (bs*M, D, nq, P)

        attn = attention_weights.transpose(1, 2).reshape(bs * M, 1, nq, L * P)
        output = (
            (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn)
            .sum(-1)
            .view(bs, M * D, nq)
        )
        return output.transpose(1, 2).contiguous()

    @staticmethod
    def symbolic(
        g,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
    ):
        return g.op(
            "mmdeploy::MMCVMultiScaleDeformableAttention",
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step_i=64,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Inference-only: these tests only ever call this Function under
        # torch.no_grad() (or during export tracing, which never invokes
        # backward), so no gradient implementation is needed.
        raise NotImplementedError(
            "MSDeformAttnFunction is inference-only; no backward pass"
        )


# --------------------------------------------------------------------------- #
# 2. MMCVDeformConv2d (DCNv1) / MMCVModulatedDeformConv2d (DCNv2) -- see
#    onnxsim/passes/rewrite_deform_conv_to_gather.h.
#
# Op spec matched (read off ``TryMatch`` there):
#   DCNv1 inputs: (input, offset, weight[, bias]).
#   DCNv2 inputs: (input, offset, mask, weight[, bias]).
#   attributes: stride, padding, dilation (2 ints each), groups,
#   deform_groups (or the synonym "deformable_groups"). domain "" or
#   "mmdeploy". **Scoped to groups == 1** -- arbitrary deform_groups is fine.
# --------------------------------------------------------------------------- #


def _bilinear_sample_zero_pad(img, y, x):
    """4-corner bilinear sample of ``img`` (H,W) at float coords ``y``/``x``
    (broadcastable tensors), zero-padded outside ``[0,H-1]``/``[0,W-1]`` --
    the exact algorithm ``tests/test_deform_conv_to_gather.py``'s
    ``_bilinear_sample_zero_pad`` validates independently, ported here to
    plain ``torch`` tensor ops instead of NumPy."""
    H, W = img.shape[-2], img.shape[-1]
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1

    def get(yy, xx):
        valid = (yy >= 0) & (yy <= H - 1) & (xx >= 0) & (xx <= W - 1)
        yyc = yy.clamp(0, H - 1).long()
        xxc = xx.clamp(0, W - 1).long()
        val = img[yyc, xxc]
        return torch.where(valid, val, torch.zeros_like(val))

    return (
        get(y0, x0) * wx0 * wy0
        + get(y0, x1) * wx1 * wy0
        + get(y1, x0) * wx0 * wy1
        + get(y1, x1) * wx1 * wy1
    )


def _deform_conv2d_core(
    x, offset, mask, weight, bias, stride, padding, dilation, deform_groups
):
    """Standard (modulated, if ``mask`` is not None) deformable convolution,
    ``groups == 1`` only (matching this pass's own scope limit) -- a direct
    torch-tensor port of ``tests/test_deform_conv_to_gather.py``'s
    ``_deform_conv2d_reference`` NumPy implementation of the same
    well-established algorithm (Zhu et al.'s DCNv2; DCNv1 is the same
    algorithm with an implicit all-ones mask)."""
    N, Cin, H, W = x.shape
    Cout, _Cin_g, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Hout, Wout = offset.shape[2], offset.shape[3]
    Cin_dg = Cin // deform_groups

    ho = torch.arange(Hout, dtype=x.dtype, device=x.device).reshape(-1, 1)
    wo = torch.arange(Wout, dtype=x.dtype, device=x.device).reshape(1, -1)

    out = x.new_zeros((N, Cout, Hout, Wout))
    for n in range(N):
        acc = x.new_zeros((Cout, Hout, Wout))
        for cin in range(Cin):
            dg = cin // Cin_dg
            for i in range(kh):
                for j in range(kw):
                    k = i * kw + j
                    dy = offset[n, dg * 2 * kh * kw + 2 * k]
                    dx = offset[n, dg * 2 * kh * kw + 2 * k + 1]
                    y = ho * sh - ph + i * dh + dy
                    xx = wo * sw - pw + j * dw + dx
                    sampled = _bilinear_sample_zero_pad(x[n, cin], y, xx)
                    if mask is not None:
                        sampled = sampled * mask[n, dg * kh * kw + k]
                    acc = acc + weight[:, cin, i, j].reshape(
                        -1, 1, 1
                    ) * sampled.unsqueeze(0)
        out[n] = acc
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


class DeformConvFunction(torch.autograd.Function):
    """DCNv1 (unmodulated, no mask) -- emits ``MMCVDeformConv2d``."""

    @staticmethod
    def forward(
        ctx, x, offset, weight, bias, stride, padding, dilation, groups, deform_groups
    ):
        return _deform_conv2d_core(
            x,
            offset,
            None,
            weight,
            bias,
            stride,
            padding,
            dilation,
            deform_groups,
        )

    @staticmethod
    def symbolic(
        g, x, offset, weight, bias, stride, padding, dilation, groups, deform_groups
    ):
        args = [x, offset, weight]
        if bias is not None:
            args.append(bias)
        return g.op(
            "mmdeploy::MMCVDeformConv2d",
            *args,
            stride_i=list(stride),
            padding_i=list(padding),
            dilation_i=list(dilation),
            groups_i=groups,
            deform_groups_i=deform_groups,
        )

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "DeformConvFunction is inference-only; no backward pass"
        )


class ModulatedDeformConvFunction(torch.autograd.Function):
    """DCNv2 (modulated, with mask) -- emits ``MMCVModulatedDeformConv2d``."""

    @staticmethod
    def forward(
        ctx,
        x,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    ):
        return _deform_conv2d_core(
            x,
            offset,
            mask,
            weight,
            bias,
            stride,
            padding,
            dilation,
            deform_groups,
        )

    @staticmethod
    def symbolic(
        g,
        x,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    ):
        args = [x, offset, mask, weight]
        if bias is not None:
            args.append(bias)
        return g.op(
            "mmdeploy::MMCVModulatedDeformConv2d",
            *args,
            stride_i=list(stride),
            padding_i=list(padding),
            dilation_i=list(dilation),
            groups_i=groups,
            deform_groups_i=deform_groups,
        )

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "ModulatedDeformConvFunction is inference-only; no backward pass"
        )


# --------------------------------------------------------------------------- #
# 3. TRTBatchedNMS -- see onnxsim/passes/rewrite_trt_batched_nms.h.
#
# Op spec matched (read off ``patternMatchPredicate``/``runTransform``
# there): 2 inputs (boxes, scores), 4 outputs (num_detections, nmsed_boxes,
# nmsed_scores, nmsed_classes). Attributes: background_label_id, num_classes,
# topK, keepTopK, scoreThreshold, iouThreshold. domain "" or "mmdeploy".
# **Scoped to class-agnostic boxes** (boxes' 3rd dim == 1, statically known)
# **and statically-known batch size N.**
# --------------------------------------------------------------------------- #


def _iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _greedy_nms(boxes_c, scores_c, iou_threshold, top_k):
    """Standard greedy NMS: sort by score descending, keep a box, suppress
    any remaining box with IoU > iou_threshold against it. Returns indices
    into boxes_c/scores_c, at most top_k of them. Ported verbatim from
    ``tests/test_trt_batched_nms.py``'s ``_greedy_nms``."""
    order = np.argsort(-scores_c, kind="stable")
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for pos, i in enumerate(order):
        if suppressed[pos]:
            continue
        keep.append(i)
        if len(keep) >= top_k:
            break
        for pos2 in range(pos + 1, len(order)):
            if suppressed[pos2]:
                continue
            j = order[pos2]
            if _iou(boxes_c[i], boxes_c[j]) > iou_threshold:
                suppressed[pos2] = True
    return keep


def reference_trt_batched_nms(
    boxes,
    scores,
    background_label_id,
    top_k,
    keep_top_k,
    score_threshold,
    iou_threshold,
):
    """boxes: (N, num_boxes, 4) float32 (already squeezed -- class-agnostic).
    scores: (N, num_boxes, num_classes) float32. Implements the "per-class
    greedy NMS, then global per-batch top-K merge" algorithm this op is
    documented to perform. Ported verbatim from
    ``tests/test_trt_batched_nms.py``'s ``reference_trt_batched_nms``."""
    n, num_boxes, num_classes = scores.shape
    out_boxes = np.zeros((n, keep_top_k, 4), dtype=np.float32)
    out_scores = np.zeros((n, keep_top_k), dtype=np.float32)
    out_classes = np.full((n, keep_top_k), -1.0, dtype=np.float32)
    out_numdet = np.zeros((n, 1), dtype=np.int64)

    for b in range(n):
        pooled = []  # (score, class, box_index)
        for c in range(num_classes):
            if c == background_label_id:
                continue
            sc = scores[b, :, c]
            valid_idx = np.where(sc > score_threshold)[0]
            if len(valid_idx) == 0:
                continue
            kept_local = _greedy_nms(
                boxes[b, valid_idx], sc[valid_idx], iou_threshold, top_k
            )
            for li in kept_local:
                orig_idx = valid_idx[li]
                pooled.append((float(sc[orig_idx]), c, int(orig_idx)))
        pooled.sort(key=lambda t: -t[0])
        pooled = pooled[:keep_top_k]
        out_numdet[b, 0] = len(pooled)
        for k, (sc, c, bidx) in enumerate(pooled):
            out_boxes[b, k] = boxes[b, bidx]
            out_scores[b, k] = sc
            out_classes[b, k] = float(c)
    return out_numdet, out_boxes, out_scores, out_classes


def assert_same_detections(actual, expected, atol=1e-4):
    """Compares kept detections per batch item as an unordered set (sorted
    by score), not by raw index position -- the same technique
    ``tests/test_trt_batched_nms.py``'s ``_assert_same_detections`` uses,
    since the underlying real ``TRTBatchedNMS`` TensorRT plugin's exact
    tie-breaking order is not fully specifiable (see that file's module
    docstring); ``actual``/``expected`` are each
    ``(num_detections, boxes, scores, classes)`` tuples."""
    a_numdet, a_boxes, a_scores, a_classes = actual
    e_numdet, e_boxes, e_scores, e_classes = expected
    n = e_numdet.shape[0]
    for b in range(n):
        assert int(a_numdet[b, 0]) == int(e_numdet[b, 0]), (
            f"batch {b}: num_detections mismatch: "
            f"{int(a_numdet[b, 0])} vs {int(e_numdet[b, 0])}"
        )
        cnt = int(e_numdet[b, 0])

        def _rows(scores_row, boxes_row, classes_row, cnt=cnt):
            rows = [
                (float(scores_row[k]), float(classes_row[k]), tuple(boxes_row[k]))
                for k in range(cnt)
            ]
            rows.sort(key=lambda t: (-t[0], t[1]))
            return rows

        a_rows = _rows(a_scores[b], a_boxes[b], a_classes[b])
        e_rows = _rows(e_scores[b], e_boxes[b], e_classes[b])
        assert len(a_rows) == len(e_rows)
        for (a_s, a_c, a_bx), (e_s, e_c, e_bx) in zip(a_rows, e_rows):
            assert a_c == e_c, f"batch {b}: class mismatch {a_c} vs {e_c}"
            assert abs(a_s - e_s) <= atol, f"batch {b}: score mismatch"
            np.testing.assert_allclose(a_bx, e_bx, atol=atol)

        if cnt < a_boxes.shape[1]:
            np.testing.assert_allclose(a_boxes[b, cnt:], 0.0)
            np.testing.assert_allclose(a_scores[b, cnt:], 0.0)
            np.testing.assert_allclose(a_classes[b, cnt:], -1.0)


class TRTBatchedNMSFunction(torch.autograd.Function):
    """Forward is a thin torch<->numpy bridge around
    ``reference_trt_batched_nms`` above (fine for an inference-only,
    no-gradient-needed custom op)."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_label_id,
        num_classes,
        top_k,
        keep_top_k,
        score_threshold,
        iou_threshold,
    ):
        device = boxes.device
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        boxes_sq = boxes_np[:, :, 0, :]  # class-agnostic: squeeze axis 2

        numdet, nboxes, nscores, nclasses = reference_trt_batched_nms(
            boxes_sq,
            scores_np,
            background_label_id,
            top_k,
            keep_top_k,
            score_threshold,
            iou_threshold,
        )
        return (
            torch.from_numpy(numdet.astype(np.int32)).to(device),
            torch.from_numpy(nboxes).to(device),
            torch.from_numpy(nscores).to(device),
            torch.from_numpy(nclasses).to(device),
        )

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_label_id,
        num_classes,
        top_k,
        keep_top_k,
        score_threshold,
        iou_threshold,
    ):
        return g.op(
            "mmdeploy::TRTBatchedNMS",
            boxes,
            scores,
            background_label_id_i=background_label_id,
            num_classes_i=num_classes,
            topK_i=top_k,
            keepTopK_i=keep_top_k,
            scoreThreshold_f=score_threshold,
            iouThreshold_f=iou_threshold,
            outputs=4,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "TRTBatchedNMSFunction is inference-only; no backward pass"
        )
