"""Tests for the ``rewrite_trt_batched_nms`` C++ pass
(``onnxsim/passes/rewrite_trt_batched_nms.h``) -- decomposes mmdeploy's
custom ``TRTBatchedNMS`` op (a closed TensorRT-plugin post-processing op with
no ONNX Runtime kernel, emitted at the end of most 2-D/BEV detection
pipelines) into a subgraph built from standard ONNX ops centered on
``NonMaxSuppression``. This is an opt-in ``PassType::Other`` rewrite --
``extra_optimizers=["rewrite_trt_batched_nms"]``.

**Why this test file can't use onnxsim's usual pre/post equivalence check**:
``TRTBatchedNMS`` has no ONNX Runtime kernel and no ``onnx.reference``
kernel -- there is *no way to execute the original graph at all*, so
``onnxsim.simplify``'s own ``check_n``-based comparison (which runs the
*original* graph to compare against) cannot be used here (every call below
passes ``check_n=0``). Instead, each test below implements its own
independent NumPy reference of the standard "per-class greedy NMS, then
global per-batch top-K merge" algorithm the op is documented to perform (see
``reference_trt_batched_nms`` below), runs it on the same random boxes/
scores used to build the test model, and compares the *simplified* graph's
output -- executed through ``onnxruntime``, which does have a
``NonMaxSuppression`` kernel -- against that independent reference.

**Known limitation, inherited from the pass itself**: ``TRTBatchedNMS`` is a
closed TensorRT plugin (see NVIDIA's TensorRT OSS repo,
``plugin/batchedNMSPlugin`` / ``efficientNMSPlugin``, for the closest public
description); its exact tie-breaking order for equal-or-near-equal scores
and its internal floating-point rounding are not fully specifiable from
outside. This pass -- and this test -- therefore only claim the same SET of
kept detections (by box + class + score, up to padding/count) as the
NumPy reference algorithm below, not bit-identical order against a real
TensorRT-plugin-produced reference (which is not available here anyway,
precisely because the op has no runnable kernel). To keep this comparison
meaningful, test boxes/scores are randomized with enough separation that
near-ties (which would make even *this* comparison order-sensitive) are
vanishingly unlikely; comparisons below match kept detections as an
unordered multiset per batch item, sorted by score, rather than assuming a
specific tie order.
"""

import numpy as np
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; running the decomposed graph
# (built around NonMaxSuppression) needs a real kernel.
ort = pytest.importorskip("onnxruntime")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _model(body, initializer=(), opset=13, ir_version=10, extra_opsets=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_opsets}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _trt_batched_nms_model(
    n,
    num_boxes,
    num_classes,
    keep_top_k,
    top_k=200,
    score_threshold=0.05,
    iou_threshold=0.5,
    background_label_id=-1,
    num_boxes_dyn=False,
    n_dyn=False,
    boxes_class_dim=1,
    domain="mmdeploy",
):
    """Builds a single-node graph: `boxes, scores -> TRTBatchedNMS -> 4 outputs`."""
    n_sym = "N" if n_dyn else str(n)
    b_sym = "B" if num_boxes_dyn else str(num_boxes)
    op = f"{domain}.TRTBatchedNMS" if domain else "TRTBatchedNMS"
    extra_opsets = f', "{domain}": 1' if domain else ""
    return _model(
        f"""
        agraph (float[{n_sym},{b_sym},{boxes_class_dim},4] boxes,
                float[{n_sym},{b_sym},{num_classes}] scores)
              => (int32[{n_sym},1] num_detections,
                  float[{n_sym},{keep_top_k},4] nmsed_boxes,
                  float[{n_sym},{keep_top_k}] nmsed_scores,
                  float[{n_sym},{keep_top_k}] nmsed_classes)
        {{
          num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = {op}
            <background_label_id={background_label_id}, num_classes={num_classes},
             topK={top_k}, keepTopK={keep_top_k}, scoreThreshold={score_threshold},
             iouThreshold={iou_threshold}>
            (boxes, scores)
        }}
        """,
        extra_opsets=extra_opsets,
    )


def _random_boxes_scores(rng, n, num_boxes, num_classes, box_scale=10.0):
    """Random well-separated corner-format boxes (shared across classes,
    i.e. num_classes_or_1 == 1) and per-class scores."""
    centers = rng.uniform(0, box_scale, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    x1y1 = centers - sizes / 2
    x2y2 = centers + sizes / 2
    boxes = np.concatenate([x1y1, x2y2], axis=-1).astype(np.float32)  # (N,B,4)
    boxes_5d = boxes[:, :, None, :]  # (N,B,1,4)
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)
    return boxes_5d, boxes, scores


# ---------------------------------------------------------------------------
# Independent NumPy reference implementation (see module docstring)
# ---------------------------------------------------------------------------


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
    into boxes_c/scores_c, at most top_k of them."""
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
    scores: (N, num_boxes, num_classes) float32.
    Implements the "per-class greedy NMS, then global per-batch top-K merge"
    algorithm this op is documented to perform -- see this module's and
    ``rewrite_trt_batched_nms.h``'s own header comments.
    """
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


# ---------------------------------------------------------------------------
# Comparison: unordered multiset per batch item, by (score, class, box)
# ---------------------------------------------------------------------------


def _assert_same_detections(actual, expected, atol=1e-4):
    """Compares the pass's simplified-graph output against the reference,
    per batch item, as an unordered set of kept detections (see module
    docstring: tie order is not guaranteed to match)."""
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
            assert a_c == pytest.approx(e_c), f"batch {b}: class mismatch"
            assert a_s == pytest.approx(e_s, abs=atol), f"batch {b}: score mismatch"
            np.testing.assert_allclose(a_bx, e_bx, atol=atol)

        # Padding past num_detections must be zero (boxes/scores) / -1 (class).
        if cnt < a_boxes.shape[1]:
            np.testing.assert_allclose(a_boxes[b, cnt:], 0.0)
            np.testing.assert_allclose(a_scores[b, cnt:], 0.0)
            np.testing.assert_allclose(a_classes[b, cnt:], -1.0)


def _run_simplified(model, boxes_5d, scores):
    simplified, ok = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["rewrite_trt_batched_nms"]
    )
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedNMS" not in op_types, op_types
    assert "NonMaxSuppression" in op_types, op_types

    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {"boxes": boxes_5d, "scores": scores})
    return simplified, tuple(outputs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_batch_item():
    rng = np.random.default_rng(0)
    n, num_boxes, num_classes, keep_top_k = 1, 40, 4, 20
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(n, num_boxes, num_classes, keep_top_k)
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


def test_multiple_batch_items():
    rng = np.random.default_rng(1)
    n, num_boxes, num_classes, keep_top_k = 3, 30, 5, 15
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(n, num_boxes, num_classes, keep_top_k)
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


def test_background_label_id_negative_one_keeps_all_classes():
    rng = np.random.default_rng(2)
    n, num_boxes, num_classes, keep_top_k = 2, 25, 4, 10
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(
        n, num_boxes, num_classes, keep_top_k, background_label_id=-1
    )
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)
    # Sanity: every one of num_classes appears somewhere as a kept class
    # across enough random trials (background_label_id=-1 excludes nothing).
    assert set(np.unique(expected[3][expected[3] >= 0])) <= set(range(num_classes))


def test_background_label_id_excludes_that_class():
    rng = np.random.default_rng(3)
    n, num_boxes, num_classes, keep_top_k = 2, 30, 4, 12
    background_label_id = 1
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(
        n,
        num_boxes,
        num_classes,
        keep_top_k,
        background_label_id=background_label_id,
    )
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=background_label_id,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)

    _, a_boxes, a_scores, a_classes = actual
    # The excluded class must never appear among the kept detections.
    assert not np.any(a_classes == float(background_label_id))


def test_fewer_detections_than_keep_top_k_exercises_padding():
    rng = np.random.default_rng(4)
    n, num_boxes, num_classes = 2, 8, 2
    keep_top_k = 50  # far more than num_boxes * num_classes can ever supply
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(
        n, num_boxes, num_classes, keep_top_k, score_threshold=0.3
    )
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.3,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)

    a_numdet = actual[0]
    assert np.all(a_numdet < keep_top_k), a_numdet
    assert np.any(a_numdet < keep_top_k)


def test_domain_empty_string_also_matches():
    """The pass matches domain "" as well as "mmdeploy" (see header comment)."""
    rng = np.random.default_rng(5)
    n, num_boxes, num_classes, keep_top_k = 1, 20, 3, 10
    boxes_5d, boxes, scores = _random_boxes_scores(rng, n, num_boxes, num_classes)

    model = _trt_batched_nms_model(n, num_boxes, num_classes, keep_top_k, domain=None)
    _, actual = _run_simplified(model, boxes_5d, scores)

    expected = reference_trt_batched_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=200,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


# ---------------------------------------------------------------------------
# Negative tests: predicate must decline outside this pass's documented scope
# ---------------------------------------------------------------------------


def test_declines_when_boxes_have_per_class_boxes():
    """num_classes_or_1 > 1 (statically known) is out of scope -- ONNX
    NonMaxSuppression has no notion of per-class boxes."""
    n, num_boxes, num_classes, keep_top_k = 1, 10, 3, 5
    model = _trt_batched_nms_model(
        n, num_boxes, num_classes, keep_top_k, boxes_class_dim=num_classes
    )
    simplified, ok = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["rewrite_trt_batched_nms"]
    )
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedNMS" in op_types, op_types
    assert "NonMaxSuppression" not in op_types, op_types


def test_declines_when_batch_size_is_dynamic():
    """N (batch size) not statically known is out of scope -- this pass
    unrolls a per-batch-item C++ loop at pass-build time."""
    n, num_boxes, num_classes, keep_top_k = 2, 10, 3, 5
    model = _trt_batched_nms_model(n, num_boxes, num_classes, keep_top_k, n_dyn=True)
    simplified, ok = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["rewrite_trt_batched_nms"]
    )
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedNMS" in op_types, op_types
    assert "NonMaxSuppression" not in op_types, op_types


def test_extra_optimizers_required_to_fire():
    """The pass is opt-in: plain simplify() must leave TRTBatchedNMS alone."""
    n, num_boxes, num_classes, keep_top_k = 1, 10, 3, 5
    model = _trt_batched_nms_model(n, num_boxes, num_classes, keep_top_k)
    simplified, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedNMS" in op_types, op_types
