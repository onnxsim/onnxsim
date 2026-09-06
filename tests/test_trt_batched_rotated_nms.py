"""Tests for the ``rewrite_trt_batched_rotated_nms`` C++ pass
(``onnxsim/passes/rewrite_trt_batched_rotated_nms.h``) -- decomposes
mmdeploy's rotated-box ``TRTBatchedRotatedNMS`` plugin op into a subgraph
built from standard ONNX ops, computing rotated-box IoU from scratch via
Sutherland-Hodgman polygon clipping (ONNX has no primitive for either
rotated IoU or generic greedy suppression -- see the pass's own header
comment for the full derivation). This is an opt-in ``PassType::Other``
rewrite -- ``extra_optimizers=["rewrite_trt_batched_rotated_nms"]``.

**Why the isolated IoU tests use a threshold-sweep, not a standalone op**:
the rotated-IoU computation lives entirely inside the pass's C++ node-
construction code -- there is no separate "RotatedIoU" op this file could
build a tiny graph around and read the IoU value directly out of. Instead,
each isolated test below builds the *smallest possible* ``TRTBatchedRotated
NMS`` instance that still exercises the IoU computation in a controlled,
observable way: exactly 2 boxes, 1 class, ``keepTopK=topK=2``, box 0 given a
higher score than box 1. Under greedy NMS, box 1 survives (``num_detections
== 2``) iff ``IoU(box0, box1) <= iouThreshold``, and gets suppressed
(``num_detections == 1``) iff ``IoU(box0, box1) > iouThreshold``. Sweeping
``iouThreshold`` across a value just below and just above an independently
computed reference IoU therefore brackets the pass's own internal IoU value
to arbitrary precision (limited only by how tight a bracket the test
chooses), without needing any special test-only op or hook.

**The independent reference** (``_iou_ref``, module-level below) is a
plain, textbook, variable-length-list Sutherland-Hodgman clip + shoelace
formula -- deliberately written differently from the pass's own fixed-8-
slot, mask-free, vectorized implementation (the point of "independent" is
to not share the fixed-buffer compaction trick, which is the risky/novel
part of the C++ side). It is itself cross-checked against a second,
*entirely different* method -- Monte Carlo sampling (``_monte_carlo_iou``)
-- for the rotated-partial-overlap and fully-inside cases below, and
against plain closed-form rectangle IoU for the axis-aligned case, before
being trusted as the reference for both the isolated bracketing tests and
the full end-to-end greedy-NMS reference (``reference_trt_batched_rotated_
nms``) used by the full-pass tests further down.

**Known limitation, inherited from the pass itself**: exact tie-breaking
order for equal-or-near-equal scores/IoUs is not guaranteed to match any
particular reference implementation (see the pass's own header comment).
Full-pass tests below use well-separated random scores/boxes and compare
kept detections as an unordered-by-score-then-class multiset per batch
item, exactly mirroring ``tests/test_trt_batched_nms.py``'s own approach.
"""

import math

import numpy as np
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

PASS_NAME = "rewrite_trt_batched_rotated_nms"


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


def _rotated_nms_model(
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
    num_classes_dyn=False,
    boxes_class_dim=1,
    domain="mmdeploy",
):
    """Builds a single-node graph: `boxes, scores -> TRTBatchedRotatedNMS ->
    4 outputs`. `boxes` carries 5 values (cx,cy,w,h,theta) per box."""
    n_sym = "N" if n_dyn else str(n)
    b_sym = "B" if num_boxes_dyn else str(num_boxes)
    c_sym = "C" if num_classes_dyn else str(num_classes)
    op = f"{domain}.TRTBatchedRotatedNMS" if domain else "TRTBatchedRotatedNMS"
    extra_opsets = f', "{domain}": 1' if domain else ""
    return _model(
        f"""
        agraph (float[{n_sym},{b_sym},{boxes_class_dim},5] boxes,
                float[{n_sym},{b_sym},{c_sym}] scores)
              => (int32[{n_sym},1] num_detections,
                  float[{n_sym},{keep_top_k},5] nmsed_boxes,
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


def _simplify(model):
    return onnxsim.simplify(model, check_n=0, extra_optimizers=[PASS_NAME])


def _run(model, boxes, scores):
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" not in op_types, op_types
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {"boxes": boxes, "scores": scores})
    return simplified, tuple(outputs)


# ---------------------------------------------------------------------------
# Independent rotated-IoU reference: textbook Sutherland-Hodgman + shoelace
# (deliberately NOT the fixed-8-slot vectorized form the pass itself uses).
# ---------------------------------------------------------------------------


def _corners(box):
    cx, cy, w, h, theta = box
    dx, dy = w / 2.0, h / 2.0
    local = [(dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)]
    c, s = math.cos(theta), math.sin(theta)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in local]


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def _seg_intersect(p1, p2, q1, q2):
    """Point where line p1->p2 crosses the infinite line through q1->q2."""
    dx, dy = q2[0] - q1[0], q2[1] - q1[1]
    c1 = _cross2(dx, dy, p1[0] - q1[0], p1[1] - q1[1])
    c2 = _cross2(dx, dy, p2[0] - q1[0], p2[1] - q1[1])
    t = c1 / (c1 - c2)
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def _sh_clip(subject, clip):
    """Textbook (variable-length-list) Sutherland-Hodgman: clip `subject`
    (a CCW convex polygon, list of (x,y)) against every edge of `clip`
    (also CCW)."""
    output = subject
    cn = len(clip)
    for i in range(cn):
        if not output:
            break
        q1, q2 = clip[i], clip[(i + 1) % cn]
        edge_dx, edge_dy = q2[0] - q1[0], q2[1] - q1[1]

        def inside(p):
            return _cross2(edge_dx, edge_dy, p[0] - q1[0], p[1] - q1[1]) >= 0

        input_list = output
        output = []
        n = len(input_list)
        for j in range(n):
            cur = input_list[j]
            prev = input_list[j - 1]
            cur_in = inside(cur)
            prev_in = inside(prev)
            if cur_in:
                if not prev_in:
                    output.append(_seg_intersect(prev, cur, q1, q2))
                output.append(cur)
            elif prev_in:
                output.append(_seg_intersect(prev, cur, q1, q2))
    return output


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _iou_ref(box1, box2):
    c1, c2 = _corners(box1), _corners(box2)
    inter = _sh_clip(c1, c2)
    inter_area = _polygon_area(inter)
    a1, a2 = _polygon_area(c1), _polygon_area(c2)
    denom = a1 + a2 - inter_area
    return 0.0 if denom < 1e-9 else inter_area / denom


def _points_in_poly(pts, poly):
    inside = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cr = (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)
        inside &= cr >= -1e-9
    return inside


def _monte_carlo_iou(box1, box2, rng, n=400000):
    """Fully independent (no shared geometry code with `_iou_ref`) sanity
    check: uniform random sampling over the union's bounding box."""
    c1 = np.asarray(_corners(box1))
    c2 = np.asarray(_corners(box2))
    allc = np.vstack([c1, c2])
    lo, hi = allc.min(0), allc.max(0)
    pts = rng.uniform(lo, hi, size=(n, 2))
    in1 = _points_in_poly(pts, c1)
    in2 = _points_in_poly(pts, c2)
    inter, union = np.sum(in1 & in2), np.sum(in1 | in2)
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Threshold-sweep harness against the actual pass (see module docstring).
# ---------------------------------------------------------------------------


def _box1_survives(box0, box1, iou_threshold, score0=1.0, score1=0.9):
    """True iff box1 is kept (not suppressed by box0) under the pass's own
    compiled-and-executed subgraph, for a minimal 2-box/1-class/keepTopK=2
    instance."""
    model = _rotated_nms_model(
        1,
        2,
        1,
        keep_top_k=2,
        top_k=2,
        score_threshold=-1.0,
        iou_threshold=iou_threshold,
    )
    boxes = np.array([[[list(box0)], [list(box1)]]], dtype=np.float32)
    scores = np.array([[[score0], [score1]]], dtype=np.float32)
    _, actual = _run(model, boxes, scores)
    numdet = int(actual[0][0, 0])
    assert numdet in (1, 2)
    return numdet == 2


def _assert_brackets(box0, box1, ref_iou, margin=0.03):
    """Confirms the pass's own IoU is within `margin` of `ref_iou`: box1
    must be suppressed just below ref_iou and kept just above it."""
    lo = max(0.0, ref_iou - margin)
    hi = min(0.999, ref_iou + margin)
    assert not _box1_survives(box0, box1, lo), (
        f"expected suppression at iouThreshold={lo} (ref IoU={ref_iou})"
    )
    assert _box1_survives(box0, box1, hi), (
        f"expected box1 kept at iouThreshold={hi} (ref IoU={ref_iou})"
    )


# ---------------------------------------------------------------------------
# Isolated rotated-IoU tests (see module docstring for methodology).
# ---------------------------------------------------------------------------


def test_iou_axis_aligned_matches_ordinary_rectangle_iou():
    """theta=0 for both boxes should reduce to ordinary axis-aligned IoU --
    a strong sanity check on the rotation/corner math."""
    box0 = (0.0, 0.0, 4.0, 2.0, 0.0)
    box1 = (2.0, 0.0, 4.0, 2.0, 0.0)  # x-overlap [0,2], full y-overlap
    # Exact closed-form: inter=2*2=4, area=8 each, union=8+8-4=12, iou=1/3.
    ordinary_iou = 4.0 / 12.0
    ref_iou = _iou_ref(box0, box1)
    assert ref_iou == pytest.approx(ordinary_iou, abs=1e-9)
    _assert_brackets(box0, box1, ordinary_iou)


def test_iou_axis_aligned_no_overlap():
    box0 = (0.0, 0.0, 2.0, 2.0, 0.0)
    box1 = (10.0, 0.0, 2.0, 2.0, 0.0)
    assert _iou_ref(box0, box1) == 0.0
    assert _box1_survives(box0, box1, 0.0)


def test_iou_rotated_non_overlapping():
    """Rotated boxes far apart: IoU should be (numerically) 0 regardless of
    orientation."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta0 = rng.uniform(-math.pi, math.pi)
        theta1 = rng.uniform(-math.pi, math.pi)
        box0 = (0.0, 0.0, 2.0, 2.0, theta0)
        box1 = (20.0, 20.0, 2.0, 2.0, theta1)
        assert _iou_ref(box0, box1) == 0.0
        assert _box1_survives(box0, box1, 0.0)


def test_iou_rotated_partial_overlap():
    """Rotated, partially overlapping boxes -- reference cross-checked via
    an independent Monte Carlo estimate before bracketing the pass."""
    rng = np.random.default_rng(1)
    configs = [
        ((0.0, 0.0, 4.0, 2.0, 0.0), (1.0, 1.0, 3.0, 2.0, math.radians(30))),
        (
            (0.0, 0.0, 5.0, 3.0, math.radians(15)),
            (2.0, 0.5, 4.0, 2.0, math.radians(-40)),
        ),
        (
            (1.0, -1.0, 6.0, 2.0, math.radians(60)),
            (0.0, 0.0, 3.0, 3.0, math.radians(10)),
        ),
    ]
    for box0, box1 in configs:
        ref_iou = _iou_ref(box0, box1)
        mc_iou = _monte_carlo_iou(box0, box1, rng)
        assert ref_iou == pytest.approx(mc_iou, abs=0.01), (
            "independent Monte Carlo check disagrees with the textbook SH "
            f"reference: {ref_iou} vs {mc_iou}"
        )
        assert 0.02 < ref_iou < 0.9, ref_iou  # sanity: genuinely partial
        _assert_brackets(box0, box1, ref_iou)


def test_iou_fully_inside():
    """A small box concentric with (and sharing the rotation of) a much
    larger one is fully inside it -- exact IoU = small_area / large_area."""
    theta = math.radians(37)
    big = (0.0, 0.0, 10.0, 10.0, theta)
    small = (0.0, 0.0, 2.0, 2.0, theta)
    exact_iou = (2.0 * 2.0) / (10.0 * 10.0)
    ref_iou = _iou_ref(big, small)
    assert ref_iou == pytest.approx(exact_iou, abs=1e-9)
    mc_iou = _monte_carlo_iou(big, small, np.random.default_rng(2))
    assert ref_iou == pytest.approx(mc_iou, abs=0.01)
    _assert_brackets(big, small, exact_iou)


# ---------------------------------------------------------------------------
# Full end-to-end pass tests: independent NumPy reference of the whole
# algorithm (per-class greedy rotated NMS, capped at min(topK,keepTopK) per
# the pass's own documented restructuring, then per-batch top-K merge across
# classes), mirroring tests/test_trt_batched_nms.py's reference structure.
# ---------------------------------------------------------------------------


def _greedy_rotated_nms(boxes_c, scores_c, iou_threshold, cap):
    order = np.argsort(-scores_c, kind="stable")
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for pos, i in enumerate(order):
        if suppressed[pos]:
            continue
        keep.append(i)
        if len(keep) >= cap:
            break
        for pos2 in range(pos + 1, len(order)):
            if suppressed[pos2]:
                continue
            j = order[pos2]
            if _iou_ref(boxes_c[i], boxes_c[j]) > iou_threshold:
                suppressed[pos2] = True
    return keep


def reference_trt_batched_rotated_nms(
    boxes,
    scores,
    background_label_id,
    top_k,
    keep_top_k,
    score_threshold,
    iou_threshold,
):
    """boxes: (N, num_boxes, 5) float32 (already squeezed -- class-
    agnostic). scores: (N, num_boxes, num_classes) float32."""
    n, num_boxes, num_classes = scores.shape
    out_boxes = np.zeros((n, keep_top_k, 5), dtype=np.float32)
    out_scores = np.zeros((n, keep_top_k), dtype=np.float32)
    out_classes = np.full((n, keep_top_k), -1.0, dtype=np.float32)
    out_numdet = np.zeros((n, 1), dtype=np.int64)
    per_class_cap = min(top_k, keep_top_k)

    for b in range(n):
        pooled = []  # (score, class, box_index)
        for c in range(num_classes):
            if c == background_label_id:
                continue
            sc = scores[b, :, c]
            valid_idx = np.where(sc > score_threshold)[0]
            if len(valid_idx) == 0:
                continue
            kept_local = _greedy_rotated_nms(
                boxes[b, valid_idx], sc[valid_idx], iou_threshold, per_class_cap
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


def _random_rotated_boxes_scores(rng, n, num_boxes, num_classes, box_scale=10.0):
    centers = rng.uniform(0, box_scale, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    thetas = rng.uniform(-math.pi, math.pi, size=(n, num_boxes, 1))
    boxes = np.concatenate([centers, sizes, thetas], axis=-1).astype(np.float32)
    boxes_5d = boxes[:, :, None, :]  # (N,B,1,5)
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)
    return boxes_5d, boxes, scores


def _assert_same_detections(actual, expected, atol=1e-3):
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

        if cnt < a_boxes.shape[1]:
            np.testing.assert_allclose(a_boxes[b, cnt:], 0.0)
            np.testing.assert_allclose(a_scores[b, cnt:], 0.0)
            np.testing.assert_allclose(a_classes[b, cnt:], -1.0)


def _run_full(model, boxes_5d, scores):
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" not in op_types, op_types
    assert "ArgMax" in op_types, op_types
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {"boxes": boxes_5d, "scores": scores})
    return simplified, tuple(outputs)


def test_full_single_batch_item():
    rng = np.random.default_rng(10)
    n, num_boxes, num_classes, keep_top_k, top_k = 1, 8, 2, 4, 6
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


def test_full_multiple_batch_items():
    rng = np.random.default_rng(11)
    n, num_boxes, num_classes, keep_top_k, top_k = 2, 8, 2, 3, 5
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


def test_full_background_label_id_excludes_that_class():
    rng = np.random.default_rng(12)
    n, num_boxes, num_classes, keep_top_k, top_k = 1, 8, 3, 4, 6
    background_label_id = 1
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(
        n,
        num_boxes,
        num_classes,
        keep_top_k,
        top_k=top_k,
        background_label_id=background_label_id,
    )
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=background_label_id,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)

    a_classes = actual[3]
    assert not np.any(a_classes == float(background_label_id))


def test_full_fewer_detections_than_keep_top_k_exercises_padding():
    rng = np.random.default_rng(13)
    n, num_boxes, num_classes = 1, 5, 2
    keep_top_k = 12  # far more than num_boxes*num_classes can ever supply
    top_k = 12
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(
        n, num_boxes, num_classes, keep_top_k, top_k=top_k, score_threshold=0.3
    )
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.3,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)

    a_numdet = actual[0]
    assert np.all(a_numdet < keep_top_k), a_numdet
    assert np.any(a_numdet < keep_top_k)


def test_full_domain_empty_string_also_matches():
    rng = np.random.default_rng(14)
    n, num_boxes, num_classes, keep_top_k, top_k = 1, 6, 2, 3, 5
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(
        n, num_boxes, num_classes, keep_top_k, top_k=top_k, domain=None
    )
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


def test_full_per_class_cap_uses_min_topk_keeptopk():
    """topK deliberately larger than keepTopK: the pass caps per-class
    unrolling at min(topK,keepTopK) (see header comment) -- verify this
    restructuring still produces the spec-correct output."""
    rng = np.random.default_rng(15)
    n, num_boxes, num_classes = 1, 8, 2
    keep_top_k, top_k = 3, 100  # topK far above keepTopK
    boxes_5d, boxes, scores = _random_rotated_boxes_scores(
        rng, n, num_boxes, num_classes
    )

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    _, actual = _run_full(model, boxes_5d, scores)

    expected = reference_trt_batched_rotated_nms(
        boxes,
        scores,
        background_label_id=-1,
        top_k=top_k,
        keep_top_k=keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detections(actual, expected)


# ---------------------------------------------------------------------------
# Negative tests: predicate must decline outside this pass's documented scope
# ---------------------------------------------------------------------------


def test_declines_when_boxes_have_per_class_boxes():
    n, num_boxes, num_classes, keep_top_k = 1, 6, 3, 4
    model = _rotated_nms_model(
        n, num_boxes, num_classes, keep_top_k, boxes_class_dim=num_classes
    )
    simplified, ok = _simplify(model)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_declines_when_batch_size_is_dynamic():
    n, num_boxes, num_classes, keep_top_k = 2, 6, 3, 4
    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, n_dyn=True)
    simplified, ok = _simplify(model)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_declines_when_num_classes_is_dynamic():
    """Stricter than rewrite_trt_batched_nms.h: num_classes must ALWAYS be
    statically known here (this pass always unrolls a per-class C++ loop),
    unlike the axis-aligned pass which only needed this conditionally."""
    n, num_boxes, num_classes, keep_top_k = 1, 6, 3, 4
    model = _rotated_nms_model(
        n, num_boxes, num_classes, keep_top_k, num_classes_dyn=True
    )
    simplified, ok = _simplify(model)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_declines_when_total_iterations_exceed_cap():
    """N * num_classes_active * min(topK,keepTopK) over the documented cap
    (2000) must decline -- see the pass's header comment."""
    n, num_boxes, num_classes, keep_top_k, top_k = 1, 6, 2001, 1, 1
    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    simplified, ok = _simplify(model)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_extra_optimizers_required_to_fire():
    n, num_boxes, num_classes, keep_top_k = 1, 6, 3, 4
    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k)
    simplified, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    op_types = [nd.op_type for nd in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types
