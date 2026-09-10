"""Formal check for RewriteTRTBatchedNMS (opt-in; onnxsim's own
``onnxsim/passes/rewrite_trt_batched_nms.h``): decomposes mmdeploy's custom
``TRTBatchedNMS`` op (a closed TensorRT-plugin multiclass-batched-NMS
post-processing op, with no ONNX Runtime kernel and no ``onnx.reference``
kernel) into a subgraph built from standard ONNX ops centered on
``NonMaxSuppression``. Opt-in ``PassType::Other`` rewrite --
``extra_optimizers=["rewrite_trt_batched_nms"]``.

**This file is structurally different from every quantization
formal-verify file in this suite**: it is a pure graph-shape rewrite with
no numeric error to bound. The claim worth proving is a SET/ORDERING
correctness claim -- "the rewritten graph keeps the same SET of detections
(by box + class + score, up to `keepTopK` and documented padding) as a
well-specified reference algorithm" -- not a bounded-numeric-error claim,
and NOT bit-identical/tie-break-identical output against the real,
closed-source TensorRT plugin. This matches the pass's own header comment
precisely (see "KNOWN LIMITATION" there): the header itself disclaims
bit-exactness and tie-order fidelity against the real plugin, and states
the goal only as matching a well-specified reference algorithm's SET of
kept detections. This file adopts exactly that scope, no more.

**What is Z3-proved vs. what is left to differential testing, and why**:
This pass leans entirely on ONNX's own ``NonMaxSuppression`` for the actual
per-class greedy-NMS/IoU work -- that correctness is ``NonMaxSuppression``'s
own contract (part of the ONNX operator spec), not this rewrite's
responsibility, and is NOT re-verified here. What IS this pass's own
responsibility -- and therefore the actual content formalized below -- is
the orchestration around it:

1. **Background-class exclusion via additive masking** (header step 3):
   proved as a general Z3 lemma (``test_background_mask_beats_any_...``)
   that ``-1e9 + s < scoreThreshold`` for any realistic score ``s`` and any
   not-absurdly-low ``scoreThreshold`` -- i.e. masking is equivalent to
   omitting the class from consideration entirely, because the masked
   class can never survive NonMaxSuppression's own pre-NMS score-threshold
   filter (header step 4).
2. **Cross-class top-K merge via the TopK op** (header step 6c): a full
   N-candidate combinatorial equivalence to a "sort everything, take the
   top keepTopK" reference is exercised concretely (by hand-verified
   inspection) in the 6-candidate differential test below, but is NOT
   itself restated as a single Z3 query -- for a *fixed, hand-picked*
   instance that is a numeric fact, not a universally-quantified one worth
   asking Z3 to reprove. What IS proved with Z3, universally (for ANY
   assignment of scores, not one concrete instance), is the abstract
   structural property that makes ANY TopK-style selector correct in the
   first place (``test_topk_selection_matches_smallest_rank_...``): a
   selector that (a) picks exactly K indices and (b) never excludes an
   index whose value exceeds an included one, is *forced* to select
   exactly the K smallest-rank (i.e. K largest) elements. N=5, K=2 is used
   -- large enough to be a genuine multi-way selection, small enough to
   state without a variadic "for all N" quantifier (Z3 has none). The
   6-candidate concrete case is exactly this same style of selection one
   size up; being honest about the difference: the Z3 lemma is a general
   theorem about the selection rule, the differential test is a witness
   that the compiled pass's actual node chain realizes that rule on one
   hand-checkable instance.
3. **Box clipping, output dtype, and padding conventions** (header steps 1
   and 6b/6d): pure structural/value facts (Clip to [0,1]; padding values
   exactly 0.0 for boxes/scores and -1.0 for classes past
   ``num_detections[n]``) confirmed empirically against the compiled pass's
   real output, not restated as Z3 claims -- there is no algebra to prove,
   only "did the actual op sequence produce the documented constant".

**Tie-break honesty**: as in the pass's own header and in
``tests/test_trt_batched_nms.py`` (this repo's non-formal-verify
differential suite for the same pass), comparisons below never assume a
specific tie order for equal-or-near-equal scores -- kept detections are
compared as a set/sorted-by-score list, never index-for-index against an
arbitrary tie-break. The hand-crafted scene below is deliberately built
with well-separated, pairwise-distinct scores precisely so this comparison
is meaningful without needing tie-break logic at all.

Model construction uses ``onnx.parser.parse_model`` throughout, including
the custom ``TRTBatchedNMS`` node itself: the text parser does not need a
registered schema for a domain-qualified op -- it only needs the domain
declared in ``opset_import`` -- so, unlike ops requiring true schema
validation, no ``onnx.helper.make_node`` fallback is needed here (confirmed
against ``tests/test_trt_batched_nms.py``, which uses the same approach).
"""

import numpy as np
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; running the decomposed
# graph (built around NonMaxSuppression) needs a real kernel -- there is no
# onnx.reference kernel for TRTBatchedNMS itself, and the rewritten graph's
# NonMaxSuppression has no onnx.reference kernel either.
ort = pytest.importorskip("onnxruntime")

PASS_NAME = "rewrite_trt_batched_nms"


# ---------------------------------------------------------------------------
# Part 1: Z3 lemmas for the pass's own orchestration logic
# ---------------------------------------------------------------------------


def test_background_mask_beats_any_realistic_score_threshold():
    """Header step 3: background_label_id's class is excluded by adding a
    ``-1e9`` mask to its scores before NonMaxSuppression, rather than a
    separate code branch. This is only equivalent to *omitting* that class
    from consideration if the masked score can never pass the pre-NMS
    ``score_threshold`` filter NonMaxSuppression itself applies (header
    step 4).

    Concrete hypotheses (stated, not assumed silently):
      * a realistic detector score ``s`` is bounded, e.g. ``|s| <= 1e6``
        (already enormous headroom over the ``[0,1]``-probability scores
        mmdeploy's own detection heads actually emit);
      * ``scoreThreshold`` is not itself set absurdly low, e.g.
        ``scoreThreshold > -1e8`` (still far more permissive than any sane
        NMS config, which in practice always uses a small positive
        threshold).

    Under those two hypotheses, ``-1e9 + s`` is strictly below
    ``scoreThreshold`` for EVERY ``s`` and EVERY ``scoreThreshold`` in that
    range -- not just the specific numeric values this file's differential
    test happens to exercise below.
    """
    s, threshold = z3.Reals("s threshold")
    realistic_score = z3.And(s >= -1e6, s <= 1e6)
    realistic_threshold = threshold > -1e8
    masked_score = -1e9 + s

    prove(
        z3.Implies(
            z3.And(realistic_score, realistic_threshold),
            masked_score < threshold,
        ),
        msg="background additive mask does not dominate scoreThreshold",
    )


def test_topk_selection_matches_smallest_rank_characterization():
    """Header step 6c: the cross-class merge is exactly
    ``TopK(scores, k=keepTopK, largest=1, sorted=1)`` -- pick the K
    highest-scoring rows. Rather than re-deriving one concrete N-candidate
    instance (that is what the differential test below does, by hand), what
    is proved here -- universally, via free real-valued variables rather
    than one numeric example -- is the abstract structural property that
    makes ANY TopK-style selector correct in the first place:

    A selection of exactly K indices that never excludes an index whose
    value exceeds an included index's value is *forced* to be exactly the
    K indices of smallest rank (``rank(i)`` = how many other candidates
    strictly beat candidate ``i``), i.e. exactly the K largest values under
    the standard total order on floats.

    N=5, K=2: large enough for "the K largest" to be a genuine multi-way
    selection (not a degenerate 1-vs-rest case), small enough to state
    without a variadic "for all N" quantifier over sums (Z3 has none).
    Scores are assumed pairwise distinct, sidestepping tie-break ambiguity
    -- exact tie-break order against a real TensorRT plugin is explicitly
    out of scope for this pass and this file (see module docstring).
    """
    n, k = 5, 2
    v = z3.Reals(" ".join(f"v{i}" for i in range(n)))
    sel = z3.Bools(" ".join(f"sel{i}" for i in range(n)))

    distinct = z3.And([v[i] != v[j] for i in range(n) for j in range(i + 1, n)])
    exactly_k_selected = z3.Sum([z3.If(s, 1, 0) for s in sel]) == k
    # The one structural property any correct Top-K selector has: never
    # exclude an index whose value exceeds an included index's value.
    never_excludes_a_larger_value = z3.And(
        [
            z3.Implies(z3.And(z3.Not(sel[i]), sel[j]), v[i] <= v[j])
            for i in range(n)
            for j in range(n)
            if i != j
        ]
    )

    def rank(i):
        return z3.Sum([z3.If(v[j] > v[i], 1, 0) for j in range(n) if j != i])

    hypotheses = z3.And(distinct, exactly_k_selected, never_excludes_a_larger_value)
    prove(
        z3.Implies(
            hypotheses,
            z3.And([sel[i] == (rank(i) < k) for i in range(n)]),
        ),
        msg="a correct-by-construction Top-K selector must equal the smallest-rank set",
    )


# ---------------------------------------------------------------------------
# Part 2: registration / opt-in sanity
# ---------------------------------------------------------------------------


def test_pass_is_registered_as_opt_in_not_default():
    assert PASS_NAME in C._list_other_optimizers()
    assert PASS_NAME not in C._list_optimizers()


def test_plain_simplify_leaves_trt_batched_nms_alone():
    """The pass is opt-in: plain ``simplify()`` (no ``extra_optimizers``)
    must never touch ``TRTBatchedNMS``."""
    model = _trt_batched_nms_model(n=1, num_boxes=5, num_classes=2, keep_top_k=3)
    sim_model, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    op_types = [nd.op_type for nd in sim_model.graph.node]
    assert "TRTBatchedNMS" in op_types, op_types


# ---------------------------------------------------------------------------
# Part 3: from-scratch NumPy reference, built fresh for this file from the
# pass header's own step-by-step spec (NOT imported from
# tests/test_trt_batched_nms.py's own independent reference, though both
# necessarily implement the same documented algorithm).
# ---------------------------------------------------------------------------


def _iou_xyxy(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _greedy_nms_indices(boxes, class_scores, score_threshold, iou_threshold, top_k):
    """Standard greedy NMS for one (batch, class) pair: sort surviving
    candidates (score > score_threshold) by score descending, greedily keep
    a box unless it overlaps (IoU > iou_threshold) a box already kept, and
    stop once ``top_k`` have been kept. This is exactly what
    NonMaxSuppression is documented to do internally, per-class -- header
    step 4 -- reimplemented here independently so the differential tests
    below have a from-scratch oracle."""
    survivors = [
        i for i in range(len(class_scores)) if class_scores[i] > score_threshold
    ]
    survivors.sort(key=lambda i: -class_scores[i])
    kept = []
    for i in survivors:
        if len(kept) >= top_k:
            break
        if all(_iou_xyxy(boxes[i], boxes[k]) <= iou_threshold for k in kept):
            kept.append(i)
    return kept


def reference_kept_detections(
    boxes,
    scores,
    background_label_id,
    top_k,
    keep_top_k,
    score_threshold,
    iou_threshold,
):
    """From-scratch reference for ONE batch item, implementing the
    "per-class greedy NMS, then cross-class score-sorted top-keepTopK
    merge" algorithm documented in ``rewrite_trt_batched_nms.h``'s header
    comment (steps 3-6).

    boxes: (num_boxes, 4) corner format. scores: (num_boxes, num_classes).
    Returns a list of ``(score, class, box_index)`` tuples, sorted by score
    descending, length <= ``keep_top_k`` -- the SET this pass's real output
    is claimed to match (see module docstring for the tie-break caveat).
    """
    num_boxes, num_classes = scores.shape
    pooled = []
    for c in range(num_classes):
        if c == background_label_id:
            continue
        kept = _greedy_nms_indices(
            boxes, scores[:, c], score_threshold, iou_threshold, top_k
        )
        for i in kept:
            pooled.append((float(scores[i, c]), c, i))
    pooled.sort(key=lambda t: -t[0])
    return pooled[:keep_top_k]


def reference_full_batch(
    boxes,
    scores,
    background_label_id,
    top_k,
    keep_top_k,
    score_threshold,
    iou_threshold,
):
    """Applies ``reference_kept_detections`` per batch item and assembles
    the (numdet, boxes, scores, classes) arrays in the op's own output
    layout, zero/-1.0-padded past each batch item's own count -- the same
    padding convention documented in the header (step 6d) and confirmed
    structurally below against the real compiled pass."""
    n, num_boxes, _ = boxes.shape
    out_boxes = np.zeros((n, keep_top_k, 4), dtype=np.float32)
    out_scores = np.zeros((n, keep_top_k), dtype=np.float32)
    out_classes = np.full((n, keep_top_k), -1.0, dtype=np.float32)
    out_numdet = np.zeros((n, 1), dtype=np.int64)
    for b in range(n):
        pooled = reference_kept_detections(
            boxes[b],
            scores[b],
            background_label_id,
            top_k,
            keep_top_k,
            score_threshold,
            iou_threshold,
        )
        out_numdet[b, 0] = len(pooled)
        for k, (sc, c, bidx) in enumerate(pooled):
            out_boxes[b, k] = boxes[b, bidx]
            out_scores[b, k] = sc
            out_classes[b, k] = float(c)
    return out_numdet, out_boxes, out_scores, out_classes


def _assert_matches_reference(actual, expected, atol=1e-5):
    """Compares the pass's simplified-graph output against the reference,
    per batch item, as a score-sorted list (tie order is never assumed --
    see module docstring), and confirms the documented padding convention
    past each batch item's own detection count."""
    a_numdet, a_boxes, a_scores, a_classes = actual
    e_numdet, e_boxes, e_scores, e_classes = expected
    n = e_numdet.shape[0]
    for b in range(n):
        assert int(a_numdet[b, 0]) == int(e_numdet[b, 0]), (
            f"batch {b}: num_detections {int(a_numdet[b, 0])} vs {int(e_numdet[b, 0])}"
        )
        cnt = int(e_numdet[b, 0])

        def _sorted_rows(scores_row, boxes_row, classes_row, cnt=cnt):
            rows = [
                (
                    float(scores_row[k]),
                    float(classes_row[k]),
                    tuple(float(x) for x in boxes_row[k]),
                )
                for k in range(cnt)
            ]
            rows.sort(key=lambda t: -t[0])
            return rows

        a_rows = _sorted_rows(a_scores[b], a_boxes[b], a_classes[b])
        e_rows = _sorted_rows(e_scores[b], e_boxes[b], e_classes[b])
        for (a_s, a_c, a_bx), (e_s, e_c, e_bx) in zip(a_rows, e_rows):
            assert a_c == pytest.approx(e_c), f"batch {b}: class mismatch"
            assert a_s == pytest.approx(e_s, abs=atol), f"batch {b}: score mismatch"
            np.testing.assert_allclose(a_bx, e_bx, atol=atol)

        # Documented padding convention (header step 6d): 0.0 for
        # boxes/scores, -1.0 for classes, past this batch item's own count.
        if cnt < a_boxes.shape[1]:
            np.testing.assert_allclose(a_boxes[b, cnt:], 0.0)
            np.testing.assert_allclose(a_scores[b, cnt:], 0.0)
            np.testing.assert_allclose(a_classes[b, cnt:], -1.0)


# ---------------------------------------------------------------------------
# Part 4: model construction
# ---------------------------------------------------------------------------


def _trt_batched_nms_model(
    n,
    num_boxes,
    num_classes,
    keep_top_k,
    top_k=50,
    score_threshold=0.1,
    iou_threshold=0.5,
    background_label_id=-1,
    boxes_class_dim=1,
    n_dyn=False,
):
    """Single-node graph: ``boxes, scores -> mmdeploy.TRTBatchedNMS -> 4
    outputs``, matching the op spec in ``rewrite_trt_batched_nms.h``'s own
    header comment (attribute names, input/output shapes and dtypes). Only
    shapes/attributes go into the model; the actual box/score values used
    at runtime are built separately by each test and fed straight to
    onnxruntime. ``onnx.parser`` handles the domain-qualified custom op
    directly -- it only needs "mmdeploy" declared in ``opset_import``, not
    a registered schema -- so no ``onnx.helper.make_node`` fallback is
    needed (see module docstring)."""
    n_sym = "N" if n_dyn else str(n)
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13, "mmdeploy": 1]
        >
        agraph (float[{n_sym},{num_boxes},{boxes_class_dim},4] boxes,
                float[{n_sym},{num_boxes},{num_classes}] scores)
              => (int32[{n_sym},1] num_detections,
                  float[{n_sym},{keep_top_k},4] nmsed_boxes,
                  float[{n_sym},{keep_top_k}] nmsed_scores,
                  float[{n_sym},{keep_top_k}] nmsed_classes)
        {{
          num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = mmdeploy.TRTBatchedNMS
            <background_label_id={background_label_id}, num_classes={num_classes},
             topK={top_k}, keepTopK={keep_top_k}, scoreThreshold={score_threshold},
             iouThreshold={iou_threshold}>
            (boxes, scores)
        }}
        """
    )


def _run_rewritten(model, boxes_3d, scores):
    """Runs the pass alone (``simplify_isolated_extra``, ``check_n=0`` --
    onnxsim's own numeric ``--check`` cannot run here since there is no
    kernel for the *original* graph's TRTBatchedNMS node to compare
    against), confirms the custom op is fully gone and NonMaxSuppression is
    present, then executes the rewritten graph through onnxruntime with
    graph optimization disabled (``ORT_DISABLE_ALL`` -- this suite's
    established precedent for quantized/decomposed graphs whose exact node
    shape the proof reasons about; see e.g.
    tests/test_formal_verify_qoperator_quantize_where.py's docstring)."""
    sim_model, ops = simplify_isolated_extra(model, PASS_NAME, check_n=0)
    assert ops["TRTBatchedNMS"] == 0, ops
    assert ops["NonMaxSuppression"] >= 1, ops
    # Structural spot-check (header's own opset-floor comment): the merge
    # actually goes through Pad/TopK/GatherND/Compress, not just NMS alone.
    for expected_op in ("Pad", "TopK", "GatherND", "Compress"):
        assert ops[expected_op] >= 1, (expected_op, ops)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        sim_model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    boxes_5d = boxes_3d[:, :, None, :]
    outputs = sess.run(None, {"boxes": boxes_5d, "scores": scores})
    return sim_model, tuple(outputs)


# ---------------------------------------------------------------------------
# Part 5: differential tests
# ---------------------------------------------------------------------------


def test_hand_crafted_scene_matches_reference_with_background_exclusion():
    """A hand-picked 7-box, 3-class scene, small enough to hand-verify by
    inspection (worked out and cross-checked against a standalone
    simulation before writing this test):

      box0=[0,0,10,10]      box1=[1,1,11,11]     (IoU(box0,box1) ~ 0.68)
      box2=[50,50,60,60]    box3=[51,51,61,61]   (IoU(box2,box3) ~ 0.68)
      box4=[100,100,110,110] box5=[100,101,110,111] (IoU(box4,box5) ~ 0.82)
      box6=[200,200,205,205] (isolated, all-class score 0.02 -- filtered by
        score_threshold=0.1 regardless of class)

    scores (columns: class0, class1, class2=background):
      box0: 0.90, 0.15, 0.95     box1: 0.80, 0.05, 0.99
      box2: 0.70, 0.60, 0.40     box3: 0.20, 0.65, 0.30
      box4: 0.40, 0.30, 0.20     box5: 0.35, 0.25, 0.15
      box6: 0.02, 0.02, 0.02

    With background_label_id=2, iou_threshold=0.5, score_threshold=0.1,
    top_k=50 (no per-class limiting), keep_top_k=8 (2 more than the 6 total
    surviving detections, to exercise padding): per-class greedy NMS keeps
    class0={box0,box2,box4}, class1={box3,box4,box0}; pooled and sorted by
    score descending: (0.90,c0,box0), (0.70,c0,box2), (0.65,c1,box3),
    (0.40,c0,box4), (0.30,c1,box4), (0.15,c1,box0) -- 6 detections, rows 6
    and 7 padded.

    Crucially: box1 has the single highest raw score anywhere in the scene
    (0.99, class2) -- if class2 were NOT background, box1/class2 would be
    the #1 overall detection (confirmed below by calling the reference
    with background_label_id=-1). With background_label_id=2 in the actual
    model, class2 is masked before NonMaxSuppression ever runs, so box1
    never appears in the real output at all -- this is the concrete
    instance of the masking equivalence proved abstractly in
    test_background_mask_beats_any_realistic_score_threshold.
    """
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
            [51, 51, 61, 61],
            [100, 100, 110, 110],
            [100, 101, 110, 111],
            [200, 200, 205, 205],
        ],
        dtype=np.float32,
    )[None, :, :]  # (1, 7, 4)
    scores = np.array(
        [
            [0.90, 0.15, 0.95],
            [0.80, 0.05, 0.99],
            [0.70, 0.60, 0.40],
            [0.20, 0.65, 0.30],
            [0.40, 0.30, 0.20],
            [0.35, 0.25, 0.15],
            [0.02, 0.02, 0.02],
        ],
        dtype=np.float32,
    )[None, :, :]  # (1, 7, 3)

    common = dict(top_k=50, keep_top_k=8, score_threshold=0.1, iou_threshold=0.5)
    model = _trt_batched_nms_model(
        n=1,
        num_boxes=7,
        num_classes=3,
        background_label_id=2,
        **common,
    )
    _, actual = _run_rewritten(model, boxes, scores)
    a_numdet, a_boxes, a_scores, a_classes = actual

    expected = reference_full_batch(boxes, scores, background_label_id=2, **common)
    _assert_matches_reference(actual, expected)

    # Explicit, literal padding check (header step 6d), in addition to the
    # generic tail-check inside _assert_matches_reference: rows 6 and 7 are
    # exactly the documented sentinel values, not merely "close to zero".
    assert int(a_numdet[0, 0]) == 6
    np.testing.assert_array_equal(a_boxes[0, 6:8], np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(a_scores[0, 6:8], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(a_classes[0, 6:8], np.full(2, -1.0, dtype=np.float32))

    # The background-exclusion path, explicitly: box1/class2 would be the
    # #1 overall detection if class2 weren't masked out as background --
    # confirm that via the reference with background disabled...
    unmasked = reference_kept_detections(
        boxes[0], scores[0], background_label_id=-1, **common
    )
    assert unmasked[0] == (pytest.approx(0.99), 2, 1), unmasked[0]
    # ...then confirm it never appears in the real (background-excluding)
    # output at all.
    assert not np.any(a_classes[0, :6] == 2.0), a_classes[0]


def test_random_multi_batch_matches_reference():
    """A second, randomized differential check across multiple batch items
    (no background exclusion here -- that path is covered by the
    hand-crafted scene above), for extra confidence beyond one hand-picked
    instance. Boxes are randomly placed with a wide spread and reused
    across classes (num_classes_or_1 == 1, this pass's only supported
    shape), and scores are independently randomized per class."""
    rng = np.random.default_rng(0)
    n, num_boxes, num_classes, keep_top_k = 2, 6, 2, 4
    centers = rng.uniform(0, 20.0, size=(n, num_boxes, 2))
    half_sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    boxes = np.concatenate(
        [centers - half_sizes, centers + half_sizes], axis=-1
    ).astype(np.float32)  # (n, num_boxes, 4)
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)

    common = dict(
        top_k=50, keep_top_k=keep_top_k, score_threshold=0.1, iou_threshold=0.5
    )
    model = _trt_batched_nms_model(
        n=n,
        num_boxes=num_boxes,
        num_classes=num_classes,
        background_label_id=-1,
        **common,
    )
    _, actual = _run_rewritten(model, boxes, scores)
    expected = reference_full_batch(boxes, scores, background_label_id=-1, **common)
    _assert_matches_reference(actual, expected)


# ---------------------------------------------------------------------------
# Part 6: scope restrictions (predicate must decline outside documented scope)
# ---------------------------------------------------------------------------


def test_declines_when_boxes_have_per_class_boxes():
    """``num_classes_or_1 > 1`` (statically known) is out of scope -- ONNX
    ``NonMaxSuppression`` has no notion of per-class boxes (header
    "Scope" section)."""
    num_boxes, num_classes, keep_top_k = 6, 3, 4
    model = _trt_batched_nms_model(
        n=1,
        num_boxes=num_boxes,
        num_classes=num_classes,
        keep_top_k=keep_top_k,
        boxes_class_dim=num_classes,
    )
    sim_model, ops = simplify_isolated_extra(model, PASS_NAME, check_n=0)
    assert ops["TRTBatchedNMS"] == 1, ops
    assert ops["NonMaxSuppression"] == 0, ops


def test_declines_when_batch_size_is_dynamic():
    """``N`` (batch size) not statically known is out of scope --
    ``runTransform`` unrolls a per-batch-item C++ loop at pass-build time
    (header "Scope" section)."""
    num_boxes, num_classes, keep_top_k = 6, 3, 4
    model = _trt_batched_nms_model(
        n=1,
        num_boxes=num_boxes,
        num_classes=num_classes,
        keep_top_k=keep_top_k,
        n_dyn=True,
    )
    sim_model, ops = simplify_isolated_extra(model, PASS_NAME, check_n=0)
    assert ops["TRTBatchedNMS"] == 1, ops
    assert ops["NonMaxSuppression"] == 0, ops
