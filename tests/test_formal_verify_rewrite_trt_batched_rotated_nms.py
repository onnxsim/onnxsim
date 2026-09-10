"""Formal check for RewriteTRTBatchedRotatedNMS (opt-in; onnxsim's own
``onnxsim/passes/rewrite_trt_batched_rotated_nms.h``) -- the rotated-box
sibling of ``rewrite_trt_batched_nms.h``, decomposing mmdeploy's
``TRTBatchedRotatedNMS`` op into a fixed (unrolled, never ``Loop``) subgraph
of standard ONNX ops. Unlike the axis-aligned op, ONNX has no primitive for
either half of this pass's job, so the pass implements BOTH from scratch:

  1. Rotated-box IoU via Sutherland-Hodgman polygon clipping + the shoelace
     area formula (the pass's header comment derives this in full detail,
     including a fixed-8-slot, mask-free vectorization of the classic
     variable-length-list clipping algorithm).
  2. Greedy NMS's "pick current best, suppress everything too close to it,
     repeat" loop, unrolled ``min(topK, keepTopK)`` times per class (never a
     data-dependent ``Loop``).

**Honesty about scope, mirroring this suite's own established precedent for
passes where a single monolithic symbolic proof is not realistic (see
``test_formal_verify_qoperator_quantize_softmax.py``'s own docstring for
this framing)**: general Sutherland-Hodgman polygon clipping is genuinely
not something Z3 can be pointed at wholesale (it is a case-split-heavy,
transcendental-function-laden geometric algorithm, not a closed-form
arithmetic expression). So this file's Z3 content is deliberately a set of
SMALL, SEPARATE, exactly-checkable facts, not one "IoU is correct" theorem:

  - The corner-computation rotation formula, confirmed against the standard
    CCW rotation-matrix convention at a few CONCRETE angles (0, +-pi/2, pi)
    where sin/cos are exactly rational -- Z3 cannot reason about sin/cos of
    a free symbolic angle, so this is a documented limitation, not an
    oversight (same convention this suite uses elsewhere for transcendental
    functions: concrete instantiation, not full symbolic reasoning).
  - A fully SYMBOLIC (no concrete angle needed) fact that survives for any
    theta: rotation preserves each corner's distance to the box center
    (follows purely from ``cos^2+sin^2=1``, no trig identity needed beyond
    that), which is exactly what makes the box's circumscribed-circle radius
    ``sqrt((w/2)^2+(h/2)^2)`` orientation-independent -- the fact the
    bounding-circle-separation lemma below relies on.
  - The shoelace area formula's sign/orientation convention: SYMBOLICALLY
    proved that the code's own corner order is CCW (signed shoelace sum on
    the un-rotated local corners is exactly ``+w*h``, matching the pass's
    header comment's own claim), plus one concrete hand-computed polygon
    (a trapezoid, checked via exact ``fractions.Fraction`` arithmetic, not
    Z3 -- plain rational arithmetic is the right tool for "does this one
    concrete polygon's area come out right", not an SMT solver).
  - Two exactly-solvable IoU special cases, at the "assembly formula" level
    (``IoU = intersection/(areaA+areaB-intersection)``) plus a geometric
    argument for the intersection value itself: (a) identical boxes have
    intersection area == own area (Sutherland-Hodgman clipping a convex
    polygon against an identical copy returns the polygon unchanged --
    stated as a premised geometric fact, not re-derived from the clipping
    algorithm), giving IoU == 1 exactly; (b) boxes whose circumscribed
    circles don't overlap have empty intersection (via the standard
    Euclidean triangle inequality, PREMISED as a hypothesis -- Z3 cannot
    derive facts about square roots of arbitrary reals on its own, so this
    mirrors this suite's convention of axiomizing transcendental/analytic
    facts rather than re-deriving them), giving IoU == 0 exactly.
  - The unrolled greedy-suppression ITERATION/SELECTION logic (ArgMax over a
    working array that self-masks the picked index and everything it
    suppresses) proved, for concrete small (num_boxes, keepTopK) pairs and
    FREE symbolic scores plus an OPAQUE free-boolean "IoU exceeds threshold"
    matrix per box pair, to select the exact same ordered sequence of boxes
    as an independently-structured rank-order reference (no working-array
    mutation, no ArgMax -- a static per-box rank plus a single forward
    suppression scan). Treating "is box i suppressed by box j" as an opaque
    boolean -- never computing it from IoU inside Z3 -- is exactly the
    "separate concerns" discipline this suite uses elsewhere: it checks the
    SELECTION LOGIC is right independent of how IoU itself is computed.

**What this file does NOT attempt in Z3**: general partial-overlap IoU
correctness (any two arbitrary rotated boxes). That is filled in by heavy,
independent differential testing instead: a from-scratch (fresh, NOT copied
from the C++'s own fixed-8-slot vectorized structure, and NOT copied from
``tests/test_trt_batched_rotated_nms.py``'s own textbook reference either,
though both necessarily implement the same well-known Sutherland-Hodgman +
shoelace algorithm) Python reimplementation of rotated IoU, cross-checked
against an independent Monte Carlo estimator, used as ground truth for (i) a
threshold-sweep bracketing technique that reads the REAL compiled pass's
internal IoU value indirectly (there is no standalone "RotatedIoU" op to
inspect directly -- see ``_box1_survives`` below) and (ii) a full
per-class-greedy-NMS-then-top-K-merge reference used against the real,
compiled-and-executed (onnxruntime, ``CPUExecutionProvider``) subgraph on
small multi-box scenes. Neither ``shapely`` nor ``cv2`` is available in this
environment (confirmed below) -- if either becomes available, cross-checking
against it would be a welcome addition, but this file's own from-scratch
implementation is written to stand on its own regardless.

**The box-format assumption -- flagged exactly as prominently as the
pass's own header comment flags it, no more confirmed here than there**:
``(cx, cy, w, h, theta)``, OpenCV-style rotated-rect convention. The header
comment is explicit that this is a DOCUMENTED ASSUMPTION about mmdeploy's
actual TensorRT-plugin export, not an independently-verified fact against a
real mmdeploy-exported model -- this file inherits that same uncertainty
verbatim and does not resolve it; every test below constructs its own boxes
in this assumed format and checks the pass's internal self-consistency (does
its output match ITS OWN documented algorithm on ITS OWN assumed format),
which cannot by itself confirm the format assumption is correct for a real
export.

**Known limitation, inherited from the pass itself and this suite's own
established non-goal for TensorRT-plugin-shaped passes**: exact tie-breaking
order for equal-or-near-equal scores/IoUs is not guaranteed to match any
particular reference. Full-pipeline tests below compare kept detections as
an unordered-by-score-then-class SET per batch item, mirroring both
``tests/test_trt_batched_nms.py`` and ``tests/test_trt_batched_rotated_nms.
py``'s own approach -- this is a SET-of-kept-detections claim, not a
bit-exact ordering claim.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

PASS_NAME = "rewrite_trt_batched_rotated_nms"

ort = pytest.importorskip("onnxruntime")


def test_shapely_and_cv2_are_not_available_here():
    """Documents (rather than assumes) the environment this file's own
    from-scratch IoU implementation must stand on its own in -- see module
    docstring. If this ever starts failing (one of these becomes
    importable), that is a green light to ADD an extra shapely/cv2
    cross-check somewhere below, not a reason to change this test."""
    with pytest.raises(ImportError):
        import shapely  # noqa: F401
    with pytest.raises(ImportError):
        import cv2  # noqa: F401


# ===========================================================================
# Z3 part 1: rotated-box geometry -- small, separate, exact lemmas.
# ===========================================================================


def test_corner_rotation_formula_matches_ccw_convention_at_concrete_angles():
    """The pass's own corner formula (``rewrite_trt_batched_rotated_nms.h``'s
    ``Corners()``): for a local point ``(lx, ly)`` (one of the 4
    ``(+-w/2, +-h/2)`` corners), the rotated-and-translated corner is::

        X = lx*cos(theta) - ly*sin(theta) + cx
        Y = lx*sin(theta) + ly*cos(theta) + cy

    Z3 cannot reason about sin/cos of a free symbolic ``theta`` -- this is
    exactly the same transcendental-function limitation this suite handles
    elsewhere via concrete instantiation, not full symbolic reasoning. At
    ``theta`` in ``{0, pi/2, pi, -pi/2}``, sin/cos are exactly rational, so
    each case becomes a fully decidable equality. The EXPECTED corner is
    derived independently of the code's own formula, from the elementary
    definition of a CCW rotation by that angle (identity / 90-deg-CCW /
    180-deg / 90-deg-CW respectively) -- confirming which rotation
    DIRECTION convention (``R(theta) = [[cos,-sin],[sin,cos]]``, standard
    CCW-positive) the code follows, not merely restating its own formula."""
    cx, cy, lx, ly = z3.Reals("cx cy lx ly")

    def code_corner(cos_t, sin_t):
        return (lx * cos_t - ly * sin_t + cx, lx * sin_t + ly * cos_t + cy)

    # theta = 0: identity.
    x0, y0 = code_corner(z3.RealVal(1), z3.RealVal(0))
    prove(z3.And(x0 == lx + cx, y0 == ly + cy), "theta=0 (identity)")

    # theta = pi/2: standard CCW 90-degree rotation, (x,y) -> (-y,x).
    x1, y1 = code_corner(z3.RealVal(0), z3.RealVal(1))
    prove(z3.And(x1 == -ly + cx, y1 == lx + cy), "theta=pi/2 (CCW 90)")

    # theta = pi: 180-degree rotation, (x,y) -> (-x,-y).
    x2, y2 = code_corner(z3.RealVal(-1), z3.RealVal(0))
    prove(z3.And(x2 == -lx + cx, y2 == -ly + cy), "theta=pi (180)")

    # theta = -pi/2: CW 90-degree rotation, (x,y) -> (y,-x). Confirms the
    # convention is genuinely signed (not accidentally direction-agnostic).
    x3, y3 = code_corner(z3.RealVal(0), z3.RealVal(-1))
    prove(z3.And(x3 == ly + cx, y3 == -lx + cy), "theta=-pi/2 (CW 90)")


def test_corner_distance_to_center_is_rotation_invariant_symbolically():
    """Fully SYMBOLIC (no concrete angle needed): for ANY ``theta`` (i.e. any
    ``(cos_t, sin_t)`` on the unit circle), the rotated corner's squared
    distance to the box center equals the un-rotated local point's own
    squared norm -- ``lx^2 + ly^2`` -- exactly. This is what makes a box's
    circumscribed-circle radius ``sqrt((w/2)^2+(h/2)^2)`` well-defined
    independent of rotation, which the bounding-circle-separation lemma
    below relies on. Provable by Z3 as a pure polynomial identity once
    ``cos^2+sin^2=1`` is given -- no trig beyond that Pythagorean identity is
    needed, so this genuinely does not require a concrete angle."""
    lx, ly, cos_t, sin_t = z3.Reals("lx ly cos_t sin_t")
    pythagorean = cos_t * cos_t + sin_t * sin_t == 1
    rx = lx * cos_t - ly * sin_t
    ry = lx * sin_t + ly * cos_t
    prove(
        z3.Implies(pythagorean, rx * rx + ry * ry == lx * lx + ly * ly),
        "rotation preserves corner-to-center distance",
    )


def test_box_corner_order_is_ccw_matching_the_headers_own_claim():
    """The pass's header comment claims its corner order -- ``(dx,dy),
    (-dx,dy),(-dx,-dy),(dx,-dy)`` for ``dx=w/2,dy=h/2`` -- traces the
    rectangle CCW, "verified via the shoelace formula on the un-rotated
    rectangle giving a positive sum, i.e. +wh". This is a fully SYMBOLIC
    (free ``dx,dy > 0``) polynomial fact -- no rotation or trig involved at
    all, since orientation is a property of the un-rotated local corners
    (rotation, being a proper/orientation-preserving linear map, cannot
    change CCW-ness anyway -- confirmed separately by this same signed sum
    being degree-0 in theta once expanded, though this test only needs the
    un-rotated case the header itself appeals to)."""
    dx, dy = z3.Reals("dx dy")
    pts = [(dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)]
    signed_sum = z3.RealVal(0)
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        signed_sum = signed_sum + (x0 * y1 - x1 * y0)
    signed_area = signed_sum / 2
    prove(
        z3.Implies(z3.And(dx > 0, dy > 0), signed_area == (2 * dx) * (2 * dy)),
        "corner order is CCW (signed shoelace sum == +w*h)",
    )


def test_shoelace_area_matches_hand_computed_trapezoid():
    """One concrete, non-trivial (non-rectangular) polygon, checked with
    exact rational arithmetic (``fractions.Fraction`` -- not Z3: this is
    "does this one concrete polygon's shoelace sum equal its hand-derived
    area", ordinary exact arithmetic is the right tool, not an SMT solver).
    Trapezoid, CCW, parallel sides on y=0 and y=3: (0,0),(6,0),(4,3),(1,3).
    Bottom side length 6, top side length 3 (from x=1 to x=4), height 3 ->
    textbook trapezoid area = (6+3)/2 * 3 = 27/2, independent of shoelace."""
    poly = [
        (Fraction(0), Fraction(0)),
        (Fraction(6), Fraction(0)),
        (Fraction(4), Fraction(3)),
        (Fraction(1), Fraction(3)),
    ]
    total = Fraction(0)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    shoelace_area = abs(total) / 2
    trapezoid_area = Fraction(6 + 3, 2) * 3
    assert shoelace_area == trapezoid_area == Fraction(27, 2)


def test_identical_boxes_have_iou_exactly_one():
    """Special case (a): two identical boxes. Geometric argument (premised,
    not re-derived from the clipping algorithm itself -- a standard fact
    about convex polygon clipping): Sutherland-Hodgman-clipping a convex
    polygon against an identical copy of itself returns the polygon
    unchanged (every one of the copy's edges has the whole original polygon
    on its inside, since the two polygons coincide exactly), so
    ``intersection_area == areaA == areaB``. Given that as a hypothesis, the
    IoU ASSEMBLY formula itself (``intersection/(areaA+areaB-intersection)``)
    algebraically forces IoU == 1 whenever the shared area is positive --
    this part Z3 checks directly, no geometry left to reason about."""
    area = z3.Real("area")
    intersection = area  # premised: identical boxes clip to themselves
    denom = area + area - intersection
    iou = intersection / denom
    prove(z3.Implies(area > 0, iou == 1), "identical boxes -> IoU == 1")


def test_bounding_circle_separated_boxes_have_iou_exactly_zero():
    """Special case (b): boxes whose circumscribed circles don't overlap.
    Part 1 (this test): any point ``p`` in box A's circumscribed disk and
    any point ``q`` in box B's circumscribed disk are DISTINCT once the two
    disks are separated (center distance exceeds the sum of radii) -- via
    the standard Euclidean triangle inequality for distances, PREMISED here
    as a hypothesis (Z3 cannot derive facts about square roots of arbitrary
    reals on its own; this mirrors this suite's convention of axiomizing
    transcendental/analytic facts rather than re-deriving them from
    scratch). Combined with the (separately, symbolically, proved above)
    fact that every corner of a box lies exactly on its circumscribed circle
    for ANY rotation, and the standard convexity fact that a disk contains
    the convex hull of any points on its boundary (so the whole box, being
    the convex hull of 4 such corners, lies within the disk) -- both stated
    here as premises, not re-derived -- circle separation implies the two
    boxes share no point at all, so intersection_area == 0 exactly. Given
    that, the same IoU assembly-formula argument as the identical-boxes case
    above forces IoU == 0 (as long as the boxes aren't BOTH degenerate
    zero-area, i.e. ``areaA + areaB > 0``)."""
    d_pA, d_qB, d_AB, d_pq, rA, rB = z3.Reals("d_pA d_qB d_AB d_pq rA rB")
    triangle_inequality = d_AB <= d_pA + d_pq + d_qB  # premised Euclidean fact
    hyps = z3.And(
        triangle_inequality,
        d_pA <= rA,  # p is within A's circumscribed disk
        d_qB <= rB,  # q is within B's circumscribed disk
        d_AB > rA + rB,  # the two disks are separated
        rA >= 0,
        rB >= 0,
        d_pq >= 0,
    )
    prove(z3.Implies(hyps, d_pq > 0), "circle-separated disks share no point")

    # Assembly-formula half: given intersection_area == 0 (from the argument
    # above) and areaA + areaB > 0, IoU == 0.
    area_a, area_b = z3.Reals("area_a area_b")
    intersection = z3.RealVal(0)
    denom = area_a + area_b - intersection
    iou = z3.If(denom < Fraction(1, 10**9), z3.RealVal(0), intersection / denom)
    prove(
        z3.Implies(area_a + area_b > 0, iou == 0),
        "circle-separated boxes -> IoU == 0",
    )


# ===========================================================================
# Z3 part 2: the unrolled greedy-suppression ITERATION/SELECTION logic.
# ===========================================================================
#
# Modeled abstractly per the header comment's own derivation: repeated
# ArgMax over a "working" array that, once a box is picked, gets that box's
# own slot AND every slot it suppresses masked out for all later iterations.
# "Is box i suppressed by box j" (i.e. "IoU(i,j) > iouThreshold") is an
# OPAQUE free boolean per unordered pair here -- never computed from IoU --
# so this checks the selection/iteration logic independent of how IoU
# itself is computed, matching this suite's "separate concerns" discipline.


def _argmax_over_active(values, active, n):
    """Nested-ITE fold computing (index of max `values[i]` among `active[i]`,
    whether any candidate was active at all). Well-defined even when no
    candidate is active (deterministically falls through to index n-1),
    which matters below only insofar as the "avail" hypothesis excludes that
    branch from the equivalence claim -- see its comment."""
    best_idx = z3.IntVal(0)
    best_val = values[0]
    best_active = active[0]
    for c in range(1, n):
        take = z3.Or(z3.Not(best_active), z3.And(active[c], values[c] > best_val))
        best_idx = z3.If(take, z3.IntVal(c), best_idx)
        best_val = z3.If(take, values[c], best_val)
        best_active = z3.Or(best_active, active[c])
    return best_idx, best_active


def _greedy_selection_equivalence_claim(n, k):
    """Builds the Z3 claim that, for `n` candidate boxes and `k` greedy
    picks, method A (mirrors the pass's own unrolled-ArgMax-with-masking
    construction) and method B (an independently-structured, non-mutating
    rank-order reference: rank every box once by score, then make a single
    forward pass keeping each rank-ordered box unless an earlier KEPT box
    suppresses it) pick the exact same ordered sequence of indices."""
    scores = z3.Reals(" ".join(f"s{i}" for i in range(n)))
    distinct_scores = z3.Distinct(*scores)

    # Opaque, symmetric "IoU(i,j) > iouThreshold" predicate -- IoU is
    # symmetric, so this predicate must be too; the pass's own self-mask
    # (an explicit index-equality check, not reliance on self-IoU == 1) is
    # modeled the same way here, separately from `over`.
    raw_over = z3.Function("raw_over", z3.IntSort(), z3.IntSort(), z3.BoolSort())

    def over(i, j):
        return z3.If(i == j, False, z3.Or(raw_over(i, j), raw_over(j, i)))

    # ---- Method A: repeated ArgMax over a self-masking working array. ----
    active = [z3.BoolVal(True) for _ in range(n)]
    method_a_kept = []
    avail = []  # avail[k] == "a real (active) candidate existed at step k"
    for _ in range(k):
        idx, best_active = _argmax_over_active(scores, active, n)
        method_a_kept.append(idx)
        avail.append(best_active)
        active = [
            z3.And(
                active[m], z3.Not(z3.Or(z3.IntVal(m) == idx, over(idx, z3.IntVal(m))))
            )
            for m in range(n)
        ]

    # ---- Method B: static rank + single forward suppression scan. ----
    def rank(i):
        return sum(z3.If(scores[j] > scores[i], 1, 0) for j in range(n) if j != i)

    ranks = [rank(i) for i in range(n)]

    def inv_rank(r):
        expr = z3.IntVal(n - 1)
        for i in reversed(range(n - 1)):
            expr = z3.If(ranks[i] == r, z3.IntVal(i), expr)
        return expr

    order = [inv_rank(r) for r in range(n)]  # order[0] = best-scoring index

    suppressed_b = [z3.BoolVal(False) for _ in range(n)]
    kept_count = z3.IntVal(0)
    kept_list = [z3.IntVal(-1) for _ in range(k)]
    for r in range(n):
        idx_r = order[r]
        is_suppressed = z3.Or(
            *[z3.And(z3.IntVal(m) == idx_r, suppressed_b[m]) for m in range(n)]
        )
        will_keep = z3.And(z3.Not(is_suppressed), kept_count < k)
        kept_list = [
            z3.If(z3.And(will_keep, kept_count == s), idx_r, kept_list[s])
            for s in range(k)
        ]
        suppressed_b = [
            z3.Or(suppressed_b[m], z3.And(will_keep, over(idx_r, z3.IntVal(m))))
            for m in range(n)
        ]
        kept_count = z3.If(will_keep, kept_count + 1, kept_count)

    # "avail[k]" (a real candidate existed at each of method A's k steps) is
    # a genuine, load-bearing non-degeneracy hypothesis, not a formality --
    # confirmed empirically (see this file's own development notes / the
    # task's differential-testing methodology): dropping it, or dropping
    # `distinct_scores`, each independently yields a real counterexample
    # (score/over-matrix assignment where the two methods disagree), so both
    # premises are doing real work, not vacuously true. The scenario they
    # exclude -- suppression exhausting every remaining candidate before k
    # picks are made -- is a real but harmless edge case in the actual pass
    # too (ArgMax then returns among sentinel-valued (-1e9) entries, which
    # the pass's own later `valid = scores > kSentinelCheck` filtering
    # discards regardless of which sentinel-valued index gets picked).
    domain = z3.And(distinct_scores, *avail)
    return z3.Implies(
        domain, z3.And(*[method_a_kept[i] == kept_list[i] for i in range(k)])
    )


@pytest.mark.parametrize("n,k", [(4, 2), (4, 3), (3, 2)])
def test_unrolled_greedy_selection_matches_rank_order_reference(n, k):
    prove(
        _greedy_selection_equivalence_claim(n, k),
        f"unrolled greedy selection (n={n}, k={k}) diverges from the "
        "rank-order reference",
    )


def _greedy_selection_claim_with_dropped_hypothesis(n, k, drop):
    """Rebuilds the same claim as `_greedy_selection_equivalence_claim`, but
    with one domain hypothesis removed (`drop` in {"distinct", "avail"}) --
    used only to confirm each hypothesis is load-bearing, not to prove
    anything."""
    scores = z3.Reals(" ".join(f"s{i}" for i in range(n)))
    distinct_scores = z3.Distinct(*scores)
    raw_over = z3.Function("raw_over", z3.IntSort(), z3.IntSort(), z3.BoolSort())

    def over(i, j):
        return z3.If(i == j, False, z3.Or(raw_over(i, j), raw_over(j, i)))

    active = [z3.BoolVal(True) for _ in range(n)]
    method_a_kept, avail = [], []
    for _ in range(k):
        idx, best_active = _argmax_over_active(scores, active, n)
        method_a_kept.append(idx)
        avail.append(best_active)
        active = [
            z3.And(
                active[m], z3.Not(z3.Or(z3.IntVal(m) == idx, over(idx, z3.IntVal(m))))
            )
            for m in range(n)
        ]

    def rank(i):
        return sum(z3.If(scores[j] > scores[i], 1, 0) for j in range(n) if j != i)

    ranks = [rank(i) for i in range(n)]

    def inv_rank(r):
        expr = z3.IntVal(n - 1)
        for i in reversed(range(n - 1)):
            expr = z3.If(ranks[i] == r, z3.IntVal(i), expr)
        return expr

    order = [inv_rank(r) for r in range(n)]
    suppressed_b = [z3.BoolVal(False) for _ in range(n)]
    kept_count = z3.IntVal(0)
    kept_list = [z3.IntVal(-1) for _ in range(k)]
    for r in range(n):
        idx_r = order[r]
        is_suppressed = z3.Or(
            *[z3.And(z3.IntVal(m) == idx_r, suppressed_b[m]) for m in range(n)]
        )
        will_keep = z3.And(z3.Not(is_suppressed), kept_count < k)
        kept_list = [
            z3.If(z3.And(will_keep, kept_count == s), idx_r, kept_list[s])
            for s in range(k)
        ]
        suppressed_b = [
            z3.Or(suppressed_b[m], z3.And(will_keep, over(idx_r, z3.IntVal(m))))
            for m in range(n)
        ]
        kept_count = z3.If(will_keep, kept_count + 1, kept_count)

    hyps = []
    if drop != "distinct":
        hyps.append(distinct_scores)
    if drop != "avail":
        hyps.extend(avail)
    domain = z3.And(*hyps) if hyps else z3.BoolVal(True)
    return z3.Implies(
        domain, z3.And(*[method_a_kept[i] == kept_list[i] for i in range(k)])
    )


@pytest.mark.parametrize("drop", ["distinct", "avail"])
def test_greedy_selection_hypotheses_are_load_bearing(drop):
    """Confirms each domain hypothesis in the equivalence proof above
    genuinely excludes real counterexamples (i.e. is not vacuously
    unnecessary): dropping either one, on its own, must yield a genuine
    counterexample -- `prove` succeeding here would mean that hypothesis was
    never doing any work."""
    claim = _greedy_selection_claim_with_dropped_hypothesis(4, 2, drop)
    solver = z3.Solver()
    solver.add(z3.Not(claim))
    assert solver.check() == z3.sat, f"dropping {drop!r} did not weaken the claim"


# ===========================================================================
# Independent (from-scratch) rotated-box IoU reference.
# ===========================================================================
#
# Deliberately NOT the pass's own fixed-8-slot, mask-free, vectorized
# construction (see the pass's header comment) -- a plain, textbook,
# variable-length-list Sutherland-Hodgman clip plus shoelace area, written
# fresh for this file. Also deliberately not copied from `tests/
# test_trt_batched_rotated_nms.py`'s own reference, though both necessarily
# implement the same well-known algorithm.


def _box_corners(box):
    cx, cy, w, h, theta = box
    hw, hh = w / 2.0, h / 2.0
    local = ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh))
    ct, st = math.cos(theta), math.sin(theta)
    return [(cx + lx * ct - ly * st, cy + lx * st + ly * ct) for lx, ly in local]


def _clip_against_one_edge(poly, edge_start, edge_end):
    """Keeps the portion of convex polygon `poly` on the CCW-interior side
    of the directed edge `edge_start -> edge_end`."""
    if not poly:
        return poly
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]

    def side(p):
        return ex * (p[1] - edge_start[1]) - ey * (p[0] - edge_start[0])

    output = []
    n = len(poly)
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        s_cur, s_prev = side(cur), side(prev)
        if s_cur >= 0:
            if s_prev < 0:
                t = s_prev / (s_prev - s_cur)
                output.append(
                    (prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]))
                )
            output.append(cur)
        elif s_prev >= 0:
            t = s_prev / (s_prev - s_cur)
            output.append(
                (prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]))
            )
    return output


def _sutherland_hodgman(subject, clip):
    poly = list(subject)
    n = len(clip)
    for i in range(n):
        poly = _clip_against_one_edge(poly, clip[i], clip[(i + 1) % n])
    return poly


def _shoelace(poly):
    if len(poly) < 3:
        return 0.0
    total = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _my_iou(box_a, box_b):
    corners_a, corners_b = _box_corners(box_a), _box_corners(box_b)
    intersection = _sutherland_hodgman(corners_a, corners_b)
    inter_area = _shoelace(intersection)
    area_a, area_b = _shoelace(corners_a), _shoelace(corners_b)
    denom = area_a + area_b - inter_area
    return 0.0 if denom < 1e-9 else inter_area / denom


def _points_in_convex_poly(pts, poly):
    inside = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (pts[:, 1] - y0) - (y1 - y0) * (pts[:, 0] - x0)
        inside &= cross >= -1e-9
    return inside


def _monte_carlo_iou(box_a, box_b, rng, n=300000):
    """Fully independent (no shared code with `_my_iou`'s clipping/shoelace
    machinery) sanity check: uniform sampling over the union's bounding
    box, classified by a plain half-plane point-in-convex-polygon test."""
    corners_a = np.asarray(_box_corners(box_a))
    corners_b = np.asarray(_box_corners(box_b))
    allc = np.vstack([corners_a, corners_b])
    lo, hi = allc.min(0), allc.max(0)
    pts = rng.uniform(lo, hi, size=(n, 2))
    in_a = _points_in_convex_poly(pts, corners_a)
    in_b = _points_in_convex_poly(pts, corners_b)
    inter, union = np.sum(in_a & in_b), np.sum(in_a | in_b)
    return float(inter) / float(union) if union > 0 else 0.0


def test_my_iou_matches_known_exact_values():
    """Sanity-checks `_my_iou` itself against closed-form values before
    trusting it as ground truth for anything below."""
    # Axis-aligned: two 4x2 rectangles, x-overlap [0,2], full y-overlap.
    assert _my_iou(
        (0.0, 0.0, 4.0, 2.0, 0.0), (2.0, 0.0, 4.0, 2.0, 0.0)
    ) == pytest.approx(4.0 / 12.0, abs=1e-9)
    # Fully inside, same center/orientation: iou == small_area/big_area.
    theta = math.radians(25)
    assert _my_iou(
        (0.0, 0.0, 8.0, 8.0, theta), (0.0, 0.0, 2.0, 3.0, theta)
    ) == pytest.approx(6.0 / 64.0, abs=1e-9)
    # Identical boxes: iou == 1 exactly.
    box = (1.0, -2.0, 3.0, 4.0, math.radians(37))
    assert _my_iou(box, box) == pytest.approx(1.0, abs=1e-9)
    # Bounding circles clearly separated: iou == 0 exactly.
    box0 = (0.0, 0.0, 4.0, 2.0, math.radians(12))
    box1 = (15.0, 0.0, 3.0, 3.0, math.radians(80))
    r0, r1 = math.hypot(2.0, 1.0), math.hypot(1.5, 1.5)
    assert math.hypot(15.0, 0.0) > r0 + r1  # the separation certificate itself
    assert _my_iou(box0, box1) == 0.0
    # Rotating one box by 90 degrees on an otherwise-identical axis-aligned
    # pair genuinely changes the IoU (swaps its effective footprint) -- this
    # is the same rotation-sensitivity exploited by the differential test
    # below; if a bug ever made the code ignore theta, this would still (by
    # coincidence) be 1/3 for theta=0 but would become 1.0, not 1/3, for
    # theta=pi/2.
    assert _my_iou(
        (0.0, 0.0, 4.0, 2.0, 0.0), (0.0, 0.0, 4.0, 2.0, 0.0)
    ) == pytest.approx(1.0, abs=1e-9)
    assert _my_iou(
        (0.0, 0.0, 4.0, 2.0, 0.0), (0.0, 0.0, 4.0, 2.0, math.pi / 2)
    ) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_my_iou_cross_checked_against_monte_carlo_on_partial_overlaps():
    rng = np.random.default_rng(7)
    configs = [
        (
            (0.0, 0.0, 4.0, 2.0, math.radians(20)),
            (1.5, 0.5, 3.0, 2.0, math.radians(-35)),
        ),
        (
            (0.0, 0.0, 5.0, 3.0, math.radians(50)),
            (2.0, -1.0, 4.0, 2.0, math.radians(10)),
        ),
        ((0.0, 0.0, 3.0, 3.0, 0.0), (0.0, 0.0, 3.0, 3.0, math.radians(45))),
    ]
    for box_a, box_b in configs:
        ref = _my_iou(box_a, box_b)
        mc = _monte_carlo_iou(box_a, box_b, rng)
        assert ref == pytest.approx(mc, abs=0.01), (box_a, box_b, ref, mc)
        assert 0.02 < ref < 0.98, (box_a, box_b, ref)  # genuinely partial


# ===========================================================================
# Model construction (onnx.parser, per CLAUDE.md) and pass-running helpers.
# ===========================================================================
#
# The custom domain-qualified op parses fine directly via the ONNX text
# format (a domain-qualified op name plus an extra `opset_import` entry for
# that domain) -- no `onnx.helper.make_node` fallback is needed here.


def _model(body, opset=13, ir_version=10, extra_opsets=""):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_opsets}]
        >
        {body}
        """
    )


def _rotated_nms_model(
    n,
    num_boxes,
    num_classes,
    keep_top_k,
    top_k=200,
    score_threshold=0.05,
    iou_threshold=0.5,
    background_label_id=-1,
):
    return _model(
        f"""
        agraph (float[{n},{num_boxes},1,5] boxes,
                float[{n},{num_boxes},{num_classes}] scores)
              => (int32[{n},1] num_detections,
                  float[{n},{keep_top_k},5] nmsed_boxes,
                  float[{n},{keep_top_k}] nmsed_scores,
                  float[{n},{keep_top_k}] nmsed_classes)
        {{
          num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = mmdeploy.TRTBatchedRotatedNMS
            <background_label_id={background_label_id}, num_classes={num_classes},
             topK={top_k}, keepTopK={keep_top_k}, scoreThreshold={score_threshold},
             iouThreshold={iou_threshold}>
            (boxes, scores)
        }}
        """,
        extra_opsets=', "mmdeploy": 1',
    )


def _simplify(model, check_n=0):
    # check_n=0: onnxsim's own random-input equivalence check cannot run the
    # ORIGINAL model (it contains the custom, kernel-less `mmdeploy` domain
    # op), mirroring this suite's own established `check_n=0` convention for
    # every other opt-in custom-op-firing pass (see e.g.
    # test_formal_verify_double_quantization.py, test_formal_verify_
    # dynamic_quantize_matmul.py). The differential tests below independently
    # execute the REWRITTEN subgraph via onnxruntime and check it against
    # this file's own reference, which is the real correctness check here.
    return onnxsim.simplify(model, check_n=check_n, extra_optimizers=[PASS_NAME])


def _run(model, boxes, scores):
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" not in op_types, op_types
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"boxes": boxes, "scores": scores})


def test_extra_optimizers_required_to_fire():
    """Confirms registration: the op survives a plain (default-passes-only)
    `simplify()` call, and is named in the opt-in list."""
    model = _rotated_nms_model(1, 4, 1, keep_top_k=2)
    simplified, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types
    sim2, ops = simplify_isolated_extra(model, PASS_NAME, check_n=0)
    assert ops["TRTBatchedRotatedNMS"] == 0
    assert ops["ArgMax"] > 0


# ===========================================================================
# Threshold-sweep harness: reads the real pass's internal IoU indirectly.
# ===========================================================================
#
# There is no standalone "RotatedIoU" op the pass constructs -- the
# computation lives entirely inline in the rewritten subgraph. Instead:
# build the smallest possible instance that still exercises it in an
# observable way (2 boxes, 1 class, keepTopK=topK=2, box 0 scored higher
# than box 1). Under greedy NMS, box 1 survives (num_detections==2) iff
# IoU(box0,box1) <= iouThreshold. Sweeping iouThreshold around an
# independently-computed reference IoU therefore brackets the pass's own
# internal value to arbitrary precision.


def _box1_survives(box0, box1, iou_threshold):
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
    scores = np.array([[[1.0], [0.9]]], dtype=np.float32)
    outputs = _run(model, boxes, scores)
    numdet = int(outputs[0][0, 0])
    assert numdet in (1, 2)
    return numdet == 2


def _assert_brackets(box0, box1, ref_iou, margin=0.03):
    lo = max(0.0, ref_iou - margin)
    hi = min(0.999, ref_iou + margin)
    assert not _box1_survives(box0, box1, lo), (
        f"expected suppression at iouThreshold={lo} (ref IoU={ref_iou})"
    )
    assert _box1_survives(box0, box1, hi), (
        f"expected box1 kept at iouThreshold={hi} (ref IoU={ref_iou})"
    )


def test_pass_iou_axis_aligned_matches_ordinary_rectangle_iou():
    box0 = (0.0, 0.0, 4.0, 2.0, 0.0)
    box1 = (2.0, 0.0, 4.0, 2.0, 0.0)
    ref_iou = _my_iou(box0, box1)
    assert ref_iou == pytest.approx(1.0 / 3.0, abs=1e-9)
    _assert_brackets(box0, box1, ref_iou)


def test_pass_iou_rotated_partial_overlap():
    rng = np.random.default_rng(3)
    configs = [
        (
            (0.0, 0.0, 4.0, 2.0, math.radians(20)),
            (1.5, 0.5, 3.0, 2.0, math.radians(-35)),
        ),
        (
            (0.0, 0.0, 5.0, 3.0, math.radians(50)),
            (2.0, -1.0, 4.0, 2.0, math.radians(10)),
        ),
    ]
    for box0, box1 in configs:
        ref_iou = _my_iou(box0, box1)
        mc_iou = _monte_carlo_iou(box0, box1, rng)
        assert ref_iou == pytest.approx(mc_iou, abs=0.01)
        assert 0.02 < ref_iou < 0.9
        _assert_brackets(box0, box1, ref_iou)


def test_pass_iou_fully_inside():
    theta = math.radians(25)
    big = (0.0, 0.0, 8.0, 8.0, theta)
    small = (0.0, 0.0, 2.0, 3.0, theta)
    ref_iou = _my_iou(big, small)
    assert ref_iou == pytest.approx(6.0 / 64.0, abs=1e-9)
    _assert_brackets(big, small, ref_iou)


def test_pass_iou_identical_boxes_is_essentially_exactly_one():
    """Special case (a) against the real compiled subgraph: box1 must be
    suppressed well below IoU 1 (0.95) and must survive at any threshold
    that can never be exceeded by a true IoU (1.05) -- avoids fragility to
    tiny floating-point noise around the theoretical maximum while still
    being a tight, meaningful bracket."""
    box = (1.0, -2.0, 3.0, 4.0, math.radians(37))
    assert not _box1_survives(box, box, 0.95)
    assert _box1_survives(box, box, 1.05)


def test_pass_iou_bounding_circle_separated_is_exactly_zero():
    """Special case (b) against the real compiled subgraph."""
    box0 = (0.0, 0.0, 4.0, 2.0, math.radians(12))
    box1 = (15.0, 0.0, 3.0, 3.0, math.radians(80))
    r0, r1 = math.hypot(2.0, 1.0), math.hypot(1.5, 1.5)
    assert math.hypot(15.0, 0.0) > r0 + r1  # separation certificate
    assert _my_iou(box0, box1) == 0.0
    assert _box1_survives(box0, box1, 0.0)


def test_pass_genuinely_uses_rotation_not_just_axis_aligned_bbox():
    """The sharpest possible rotation-sensitivity check: two boxes share
    the exact same center and (w, h), differing ONLY in that box1 is
    rotated 90 degrees. A correct rotated-IoU implementation must see this
    as *different* footprints (IoU == 1/3, exactly the same value as the
    plain axis-aligned example above -- box1's rotated footprint exactly
    swaps into a 2-wide x 4-tall axis-aligned rectangle centered at the
    same point) -- NOT identical boxes (which a `theta`-ignoring bug would
    compute, giving IoU == 1.0 instead). At iouThreshold=0.5 these two
    (correct vs theta-ignoring) computations produce OPPOSITE keep/suppress
    decisions, so this is a real, discriminating test of rotation handling,
    not merely of the rotation-formula lemma in isolation."""
    box0 = (0.0, 0.0, 4.0, 2.0, 0.0)
    box1 = (0.0, 0.0, 4.0, 2.0, math.pi / 2)
    ref_iou = _my_iou(box0, box1)
    assert ref_iou == pytest.approx(1.0 / 3.0, abs=1e-9)
    # A theta-ignoring bug would instead see two identical boxes (IoU==1.0),
    # which would be suppressed at threshold 0.5 -- the correct answer must
    # NOT be suppressed here.
    assert _box1_survives(box0, box1, 0.5)
    assert not _box1_survives(box0, box1, 0.2)


# ===========================================================================
# Full end-to-end differential tests: independent reference implementation
# of the whole pass (per-class greedy rotated NMS via `_my_iou`, capped at
# min(topK,keepTopK), then per-batch top-K merge across classes) against the
# real compiled-and-executed subgraph.
# ===========================================================================


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
            if _my_iou(boxes_c[i], boxes_c[j]) > iou_threshold:
                suppressed[pos2] = True
    return keep


def _reference_full_pass(
    boxes,
    scores,
    background_label_id,
    top_k,
    keep_top_k,
    score_threshold,
    iou_threshold,
):
    """`boxes`: (N,num_boxes,5) float32 (class-agnostic). `scores`:
    (N,num_boxes,num_classes) float32."""
    n, num_boxes, num_classes = scores.shape
    out_boxes = np.zeros((n, keep_top_k, 5), dtype=np.float32)
    out_scores = np.zeros((n, keep_top_k), dtype=np.float32)
    out_classes = np.full((n, keep_top_k), -1.0, dtype=np.float32)
    out_numdet = np.zeros((n, 1), dtype=np.int64)
    per_class_cap = min(top_k, keep_top_k)

    for b in range(n):
        pooled = []
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


def _run_full(model, boxes_5d, scores):
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" not in op_types, op_types
    assert "ArgMax" in op_types, op_types
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"boxes": boxes_5d, "scores": scores})


def _assert_same_detection_sets(actual, expected, atol=1e-3):
    """SET-of-kept-detections comparison, per batch item -- tolerant of
    tie-break ordering (see module docstring's non-goal note)."""
    a_numdet, a_boxes, a_scores, a_classes = actual
    e_numdet, e_boxes, e_scores, e_classes = expected
    n = e_numdet.shape[0]
    for b in range(n):
        assert int(a_numdet[b, 0]) == int(e_numdet[b, 0]), (
            f"batch {b}: num_detections {int(a_numdet[b, 0])} vs {int(e_numdet[b, 0])}"
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
        for (a_s, a_c, a_bx), (e_s, e_c, e_bx) in zip(a_rows, e_rows):
            assert a_c == pytest.approx(e_c)
            assert a_s == pytest.approx(e_s, abs=atol)
            np.testing.assert_allclose(a_bx, e_bx, atol=atol)

        if cnt < a_boxes.shape[1]:
            np.testing.assert_allclose(a_boxes[b, cnt:], 0.0)
            np.testing.assert_allclose(a_scores[b, cnt:], 0.0)
            np.testing.assert_allclose(a_classes[b, cnt:], -1.0)


def test_full_pass_matches_reference_including_axis_aligned_bbox_rotated_pair():
    """6 boxes, 2 classes, varied angles -- includes (indices 0,1) a pair
    that is axis-aligned-bounding-box-overlapping but genuinely differently
    rotated (same center/size, 45-degree relative rotation, true IoU ~0.71
    -- see `test_my_iou_cross_checked_against_monte_carlo_on_partial_
    overlaps`), so this scene cannot pass by accident if the pipeline were
    to ignore rotation somewhere along the way."""
    boxes = np.array(
        [
            [0.0, 0.0, 3.0, 3.0, 0.0],
            [0.0, 0.0, 3.0, 3.0, math.radians(45)],
            [10.0, 10.0, 2.0, 2.0, math.radians(15)],
            [10.5, 10.2, 2.0, 2.0, math.radians(-20)],
            [-8.0, 4.0, 3.0, 1.5, math.radians(70)],
            [30.0, -5.0, 2.0, 4.0, math.radians(10)],
        ],
        dtype=np.float32,
    )
    scores = np.array(
        [[0.95, 0.4], [0.80, 0.3], [0.85, 0.6], [0.60, 0.7], [0.55, 0.2], [0.50, 0.9]],
        dtype=np.float32,
    )[None]
    boxes_5d = boxes[None, :, None, :]
    n, num_boxes, num_classes = 1, 6, 2
    keep_top_k, top_k = 4, 6

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    actual = _run_full(model, boxes_5d, scores)
    expected = _reference_full_pass(
        boxes[None],
        scores,
        -1,
        top_k,
        keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detection_sets(actual, expected)

    # The rotated-overlap pair (box0 score 0.95, box1 score 0.80, both class
    # 0): box1 must be suppressed by box0 -- confirms this scene actually
    # exercises the suppression path (true IoU ~0.71 > iouThreshold 0.5),
    # not merely a coincidentally-passing trivial case where nothing gets
    # suppressed at all.
    cnt = int(actual[0][0, 0])
    kept_pairs = {
        (round(float(actual[2][0][k]), 3), int(actual[3][0][k])) for k in range(cnt)
    }
    assert (round(0.95, 3), 0) in kept_pairs  # box0 kept
    assert (round(0.80, 3), 0) not in kept_pairs  # box1 suppressed by box0


def test_full_pass_matches_reference_multiple_batch_items():
    rng = np.random.default_rng(11)
    n, num_boxes, num_classes, keep_top_k, top_k = 2, 8, 2, 3, 5
    centers = rng.uniform(0, 10.0, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    thetas = rng.uniform(-math.pi, math.pi, size=(n, num_boxes, 1))
    boxes = np.concatenate([centers, sizes, thetas], axis=-1).astype(np.float32)
    boxes_5d = boxes[:, :, None, :]
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    actual = _run_full(model, boxes_5d, scores)
    expected = _reference_full_pass(
        boxes, scores, -1, top_k, keep_top_k, score_threshold=0.05, iou_threshold=0.5
    )
    _assert_same_detection_sets(actual, expected)


def test_full_pass_background_label_id_excludes_that_class():
    rng = np.random.default_rng(12)
    n, num_boxes, num_classes, keep_top_k, top_k = 1, 8, 3, 4, 6
    background_label_id = 1
    centers = rng.uniform(0, 10.0, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    thetas = rng.uniform(-math.pi, math.pi, size=(n, num_boxes, 1))
    boxes = np.concatenate([centers, sizes, thetas], axis=-1).astype(np.float32)
    boxes_5d = boxes[:, :, None, :]
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)

    model = _rotated_nms_model(
        n,
        num_boxes,
        num_classes,
        keep_top_k,
        top_k=top_k,
        background_label_id=background_label_id,
    )
    actual = _run_full(model, boxes_5d, scores)
    expected = _reference_full_pass(
        boxes,
        scores,
        background_label_id,
        top_k,
        keep_top_k,
        score_threshold=0.05,
        iou_threshold=0.5,
    )
    _assert_same_detection_sets(actual, expected)
    assert not np.any(actual[3] == float(background_label_id))


def test_full_pass_fewer_detections_than_keep_top_k_exercises_padding():
    rng = np.random.default_rng(13)
    n, num_boxes, num_classes = 1, 5, 2
    keep_top_k, top_k = 12, 12  # far more than num_boxes*num_classes can supply
    centers = rng.uniform(0, 10.0, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    thetas = rng.uniform(-math.pi, math.pi, size=(n, num_boxes, 1))
    boxes = np.concatenate([centers, sizes, thetas], axis=-1).astype(np.float32)
    boxes_5d = boxes[:, :, None, :]
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)

    model = _rotated_nms_model(
        n, num_boxes, num_classes, keep_top_k, top_k=top_k, score_threshold=0.3
    )
    actual = _run_full(model, boxes_5d, scores)
    expected = _reference_full_pass(
        boxes, scores, -1, top_k, keep_top_k, score_threshold=0.3, iou_threshold=0.5
    )
    _assert_same_detection_sets(actual, expected)
    assert np.all(actual[0] < keep_top_k)
    assert np.any(actual[0] < keep_top_k)


def test_full_pass_per_class_cap_uses_min_topk_keeptopk():
    """topK deliberately larger than keepTopK: the pass caps per-class
    unrolling at min(topK,keepTopK) (see header comment) -- verify the
    reference (which implements this same capping rule) still matches."""
    rng = np.random.default_rng(15)
    n, num_boxes, num_classes = 1, 8, 2
    keep_top_k, top_k = 3, 100
    centers = rng.uniform(0, 10.0, size=(n, num_boxes, 2))
    sizes = rng.uniform(0.5, 2.0, size=(n, num_boxes, 2))
    thetas = rng.uniform(-math.pi, math.pi, size=(n, num_boxes, 1))
    boxes = np.concatenate([centers, sizes, thetas], axis=-1).astype(np.float32)
    boxes_5d = boxes[:, :, None, :]
    scores = rng.uniform(0.0, 1.0, size=(n, num_boxes, num_classes)).astype(np.float32)

    model = _rotated_nms_model(n, num_boxes, num_classes, keep_top_k, top_k=top_k)
    actual = _run_full(model, boxes_5d, scores)
    expected = _reference_full_pass(
        boxes, scores, -1, top_k, keep_top_k, score_threshold=0.05, iou_threshold=0.5
    )
    _assert_same_detection_sets(actual, expected)


# ===========================================================================
# Structural/decline tests -- a light touch (this pass's scope is exercised
# much more exhaustively by tests/test_trt_batched_rotated_nms.py already;
# these just confirm the predicate declines outside documented scope).
# ===========================================================================


def test_declines_when_boxes_have_per_class_boxes():
    model = _model(
        """
        agraph (float[1,6,3,5] boxes, float[1,6,3] scores)
              => (int32[1,1] num_detections, float[1,4,5] nmsed_boxes,
                  float[1,4] nmsed_scores, float[1,4] nmsed_classes)
        {
          num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = mmdeploy.TRTBatchedRotatedNMS
            <background_label_id=-1, num_classes=3, topK=6, keepTopK=4,
             scoreThreshold=0.05, iouThreshold=0.5>
            (boxes, scores)
        }
        """,
        extra_opsets=', "mmdeploy": 1',
    )
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_declines_when_total_iterations_exceed_cap():
    """N * num_classes_active * min(topK,keepTopK) over the documented cap
    (2000, see the pass's header comment) must decline."""
    model = _rotated_nms_model(1, 6, 2001, keep_top_k=1, top_k=1)
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types


def test_declines_when_opset_below_13():
    model = _rotated_nms_model(1, 6, 3, keep_top_k=4)
    model.opset_import[0].version = 12
    simplified, ok = _simplify(model)
    assert ok
    op_types = [n.op_type for n in simplified.graph.node]
    assert "TRTBatchedRotatedNMS" in op_types, op_types
