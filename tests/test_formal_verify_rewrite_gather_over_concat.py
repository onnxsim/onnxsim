"""Formal check for RewriteGatherOverConcat (opt-in; onnxsim's own
``onnxsim/passes/rewrite_gather_over_concat.h``): rewrites
``Gather(Concat(x_0, ..., x_{n-1}, axis=ca), indices, axis=ga)`` into
``Gather(x_i, indices - offset_i, axis=ga)`` whenever ``ga == ca`` and the
compile-time-constant ``indices`` all fall inside the same single Concat
input segment ``x_i`` -- ``offset_i`` being the cumulative sum of the
axis-``ca`` sizes of every Concat input before ``x_i``. If the resolved
indices span more than one segment, the pass declines outright: a single
``Gather`` node can only depend on one upstream value, so there is nothing
to rewire it to.

Formal content: this is the same concatenation-offset arithmetic as
``test_formal_verify_fuse_consecutive_concats.py`` (each segment an
uninterpreted ``Int -> Real`` function, a concatenated read a nested
``z3.If`` case-split on which segment a global offset falls into), combined
with ``Gather``'s own single-index-selection semantics: reading
``Concat(A, B, C)`` at some constant global offset and then selecting that
one "row" is the same thing as reading the one segment that offset falls
into directly, at the offset shifted down by that segment's own cumulative
size. Three segments ``A``, ``B``, ``C`` (concrete lengths 2, 3, 2 -- as
concrete as ``fuse_consecutive_concats``'s own ``_LEN_*`` constants, for the
same tractability reason) are modeled along a single axis; the proof below
targets the middle segment ``B`` (offsets ``[len(A), len(A)+len(B))``) since
it is the only one of the three with a segment on *both* sides, exercising
both branches of ``ResolveSingleSegment``'s cumulative-offset scan rather
than just one boundary. A negative control confirms the segment-selection
arithmetic is load-bearing, not vacuously true regardless of which segment
is read: reading a *different* segment (``A``) at the same local offset is
*not* generally equal to the correct segment's (``B``'s) value there.

The differential checks below additionally confirm, against the real
compiled pass, two things this proof does not itself state: (1) the pass's
own negative-index normalization (``AddYIfNegative`` against the *total*
concatenated size, matching ``Gather``'s and ``GatherND``'s own negative-index
convention -- see ``test_formal_verify_rewrite_gathernd_to_gather.py``) picks
the same segment/local-index pair as normalizing by hand would; and (2) the
rewrite only rewires the ``Gather`` node's own two inputs (input 0 to the
winning Concat input directly, input 1 to a fresh local-indices constant) --
the ``Concat`` node and its *other* inputs are left dangling, not destroyed,
exactly as the header comment describes (a later dead-code pass is expected
to clean them up; isolating this one opt-in pass via
``simplify_isolated_extra`` runs it without that companion, so the dangling
``Concat`` survives into the simplified model and is asserted on directly).
"""

import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

_LEN_A = 2
_LEN_B = 3
_LEN_C = 2


def test_rewrite_gather_over_concat_is_sound():
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    C = z3.Function("C", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    global_offset = z3.Int("global_offset")

    def concat_read(i):
        # y = Concat(A, B, C, axis=k), read at global offset i -- what the
        # original Gather(Concat(...), indices, axis=k) computes for one
        # constant index value in `indices`.
        return z3.If(
            i < _LEN_A,
            A(i),
            z3.If(i < _LEN_A + _LEN_B, B(i - _LEN_A), C(i - _LEN_A - _LEN_B)),
        )

    # B's own segment: global offsets in [len(A), len(A)+len(B)).
    local_offset = global_offset - _LEN_A
    in_b_segment = z3.And(global_offset >= _LEN_A, global_offset < _LEN_A + _LEN_B)

    # runTransform's rewritten form: Gather(B, indices - offset_B, axis=k),
    # composed with an arbitrary consumer for substitution safety (the same
    # style as the other formal-verify proofs in this suite).
    prove(
        z3.Implies(
            in_b_segment,
            consumer(concat_read(global_offset)) == consumer(B(local_offset)),
        )
    )


def test_rewrite_gather_over_concat_segment_selection_is_load_bearing():
    # Negative control: reading a *different* segment (A) at the same local
    # offset is not generally the correct answer -- confirming
    # ResolveSingleSegment's segment scan (picking exactly one segment, not
    # any segment) is load-bearing, not a vacuously-true rewrite.
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    C = z3.Function("C", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    global_offset = z3.Int("global_offset")

    def concat_read(i):
        return z3.If(
            i < _LEN_A,
            A(i),
            z3.If(i < _LEN_A + _LEN_B, B(i - _LEN_A), C(i - _LEN_A - _LEN_B)),
        )

    local_offset = global_offset - _LEN_A
    in_b_segment = z3.And(global_offset >= _LEN_A, global_offset < _LEN_A + _LEN_B)
    wrong_segment_claim = z3.Implies(
        in_b_segment,
        consumer(concat_read(global_offset)) == consumer(A(local_offset)),
    )

    solver = z3.Solver()
    solver.add(z3.Not(wrong_segment_claim))
    assert solver.check() == z3.sat, (
        "reading the wrong segment should not generally match -- segment "
        "selection would be vacuous"
    )


def _model(indices_text):
    # A, B, C have axis-0 sizes _LEN_A, _LEN_B, _LEN_C respectively (2, 3, 2).
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{_LEN_A},4] A, float[{_LEN_B},4] B, float[{_LEN_C},4] C)
          => (float[2,4] Y)
        <int64[2] indices = {indices_text}>
        {{
          cc = Concat<axis=0>(A, B, C)
          Y = Gather<axis=0>(cc, indices)
        }}
        """
    )


def test_rewrite_gather_over_concat_pass_matches():
    # indices = [3, 4]: both fall inside B's own segment (offsets [2, 5)),
    # at local offsets 1 and 2. The pass should rewire Gather to read B
    # directly with local indices [1, 2], leaving Concat (and A, C) dangling.
    model = _model("{3, 4}")
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gather_over_concat")
    assert ops["Concat"] == 1  # dangling, per the pass's own design
    assert ops["Gather"] == 1

    gather_node = producer(sim_model, "Y")
    assert gather_node.op_type == "Gather"
    assert gather_node.input[0] == "B"  # rewired straight to the one segment

    new_indices_name = gather_node.input[1]
    assert new_indices_name != "indices"  # rewired to a fresh constant
    (new_indices_init,) = [
        init for init in sim_model.graph.initializer if init.name == new_indices_name
    ]
    assert list(onnx.numpy_helper.to_array(new_indices_init)) == [1, 2]

    # The Concat node itself, and its other inputs A/C, survive untouched.
    (concat_node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(concat_node.input) == ["A", "B", "C"]


def test_rewrite_gather_over_concat_pass_matches_negative_index():
    # indices = [3, -3]: 3 is already in [0, total=7) (local index 1 in B);
    # -3 normalizes to total + (-3) = 4, also in B's segment (local index
    # 4 - 2 = 2). Confirms AddYIfNegative's normalization against the
    # *total* concatenated size, not any one segment's own size.
    model = _model("{3, -3}")
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gather_over_concat")
    assert ops["Gather"] == 1

    gather_node = producer(sim_model, "Y")
    assert gather_node.input[0] == "B"
    new_indices_name = gather_node.input[1]
    (new_indices_init,) = [
        init for init in sim_model.graph.initializer if init.name == new_indices_name
    ]
    assert list(onnx.numpy_helper.to_array(new_indices_init)) == [1, 2]


def test_rewrite_gather_over_concat_declines_indices_spanning_two_segments():
    # indices = [1, 3]: 1 falls in A's segment ([0, 2)), 3 falls in B's
    # ([2, 5)) -- ResolveSingleSegment sees two different segments and
    # returns false, so the pass must not fire at all.
    model = _model("{1, 3}")
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gather_over_concat")
    assert ops["Concat"] == 1
    assert ops["Gather"] == 1
    gather_node = producer(sim_model, "Y")
    assert gather_node.input[0] == "cc"
    assert gather_node.input[1] == "indices"


def test_rewrite_gather_over_concat_declines_different_axis():
    # Concat joins on axis=1, but Gather selects along axis=2: ga != ca, so
    # the predicate declines regardless of what indices resolve to.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2,4] A, float[2,3,4] B) => (float[2,2,2] Y)
        <int64[2] indices = {0, 1}>
        {
          cc = Concat<axis=1>(A, B)
          Y = Gather<axis=2>(cc, indices)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gather_over_concat")
    assert ops["Concat"] == 1
    assert ops["Gather"] == 1
    gather_node = producer(sim_model, "Y")
    assert gather_node.input[0] == "cc"


def test_rewrite_gather_over_concat_declines_non_constant_indices():
    # `indices` is a graph input, not a compile-time constant: FetchConstantTensor
    # returns null and the predicate declines.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{_LEN_A},4] A, float[{_LEN_B},4] B, int64[2] indices)
          => (float[2,4] Y)
        {{
          cc = Concat<axis=0>(A, B)
          Y = Gather<axis=0>(cc, indices)
        }}
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gather_over_concat")
    assert ops["Concat"] == 1
    assert ops["Gather"] == 1
    gather_node = producer(sim_model, "Y")
    assert gather_node.input[0] == "cc"
    assert gather_node.input[1] == "indices"
