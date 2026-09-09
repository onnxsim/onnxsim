"""Formal check for FuseConsecutiveConcats (fuse_consecutive_concats.h).

``patternMatchPredicate`` matches essentially every ``Concat`` node (any node
of kind ``Concat`` that has an ``axis`` attribute -- true for any valid
Concat). ``runTransform`` then loops over the outer Concat's own inputs; for
each input ``i`` that is itself the output of another Concat node (the
"inner" Concat) satisfying:

* ``cur_input_value->uses().size() == 1`` -- the inner Concat's output is
  consumed **only** by this outer Concat (the same single-consumer
  precondition used throughout this fusion-pass family, e.g.
  ``fuse_matmul_add_bias_into_gemm``'s ``orig_conv->uses().size() <= 1``);
  and
* the inner Concat has the same ``axis`` attribute value as the outer one,

the inner Concat's own inputs are spliced into the outer Concat's input list
**in place of, and at the position of,** that single input ``i`` (via
``insertInput``, which shifts every later input rightward to make room),
preserving the inner inputs' original order, and the now-dead inner Concat
node is destroyed. This can fire for multiple qualifying inputs of the outer
Concat within one ``runTransform`` call (the ``for`` loop over
``concat_node->inputs()`` doesn't stop after the first match) -- verified
empirically below (``test_fuse_consecutive_concats_pass_matches_two_fusable_inputs``)
-- but only replaces *direct* children, not grandchildren, so a chain nested
more than one Concat deep needs more than one pass invocation to fully
flatten; that multi-level case is out of scope here, one level (an outer
Concat with a single inner-Concat input) already captures the rewrite's
soundness content.

Formal content: Concat along a fixed axis is associative in the following
precise sense -- concatenating ``[A, B]`` into ``AB = Concat(A, B, axis=k)``
and then computing ``Concat(X, AB, Y, axis=k)`` (``X``, ``Y`` the outer
Concat's other inputs, possibly empty) produces *exactly* the same result,
elementwise, as ``Concat(X, A, B, Y, axis=k)`` directly. This is modeled
along a single axis (1-D, exactly mirroring how the existing
``fuse_consecutive_squeezes``/``fuse_consecutive_transposes`` proofs use
concrete small examples rather than fully general N-D symbolic shapes):
each named tensor becomes an uninterpreted Z3 function ``Int -> Real``, and
reading a concatenation at some global index is an offset-arithmetic case
split -- e.g. ``AB(i) = If(i < len(A), A(i), B(i - len(A)))`` -- the same
stride/offset-remapping style of argument as
``test_formal_verify_rewrite_gathernd_to_gather.py``'s flat-buffer proof,
just for concatenation offsets instead of row-major strides. The four
segment lengths (``len(X)=3, len(A)=2, len(B)=4, len(Y)=1``) are concrete,
not symbolic, to keep this tractable for Z3 -- exactly as the
squeeze/transpose proofs use concrete axes lists -- so this proves the
concatenation-splicing identity for these particular segment lengths,
generalizing the same way those proofs' concrete-axes results do (nothing
in the offset arithmetic depends on the specific lengths chosen; any other
concrete lengths would go through identically).

Concat along axis ``k`` of an N-D tensor reduces to exactly this same 1-D
argument applied independently to every fixed combination of the other
axes' indices (each "row" along axis ``k`` is sliced out and concatenated
on its own) -- that slicing reduction itself is not formalized here, only
the underlying 1-D offset identity it reduces to.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser

_LEN_X = 3
_LEN_A = 2
_LEN_B = 4
_LEN_Y = 1


def test_fuse_consecutive_concats_is_sound():
    X = z3.Function("X", z3.IntSort(), z3.RealSort())
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    Y = z3.Function("Y", z3.IntSort(), z3.RealSort())
    idx = z3.Int("idx")

    def AB(i):
        # y = Concat(A, B, axis=k), read at offset i into y.
        return z3.If(i < _LEN_A, A(i), B(i - _LEN_A))

    def direct(i):
        # Concat(X, A, B, Y, axis=k) -- what runTransform produces.
        return z3.If(
            i < _LEN_X,
            X(i),
            z3.If(
                i < _LEN_X + _LEN_A,
                A(i - _LEN_X),
                z3.If(
                    i < _LEN_X + _LEN_A + _LEN_B,
                    B(i - _LEN_X - _LEN_A),
                    Y(i - _LEN_X - _LEN_A - _LEN_B),
                ),
            ),
        )

    def via_inner_concat(i):
        # Concat(X, AB, Y, axis=k) -- what the graph computes before fusion.
        return z3.If(
            i < _LEN_X,
            X(i),
            z3.If(
                i < _LEN_X + _LEN_A + _LEN_B,
                AB(i - _LEN_X),
                Y(i - _LEN_X - _LEN_A - _LEN_B),
            ),
        )

    prove(direct(idx) == via_inner_concat(idx))


def test_fuse_consecutive_concats_pass_matches():
    # y = Concat(A, B, axis=0), Z = Concat(X, y, axis=0): y's only use is
    # this outer Concat and both share axis=0, so runTransform fires,
    # splicing A, B into Z's input list in y's place and destroying y's
    # producer node.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{_LEN_X},4] X, float[{_LEN_A},4] A, float[{_LEN_B},4] B)
          => (float[{_LEN_X + _LEN_A + _LEN_B},4] Z)
        {{
          y = Concat<axis=0>(A, B)
          Z = Concat<axis=0>(X, y)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 1
    (fused_node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(fused_node.input) == ["X", "A", "B"]


def test_fuse_consecutive_concats_pass_matches_with_trailing_segment():
    # Same as above but with a non-empty trailing segment Y too, i.e. the
    # general X, AB, Y -> X, A, B, Y splice (not just X, AB -> X, A, B):
    # confirms insertInput's in-place splice preserves both X before and Y
    # after the fused-in A, B, rather than e.g. appending them at the end.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{_LEN_X},4] X, float[{_LEN_A},4] A, float[{_LEN_B},4] B, float[{_LEN_Y},4] Y)
          => (float[{_LEN_X + _LEN_A + _LEN_B + _LEN_Y},4] Z)
        {{
          ab = Concat<axis=0>(A, B)
          Z = Concat<axis=0>(X, ab, Y)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 1
    (fused_node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(fused_node.input) == ["X", "A", "B", "Y"]


def test_fuse_consecutive_concats_pass_matches_two_fusable_inputs():
    # Both ab = Concat(A, B, axis=0) and cd = Concat(C, D, axis=0) feed the
    # outer Concat(ab, M, cd, axis=0), each with axis=0 and a single use --
    # runTransform's loop over the outer Concat's inputs fires for both in
    # one call, fusing away both inner Concat nodes at once.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,4] A, float[2,4] B, float[3,4] M, float[1,4] C, float[1,4] D)
          => (float[9,4] Z)
        {
          ab = Concat<axis=0>(A, B)
          cd = Concat<axis=0>(C, D)
          Z = Concat<axis=0>(ab, M, cd)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 1
    (fused_node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(fused_node.input) == ["A", "B", "M", "C", "D"]


def test_fuse_consecutive_concats_declines_different_axis():
    # y = Concat(A, B, axis=1) but Z = Concat(X, y, axis=0): the inner and
    # outer axis attributes differ, so
    # cur_input_node->i(kaxis) == concat_node->i(kaxis) fails and
    # runTransform declines for this input -- both Concat nodes survive.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] X, float[2,2] A, float[2,2] B) => (float[5,4] Z)
        {
          y = Concat<axis=1>(A, B)
          Z = Concat<axis=0>(X, y)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 2


def test_fuse_consecutive_concats_declines_multi_use_inner_concat():
    # y = Concat(A, B, axis=0) is consumed both by the outer Concat and
    # directly as a second graph output, so y's producer has two uses
    # (cur_input_value->uses().size() == 1 fails) and runTransform declines
    # -- both Concat nodes survive.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{_LEN_X},4] X, float[{_LEN_A},4] A, float[{_LEN_B},4] B)
          => (float[{_LEN_X + _LEN_A + _LEN_B},4] Z, float[{_LEN_A + _LEN_B},4] y)
        {{
          y = Concat<axis=0>(A, B)
          Z = Concat<axis=0>(X, y)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 2


def test_fuse_consecutive_concats_declines_non_concat_input():
    # Sanity check on the pass itself: an outer Concat whose input is not
    # the output of another Concat at all has nothing to fuse, and
    # runTransform's per-input check (cur_input_node->kind() == kConcat)
    # simply never fires -- the single Concat node survives unchanged.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] X, float[2,4] A) => (float[5,4] Z)
        {
          Z = Concat<axis=0>(X, A)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_concats")
    assert ops["Concat"] == 1
    (node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(node.input) == ["X", "A"]
