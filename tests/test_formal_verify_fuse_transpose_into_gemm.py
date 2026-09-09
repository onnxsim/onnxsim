"""Formal check for FuseTransposeIntoGemm (fuse_transpose_into_gemm.h).

The pass matches every ``Gemm`` node unconditionally, then checks its first
two inputs (A at index 0, B at index 1) independently: if an input is the
output of a ``Transpose`` node whose ``perm`` is exactly ``[1, 0]``, the pass
rewires Gemm's input to read directly from that Transpose's own input
(skipping it) and toggles Gemm's corresponding boolean attribute
(``transA``/``transB``) -- ``n->i_(trans, n->hasAttribute(trans) ?
!n->i(trans) : 1)``: flip the existing value, or set it to ``1`` if the
attribute was absent (equivalently "was false"). The old Transpose node is
destroyed only if it has no remaining uses; a Transpose that also feeds
something else is left alive with its output still wired to that other use.

ONNX Gemm computes ``alpha * op(A, transA) @ op(B, transB) + beta * C``,
where ``op(X, flag)`` is ``X`` if ``flag`` is false and ``X``'s 2-D transpose
if ``flag`` is true. The rewrite's soundness rests on one algebraic fact:
reading through an explicit ``Transpose`` node and applying Gemm's own
``op`` with the *original* flag is exactly the same value, at every index,
as reading the pre-Transpose tensor directly and applying ``op`` with the
*toggled* flag -- because transposing twice cancels out (flags disagree) and
transposing once matches once (flags agree). The flag only ever takes one of
two effective values (false/absent, or true), so this is proved by an
explicit 2-way case split (not a fully symbolic boolean), mirroring
``tests/test_formal_verify_fuse_consecutive_transposes.py``'s use of an
uninterpreted 2-D tensor function plus index-swapping for ``transpose``.

That per-operand identity is then composed with a genuine (if small,
concrete-sized) Gemm computation -- ``alpha * A' @ B' + beta * C`` summed out
over a concrete contraction dimension -- rather than an opaque uninterpreted
"consumer" stand-in, matching the embedding style of
test_formal_verify_fuse_matmul_into_conv.py / fuse_bn_into_conv.py: this pass
is *specifically* about which value each of Gemm's own two matmul operands
reads before Gemm's actual arithmetic runs, so showing the identity survives
substitution into that arithmetic (for both operands, and all 4 combinations
of their original flag states -- covering both firing together, as in the
"both A and B transposed" differential test below) ties the proof directly
to what Gemm computes, not just to an abstract "substitution is safe" lemma.
"""

from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser

_M = 2
_K = 2
_N = 2


def _transposed_read(tensor):
    """The output of ``Transpose(tensor, perm=[1, 0])``: index (x, y) reads
    ``tensor`` at (y, x), ONNX/numpy Transpose semantics for a plain 2-D
    swap."""

    def read(x, y):
        return tensor(y, x)

    return read


def _op(tensor, flag):
    """ONNX Gemm's per-operand ``op``: transpose ``tensor`` if ``flag`` is
    true, else read it as-is."""

    def read(x, y):
        return tensor(y, x) if flag else tensor(x, y)

    return read


def test_transpose_toggle_identity_is_sound():
    """``op(Transpose(A0), original_flag) == op(A0, toggled_flag)`` at every
    index, for both possible effective starting states of the flag (false --
    which also covers "attribute absent", since runTransform's
    ``hasAttribute(trans) ? !v : 1`` sets the same toggled value 1 whether
    absent or explicitly 0 -- and true). This is exactly the substitution
    runTransform performs on each of Gemm's first two inputs.
    """
    A0 = z3.Function("A0", z3.IntSort(), z3.IntSort(), z3.RealSort())
    i, j = z3.Ints("i j")
    transpose_out = _transposed_read(A0)  # Transpose(A0, perm=[1,0])'s own output

    for original_flag in (False, True):
        toggled_flag = not original_flag  # !n->i(trans), or the absent->1 case
        lhs = _op(transpose_out, original_flag)(i, j)  # pre-rewrite read
        rhs = _op(A0, toggled_flag)(i, j)  # post-rewrite read
        prove(
            lhs == rhs, msg=f"toggle identity fails for original_flag={original_flag}"
        )


def test_fuse_transpose_into_gemm_gemm_composition_is_sound():
    """The per-operand identity above survives substitution into a genuine
    Gemm computation, for both operands simultaneously and for all 4
    combinations of their original flag states -- the pass rewires and
    toggles A and B independently in one ``runTransform`` call, and this
    covers both firing together (as well as, degenerately, either alone,
    since each operand's term is handled identically regardless of the
    other's flag value).
    """
    A0 = z3.Function("A0", z3.IntSort(), z3.IntSort(), z3.RealSort())
    B0 = z3.Function("B0", z3.IntSort(), z3.IntSort(), z3.RealSort())
    C = z3.Function("C", z3.IntSort(), z3.IntSort(), z3.RealSort())
    alpha, beta = z3.Reals("alpha beta")

    a_transposed = _transposed_read(A0)  # Transpose(A0, perm=[1,0])'s own output
    b_transposed = _transposed_read(B0)  # Transpose(B0, perm=[1,0])'s own output

    def gemm(a_read, transA, b_read, transB):
        a = _op(a_read, transA)
        b = _op(b_read, transB)

        def out(p, q):
            return alpha * sum(a(p, k) * b(k, q) for k in range(_K)) + beta * C(p, q)

        return out

    for orig_transA in (False, True):
        for orig_transB in (False, True):
            pre = gemm(a_transposed, orig_transA, b_transposed, orig_transB)
            post = gemm(A0, not orig_transA, B0, not orig_transB)
            claim = z3.And(
                *[pre(p, q) == post(p, q) for p in range(_M) for q in range(_N)]
            )
            prove(
                claim,
                msg=(
                    "gemm composition fails for "
                    f"transA={orig_transA}, transB={orig_transB}"
                ),
            )


def test_fuse_transpose_into_gemm_toggles_absent_trans_attribute():
    # No transA attribute to start (defaults false) -- the pass should
    # rewire input 0 to A0 directly and set transA=1 (absent/false ->
    # toggled true).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3] A0, float[2,4] B) => (float[3,4] Y)
        {
          At = Transpose<perm = [1, 0]>(A0)
          Y = Gemm(At, B)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_transpose_into_gemm")
    assert ops["Transpose"] == 0

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    assert list(gemm_node.input) == ["A0", "B"]
    attrs = {a.name: a.i for a in gemm_node.attribute}
    assert attrs.get("transA") == 1
    assert "transB" not in attrs


def test_fuse_transpose_into_gemm_flips_existing_trans_attribute():
    # transB=1 already present -- the pass should rewire input 1 to B0
    # directly and flip transB to 0 (the "already had the attribute, flip
    # it" branch, not "attribute absent, set to 1").
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] A, float[4,3] B0) => (float[3,3] Y)
        {
          Bt = Transpose<perm = [1, 0]>(B0)
          Y = Gemm<transB = 1>(A, Bt)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_transpose_into_gemm")
    assert ops["Transpose"] == 0

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    assert list(gemm_node.input) == ["A", "B0"]
    attrs = {a.name: a.i for a in gemm_node.attribute}
    assert attrs.get("transB") == 0
    assert "transA" not in attrs


def test_fuse_transpose_into_gemm_fires_on_both_inputs_at_once():
    # Both A and B fed by their own [1, 0]-perm Transpose -- both should
    # fire in the same runTransform call: both inputs rewired, both flags
    # toggled from absent to 1.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3] A0, float[4,2] B0) => (float[3,4] Y)
        {
          At = Transpose<perm = [1, 0]>(A0)
          Bt = Transpose<perm = [1, 0]>(B0)
          Y = Gemm(At, Bt)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_transpose_into_gemm")
    assert ops["Transpose"] == 0

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    assert list(gemm_node.input) == ["A0", "B0"]
    attrs = {a.name: a.i for a in gemm_node.attribute}
    assert attrs.get("transA") == 1
    assert attrs.get("transB") == 1


def test_fuse_transpose_into_gemm_declines_non_matching_perm():
    # perm=[0,1] is a no-op transpose but is NOT the literal [1, 0] this
    # pass exact-matches on -- must not fire.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,4] A0, float[4,3] B) => (float[2,3] Y)
        {
          At = Transpose<perm = [0, 1]>(A0)
          Y = Gemm(At, B)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_transpose_into_gemm")
    assert ops["Transpose"] == 1

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    assert list(gemm_node.input) == ["At", "B"]
    assert not any(a.name == "transA" for a in gemm_node.attribute)


def test_fuse_transpose_into_gemm_keeps_multi_use_transpose_alive():
    # The Transpose feeding Gemm's A input also feeds a second graph output
    # -- the input rewiring isn't conditioned on single-use (only node
    # destruction is), so Gemm's input should still be redirected to A0 and
    # transA toggled, but the Transpose node itself must survive since it
    # still has a use.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3] A0, float[2,4] B) => (float[3,4] Y, float[3,2] Z)
        {
          At = Transpose<perm = [1, 0]>(A0)
          Y = Gemm(At, B)
          Z = Identity(At)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_transpose_into_gemm")
    assert ops["Transpose"] == 1

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    assert list(gemm_node.input) == ["A0", "B"]
    attrs = {a.name: a.i for a in gemm_node.attribute}
    assert attrs.get("transA") == 1

    z_producer = producer(sim_model, "Z")
    assert z_producer.op_type == "Identity"
    transpose_node = producer(sim_model, z_producer.input[0])
    assert transpose_node.op_type == "Transpose"
    assert list(transpose_node.input) == ["A0"]
