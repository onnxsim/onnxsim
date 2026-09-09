"""Formal check for EliminateNopPad (eliminate_nop_pad.h).

``patternMatchPredicate`` matches any ``Pad`` node unconditionally; the real
work is in ``is_nop_pad``, called from ``runTransform``. It fetches ``pads``
via ``GetValueFromAttrOrInput(node, kpads, 1, pads)`` -- the ``pads``
attribute (opset <= 10) if present, else falling back to the second input
(opset 11+), which only succeeds if that input is a ``Constant`` node's
output or a constant initializer (``FetchConstantTensor``) -- and declines
(returns ``false``, so ``runTransform`` leaves the node untouched) if that
lookup fails, if the resulting vector is empty, or if any individual entry
is nonzero (``for (const auto& p : pads) if (p != 0) return false;``). It
does NOT look at ``mode`` at all: an all-zero-pads ``Pad`` is treated as a
no-op regardless of whether ``mode`` is "constant", "reflect", or "edge" --
which is semantically correct, since those modes only differ in how they
synthesize values for *added* padding, and there is none to synthesize when
every pad amount is exactly zero. ``runTransform`` then does exactly what
eliminate_identity.h does: ``tryReplacingAllUsesWith(node->output(),
node->inputs()[0])`` (input 0, the data tensor -- not the ``pads`` or the
optional ``value`` input) and destroys the Pad node
(``NodeDestroyType::DestroyOne``); the ``pads``/``value`` initializer or
Constant node, if any, is left dangling in the graph, the same caveat noted
in test_formal_verify_fuse_pad_into_conv.py.

Two edge cases the predicate declines on, both exercised below: (1) any pad
amount that is not *literally* zero, including one that would net to zero
length change some other way -- ``is_nop_pad`` checks each entry with
``p != 0``, there is no netting/cancellation logic; and (2) a ``pads``
input whose runtime value is provably always zero but isn't a
``Constant``/initializer the pass can see at compile time (e.g. it's the
output of some arithmetic) -- the pass has no evaluator, so it
conservatively declines rather than trying to prove the value is zero.

Soundness: for a single axis with begin/end pad amounts ``pad_begin``/
``pad_end``, a Pad's output at position ``j`` reads input position
``j - pad_begin`` when that position is in bounds, and otherwise falls back
to some mode-specific rule -- modeled below as ``boundary``, an
*uninterpreted* function standing in for whatever "constant"/"reflect"/
"edge" computes there, since the proof does not need to know which. When
``pad_begin == pad_end == 0`` (the exact hypothesis ``is_nop_pad`` checks,
applied per axis), the output has the same length as the input, so every
output position ``j`` lies in ``[0, length)``, and ``j - pad_begin == j`` is
therefore always in bounds too -- so the out-of-bounds branch, and hence
``boundary``, is never reached, regardless of what it computes. That is the
formal content of "mode doesn't matter when pads are zero": the proof holds
for an arbitrary ``boundary``, not just the three concrete modes ONNX
happens to define. Composing that pointwise identity with an arbitrary
uninterpreted downstream consumer (as in
test_formal_verify_eliminate_identity.py) then proves substitution
soundness for every possible consumer, not only the one this file's
differential checks happen to use.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_pad_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    boundary = z3.Function("boundary", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    length, pad_begin, pad_end, j = z3.Ints("length pad_begin pad_end j")

    # j ranges over the padded output's valid positions for this axis.
    domain = z3.And(length > 0, 0 <= j, j < length + pad_begin + pad_end)
    src = j - pad_begin
    in_bounds = z3.And(src >= 0, src < length)
    # Generic 1-axis Pad semantics, for ANY mode: in-bounds reads pass
    # through to the input; out-of-bounds is handled by an arbitrary
    # mode-specific rule the proof never needs to inspect.
    pad_output = z3.If(in_bounds, x(src), boundary(src))

    # is_nop_pad's actual hypothesis, applied per axis: every pad amount
    # individually equals zero (not merely nets to zero).
    is_nop_pad = z3.And(pad_begin == 0, pad_end == 0)

    prove(
        z3.Implies(
            z3.And(domain, is_nop_pad),
            consumer(pad_output) == consumer(x(j)),
        )
    )


def test_eliminate_nop_pad_pass_matches():
    # Differential check: an all-zero constant `pads` input makes is_nop_pad
    # true, so the compiled pass, run alone, removes the Pad node and
    # rewires its consumer directly to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[4] pads = {0, 0, 0, 0}>
        {
          p = Pad<mode = "constant">(X, pads)
          Y = Relu(p)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_pad")
    assert ops["Pad"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_pad_pass_matches_regardless_of_mode():
    # Same as above but mode="edge" -- is_nop_pad never inspects `mode`, so
    # the pass eliminates this exactly as readily as the "constant" case,
    # confirming the proof's central claim (mode is irrelevant when there is
    # nothing to pad) against the actual compiled pass, not just the model.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[4] pads = {0, 0, 0, 0}>
        {
          p = Pad<mode = "edge">(X, pads)
          Y = Relu(p)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_pad")
    assert ops["Pad"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_pad_declines_on_nonzero_pads():
    # Edge case from is_nop_pad: one pad amount (axis 1, begin) is nonzero,
    # so `for (const auto& p : pads) if (p != 0) return false;` trips on it
    # -- the predicate declines and the pass, run alone, must leave Pad
    # untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,10] Y)
        <int64[4] pads = {0, 1, 0, 1}>
        {
          Y = Pad<mode = "constant">(X, pads)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_pad")
    assert ops["Pad"] == 1


def test_eliminate_nop_pad_declines_on_non_constant_pads():
    # Edge case from is_nop_pad / GetValueFromAttrOrInput: `pads` here is
    # provably all-zero at runtime for any `s` (0 times anything is 0), but
    # its producer is a Mul node, not a Constant node or an initializer --
    # FetchConstantTensor only recognizes those two -- so the pass has no
    # way to see that at compile time and conservatively declines, even
    # though eliminating it would in fact be sound.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X, int64[4] s) => (float[4,8] Y)
        <int64[4] zero_pads = {0, 0, 0, 0}>
        {
          pads = Mul(zero_pads, s)
          p = Pad<mode = "constant">(X, pads)
          Y = Relu(p)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_pad")
    assert ops["Pad"] == 1
