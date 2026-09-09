"""Formal check for EliminateNopExpand (eliminate_nop_expand.h).

``patternMatchPredicate`` matches an ``Expand`` node only when its second
input (the target ``shape``) is a compile-time constant tensor
(``IsConstantTensor(node, 1)`` -- a ``Constant`` node's output or a constant
initializer, per ``FetchConstantTensor``/``pass_util.h``). ``runTransform``
then parses that constant as ``int64_t``s and calls
``isABroadcastToB(shape_as_ints, input_value->sizes())`` (see
``pass_util.h``), declining -- leaving the ``Expand`` untouched -- if that's
false or if the input's own shape isn't statically known at all. Given
``dims_a`` (the target ``shape``) and ``dims_b`` (X's own statically-known
shape), ``isABroadcastToB``:

- returns false immediately if ``dims_a`` has more entries than ``dims_b``;
- otherwise right-aligns the two (numpy-broadcast style) and, for each
  aligned axis, requires *either* ``dims_a``'s entry to be exactly ``1``
  (`"keep whatever X already has here"`), *or* ``dims_b``'s corresponding
  entry to be a statically-known int equal to ``dims_a``'s entry
  (`"redundantly restates X's actual size"`);
- any of ``dims_b``'s *leading* entries left unaligned (when ``dims_a`` is
  shorter) are entirely unconstrained.

So ``isABroadcastToB`` holds exactly when broadcasting ``shape`` against X's
own existing shape reproduces that same shape unchanged -- i.e. exactly when
``Expand(X, shape)`` is a genuine no-op, as opposed to a real broadcast that
grows some size-1 axis of X to a larger size (which ``isABroadcastToB``
explicitly rejects: a target entry that is neither ``1`` nor equal to X's own
already-known size at that axis, e.g. growing a size-1 axis, fails both
branches). When it holds, ``runTransform`` does exactly what
``eliminate_identity.h`` does: ``tryReplacingAllUsesWith(node->output(),
input_value)``.

Soundness: ONNX/numpy broadcasting's per-axis index rule is that an input
axis of size ``d`` reads source index ``0`` when ``d == 1`` (regardless of
the output size at that axis), and otherwise reads the output index directly
(since the input and output sizes coincide on that axis). Because
``isABroadcastToB`` only accepts a target entry equal to ``1`` or to X's own
already-known size ``d`` at that axis, the output size at every axis is
always exactly ``d`` -- never anything genuinely broadcast-grown -- so the
"direct passthrough" half of that rule is what actually fires whenever
``d != 1``, for either of ``isABroadcastToB``'s two OR-branches. Modeling X
as a rank-2 tensor with a known concrete shape ``[D0, D1]`` (concrete
integers, not symbolic, matching how the existing squeeze/transpose proofs
use concrete example shapes/axes to keep the arithmetic tractable) and
picking a concrete target ``shape = [1, D1]`` -- axis 0 exercises the
"``== 1``" branch, axis 1 the "statically equal" branch, since ``D1 != 1`` --
this reduces to a genuine identity: ``Expand(X, shape)[i, j] == X[i, j]`` for
every valid index. Composing that with an arbitrary uninterpreted
``consumer`` (as in test_formal_verify_eliminate_identity.py) then proves
substitution soundness for every possible downstream consumer, not only the
one this file's differential checks happen to use.
"""

import collections

from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import parser

import onnxsim


def test_eliminate_nop_expand_is_sound():
    D0, D1 = 4, 8  # X's own statically-known shape
    a0, a1 = 1, D1  # Expand's target `shape` input

    # isABroadcastToB's own per-axis check, spelled out exactly as
    # pass_util.h computes it (X's own dims are always statically known
    # here, i.e. `Dimension.is_int` is true for both axes) -- confirming the
    # chosen example actually satisfies the real predicate, and that it
    # exercises BOTH of its OR-branches rather than the same one twice.
    assert a0 == 1  # branch: target says "keep whatever's there"
    assert a1 != 1 and a1 == D1  # branch: target restates X's actual size

    x = z3.Function("x", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")
    domain = z3.And(0 <= i, i < D0, 0 <= j, j < D1)

    def broadcast_src(d, idx):
        # ONNX/numpy broadcasting's per-axis index rule: an input axis of
        # size 1 always reads source index 0, regardless of the output size
        # there; otherwise (the only other case isABroadcastToB allows,
        # since a target entry must be 1 or equal to d) input and output
        # sizes coincide on that axis, and the index passes straight
        # through unchanged.
        return 0 if d == 1 else idx

    expand_output = x(broadcast_src(D0, i), broadcast_src(D1, j))

    prove(z3.Implies(domain, consumer(expand_output) == consumer(x(i, j))))


def test_eliminate_nop_expand_pass_matches_keep_axis():
    # Differential check: shape=[1,8] against X's actual [4,8]. Axis 0 (a0=1)
    # passes isABroadcastToB's "==1" branch trivially regardless of X's own
    # dim 0; axis 1 (a1=8) passes via the "statically equal" branch since it
    # restates X's actual dim 1 (8). isABroadcastToB holds, so the compiled
    # pass, run alone, removes the Expand node and rewires its consumer
    # directly to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] shape = {1, 8}>
        {
          e = Expand(X, shape)
          Y = Relu(e)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_expand")
    assert ops["Expand"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_expand_pass_matches_restate_both_axes():
    # shape=[4,8] exactly restates X's own full shape -- both axes pass via
    # isABroadcastToB's "statically equal" branch (neither entry is 1). The
    # pass should fire just as readily as the "keep axis" case above.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] shape = {4, 8}>
        {
          e = Expand(X, shape)
          Y = Relu(e)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_expand")
    assert ops["Expand"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_expand_declines_on_real_broadcast():
    # X's own dim 1 is genuinely 1 (a broadcastable axis), and shape=[4,16]
    # actually grows it: dims_a[1]=16 is neither 1 nor equal to X's dim 1
    # (1), so isABroadcastToB is false -- this is a real, valid shape change
    # (1 -> 16), exactly the case Expand exists for, and a good negative
    # control since the predicate must decline rather than firing.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,1] X) => (float[4,16] Y)
        <int64[2] shape = {4, 16}>
        {
          Y = Expand(X, shape)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_expand")
    assert ops["Expand"] == 1


def test_eliminate_nop_expand_declines_on_non_constant_shape():
    # `shape` here is provably always [1, 8] at runtime (an Add of two
    # constant initializers), which would make Expand a genuine no-op -- but
    # it is neither a Constant node nor an initializer itself, so
    # IsConstantTensor/FetchConstantTensor can't see it at compile time and
    # the predicate declines, leaving Expand (and the Add) in place. This
    # bypasses simplify_isolated (which only controls the *optimizer pass*
    # list) and calls onnxsim.simplify directly with
    # skip_constant_folding=True -- onnxsim's own constant folding is a
    # separate step that runs before the optimizer passes regardless of
    # skipped_optimizers, and would otherwise fold the Add away (as
    # confirmed empirically while developing this test) before
    # eliminate_nop_expand ever saw it, making this an unreliable negative
    # control without that flag.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] shape_a = {0, 3}, int64[2] shape_b = {1, 5}>
        {
          shape = Add(shape_a, shape_b)
          e = Expand(X, shape)
          Y = Relu(e)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_nop_expand"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Expand"] == 1
    assert ops["Add"] == 1
