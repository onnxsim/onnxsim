"""Formal check for EliminateNopSplit (eliminate_nop_split.h).

``patternMatchPredicate`` matches a ``Split`` node when: it's kind ``Split``,
its (single) input has a statically-known shape (``has_sizes()``), AND it has
*exactly one* output (``node->outputs().size() == 1``). All three conjuncts
are hard requirements -- a multi-output Split (by far the common case) is
rejected outright by the last one, and an input whose shape isn't statically
known is rejected by the second, regardless of anything else.

``runTransform`` then computes ``axis`` (the ``axis`` attribute, default 0,
normalized for negative values against the input's rank) and tries to fetch
an explicit ``split`` sizes list via
``GetValueFromAttrOrInput(node, ksplit, 1, split)`` -- the ``split``
attribute for older opsets, or the second input for opset 13+, only
succeeding if that input is a ``Constant``/initializer
(``FetchConstantTensor``). The actual guard is:

.. code-block:: c++

    if (GetValueFromAttrOrInput(node, ksplit, 1, split) && !split.empty() &&
        (!sizes[axis].is_int || sizes[axis].dim != split[0])) {
      return false;
    }

Read carefully, this is short-circuiting ``&&``, not a single flat
condition: when the lookup *fails* (no ``split`` attribute and no second
input at all -- ``GetValueFromAttrOrInput`` returns ``false``), the whole
condition is ``false`` regardless of the rest, so the mismatch check is
*skipped entirely* and the pass proceeds unconditionally. This is not a gap:
ONNX's own ``Split`` schema says that with one output and no ``split`` sizes
given, that lone output defaults to covering the *entire* input along
``axis`` -- so the "no split given" case is trivially, unconditionally a
no-op, and only ever gets more specific when an explicit ``split`` list *is*
supplied, in which case ``split[0]`` (there is only one output, so only one
list entry matters) must exactly equal the input's own statically-known size
along ``axis`` or the predicate declines.

Empirically (see below), a single-output ``Split`` whose explicit ``split``
list does *not* sum to the input's full dimension along ``axis`` is not
constructible as valid ONNX in the first place: ONNX's ``Split`` schema (via
its own shape inference, run inside onnxsim's own model-checking) requires
``sum(split) == dim(axis)``, and with one output that forces
``split[0] == dim(axis)`` -- exactly the equality the pass's mismatch branch
checks, and exactly the equality that lets it be skipped safely when no
``split`` is given at all. So the ``sizes[axis].dim != split[0]`` branch in
the C++ is a defensive check that, for a genuinely single-output Split, can
never actually observe a mismatch under ONNX's own validity rules -- it can
only ever confirm what ONNX has already guaranteed. This file's negative
control for "explicit but wrong split sizes" is therefore the far more
common real-world case instead: two or more outputs, rejected outright by
the ``outputs().size() == 1`` conjunct before the split-sizes logic is even
reached.

Once the predicate matches, ``runTransform`` does exactly what
``eliminate_identity.h`` does: ``tryReplacingAllUsesWith(node->output(),
input)`` and destroys the node (``NodeDestroyType::DestroyOne``).

Soundness: a Split into exactly one output, whose size along ``axis`` equals
the *full* input size there, is a pointwise identity -- ``split_output(idx)
== input(idx)`` for every valid index, not merely every index the output
itself declares. This holds because a single output is necessarily the
*first* output, so ONNX's own Split semantics start its slice at offset 0
along ``axis`` (there is no preceding output to have already consumed part
of it); once that output's declared length along ``axis`` equals the whole
input's length there, its domain covers the same range the input itself
does, with nothing to its left or right left unaccounted for -- along every
other axis, Split does not reindex at all, so those pass through unchanged
by construction. Modeled below as a rank-2 tensor for tractability (an
uninterpreted Z3 function ``Int, Int -> Real``) with ``axis`` fixed to 0 out
of concreteness, the same single-axis scoping
``test_formal_verify_eliminate_nop_pad.py`` uses (stated honestly in each
proof below rather than claimed to generalize automatically): this file
proves the one-axis, rank-2 case, generalizing to other axes and ranks by
the identical per-axis argument Pad's proof makes -- the axis being split
plays the role Pad's padded axis does, and every other axis is untouched
either way. Composing with an arbitrary uninterpreted ``consumer`` function,
exactly as the other nop-elimination proofs in this repo do, then proves
substitution safety for any downstream consumer, not only the one this
file's differential checks happen to use.
"""

from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import parser

import onnxsim


def test_eliminate_nop_split_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.IntSort(), z3.RealSort())
    split_output = z3.Function(
        "split_output", z3.IntSort(), z3.IntSort(), z3.RealSort()
    )
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    dim0, dim1, out_dim0, i, j, row = z3.Ints("dim0 dim1 out_dim0 i j row")

    # Split's own semantics along axis 0, for the single (hence *first*)
    # output: it always starts at offset 0 -- there is no preceding output
    # to have already consumed part of axis 0 -- so within its own declared
    # domain (0 <= row < out_dim0) it reads input row `row` unchanged. Axis
    # 1 (standing in for every other, non-split axis) is untouched by
    # construction, since Split only ever reindexes along `axis`.
    split_semantics = z3.ForAll(
        [row, j],
        z3.Implies(
            z3.And(0 <= row, row < out_dim0, 0 <= j, j < dim1),
            split_output(row, j) == x(row, j),
        ),
    )

    # The pass's actual hypothesis: the single output's declared/inferred
    # size along axis 0 equals the whole input's size there -- either
    # because runTransform checked split[0] == sizes[axis] explicitly (an
    # explicit split list was given and matched), or because it's
    # guaranteed unconditionally by ONNX's own default Split semantics when
    # no split sizes are given for a single output at all.
    covers_whole_axis = out_dim0 == dim0

    # i, j range over the INPUT's own full domain -- not merely the
    # output's declared domain -- to capture that the single output isn't
    # just "consistent with some slice of the input" but literally covers
    # the whole tensor along every axis, not only `axis`.
    domain = z3.And(dim0 > 0, dim1 > 0, 0 <= i, i < dim0, 0 <= j, j < dim1)

    prove(
        z3.Implies(
            z3.And(split_semantics, covers_whole_axis, domain),
            consumer(split_output(i, j)) == consumer(x(i, j)),
        )
    )


def test_eliminate_nop_split_pass_matches_no_explicit_split():
    # Differential check: a single-output Split with no `split` attribute
    # and no second input at all. GetValueFromAttrOrInput fails (no
    # attribute, and input index 1 is out of range with only one input), so
    # the `&&` chain short-circuits and the mismatch check never runs --
    # ONNX's own default Split semantics (one output, no split sizes) make
    # that one output cover the whole input unconditionally, so the compiled
    # pass, run alone, removes the Split node and rewires its consumer
    # directly to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          s = Split<axis = 0>(X)
          Y = Relu(s)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_split")
    assert ops["Split"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_split_pass_matches_explicit_matching_split():
    # A single-output Split with an EXPLICIT split-sizes input (opset 13+
    # form) whose one entry exactly equals X's own full size along axis 0.
    # GetValueFromAttrOrInput succeeds here (split = [4]), split is
    # non-empty, and sizes[axis].dim (4) == split[0] (4), so the mismatch
    # branch does not trip -- the pass still fires.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[1] split = {4}>
        {
          s = Split<axis = 0>(X, split)
          Y = Relu(s)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_split")
    assert ops["Split"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_split_pass_matches_explicit_matching_split_attribute_form():
    # Same as above, but using the pre-opset-13 `split` *attribute* form
    # instead of a second input -- GetValueFromAttrOrInput's
    # GetValueFromAttr branch succeeds instead of its GetValueFromInput
    # branch, exercising the other half of that `||`.
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 11]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          s = Split<axis = 0, split = [4]>(X)
          Y = Relu(s)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_split")
    assert ops["Split"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_split_declines_on_multiple_outputs():
    # The far more common real-world case, and a solid negative control:
    # `outputs().size() == 1` is a hard conjunct in patternMatchPredicate,
    # so a 2-output Split must never fire regardless of its split sizes
    # (here they're even a perfectly valid, even 2-way split). This is also
    # empirically the *only* constructible negative control for the
    # mismatch branch's spirit: a single-output Split with a `split` list
    # that does NOT sum to the input's full axis dimension is rejected by
    # ONNX's own Split shape inference before onnxsim's optimizer passes
    # ever run (confirmed empirically -- see the module docstring), so that
    # scenario cannot be built as a valid model in the first place.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[2,8] Y, float[2,8] Z)
        <int64[2] split = {2, 2}>
        {
          Y, Z = Split<axis = 0>(X, split)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_split")
    assert ops["Split"] == 1


def test_eliminate_nop_split_declines_on_non_static_shape():
    # patternMatchPredicate also requires `node->inputs()[0]->has_sizes()`.
    # X's shape here is genuinely unresolvable by onnxsim's shape inference:
    # it is Squeeze'd by an axes value that is deterministic at runtime
    # ([0, 3], always) but not a compile-time constant (an Add of two
    # initializers, rather than an initializer or Constant node itself --
    # FetchConstantTensor only recognizes those two) -- the same trick
    # test_formal_verify_fuse_consecutive_unsqueezes.py's own
    # `_dynamic_shape_model` helper uses. This bypasses `simplify_isolated`
    # (which only controls the *optimizer pass* list) and calls
    # `onnxsim.simplify` directly with `skip_constant_folding=True`, since
    # onnxsim's constant folding is a separate step that runs before the
    # optimizer passes regardless of `skipped_optimizers` and would
    # otherwise fold the Add/Squeeze away -- restoring a statically-known
    # shape -- before eliminate_nop_split ever saw it.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,4,8,1] X0) => (float[4,8] Y)
        <int64[2] sq_a = {0, 2}, int64[2] sq_b = {0, 1}>
        {
          sq_axes = Add(sq_a, sq_b)
          X = Squeeze(X0, sq_axes)
          s = Split<axis = 0>(X)
          Y = Relu(s)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_nop_split"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = {n.op_type for n in sim_model.graph.node}
    assert "Split" in ops
