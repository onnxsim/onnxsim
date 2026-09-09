"""Formal check for FuseConcatIntoReshape (fuse_concat_into_reshape.h).

``patternMatchPredicate`` matches ``Reshape(X, Concat(seg_0, ..., seg_n))``
(``matchConcatReshape``, the Concat's own ``axis`` attribute must be 0) or
``Reshape(X, Cast(Concat(seg_0, ..., seg_n), to=INT64))``
(``matchConcatCastReshape`` -- same Concat shape, with an intervening Cast to
INT64 in between).

``runTransform`` walks the Concat's own input segments ``v`` one at a time,
building a flat ``std::vector<int64_t> shapes``:

* if ``FetchConstantTensor(v)`` succeeds (``v`` is a ``Constant`` node or a
  constant-treated initializer), its raw values are read out and converted
  to ``int64_t`` (via the ``DO_CASE`` switch, which accepts FLOAT/INT32/
  INT64/DOUBLE/UINT8/INT8/UINT16/INT16/UINT32/UINT64) and appended to
  ``shapes``. Without an intervening Cast the tensor must *already* be
  INT64-typed or the pass declines outright -- moot in valid ONNX, though,
  since ``Concat`` requires every input to share one element type, and
  Reshape's own shape input must be INT64, so a non-cast Concat feeding
  Reshape directly can only ever be all-INT64 to begin with.
* otherwise (``v`` is not a compile-time constant): if ``v`` has statically
  known shape ``[1]`` (exactly one element), a literal ``-1`` is appended to
  ``shapes`` *in place of* ``v``'s real, unknown-at-compile-time value --
  ONNX Reshape's own "infer this axis" sentinel. If ``v`` is non-constant
  and does NOT have exactly one element, ``runTransform`` gives up
  (``return false``) outright; the header's own TODO notes a smarter
  partial-segment case (``[?, 2, 3]`` concatenated with ``[4]`` producing
  ``[-1, 2, 3, 4]``) is not implemented, only whole-segment substitution.

After the walk, if more than one ``-1`` ended up in ``shapes``, the pass
declines (``unknown_dim_count > 1``) -- the same restriction
``eliminate_nop_reshape``'s own ``-1``-inference reason (c) imposes, since
Reshape's ``-1`` is only well-defined with exactly one unknown axis.
Otherwise it builds a brand-new INT64 initializer holding exactly ``shapes``
and rewires the ``Reshape``'s second input to it directly
(``node->replaceInput(1, value)``), with ``destroy_current =
NodeDestroyType::DestroyZero`` -- the old ``Concat``/``Cast`` chain is left
dangling in the graph, not itself destroyed, the same dangling-producer
pattern already seen in e.g. ``fuse_consecutive_slices`` and
``fuse_pad_into_conv``.

Formal content, in two parts:

  (a) **plain constant folding** (trivial/definitional): for every segment
      the pass reads via ``FetchConstantTensor``, that call's own guarantee
      *is* that the fetched compile-time value equals the value the segment
      evaluates to at runtime -- a compile-time constant is, by
      construction, the same value every time the graph runs. Substituting
      the folded literal for the live ``Concat`` segment therefore changes
      nothing observable. There is no interesting algebra here beyond this
      definitional fact (and, for the Cast variant, that the numeric
      cast/parse round-trips exactly for the small integer-valued literals
      any real target shape uses -- exercised empirically below via a
      FLOAT-typed Concat segment converted through the Cast path).

  (b) **the nontrivial -1-inference argument** (the interesting content,
      shared in spirit with ``eliminate_nop_reshape``'s own reason (c)):
      for the *at most one* non-constant, single-element segment that gets
      replaced by a literal ``-1``, soundness rests on exactly the same
      auxiliary lemma ``test_formal_verify_eliminate_nop_reshape.py`` proves
      for its own ``-1``-inference case -- ``new_dim == -1 ∧
      other_axes_pinned ∧ total_size_preserved ⇒ inferred_dim ==
      true_dim_at_that_axis`` -- adapted here to a setup where the "true
      dim" being inferred is not ``old_shape[i]`` (Reshape's own input
      shape, as in the sibling file) but the *Concat segment*'s true,
      unknown-at-compile-time-but-deterministic-at-runtime value. The
      argument is otherwise identical: every *other* axis of the target
      shape is now a literal, folded-constant value (part (a) above), so
      Reshape's own size-preservation law (the fused shape's element count
      must equal ``X``'s total element count -- exactly what made the
      *original*, pre-fusion ``Reshape(X, Concat(...))`` a valid reshape in
      the first place, since the original target shape's product must
      *also* equal that same total) pins the resolved ``-1`` axis to
      exactly the unknown segment's true value. Once that is established,
      the fused shape list and the original (pre-fusion) shape list are
      *literally the same list of numbers*, so the two Reshapes trivially
      compute the same output -- unlike the sibling file, no row-major
      flatten/unflatten argument is needed here to bridge two
      differently-shaped-looking descriptions of the same shape, because
      here both sides land on the exact same shape after the lemma is
      applied. This is modeled below with a 3-axis target shape ``[c0, u1,
      c2]`` -- two known/constant-folded axes ``c0``, ``c2`` flanking one
      unknown, single-element axis ``u1`` -- concrete enough to exercise
      part (a) (``c0``, ``c2``) combining with part (b) (``u1`` /
      ``resolved1``) in one shape, the same way the sibling file's rank-2
      example was "enough to exercise all three reasons" for its own
      proof. Reshape's flat-buffer semantics itself (input and output share
      one row-major buffer of ``D`` elements whenever the target shape's
      product is ``D``) is modeled directly as a 1-D uninterpreted function
      ``x: Int -> Real`` indexed by the flattened offset, rather than via
      the sibling's 2-D div/mod unflattening -- the two shapes being
      compared here (fused vs. original) are reshapes of the *same* ``X``
      into shapes that the lemma proves are numerically identical, so a
      single flat-buffer view suffices; there is no second, differently
      laid out original shape to bridge against as in the sibling proof.

A key empirical surprise (see the differential tests below): a *fully*
constant ``Concat`` feeding ``Reshape`` -- and a ``Concat`` segment derived
from ``Shape(X)`` (used here to construct a genuinely non-constant,
single-element segment) -- are BOTH resolved away by onnxsim's own separate
constant-folding step (``_EvalPartialShape`` + the ordinary constant folder,
see ``onnxsim.cpp``'s ``FoldConstant``) before ``fuse_concat_into_reshape``
ever gets a chance to run, when that step is left enabled (as
``simplify_isolated`` always leaves it -- it has no ``skip_constant_folding``
knob, see ``tests/_formal_verify_common.py``). ``_EvalPartialShape``
specifically turns ``Shape``/``Gather``-on-shape into constants even for a
plain (non-initializer, non-``Constant``) graph input, since it works from
static shape *metadata*, not from the value having to already be a
compile-time constant per ``IsConstantTensor``. So every differential test
below bypasses ``simplify_isolated`` and calls ``onnxsim.simplify`` directly
with ``skip_constant_folding=True`` -- the same workaround
``test_formal_verify_fuse_consecutive_slices.py`` already uses to inspect a
fusion pass's own raw output structure rather than a structure any separate
folding step could have produced instead. With constant folding off, a
``Shape<start=0,end=1>(X)`` segment survives as a real, un-resolved node
feeding ``Concat`` -- non-constant per ``FetchConstantTensor`` (it is
neither a ``Constant`` node nor a constant-treated initializer), with a
statically-known-shape-``[1]`` output -- exactly the shape this pass's own
``-1``-substitution branch is written to handle, confirmed below to fire.
"""

import collections

from _formal_verify_common import isolate, producer, prove, z3
from onnx import numpy_helper, parser

import onnxsim


def test_fuse_concat_into_reshape_is_sound():
    # x: the flat, row-major data buffer any target shape with the right
    # total element count reinterprets -- ONNX Reshape's own semantics is
    # exactly "same flat buffer, different shape labeling."
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    c0, c2 = z3.Ints("c0 c2")  # the two constant-folded (part (a)) axes
    u1 = z3.Int("u1")  # the unknown segment's true runtime value
    resolved1 = z3.Int("resolved1")  # the fused Reshape's own -1 inference
    D = z3.Int("D")  # X's total element count
    i, j, k = z3.Ints("i j k")

    def reshape_val(s0, s1, s2, ii, jj, kk):
        flat = (ii * s1 + jj) * s2 + kk
        return x(flat)

    domain = z3.And(
        c0 > 0, c2 > 0, u1 > 0, 0 <= i, i < c0, 0 <= j, j < u1, 0 <= k, k < c2
    )

    # The pre-fusion graph is valid ONNX: Reshape(X, Concat(c0, u1, c2))
    # already has target-shape product == X's total size D. This is what
    # made the *original* Concat->Reshape a legal reshape in the first
    # place -- not something the pass invents.
    original_reshape_is_valid = c0 * u1 * c2 == D

    # Reshape's own size-preservation law applied to the FUSED shape
    # [c0, -1, c2]: whatever resolved1 the runtime infers for the sole -1
    # axis must make the fused target shape's product equal D too.
    size_preservation_law = c0 * resolved1 * c2 == D

    prove(
        z3.Implies(
            z3.And(domain, original_reshape_is_valid, size_preservation_law),
            consumer(reshape_val(c0, resolved1, c2, i, j, k))
            == consumer(reshape_val(c0, u1, c2, i, j, k)),
        )
    )


def _simplify_isolated_no_cf(model, check_n=3):
    # Like _formal_verify_common.simplify_isolated, but with constant
    # folding (and the partial-shape-eval step nested inside it) turned off
    # -- see this file's docstring for why that is required to observe
    # fuse_concat_into_reshape's own runTransform output rather than a
    # structure onnxsim's separate folding step already produced.
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        skipped_optimizers=isolate("fuse_concat_into_reshape"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _shape_tensor(sim_model, reshape_node):
    # The fused Reshape's second input, resolved to a numpy array via the
    # fresh initializer runTransform creates for it.
    initializers = {i.name: i for i in sim_model.graph.initializer}
    shape_input = reshape_node.input[1]
    assert shape_input in initializers, (
        f"expected {shape_input!r} to be a fresh constant initializer"
    )
    return numpy_helper.to_array(initializers[shape_input])


def test_fuse_concat_into_reshape_pass_matches_fully_constant():
    # Concat(a, b) is entirely compile-time constant (both INT64
    # initializers) -- runTransform should fold it directly into a fresh
    # [2, 3, 4] initializer and rewire Reshape's second input to it,
    # leaving the old Concat dangling (DestroyZero; eliminate_deadend is
    # skipped by isolate()).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <int64[2] a = {2, 3}, int64[1] b = {4}>
        {
          shp = Concat<axis=0>(a, b)
          Y = Reshape(X, shp)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1
    assert ops["Concat"] == 1  # the old Concat is left dangling, not destroyed

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.op_type == "Reshape"
    assert reshape_node.input[1] != "shp"  # rewired away from the old Concat
    assert list(_shape_tensor(sim_model, reshape_node)) == [2, 3, 4]

    dangling = [n for n in sim_model.graph.node if n.output[0] == "shp"]
    assert len(dangling) == 1 and dangling[0].op_type == "Concat"


def test_fuse_concat_into_reshape_pass_matches_mixed_constant_and_unknown():
    # dim0 = Shape<start=0,end=1>(X): a genuinely non-constant (per
    # FetchConstantTensor), statically-shaped-[1] segment -- with constant
    # folding off (see this file's docstring), it survives as a real Shape
    # node rather than being resolved by _EvalPartialShape. Concatenated
    # with two INT64 constants b=3, c=4. runTransform should still fire,
    # substituting -1 for dim0's own unknown-at-compile-time value.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <int64[1] b = {3}, int64[1] c = {4}>
        {
          dim0 = Shape<start=0,end=1>(X)
          shp = Concat<axis=0>(dim0, b, c)
          Y = Reshape(X, shp)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1
    assert ops["Shape"] == 1  # dangling
    assert ops["Concat"] == 1  # dangling

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.input[1] != "shp"
    # -1 stands in for dim0's own (unknown-to-the-pass, but actually 2)
    # value; b, c pass through as plain folded constants.
    assert list(_shape_tensor(sim_model, reshape_node)) == [-1, 3, 4]


def test_fuse_concat_into_reshape_pass_matches_cast_variant():
    # Same fully-constant Concat as the basic case, but with an
    # intervening Cast<to=INT64> -- exercises matchConcatCastReshape
    # instead of matchConcatReshape. Must fire and land on the exact same
    # [2, 3, 4] result.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <int64[2] a = {2, 3}, int64[1] b = {4}>
        {
          shp = Concat<axis=0>(a, b)
          shp64 = Cast<to=7>(shp)
          Y = Reshape(X, shp64)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1
    assert ops["Concat"] == 1  # dangling
    assert ops["Cast"] == 1  # dangling

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.input[1] != "shp64"
    assert list(_shape_tensor(sim_model, reshape_node)) == [2, 3, 4]


def test_fuse_concat_into_reshape_pass_matches_cast_variant_float_segments():
    # Same as the Cast variant above, but the Concat's own segments are
    # FLOAT-typed -- only legal ONNX because of the intervening Cast (a
    # non-cast Concat->Reshape requires INT64 throughout, per Concat's own
    # single-element-type rule and Reshape's INT64-only shape input).
    # Exercises runTransform's DO_CASE FLOAT->int64_t conversion path,
    # confirming the numeric round-trip is exact for these constants.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <float[2] a = {2.0, 3.0}, float[1] b = {4.0}>
        {
          shp = Concat<axis=0>(a, b)
          shp64 = Cast<to=7>(shp)
          Y = Reshape(X, shp64)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.input[1] != "shp64"
    assert list(_shape_tensor(sim_model, reshape_node)) == [2, 3, 4]


def test_fuse_concat_into_reshape_declines_two_unknown_segments():
    # Two independent Shape-derived single-element segments (dim0, dim1)
    # plus one constant: unknown_dim_count would reach 2, so runTransform
    # declines outright (more than one -1 is not well-defined for
    # Reshape) -- Reshape's second input is left pointing at the original
    # (un-fused) Concat output.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <int64[1] b = {4}>
        {
          dim0 = Shape<start=0,end=1>(X)
          dim1 = Shape<start=1,end=2>(X)
          shp = Concat<axis=0>(dim0, dim1, b)
          Y = Reshape(X, shp)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1
    assert ops["Concat"] == 1
    assert ops["Shape"] == 2

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.input[1] == "shp"  # unchanged: the pass declined


def test_fuse_concat_into_reshape_declines_multi_element_unknown_segment():
    # dims01 = Shape<start=0,end=2>(X) is non-constant AND has 2 elements
    # (not the eligible-for-substitution shape [1]) -- runTransform's own
    # `v->sizes().size() != 1` check trips and it declines outright,
    # regardless of unknown_dim_count. Reshape's second input is left
    # pointing at the original Concat output.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 15]
        >
        g (float[2,3,4] X) => (float[?,?,?] Y)
        <int64[1] b = {4}>
        {
          dims01 = Shape<start=0,end=2>(X)
          shp = Concat<axis=0>(dims01, b)
          Y = Reshape(X, shp)
        }
        """
    )
    sim_model, ops = _simplify_isolated_no_cf(model)
    assert ops["Reshape"] == 1
    assert ops["Concat"] == 1
    assert ops["Shape"] == 1

    reshape_node = producer(sim_model, "Y")
    assert reshape_node.input[1] == "shp"  # unchanged: the pass declined
