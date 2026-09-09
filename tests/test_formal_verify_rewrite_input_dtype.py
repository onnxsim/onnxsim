"""Formal check for RewriteInputDtype (opt-in;
``rewrite_input_dtype.h``): a ``FullGraphBasedPass`` that, for every
graph-level *input* (not an initializer, not an interior value) whose
declared element type is INT64, inserts a new ``Cast<to=INT64>`` node
reading that input, redirects every existing consumer of the input to read
the Cast's output instead, and then re-declares the input's *own* type as
INT32 (``value->setElemType(TensorProto_DataType_INT32)``). An input that is
also an initializer name (the same "optional input with a default value"
exclusion ``rename_input_output``/``eliminate_duplicate_initializer`` apply
to their own inputs) is skipped outright -- confirmed by re-reading the C++
above: ``initializer_names.count(value->uniqueName()) > 0`` is checked
first, short-circuiting before the dtype check even runs.

Note the direction each type moves: the *inserted Cast*'s ``to`` attribute
is INT64 (casting back up), while the *input*'s own declared type becomes
INT32 (narrower) -- so the graph ends up declaring this input as
int32-typed, but immediately upcasts it back to int64 before anything else
reads it. A grep of this repo turns up no caller, doc, or comment beyond the
pass's own ~25 lines explaining why; the plausible reading (stated here as
an inference, not a documented fact) is that this is a compatibility shim
for a runtime/binding that can only bind int32-typed model inputs, while the
graph's own internal computation still needs int64 (ONNX's shape/indexing
ops are int64-heavy) -- a mismatch that shows up e.g. between certain
mobile/edge runtimes' preferred integer width and ONNX's own conventions.

**Formal content -- and a real subtlety this pass's own code does not flag**:
naively this looks like ``rename_input_output``'s kind of cosmetic,
unconditionally value-preserving relabeling. It is NOT that in general. The
inserted Cast's job is to undo the just-introduced INT32 declaration by
casting back to INT64 -- but "declaring the input as INT32" is not a free
annotation: it changes what value the *runtime* is actually allowed to bind
there. A caller that hands this rewritten model an actual int32-range int64
value gets the same computation as before (proved below, under an explicit
range hypothesis). A caller whose real int64 data does *not* fit in int32
(e.g. a large shape/index value, or any value at or past 2**31) gets
something else entirely: truncating to int32 and back is a two's-complement
*wraparound*, not a lossless round trip, for such a value. This is a real,
provable precondition of this pass's own soundness that its C++ implementation
does not state or check anywhere -- exactly the kind of gap a translation-
validation proof is for surfacing, not a reason to soften the claim below:

1. **Positive claim, under an explicit hypothesis.** ONNX's own ``Cast``
   semantics define integer-to-integer casting as an exact, lossless
   conversion whenever the value is representable in both the source and
   destination types (no rounding or quantization is involved at all, unlike
   e.g. a float16 round trip). int32's representable range,
   ``[-2**31, 2**31 - 1]``, is a strict subset of int64's, so for any actual
   input value ``v`` inside that range, casting down to int32 and back up to
   int64 is the identity: ``Cast<to=INT64>(Cast<to=INT32>(v)) == v``. The
   hypothesis ``-2**31 <= v < 2**31`` is modeled explicitly below as a real
   precondition of the claim, not left as an unstated assumption.
2. **Negative control: outside that range, it is provably NOT the identity.**
   Truncating an out-of-range int64 value to int32 wraps around
   (two's-complement truncation -- the same semantics as
   ``static_cast<int32_t>(v)`` in the C++ this pass's own runtime consumer
   presumably uses, and as onnxruntime/numpy's own int64->int32 casts). This
   is modeled directly as the standard truncate-then-sign-extend modular
   formula, ``int32_roundtrip(v) := ((v + 2**31) mod 2**32) - 2**31``
   (identical to the round trip above wherever the range hypothesis holds,
   since ``Cast<to=INT64>`` widening int32 back to int64 is always itself
   exact -- int32 is a strict subset of int64), and Z3 confirms it is *not*
   the identity once ``v`` leaves int32's range: ``v = 2**31`` -- one past
   int32's max -- truncates to ``-2**31``, a genuine, concrete wraparound.
3. The value-preserving direction (1) is composed with an arbitrary
   uninterpreted ``consumer``, this suite's usual substitution-safety idiom.

The differential tests below run the actual compiled pass and confirm: it
fires on an INT64 input (inserting the Cast and flipping the input's own
declared type, with the original consumer now reading the Cast's output);
it preserves the graph's numeric behavior for an in-range value (feeding the
*original* model an int64 array and the *rewritten* model an int32 array of
the same numbers, since the rewritten model's own declared input dtype has
changed -- the same countermeasure ``test_formal_verify_rename_input_output.py``
uses for its own boundary-relabeling pass, and for the same reason: onnxsim's
built-in random-input equivalence check feeds by the *original* model's
declared dtype, so it can't be used unmodified here either, confirmed
empirically below via ``check_n=0``); it leaves an input alone when that
input's name is also an initializer name, even though its dtype is INT64
(the short-circuited initializer-name check); and it leaves non-INT64 inputs
(FLOAT, already-INT32) completely untouched.
"""

import numpy as np
import onnxruntime as ort
from _formal_verify_common import isolate, producer, prove, simplify_isolated_extra, z3
from onnx import TensorProto, numpy_helper, parser

import onnxsim


def test_rewrite_input_dtype_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "rewrite_input_dtype" in C._list_other_optimizers()
    assert "rewrite_input_dtype" not in C._list_optimizers()


def _int32_roundtrip(v):
    """Two's-complement truncate-to-int32-then-sign-extend-back-to-int64,
    expressed as modular arithmetic: the same operation the pass's inserted
    ``Cast<to=INT64>(Cast<to=INT32>(v))`` performs at the value level, once
    the input's own declared type has been narrowed to INT32.
    """
    return ((v + 2**31) % (2**32)) - 2**31


def test_rewrite_input_dtype_is_sound_within_int32_range():
    # The explicit precondition this pass's own soundness actually rests on,
    # stated as a real hypothesis rather than left implicit: the runtime
    # value must already fit in int32's representable range. Under that
    # hypothesis, truncating to int32 and casting back to int64 is exact.
    v = z3.Int("v")
    consumer = z3.Function("consumer", z3.IntSort(), z3.IntSort())

    in_range = z3.And(v >= -(2**31), v < 2**31)
    roundtrip = _int32_roundtrip(v)

    prove(z3.Implies(in_range, roundtrip == v))
    prove(z3.Implies(in_range, consumer(roundtrip) == consumer(v)))


def test_rewrite_input_dtype_negative_control_out_of_range_wraps():
    # Without the range hypothesis, the round-trip claim is NOT a theorem:
    # Z3 must find a genuine counterexample, confirming the hypothesis in
    # the test above is load-bearing, not decorative.
    v = z3.Int("v")
    solver = z3.Solver()
    solver.add(_int32_roundtrip(v) != v)
    assert solver.check() == z3.sat, (
        "round trip holds for every v with no range hypothesis -- "
        "negative control is vacuous"
    )

    # A concrete demonstration of the same fact, independent of Z3: v =
    # 2**31 is one past int32's max (2**31 - 1). Two's-complement truncation
    # wraps it all the way around to int32's *minimum*, -2**31 -- not merely
    # "some other value", but the specific, well-known wraparound behavior
    # this formula (and real int64->int32 casts) actually exhibit.
    v_concrete = 2**31
    assert _int32_roundtrip(v_concrete) == -(2**31)
    assert _int32_roundtrip(v_concrete) != v_concrete


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def test_rewrite_input_dtype_pass_rewrites_input_and_preserves_numerics_in_range():
    # A single INT64 input feeding a Cast<to=FLOAT> consumer -- the pass,
    # isolated, should insert its own Cast<to=INT64> reading X, redirect the
    # original Cast to read that new Cast's output, and flip X's own
    # declared type to INT32.
    model = _model(
        """
        g (int64[4] X) => (float[4] Y)
        {
          Y = Cast<to = 1>(X)
        }
        """
    )
    # check_n=0: onnxsim's own random-input equivalence check feeds inputs
    # by the *original* model's declared dtype (int64), but this pass
    # changes the rewritten model's own input dtype to int32 -- see module
    # docstring. The numeric check below (each model fed its own correct
    # dtype) is the real soundness check here.
    sim_model, ops = simplify_isolated_extra(model, "rewrite_input_dtype", check_n=0)
    assert ops == {"Cast": 2}

    (x_input,) = [i for i in sim_model.graph.input if i.name == "X"]
    assert x_input.type.tensor_type.elem_type == TensorProto.INT32

    inserted_cast = next(
        n
        for n in sim_model.graph.node
        if n.op_type == "Cast" and list(n.input) == ["X"]
    )
    (to_attr,) = [a for a in inserted_cast.attribute if a.name == "to"]
    assert to_attr.i == TensorProto.INT64

    y_node = producer(sim_model, "Y")
    assert y_node.op_type == "Cast"
    assert list(y_node.input) == [inserted_cast.output[0]]
    assert y_node.input[0] != "X"

    # Numeric check: an in-range int64 value round-trips through
    # int32-truncate-then-int64-widen exactly, so the rewritten model's
    # output should match the original model's, value for value, given the
    # SAME numbers -- fed as int64 to the original (its declared dtype) and
    # as int32 to the rewritten model (its own new declared dtype).
    rng = np.random.default_rng(0)
    x = rng.integers(-1000, 1000, size=4).astype(np.int64)

    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (orig_out,) = orig_sess.run(None, {"X": x})

    new_sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (new_out,) = new_sess.run(None, {"X": x.astype(np.int32)})

    np.testing.assert_array_equal(orig_out, new_out)


def test_rewrite_input_dtype_skips_input_that_is_also_initializer():
    # Bias is INT64 (would otherwise qualify) but its name is also an
    # initializer name (the "optional input with a default value"
    # convention) -- the pass's own initializer-name check short-circuits
    # before the dtype check ever runs, so Bias must be left completely
    # untouched while X (a plain INT64 input) is still rewritten.
    #
    # mutable_initializer=True is required to reach this guard at all:
    # onnxsim's default preprocessing strips any graph.input entry that
    # duplicates an initializer name *before* the C++ optimizer ever runs
    # (the same reachability subtlety documented in
    # test_formal_verify_rename_input_output.py) -- without it, Bias's
    # duplicate input entry would already be gone by the time this pass
    # runs. simplify_isolated_extra doesn't expose mutable_initializer, so
    # this calls onnxsim.simplify directly, reusing isolate() for the skip
    # list.
    model = _model(
        """
        g (int64[4] X, int64[4] Bias) => (int64[4] Z)
        {
          Z = Add(X, Bias)
        }
        """,
        initializer=[
            numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64), "Bias")
        ],
    )

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=["rewrite_input_dtype"],
        skipped_optimizers=isolate(),
        mutable_initializer=True,
    )
    assert check_ok

    inputs_by_name = {i.name: i for i in sim_model.graph.input}
    assert inputs_by_name["X"].type.tensor_type.elem_type == TensorProto.INT32
    assert inputs_by_name["Bias"].type.tensor_type.elem_type == TensorProto.INT64

    add_node = producer(sim_model, "Z")
    assert add_node.input[1] == "Bias"  # untouched: no Cast, same name
    assert add_node.input[0] != "X"  # redirected to the inserted Cast's output

    inserted_cast = next(
        n
        for n in sim_model.graph.node
        if n.op_type == "Cast" and list(n.input) == ["X"]
    )
    assert add_node.input[0] == inserted_cast.output[0]
    # No Cast reads Bias at all.
    assert not any(
        n.op_type == "Cast" and list(n.input) == ["Bias"] for n in sim_model.graph.node
    )


def test_rewrite_input_dtype_skips_non_int64_inputs():
    # Neither input is INT64 (one FLOAT, one already INT32) -- the
    # predicate's dtype check declines both, so the pass should be a
    # complete no-op: no Cast inserted, no input's declared type touched.
    model = _model(
        """
        g (float[4] X, int32[4] W) => (float[4] Y, int32[4] Z)
        {
          Y = Identity(X)
          Z = Identity(W)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rewrite_input_dtype")
    assert ops == {"Identity": 2}

    inputs_by_name = {i.name: i for i in sim_model.graph.input}
    assert inputs_by_name["X"].type.tensor_type.elem_type == TensorProto.FLOAT
    assert inputs_by_name["W"].type.tensor_type.elem_type == TensorProto.INT32

    y_node = producer(sim_model, "Y")
    z_node = producer(sim_model, "Z")
    assert list(y_node.input) == ["X"]
    assert list(z_node.input) == ["W"]
