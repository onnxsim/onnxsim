"""Formal check for LiftLexicalReferences (opt-in;
``lift_lexical_references.h``): walks every ``If``/``Loop`` control-flow node
and its subgraph body/bodies, and for every value referenced *inside* a body
that is actually defined in some *enclosing* (lexical parent) scope, adds
that value's name to a new, non-standard node attribute -- ``__control_inputs``,
a list of strings -- on the control-flow node itself. Per the header
comment's own algorithm (``liftReferences``): a body's own formal inputs
(``iter_num``/``cond`` for ``Loop``, no extra inputs for ``If``) and anything
the body produces itself are resolved within the body's own frame and never
become control inputs; only genuine outer-scope captures do. A name that
resolves in *neither* the body's own frame nor any enclosing frame is a
genuinely unresolved reference and makes ``runPass`` throw
``std::runtime_error`` -- the header comment's documented signal for "this
graph is malformed", not something this pass itself ever creates (see below
for why that path isn't independently exercisable here).

Like ``eliminate_duplicate_initializer``, ``nop``, and ``rename_input_output``,
this is a ``FullGraphBasedPass`` (whole-graph ``runPass(Graph&)``); its
``PassType`` is ``Separate`` (opt-in, confirmed empirically below via
``_list_other_optimizers()``/``_list_optimizers()``, mirroring
``rename_input_output``'s own such test).

Formal content, and why the proof here is intentionally thin: this pass
**does not change any computed value in the graph at all**. It never
rewires a node's ``inputs()``/``outputs()``, never touches a subgraph's own
body nodes, and never adds a graph input/output -- it only writes a new,
non-standard attribute (a list of strings) onto the control-flow node whose
body did the capturing. The header comment is explicit that this "yields a
graph that does not conform to the ONNX spec" (``__control_inputs`` isn't a
real ``If``/``Loop`` attribute per the operator schemas), but it is harmless
to actual execution: an unrecognized attribute is simply ignored by
onnxruntime and by onnx's own reference evaluator, and existing lexical
scoping -- an ``If``/``Loop`` body already implicitly "sees" outer-scope
values by name lookup through the enclosing graph, per ONNX's own IR/
interpreter semantics -- is exactly what's being exposed here, not altered.
So the claim to formalize is not an operator-semantics algebra (there is
none to get right or wrong); it is: attaching an inert, execution-irrelevant
metadata field to a node cannot change what that node -- or, by extension,
the graph composed with anything downstream -- computes, because nothing in
ONNX's execution semantics ever reads that field. This is modeled below as
an uninterpreted ``annotate(inputs, metadata)`` function standing for "the
node's real output, augmented with an irrelevant metadata slot", constrained
by an *explicit* axiom that it is metadata-invariant (``annotate`` never
actually varies with its second argument) -- because that axiom **is**
precisely the real, checkable claim about this pass: its own new attribute
is never consulted by anything that computes the graph's values. This is
intentionally abstract/generic (per this suite's established convention for
"thin" proofs, e.g. ``rename_input_output``'s and ``nop``'s own docstrings)
rather than dressed up with per-operator machinery this pass has no use for;
a second, ``Loop``-body-flavored restatement of the same argument is
included as a bonus, since the header comment specifically motivates the
pass with a per-iteration-body-scheduling use case.

Given how thin the algebra is, essentially all of the real verification
value is in the differential tests below, run against the actually compiled
pass:

- **The clean case (``If``)**: ``simplify_isolated_extra`` (isolate this one
  opt-in pass, ``check_n=3``) works completely normally for an ``If`` model
  whose ``then``/``else`` branches each capture a distinct outer-scope value
  -- unlike ``rename_input_output``, this pass never changes any graph-level
  input/output name, so onnxsim's own random-input equivalence check (which
  always addresses inputs/outputs by the *original* model's names) applies
  with no countermeasures needed. Confirmed empirically below: the resulting
  ``If`` node's ``__control_inputs`` attribute contains exactly the two
  genuine outer-scope captures, and a value produced *inside* a branch
  (including one purely local to that branch, referenced only by another
  node in the same branch) is correctly excluded.
- **The ``Loop`` case needs a countermeasure, but not because of anything
  ``lift_lexical_references`` itself does**: reproducing the header
  comment's own worked example (a ``Loop`` whose body captures two outer
  values) via ``simplify_isolated_extra`` fails with
  ``RuntimeError: Unresolved value references: _v_8,`` -- *not* a lexical
  capture inside the body, but the ``Loop`` node's own ``trip_count``/
  ``condition`` inputs. Found empirically: onnxsim's constant-folding stage
  (a preprocessing step wholly independent of ``skipped_optimizers``/
  ``extra_optimizers`` -- it isn't one of the named optimizer passes at all)
  folds the scalar ``Constant`` nodes producing ``trip_count``/``condition``
  into freshly-named graph initializers (e.g. ``_v_8``) *before*
  ``lift_lexical_references`` ever runs. Per ``ir_pb_converter.cc``/``ir.h``,
  such an initializer becomes a ``Value`` created via
  ``addInitializerAndCreateValue`` off a special captured/undefined node --
  which is exactly the node kind this pass's own ``liftReferences`` loop
  explicitly skips (``kUndefined``/``kCaptured``, "Skip optional input/
  captured value node") without registering its outputs in the environment
  stack at all. So the folded initializer is invisible to the top-level
  frame the pass builds from ``g->inputs()``, and the ``Loop`` node's own
  (perfectly ordinary, non-lexical) input becomes "unresolved" purely as an
  artifact of *when* constant folding ran relative to this pass -- correctly
  triggering the pass's own documented failure mode, but for a reason that
  has nothing to do with lexical scoping. ``skip_constant_folding=True``
  avoids this (confirmed empirically below); since ``simplify_isolated_extra``
  doesn't expose that parameter, the ``Loop`` differential test calls
  ``onnxsim.simplify`` directly instead, reusing ``isolate()`` for the skip
  list -- the same fallback pattern ``rename_input_output``'s and
  ``eliminate_duplicate_initializer``'s own test files use for parameters
  their shared helpers don't plumb through.
- **The genuinely-unresolved-reference path is not independently
  exercisable via the public API, confirmed empirically**: a body subgraph
  node reading a name that is undefined *anywhere* (not merely uncaptured)
  is already rejected by ONNX's own generic topological-sort/structural
  graph validity checking -- which ``onnxsim.simplify`` applies to every
  model regardless of which passes are requested -- before the model ever
  reaches this pass's ``runPass``. So there is no dedicated negative-control
  differential test for that ``std::runtime_error`` path here: exercising it
  would only demonstrate ONNX's general graph-validity checking, not
  anything specific to this pass's own algorithm. (The Z3 side does still
  carry its own, meaningful negative control -- see
  ``test_lift_lexical_references_negative_control_needs_metadata_invariance_axiom``
  below -- confirming the *soundness argument itself* is not vacuously true.)

On subgraph model construction: per ``CLAUDE.md``, ``onnx.parser.parse_model()``
was tried first and confirmed (empirically, standalone) to support both
``If`` and ``Loop`` nodes with nested body ``GraphProto``s, including bodies
that reference outer-scope names -- so the text form is used throughout below
rather than falling back to ``onnx.helper``.
"""

import numpy as np
import onnxruntime as ort
from _formal_verify_common import isolate, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim


def _control_inputs(node):
    """The node's ``__control_inputs`` strings attribute value, or None."""
    for attr in node.attribute:
        if attr.name == "__control_inputs":
            return sorted(s.decode() for s in attr.strings)
    return None


def test_lift_lexical_references_is_sound():
    # annotate(inputs, metadata) stands for "the node's real output value,
    # with an extra inert metadata slot (the new __control_inputs list)
    # alongside its genuine inputs". The axiom below is the actual claim
    # this pass's soundness rests on: annotate never actually varies with
    # its metadata argument, for any inputs and any two metadata values --
    # i.e. nothing that computes the graph's values ever reads
    # __control_inputs. Given that axiom, a downstream consumer composed
    # with annotate cannot tell which metadata was attached.
    inputs, m1, m2 = z3.Ints("inputs m1 m2")
    annotate = z3.Function("annotate", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    metadata_invariant = z3.ForAll(
        [inputs, m1, m2], annotate(inputs, m1) == annotate(inputs, m2)
    )
    prove(
        z3.Implies(
            metadata_invariant,
            consumer(annotate(inputs, m1)) == consumer(annotate(inputs, m2)),
        )
    )


def test_lift_lexical_references_negative_control_needs_metadata_invariance_axiom():
    # Without the metadata-invariance axiom, annotate is a fully
    # unconstrained uninterpreted function of two arguments: the claim must
    # NOT be valid then (Z3 must find a counterexample where two different
    # metadata values really do produce different consumer results), or the
    # proof above would be vacuously true regardless of what the axiom says
    # -- exactly the same shape of negative control
    # eliminate_duplicate_initializer's own test file uses for its
    # content-equality hypothesis.
    inputs, m1, m2 = z3.Ints("inputs m1 m2")
    annotate = z3.Function("annotate", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    solver = z3.Solver()
    solver.add(m1 != m2)
    solver.add(z3.Not(consumer(annotate(inputs, m1)) == consumer(annotate(inputs, m2))))
    assert solver.check() == z3.sat


def test_lift_lexical_references_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "lift_lexical_references" in C._list_other_optimizers()
    assert "lift_lexical_references" not in C._list_optimizers()


def test_lift_lexical_references_if_captures_outer_scope_values():
    # then_branch captures X via an extra purely-local node (L, referenced
    # only by T within the same branch -- confirms a genuinely local value
    # is excluded, not just a branch's own output); else_branch captures Y.
    # Neither branch has any input of its own (If's body subgraphs take no
    # formal inputs at all), so every name a branch's nodes read that isn't
    # one of that branch's own node outputs must resolve as an outer-scope
    # capture -- exactly the case this pass exists to expose.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[5] X, bool Cond) => (float[5] Y, float[5] Z)
        {
          Y = Identity(X)
          Z = If(Cond) <
            then_branch = then_graph () => (float[5] T) {
              L = Identity(X)
              T = Identity(L)
            },
            else_branch = else_graph () => (float[5] E) {
              E = Identity(Y)
            }
          >
        }
        """
    )
    # ops only counts top-level graph.node op types -- the branch-local
    # Identity nodes (L, T, E) live inside the If node's own subgraph
    # attributes, not the top-level node list.
    sim_model, ops = simplify_isolated_extra(
        model, "lift_lexical_references", check_n=3
    )
    assert ops == {"Identity": 1, "If": 1}

    (if_node,) = [n for n in sim_model.graph.node if n.op_type == "If"]
    # Exactly the two genuine outer-scope captures -- no local value (L, T,
    # E), and no double-counting across the two branches.
    assert _control_inputs(if_node) == ["X", "Y"]

    # onnxsim's own random-input equivalence check (via simplify_isolated_extra
    # above, check_n=3) already confirmed this, but a direct onnxruntime
    # comparison against both branches makes the "graph's own computed
    # outputs are numerically identical before/after" claim explicit and
    # addresses it by the (unrenamed, by this pass) original input/output
    # names -- as rename_input_output's own test file does for its
    # differential checks that simplify_isolated's default doesn't cover.
    rng = np.random.default_rng(0)
    x = rng.random(5, dtype=np.float32)
    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    new_sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for cond in (True, False):
        feed = {"X": x, "Cond": np.array(cond)}
        orig_y, orig_z = orig_sess.run(None, feed)
        new_y, new_z = new_sess.run(None, feed)
        np.testing.assert_array_equal(orig_y, new_y)
        np.testing.assert_array_equal(orig_z, new_z)


def test_lift_lexical_references_loop_captures_outer_scope_values():
    # Adapted directly from the header comment's own worked example: the
    # Loop body captures both X (the outer graph input) and Y (an outer
    # node's output), while _Y2/_Y3 (produced inside the body) and the
    # body's own formal inputs (i, cond) must NOT be treated as captures.
    #
    # skip_constant_folding=True and a direct onnxsim.simplify call (instead
    # of simplify_isolated_extra) are both required to reach this pass's
    # actual capture logic here -- see the module docstring's "The Loop case
    # needs a countermeasure" section: without it, onnxsim's own constant
    # folding turns trip_count/condition into freshly-named initializers
    # *before* this pass runs, and the Loop node's own (non-lexical) input
    # to those initializers is reported as "unresolved" -- an artifact of
    # constant-folding ordering, not a lexical-capture bug.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[5] X) => (float[5] Y, float[5] Y2, float[5] Y3)
        {
          Y = Identity(X)
          trip_count = Constant<value = int64{3}>()
          condition = Constant<value = bool{1}>()
          Y2, Y3 = Loop(trip_count, condition) <
            body = body_graph (int64 i, bool cond) => (bool cond_out, float[5] _Y2, float[5] _Y3)
            {
              _Y2 = Identity(X)
              _Y3 = Identity(Y)
              cond_out = Identity(cond)
            }
          >
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        extra_optimizers=["lift_lexical_references"],
        skipped_optimizers=isolate(),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    (loop_node,) = [n for n in sim_model.graph.node if n.op_type == "Loop"]
    # Exactly the two genuine outer-scope captures -- neither _Y2/_Y3 (local
    # to the body) nor i/cond (the body's own formal parameters, already
    # resolved within its own frame) appear.
    assert _control_inputs(loop_node) == ["X", "Y"]

    # Direct onnxruntime comparison, same rationale as the If test above:
    # makes the "graph's own computed outputs are unchanged" claim explicit,
    # independent of onnxsim's own internal check_ok above.
    rng = np.random.default_rng(0)
    x = rng.random(5, dtype=np.float32)
    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    new_sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    orig_outs = orig_sess.run(None, {"X": x})
    new_outs = new_sess.run(None, {"X": x})
    for orig_out, new_out in zip(orig_outs, new_outs):
        np.testing.assert_array_equal(orig_out, new_out)
