"""Formal check for RenameInputOutput (opt-in; ``rename_input_output.h``):
renames every graph-level *input* Value's own name to a pattern-generated
name (default ``input_%d``, ``%d`` = 0-based positional index; customizable
via the ``OPTIMIZER_RENAME_INPUT_PATTERN`` environment variable, read once
via ``getenv``), skipping any input whose name is also an initializer name
(the same "optional input with a default value" exclusion
``eliminate_duplicate_initializer`` applies to its own inputs -- see
``test_formal_verify_eliminate_duplicate_initializer.py``). It does the same
for every graph-level *output* (default ``output_%d``,
``OPTIMIZER_RENAME_OUTPUT_PATTERN``), with **no** such initializer-name
exclusion for outputs -- confirmed empirically below, this is a real,
deliberate asymmetry, not an oversight to route around.

Like ``eliminate_duplicate_initializer`` and ``nop``, this is a
``FullGraphBasedPass`` (whole-graph ``runPass(Graph&)``), not a
``PredicateBasedPass``.

Formal content, and why the proof here is intentionally thin: this is a
purely cosmetic/labeling rewrite, not an algebraic identity about operator
semantics. ``rename_input_output`` calls ``value->setUniqueName(new_name)``
on the graph-boundary ``Value`` objects themselves. A ``Value``'s uses (which
nodes read/write it) are tracked by object identity in ONNX's in-memory IR,
not by name lookup -- so every node that used to consume/produce that Value
under its old name automatically consumes/produces the *same* Value object
under its new name. No node is added or removed, no node's attributes
change, and no edge is rewired: only a string field on some boundary Value
objects changes. The claim is modeled below as: an uninterpreted
"compute-by-position" function ``compute(i)`` -- standing for "whatever
value the data-flow graph produces at boundary position ``i``", determined
purely by nodes-and-edges-by-object-identity -- that, by construction, is
not a function of any name-of-position function at all, so it is invariant
under an arbitrary change to the naming functions ``name_before``/
``name_after``. This is nearly tautological once ONNX's identity-based (not
name-based) edge representation is taken as given -- stated honestly as
such, in the same spirit as ``test_formal_verify_nop.py``'s and
``test_formal_verify_eliminate_duplicate_initializer.py``'s own thin proofs,
rather than dressed up with machinery this pass has no real use for. There
is no meaningful "wrong version" of a pure rename to use as a negative
control (unlike e.g. a fusion whose algebra could plausibly be gotten
wrong); the practical soundness check that actually matters for this pass is
the differential test running the *renamed* model through onnxruntime and
confirming the numeric output is unchanged, addressed by the new names.

Surprises found empirically while developing this test (see the throwaway
debug script mentioned in this session, not committed):

- ``simplify_isolated``/``simplify_isolated_extra``'s default ``check_n=3``
  cannot be used with this pass at all: onnxsim's own internal equivalence
  check (``model_checking.compare``) always feeds its randomly generated
  inputs to *both* the original and simplified model using the *original*
  model's input names -- but this pass deliberately changes the simplified
  model's input names, so onnxruntime's ``session.run`` on the simplified
  model raises ``ValueError: Required inputs [...] are missing from input
  feed [...]`` before any comparison happens. Every differential test below
  therefore passes ``check_n=0`` explicitly (falls back to a structural
  ``onnx.checker.check_model`` only) and instead does its own numeric check
  via onnxruntime, run once against each model addressed by its own correct
  (old vs. new) input/output names.
- The input-name-is-also-an-initializer-name exclusion has the same
  reachability subtlety as ``eliminate_duplicate_initializer``'s own input
  exclusion: onnxsim's Python-level preprocessing
  (``remove_initializer_from_input``) unconditionally strips any
  ``graph.input`` entry whose name duplicates an initializer name *before*
  the model reaches the C++ optimizer -- confirmed empirically, without
  countermeasures the duplicate input entry is simply gone by the time this
  pass runs, so its own initializer-name guard is never exercised. Passing
  ``mutable_initializer=True`` (plumbed straight to the C++ core, not part
  of ``simplify_isolated``'s signature) disables that stripping and is what
  lets the exclusion test below actually reach the pass's own guard, calling
  ``onnxsim.simplify`` directly (reusing ``isolate()`` for the skip list)
  instead of the shared helper.
- The per-input/per-output index used in the rename pattern is the input's
  or output's raw 0-based *position* in ``graph.inputs()``/``graph.outputs()``
  -- confirmed empirically: skipping an excluded (initializer-named) input
  does not renumber the inputs that come after it. An excluded input at
  position 0 leaves the input at position 1 renamed to ``input_1``, not
  ``input_0``.
- No such preprocessing exists for graph outputs, and outputs get no
  initializer-name exclusion at all: a graph output produced directly by an
  initializer (no producing node, the same pattern
  ``test_formal_verify_eliminate_duplicate_initializer.py`` uses for its own
  output-exclusion test) *is* renamed by this pass -- and since the same
  ``Value`` object backs both the ``graph.output`` entry and the
  initializer tensor, the initializer entry's name changes right along with
  it. Confirmed empirically below.
- Environment-variable customization (``OPTIMIZER_RENAME_INPUT_PATTERN`` /
  ``OPTIMIZER_RENAME_OUTPUT_PATTERN``) round-trips cleanly through
  ``simplify_isolated_extra`` with no other countermeasures needed; the test
  below uses pytest's ``monkeypatch`` fixture, which restores the prior
  environment automatically (including a fully absent variable) even if the
  test body raises.
"""

import numpy as np
import onnxruntime as ort
from _formal_verify_common import isolate, prove, simplify_isolated_extra, z3
from onnx import numpy_helper, parser

import onnxsim


def test_rename_input_output_output_by_position_is_unaffected():
    # `compute` stands for "the value the data-flow graph produces at
    # boundary position i", determined purely by nodes and edges tracked by
    # object identity -- and, crucially, it is syntactically NOT a function
    # of any naming function: it only ever takes the position `i`. That is
    # exactly the structural fact that makes renaming sound: whatever a
    # graph computes at a given input/output position cannot depend on what
    # string some Value's uniqueName field happens to hold.
    #
    # name_before/name_after model the pass's actual, real effect on the
    # graph (some position's name really does change) -- the claim survives
    # even when they are forced apart at position i, precisely because
    # `compute` never reads a name in the first place.
    compute = z3.Function("compute", z3.IntSort(), z3.RealSort())
    name_before = z3.Function("name_before", z3.IntSort(), z3.StringSort())
    name_after = z3.Function("name_after", z3.IntSort(), z3.StringSort())
    i = z3.Int("i")

    renamed_at_i = name_before(i) != name_after(i)
    prove(z3.Implies(renamed_at_i, compute(i) == compute(i)))


def test_rename_input_output_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "rename_input_output" in C._list_other_optimizers()
    assert "rename_input_output" not in C._list_optimizers()


def test_rename_input_output_pass_matches_default_pattern_and_preserves_numerics():
    # Two plain (non-initializer) inputs, one output: the compiled pass,
    # isolated, should rename both inputs and the output to the default
    # patterns, leaving the node structure (same op, same edges by
    # position) completely untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X, float[2,2] Y) => (float[2,2] Z)
        {
          Z = Add(X, Y)
        }
        """
    )
    # check_n=0: onnxsim's own random-input equivalence check feeds the
    # simplified model using the *original* input names, which this pass
    # deliberately changes -- see module docstring. The numeric check below
    # (run via onnxruntime against each model's own correct names) is the
    # real soundness check for this pass.
    sim_model, ops = simplify_isolated_extra(model, "rename_input_output", check_n=0)

    assert [i.name for i in sim_model.graph.input] == ["input_0", "input_1"]
    assert [o.name for o in sim_model.graph.output] == ["output_0"]
    assert ops == {"Add": 1}
    (add_node,) = sim_model.graph.node
    assert list(add_node.input) == ["input_0", "input_1"]
    assert list(add_node.output) == ["output_0"]

    # Numeric check: run both the original and the renamed model with the
    # same data, addressed by each model's own (old vs. new) names, and
    # confirm the results agree exactly -- the data flow is untouched, only
    # the boundary labels changed.
    rng = np.random.default_rng(0)
    x = rng.random((2, 2), dtype=np.float32)
    y = rng.random((2, 2), dtype=np.float32)

    orig_sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (orig_out,) = orig_sess.run(None, {"X": x, "Y": y})

    new_sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (new_out,) = new_sess.run(None, {"input_0": x, "input_1": y})

    np.testing.assert_array_equal(orig_out, new_out)


def test_rename_input_output_skips_input_that_is_also_initializer():
    # Bias's name is also an initializer name (the "optional input with a
    # default value" pattern) -- Bias must be left untouched while X (a
    # plain input) and the output are still renamed.
    #
    # mutable_initializer=True is required to reach this guard at all:
    # onnxsim's default preprocessing strips any graph.input entry that
    # duplicates an initializer name *before* the C++ optimizer ever runs
    # (see module docstring) -- with the default mutable_initializer=False,
    # Bias's duplicate input entry would already be gone by the time this
    # pass runs, defeating the point of this test (confirmed empirically).
    # simplify_isolated_extra doesn't expose mutable_initializer, so this
    # calls onnxsim.simplify directly, reusing isolate() for the skip list.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X, float[2,2] Bias) => (float[2,2] Z)
        {
          Z = Add(X, Bias)
        }
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2),
            name="Bias",
        )
    )

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=["rename_input_output"],
        skipped_optimizers=isolate(),
        mutable_initializer=True,
    )
    assert check_ok

    assert [i.name for i in sim_model.graph.input] == ["input_0", "Bias"]
    assert [o.name for o in sim_model.graph.output] == ["output_0"]
    (add_node,) = sim_model.graph.node
    assert list(add_node.input) == ["input_0", "Bias"]  # Bias untouched
    assert list(add_node.output) == ["output_0"]


def test_rename_input_output_index_is_positional_not_renumbered_after_skip():
    # Bias (excluded, position 0) comes first; X (a plain input, position 1)
    # comes second. If the index were renumbered after skipping excluded
    # inputs, X would become input_0; instead it stays input_1, confirming
    # the pattern's %d is the raw 0-based position, not a "count of renamed
    # inputs so far".
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] Bias, float[2,2] X) => (float[2,2] Z)
        {
          Z = Add(X, Bias)
        }
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2),
            name="Bias",
        )
    )

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=["rename_input_output"],
        skipped_optimizers=isolate(),
        mutable_initializer=True,
    )
    assert check_ok
    assert [i.name for i in sim_model.graph.input] == ["Bias", "input_1"]


def test_rename_input_output_has_no_initializer_exclusion_for_outputs():
    # Y is a graph output produced directly by an initializer (no producing
    # node) -- the same pattern used for eliminate_duplicate_initializer's
    # own output-exclusion test -- but unlike that pass, rename_input_output
    # applies NO initializer-name exclusion to outputs: Y is renamed anyway,
    # and since the same Value object backs both the graph.output entry and
    # the initializer tensor, the initializer's own name changes with it.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Y, float[2,2] W)
        <float[2,2] Y = {1.0, 2.0, 3.0, 4.0}>
        {
          W = Relu(X)
        }
        """
    )

    sim_model, ops = simplify_isolated_extra(model, "rename_input_output", check_n=0)

    assert [i.name for i in sim_model.graph.input] == ["input_0"]
    assert [o.name for o in sim_model.graph.output] == ["output_0", "output_1"]
    assert [i.name for i in sim_model.graph.initializer] == ["output_0"]
    assert ops == {"Relu": 1}
    (relu_node,) = sim_model.graph.node
    assert list(relu_node.input) == ["input_0"]
    assert list(relu_node.output) == ["output_1"]


def test_rename_input_output_env_var_customizes_pattern(monkeypatch):
    # OPTIMIZER_RENAME_INPUT_PATTERN/OPTIMIZER_RENAME_OUTPUT_PATTERN are read
    # once via getenv inside the pass; monkeypatch.setenv restores the prior
    # environment (including "was unset") automatically after the test, even
    # if it raises, so no manual try/finally is needed here.
    monkeypatch.setenv("OPTIMIZER_RENAME_INPUT_PATTERN", "my_in_%d")
    monkeypatch.setenv("OPTIMIZER_RENAME_OUTPUT_PATTERN", "my_out_%d")

    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Z)
        {
          Z = Relu(X)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rename_input_output", check_n=0)

    assert [i.name for i in sim_model.graph.input] == ["my_in_0"]
    assert [o.name for o in sim_model.graph.output] == ["my_out_0"]
    assert ops == {"Relu": 1}
