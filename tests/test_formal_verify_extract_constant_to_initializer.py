"""Formal check for ExtractConstantToInitializer (extract_constant_to_initializer.h).

``patternMatchPredicate`` matches a ``Constant`` node that carries a ``value``
(Tensor) attribute. ``runTransform`` takes that embedded tensor and moves it
into the GRAPH's initializer list as a new named value, rewires every
consumer of the Constant node's output onto that initializer, and destroys
the Constant node. (The ``reserved_names_``/``nextReservedName``/
``initializePass`` machinery in the header is purely a batched-name-
reservation performance optimization with no semantic effect -- not modeled
or tested here.)

Formal content, and why the proof here is intentionally thin (the same
framing ``test_formal_verify_eliminate_duplicate_initializer.py`` uses for
its own comparable claim): this is a definitional-equality proof, not an
algebraic derivation. ONNX's own semantics define a ``Constant`` node's
output as being exactly the tensor embedded in its ``value`` attribute, and
separately define a graph INITIALIZER's associated value as being exactly
the tensor it holds -- both are simply "this name denotes this fixed
tensor," with no operational difference between the two as far as any
CONSUMER of the name is concerned. The pass just changes which graph-level
mechanism supplies a given fixed value, not the value itself. Modeled below
as: an uninterpreted tensor function ``T : Int -> Real`` standing for the
embedded constant's content, two ways of naming/supplying that same content
(``constant_node_output`` and ``initializer_value``, both constrained to
equal ``T`` pointwise -- mirroring how ``eliminate_duplicate_initializer``'s
proof modeled "two initializer entries with the same content"), and an
arbitrary uninterpreted ``consumer``. The content-equality hypothesis does
essentially all the work here -- there is no nontrivial algebra to prove.
The negative control below (independent, unconstrained
``constant_node_output``/``initializer_value``, no hypothesis) confirms the
claim is not vacuously true regardless of that hypothesis.

A structural surprise, found while writing this file's differential tests,
that has no counterpart in any other file in this suite: unlike every other
targeted pass, ``extract_constant_to_initializer`` is UNREACHABLE through
onnxsim's own Python API, by deliberate design. ``onnxsim.cpp``'s
``SimplifyImpl`` builds ``config.optimizer_passes`` from
``onnx::optimization::GetFuseAndEliminationPass()`` filtered through a fixed
``always_disabled_passes = {"eliminate_shape_gather",
"extract_constant_to_initializer"}`` list -- and that filter is applied not
only to the default pass set but also to ``extra_optimizers`` (the general
opt-in mechanism every other non-default pass in this suite is exercised
through), so there is no combination of ``skipped_optimizers``/``isolate()``
and/or ``extra_optimizers`` that makes it run. The comment right above that
list explains why: onnxsim's own constant folder deliberately leaves a
``Constant`` node in producer form (rather than baking it into an
initializer) so a fold's origin stays visible in the output model, and
running this pass would erase that distinction right back -- including for
``Constant`` nodes onnxsim's own folder just produced -- so it is always
dropped, unconditionally.

Consequently, unlike every sibling file in this suite, there is no way to
run "the real compiled pass, in isolation" against a concrete model *through
onnxsim's own public Python surface* -- the surface this suite otherwise
insists on testing against (see CLAUDE.md and every other
``test_formal_verify_*.py`` file). An earlier version of this file drove
onnxoptimizer's ``ExtractConstantToInitializer`` class directly via a small
C++ harness compiled at test time against this checkout's
``.setuptools-cmake-build`` artifacts; that approach was dropped because it
only works when a source build's ``compile_commands.json`` and static
libraries happen to be present in the CWD -- true in a local dev checkout,
but never true in CI's actual test job (``CIBW_TEST_COMMAND`` in
``.github/workflows/build-and-test.yml`` installs a prebuilt wheel and runs
``pytest`` against it, with no leftover CMake build tree or C++ toolchain
available), so those tests would silently skip on every real CI run and
provide no ongoing verification value there, while adding real fragility
(compiler flags, link order, static-lib layout) for a local-only benefit.

Instead, ``test_extract_constant_to_initializer_unreachable_via_onnxsim_
simplify`` below tests the property that actually matters for onnxsim's own
users and behavior -- that this pass never fires through the public API --
directly and unconditionally (no special build machinery, so it always runs
in CI): if onnxsim's own ``always_disabled_passes`` filter (``onnxsim.cpp``)
ever stopped excluding this pass, this is the test that would catch it, by
observing a ``Constant`` node unexpectedly disappearing in favor of a fresh
initializer.
"""

from _formal_verify_common import isolate, prove, z3
from onnx import parser

import onnxsim


def test_extract_constant_to_initializer_is_sound():
    T = z3.Function("T", z3.IntSort(), z3.RealSort())
    constant_node_output = z3.Function(
        "constant_node_output", z3.IntSort(), z3.RealSort()
    )
    initializer_value = z3.Function("initializer_value", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    # "constant_node_output and initializer_value denote the same tensor
    # content": both equal the shared, uninterpreted content function T at
    # every index -- the ONNX-spec-level fact that a Constant node's value
    # attribute and an initializer entry are both just "this name denotes
    # this fixed tensor".
    same_content = z3.ForAll(
        [j],
        z3.And(
            constant_node_output(j) == T(j),
            initializer_value(j) == T(j),
        ),
    )
    prove(
        z3.Implies(
            same_content,
            consumer(constant_node_output(i)) == consumer(initializer_value(i)),
        )
    )


def test_extract_constant_to_initializer_negative_control_needs_hypothesis():
    # Without the content-equality hypothesis, constant_node_output and
    # initializer_value are two independent, fully unconstrained
    # uninterpreted functions: the claim must NOT be valid then, or the
    # "proof" above would be vacuously true regardless of what same_content
    # says. Z3 should find a sat counterexample negating the claim.
    constant_node_output = z3.Function(
        "constant_node_output", z3.IntSort(), z3.RealSort()
    )
    initializer_value = z3.Function("initializer_value", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i = z3.Int("i")

    solver = z3.Solver()
    solver.add(
        z3.Not(consumer(constant_node_output(i)) == consumer(initializer_value(i)))
    )
    assert solver.check() == z3.sat


def test_extract_constant_to_initializer_unreachable_via_onnxsim_simplify():
    # Regression guard for the module docstring's central claim: unlike
    # every other pass in this suite, no combination of onnxsim.simplify's
    # own knobs runs this pass. isolate("extract_constant_to_initializer")
    # skips every OTHER default pass, nominally leaving only this one
    # active -- and skip_constant_folding=True keeps onnxsim's separate
    # constant folder from folding Relu(Constant) away on its own, which
    # would otherwise obscure whether this pass itself ever touched
    # anything. If onnxsim's own always_disabled_passes filter (onnxsim.cpp)
    # ever stopped excluding this pass, this test would start failing here
    # (Constant would disappear and an initializer would appear) rather than
    # silently changing onnxsim's constant-node-preservation guarantee.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Y)
        {
            A = Constant <value = float[4] {1.0, 2.0, 3.0, 4.0}> ()
            Y = Relu(A)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("extract_constant_to_initializer"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    assert list(sim_model.graph.initializer) == []
    ops = {n.op_type for n in sim_model.graph.node}
    assert ops == {"Constant", "Relu"}
