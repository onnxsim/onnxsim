"""The Python half of the Python<->C++ step-graph emitter parity check.

``onnxsim/qat_graph_builder.{h,cpp}`` re-implements the emitter half of
:mod:`onnxsim.qat_graph` in C++, so the browser converter can build a training
step graph without a Python round trip. Two implementations of one emitter that
quietly disagree is the whole hazard of having done that: both graphs would be
valid, both would run, and the browser would train a model differently from the
Python with nothing saying so.

``onnxsim/qat_parity_fixtures.txt`` is the shared reference both sides are
measured against. This file asserts *fixture == Python*; the C++
``qat_graph_parity_test`` asserts *fixture == C++*. Together they give
Python == C++, which is the property actually wanted and which neither test
establishes on its own.

The failure this file exists to catch is specifically **a stale fixture**. The
C++ test alone cannot distinguish "the port is correct" from "the port matches
a fixture that stopped describing the Python three commits ago" -- in the
second case the C++ test still passes, cheerfully, while the two emitters have
diverged. So changing :mod:`onnxsim.qat_graph`'s emission without regenerating
must fail *here*, loudly, with instructions.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GENERATOR = os.path.join(_ROOT, "scripts", "make_qat_parity_fixtures.py")
_FIXTURE = os.path.join(_ROOT, "onnxsim", "qat_parity_fixtures.txt")

_REGENERATE = (
    "Run `python3 scripts/make_qat_parity_fixtures.py` and commit the result "
    "-- and if onnxsim/qat_graph_builder.cpp is meant to emit the same thing, "
    "update it in the same change, because the C++ parity test will now fail "
    "against the new fixture."
)


def _generator():
    """The fixture generator, imported by path.

    It lives in ``scripts/`` rather than in the package, so there is no import
    to do; loading it by path keeps the generator and this test reading the
    same case definitions instead of two copies that can disagree about what
    they describe.
    """
    spec = importlib.util.spec_from_file_location("_qat_parity_gen", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _generator()


@pytest.fixture(scope="module")
def committed_text():
    with open(_FIXTURE) as f:
        return f.read()


@pytest.fixture(scope="module")
def committed(gen, committed_text):
    """The freshly-built structure, for the tests below that inspect one case.

    Not parsed back out of the committed text -- the text has no parser on
    either side, by design. Using the rebuilt structure here is only sound
    because ``test_the_committed_fixture_still_describes_what_python_emits``
    pins it to the committed text; if that test fails, treat every assertion
    below it as describing the working tree rather than the fixture.
    """
    return gen.build()


@pytest.fixture(scope="module")
def regenerated_text(gen):
    return gen.render(gen.build())


def test_the_committed_fixture_still_describes_what_python_emits(
    committed_text, regenerated_text
):
    """The fixture is not stale.

    If this fails, :mod:`onnxsim.qat_graph`'s emission changed and the
    reference did not follow it. That is only a problem to *fix* rather than a
    bug per se -- but leaving it unfixed would quietly turn the C++ parity test
    into a test of nothing, since it would keep agreeing with a description of
    an emitter that no longer exists.
    """
    assert regenerated_text == committed_text, (
        "onnxsim/qat_parity_fixtures.txt no longer matches what "
        "onnxsim/qat_graph.py emits. " + _REGENERATE
    )


def test_every_case_is_present_on_both_sides(committed_text, regenerated_text):
    """Cases are not silently dropped.

    A whole-object comparison already covers this, but it reports as one
    enormous diff; naming the missing case makes the common edit (adding a
    builder method and a case for it, then forgetting to regenerate) diagnose
    itself.
    """

    def cases(text):
        return sorted(
            line.split(" ", 1)[1]
            for line in text.splitlines()
            if line.startswith("case ")
        )

    assert cases(committed_text) == cases(regenerated_text), _REGENERATE


def test_the_fixture_pins_the_operator_allowlist(committed_text):
    """``EP_FRIENDLY_OPS`` is part of the contract, not just the graphs.

    The C++ restates the allowlist, and a member present on one side only would
    let one emitter produce a graph the other's own tests reject. Pinning it in
    the shared fixture makes that a parity failure rather than something nobody
    notices until a WebGPU run falls over.
    """
    from onnxsim.qat_graph import EP_FRIENDLY_OPS

    line = next(line for line in committed_text.splitlines() if line.startswith("ops "))
    assert line.split(" ", 1)[1].split(",") == sorted(EP_FRIENDLY_OPS), _REGENERATE


def test_no_case_emits_an_operator_outside_the_allowlist(committed):
    """The fixture cannot itself assert something false.

    If a case emitted an op outside the allowlist, the C++ test would be
    verifying parity on a graph that the allowlist says should never have been
    built -- so the reference would be enforcing agreement on a bug.
    """
    from onnxsim.qat_graph import EP_FRIENDLY_OPS

    emitted = {
        node["op_type"]
        for case in committed["cases"].values()
        for node in case.get("nodes", [])
    }
    assert not (emitted - set(EP_FRIENDLY_OPS))


def test_the_rounding_case_contains_no_round_node(committed):
    """The composed rounding stayed composed.

    ``Round`` is absent from ``EP_FRIENDLY_OPS`` because WebNN has no rounding
    operator at all, which is the entire reason
    :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` spells it out of
    Abs/Add/Cast/Sign/Cast/Mul. A port -- or a future simplification on either
    side -- that "cleaned this up" into a single ``Round`` would pass every
    numerical test and be unrunnable on the backends the allowlist exists for.
    """
    ops = [node["op_type"] for node in committed["cases"]["round_to_nearest"]["nodes"]]
    assert "Round" not in ops
    assert ops == ["Abs", "Add", "Cast", "Sign", "Cast", "Mul"]


def test_adams_one_minus_beta_constants_are_computed_in_double(committed_text):
    """The narrowing order is pinned, because it is invisible and it matters.

    ``1 - beta`` is computed in double precision and then narrowed to float32.
    Doing the subtraction in float32 instead lands on a different number --
    0.100000024 rather than 0.1 -- which would leave both emitters producing
    perfectly valid graphs that take subtly different optimizer steps forever
    after. Nothing else in either test suite would notice.
    """
    # The fixture writes floats as IEEE-754 bit patterns precisely so this
    # distinction is legible. Both constants are checked, and each against the
    # *specific* value the mistake produces rather than merely "not the right
    # one" -- a wrong-value assertion naming a pattern that can never occur
    # passes forever and tests nothing.
    #   1 - 0.9   : 0x3dcccccd narrowed from double, 0x3dccccd0 in float32
    #   1 - 0.999 : 0x3a83126f narrowed from double, 0x3a831200 in float32
    assert "0x3dcccccd" in committed_text
    assert "0x3dccccd0" not in committed_text
    assert "0x3a83126f" in committed_text
    assert "0x3a831200" not in committed_text
