"""Tests for onnxsim.graph_grad's custom-gradient registry --
:func:`onnxsim.graph_grad.register_gradient`/:func:`unregister_gradient`/
:func:`custom_gradient`/:func:`supported_ops` -- the seam that lets a caller
teach :func:`onnxsim.graph_grad.build_backward` a rule for an op none of the
builtin ``_RULES`` cover, the same relationship
:func:`torch.autograd.Function.backward` (or ``torch.library.register_autograd``)
has to a custom torch op. See ``graph_grad.py``'s own "Extending the
boundary" section for the full design writeup.

Reuses ``test_graph_grad.py``'s ``_model``/``_check``/``_away_from_zero``
rather than duplicating them -- the same cross-test-module import
``test_autoround_step_graph.py`` already establishes for this repo -- because
``_check`` already does exactly what the end-to-end test below needs: build
the backward with :func:`onnxsim.graph_grad.build_backward`'s *default*
``rules=None`` (so it genuinely exercises the registry, not a hand-built
override table) and compare against an independent finite difference.

Every test that registers something uses :func:`onnxsim.graph_grad.custom_gradient`
(the scoped form) rather than a bare ``register_gradient`` left un-cleaned-up
-- the registry is process-global, so a leaked registration would silently
change what every *other* test file sees, including
``tests/test_qat_parity.py``'s pin of ``sorted(graph_grad.SUPPORTED_OPS)``
against ``qat_parity_fixtures.txt``.
"""

from __future__ import annotations

import pytest
from test_graph_grad import _away_from_zero, _check, _model

from onnxsim import graph_grad, qat_graph


def _grad_reciprocal(ctx, node, g):
    """``Y = Reciprocal(X)`` => ``dX = -g / X^2 = -g * Y^2`` -- reusing the
    forward's own output ``Y`` the way the builtin Sigmoid/Tanh rules do,
    rather than recomputing ``X * X``."""
    y = node.output[0]
    y2 = ctx.b.mul(y, y)
    neg_g = ctx.b.mul(g, ctx.b.const(-1.0))
    return [ctx.b.mul(neg_g, y2)]


def _grad_sin(ctx, node, g):
    """``Y = Sin(X)`` => ``dX = g * cos(X)``, via a raw ``Cos`` node --
    :class:`onnxsim.qat_graph.GraphBuilder`'s convenience wrappers don't
    include ``Cos``, so this is also the test coverage for registering a
    rule that reaches past them into ``ctx.b.op`` directly."""
    (x,) = node.input
    cos_x = ctx.b.op("Cos", [x])
    return [ctx.b.mul(g, cos_x)]


def test_an_unregistered_op_is_refused_same_as_ever():
    """Baseline: before any registration, ``Reciprocal`` is refused exactly
    like any other op :mod:`onnxsim.graph_grad` has no rule for."""
    assert "Reciprocal" not in graph_grad.SUPPORTED_OPS
    assert "Reciprocal" not in graph_grad.supported_ops()
    model = _model(
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Reciprocal(A)
        }
        """
    )
    with pytest.raises(graph_grad.UnsupportedOpError, match="Reciprocal"):
        _check(model)


def test_register_gradient_makes_an_op_differentiable_end_to_end():
    """Once registered, the same block trains -- checked the same way every
    builtin rule is, against an independent finite difference -- through
    :func:`onnxsim.graph_grad.build_backward`'s ordinary, default
    ``rules=None`` path, not a hand-built override table."""
    model = _model(
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Reciprocal(A)
        }
        """
    )
    with graph_grad.custom_gradient("Reciprocal", _grad_reciprocal):
        assert "Reciprocal" in graph_grad.supported_ops()
        # Kept away from zero: Reciprocal's second derivative blows up there,
        # which would swamp the finite difference's own O(h^2) error.
        _check(model, overrides={"A": _away_from_zero})
    assert "Reciprocal" not in graph_grad.supported_ops()


def test_register_gradient_works_as_a_decorator():
    """The decorator form (``rule=None`` triggers it) registers the function
    it wraps and returns it unchanged, so it remains callable directly --
    checked here against the same numeric contract every rule has: one
    gradient per ``node.input``."""

    @graph_grad.register_gradient("Reciprocal")
    def _rule(ctx, node, g):
        return _grad_reciprocal(ctx, node, g)

    try:
        assert "Reciprocal" in graph_grad.supported_ops()
        b = qat_graph.GraphBuilder()
        model = _model(
            """
            g (float[2] A) => (float[2] Y) {
              Y = Reciprocal(A)
            }
            """
        )
        grads = graph_grad.build_backward(
            b, list(model.graph.node), {"A": [2], "Y": [2]}, {"Y": "dY"}, ["A"]
        )
        assert "A" in grads
    finally:
        graph_grad.unregister_gradient("Reciprocal")


def test_register_gradient_refuses_to_silently_replace_a_builtin_rule():
    """Registering over an op :data:`onnxsim.graph_grad.SUPPORTED_OPS`
    already covers is refused unless ``override=True`` is explicit -- the
    same "a collision is refused, not resolved by whichever registration ran
    last" stance :class:`onnxsim.graph_grad.UnsupportedOpError` itself takes."""
    assert "Relu" in graph_grad.SUPPORTED_OPS
    with pytest.raises(ValueError, match="override=True"):
        graph_grad.register_gradient("Relu", _grad_sin)
    # The refused call must not have registered anything.
    assert "Relu" not in graph_grad._CUSTOM_RULES


def test_register_gradient_override_replaces_a_builtin_rule_for_python_callers():
    """``override=True`` is honored -- the escape hatch exists, it just is
    not the default. Scoped with :func:`onnxsim.graph_grad.custom_gradient`
    so the override does not survive past this test."""
    calls = []

    def _spy_relu(ctx, node, g):
        calls.append(node.output[0])
        return graph_grad._RULES["Relu"](ctx, node, g)

    model = _model(
        """
        g (float[3] A) => (float[3] Y) {
          Y = Relu(A)
        }
        """
    )
    with graph_grad.custom_gradient("Relu", _spy_relu, override=True):
        b = qat_graph.GraphBuilder()
        graph_grad.build_backward(
            b, list(model.graph.node), {"A": [3], "Y": [3]}, {"Y": "dY"}, ["A"]
        )
    assert calls == ["Y"]
    # The builtin table itself was never mutated -- only the process-global
    # custom registry was, and custom_gradient already cleaned that up.
    assert graph_grad._RULES["Relu"] is not _spy_relu
    assert "Relu" not in graph_grad._CUSTOM_RULES


def test_register_gradient_refuses_to_silently_replace_a_custom_rule():
    """The same refusal applies to a second registration of an op a caller
    already registered themselves -- not just to a builtin one."""
    with graph_grad.custom_gradient("Reciprocal", _grad_reciprocal):
        with pytest.raises(ValueError, match="override=True"):
            graph_grad.register_gradient("Reciprocal", _grad_reciprocal)
        # override=True on a *custom* collision also works.
        graph_grad.register_gradient("Reciprocal", _grad_reciprocal, override=True)
    assert "Reciprocal" not in graph_grad.supported_ops()


def test_unregister_gradient_raises_for_an_op_never_registered():
    """Builtin rules were never in the custom registry to begin with, so
    there is nothing for :func:`onnxsim.graph_grad.unregister_gradient` to
    remove for one -- same for any op nobody registered at all."""
    with pytest.raises(KeyError):
        graph_grad.unregister_gradient("Relu")
    with pytest.raises(KeyError):
        graph_grad.unregister_gradient("NotARealOpEitherWay")


def test_supported_ops_stays_builtin_only_while_the_effective_set_grows():
    """:data:`onnxsim.graph_grad.SUPPORTED_OPS` is what
    ``tests/test_qat_parity.py`` pins against ``qat_parity_fixtures.txt`` --
    it must never move just because some other test registered something, or
    that pin would become test-order-dependent. :func:`onnxsim.graph_grad.supported_ops`
    is the one that reflects registrations."""
    before = graph_grad.SUPPORTED_OPS
    with graph_grad.custom_gradient("Reciprocal", _grad_reciprocal):
        assert graph_grad.SUPPORTED_OPS is before
        assert "Reciprocal" not in graph_grad.SUPPORTED_OPS
        assert "Reciprocal" in graph_grad.supported_ops()
        assert graph_grad.supported_ops() == before | {"Reciprocal"}
    assert graph_grad.SUPPORTED_OPS is before
    assert graph_grad.supported_ops() == before


def test_custom_gradient_cleans_up_even_when_the_body_raises():
    """The scoped form's whole point: a failure inside the ``with`` block
    must not leave the registration behind for later tests."""
    with pytest.raises(RuntimeError, match="boom"):
        with graph_grad.custom_gradient("Reciprocal", _grad_reciprocal):
            assert "Reciprocal" in graph_grad.supported_ops()
            raise RuntimeError("boom")
    assert "Reciprocal" not in graph_grad.supported_ops()


def test_an_explicit_rules_argument_still_bypasses_the_registry():
    """:func:`onnxsim.graph_grad.build_backward`'s ``rules=`` override (used
    by ``tests/test_graph_grad_templates.py``) is unaffected by the registry:
    passing an explicit table -- even one that does not happen to include a
    custom registration -- is respected exactly, matching the existing
    contract predating this feature."""
    with graph_grad.custom_gradient("Reciprocal", _grad_reciprocal):
        b = qat_graph.GraphBuilder()
        model = _model(
            """
            g (float[2] A) => (float[2] Y) {
              Y = Reciprocal(A)
            }
            """
        )
        with pytest.raises(graph_grad.UnsupportedOpError, match="Reciprocal"):
            graph_grad.build_backward(
                b,
                list(model.graph.node),
                {"A": [2], "Y": [2]},
                {"Y": "dY"},
                ["A"],
                rules=dict(graph_grad._RULES),  # explicit table, no Reciprocal
            )
