"""Tests cross-checking onnxsim.graph_grad's *hand-written* gradient rules
against ``torch.autograd`` on the identical math -- extending the coverage
tests/test_graph_grad_templates.py established for the templated
BatchNormalization rule (and tests/test_lora.py for the LoRA composition's
MatMul/Add) to the rest of :data:`onnxsim.graph_grad.SUPPORTED_OPS`.

tests/test_graph_grad.py already checks every rule against an independent
finite difference, which is enough to catch a rule that is finite, the right
shape, and numerically wrong -- but a finite difference is still just a
different numerical method applied to *the same formula* the rule itself
computes, so a bug shared between the rule and the reader's mental model of
it (a wrong factor both "look right") is not something it can catch on its
own. ``torch.autograd`` is a second, genuinely independent autodiff
implementation -- not a numerical approximation of the same formula, a
different system differentiating a different but equivalent graph -- which
is exactly what caught this repo's own ``BatchNormalization`` dvar-derivation
bug (see ``graph_grad.py``'s "Templated rules" section). This file applies
that same check to the rest of the builtin rules, one op family at a time,
rather than only to the op whose bug motivated it.

Reuses tests/test_graph_grad.py's ``_model``/``_static_shapes``/
``_backward_model``/``_feeds``/``_positive``/``_away_from_zero`` rather than
duplicating them -- the same cross-test-module import
tests/test_graph_grad_registry.py already establishes for this repo.

Not attempted here: ``Add`` and ``BatchNormalization`` (already covered by
tests/test_graph_grad_templates.py), and a handful of ops
(``Identity``/``Reshape``/``Transpose``-adjacent plumbing already implied by
every other case below reaching them through a chain) where a finite
difference already gives full confidence and there is no independent-formula
detail left for ``torch.autograd`` to disagree about.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn.functional as F
from test_graph_grad import (
    _away_from_zero,
    _backward_model,
    _feeds,
    _model,
    _positive,
    _static_shapes,
)

ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip(
    "torch", reason="this whole module compares against torch.autograd"
)


def _check_against_torch(
    body, torch_fn, targets=None, overrides=None, seed=0, rtol=2e-3, atol=2e-4
):
    """The whole experiment, torch flavored: emit ``body``'s backward with
    :func:`onnxsim.graph_grad.build_backward` (via
    ``test_graph_grad._backward_model``, which also re-checks the operator
    allowlist), run it, and compare against ``torch.autograd.grad`` of
    ``torch_fn`` -- a plain-torch-ops reimplementation of the same forward --
    seeded with the same random inputs and the same upstream gradient."""
    model = _model(body)
    targets = targets or [value.name for value in model.graph.input]
    rng = np.random.default_rng(seed)
    feeds = _feeds(model, rng, overrides)

    backward = _backward_model(model, targets)
    output_shape = _static_shapes(model)[model.graph.output[0].name]
    grad_seed = rng.standard_normal(output_shape).astype(np.float32)
    session = ort.InferenceSession(
        backward.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    analytic = session.run([f"grad_{t}" for t in targets], dict(feeds, dY=grad_seed))

    tensors = {
        name: torch.tensor(value, requires_grad=(name in targets))
        for name, value in feeds.items()
    }
    y = torch_fn(tensors)
    loss = (y * torch.tensor(grad_seed)).sum()
    torch_grads = torch.autograd.grad(loss, [tensors[t] for t in targets])
    for target, got, expected in zip(targets, analytic, torch_grads):
        got = np.asarray(got)
        expected = expected.detach().numpy()
        assert got.shape == expected.shape, f"{target}: {got.shape} != {expected.shape}"
        np.testing.assert_allclose(
            got, expected, rtol=rtol, atol=atol, err_msg=f"{target}: vs torch.autograd"
        )


# --- Elementwise and structural ops -------------------------------------
#
# One small graph each, the same shapes tests/test_graph_grad.py's own
# _CASES uses, paired with a plain-torch-ops forward that computes the same
# thing an independent way.

_SIMPLE_CASES = {
    "relu": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Relu(A)
        }
        """,
        None,
        lambda t: torch.relu(t["A"]),
    ),
    "sigmoid": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Sigmoid(A)
        }
        """,
        None,
        lambda t: torch.sigmoid(t["A"]),
    ),
    "tanh": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Tanh(A)
        }
        """,
        None,
        lambda t: torch.tanh(t["A"]),
    ),
    "exp": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Exp(A)
        }
        """,
        None,
        lambda t: torch.exp(t["A"]),
    ),
    "log": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Log(A)
        }
        """,
        {"A": _positive},
        lambda t: torch.log(t["A"]),
    ),
    "sqrt": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Sqrt(A)
        }
        """,
        {"A": _positive},
        lambda t: torch.sqrt(t["A"]),
    ),
    "erf": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Erf(A)
        }
        """,
        None,
        lambda t: torch.erf(t["A"]),
    ),
    "neg": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Neg(A)
        }
        """,
        None,
        lambda t: -t["A"],
    ),
    "sub": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Sub(A, B)
        }
        """,
        None,
        lambda t: t["A"] - t["B"],
    ),
    "mul": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Mul(A, B)
        }
        """,
        None,
        lambda t: t["A"] * t["B"],
    ),
    "div": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Div(A, B)
        }
        """,
        {"B": _away_from_zero},
        lambda t: t["A"] / t["B"],
    ),
    "reshape": (
        """
        g (float[2,3,4] A) => (float[6,4] Y)
        <int64[2] target = {6, 4}>
        {
          Y = Reshape(A, target)
        }
        """,
        None,
        lambda t: t["A"].reshape(6, 4),
    ),
    "transpose_default": (
        """
        g (float[2,3,4] A) => (float[4,3,2] Y) {
          Y = Transpose(A)
        }
        """,
        None,
        lambda t: t["A"].permute(2, 1, 0),
    ),
    "transpose_perm": (
        """
        g (float[2,3,4] A) => (float[3,2,4] Y) {
          Y = Transpose <perm = [1, 0, 2]> (A)
        }
        """,
        None,
        lambda t: t["A"].permute(1, 0, 2),
    ),
    "reducesum_keepdims": (
        """
        g (float[2,3,4] A) => (float[2,1,4] Y)
        <int64[1] axes = {1}>
        {
          Y = ReduceSum <keepdims = 1> (A, axes)
        }
        """,
        None,
        lambda t: t["A"].sum(dim=1, keepdim=True),
    ),
    "reducesum_dropdims": (
        """
        g (float[2,3,4] A) => (float[2,4] Y)
        <int64[1] axes = {1}>
        {
          Y = ReduceSum <keepdims = 0> (A, axes)
        }
        """,
        None,
        lambda t: t["A"].sum(dim=1, keepdim=False),
    ),
    "reducemean_axes_attribute": (
        """
        g (float[2,3,4] A) => (float[2,3,1] Y) {
          Y = ReduceMean <axes = [2], keepdims = 1> (A)
        }
        """,
        None,
        lambda t: t["A"].mean(dim=2, keepdim=True),
    ),
    "reducemean_dropdims": (
        """
        g (float[2,3,4] A) => (float[2,4] Y) {
          Y = ReduceMean <axes = [1], keepdims = 0> (A)
        }
        """,
        None,
        lambda t: t["A"].mean(dim=1, keepdim=False),
    ),
    "softmax_last_axis": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Softmax (A)
        }
        """,
        None,
        lambda t: torch.softmax(t["A"], dim=-1),
    ),
    "softmax_first_axis": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Softmax <axis = 0> (A)
        }
        """,
        None,
        lambda t: torch.softmax(t["A"], dim=0),
    ),
    # Continuous random data is never exactly on a clip bound, so the one
    # place onnxsim's mask (strict inequality -- see _grad_clip) and torch's
    # clamp gradient (inclusive of the bound) actually disagree is a
    # zero-probability event here, not something this comparison risks.
    "clip": (
        """
        g (float[3,4] A) => (float[3,4] Y)
        <float lo = {-0.5}, float hi = {0.5}>
        {
          Y = Clip(A, lo, hi)
        }
        """,
        None,
        lambda t: torch.clamp(t["A"], min=-0.5, max=0.5),
    ),
    "clip_lower_only": (
        """
        g (float[3,4] A) => (float[3,4] Y)
        <float lo = {-0.25}>
        {
          Y = Clip(A, lo)
        }
        """,
        None,
        lambda t: torch.clamp(t["A"], min=-0.25),
    ),
}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", sorted(_SIMPLE_CASES), ids=sorted(_SIMPLE_CASES))
def test_simple_rule_matches_torch_autograd(case, seed):
    body, overrides, torch_fn = _SIMPLE_CASES[case]
    _check_against_torch(body, torch_fn, overrides=overrides, seed=seed)


# --- MatMul --------------------------------------------------------------

_MATMUL_CASES = {
    "matmul": """
        g (float[3,4] A, float[4,5] B) => (float[3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
    "matmul_batched": """
        g (float[2,3,4] A, float[2,4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
    # Torch's own matmul broadcasts leading batch dims exactly the way
    # ONNX's MatMul spec (and _grad_matmul's own np.broadcast_shapes call)
    # does, so the same lambda covers the broadcasting and rank-mixed cases.
    "matmul_broadcast_batch": """
        g (float[1,3,4] A, float[2,4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
    "matmul_rank_mix": """
        g (float[2,3,4] A, float[4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
}


@pytest.mark.parametrize("case", sorted(_MATMUL_CASES), ids=sorted(_MATMUL_CASES))
def test_matmul_matches_torch_autograd(case):
    _check_against_torch(_MATMUL_CASES[case], lambda t: t["A"] @ t["B"])


# --- Gemm ------------------------------------------------------------------


@pytest.mark.parametrize(
    "attributes",
    [
        "",
        "<alpha = 0.75>",
        "<beta = 0.5>",
        "<alpha = 1.5, beta = -0.25>",
        "<transB = 1>",
        "<transA = 1>",
        "<transA = 1, transB = 1, alpha = 0.5, beta = 2.0>",
    ],
    ids=["plain", "alpha", "beta", "alpha_beta", "transB", "transA", "everything"],
)
def test_gemm_matches_torch_autograd(attributes):
    trans_a = "transA = 1" in attributes
    trans_b = "transB = 1" in attributes
    alpha = (
        0.75
        if "alpha = 0.75" in attributes
        else (
            1.5
            if "alpha = 1.5" in attributes
            else (0.5 if "alpha = 0.5" in attributes else 1.0)
        )
    )
    beta = (
        0.5
        if "beta = 0.5" in attributes
        else (
            -0.25
            if "beta = -0.25" in attributes
            else (2.0 if "beta = 2.0" in attributes else 1.0)
        )
    )
    a_shape = "4,3" if trans_a else "3,4"
    b_shape = "5,4" if trans_b else "4,5"

    def torch_fn(t):
        a = t["A"].t() if trans_a else t["A"]
        b = t["B"].t() if trans_b else t["B"]
        return alpha * (a @ b) + beta * t["C"]

    _check_against_torch(
        f"""
        g (float[{a_shape}] A, float[{b_shape}] B, float[5] C)
            => (float[3,5] Y) {{
          Y = Gemm {attributes} (A, B, C)
        }}
        """,
        torch_fn,
    )


# --- Conv --------------------------------------------------------------
#
# ONNX Conv and torch's conv{1,2,3}d share both the weight-tensor layout
# ([out_channels, in_channels/groups, *kernel]) and the padding/stride/
# dilation/groups semantics, so this is a genuinely different implementation
# of the *same* op (torch's own native conv + its autodiff, not im2col
# reconstructed by hand the way _grad_conv is) rather than a reformulation --
# a strong, cheap cross-check for a rule whose whole design (see _grad_conv's
# own docstring) is "no convolution op in the backward graph at all". Every
# case below keeps padding symmetric so a plain ``padding=`` tuple expresses
# it exactly; the asymmetric SAME_UPPER/SAME_LOWER cases stay covered by
# tests/test_graph_grad.py's finite differences only.
_CONV_FN = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}

_CONV_CASES = {
    "conv": dict(
        body="""
        g (float[1,2,4,4] A, float[3,2,3,3] B) => (float[1,3,2,2] Y) {
          Y = Conv(A, B)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        padding=(0, 0),
        has_bias=False,
    ),
    "conv_bias": dict(
        body="""
        g (float[1,1,4,4] A, float[2,1,3,3] B, float[2] C)
            => (float[1,2,2,2] Y) {
          Y = Conv(A, B, C)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        padding=(0, 0),
        has_bias=True,
    ),
    "conv_pointwise": dict(
        body="""
        g (float[1,2,3,3] A, float[3,2,1,1] B) => (float[1,3,3,3] Y) {
          Y = Conv(A, B)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        padding=(0, 0),
        has_bias=False,
    ),
    "conv_strided_padded": dict(
        body="""
        g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
          Y = Conv <strides = [2, 2], pads = [1, 1, 1, 1]> (A, B)
        }
        """,
        spatial=2,
        stride=(2, 2),
        dilation=(1, 1),
        groups=1,
        padding=(1, 1),
        has_bias=False,
    ),
    "conv_dilated": dict(
        body="""
        g (float[1,1,5,5] A, float[1,1,2,2] B) => (float[1,1,3,3] Y) {
          Y = Conv <dilations = [2, 2]> (A, B)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(2, 2),
        groups=1,
        padding=(0, 0),
        has_bias=False,
    ),
    "conv_grouped": dict(
        body="""
        g (float[1,4,3,3] A, float[4,2,2,2] B) => (float[1,4,2,2] Y) {
          Y = Conv <group = 2> (A, B)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(1, 1),
        groups=2,
        padding=(0, 0),
        has_bias=False,
    ),
    "conv_depthwise": dict(
        body="""
        g (float[1,3,3,3] A, float[3,1,2,2] B, float[3] C)
            => (float[1,3,2,2] Y) {
          Y = Conv <group = 3> (A, B, C)
        }
        """,
        spatial=2,
        stride=(1, 1),
        dilation=(1, 1),
        groups=3,
        padding=(0, 0),
        has_bias=True,
    ),
    "conv_1d": dict(
        body="""
        g (float[1,2,7] A, float[2,2,3] B) => (float[1,2,3] Y) {
          Y = Conv <strides = [2]> (A, B)
        }
        """,
        spatial=1,
        stride=(2,),
        dilation=(1,),
        groups=1,
        padding=(0,),
        has_bias=False,
    ),
    "conv_3d": dict(
        body="""
        g (float[1,1,3,3,3] A, float[2,1,2,2,2] B) => (float[1,2,2,2,2] Y) {
          Y = Conv(A, B)
        }
        """,
        spatial=3,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=1,
        padding=(0, 0, 0),
        has_bias=False,
    ),
}


@pytest.mark.parametrize("case", sorted(_CONV_CASES), ids=sorted(_CONV_CASES))
def test_conv_matches_torch_autograd(case):
    spec = _CONV_CASES[case]
    conv = _CONV_FN[spec["spatial"]]

    def torch_fn(t, spec=spec, conv=conv):
        bias = t["C"] if spec["has_bias"] else None
        return conv(
            t["A"],
            t["B"],
            bias=bias,
            stride=spec["stride"],
            padding=spec["padding"],
            dilation=spec["dilation"],
            groups=spec["groups"],
        )

    _check_against_torch(spec["body"], torch_fn)


# --- MaxPool / AveragePool -----------------------------------------------
#
# Same rationale as Conv: torch's own pooling ops share ONNX's geometry
# attributes directly, so this is an independent implementation of the same
# op, not a hand-transcription of _grad_maxpool/_grad_averagepool's own
# col2im construction.
_MAXPOOL_FN = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}

_MAXPOOL_CASES = {
    "maxpool": dict(
        body="""
        g (float[1,2,4,4] A) => (float[1,2,2,2] Y) {
          Y = MaxPool <kernel_shape = [2, 2], strides = [2, 2]> (A)
        }
        """,
        spatial=2,
        kernel=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
    ),
    "maxpool_strided_padded": dict(
        body="""
        g (float[1,2,5,5] A) => (float[1,2,3,3] Y) {
          Y = MaxPool <kernel_shape = [3, 3], strides = [2, 2],
                       pads = [1, 1, 1, 1]> (A)
        }
        """,
        spatial=2,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
    ),
    "maxpool_dilated": dict(
        body="""
        g (float[1,1,5,5] A) => (float[1,1,3,3] Y) {
          Y = MaxPool <kernel_shape = [2, 2], dilations = [2, 2]> (A)
        }
        """,
        spatial=2,
        kernel=(2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(2, 2),
    ),
    "maxpool_1d": dict(
        body="""
        g (float[1,2,7] A) => (float[1,2,3] Y) {
          Y = MaxPool <kernel_shape = [3], strides = [2]> (A)
        }
        """,
        spatial=1,
        kernel=(3,),
        stride=(2,),
        padding=(0,),
        dilation=(1,),
    ),
    "maxpool_3d": dict(
        body="""
        g (float[1,1,4,4,4] A) => (float[1,1,2,2,2] Y) {
          Y = MaxPool <kernel_shape = [2, 2, 2], strides = [2, 2, 2]> (A)
        }
        """,
        spatial=3,
        kernel=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
    ),
}


@pytest.mark.parametrize("case", sorted(_MAXPOOL_CASES), ids=sorted(_MAXPOOL_CASES))
def test_maxpool_matches_torch_autograd(case):
    spec = _MAXPOOL_CASES[case]
    pool = _MAXPOOL_FN[spec["spatial"]]

    def torch_fn(t, spec=spec, pool=pool):
        return pool(
            t["A"],
            kernel_size=spec["kernel"],
            stride=spec["stride"],
            padding=spec["padding"],
            dilation=spec["dilation"],
        )

    _check_against_torch(spec["body"], torch_fn)


_AVGPOOL_FN = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}

_AVGPOOL_CASES = {
    "averagepool": dict(
        body="""
        g (float[1,2,4,4] A) => (float[1,2,2,2] Y) {
          Y = AveragePool <kernel_shape = [2, 2], strides = [2, 2]> (A)
        }
        """,
        spatial=2,
        kernel=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        count_include_pad=False,
    ),
    "averagepool_count_exclude_pad": dict(
        body="""
        g (float[1,2,5,5] A) => (float[1,2,3,3] Y) {
          Y = AveragePool <kernel_shape = [3, 3], strides = [2, 2],
                           pads = [1, 1, 1, 1]> (A)
        }
        """,
        spatial=2,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        count_include_pad=False,
    ),
    "averagepool_count_include_pad": dict(
        body="""
        g (float[1,2,5,5] A) => (float[1,2,3,3] Y) {
          Y = AveragePool <kernel_shape = [3, 3], strides = [2, 2],
                           pads = [1, 1, 1, 1], count_include_pad = 1> (A)
        }
        """,
        spatial=2,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        count_include_pad=True,
    ),
    "averagepool_1d": dict(
        body="""
        g (float[1,2,7] A) => (float[1,2,3] Y) {
          Y = AveragePool <kernel_shape = [3], strides = [2]> (A)
        }
        """,
        spatial=1,
        kernel=(3,),
        stride=(2,),
        padding=(0,),
        count_include_pad=False,
    ),
    "averagepool_3d": dict(
        body="""
        g (float[1,1,4,4,4] A) => (float[1,1,2,2,2] Y) {
          Y = AveragePool <kernel_shape = [2, 2, 2], strides = [2, 2, 2]> (A)
        }
        """,
        spatial=3,
        kernel=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        count_include_pad=False,
    ),
}


@pytest.mark.parametrize("case", sorted(_AVGPOOL_CASES), ids=sorted(_AVGPOOL_CASES))
def test_averagepool_matches_torch_autograd(case):
    spec = _AVGPOOL_CASES[case]
    pool = _AVGPOOL_FN[spec["spatial"]]

    def torch_fn(t, spec=spec, pool=pool):
        return pool(
            t["A"],
            kernel_size=spec["kernel"],
            stride=spec["stride"],
            padding=spec["padding"],
            count_include_pad=spec["count_include_pad"],
        )

    _check_against_torch(spec["body"], torch_fn)


# --- LayerNormalization / InstanceNormalization --------------------------
#
# Both share BatchNormalization's own risk profile -- dx depends on every
# element in its normalization group through both a mean and a variance term
# computed from x itself -- so both get the same torch.autograd cross-check
# tests/test_graph_grad_templates.py gave BatchNormalization, written out in
# plain torch ops (mean/var/rsqrt) rather than a fused
# torch.nn.functional.layer_norm/instance_norm, for the same reason that
# file's own comment gives: independence from whatever a fused kernel does
# or does not support differentiating.


def _layer_norm_torch_fn(axis, eps, has_bias):
    def fn(t):
        x = t["A"]
        dims = tuple(range(axis, x.dim()))
        xc = x - x.mean(dim=dims, keepdim=True)
        var = (xc * xc).mean(dim=dims, keepdim=True)
        xhat = xc / torch.sqrt(var + eps)
        y = xhat * t["S"]
        if has_bias:
            y = y + t["B"]
        return y

    return fn


_LAYER_NORM_CASES = {
    "layer_norm": dict(
        body="""
        g (float[2,3,4] A, float[4] S, float[4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization (A, S, B)
        }
        """,
        axis=2,
        eps=1e-5,
        has_bias=True,
    ),
    "layer_norm_no_bias": dict(
        body="""
        g (float[2,3,4] A, float[4] S) => (float[2,3,4] Y) {
          Y = LayerNormalization (A, S)
        }
        """,
        axis=2,
        eps=1e-5,
        has_bias=False,
    ),
    "layer_norm_axis_1": dict(
        body="""
        g (float[2,3,4] A, float[3,4] S, float[3,4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization <axis = 1> (A, S, B)
        }
        """,
        axis=1,
        eps=1e-5,
        has_bias=True,
    ),
    "layer_norm_epsilon": dict(
        body="""
        g (float[2,3,4] A, float[4] S, float[4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization <epsilon = 0.001> (A, S, B)
        }
        """,
        axis=2,
        eps=0.001,
        has_bias=True,
    ),
}


@pytest.mark.parametrize(
    "case", sorted(_LAYER_NORM_CASES), ids=sorted(_LAYER_NORM_CASES)
)
def test_layer_normalization_matches_torch_autograd(case):
    spec = _LAYER_NORM_CASES[case]
    _check_against_torch(
        spec["body"],
        _layer_norm_torch_fn(spec["axis"], spec["eps"], spec["has_bias"]),
    )


def _instance_norm_torch_fn(eps):
    def fn(t):
        x = t["X"]
        spatial = tuple(range(2, x.dim()))
        xc = x - x.mean(dim=spatial, keepdim=True)
        var = (xc * xc).mean(dim=spatial, keepdim=True)
        xhat = xc / torch.sqrt(var + eps)
        bshape = [1, -1] + [1] * (x.dim() - 2)
        return xhat * t["S"].reshape(bshape) + t["Bs"].reshape(bshape)

    return fn


_INSTANCE_NORM_CASES = {
    "instance_norm": dict(
        body="""
        g (float[2,3,4,4] X, float[3] S, float[3] Bs) => (float[2,3,4,4] Y) {
          Y = InstanceNormalization(X, S, Bs)
        }
        """,
        eps=1e-5,
    ),
    "instance_norm_channel_count": dict(
        body="""
        g (float[2,5,3,3] X, float[5] S, float[5] Bs) => (float[2,5,3,3] Y) {
          Y = InstanceNormalization(X, S, Bs)
        }
        """,
        eps=1e-5,
    ),
    "instance_norm_1d_spatial": dict(
        body="""
        g (float[2,3,5] X, float[3] S, float[3] Bs) => (float[2,3,5] Y) {
          Y = InstanceNormalization(X, S, Bs)
        }
        """,
        eps=1e-5,
    ),
    "instance_norm_epsilon": dict(
        body="""
        g (float[2,3,4,4] X, float[3] S, float[3] Bs) => (float[2,3,4,4] Y) {
          Y = InstanceNormalization <epsilon = 0.001> (X, S, Bs)
        }
        """,
        eps=0.001,
    ),
}


@pytest.mark.parametrize(
    "case", sorted(_INSTANCE_NORM_CASES), ids=sorted(_INSTANCE_NORM_CASES)
)
def test_instance_normalization_matches_torch_autograd(case):
    spec = _INSTANCE_NORM_CASES[case]
    _check_against_torch(spec["body"], _instance_norm_torch_fn(spec["eps"]))


# --- Gather ---------------------------------------------------------------
#
# torch.index_select is the same embedding-lookup semantics _grad_gather
# documents itself as implementing (a scatter-add gradient for repeated
# reads), built by torch's own indexing rather than the one-hot MatMul
# _grad_gather uses to stay inside BACKWARD_OPS -- an independent
# implementation of the same rule, not the same construction restated.
# ``idx``/``axis`` are the model's own initializer/attribute, not a graph
# input _feeds could randomize, so they are given directly to torch here the
# same fixed way the ONNX text spells them.

_GATHER_CASES = {
    "gather_embedding": dict(
        body="""
        g (float[5,3] data) => (float[3,3] Y)
        <int64[3] idx = {0, 2, 4}>
        {
          Y = Gather(data, idx)
        }
        """,
        axis=0,
        index=[0, 2, 4],
    ),
    "gather_repeated_index": dict(
        body="""
        g (float[4,2] data) => (float[3,2] Y)
        <int64[3] idx = {1, 1, 3}>
        {
          Y = Gather(data, idx)
        }
        """,
        axis=0,
        index=[1, 1, 3],
    ),
    # axis=-2 on a rank-3 input resolves to axis 1 (size 5); idx=-1 resolves
    # to 4 -- both resolutions done here, by hand, so torch is given the
    # same lookup in its own positive-axis/positive-index spelling.
    "gather_negative_axis_and_index": dict(
        body="""
        g (float[2,5,3] data) => (float[2,2,3] Y)
        <int64[2] idx = {-1, 0}>
        {
          Y = Gather <axis = -2> (data, idx)
        }
        """,
        axis=1,
        index=[4, 0],
    ),
}


@pytest.mark.parametrize("case", sorted(_GATHER_CASES), ids=sorted(_GATHER_CASES))
def test_gather_matches_torch_autograd(case):
    spec = _GATHER_CASES[case]
    idx = torch.tensor(spec["index"], dtype=torch.long)

    def torch_fn(t, axis=spec["axis"], idx=idx):
        return t["data"].index_select(axis, idx)

    _check_against_torch(spec["body"], torch_fn)
