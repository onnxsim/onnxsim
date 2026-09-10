#!/usr/bin/env python3
"""Print, side by side, the backward graph :mod:`onnxsim.graph_grad` emits
for a handful of ops next to the backward graph torch's own AOTAutograd
produces for the identical math.

This is a reading aid, not a test. tests/test_graph_grad_torch_autograd.py
already checks the *numbers* onnxsim's rules produce against
``torch.autograd.grad`` -- and since AOTAutograd's backward is still
PyTorch's own autograd underneath (just traced through functionalization
and partitioned into forward/backward halves), tracing it again here would
not add a second independent numeric oracle. What AOTAutograd's trace does
give, that plain ``torch.autograd.grad`` does not, is *visibility*: with
``torch._decomp.core_aten_decompositions()`` applied, an op whose eager
backward is a single opaque fused kernel call gets expanded into the
primitive aten ops that kernel is defined in terms of (see the Relu case
below: its backward decomposes into an explicit ``le``/``where`` mask, the
same shape of thing :func:`onnxsim.graph_grad._grad_relu` builds by hand).
Reading the two lists next to each other is a quick sanity check when
touching a rule -- structurally implausible output is usually visible at a
glance -- but it is not asserted anywhere, on purpose:

* ``torch._functorch.aot_autograd.aot_export_module``,
  ``torch._functorch.partitioners.default_partition`` and
  ``torch._decomp.core_aten_decompositions`` are all private torch APIs with
  no stability guarantee across releases (their exact call signature
  already changed once during this file's own development). A test pinned
  to their current shape would break on an unrelated torch bump, not on an
  onnxsim regression.
* AOTAutograd's decomposition is not the same VJP construction onnxsim's
  own rules use even where both land on plain arithmetic (different
  op vocabulary, different intermediate naming, sometimes a different but
  equivalent factoring), and for Conv/MaxPool/AveragePool it does not
  decompose at all -- ``aten.convolution_backward``/
  ``aten.max_pool2d_with_indices_backward`` are themselves part of core
  aten, so those cases print one opaque call each, in deliberate contrast
  to onnxsim's own explicit im2col-based graph for the same op (see
  ``_grad_conv``'s docstring for why onnxsim decomposes there instead of
  reusing a fused kernel: no such fused kernel exists in
  :data:`onnxsim.graph_grad.BACKWARD_OPS`). There is no useful op-for-op
  equality to assert in either direction, so this script only prints.

Every AOTAutograd call is wrapped so a version mismatch prints a "skipped"
line for that case and moves on rather than aborting the whole run --
consistent with the above, this stays a best-effort look, never a gate.

Usage: ``python scripts/dump_aot_autograd_grad_graphs.py`` (needs the built
``onnxsim_cpp2py_export`` extension importable, same as running the tests,
and ``torch`` installed for the AOTAutograd half -- without it, this still
prints the onnxsim side of each case).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnx  # noqa: E402
import onnx.parser  # noqa: E402
import onnx.shape_inference  # noqa: E402

from onnxsim import graph_grad, qat_graph  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

_HEADER = '<ir_version: 8, opset_import: ["": 17]>'


def _model(body: str) -> onnx.ModelProto:
    return onnx.parser.parse_model(f"{_HEADER}\n{body}")


def _static_shapes(model: onnx.ModelProto) -> dict:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _onnxsim_backward_ops(body: str, targets: list) -> list:
    """The op_type of every node :func:`onnxsim.graph_grad.build_backward`
    emits for ``body``, in emission order."""
    model = _model(body)
    shapes = _static_shapes(model)
    output = model.graph.output[0].name
    b = qat_graph.GraphBuilder("bw_")
    graph_grad.build_backward(
        b, list(model.graph.node), shapes, {output: "dY"}, targets
    )
    return [n.op_type for n in b.nodes]


def _aot_backward_ops(fn, n_inputs: int, shapes: list):
    """AOTAutograd's own decomposed backward graph for ``sum(fn(*xs) *
    seed)``, seeded the same way ``_onnxsim_backward_ops`` is -- a single
    upstream gradient over the whole output, not a scalar loss with an
    implicit gradient of 1. ``shapes`` is ``[*input shapes, seed/output
    shape]``. Returns ``None`` (after printing why) instead of raising if
    torch, or the private APIs this leans on, are not available in the
    shape this script expects -- see the module docstring.
    """
    try:
        import torch.nn as nn
        from torch._decomp import core_aten_decompositions
        from torch._functorch.aot_autograd import (
            aot_export_module,
            default_partition,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped -- torch/aot_autograd unavailable: {e})")
        return None

    class _Loss(nn.Module):
        def forward(self, *args):
            xs, seed = args[:n_inputs], args[n_inputs]
            y = fn(*xs)
            return ((y * seed).sum(),)

    try:
        torch.manual_seed(0)
        args = tuple(
            torch.randn(*shape, requires_grad=True) for shape in shapes[:n_inputs]
        )
        seed = torch.randn(*shapes[n_inputs])
        gm, _sig = aot_export_module(
            _Loss(),
            (*args, seed),
            trace_joint=True,
            output_loss_index=0,
            decompositions=core_aten_decompositions(),
        )
        _fw, bw = default_partition(gm, (*args, seed), num_fwd_outputs=1)
        return [
            str(node.target) for node in bw.graph.nodes if node.op == "call_function"
        ]
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped -- aot_autograd internals raised: {e})")
        return None


def _layer_norm_fn(axis, eps):
    def fn(x, scale, bias):
        dims = tuple(range(axis, x.dim()))
        xc = x - x.mean(dim=dims, keepdim=True)
        var = (xc * xc).mean(dim=dims, keepdim=True)
        return (xc / torch.sqrt(var + eps)) * scale + bias

    return fn


def _instance_norm_fn(eps):
    def fn(x, scale, bias):
        spatial = tuple(range(2, x.dim()))
        xc = x - x.mean(dim=spatial, keepdim=True)
        var = (xc * xc).mean(dim=spatial, keepdim=True)
        xhat = xc / torch.sqrt(var + eps)
        bshape = [1, -1] + [1] * (x.dim() - 2)
        return xhat * scale.reshape(bshape) + bias.reshape(bshape)

    return fn


def _batch_norm_fn(eps):
    # mean/var are the node's own *inputs* here, not reductions of x -- the
    # same distinction _grad_batch_normalization's own docstring draws
    # against _grad_layer_normalization.
    def fn(x, scale, bias, mean, var):
        bshape = [1, -1] + [1] * (x.dim() - 2)
        xhat = (x - mean.reshape(bshape)) / torch.sqrt(var.reshape(bshape) + eps)
        return xhat * scale.reshape(bshape) + bias.reshape(bshape)

    return fn


# Each case: an onnxsim ONNX-text forward + the targets to differentiate,
# and a positional-argument torch function computing the identical forward
# (last shape in "shapes" is always the seed/output shape). A curated
# handful -- the ops with the most structurally interesting backward
# (Norm family's mean/var coupling, Conv/Pool's fused-vs-decomposed
# contrast) plus a couple of plain ones (Relu, MatMul) for a baseline.
CASES = [
    dict(
        name="relu",
        onnx_body="""
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Relu(A)
        }
        """,
        onnx_targets=["A"],
        n_inputs=1,
        torch_fn=lambda x: torch.relu(x),
        shapes=[(3, 4), (3, 4)],
    ),
    dict(
        name="softmax",
        onnx_body="""
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Softmax (A)
        }
        """,
        onnx_targets=["A"],
        n_inputs=1,
        torch_fn=lambda x: torch.softmax(x, dim=-1),
        shapes=[(3, 4), (3, 4)],
    ),
    dict(
        name="matmul",
        onnx_body="""
        g (float[3,4] A, float[4,5] B) => (float[3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
        onnx_targets=["A", "B"],
        n_inputs=2,
        torch_fn=lambda a, b: a @ b,
        shapes=[(3, 4), (4, 5), (3, 5)],
    ),
    dict(
        name="gemm",
        onnx_body="""
        g (float[3,4] A, float[4,5] B, float[5] C) => (float[3,5] Y) {
          Y = Gemm(A, B, C)
        }
        """,
        onnx_targets=["A", "B", "C"],
        n_inputs=3,
        torch_fn=lambda a, b, c: a @ b + c,
        shapes=[(3, 4), (4, 5), (5,), (3, 5)],
    ),
    dict(
        name="layer_norm",
        onnx_body="""
        g (float[2,3,4] A, float[4] S, float[4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization (A, S, B)
        }
        """,
        onnx_targets=["A", "S", "B"],
        n_inputs=3,
        torch_fn=_layer_norm_fn(axis=2, eps=1e-5),
        shapes=[(2, 3, 4), (4,), (4,), (2, 3, 4)],
    ),
    dict(
        name="instance_norm",
        onnx_body="""
        g (float[2,3,4,4] X, float[3] S, float[3] Bs) => (float[2,3,4,4] Y) {
          Y = InstanceNormalization(X, S, Bs)
        }
        """,
        onnx_targets=["X", "S", "Bs"],
        n_inputs=3,
        torch_fn=_instance_norm_fn(eps=1e-5),
        shapes=[(2, 3, 4, 4), (3,), (3,), (2, 3, 4, 4)],
    ),
    dict(
        name="batch_norm",
        onnx_body="""
        g (float[2,3,4,4] X, float[3] S, float[3] Bs, float[3] M, float[3] V)
            => (float[2,3,4,4] Y) {
          Y = BatchNormalization(X, S, Bs, M, V)
        }
        """,
        onnx_targets=["X", "S", "Bs", "M", "V"],
        n_inputs=5,
        torch_fn=_batch_norm_fn(eps=1e-5),
        shapes=[(2, 3, 4, 4), (3,), (3,), (3,), (3,), (2, 3, 4, 4)],
    ),
    dict(
        name="conv",
        onnx_body="""
        g (float[1,2,4,4] A, float[3,2,3,3] B) => (float[1,3,2,2] Y) {
          Y = Conv(A, B)
        }
        """,
        onnx_targets=["A", "B"],
        n_inputs=2,
        torch_fn=lambda x, w: F.conv2d(x, w),
        shapes=[(1, 2, 4, 4), (3, 2, 3, 3), (1, 3, 2, 2)],
    ),
    dict(
        name="maxpool",
        onnx_body="""
        g (float[1,2,4,4] A) => (float[1,2,2,2] Y) {
          Y = MaxPool <kernel_shape = [2, 2], strides = [2, 2]> (A)
        }
        """,
        onnx_targets=["A"],
        n_inputs=1,
        torch_fn=lambda x: F.max_pool2d(x, kernel_size=2, stride=2),
        shapes=[(1, 2, 4, 4), (1, 2, 2, 2)],
    ),
    dict(
        name="averagepool",
        onnx_body="""
        g (float[1,2,4,4] A) => (float[1,2,2,2] Y) {
          Y = AveragePool <kernel_shape = [2, 2], strides = [2, 2]> (A)
        }
        """,
        onnx_targets=["A"],
        n_inputs=1,
        torch_fn=lambda x: F.avg_pool2d(x, kernel_size=2, stride=2),
        shapes=[(1, 2, 4, 4), (1, 2, 2, 2)],
    ),
]


def main() -> None:
    for case in CASES:
        print(f"=== {case['name']} ===")
        onnx_ops = _onnxsim_backward_ops(case["onnx_body"], case["onnx_targets"])
        print(f"onnxsim      ({len(onnx_ops)} nodes): {onnx_ops}")
        aot_ops = _aot_backward_ops(case["torch_fn"], case["n_inputs"], case["shapes"])
        if aot_ops is not None:
            print(f"aot_autograd ({len(aot_ops)} nodes): {aot_ops}")
        print()


if __name__ == "__main__":
    main()
