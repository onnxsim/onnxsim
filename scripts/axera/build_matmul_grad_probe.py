#!/usr/bin/env python3
"""Minimal MatMul-gradient probe, isolating the exact node the training-step
gradient-underflow ceiling lives at (`docs/axera-on-device-training-handoff.md`'s
"The ceiling: the gradient dies" section) from every other legalization/speed
change elsewhere in that pipeline.

Graph: `y = MatMul(x, w)`; `loss = ReduceSum(y*y)`; `dW = grad(loss, w)` via
`graph_grad.build_backward`, with the loss's own incoming gradient seeded by
a real runtime scalar `grad_seed` (a graph input, not baked in -- see the
`grad_seed` fix in this same doc's "FP32 gradient seed" section). Outputs:
`dW` only -- `loss` stays an internal tensor, not a graph output, because a
true-scalar (rank-0) *output* tensor triggers Pulsar2's own
"zero-dimensional tensor cannot be concatenated" calibration failure, the
same class of issue as the scalar-*input* fix `grad_seed`/`lr` already needed
(shape `[1]`, not `[]`), just on the output side.

Used to test whether `quant.layer_configs`' `output_data_type: "FP32"` field
(documented for `Conv`, not textually restricted to it) also applies when
targeting the gradient-producing `MatMul` -- see the handoff doc's "Two more
angles on controlling gradient quantization directly" section for the
result (both angles tried there closed negative, with real compiler-level
evidence). Kept as a standalone script rather than folded into
`build_resident_train_step.py`: this graph has nothing to do with a resident
training step (no state, no SGD update, no legalization) -- it exists purely
to put one `MatMul` and its own gradient in front of Pulsar2 with as little
else in the way as possible.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim  # noqa: E402

ensure_repo_onnxsim()

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from onnxsim import graph_grad, qat_graph  # noqa: E402

N, K, M = 8, 16, 8


def build() -> tuple[onnx.ModelProto, str, str]:
    b = qat_graph.GraphBuilder()
    x = "x"
    w = "w"
    y = b.matmul(x, w)
    sq = b.mul(y, y)
    loss = b.op("ReduceSum", [sq], keepdims=0)
    forward_nodes = list(b.nodes)

    grads = graph_grad.build_backward(
        b,
        nodes=forward_nodes,
        shapes={
            x: (N, K),
            w: (K, M),
            y: (N, M),
            sq: (N, M),
            loss: (),
        },
        grad_outputs={loss: "grad_seed"},
        targets=[w],
    )
    dw = grads[w]
    assert dw is not None

    inputs = [
        helper.make_tensor_value_info(x, TensorProto.FLOAT, [N, K]),
        helper.make_tensor_value_info(w, TensorProto.FLOAT, [K, M]),
        helper.make_tensor_value_info("grad_seed", TensorProto.FLOAT, [1]),
    ]
    outputs = [helper.make_tensor_value_info(dw, TensorProto.FLOAT, [K, M])]
    graph = helper.make_graph(
        b.nodes, "matmul_grad_probe", inputs, outputs, initializer=b.initializer
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model, loss, dw


if __name__ == "__main__":
    result_model, loss_name, dw_name = build()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "matmul_grad_probe.onnx"
    onnx.save(result_model, out_path)
    print(f"saved {out_path}  loss={loss_name!r}  dW={dw_name!r}")
