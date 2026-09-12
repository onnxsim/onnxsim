#!/usr/bin/env python3
"""Build the same small training-step graph as
`build_multiphase_calib_swap_probe.py`, but with an extra, graph-computed
diagnostic output per trainable weight: ``ReduceMax(Abs(grad))`` on the
*pre-quantization* gradient tensor, tapped before it is multiplied by ``lr``.

This is the fix the very first handoff doc named and nobody built: "fixed-
point clipping is silent and local... a graph that wants reliable back-off
needs to export ``ReduceMax(|t|)`` on those tensors as extra outputs." The
question this script exists to test is whether that signal is a *leading*
indicator of the gradient dying (it crosses some threshold before the
returned gradient's own zero-fraction proxy would fire) or just a redundant,
concurrent one -- answered in `docs/axera-on-device-training-handoff.md`'s
"Graph-computed calibration/saturation signals" section, not here; this
module only builds the instrumented graph.

`onnxsim.qat_graph.make_step_graph` has no generic "extra output" parameter
(only `state` and `loss`), so this reimplements `build_resident_step`'s body
directly rather than extending that shared function -- the same choice
`build_multiphase_calib_swap_probe.py` and `build_matmul_grad_probe.py` made
for their own one-off experiments.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, parser

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim, fresh  # noqa: E402

ensure_repo_onnxsim()

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules sharing one `sys.modules["legalize"]` entry
# -- a plain `import legalize` risks silently getting axelera's copy if
# something upstream already claimed that bare name. `fresh` reloads
# directly from this file's own directory regardless of what's cached.
legalize = fresh("legalize", HERE)

from onnxsim import graph_grad, qat_graph  # noqa: E402


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def forward_model() -> onnx.ModelProto:
    """Identical to `build_multiphase_calib_swap_probe.forward_model` --
    same seed, same shapes -- so results are directly comparable."""
    rng = np.random.default_rng(0)
    cw = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.3
    gw = rng.standard_normal((10, 32)).astype(np.float32) * 0.2
    gb = rng.standard_normal((10,)).astype(np.float32) * 0.1
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,1,4,4] x) => (float[1,10] logits)
        {
          h = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, cw)
          r = Relu(h)
          f = Flatten<axis=1>(r)
          logits = Gemm<transB=1>(f, gw, gb)
        }
        """
    )
    model.graph.initializer.extend([_f32(cw, "cw"), _f32(gw, "gw"), _f32(gb, "gb")])
    return model


def build_instrumented_step(
    forward_and_loss: onnx.ModelProto,
    params: Sequence[str],
    loss_output: str = "loss",
) -> Tuple[onnx.ModelProto, Dict[str, str], Dict[str, str]]:
    """Like `build_resident_train_step.build_resident_step`, plus a
    `ReduceMax(Abs(grad))` output per trainable weight.

    :returns: `(step_model, state, signals)` -- `signals` maps each
            trainable weight's name to its diagnostic output's name.
    """
    from build_resident_train_step import _POST_BACKWARD_RULES, _static_shapes_and_types

    model = onnx.ModelProto()
    model.CopyFrom(forward_and_loss)
    legalize.avgpool_ceil_to_floor(model)
    legalize.flatten_to_reshape(model)
    legalize.global_pool_to_reduce(model)
    onnx.checker.check_model(model)

    shapes, elem_types = _static_shapes_and_types(model)
    initializers = {t.name: t for t in model.graph.initializer}

    b = qat_graph.GraphBuilder(prefix="calib_signal__")
    b.nodes = list(model.graph.node)
    trained = set(params)
    b.initializer = [t for t in model.graph.initializer if t.name not in trained]

    seed = b.const(np.array(1.0, dtype=np.float32), "loss_seed")
    grads = graph_grad.build_backward(
        b,
        nodes=list(model.graph.node),
        shapes=shapes,
        grad_outputs={loss_output: seed},
        targets=list(params),
    )

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    signals: Dict[str, str] = {}
    for p in params:
        w_shape = tuple(int(d) for d in shapes[p])
        # Tap the gradient *before* it is scaled by lr or subtracted --
        # this is the tensor a compiled model quantizes as its own output,
        # the same one the ceiling investigation's "silent clipping" note
        # is about.
        abs_grad = b.op("Abs", [grads[p]], hint=f"{p}_absgrad")
        max_grad = b.op("ReduceMax", [abs_grad], hint=f"{p}_maxgrad", keepdims=0)
        signals[p] = max_grad
        step = b.mul("lr", grads[p])
        w_next = b.sub(p, step)
        state[p] = (w_shape, w_next)

    constants: Dict[str, Tuple[Sequence[int], int]] = {}
    for inp in model.graph.input:
        if inp.name in initializers:
            continue
        shape = shapes.get(inp.name)
        constants[inp.name] = (
            tuple(int(d) for d in shape),
            elem_types.get(inp.name, TensorProto.FLOAT),
        )

    # simplify=False: add the signal outputs to the declared graph outputs
    # *before* running simplify() ourselves, so dead-code elimination
    # cannot drop a branch that (without this) heads nowhere else.
    step_graph = qat_graph.make_step_graph(
        b,
        constants=constants,
        state=state,
        scalars=["lr"],
        loss=loss_output,
        name="resident_train_step_instrumented",
        simplify=False,
    )
    step_model = step_graph.model
    for p, sig_name in signals.items():
        step_model.graph.output.append(
            onnx.helper.make_tensor_value_info(sig_name, TensorProto.FLOAT, [])
        )

    for inp in step_model.graph.input:
        if inp.name == "lr":
            del inp.type.tensor_type.shape.dim[:]
            inp.type.tensor_type.shape.dim.add().dim_value = 1

    legalize.legalize(step_model, _POST_BACKWARD_RULES)
    onnx.checker.check_model(step_model)

    from onnxsim import simplify as _simplify

    step_model, ok = _simplify(
        step_model,
        skipped_optimizers=[
            "fuse_matmul_add_bias_into_gemm",
            "fuse_transpose_into_gemm",
        ],
    )
    if not ok:
        raise RuntimeError("post-legalize simplify() failed its own correctness check")

    return step_model, step_graph.state, signals


def main(argv=None) -> int:
    out_path = (argv or sys.argv[1:])[0]
    forward = forward_model()
    from build_resident_train_step import add_mse_loss

    with_loss = add_mse_loss(forward, "logits", num_classes=10)
    step_model, state, signals = build_instrumented_step(with_loss, ["cw", "gw"])
    print(f"step graph: {len(step_model.graph.node)} nodes")
    for p, out_name in state.items():
        print(f"  state: {p} -> {out_name}")
    for p, out_name in signals.items():
        print(f"  signal: {p} -> {out_name}")
    onnx.save(step_model, out_path)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
