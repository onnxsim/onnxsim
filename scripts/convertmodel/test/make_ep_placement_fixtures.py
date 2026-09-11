#!/usr/bin/env python3
"""Emit minimal ONNX fixtures for the two documented EP fallback gaps
``onnxsim.check_webgpu_attention_support``/``onnxsim.check_webnn_support``
(``onnxsim/webgpu_target.py``/``onnxsim/webnn_target.py``) check for
statically, so ``webgpu_attention_placement.test.mjs``/
``webnn_reshape_placement.test.mjs`` can confirm the claim against a real
browser's execution-provider assignment rather than trusting ORT's own docs
alone.

Each gap gets a matched pair: a "flagged" fixture exercising the documented
gap, and a "clean" control identical in every other respect (same op, same
shapes) but without it. The control matters because "this op didn't run on
the GPU" is meaningless on its own -- it could mean "this configuration isn't
supported" (the claim under test) or just "this op is never supported at
all". Only the contrast between the two proves the *gap*, not just the op.

These are small, hand-built repro cases for a specific known EP gap, not
graphs produced by any onnxsim builder -- unlike ``make_step_graph_fixtures.py``
(whose docstring explains why its own fixtures must come from onnxsim's own
functions), there is no "real caller" graph shape to be faithful to here, so
building the minimal graph the gap needs directly, via ``onnx.parser``
(matching the convention ``tests/test_webgpu_target.py``/
``tests/test_webnn_target.py`` already use for the same two gaps in Python),
is exactly the right amount of graph.

Regenerate (only the ``onnx`` package is needed, not onnxsim itself)::

    python3 make_ep_placement_fixtures.py
"""

import json
import os

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))

B, S, H, NH = 1, 4, 8, 2


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _attention_model(with_mask):
    rng = np.random.default_rng(0)
    wqkv = _f32(rng.standard_normal((H, H * 3)) * 0.1, "wqkv")
    bqkv = _f32(rng.standard_normal(H * 3) * 0.1, "bqkv")
    attn_inputs = "pre, wqkv, bqkv, mask_index" if with_mask else "pre, wqkv, bqkv"
    graph_inputs = f"float[{B},{S},{H}] x"
    if with_mask:
        graph_inputs += f", int32[{B}] mask_index"
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g ({graph_inputs}) => (float[{B},{S},{H}] y)
        {{
          pre = Identity(x)
          attn = com.microsoft.Attention<num_heads = {NH}, qkv_hidden_sizes = [{H}, {H}, {H}]>({attn_inputs})
          y = Identity(attn)
        }}
        """
    )
    model.graph.initializer.extend([wqkv, bqkv])
    onnx.checker.check_model(model)
    return model


def _reshape_model(dynamic_shape):
    if dynamic_shape:
        # `shape` is computed at runtime (Shape() of another input), so it
        # isn't a constant -- the gap onnxruntime-web's WebNN operator table
        # documents for Reshape.
        body = """
        g (float[2,3] x, float[6] like) => (float[6] y)
        {
          pre = Identity(x)
          shape = Shape(like)
          reshaped = Reshape(pre, shape)
          y = Identity(reshaped)
        }
        """
        initializer = []
    else:
        shape = onnx.numpy_helper.from_array(np.array([6], dtype=np.int64), "shape")
        body = """
        g (float[2,3] x) => (float[6] y)
        {
          pre = Identity(x)
          reshaped = Reshape(pre, shape)
          y = Identity(reshaped)
        }
        """
        initializer = [shape]
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _save(model, name):
    path = os.path.join(HERE, name)
    onnx.save(model, path)
    return name


def main():
    manifest = {
        "webgpuAttention": {
            "flaggedOp": "Attention",
            "cases": {
                "clean": {
                    "file": _save(_attention_model(with_mask=False), "webgpu_attention_clean.onnx"),
                    "inputs": {"x": {"dims": [B, S, H], "dtype": "float32"}},
                },
                "mask": {
                    "file": _save(_attention_model(with_mask=True), "webgpu_attention_mask.onnx"),
                    "inputs": {
                        "x": {"dims": [B, S, H], "dtype": "float32"},
                        "mask_index": {"dims": [B], "dtype": "int32"},
                    },
                },
            },
        },
        "webnnReshape": {
            "flaggedOp": "Reshape",
            "cases": {
                "clean": {
                    "file": _save(_reshape_model(dynamic_shape=False), "webnn_reshape_constant.onnx"),
                    "inputs": {"x": {"dims": [2, 3], "dtype": "float32"}},
                },
                "dynamic": {
                    "file": _save(_reshape_model(dynamic_shape=True), "webnn_reshape_dynamic.onnx"),
                    "inputs": {
                        "x": {"dims": [2, 3], "dtype": "float32"},
                        "like": {"dims": [6], "dtype": "float32"},
                    },
                },
            },
        },
    }
    with open(os.path.join(HERE, "ep_placement_fixtures.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("wrote", ", ".join(c["file"] for g in manifest.values() for c in g["cases"].values()))
    print("wrote ep_placement_fixtures.json")


if __name__ == "__main__":
    main()
