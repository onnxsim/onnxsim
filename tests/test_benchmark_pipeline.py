import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "apple" / "benchmark_onnx_pipeline.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("benchmark_onnx_pipeline", MODULE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def test_resolve_backends_defaults_to_manifest_choice():
    # Without --fallback-to-tinygrad the manifest's choice stands even when that
    # backend has no runner, so a Core ML measurement can never silently become a
    # tinygrad one.
    runners = {("head", "coreml"): object()}
    selected, subs = PIPELINE._resolve_backends(
        ["head"], {"head": "coreml"}, runners, False
    )
    assert selected == {"head": "coreml"}
    assert subs == []


def test_resolve_backends_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported backend"):
        PIPELINE._resolve_backends(["head"], {"head": "nope"}, {}, False)


def test_resolve_backends_prefers_jit_then_eager():
    # A stage whose Core ML runner is missing (e.g. RoiAlign cannot be lowered)
    # falls back to Metal JIT when that built, and to eager otherwise.
    runners = {("head", "tinygrad_metal_jit"): object()}
    selected, subs = PIPELINE._resolve_backends(
        ["head"], {"head": "coreml"}, runners, True
    )
    assert selected == {"head": "tinygrad_metal_jit"}
    assert subs == [
        {
            "stage": "head",
            "requested_backend": "coreml",
            "used_backend": "tinygrad_metal_jit",
        }
    ]

    runners = {("head", "tinygrad_metal"): object()}
    selected, subs = PIPELINE._resolve_backends(
        ["head"], {"head": "coreml"}, runners, True
    )
    assert selected == {"head": "tinygrad_metal"}
    assert subs[0]["used_backend"] == "tinygrad_metal"


def test_resolve_backends_leaves_working_stages_alone():
    # Fallback is per stage and only for stages that actually failed, so a
    # working Core ML stage is never swapped out.
    runners = {
        ("backbone", "coreml"): object(),
        ("head", "tinygrad_metal_jit"): object(),
    }
    selected, subs = PIPELINE._resolve_backends(
        ["backbone", "head"],
        {"backbone": "coreml", "head": "coreml"},
        runners,
        True,
    )
    assert selected == {"backbone": "coreml", "head": "tinygrad_metal_jit"}
    assert [s["stage"] for s in subs] == ["head"]


def test_resolve_backends_leaves_unfixable_stage_for_the_error_path():
    # If no tinygrad backend has a runner either, the stage keeps its requested
    # backend and the existing "selected backend unavailable" error still fires
    # (that check looks runners up by the selected name).
    selected, subs = PIPELINE._resolve_backends(["head"], {"head": "coreml"}, {}, True)
    assert selected == {"head": "coreml"}
    assert subs == []


def _two_input_model(name, input_names, output_name):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", input_names, [output_name])],
        name,
        [
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [2, 3]
            )
            for input_name in input_names
        ],
        [
            onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _model(name, input_name, output_name, op):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node(op, [input_name], [output_name])],
        name,
        [
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, [2, 3]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def test_fuse_stage_models_connects_outputs_and_explicit_inputs():
    first = _model("first", "x", "features", "Identity")
    second = _model("second", "features", "head", "Relu")
    fused, sources, outputs = PIPELINE._fuse_stage_models(
        ["first", "second"],
        {"first": first, "second": second},
        [{"from": ["first", "features"], "to": ["second", "features"]}],
        [],
    )
    assert [value.name for value in fused.graph.output] == ["fused_1_head"]
    assert outputs == ["head"]
    assert sources == {"fused_0_x": ("first", "x")}
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ort.InferenceSession(
        fused.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"fused_0_x": x})[0]
    np.testing.assert_array_equal(result, x)


def test_fuse_stage_models_connects_explicit_shared_input():
    first = _model("first", "x", "features", "Identity")
    second = _two_input_model("second", ["features", "shared"], "head")
    fused, sources, _ = PIPELINE._fuse_stage_models(
        ["first", "second"],
        {"first": first, "second": second},
        [{"from": ["first", "features"], "to": ["second", "features"]}],
        [{"from": ["first", "x"], "to": ["second", "shared"]}],
    )
    assert sources["fused_0_x"] == ("first", "x")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ort.InferenceSession(
        fused.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"fused_0_x": x})[0]
    np.testing.assert_array_equal(result, x * 2)


def test_fuse_stage_models_rejects_forward_input_connection_without_mapping():
    first = _model("first", "x", "features", "Identity")
    second = _model("second", "features", "head", "Relu")
    with pytest.raises(ValueError, match="no connection"):
        PIPELINE._fuse_stage_models(
            ["first", "second"],
            {"first": first, "second": second},
            [],
            [],
        )
