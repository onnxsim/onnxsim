import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

import onnxsim  # noqa: E402
from onnxsim.pytree_io import (  # noqa: E402
    PYTREE_PATH_KEY,
    annotate_pytree_io,
    export_exported_program_to_onnx,
    pytree_io_paths,
)


class _Nested(torch.nn.Module):
    def forward(self, x, cfg):
        y = x * cfg["scale"][0] + cfg["bias"]
        return {"out": (y, [y.sum()]), "aux": cfg["scale"]}


class _Single(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _nested_program():
    x = torch.randn(2, 3)
    cfg = {"scale": (torch.tensor([2.0]),), "bias": torch.ones(3)}
    return torch.export.export(_Nested(), (x,), {"cfg": cfg})


def _paths_on_io(model):
    def paths(values):
        return [
            [p.value for p in v.metadata_props if p.key == PYTREE_PATH_KEY]
            for v in values
        ]

    return paths(model.graph.input), paths(model.graph.output)


def test_pytree_io_paths_follow_args_kwargs_and_outputs():
    input_paths, output_paths = pytree_io_paths(_nested_program())
    assert input_paths == ["args/0", "kwargs/cfg/scale/0", "kwargs/cfg/bias"]
    assert output_paths == ["out/0", "out/1/0", "aux/0"]


def test_export_helper_annotates_graph_io_and_simplify_keeps_it():
    ep = _nested_program()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested.onnx")
        export_exported_program_to_onnx(ep, path)
        model = onnx.load(path)
    inputs, outputs = _paths_on_io(model)
    assert inputs == [["args/0"], ["kwargs/cfg/scale/0"], ["kwargs/cfg/bias"]]
    assert outputs == [["out/0"], ["out/1/0"], ["aux/0"]]

    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    sim_inputs, sim_outputs = _paths_on_io(simplified)
    assert sim_inputs == inputs
    assert sim_outputs == outputs


def test_export_helper_can_skip_annotation():
    ep = _nested_program()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plain.onnx")
        export_exported_program_to_onnx(ep, path, annotate_pytree=False)
        model = onnx.load(path)
    inputs, outputs = _paths_on_io(model)
    assert all(paths == [] for paths in inputs + outputs)


def test_single_tensor_output_gets_root_path():
    ep = torch.export.export(_Single(), (torch.randn(2),))
    assert pytree_io_paths(ep) == (["args/0"], [""])


def test_annotate_refuses_mismatched_graph_io():
    ep = _nested_program()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested.onnx")
        export_exported_program_to_onnx(ep, path, annotate_pytree=False)
        model = onnx.load(path)
    del model.graph.output[-1]
    with pytest.raises(ValueError, match="graph I/O does not match"):
        annotate_pytree_io(model, ep)
