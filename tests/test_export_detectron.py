# Regression test for onnxsim.export_detectron_model: the Detectron2
# counterpart of onnxsim.export_transformers_model /
# onnxsim.export_diffusion_model. Checks the wrapper does the same thing as
# the manual build-model-then-trace-with-TracingAdapter-then-simplify recipe
# Detectron2's own tools/deploy/export_model.py runs by hand -- produces a
# valid, strictly-simplified ONNX file and returns the check_ok result --
# using a randomly-initialized model (no weights=, so no network call) to
# keep this runnable offline.
#
# torch and detectron2 are not normal test dependencies (detectron2 isn't
# even on PyPI -- see onnxsim.export_detectron_model's docstring), so this
# skips unless they're already importable. To run it locally::
#
#     pip install torch 'git+https://github.com/facebookresearch/detectron2.git'
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_export_detectron.py -v

import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("detectron2")

_CONFIG = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"


def test_export_detectron_model_simplifies(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    check_ok = onnxsim.export_detectron_model(_CONFIG, out_path)

    assert check_ok is True
    model = onnx.load(out_path, load_external_data=False)
    assert len(model.graph.node) > 0


def test_export_detectron_model_returns_check_results(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    check_ok = onnxsim.export_detectron_model(_CONFIG, out_path, check_n=2)

    assert check_ok is True


def test_export_detectron_model_save_as_external_data(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    onnxsim.export_detectron_model(_CONFIG, out_path, save_as_external_data=True)

    assert os.path.exists(out_path + ".data")
    model, _pool = onnxsim.load_model(out_path)
    assert len(model.graph.node) > 0
