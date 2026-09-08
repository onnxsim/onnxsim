# Regression test for onnxsim.export_sam2_model: the SAM 2 counterpart of
# onnxsim.export_detectron_model. Checks the wrapper does the same thing as
# the manual build-model-then-trace-encoder-and-decoder-then-simplify recipe
# samexporter's own export_sam2 CLI script runs by hand -- produces the
# expected encoder.onnx/decoder.onnx files, simplifies each, and returns a
# per-file check_ok map -- using a randomly-initialized model (no
# checkpoint=, so no network call) to keep this runnable offline.
#
# torch, samexporter, hydra-core, and the real (source-installed) sam2
# package are not normal test dependencies -- sam2 isn't even on PyPI under
# its official name, see onnxsim.export_sam2_model's docstring -- so this
# skips unless they're already importable. To run it locally::
#
#     pip install torch samexporter hydra-core
#     pip install 'git+https://github.com/facebookresearch/sam2.git'
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_export_sam2.py -v

import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("samexporter")
pytest.importorskip("sam2")

_MODEL_TYPE = "sam2.1_hiera_tiny"

_EXPECTED_FILES = {"encoder.onnx", "decoder.onnx"}


def test_export_sam2_model_simplifies(tmp_path):
    out_dir = str(tmp_path)

    results = onnxsim.export_sam2_model(_MODEL_TYPE, out_dir)

    assert set(results.keys()) == _EXPECTED_FILES
    assert all(results.values()), results
    for name in _EXPECTED_FILES:
        model = onnx.load(os.path.join(out_dir, name), load_external_data=False)
        assert len(model.graph.node) > 0


def test_export_sam2_model_returns_check_results(tmp_path):
    out_dir = str(tmp_path)

    results = onnxsim.export_sam2_model(_MODEL_TYPE, out_dir, check_n=2)

    assert set(results.keys()) == _EXPECTED_FILES
    assert all(results.values()), results


def test_export_sam2_model_save_as_external_data(tmp_path):
    out_dir = str(tmp_path)

    onnxsim.export_sam2_model(_MODEL_TYPE, out_dir, save_as_external_data=True)

    for name in _EXPECTED_FILES:
        assert os.path.exists(os.path.join(out_dir, name + ".data"))
        model, _pool = onnxsim.load_model(os.path.join(out_dir, name))
        assert len(model.graph.node) > 0


def test_export_sam2_model_rejects_unknown_model_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown model_type"):
        onnxsim.export_sam2_model("not-a-real-model-type", str(tmp_path))
