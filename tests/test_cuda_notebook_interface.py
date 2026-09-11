"""CPU-only rehearsal of examples/cuda_feature_tests/cuda_feature_tests.ipynb.

That notebook is meant to be run by hand against a real GPU (see its own
README for why it isn't wired into CI directly), so ordinary CPU-only CI never
actually executes it. Left alone, a rename/removal of the interfaces it
depends on -- the CLI's ``--providers``/``--cuda`` flags, ``Runner``/
``as_ort_value``, ``measure_accuracy_drop``'s ``providers`` kwarg -- would
only surface the next time someone happens to open the notebook on a GPU
machine, which could be a long time after the breaking change landed.

These tests call the exact same interfaces with ``CPUExecutionProvider``
standing in for ``CUDAExecutionProvider``, so a signature/flag change breaks
an ordinary CPU test run immediately. They are not a substitute for actually
running the notebook: onnxruntime accepts an unavailable provider name at the
API level identically regardless of whether real CUDA hardware and driver
sit behind it, so nothing here proves CUDA execution itself still works.
Provider list/tuple handling that doesn't need the CLI or the ``Runner``/
``measure_accuracy_drop`` combination is already covered by
tests/test_backend.py; this file only adds the notebook-specific interfaces
that aren't exercised anywhere else.
"""

import sys

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim import backend
from onnxsim.accuracy import measure_accuracy_drop


def _model(a_value: float = 1.0) -> onnx.ModelProto:
    """A model whose ``a + b`` can be constant-folded, then added to input."""
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 18]>
        foldable (float[2,2] x) => (float[2,2] y)
        <float[2,2] a = {{{a_value}, {a_value}, {a_value}, {a_value}}}, float[2,2] b = {{2.0, 2.0, 2.0, 2.0}}>
        {{
          c = Add(a, b)
          y = Add(c, x)
        }}
        """
    )
    onnx.checker.check_model(model)
    return model


X = np.arange(4, dtype=np.float32).reshape(2, 2)


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_cli_providers_flag_folds_model():
    # Mirrors the notebook's CLI test (Test D): --providers is wired all the
    # way through to constant folding, not just parsed and ignored.
    import tempfile
    from pathlib import Path

    from onnxsim import onnx_simplifier

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "in.onnx")
        output_path = str(Path(tmpdir) / "out.onnx")
        onnx.save(_model(), input_path)

        argv = sys.argv
        try:
            sys.argv = [
                "onnxsim",
                input_path,
                output_path,
                "--providers",
                "CPUExecutionProvider",
            ]
            onnx_simplifier.main()
        finally:
            sys.argv = argv

        opt = onnx.load(output_path)
        assert len(opt.graph.node) == 1
        out = backend.run_model(opt, {"x": X})
        np.testing.assert_allclose(out["y"], X + 3.0)


def test_cli_cuda_and_providers_are_mutually_exclusive():
    # Mirrors the notebook's use of the --cuda shortcut (Test D): this proves
    # --cuda still exists and is still wired to the same mutual-exclusivity
    # check as --providers, without needing a CUDA build to exercise it.
    import tempfile
    from pathlib import Path

    from onnxsim import onnx_simplifier

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "in.onnx")
        output_path = str(Path(tmpdir) / "out.onnx")
        onnx.save(_model(), input_path)

        argv = sys.argv
        try:
            sys.argv = [
                "onnxsim",
                input_path,
                output_path,
                "--cuda",
                "--providers",
                "CPUExecutionProvider",
            ]
            with pytest.raises(RuntimeError, match="mutually exclusive"):
                onnx_simplifier.main()
        finally:
            sys.argv = argv


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for OrtValue/Runner"
)
def test_runner_run_with_ort_values_roundtrip():
    # Mirrors the notebook's DLPack test (Test F): Runner + as_ort_value +
    # run_with_ort_values used together, the way the notebook chains them for
    # a CUDA torch.Tensor. CPUExecutionProvider stands in for CUDA here.
    runner = backend.Runner(_model(), providers=["CPUExecutionProvider"])
    assert runner.supports_ort_values()
    ort_inputs = {"x": backend.as_ort_value(X)}
    outputs = runner.run_with_ort_values(ort_inputs)
    np.testing.assert_allclose(outputs["y"].numpy(), X + 3.0)


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_measure_accuracy_drop_accepts_providers_kwarg():
    # Mirrors the notebook's Test G: measure_accuracy_drop's providers kwarg
    # is still accepted and still threaded down to both model runs.
    report = measure_accuracy_drop(
        _model(a_value=1.0),
        _model(a_value=1.01),
        calibration_data=[{"x": X}],
        providers=["CPUExecutionProvider"],
    )
    assert report.all_finite
    assert report.per_output["y"].max_abs_error > 0
