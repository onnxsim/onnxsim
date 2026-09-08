# onnxsim.export_detectron_model's optional-dependency error message,
# checked with 'detectron2' forced unimportable regardless of whether it is
# actually installed in this environment -- unlike test_export_detectron.py
# (which needs torch/detectron2 for real to exercise the export itself),
# this test's whole point is exercising the *absence* path, so it must not
# be skipped just because those heavy packages happen to be present.

import builtins

import pytest

import onnxsim


def test_export_detectron_model_needs_detectron2(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("detectron2"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"onnxsim\[detectron2\]"):
        onnxsim.export_detectron_model("some-config.yaml", "/tmp/does-not-matter.onnx")
