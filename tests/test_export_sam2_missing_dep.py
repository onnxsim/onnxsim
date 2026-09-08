# onnxsim.export_sam2_model's optional-dependency error message, checked
# with 'samexporter' forced unimportable regardless of whether it is
# actually installed in this environment -- unlike test_export_sam2.py
# (which needs torch/samexporter/sam2 for real to exercise the export
# itself), this test's whole point is exercising the *absence* path, so it
# must not be skipped just because those heavy packages happen to be
# present.

import builtins

import pytest

import onnxsim


def test_export_sam2_model_needs_samexporter(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("samexporter"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"onnxsim\[sam2\]"):
        onnxsim.export_sam2_model("sam2.1_hiera_tiny", "/tmp/does-not-matter")
