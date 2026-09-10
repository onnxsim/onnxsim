"""Tests against Voyager SDK's *real* compiler (`axelera.compiler`, from the
optional `axelera-devkit` package) -- unlike `test_axelera_voyager_op_
support.py` (which never imports voyager-sdk or axelera.compiler at all),
these tests actually run onnxsim-simplified models through Axelera's own
quantizer for genuine numeric validation. See `scripts/axelera/
voyager_backend.py`'s module docstring for what this backend can and can't
do, and for the two concrete real-compiler observations it's built from.

Skipped entirely unless `axelera.compiler` is importable -- these packages
are large (torch, CUDA toolkit packages, TVM, and more) and genuinely
optional; nothing in onnxsim's own test suite or CI is expected to install
them. Install with:

    pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt axelera-devkit[all]
"""

import os
import sys

import numpy as np
import pytest
from onnx import numpy_helper, parser

import onnxsim

_AXELERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axelera"
)
if _AXELERA_DIR not in sys.path:
    sys.path.insert(0, _AXELERA_DIR)

import voyager_backend as backend  # noqa: E402

if not backend.has_axelera_compiler():
    pytest.skip(
        "axelera.compiler (from axelera-devkit) is not installed -- see this "
        "module's docstring for the (large, optional) install command",
        allow_module_level=True,
    )

compiler = pytest.importorskip("axelera.compiler")
torch = pytest.importorskip("torch")


def _conv_bn_relu_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[1,3,16,16] x) => (float[1,8,16,16] y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w)
          bn = BatchNormalization(c, scale, bias, mean, var)
          y = Relu(bn)
        }
        """
    )
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
    scale = np.abs(rng.standard_normal((8,)).astype(np.float32)) + 0.1
    bias = rng.standard_normal((8,)).astype(np.float32)
    mean = rng.standard_normal((8,)).astype(np.float32) * 0.1
    var = np.abs(rng.standard_normal((8,)).astype(np.float32)) + 0.5
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(w, name="w"),
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(bias, name="bias"),
            numpy_helper.from_array(mean, name="mean"),
            numpy_helper.from_array(var, name="var"),
        ]
    )
    return model


def _conv_model(auto_pad):
    extra = f', auto_pad = "{auto_pad}"' if auto_pad != "NOTSET" else ""
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[1,3,16,16] x) => (float[1,8,16,16] y)
        {{
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]{extra}>(x, w)
          y = Relu(c)
        }}
        """
    )
    w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, name="w"))
    return model


def _calib_fn(shape=(1, 3, 16, 16), n=16, seed=1):
    def gen():
        rng = np.random.default_rng(seed)
        for _ in range(n):
            yield rng.standard_normal(shape).astype(np.float32)

    return gen


def test_bn_fusion_is_bit_identical_through_real_quantizer():
    """The concrete result documented in voyager_backend.py's docstring,
    pinned as a regression test: onnxsim's Conv+BatchNorm fusion must not
    change what the real Voyager SDK compiler's quantizer produces.
    """
    raw = _conv_bn_relu_model()
    simplified, check_ok = onnxsim.simplify(raw, check_n=3)
    assert check_ok
    assert [n.op_type for n in raw.graph.node] == ["Conv", "BatchNormalization", "Relu"]
    assert [n.op_type for n in simplified.graph.node] == ["Conv", "Relu"]

    test_input = (
        np.random.default_rng(99).standard_normal((1, 3, 16, 16)).astype(np.float32)
    )
    result = backend.compare_before_after_simplify(
        raw, simplified, _calib_fn(), test_input
    )
    assert result["bit_identical"]
    assert result["max_abs_diff"] == 0.0


def test_conv_notset_auto_pad_quantizes_cleanly():
    model = _conv_model("NOTSET")
    q = backend.quantize(model, _calib_fn()())
    out = q(
        torch.from_numpy(
            np.random.default_rng(2).standard_normal((1, 3, 16, 16)).astype(np.float32)
        )
    )
    assert out.shape == (1, 8, 16, 16)


def test_conv_same_upper_auto_pad_fails_quantization_matching_voyager_ops_rule(caplog):
    """Cross-checks `voyager_ops.VOYAGER_OP_SUPPORT["Conv"]["rules"]` against
    the real compiler: violating `auto_pad == "NOTSET"` was observed to (a)
    log a WARNING quoting that exact constraint string, via the compiler's
    own `onnx_validator`, and (b) raise `QtoolsError` (with a shorter,
    differently-worded message -- the constraint text lives in the logged
    warning, not the exception itself) -- not a silent CPU fallback. See
    voyager_backend.py's docstring for why that refines onnx-support.md's
    own "falls back to CPU" prose.
    """
    import logging

    import voyager_ops as ops

    rule = ops.VOYAGER_OP_SUPPORT["Conv"]["rules"][0]
    assert rule == 'auto_pad == "NOTSET"'  # the exact string the real warning quotes

    model = _conv_model("SAME_UPPER")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(compiler.exceptions.QtoolsError, match="SAME_UPPER"):
            backend.quantize(model, _calib_fn()())

    assert f"Unsatisfied constraint: {rule}" in caplog.text
