"""Tests for ``scripts/axera/build_parakeet_lstm_probe.py``'s pure-ONNX
``op_coverage`` helper -- the only piece of that script not gated on
``torch``/``transformers`` (the real Parakeet-decoder export itself is
exercised manually, following this project's convention for every other
torch/transformers-dependent axera build script: see
``docs/axera-audio-speech-op-coverage.md``'s LSTM section for that real
result).
"""

import os
import sys

from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_parakeet_lstm_probe as m  # noqa: E402


def _mixed_coverage_model():
    """One op with a backward rule and NPU support (`MatMul`), one NPU-
    supported op with no backward rule (`LSTM`, this script's whole point),
    and one op supported by neither table (`Shape`) -- covering all three
    real cells `op_coverage`'s cross-reference can produce."""
    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[2,2] x, float[2,2] w) => (float[2] shp)
        {
          y = MatMul(x, w)
          shp = Shape(y)
        }
        """
    )


def test_matmul_has_a_backward_rule_and_npu_support():
    coverage = m.op_coverage(_mixed_coverage_model())
    assert coverage["MatMul"] == (True, True)


def test_shape_has_neither_a_backward_rule_nor_npu_support():
    coverage = m.op_coverage(_mixed_coverage_model())
    assert coverage["Shape"] == (False, False)


def test_lstm_is_npu_supported_but_has_no_backward_rule():
    """The actual finding this probe exists to confirm on a real
    architecture, checked here against the coverage table directly rather
    than a real LSTM export (which needs `torch`/`transformers`)."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,2,4] x, float[1,4,4] w, float[1,4,4] r) => (float[1,1,2,4] y)
        {
          y = LSTM<hidden_size=4>(x, w, r)
        }
        """
    )
    assert m.op_coverage(model)["LSTM"] == (False, True)
