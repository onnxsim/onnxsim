"""Tests for ``scripts/axera/build_wavernn_gru_probe.py``'s pure-ONNX
``op_coverage`` helper -- the only piece of that script not gated on
``torch``/``torchaudio`` (the real WaveRNN export itself is exercised
manually, following this project's convention for every other
torch-dependent axera build script: see
``docs/axera-audio-speech-op-coverage.md``'s GRU section for that real
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

import build_wavernn_gru_probe as m  # noqa: E402


def test_conv_has_a_backward_rule_and_npu_support():
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,1,8] x, float[2,1,3] w) => (float[1,2,6] y)
        {
          y = Conv(x, w)
        }
        """
    )
    assert m.op_coverage(model)["Conv"] == (True, True)


def test_gru_has_neither_a_backward_rule_nor_npu_support():
    """The actual finding this probe exists to confirm on a real
    architecture, checked here against the coverage table directly rather
    than a real GRU export (which needs `torch`/`torchaudio`) -- and the
    stricter of the two negatives compared to `LSTM` (which is at least
    NPU-supported): a GRU-containing model cannot even run on this
    hardware at inference."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,2,4] x, float[1,12,4] w, float[1,12,4] r) => (float[1,1,2,4] y)
        {
          y = GRU<hidden_size=4>(x, w, r)
        }
        """
    )
    assert m.op_coverage(model)["GRU"] == (False, False)
