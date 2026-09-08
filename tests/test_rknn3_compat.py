"""RKNN3 (Rockchip's next-gen RK1820/RK1828/RK3572 NPU line) LLM compatibility
test.

Verifies that onnxsim's output still converts and runs through
`rknn.api.RKNN.load_llm()` -- RKNN3-Toolkit's dedicated LLM conversion entry
point, distinct from the ordinary `load_onnx()` path
`tests/test_rknn_compat.py` covers for `rknn-toolkit2` (a different,
incompatible package despite the identical import path) -- and that
simplification does not change the result on RKNN3's PC simulator. See
`scripts/rknn3/rknn3_backend.py` for the fidelity tier this does and does not
cover, and for the real, verified compatibility issues this harness works
around: a torch>=2.9 ONNX-exporter default-flip bug in the upstream export
reference implementation, an SDK-vs-`scripts/common` `sys.modules["common"]`
name collision, and why `compare_logits()`'s tolerance is looser than the
CNN harness's (a single `float16`-ULP-scale divergence from the PC
simulator's `float16`-only compute path, not an onnxsim correctness bug).

Unlike `tests/test_rknn_compat.py`'s network-free synthetic-model suite, this
module downloads a small (~10MB), real, publicly hosted Qwen2.5-architecture
checkpoint (`rknn3_backend.MODEL_NAME`) -- RKNN3's LLM path is a fixed-shape
ONNX export of a real transformer, not something a from-scratch synthetic
graph exercises meaningfully (`load_llm()` does real attention/RoPE pattern
matching on the graph). The whole module is skipped when RKNN3-Toolkit (the
`rknn-toolkit` wheel from `airockchip/rknn3-toolkit`, not on PyPI) is not
installed, so it is safe to keep in ``tests/``.
"""

import os
import sys

import pytest

_RKNN3_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "rknn3"
)
if _RKNN3_DIR not in sys.path:
    sys.path.insert(0, _RKNN3_DIR)

import rknn3_backend as rb  # noqa: E402
import worker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not rb.RKNN3_AVAILABLE,
    reason=f"RKNN3-Toolkit unavailable: {rb.unavailable_reason()}",
)


def test_rknn3_llm_compat():
    """Simplification must not break or change the `load_llm()` result for
    the fixed tiny Qwen2.5-architecture checkpoint this harness uses."""
    result = worker.check()
    # ``unsupported`` (RKNN3-Toolkit can't convert/run the graph at all) is
    # acceptable; a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_load_llm_is_the_rknn3_api():
    """Sanity check on the two toolkits' shared import path: this session's
    `rknn.api.RKNN` must be the RKNN3 one (`load_llm`), not rknn-toolkit2's
    (`load_onnx`-only) -- see rknn3_backend.py's docstring."""
    from rknn.api import RKNN

    assert hasattr(RKNN, "load_llm")
