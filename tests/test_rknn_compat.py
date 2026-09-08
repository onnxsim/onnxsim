"""RKNN (Rockchip NPU) compatibility test.

Verifies that onnxsim's output still converts and runs through
`rknn-toolkit2` -- Rockchip's real ONNX -> RKNN converter -- and that
simplification does not change the result on its PC simulator. Uses the
pip-installable ``rknn-toolkit2`` wheel: on a plain x86-64 CPU host with no
RK35xx/RV1106 device attached, ``init_runtime(target=None)`` still performs
the full offline convert + build and runs the result on Rockchip's own PC
simulator, so this runs on an ordinary CI runner. See
``scripts/rknn/rknn_backend.py`` for the fidelity tier this does and does not
cover, and for a real, verified `onnx.mapping` compatibility shim this
harness needs to import `rknn-toolkit2` at all on a modern ``onnx`` install.

The whole module is skipped when ``rknn-toolkit2`` is not installed (the
default test matrix does not pull it in), so it is safe to keep in
``tests/``. The heavier, curated model sweep lives in ``scripts/rknn`` and
runs on a schedule; this file is the fast, in-tree smoke test.
"""

import os
import sys

import numpy as np
import pytest

# The RKNN harness lives under scripts/rknn; reuse it rather than duplicate.
_RKNN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "rknn"
)
if _RKNN_DIR not in sys.path:
    sys.path.insert(0, _RKNN_DIR)
_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import rknn_backend as rb  # noqa: E402

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# scripts/axera/_local_import.py's docstring for why a plain import here can
# silently resolve to a *different* vendor's module in the full test suite.
from _local_import import fresh  # noqa: E402

models = fresh("models", _RKNN_DIR)
check = fresh("worker", _RKNN_DIR).check

pytestmark = pytest.mark.skipif(
    not rb.RKNN_AVAILABLE,
    reason=f"rknn-toolkit2 unavailable: {rb.unavailable_reason()}",
)


@pytest.mark.parametrize("name", models.names())
def test_rknn_compat_suite(name):
    """Each suite model: simplification must not break or change the RKNN result."""
    result = check(name, None)
    # ``unsupported`` (RKNN can't convert/run the graph at all) is acceptable;
    # a regression / crash / error is not.
    assert result["status"] in ("ok", "unsupported"), result


def test_rknn_matches_cpu_reference():
    """A directly-built graph: RKNN(simplified) roughly matches the ORT CPU
    reference (the PC simulator is not bit-exact -- see rknn_backend.py)."""
    from onnxsim import simplify

    model = models.conv_bn_relu()
    simp, _ = simplify(model)
    feeds = rb.random_feeds(model, seed=0)

    reference = rb.run_with_cpu(model, feeds)
    candidate = rb.run(simp, feeds)

    close, max_diff = rb.compare(reference, candidate)
    assert close, f"RKNN output diverged from CPU reference: max_abs_diff={max_diff}"


def test_simplify_matches_on_rknn():
    """Simplification should not change what the RKNN PC simulator computes."""
    from onnxsim import simplify

    model = models.redundant_transpose()
    simp, _ = simplify(model)
    feeds = rb.random_feeds(model, seed=1)

    rknn_orig = rb.run(model, feeds)
    rknn_simp = rb.run(simp, feeds)

    close, max_diff = rb.compare(rknn_orig, rknn_simp)
    assert close, f"RKNN output diverged after simplification: max_abs_diff={max_diff}"


def test_onnx_mapping_shim_is_installed():
    """The compat shim documented in rknn_backend.py's docstring must have run
    (this is what makes `import rknn.api` possible on a modern `onnx`)."""
    import onnx

    assert hasattr(onnx, "mapping")
    assert onnx.TensorProto.FLOAT in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE
    assert np.float32 in onnx.mapping.NP_TYPE_TO_TENSOR_TYPE
