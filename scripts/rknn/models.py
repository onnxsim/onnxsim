#!/usr/bin/env python3
"""RKNN-side alias for the shared synthetic model suite.

The suite lives in ``scripts/common/synthetic_models.py`` so the Apple
CoreML, Intel OpenVINO, Qualcomm QNN and Axera Pulsar2 harnesses can reuse it
without duplicating the graph builders. This module re-exports the same
public API so ``worker.py``'s ``import models`` and
``tests/test_rknn_compat.py``'s ``models.names()`` / ``models.conv_bn_relu()``
keep working unchanged.

``matmul_bias_tanh`` has a rank-2 input (no image layout to get wrong), so it
doubles as this harness's check that non-4-D graphs still convert -- RKNN is
built around CV models, and rank-2 MLP-style graphs are a real edge case for
the NCHW/NHWC handling in ``rknn_backend.run``.
"""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# See scripts/qualcomm/models.py's comment on why scripts/ is only kept on
# sys.path for the duration of this import.
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.synthetic_models import (  # noqa: E402,F401
        all_models,
        build,
        conv_bn_relu,
        foldable_shape_reshape,
        matmul_bias_tanh,
        names,
        redundant_transpose,
        sigmoid_mul_swish,
    )
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)

if __name__ == "__main__":
    for n, m in all_models().items():
        print(f"{n:24} {len(m.graph.node)} nodes")
