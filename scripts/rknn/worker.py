#!/usr/bin/env python3
"""Check one model against the RKNN PC simulator, in an isolated subprocess.

Run as ``worker.py <model_name> [onnx_path]``. With just a name, the model
comes from the built-in ``models.py`` suite; with an ``onnx_path`` the graph
is loaded from disk (so the same worker can later drive real onnxmodelzoo
models, matching the other vendor harnesses). `rknn-toolkit2`'s converter can
abort at the C-extension level on an unsupported graph, so each model runs in
its own process and the final stdout line is exactly ``__RESULT__<json>`` --
the same protocol ``scripts/qualcomm/worker.py`` uses.

Framed the same way as the QNN check: *original vs. simplified through the
same RKNN PC-simulator build*, so a fixed simulator limitation (an op it
can't convert, or its own float-vs-reference numeric slack -- see
``rknn_backend.py``'s docstring) cancels out and only an onnxsim-introduced
change fails the run. For one model:

1. ``simplify`` it with onnxsim.
2. Build deterministic random inputs.
3. Convert + run the **original** graph through RKNN.
   * If that already fails, the converter/simulator simply does not support
     this graph -- status ``unsupported`` (reported, not a failure).
4. Convert + run the **simplified** graph through RKNN.
   * If the original converted but the simplified doesn't, simplification
     broke RKNN compatibility -- status ``rknn_regression`` (a failure).
5. Compare the two RKNN outputs. Divergence beyond tolerance means
   simplification changed the simulator result -- ``rknn_regression``.
6. Also record the ONNX Runtime CPU reference diff, as information only (the
   RKNN PC simulator is not expected to match it tightly).

Status values:

* ``ok``              - original & simplified both converted/ran and agreed.
* ``rknn_regression``  - the simplified graph broke RKNN compat or changed results.
* ``simplify_error``   - onnxsim raised on this model.
* ``unsupported``      - RKNN could not convert/run even the original graph.
* ``skipped``          - rknn-toolkit2 is not available on this host.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def check(model_name: str, onnx_path: str | None) -> dict:
    res = {
        "model": model_name,
        "status": "error",
        "orig_nodes": None,
        "simp_nodes": None,
        "diff_vs_rknn_orig": None,
        "diff_vs_cpu_ref": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import onnx
        import rknn_backend as rb

        if not rb.RKNN_AVAILABLE:
            res["status"] = "skipped"
            res["error"] = rb.unavailable_reason()
            return res

        from onnxsim import simplify

        if onnx_path:
            model = onnx.load(onnx_path)
        else:
            import models

            model = models.build(model_name)
        res["orig_nodes"] = len(model.graph.node)

        try:
            simp, _check_ok = simplify(model)
        except Exception as exc:
            res["status"] = "simplify_error"
            res["error"] = f"{type(exc).__name__}: {exc}"
            return res
        res["simp_nodes"] = len(simp.graph.node)

        feeds = rb.random_feeds(model, seed=0)

        # ORT CPU reference (informational end-to-end check).
        cpu_ref = rb.run_with_cpu(model, feeds)

        # Original graph through RKNN. If this fails, the converter/simulator
        # can't handle the graph at all -- not something onnxsim did.
        try:
            rknn_orig = rb.run(model, feeds)
        except Exception as exc:
            res["status"] = "unsupported"
            res["error"] = f"original graph not supported by RKNN: {exc}"
            return res

        # Simplified graph through RKNN. The original worked, so a failure
        # here is an onnxsim-introduced compatibility regression.
        try:
            rknn_simp = rb.run(simp, feeds)
        except Exception as exc:
            res["status"] = "rknn_regression"
            res["error"] = f"simplified graph broke RKNN convert/run: {exc}"
            return res

        # Simplification must not change the simulator result.
        agree, diff_backend = rb.compare(rknn_orig, rknn_simp)
        _, diff_ref = rb.compare(cpu_ref, rknn_simp)
        res["diff_vs_rknn_orig"] = diff_backend
        res["diff_vs_cpu_ref"] = diff_ref
        if agree:
            res["status"] = "ok"
        else:
            res["status"] = "rknn_regression"
            res["error"] = (
                f"simplified RKNN output diverged from original "
                f"(max_abs_diff={diff_backend:.3g})"
            )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    name = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else None
    print("__RESULT__" + json.dumps(check(name, path)))
