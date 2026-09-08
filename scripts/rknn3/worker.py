#!/usr/bin/env python3
"""Check onnxsim against RKNN3-Toolkit's `load_llm()` path, in an isolated
subprocess. Run as ``worker.py`` (no arguments -- unlike the sibling
`scripts/rknn` harness, this one always exercises the single, fixed real HF
checkpoint named by `rknn3_backend.MODEL_NAME`; there is no synthetic
suite for the LLM path, see `rknn3_backend.py`'s docstring for why).

`load_llm()`'s own C-extension code can abort hard on an unexpected graph,
so this runs in its own process and the final stdout line is exactly
``__RESULT__<json>`` -- the same protocol `scripts/rknn/worker.py` uses.

Framed the same way as that CNN harness: *original vs. onnxsim-simplified
ONNX, both converted and run through the same real RKNN3 `load_llm()` + PC
simulator*, so a fixed simulator/converter limitation cancels out and only
an onnxsim-introduced change fails the check. For the one fixed model:

1. Export a real, tiny Qwen2.5-architecture HF checkpoint to the ONNX +
   `.config.pkl` + `.embed.bin` triple `load_llm()` expects.
2. `simplify` the ONNX with onnxsim.
3. Convert + run the **original** ONNX through `load_llm()` + `build()` +
   the PC simulator.
   * If that already fails, RKNN3-Toolkit simply doesn't support this
     graph -- status ``unsupported`` (reported, not a failure).
4. Convert + run the **simplified** ONNX the same way.
   * If the original worked but the simplified doesn't -> `rknn3_regression`
     (a failure): simplification broke `load_llm()` compatibility.
5. Compare the two prefill logits. Divergence beyond tolerance ->
   `rknn3_regression`: simplification changed the simulator result.

Status values mirror `scripts/rknn/worker.py`: ``ok``, ``rknn3_regression``,
``simplify_error``, ``unsupported``, ``skipped``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def check() -> dict:
    res = {
        "model": None,
        "status": "error",
        "diff_vs_rknn3_orig": None,
        "error": None,
        "seconds": None,
    }
    t0 = time.time()
    try:
        import numpy as np
        import rknn3_backend as rb

        res["model"] = rb.MODEL_NAME

        if not rb.RKNN3_AVAILABLE:
            res["status"] = "skipped"
            res["error"] = rb.unavailable_reason()
            return res

        from onnxsim import simplify

        with (
            tempfile.TemporaryDirectory() as orig_dir,
            tempfile.TemporaryDirectory() as simp_dir,
        ):
            artifacts = rb.export_llm_artifacts(orig_dir, seq_len=rb.SEQ_LEN)

            import onnx

            model = onnx.load(artifacts["onnx"])
            try:
                simp, _check_ok = simplify(model)
            except Exception as exc:
                res["status"] = "simplify_error"
                res["error"] = f"{type(exc).__name__}: {exc}"
                return res

            simp_onnx_path = os.path.join(simp_dir, "model.onnx")
            onnx.save(simp, simp_onnx_path)

            token_ids = np.array(
                [[9707, 11, 1879, 0]], dtype=np.int64
            )  # "Hello, world!"

            try:
                logits_orig = rb.run(
                    artifacts["onnx"],
                    artifacts["config"],
                    artifacts["embed"],
                    token_ids,
                    seq_len=rb.SEQ_LEN,
                )
            except Exception as exc:
                res["status"] = "unsupported"
                res["error"] = f"original graph not supported by load_llm(): {exc}"
                return res

            try:
                # config/embed are unaffected by simplify() (they come from
                # the checkpoint's tokenizer/config/weights, not the ONNX
                # graph) so the same sidecar files are reused here.
                logits_simp = rb.run(
                    simp_onnx_path,
                    artifacts["config"],
                    artifacts["embed"],
                    token_ids,
                    seq_len=rb.SEQ_LEN,
                )
            except Exception as exc:
                res["status"] = "rknn3_regression"
                res["error"] = f"simplified graph broke load_llm() convert/run: {exc}"
                return res

            agree, diff = rb.compare_logits(logits_orig, logits_simp)
            res["diff_vs_rknn3_orig"] = diff
            if agree:
                res["status"] = "ok"
            else:
                res["status"] = "rknn3_regression"
                res["error"] = (
                    f"simplified load_llm() output diverged from original "
                    f"(max_abs_diff={diff:.3g})"
                )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-800:]
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    print("__RESULT__" + json.dumps(check()))
