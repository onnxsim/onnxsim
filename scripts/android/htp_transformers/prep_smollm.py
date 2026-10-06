#!/usr/bin/env python3
"""Prepare SmolLM2-135M (WikiText-2, 128-token windows) variants and inputs for the Snapdragon HTP runs (run_smollm.py).

    python prep_smollm.py --work /big/disk/dir [--windows 16]

Writes <work>/models/smollm_{orig,naive,adaptive}.onnx (orig: simplified only; naive: RMSNorm decomposed with x*x; adaptive: row-max
scaled, npu_rewrite.py default) and <work>/data/{win.bin,win.manifest,ref_nll.npy}. Reuses scripts/allwinner/transformer_accuracy/
study_smollm.py; run it from a directory that is not the repository root.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ALLWINNER = REPO / "scripts" / "allwinner"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--windows", type=int, default=16)
    a = p.parse_args()
    models, data = a.work / "models", a.work / "data"
    models.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    sm = load(ALLWINNER / "transformer_accuracy" / "study_smollm.py", "study_smollm")
    npu = sm.load_npu_rewrite()
    import onnx
    from onnx import shape_inference

    import onnxsim

    model_dir, win, _ = sm.prepare(a.work, a.seq, a.windows, 1)
    wrapper = sm.build_wrapper(model_dir, a.seq)
    raw = models / "smollm_b1.onnx"
    if not raw.exists():
        sm.export_onnx(wrapper, win, 1, raw)
    simplified, ok = onnxsim.simplify(onnx.load(str(raw)))
    assert ok, "onnxsim could not validate the simplified model"
    onnx.save(simplified, models / "smollm_orig.onnx")
    for name, scaling in (("naive", "none"), ("adaptive", "max")):
        rewritten, stats = npu.rewrite(simplified, scaling)
        onnx.save(
            shape_inference.infer_shapes(rewritten), models / f"smollm_{name}.onnx"
        )
        print(f"smollm_{name}.onnx: rewrites {dict(stats)}")

    import torch

    with torch.no_grad():
        ref = np.concatenate(
            [
                sm.nll(
                    wrapper(torch.from_numpy(win[i : i + 1])).numpy(), win[i : i + 1]
                )
                for i in range(len(win))
            ]
        )
    np.save(data / "ref_nll.npy", ref)
    print(f"torch fp32 perplexity over {len(win)} windows: {np.exp(ref.mean()):.3f}")
    win.tofile(data / "win.bin")
    np.save(data / "win_ids.npy", win)
    (data / "win.manifest").write_text(f"input_ids i64 win.bin {len(win)},1,{a.seq}\n")


if __name__ == "__main__":
    main()
