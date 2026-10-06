#!/usr/bin/env python3
"""Perplexity of a small decoder LLM (SmolLM2-135M on WikiText-2) under the quantization schemes of Allwinner's Acuity toolchain.

    python study_smollm.py --work /big/disk/dir [--only "fp16,W16"] [--json results.json]

The decoder counterpart of study_sst2.py: RMSNorm, rotary embeddings, grouped-query attention, a SiLU-gated MLP, a causal mask and a
tied 49 152-entry vocabulary. Pipeline: download (safetensors only) -> fixed 128-token WikiText-2 windows -> PyTorch fp32 perplexity ->
export with a constant additive causal mask (a static-shape deployment feeds one, and it avoids transformers' mask-building ops that
the legacy ONNX exporter cannot express) -> onnxsim + scripts/allwinner/npu_rewrite.py -> calibrate -> evaluate each scheme emulated by
quantsim.py. Needs torch, transformers, huggingface_hub, pandas, pyarrow, onnx, onnxsim and onnxruntime; run it from a directory that
is not the repository root. Perplexity here is over short 128-token windows, so its absolute value (about 39) is higher than the usual
long-context figure; only the comparison between schemes matters.

This simulates Acuity's schemes with fake-quantize nodes on every tensor; it is not Acuity and not the NPU (see quantsim.py).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quantsim as qs  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
DATASET = "Salesforce/wikitext"
DATA_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def load_npu_rewrite():
    spec = importlib.util.spec_from_file_location(
        "npu_rewrite", HERE.parent / "npu_rewrite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nll(logits, ids):
    """Per-token negative log-likelihood of ids[1:] given the logits at the previous positions."""
    lg = logits[:, :-1].astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -np.take_along_axis(logp, ids[:, 1:, None], -1)[..., 0]


def prepare(work, seq, windows, calib_windows):
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    model_dir = work / "model"
    snapshot_download(
        MODEL, local_dir=model_dir, allow_patterns=["*.json", "*.txt", "*.safetensors"]
    )  # safetensors only: no pickle
    data = hf_hub_download(
        DATASET, DATA_FILE, repo_type="dataset", local_dir=work / "data"
    )
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok("\n\n".join(pd.read_parquet(data).text), return_tensors="np")[
        "input_ids"
    ][0]
    need = (windows + calib_windows) * seq
    if len(ids) < need:
        raise SystemExit(f"WikiText-2 test has {len(ids)} tokens, {need} needed")
    win = ids[: windows * seq].reshape(windows, seq).astype(np.int64)
    calib = (
        ids[windows * seq : need].reshape(calib_windows, seq).astype(np.int64)
    )  # text the evaluation never sees
    return model_dir, win, calib


def build_wrapper(model_dir, seq):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, attn_implementation="eager", dtype=torch.float32
    ).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer(
                "mask",
                torch.triu(torch.full((seq, seq), torch.finfo(torch.float32).min), 1)[
                    None, None
                ],
            )

        def forward(self, input_ids):
            mask = self.mask.expand(input_ids.shape[0], -1, -1, -1)
            return self.m(
                input_ids=input_ids, attention_mask=mask, use_cache=False
            ).logits

    return Wrap(model)


def torch_perplexity(wrapper, win, batch):
    import torch

    with torch.no_grad():
        return float(
            np.exp(
                np.concatenate(
                    [
                        nll(
                            wrapper(torch.from_numpy(win[i : i + batch])).numpy(),
                            win[i : i + batch],
                        )
                        for i in range(0, len(win), batch)
                    ]
                ).mean()
            )
        )


def export_onnx(wrapper, win, batch, path):
    import torch

    torch.onnx.export(
        wrapper,
        (torch.from_numpy(win[:batch]),),
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )


def convert(path, npu):
    import onnx
    from onnx import shape_inference

    import onnxsim

    simplified, ok = onnxsim.simplify(onnx.load(str(path)))
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified model")
    rewritten, stats = npu.rewrite(simplified)
    return (
        shape_inference.infer_shapes(rewritten),
        stats,
        npu.undocumented(rewritten),
        npu.large_dims(rewritten),
    )


def schemes(sets, names):
    s, everything = sets, set(names)
    norm = (
        s["ln_interior"] | s["rms_interior"]
    )  # the rewriter's norm interiors (and un-rewritten Pow patterns)
    fused = norm | s["silu_interior"]
    rest16 = everything - s["matmul_in"]
    return [
        # label, scheme, calibration, keep (exact float), keep_fp16
        ("fp16, every tensor + weight", "fp16", "minmax", set(), set()),
        ("bf16, every tensor + weight", "bf16", "minmax", set(), set()),
        ("int16 dfp, everything (mask unfixed)", "int16", "minmax", set(), set()),
        ("int16 dfp + softmax-input fp16", "int16", "minmax", set(), s["softmax_in"]),
        (
            "int16 dfp + softmax-in fp16 + fused norm/SiLU",
            "int16",
            "minmax",
            fused,
            s["softmax_in"],
        ),
        ("weights only: int8 per-channel", "pcq", "minmax", everything, set()),
        ("weights only: uint8 per-tensor", "uint8", "minmax", everything, set()),
        (
            "a8 all + softmax-in fp16 + fused norm/SiLU, pcq minmax",
            "pcq",
            "minmax",
            fused,
            s["softmax_in"],
        ),
        (
            "a8 all + softmax-in fp16 + fused norm/SiLU, pcq ema",
            "pcq",
            "ema",
            fused,
            s["softmax_in"],
        ),
        (
            "W8A8 matmul inputs only (rest fp16), pcq minmax",
            "pcq",
            "minmax",
            set(),
            rest16,
        ),
        ("W8A8 matmul inputs only (rest fp16), pcq ema", "pcq", "ema", set(), rest16),
        ("W16 dfp matmul inputs only (rest fp16)", "int16", "minmax", set(), rest16),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--work",
        required=True,
        type=Path,
        help="directory for downloads and intermediate files",
    )
    p.add_argument("--seq", type=int, default=128, help="tokens per window")
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="fixed batch of the exported graph (does not change the math)",
    )
    p.add_argument(
        "--windows", type=int, default=64, help="WikiText-2 test windows evaluated"
    )
    p.add_argument(
        "--calib-windows",
        type=int,
        default=16,
        help="windows (after the evaluated ones) used for calibration",
    )
    p.add_argument(
        "--only",
        help="comma-separated substrings; run only schemes whose label contains one",
    )
    p.add_argument("--json", type=Path, help="write the results here")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    npu = load_npu_rewrite()

    model_dir, win, calib = prepare(a.work, a.seq, a.windows, a.calib_windows)
    wrapper = build_wrapper(model_dir, a.seq)
    print(
        f"PyTorch fp32 perplexity ({a.windows} windows x {a.seq} tokens): {torch_perplexity(wrapper, win, a.batch):.3f}"
    )
    onnx_path = a.work / f"smollm_b{a.batch}_s{a.seq}.onnx"
    if not onnx_path.exists():
        export_onnx(wrapper, win, a.batch, onnx_path)
    model, rewrites, left, large = convert(onnx_path, npu)
    print(
        f"converted: rewrites {dict(rewrites)}; operators outside the documented set: {left or 'none'}"
    )
    print(
        "size-limit warnings:",
        sorted({(op, tuple(shape)) for _, op, _, shape in large}) or "none",
    )

    import onnxruntime as ort

    def sess(m):
        return ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )

    def batches(ids):
        return [
            {"input_ids": ids[i : i + a.batch]} for i in range(0, len(ids), a.batch)
        ]

    ref_sess = sess(model)
    vb, cb = batches(win), batches(calib)

    names = qs.float_node_outputs(model)
    sets = qs.hybrid_sets(model)
    print(
        f"{len(names)} float tensors; "
        + ", ".join(f"{k} {len(v)}" for k, v in sets.items())
    )
    ranges = qs.collect_ranges(model, cb, names)
    squares = [t for t in names if "__sq" in t]
    if squares:
        print(
            f"largest squared value after the norm rewrite: {max(max(abs(v) for v in ranges['minmax'][t]) for t in squares):.3g} (fp16 max 65504)"
        )

    def measure(built):
        s = sess(built)
        nl, ref_nl, agree, sq_err, count, lo, hi = [], [], 0, 0.0, 0, np.inf, -np.inf
        for f in vb:
            t, r = s.run(None, f)[0], ref_sess.run(None, f)[0]
            ref_nl.append(nll(r, f["input_ids"]))
            if not np.isfinite(t).all():
                return {
                    "ppl": float("nan"),
                    "top1": float("nan"),
                    "rel_rmse": float("nan"),
                }, float(np.exp(np.concatenate(ref_nl).mean()))
            nl.append(nll(t, f["input_ids"]))
            agree += int((t.argmax(-1) == r.argmax(-1)).sum())
            sq_err += float(((t.astype(np.float64) - r) ** 2).sum())
            count += t.size
            lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
        return {
            "ppl": float(np.exp(np.concatenate(nl).mean())),
            "top1": agree / (len(win) * a.seq),
            "rel_rmse": float(np.sqrt(sq_err / count) / (hi - lo)),
        }, float(np.exp(np.concatenate(ref_nl).mean()))

    base_ppl = float(
        np.exp(
            np.concatenate(
                [nll(ref_sess.run(None, f)[0], f["input_ids"]) for f in vb]
            ).mean()
        )
    )
    print(f"converted model fp32 perplexity: {base_ppl:.3f}")
    results = [
        {
            "label": "fp32 (converted model)",
            "ppl": base_ppl,
            "top1": 1.0,
            "rel_rmse": 0.0,
        }
    ]
    only = a.only.split(",") if a.only else None
    print(
        f"{'scheme':58} {'perplexity':>11} {'top-1 agree':>12} {'logit rel-RMSE':>15}"
    )
    for label, scheme, calib_name, keep, keep16 in schemes(sets, names):
        if only and not any(o in label for o in only):
            continue
        t0 = time.time()
        built = qs.build(
            model,
            ranges[calib_name],
            scheme,
            keep=frozenset(keep),
            keep_fp16=frozenset(keep16),
        )
        row, _ = measure(built)
        row["label"] = label
        results.append(row)
        print(
            f"{label:58} {row['ppl']:11.3f} {row['top1']:12.4f} {row['rel_rmse']:15.4f}   ({time.time() - t0:.0f}s)",
            flush=True,
        )
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
