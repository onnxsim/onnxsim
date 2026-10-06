#!/usr/bin/env python3
"""Accuracy of a fine-tuned transformer (DistilBERT on SST-2) under the quantization schemes of Allwinner's Acuity toolchain.

    python study_sst2.py --work /big/disk/dir [--calib 128] [--only "W8A8,fp16"] [--json results.json]

Pipeline: download the model (safetensors only) and SST-2 -> tokenize to a fixed length -> PyTorch fp32 baseline -> export to a
fixed-shape ONNX model -> onnxsim + scripts/allwinner/npu_rewrite.py (the operator set the NPU importer documents) -> calibrate ->
evaluate each scheme emulated by quantsim.py -> print a table. Needs torch, transformers, huggingface_hub, pandas, pyarrow, onnx,
onnxsim and onnxruntime; run it from a directory that is not the repository root (the repository's onnxsim/ would shadow the package).

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

MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DATASET = "stanfordnlp/sst2"


def load_npu_rewrite():
    spec = importlib.util.spec_from_file_location(
        "npu_rewrite", HERE.parent / "npu_rewrite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(work):
    from huggingface_hub import hf_hub_download, snapshot_download

    model_dir = work / "model"
    # safetensors only: a pickled .bin checkpoint would execute code when loaded
    snapshot_download(
        MODEL, local_dir=model_dir, allow_patterns=["*.json", "*.txt", "*.safetensors"]
    )
    data = {
        s: hf_hub_download(
            DATASET,
            f"data/{s}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=work / "data",
        )
        for s in ("validation", "train")
    }
    return model_dir, data


def tokenize(model_dir, data, seq, n_calib_pool, seed):
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    val = pd.read_parquet(data["validation"])
    calib = pd.read_parquet(data["train"]).sample(n_calib_pool, random_state=seed)
    longest = max(len(tok(s)["input_ids"]) for s in val.sentence)
    if longest > seq:
        raise SystemExit(
            f"--seq {seq} is shorter than the longest validation sentence ({longest} tokens)"
        )

    def enc(df):
        t = tok(
            list(df.sentence),
            padding="max_length",
            truncation=True,
            max_length=seq,
            return_tensors="np",
        )
        return {
            "ids": t["input_ids"].astype(np.int64),
            "mask": t["attention_mask"].astype(np.int64),
            "labels": df.label.values,
        }

    return enc(val), enc(calib)


def torch_baseline(model_dir, val, batch):
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        logits = np.concatenate(
            [
                model(
                    input_ids=torch.from_numpy(val["ids"][i : i + batch]),
                    attention_mask=torch.from_numpy(val["mask"][i : i + batch]),
                ).logits.numpy()
                for i in range(0, len(val["ids"]), batch)
            ]
        )
    return model, logits


def export_onnx(model, val, batch, path):
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    example = (
        torch.from_numpy(val["ids"][:batch]),
        torch.from_numpy(val["mask"][:batch]),
    )
    torch.onnx.export(
        Wrap(model),
        example,
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )


def make_batches(d, batch, n=None):
    ids, mask = (d["ids"][:n], d["mask"][:n]) if n else (d["ids"], d["mask"])
    out = []
    for i in range(0, len(ids), batch):
        a, b = ids[i : i + batch], mask[i : i + batch]
        if (
            len(a) < batch
        ):  # the last batch repeats rows to the fixed batch size; callers drop the extras
            pad = batch - len(a)
            a, b = np.concatenate([a, a[:pad]]), np.concatenate([b, b[:pad]])
        out.append({"input_ids": a, "attention_mask": b})
    return out


def convert(path, npu):
    import onnx
    from onnx import shape_inference

    import onnxsim

    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified model")
    rewritten, stats = npu.rewrite(simplified)
    return shape_inference.infer_shapes(rewritten), stats, npu.undocumented(rewritten)


def schemes(sets, names):
    s, everything = sets, set(names)
    fused = s["ln_interior"] | s["gelu_interior"]
    rest_fp16 = everything - s["matmul_in"]
    return [
        # label, scheme, calibration, keep (float), keep_fp16
        ("fp16 (all tensors + weights)", "fp16", "minmax", set(), set()),
        ("bf16 (all tensors + weights)", "bf16", "minmax", set(), set()),
        ("int16 dfp, everything (mask unfixed)", "int16", "minmax", set(), set()),
        ("int16 dfp + softmax-input fp16", "int16", "minmax", set(), s["softmax_in"]),
        (
            "int16 dfp + softmax-in fp16 + fused LN+GELU",
            "int16",
            "minmax",
            fused,
            s["softmax_in"],
        ),
        ("weights only: int8 per-channel", "pcq", "minmax", everything, set()),
        ("weights only: uint8 per-tensor", "uint8", "minmax", everything, set()),
        (
            "a8 all + softmax-in fp16 + fused LN+GELU, pcq minmax",
            "pcq",
            "minmax",
            fused,
            s["softmax_in"],
        ),
        (
            "a8 all + softmax-in fp16 + fused LN+GELU, pcq ema",
            "pcq",
            "ema",
            fused,
            s["softmax_in"],
        ),
        (
            "a8 all + softmax-in fp16 + fused LN+GELU, pcq p99.9",
            "pcq",
            "p999",
            fused,
            s["softmax_in"],
        ),
        (
            "W8A8 matmul inputs only (rest fp16), pcq minmax",
            "pcq",
            "minmax",
            set(),
            rest_fp16,
        ),
        (
            "W8A8 matmul inputs only (rest fp16), pcq ema",
            "pcq",
            "ema",
            set(),
            rest_fp16,
        ),
        (
            "W8A8 matmul inputs only (rest fp16), pcq p99.9",
            "pcq",
            "p999",
            set(),
            rest_fp16,
        ),
        (
            "W8A8 matmul inputs only (rest fp16), uint8 p99.9",
            "uint8",
            "p999",
            set(),
            rest_fp16,
        ),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--work",
        required=True,
        type=Path,
        help="directory for downloads and intermediate files",
    )
    p.add_argument(
        "--seq",
        type=int,
        default=64,
        help="fixed token length (the longest SST-2 validation sentence is 55 tokens)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="fixed batch of the exported graph (does not change the math)",
    )
    p.add_argument(
        "--calib",
        type=int,
        default=128,
        help="calibration sentences from the training split",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="which training sentences calibrate"
    )
    p.add_argument(
        "--only",
        help="comma-separated substrings; run only schemes whose label contains one",
    )
    p.add_argument("--json", type=Path, help="write the results here")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    npu = load_npu_rewrite()

    model_dir, data = download(a.work)
    val, calib = tokenize(model_dir, data, a.seq, max(a.calib, 16), a.seed)
    labels, n_val = val["labels"], len(val["labels"])
    torch_model, pt_logits = torch_baseline(model_dir, val, a.batch)
    print(
        f"PyTorch fp32 SST-2 validation accuracy: {(pt_logits.argmax(1) == labels).mean():.4f}"
    )

    onnx_path = a.work / f"distilbert_b{a.batch}_s{a.seq}.onnx"
    if not onnx_path.exists():
        export_onnx(torch_model, val, a.batch, onnx_path)
    model, rewrites, left = convert(onnx_path, npu)
    print(
        f"converted: rewrites {rewrites}; operators outside the documented set: {left or 'none'}"
    )

    vb = make_batches(val, a.batch)
    ref = qs.evaluate(model, vb)[:n_val]
    print(
        f"converted model fp32: accuracy {(ref.argmax(1) == labels).mean():.4f}, agreement with PyTorch {(ref.argmax(1) == pt_logits.argmax(1)).mean():.4f}"
    )

    names = qs.float_node_outputs(model)
    sets = qs.hybrid_sets(model)
    print(
        f"{len(names)} float tensors; softmax inputs {len(sets['softmax_in'])}, LayerNorm interior {len(sets['ln_interior'])}, GELU interior {len(sets['gelu_interior'])}, matmul inputs {len(sets['matmul_in'])}"
    )
    cb = make_batches(calib, a.batch, a.calib)
    ranges = qs.collect_ranges(model, cb, names)
    ranges["p999"] = qs.collect_percentile(model, cb[:8], names)
    widest = max(ranges["minmax"].items(), key=lambda kv: kv[1][1] - kv[1][0])
    print(
        f"calibrated on {len(cb) * a.batch} sentences; widest range {widest[1][1] - widest[1][0]:.3g} at {widest[0]}"
    )

    results = [
        {
            "label": "fp32 (converted model)",
            "acc": float((ref.argmax(1) == labels).mean()),
            "agree": 1.0,
            "rel_rmse": 0.0,
        }
    ]
    only = a.only.split(",") if a.only else None
    print(f"{'scheme':56} {'accuracy':>8} {'agree':>7} {'logit rel-RMSE':>15}")
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
        lg = qs.evaluate(built, vb)[:n_val]
        finite = bool(np.isfinite(lg).all())
        row = {
            "label": label,
            "acc": float((lg.argmax(1) == labels).mean()) if finite else float("nan"),
            "agree": float((lg.argmax(1) == ref.argmax(1)).mean())
            if finite
            else float("nan"),
            "rel_rmse": float(
                np.sqrt(np.mean((lg - ref) ** 2)) / (ref.max() - ref.min())
            )
            if finite
            else float("nan"),
        }
        results.append(row)
        print(
            f"{label:56} {row['acc']:8.4f} {row['agree']:7.4f} {row['rel_rmse']:15.4f}   ({time.time() - t0:.0f}s)",
            flush=True,
        )
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
