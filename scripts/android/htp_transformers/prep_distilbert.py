#!/usr/bin/env python3
"""Prepare DistilBERT (SST-2) variants and inputs for the Snapdragon HTP runs (run_phone.py).

    python prep_distilbert.py --work /big/disk/dir [--quant]

Writes into <work>/models:
  distilbert_orig.onnx    simplified only: LayerNormalization and Div as torch exports them
  distilbert_naive.onnx   norms decomposed in the textbook form (d*d): overflows fp16 on this model
  distilbert_adaptive.onnx   norms rewritten with per-row max scaling (npu_rewrite.py default): fp16-safe
  distilbert_w8a8.onnx / distilbert_w8a16.onnx   (with --quant) QDQ models from the adaptive graph, built with onnxruntime's QNN tooling
and, in <work>/data, the 872 SST-2 validation sentences as raw int64 arrays plus the runner manifest.
All models are batch 1, 64 tokens. Needs the packages of scripts/allwinner/transformer_accuracy/study_sst2.py; run it from a directory
that is not the repository root.
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


def calibration_reader(calib, n):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.feeds = iter(
                {
                    "input_ids": calib["ids"][i : i + 1],
                    "attention_mask": calib["mask"][i : i + 1],
                }
                for i in range(min(n, len(calib["ids"])))
            )

        def get_next(self):
            return next(self.feeds, None)

    return Reader()


def reciprocal_to_div(model):
    """Mul(a, Reciprocal(b)) -> Div(a, b). The HTP crushes the 16-bit Reciprocal -> Mul pair of the decomposed norm (accuracy 49%);
    the Div form is accurate (90.8%)."""
    producers = {o: n for n in model.graph.node for o in n.output}
    drop = set()
    for n in model.graph.node:
        if n.op_type != "Mul":
            continue
        for k in (0, 1):
            r = producers.get(n.input[k])
            if r is not None and r.op_type == "Reciprocal":
                a, b = n.input[1 - k], r.input[0]
                n.op_type = "Div"
                del n.input[:]
                n.input.extend([a, b])
                drop.add(id(r))
                break
    used = {i for n in model.graph.node if id(n) not in drop for i in n.input}
    keep = [
        n
        for n in model.graph.node
        if id(n) not in drop
        or n.output[0] in used  # a Reciprocal still read elsewhere stays
    ]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def quantize(src, dst, calib, n, activation, exclude):
    """QDQ model for the QNN EP: 8-bit weights, 8- or 16-bit activations, nodes in `exclude` left float."""
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    config = get_qnn_qdq_config(
        str(src),
        calibration_reader(calib, n),
        activation_type=QuantType.QUInt16 if activation == 16 else QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        nodes_to_exclude=sorted(exclude),
    )
    quantize(str(src), str(dst), config)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument(
        "--quant", action="store_true", help="also build the w8a8 / w8a16 QDQ models"
    )
    p.add_argument("--calib", type=int, default=128)
    a = p.parse_args()
    models, data = a.work / "models", a.work / "data"
    models.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    study = load(ALLWINNER / "transformer_accuracy" / "study_sst2.py", "study_sst2")
    npu = study.load_npu_rewrite()
    import onnx
    from onnx import shape_inference

    import onnxsim

    model_dir, files = study.download(a.work)
    val, calib = study.tokenize(model_dir, files, a.seq, max(a.calib, 16), 0)
    torch_model, _ = study.torch_baseline(
        model_dir, {"ids": val["ids"][:16], "mask": val["mask"][:16]}, 16
    )
    raw = models / "distilbert_b1.onnx"
    if not raw.exists():
        study.export_onnx(torch_model, val, 1, raw)
    simplified, ok = onnxsim.simplify(onnx.load(str(raw)))
    assert ok, "onnxsim could not validate the simplified model"
    onnx.save(simplified, models / "distilbert_orig.onnx")
    for name, scaling in (("naive", "none"), ("adaptive", "max")):
        rewritten, stats = npu.rewrite(simplified, scaling)
        onnx.save(
            shape_inference.infer_shapes(rewritten), models / f"distilbert_{name}.onnx"
        )
        print(f"distilbert_{name}.onnx: rewrites {dict(stats)}")

    if a.quant:
        divform = models / "distilbert_adaptive_div.onnx"
        onnx.save(
            reciprocal_to_div(onnx.load(str(models / "distilbert_adaptive.onnx"))),
            divform,
        )
        adaptive = onnx.load(str(divform))
        # the attention mask's -3.4e38 fill value would set the quantization range of the scores, so the masked Add and the Softmax stay float
        softmax = [n for n in adaptive.graph.node if n.op_type == "Softmax"]
        exclude = {n.name for n in softmax} | {
            n.name
            for n in adaptive.graph.node
            if n.output[0] in {s.input[0] for s in softmax}
        }
        for bits in (8, 16):
            quantize(
                divform,
                models / f"distilbert_w8a{bits}.onnx",
                calib,
                a.calib,
                bits,
                exclude,
            )
            print(
                f"distilbert_w8a{bits}.onnx written ({len(exclude)} nodes left float)"
            )

    ids, mask = val["ids"].astype(np.int64), val["mask"].astype(np.int64)
    n = len(ids)
    ids.tofile(data / "val_ids.bin")
    mask.tofile(data / "val_mask.bin")
    np.save(data / "val_labels.npy", val["labels"])
    (data / "val.manifest").write_text(
        f"input_ids i64 val_ids.bin {n},1,{a.seq}\nattention_mask i64 val_mask.bin {n},1,{a.seq}\n"
    )
    print(f"{n} validation sentences written to {data}")


if __name__ == "__main__":
    main()
