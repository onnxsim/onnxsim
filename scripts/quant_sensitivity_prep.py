#!/usr/bin/env python3
"""Data and model preparation for scripts/quant_sensitivity_bench.py (not run in CI).

Everything is written under ``--workdir`` (default: /mnt/data/cache/claude-work/qs-work); nothing is
ever written into the repository. Requires network access, ``torch``, ``torchvision``,
``huggingface_hub``, ``pyarrow``, ``pillow``, and for the text task ``transformers`` + ``datasets``.

    python scripts/quant_sensitivity_prep.py imagenet            # real ImageNet-1k validation subset
    python scripts/quant_sensitivity_prep.py cnn resnet18        # pretrained weights -> ONNX
    python scripts/quant_sensitivity_prep.py cnn mobilenet_v2
    python scripts/quant_sensitivity_prep.py bert                # DistilBERT/SST-2 + real SST-2 validation

``imagenet``: the validation set mirror ``Tsomaros/Imagenet-1k_validation`` (ungated; the official
ILSVRC/imagenet-1k repository is gated). The mirror is sorted by class, so a prefix would cover a
handful of classes; this keeps every 10th image (class stratified, 5 per class, 5000 in all),
resizes the short side to 256, center-crops 224 and stores uint8 NHWC. The float models'
accuracies on the resulting split (68.6 % ResNet18, 71.7 % MobileNetV2 on the 3000 evaluation
images) match the published ImageNet top-1 (69.8 % / 71.9 %), which is the check that the label
indexing and the preprocessing are right.
"""

import argparse
import io
import os
import time

import numpy as np


def prep_imagenet(workdir: str, stride: int = 10) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo = "Tsomaros/Imagenet-1k_validation"
    imgs, labels, g, t0 = [], [], 0, time.time()
    for shard in range(15):
        p = hf_hub_download(
            repo, f"data/validation-{shard:05d}-of-00015.parquet", repo_type="dataset"
        )
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            tab = pf.read_row_group(rg)
            im_col, lb_col = tab.column("image"), tab.column("label").to_pylist()
            for i in range(tab.num_rows):
                if g % stride == 0:
                    im = Image.open(io.BytesIO(im_col[i].as_py()["bytes"])).convert(
                        "RGB"
                    )
                    w, h = im.size
                    sc = 256 / min(w, h)
                    im = im.resize(
                        (max(256, round(w * sc)), max(256, round(h * sc))),
                        Image.BILINEAR,
                    )
                    w, h = im.size
                    left, top = (w - 224) // 2, (h - 224) // 2
                    imgs.append(
                        np.asarray(
                            im.crop((left, top, left + 224, top + 224)), dtype=np.uint8
                        )
                    )
                    labels.append(lb_col[i])
                g += 1
        print(
            f"shard {shard}: rows seen {g}, kept {len(imgs)}, {time.time() - t0:.0f}s",
            flush=True,
        )
    np.save(os.path.join(workdir, "imagenet_val5k_x.npy"), np.stack(imgs))
    np.save(
        os.path.join(workdir, "imagenet_val5k_y.npy"),
        np.asarray(labels, dtype=np.int64),
    )
    print("saved", len(imgs), "images; distinct labels", len(set(labels)))


def prep_cnn(workdir: str, name: str) -> None:
    import torch
    import torchvision

    ctor = {
        "resnet18": (
            torchvision.models.resnet18,
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        ),
        "mobilenet_v2": (
            torchvision.models.mobilenet_v2,
            torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
        ),
        "resnet50": (
            torchvision.models.resnet50,
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        ),
    }[name]
    model = ctor[0](weights=ctor[1]).eval()
    out = os.path.join(workdir, f"{name}.onnx")
    # the legacy exporter folds BatchNorm into Conv, so the sites are Conv/Gemm with plain weights
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), out, input_names=["x"], output_names=["logits"], opset_version=13,
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False,
    )  # fmt: skip
    print("exported", out)


def prep_bert(workdir: str, length: int = 64) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, attn_implementation="eager"
    ).eval()
    ds = load_dataset(
        "stanfordnlp/sst2", split="validation"
    )  # the real, labelled SST-2 validation set
    sentences, y = list(ds["sentence"]), np.asarray(list(ds["label"]), np.int64)
    enc = tok(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="np",
    )
    ids, mask = (
        enc["input_ids"].astype(np.int64),
        enc["attention_mask"].astype(np.int64),
    )
    truncated = sum(len(tok(s)["input_ids"]) > length for s in sentences)
    np.savez(
        os.path.join(workdir, "sst2_val_tok.npz"),
        input_ids=ids,
        attention_mask=mask,
        label=y,
    )
    with torch.no_grad():
        logits = model(
            input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)
        ).logits
    print(f"SST-2 validation: {len(y)} sentences, {truncated} truncated at {length} tokens; "
          f"torch float accuracy {float((logits.argmax(-1).numpy() == y).mean()):.4f}")  # fmt: skip

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    # batch 1 and a fixed length: the exported reshapes are static (the estimators then run
    # one sample at a time, which is what the per-sample Fisher estimate wants anyway)
    torch.onnx.export(
        Wrap(model), (torch.from_numpy(ids[:1]), torch.from_numpy(mask[:1])),
        os.path.join(workdir, "distilbert_sst2.onnx"), input_names=["input_ids", "attention_mask"],
        output_names=["logits"], opset_version=17, dynamo=False,
    )  # fmt: skip
    print("exported distilbert_sst2.onnx")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["imagenet", "cnn", "bert"])
    ap.add_argument("model", nargs="?", default="resnet18")
    ap.add_argument("--workdir", default="/mnt/data/cache/claude-work/qs-work")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    {"imagenet": lambda: prep_imagenet(args.workdir), "cnn": lambda: prep_cnn(args.workdir, args.model),
     "bert": lambda: prep_bert(args.workdir)}[args.what]()  # fmt: skip


if __name__ == "__main__":
    main()
