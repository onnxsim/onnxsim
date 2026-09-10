#!/usr/bin/env python3
"""Generate ONNX Runtime on-device training artifacts from a bare .onnx file.

This is the one step in the fine-tuning workflow that needs Python: building
the gradient graph requires onnxruntime.training.artifacts, which is not
exposed through the C/C++ API. Run this once, offline; the CLI tool
(onnx-finetune) that actually trains needs only the four files this script
produces, and has no Python dependency itself.

Requires a training-enabled onnxruntime build (`--enable_training_apis
--enable_pybind --build_wheel`, or the future public wheel once one exists --
`pip install onnxruntime` alone does not include this). See ../README.md.
"""

import argparse
import json
import os
import sys

import onnx
from onnxruntime.training import artifacts

LOSS_TYPES = {
    "mse": artifacts.LossType.MSELoss,
    "cross-entropy": artifacts.LossType.CrossEntropyLoss,
    "bce": artifacts.LossType.BCEWithLogitsLoss,
}
OPTIM_TYPES = {
    "adamw": artifacts.OptimType.AdamW,
    "sgd": artifacts.OptimType.SGD,
}

# onnx-finetune links a specific from-source onnxruntime build (see
# README.md's "Prerequisite" section); its bundled C++ engine rejects any
# model IR version newer than this. onnxruntime.training.artifacts.
# generate_artifacts() builds optimizer_model.onnx fresh internally using
# whichever onnx package happens to be installed, so its ir_version tracks
# that package's default (13 as of a reasonably current onnx release) --
# independent of training_model.onnx/eval_model.onnx, which inherit the
# ir_version of the model passed in and so are usually already low enough.
# Without this clamp, onnx-finetune fails at load time with "Unsupported
# model IR version" even though generate_artifacts.py itself ran cleanly --
# affects every --loss mode, not just distillation.
MAX_SUPPORTED_IR_VERSION = 10


def _clamp_ir_version(path):
    model = onnx.load(path)
    if model.ir_version > MAX_SUPPORTED_IR_VERSION:
        model.ir_version = MAX_SUPPORTED_IR_VERSION
        onnx.save(model, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="input .onnx file")
    p.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="directory to write training artifacts into",
    )
    p.add_argument(
        "--freeze-prefix",
        action="append",
        default=[],
        help="initializer name prefix to freeze (repeatable). Everything not matching a "
        "given prefix is trainable. Omit to train the whole model. Ignored if "
        "--lora-params-file is given.",
    )
    p.add_argument(
        "--lora-params-file",
        help="JSON adapter manifest from inject_lora.py -- trains exactly the LoRA lora_A/lora_B "
        "params it lists and freezes every other initializer (the real low-rank LoRA recipe, as "
        "opposed to --freeze-prefix's full-parameter subset training). Overrides --freeze-prefix.",
    )
    p.add_argument("--loss", choices=LOSS_TYPES, default="mse")
    p.add_argument("--optimizer", choices=OPTIM_TYPES, default="adamw")
    args = p.parse_args()

    model = onnx.load(args.model)
    all_params = [i.name for i in model.graph.initializer]
    if args.lora_params_file:
        with open(args.lora_params_file) as f:
            manifest = json.load(f)
        trainable = [n for pair in manifest["pairs"] for n in pair]
        frozen = [n for n in all_params if n not in trainable]
    elif args.freeze_prefix:
        frozen = [
            n
            for n in all_params
            if any(n.startswith(pfx) for pfx in args.freeze_prefix)
        ]
        trainable = [n for n in all_params if n not in frozen]
    else:
        frozen = []
        trainable = list(all_params)

    if not trainable:
        sys.exit(
            "error: --freeze-prefix matched every initializer, nothing left to train"
        )

    print(f"trainable ({len(trainable)}):", ", ".join(trainable))
    if frozen:
        print(f"frozen ({len(frozen)}):", ", ".join(frozen))

    os.makedirs(args.output_dir, exist_ok=True)

    artifacts.generate_artifacts(
        model,
        requires_grad=trainable,
        frozen_params=frozen,
        loss=LOSS_TYPES[args.loss],
        optimizer=OPTIM_TYPES[args.optimizer],
        artifact_directory=args.output_dir,
    )
    print("wrote training artifacts ->", args.output_dir)

    for name in ("training_model.onnx", "eval_model.onnx", "optimizer_model.onnx"):
        _clamp_ir_version(os.path.join(args.output_dir, name))


if __name__ == "__main__":
    main()
