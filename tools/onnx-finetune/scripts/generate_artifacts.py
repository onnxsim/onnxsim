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

import distillation_loss

LOSS_TYPES = {
    "mse": artifacts.LossType.MSELoss,
    "cross-entropy": artifacts.LossType.CrossEntropyLoss,
    "bce": artifacts.LossType.BCEWithLogitsLoss,
    # "distillation" isn't a plain LossType enum value (it needs a second,
    # frozen model's logits as an extra loss input) -- handled separately in
    # main() via distillation_loss.build_distillation_loss_class(), but
    # listed here too so it shows up in --loss's choices/help.
    "distillation": None,
}
OPTIM_TYPES = {
    "adamw": artifacts.OptimType.AdamW,
    "sgd": artifacts.OptimType.SGD,
}


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
    p.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="--loss distillation only: softmax temperature for the soft-target term (Hinton et al.)",
    )
    p.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help="--loss distillation only: weight on the soft-target loss vs. the hard-label loss "
        "(1.0 = pure distillation, 0.0 = plain supervised training on --labels alone)",
    )
    args = p.parse_args()

    if args.loss == "distillation":
        if args.distill_temperature <= 0:
            sys.exit("error: --distill-temperature must be > 0")
    elif args.distill_temperature != 2.0 or args.distill_alpha != 0.5:
        sys.exit("error: --distill-temperature/--distill-alpha only apply with --loss distillation")

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

    if args.loss == "distillation":
        DistillationLoss = distillation_loss.build_distillation_loss_class()
        artifacts.generate_artifacts(
            model,
            requires_grad=trainable,
            frozen_params=frozen,
            loss=DistillationLoss(temperature=args.distill_temperature, alpha=args.distill_alpha),
            optimizer=OPTIM_TYPES[args.optimizer],
            artifact_directory=args.output_dir,
            loss_input_names=[model.graph.output[0].name, "teacher_logits", "labels"],
            # Expose the soft/hard sub-losses as extra training outputs too,
            # for callers (onnx-finetune's --teacher-model mode) that want
            # to log the breakdown rather than only the combined total.
            additional_output_names=[
                DistillationLoss.SOFT_LOSS_OUTPUT_NAME,
                DistillationLoss.HARD_LOSS_OUTPUT_NAME,
            ],
        )
        print(
            f"wrote distillation training artifacts -> {args.output_dir} "
            f"(temperature={args.distill_temperature}, alpha={args.distill_alpha}; "
            f"training graph expects 3 external inputs at runtime: the model's own input, "
            f"'teacher_logits', 'labels' -- see ../README.md's distillation section)"
        )
    else:
        artifacts.generate_artifacts(
            model,
            requires_grad=trainable,
            frozen_params=frozen,
            loss=LOSS_TYPES[args.loss],
            optimizer=OPTIM_TYPES[args.optimizer],
            artifact_directory=args.output_dir,
        )
        print("wrote training artifacts ->", args.output_dir)


if __name__ == "__main__":
    main()
