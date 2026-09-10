#!/usr/bin/env python3
"""Write synthetic (input, labels) data for the toy classifier
(make_toy_classifier.py) / distillation demo.

labels = argmax(input @ W) for a fixed random W -- a linearly-separable-ish
rule the toy MLP can actually learn, analogous to make_synthetic_data.py's
``target = sum(input)`` for the regression toy. Unlike that script's target
file, labels.bin is raw int64 (one class index per sample), matching what
onnx-finetune's distillation mode (--loss cross-entropy family) and
onnxruntime's SoftmaxCrossEntropyLoss both expect -- not float32.
"""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input-dim", type=int, default=8)
p.add_argument("--num-classes", type=int, default=4)
p.add_argument("--num-samples", type=int, default=2048)
p.add_argument("--input-out", default="train_input.bin")
p.add_argument("--labels-out", default="train_labels.bin")
p.add_argument("--seed", type=int, default=1)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
x = rng.standard_normal((args.num_samples, args.input_dim)).astype(np.float32)
w = rng.standard_normal((args.input_dim, args.num_classes)).astype(np.float32)
labels = np.argmax(x @ w, axis=1).astype(np.int64)

x.tofile(args.input_out)
labels.tofile(args.labels_out)
print(f"wrote {args.num_samples} samples -> {args.input_out}, {args.labels_out}")
