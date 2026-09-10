#!/usr/bin/env python3
"""Write back trained weights (from ../src/main.cpp's ``--distillation-step-graph``
mode) into a fresh, inference-ready copy of the original student model.

The native CLI's plain (non-training) ``Ort::Session`` path has no
``TrainingSession::ExportModelForInferencing`` equivalent -- that call is
part of ``onnxruntime.training``'s C++ API, which this mode deliberately
does not link (see wasm/README.md and ../README.md's "Knowledge
distillation (graph_grad)" section for why). Reassembling the final
initializer values into a normal ``.onnx`` file is exactly the kind of ONNX
graph surgery this repo already does in Python everywhere else, so it stays
here rather than pulling an onnx protobuf dependency into the C++ CLI just
for this one step.

``weights_bin`` must hold each weight named by a ``weight`` line in
``manifest`` (see generate_distillation_step_graph.py's
``write_manifest_and_initial_state``), concatenated float32, row-major, in
that same order -- exactly the format ``main.cpp --output-weights`` writes.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnx.numpy_helper


def _read_manifest_weights(manifest_path: str):
    weights = []
    with open(manifest_path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "weight":
                name = parts[1]
                shape = [int(d) for d in parts[2:]]
                weights.append((name, shape))
    return weights


def apply_trained_weights(student_path: str, manifest_path: str, weights_bin_path: str, output_path: str) -> None:
    weights = _read_manifest_weights(manifest_path)
    blob = np.fromfile(weights_bin_path, dtype=np.float32)

    model = onnx.load(student_path)
    by_name = {init.name: init for init in model.graph.initializer}

    offset = 0
    for name, shape in weights:
        count = int(np.prod(shape)) if shape else 1
        values = blob[offset : offset + count].reshape(shape)
        offset += count
        if name not in by_name:
            raise ValueError(f"manifest names weight {name!r}, not an initializer of {student_path}")
        new_tensor = onnx.numpy_helper.from_array(values.astype(np.float32), name)
        by_name[name].CopyFrom(new_tensor)

    if offset != blob.size:
        raise ValueError(
            f"{weights_bin_path} has {blob.size} float32s, but the manifest's weights "
            f"account for only {offset} of them"
        )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("student", help="the original (untrained) student .onnx model")
    p.add_argument("manifest", help="the step graph's .manifest.txt")
    p.add_argument("weights_bin", help="final weights, written by main.cpp --output-weights")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    apply_trained_weights(args.student, args.manifest, args.weights_bin, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
