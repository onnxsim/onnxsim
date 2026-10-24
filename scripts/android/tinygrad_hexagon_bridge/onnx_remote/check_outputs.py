#!/usr/bin/env python3
"""Compare a runner's dumped outputs (onnx-remote-client --compile-run ... --dump out.bin) with onnxruntime on the same inputs.

The v68+ compiler path (compile_v69.sh) has no compile-time check (qemu can't run qfloat / HMX code), so this is its correctness
test. The dump is the ONNX outputs back to back in graph order, each in its program.txt dtype: a single FLOAT or UINT8 output
as it is, several outputs each as FLOAT.

usage: check_outputs.py MODEL.onnx OUT.bin --input-raw DTYPE:D0,D1,...:FILE [--input-raw ...] [--rtol 1e-4] [--ref-model REF.onnx]
  (the same --input-raw arguments as onnx-remote-client, in ONNX graph input order; DTYPE is the ONNX dtype number)
  --ref-model: compute the reference with onnxruntime on this model instead (same inputs and outputs), for a MODEL it can't run,
  e.g. one with onnxsim::MSDA nodes (bevformer_tiny/msda_hvx/export_msda_op.py) checked against the plain export
"""

import argparse
import sys

import numpy as np
import onnx
import onnxruntime as ort

NP = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("--input-raw", action="append", default=[])
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="float outputs: max |diff| / max |ref| allowed",
    )
    ap.add_argument(
        "--ref-model",
        help="run onnxruntime on this model for the reference (same graph inputs and outputs)",
    )
    a = ap.parse_args()
    model = onnx.load(a.model, load_external_data=False)
    init = {i.name for i in model.graph.initializer}
    names = [vi.name for vi in model.graph.input if vi.name not in init]
    if len(a.input_raw) != len(names):
        sys.exit(f"{len(names)} graph inputs, {len(a.input_raw)} --input-raw given")
    feeds = {}
    for name, spec in zip(names, a.input_raw):
        dt, dims, path = spec.split(":", 2)
        feeds[name] = np.fromfile(path, NP[int(dt)]).reshape(
            [int(d) for d in dims.split(",")]
        )
    refs = ort.InferenceSession(
        a.ref_model or a.model, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    raw = open(a.dump, "rb").read()
    single = len(refs) == 1 and refs[0].dtype in (np.float32, np.uint8)
    at, ok = 0, True
    for vo, ref in zip(model.graph.output, refs):
        dt = ref.dtype if single else np.float32
        got = np.frombuffer(raw, dt, ref.size, at).reshape(ref.shape)
        at += got.nbytes
        r = ref.astype(dt)
        if np.issubdtype(dt, np.integer):
            d = np.abs(got.astype(np.int64) - r.astype(np.int64))
            bad = int((d > 0).sum())
            print(f"{vo.name}: {bad} of {r.size} differ (max {int(d.max())})")
            ok &= bad == 0
        else:
            d = float(np.abs(got - r).max())
            scale = float(np.abs(r).max()) or 1.0
            cos = float(
                got.ravel()
                @ r.ravel()
                / (np.linalg.norm(got) * np.linalg.norm(r) or 1.0)
            )
            print(f"{vo.name}: max abs {d:.3g} (rel {d / scale:.3g}), cos {cos:.9f}")
            ok &= d / scale <= a.rtol
    if at != len(raw):
        print(f"dump has {len(raw)} bytes, the outputs {at}")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
