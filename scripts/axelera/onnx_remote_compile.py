#!/usr/bin/env python3
"""onnx-remote-compiler command for Axelera Metis (Voyager SDK).

Compiles a static-shape ONNX model for Metis and packs the Voyager output
directory into a single ONXAXL1 bundle, which the Axelera worker
(tools/onnx-remote/remote_axelera_worker.cpp) unpacks and loads with
axr_load_model().

Wire it into the compiler service (see tools/onnx-remote/README.md):

    onnx-remote-compiler --port 39502 --target metis-m2 --compiler-id voyager-1.9.0 \
      --cache-dir ~/.cache/onnxsim-axelera \
      --command 'python3 scripts/axelera/onnx_remote_compile.py --input {input} \
                 --output {output} --manifest {manifest} --target {target} \
                 --compiler-id {compiler_id} --shapes {shapes}'

Calibration: quantization needs representative inputs. By default this uses
random samples in [-1, 1], which is only adequate for transport and handshake
tests. For real accuracy, pass --calib-dir with .npy files (one sample each,
already in the model's input layout and dtype).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="ONNX ModelProto file")
    p.add_argument("--output", required=True, help="tar artifact to write")
    p.add_argument("--manifest", required=True, help="JSON manifest to write")
    p.add_argument("--target", default="metis-m2")
    p.add_argument("--compiler-id", default="voyager")
    p.add_argument("--shapes", default="", help="JSON input descriptors (unused for static models)")
    p.add_argument("--calib-dir", default="", help="directory of .npy calibration samples")
    p.add_argument("--calib-samples", type=int, default=16, help="random samples when --calib-dir is empty")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_calibration(args: argparse.Namespace, input_shapes: dict[str, tuple], input_dtypes: dict[str, str]):
    """Yield calibration samples as the single-input arrays quantize() expects."""
    if len(input_shapes) != 1:
        raise SystemExit("this compiler command supports single-input models only")
    name, shape = next(iter(input_shapes.items()))
    dtype = np.dtype(input_dtypes[name])
    if args.calib_dir:
        files = sorted(Path(args.calib_dir).glob("*.npy"))
        if not files:
            raise SystemExit(f"no .npy calibration samples in {args.calib_dir}")
        for f in files:
            sample = np.load(f).astype(dtype)
            if tuple(sample.shape) != tuple(shape):
                raise SystemExit(f"{f} has shape {sample.shape}, model expects {shape}")
            yield sample
        return
    rng = np.random.default_rng(args.seed)
    for _ in range(args.calib_samples):
        yield rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def static_io(onnx_path: str):
    import onnx

    model = onnx.load(onnx_path)
    init_names = {i.name for i in model.graph.initializer}
    shapes, dtypes, out_shapes = {}, {}, {}
    onnx_dtype = {1: "float32", 6: "int32", 7: "int64", 2: "uint8", 3: "int8"}
    for vi in model.graph.input:
        if vi.name in init_names:
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if not d.HasField("dim_value"):
                raise SystemExit(f"input {vi.name} has a dynamic dimension; pass a static-shape model")
            dims.append(d.dim_value)
        shapes[vi.name] = tuple(dims)
        dtypes[vi.name] = onnx_dtype[vi.type.tensor_type.elem_type]
    for vi in model.graph.output:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        out_shapes[vi.name] = tuple(dims)
    return model, shapes, dtypes, out_shapes


BUNDLE_MAGIC = b"ONXAXL1\x00"


def write_bundle(src_dir: Path, dst: Path) -> None:
    """Pack every file under src_dir into an ONXAXL1 bundle (see remote_axelera_worker.cpp).

    Layout: magic, u32 entry count, then per entry a u32 name length, the
    relative POSIX name, a u64 size and the bytes. All integers little-endian.
    """
    files = sorted(p for p in src_dir.rglob("*") if p.is_file())
    with open(dst, "wb") as f:
        f.write(BUNDLE_MAGIC)
        f.write(len(files).to_bytes(4, "little"))
        for p in files:
            name = p.relative_to(src_dir).as_posix().encode("utf-8")
            data = p.read_bytes()
            f.write(len(name).to_bytes(4, "little"))
            f.write(name)
            f.write(len(data).to_bytes(8, "little"))
            f.write(data)


def main() -> int:
    args = parse_args()
    from axelera import compiler  # Voyager SDK, from the SDK venv

    _, input_shapes, input_dtypes, output_shapes = static_io(args.input)
    cfg = compiler.CompilerConfig()
    calib = load_calibration(args, input_shapes, input_dtypes)

    work = Path(tempfile.mkdtemp(prefix="axelera-compile-"))
    try:
        quantized = compiler.quantize(model=args.input, calibration_dataset=calib, config=cfg)
        out_dir = work / "compiled"
        compiler.compile(model=quantized, config=cfg, output_dir=out_dir)

        if not (out_dir / "model.json").exists():
            raise SystemExit(f"compile produced no model.json in {out_dir}")
        write_bundle(out_dir, Path(args.output))

        digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
        voyager_version = "unknown"
        sdk_version = Path(sys.prefix).parent / "VERSION.TXT"  # venv_<ver> sits in the SDK checkout
        if sdk_version.exists():
            voyager_version = sdk_version.read_text().strip()
        manifest = {
            "compiler": {"name": "voyager", "version": voyager_version, "id": args.compiler_id},
            "target": {"backend": "axelera", "device": args.target},
            "artifact": {"format": "axelera-bundle", "entry": "model.json", "sha256": digest,
                         "bytes": os.path.getsize(args.output)},
            "io": {
                "inputs": [{"name": n, "shape": list(s), "dtype": input_dtypes[n]} for n, s in input_shapes.items()],
                "outputs": [{"name": n, "shape": list(s)} for n, s in output_shapes.items()],
            },
            "calibration": {"source": args.calib_dir or "random", "samples": args.calib_samples if not args.calib_dir else None},
        }
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"wrote {args.output} ({os.path.getsize(args.output)} bytes)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
