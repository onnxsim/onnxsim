#!/usr/bin/env python3
"""Compare Core ML and tinygrad Metal on a SAM encoder/decoder pair.

The image encoder and prompt decoder are already separate ONNX graphs in the
SAM vision-model workflow. This runner measures each backend for each stage,
then measures the practical hybrid path (Core ML encoder -> Metal decoder)
with the tensor transfer and both predictions included.

Example:
  python scripts/apple/benchmark_sam_hybrid.py ~/.cache/onnxsim-sam/edgesam \
      --output /tmp/edgesam-m4.json --repeats 12
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def _inputs(model: onnx.ModelProto, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    initializers = {x.name for x in model.graph.initializer}
    result = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if any(not n for n in shape):
            raise ValueError(f"{value.name} has a dynamic shape: {shape}")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
        if dtype == np.uint8:
            data = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 3, size=shape).astype(dtype)
        else:
            data = rng.standard_normal(shape).astype(dtype)
        result[value.name] = np.ascontiguousarray(data)
    return result


class _MetalRunner:
    def __init__(self, path: Path, jit: bool = False):
        from tinygrad import Tensor, TinyJit
        from tinygrad.nn.onnx import OnnxRunner

        self.Tensor = Tensor
        self.model = onnx.load(path)
        self.runner = OnnxRunner(str(path)).to("METAL")
        self.output_names = [x.name for x in self.model.graph.output]
        self.input_names = [
            x.name for x in self.model.graph.input
            if x.name not in {t.name for t in self.model.graph.initializer}
        ]
        self.jit = None
        if jit:
            def run(*values):
                outputs = self.runner({
                    name: value for name, value in zip(self.input_names, values)
                })
                return tuple(outputs[name] for name in self.output_names)

            self.jit = TinyJit(run)

    def __call__(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensors = {k: self.Tensor(v, device="METAL") for k, v in feeds.items()}
        if self.jit is not None:
            outputs = self.jit(*(tensors[name] for name in self.input_names))
        else:
            outputs = tuple(self.runner(tensors)[name] for name in self.output_names)
        return [value.numpy() for value in outputs]


def _coreml_runner(path: Path, compute_units: str, compute_precision: str,
                   package: Path):
    import coremltools as ct

    from onnxsim import export_coreml

    model = onnx.load(path)
    graph = model.graph
    # The SAM QNN export declares a uint8 image input but immediately applies
    # DequantizeLinear(scale=1, zero=0). Core ML's tensor interface has no
    # uint8 image input, so expose that exact same raw-pixel tensor as float32
    # and remove the identity dequantization at the boundary.
    feed_cast: set[str] = set()
    for value in graph.input:
        if value.type.tensor_type.elem_type != onnx.TensorProto.UINT8:
            continue
        dq = next((n for n in graph.node if n.op_type == "DequantizeLinear"
                   and n.input and n.input[0] == value.name), None)
        if dq is None:
            raise ValueError(f"uint8 input {value.name} has no boundary DequantizeLinear")
        input_name, output_name = value.name, dq.output[0]
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        for node in graph.node:
            if node is not dq:
                for i, name in enumerate(node.input):
                    if name == output_name:
                        node.input[i] = input_name
        graph.node.remove(dq)
        feed_cast.add(input_name)

    options = {
        "output_path": str(package),
        "compute_units": compute_units,
        "skip_model_load": False,
    }
    if compute_precision != "DEFAULT":
        options["compute_precision"] = getattr(ct.precision, compute_precision)
    mlmodel = export_coreml(model, **options)
    output_names = [v.name for v in model.graph.output]
    coreml_output_names = [v.name for v in mlmodel.get_spec().description.output]
    if len(coreml_output_names) != len(output_names):
        raise RuntimeError(
            "Core ML output count differs from ONNX: "
            f"{len(coreml_output_names)} vs {len(output_names)}"
        )

    def predict(feeds):
        coreml_feeds = {
            name: (value.astype(np.float32) if name in feed_cast else value)
            for name, value in feeds.items()
        }
        result = mlmodel.predict(coreml_feeds)
        # coremltools may rename outputs that are not valid feature names
        # (for example numeric COCO tensor identifiers such as "391").
        return [np.asarray(result[name]) for name in coreml_output_names]

    return predict


def _measure(fn, feeds, warmup: int, repeats: int):
    for _ in range(warmup):
        fn(feeds)
    times = []
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(feeds)
        times.append((time.perf_counter() - start) * 1000)
    return out, {
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "runs": repeats,
    }


def _quality(a: list[np.ndarray], b: list[np.ndarray], names: list[str]) -> dict:
    result = {}
    for name, x, y in zip(names, a, b):
        x, y = x.astype(np.float32).ravel(), y.astype(np.float32).ravel()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        result[name] = {
            "max_abs": float(np.max(np.abs(x - y))),
            "cosine": float(np.dot(x, y) / denom) if denom else 1.0,
        }
        if name == "low_res_masks":
            xa, ya = x > 0, y > 0
            union = int(np.count_nonzero(xa | ya))
            result[name]["threshold_iou"] = (
                float(np.count_nonzero(xa & ya) / union) if union else 1.0
            )
    return result


def _image_feed(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    width2, height2 = round(width * scale), round(height * scale)
    resized = np.asarray(image.resize((width2, height2), Image.Resampling.BILINEAR))
    padded = np.empty((1, size, size, 3), dtype=np.uint8)
    padded[:] = np.array([124, 116, 104], dtype=np.uint8)
    padded[0, :height2, :width2] = resized
    return padded


def _ort_runner(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda feeds: session.run(None, feeds)


def _stage(path: Path, compute_units: str, compute_precision: str, package: Path, seed: int,
           warmup: int, repeats: int, feeds_override=None) -> dict:
    model = onnx.load(path)
    feeds = feeds_override if feeds_override is not None else _inputs(model, seed)
    ref = _ort_runner(path)(feeds)
    output_names = [x.name for x in model.graph.output]
    record: dict = {
        "input_shapes": {k: list(v.shape) for k, v in feeds.items()},
        "benchmark_feeds": feeds,
    }
    for name, factory in (
        ("coreml", lambda: _coreml_runner(path, compute_units, compute_precision, package)),
        ("tinygrad_metal", lambda: _MetalRunner(path)),
        ("tinygrad_metal_jit", lambda: _MetalRunner(path, jit=True)),
    ):
        try:
            runner = factory()
            out, timing = _measure(runner, feeds, warmup, repeats)
            record[name] = {**timing, "vs_ort": _quality(out, ref, output_names)}
            record[name + "_runner"] = runner
        except Exception as exc:
            record[name] = {"error": f"{type(exc).__name__}: {exc}"}
    record["reference_outputs"] = ref
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="SAM export directory with enc.fp16.onnx and dec.sim.onnx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-units", choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE"], default="CPU_AND_NE")
    parser.add_argument("--compute-precision", choices=["DEFAULT", "FLOAT16", "FLOAT32"], default="DEFAULT")
    parser.add_argument("--image", type=Path, help="real image for accuracy checks; otherwise seeded synthetic pixels")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    enc_path, dec_path = args.model_dir / "enc.fp16.onnx", args.model_dir / "dec.sim.onnx"
    if not enc_path.is_file() or not dec_path.is_file():
        raise SystemExit(f"missing {enc_path} or {dec_path}; export the model with sam.py first")
    package_dir = args.output.with_suffix("").with_name(args.output.stem + "_coreml")
    package_dir.mkdir(parents=True, exist_ok=True)

    enc_model = onnx.load(enc_path)
    enc_feeds = _inputs(enc_model, 101)
    if args.image:
        size = enc_feeds["pixels_u8"].shape[1]
        enc_feeds["pixels_u8"] = _image_feed(args.image, size)
    enc = _stage(enc_path, args.compute_units, args.compute_precision, package_dir / "encoder.mlpackage",
                 101, args.warmup, args.repeats, enc_feeds)
    dec_feeds = _inputs(onnx.load(dec_path), 202)
    dec_feeds["image_embeddings"] = enc["reference_outputs"][0]
    if "point_coords" in dec_feeds:
        dec_feeds["point_coords"][:] = np.array([[[512, 512], [0, 0]]], dtype=np.float32)
    if "point_labels" in dec_feeds:
        dec_feeds["point_labels"][:] = np.array([[1, -1]], dtype=dec_feeds["point_labels"].dtype)
    dec = _stage(dec_path, args.compute_units, args.compute_precision, package_dir / "decoder.mlpackage",
                 202, args.warmup, args.repeats, dec_feeds)
    report = {
        "hardware": "Apple Silicon",
        "compute_units": args.compute_units,
        "compute_precision": args.compute_precision,
        "encoder": {k: v for k, v in enc.items() if k not in ("coreml_runner", "tinygrad_metal_runner", "tinygrad_metal_jit_runner", "reference_outputs", "benchmark_feeds")},
        "decoder": {k: v for k, v in dec.items() if k not in ("coreml_runner", "tinygrad_metal_runner", "tinygrad_metal_jit_runner", "reference_outputs", "benchmark_feeds")},
        "pipelines": {},
    }

    # Fixed prompt tensors; the hybrid's encoder output is passed through the
    # Core ML prediction API and into tinygrad's Metal tensors once per image.
    dec_feeds = dict(dec["benchmark_feeds"])
    for enc_name in ("coreml", "tinygrad_metal", "tinygrad_metal_jit"):
        for dec_name in ("coreml", "tinygrad_metal", "tinygrad_metal_jit"):
            er, dr = enc.get(enc_name + "_runner"), dec.get(dec_name + "_runner")
            key = f"{enc_name}_to_{dec_name}"
            if er is None or dr is None:
                report["pipelines"][key] = {"status": "unavailable"}
                continue

            def pipeline(_):
                embedding = er(enc["benchmark_feeds"])[0]
                feeds = dict(dec_feeds)
                feeds["image_embeddings"] = embedding.astype(
                    feeds["image_embeddings"].dtype, copy=False
                )
                return dr(feeds)

            # Compare the selected backends' combined output to the same ONNX
            # CPU path, so a fast but numerically broken hybrid is visible.
            ref_feeds = dict(dec_feeds)
            ref_feeds["image_embeddings"] = enc["reference_outputs"][0]
            ref_out = _ort_runner(dec_path)(ref_feeds)
            try:
                out, timing = _measure(pipeline, {}, args.warmup, args.repeats)
                report["pipelines"][key] = {
                    "status": "ok",
                    **timing,
                    "vs_ort": _quality(
                        out, ref_out, [x.name for x in onnx.load(dec_path).graph.output]
                    ),
                }
            except Exception as exc:
                report["pipelines"][key] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
