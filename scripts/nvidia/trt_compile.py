#!/usr/bin/env python3
"""Compile an ONNX model into a TensorRT serialized engine.

This is an adapter for ``onnx-remote-compiler``.  TensorRT stays an optional
dependency on the compile host; ``--help`` and the module's argument handling
remain usable on CI machines without CUDA or TensorRT installed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def compile_tensorrt(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    target: str,
    fp16: bool,
    int8: bool,
    workspace_mb: int,
) -> None:
    try:
        # Import lazily: the compiler service can validate its command template
        # and minimal CI can run --help without a GPU Python environment.
        from trt_harness import build_engine
    except Exception as exc:
        raise RuntimeError(
            "TensorRT, ONNX and NumPy are required for TensorRT compilation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    blob, info = build_engine(
        input_path, fp16=fp16, int8=int8, workspace_mb=workspace_mb
    )
    if blob is None:
        raise RuntimeError(info.get("error", "TensorRT engine build failed"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
    manifest = {
        "schema_version": 1,
        "compiler": {
            "name": "nvidia-tensorrt",
            "version": _trt_version(),
            "id": "nvidia-tensorrt",
        },
        "target": {"backend": "tensorrt", "device": target},
        "artifact": {"format": "tensorrt-engine", "abi": "tensorrt-serialized"},
        "io": {"dtype": "model-defined", "dynamic_shapes": "model-defined"},
        "capabilities": {"ops": [], "dtypes": []},
        "legalization": {"profile": target, "version": 1},
        "compile_host": {"system": platform.platform()},
        "profiling": {
            "verbosity": "detailed",
            "build_s": info.get("build_s"),
            "n_layers": info.get("n_layers"),
            "n_int8_layers": info.get("n_int8_layers"),
            "engine_bytes": info.get("engine_bytes"),
            "layers": info.get("layers", []),
        },
        "options": {
            "fp16": fp16,
            "int8": int8,
            "int8_effective": info.get("int8_effective", False),
            "workspace_mb": workspace_mb,
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _trt_version() -> str:
    try:
        import tensorrt as trt

        return getattr(trt, "__version__", "unknown")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", default="tensorrt-cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input model does not exist: {args.input}")
    if args.workspace_mb <= 0:
        parser.error("--workspace-mb must be positive")
    try:
        compile_tensorrt(
            args.input,
            args.output,
            args.manifest,
            args.target,
            args.fp16,
            args.int8,
            args.workspace_mb,
        )
    except Exception as exc:
        print(f"TensorRT compilation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
