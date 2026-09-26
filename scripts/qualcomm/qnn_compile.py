#!/usr/bin/env python3
"""Compile an ONNX model into a QNN ONNX Runtime EP-context artifact.

This is the SDK-specific command adapter for ``onnx-remote-compiler``. The
service itself remains dependency-free; invoke this script from its trusted
command template when ``onnxruntime-qnn`` is installed on the compile host.
The generated artifact is an EP-context ONNX model. With embed mode enabled,
the QNN context is carried inside that file, which makes it a single artifact
for the remote transport.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys


def compile_qnn(input_path: Path, output_path: Path, manifest_path: Path,
                target: str, strict: bool) -> None:
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime-qnn is required for QNN compilation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    backend_path = os.environ.get("QNN_BACKEND_PATH") or qnn.get_qnn_htp_path()
    if not os.path.isfile(backend_path):
        raise RuntimeError(f"QNN backend library does not exist: {backend_path}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.add_session_config_entry("ep.context_enable", "1")
    session_options.add_session_config_entry("ep.context_embed_mode", "1")
    session_options.add_session_config_entry("ep.context_file_path", str(output_path))
    if strict:
        session_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", "1"
        )

    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    provider_options = {
        "backend_path": backend_path,
        "htp_graph_finalization_optimization_mode": os.environ.get(
            "QNN_HTP_OPT_MODE", "3"
        ),
    }
    ort.InferenceSession(
        str(input_path),
        sess_options=session_options,
        providers=["QNNExecutionProvider"],
        provider_options=[provider_options],
    )
    if not output_path.is_file():
        raise RuntimeError(
            "QNN EP created a session but did not produce the context artifact: "
            f"{output_path}"
        )

    manifest = {
        "schema_version": 1,
        "compiler": {
            "name": "onnxruntime-qnn",
            "version": getattr(ort, "__version__", "unknown"),
            "id": os.environ.get("QNN_COMPILER_ID", "onnxruntime-qnn"),
        },
        "target": {"backend": "qnn", "device": target},
        "artifact": {"format": "onnxruntime-ep-context", "abi": "qnn"},
        "io": {"dtype": "model-defined", "context_embed_mode": 1},
        "capabilities": {"ops": [], "dtypes": []},
        "legalization": {"profile": target, "version": 1},
        "compile_host": {"system": platform.platform()},
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", default="qnn-htp")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input model does not exist: {args.input}")
    try:
        compile_qnn(args.input, args.output, args.manifest, args.target, args.strict)
    except Exception as exc:
        print(f"qnn compilation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
