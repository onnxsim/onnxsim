#!/usr/bin/env bash
# VNN-COMP protocol: install_tool.sh <version>. Installs onnxsim (builds its C++ extension,
# ONNX Runtime is not built: setup.py passes ONNXSIM_BUILTIN_ORT=OFF) and the runtime deps
# of onnxsim.vnnlib: onnxruntime for replay, torch for the PGD attack and alpha / beta CROWN.
set -euo pipefail
if [ "${1:-}" != "v1" ]; then
    echo "install_tool.sh: unsupported protocol version '${1:-}' (expected v1)" >&2
    exit 1
fi
repo="$(cd "$(dirname "$0")/../.." && pwd)"
python3 -m pip install "$repo" onnxruntime torch
