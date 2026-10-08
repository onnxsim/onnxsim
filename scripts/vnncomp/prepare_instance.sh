#!/usr/bin/env bash
# VNN-COMP protocol: prepare_instance.sh <version> <benchmark> <onnx> <vnnlib>. Nothing to prepare.
set -euo pipefail
if [ "${1:-}" != "v1" ]; then
    echo "prepare_instance.sh: unsupported protocol version '${1:-}' (expected v1)" >&2
    exit 1
fi
exit 0
