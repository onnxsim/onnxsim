#!/usr/bin/env bash
# VNN-COMP protocol: run_instance.sh <version> <benchmark> <onnx> <vnnlib> <results> <timeout>.
# Writes unsat, sat (then a counterexample block), unknown, error, or timeout to <results>.
set -euo pipefail
if [ "${1:-}" != "v1" ]; then
    echo "run_instance.sh: unsupported protocol version '${1:-}' (expected v1)" >&2
    exit 1
fi
exec python3 -m onnxsim.vnnlib run "$3" "$4" "$5" "$6"
