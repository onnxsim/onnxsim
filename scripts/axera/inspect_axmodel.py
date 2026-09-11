#!/usr/bin/env python3
"""Inspect a real Axera `.axmodel` file for the CPU/NPU boundary markers.

Implements steps 2-4 of the "Immediate next steps" described in the Axera
compatibility notes in ``scripts/axera/README.md``. As of writing **no real
`.axmodel` has been inspected**, so this script exists to actually do that the
moment one is available, rather than continuing to guess.

It:

1. Loads the file with plain ``onnx.load()`` -- confirms it parses as a
   standard ``ModelProto`` at all (the notes' §2: the container is claimed to
   be plain ONNX protobuf, e.g. `onnx inspect -m -n -t` works on it directly
   -- this is the first real test of that claim).
2. Reports any node with a non-standard `domain` (ONNX's documented mechanism
   for vendor extensions -- the likely home of an embedded NPU blob, notes §4).
3. Reports node op_types outside the standard ONNX operator set for the
   model's declared opset.
4. Reports attributes with suspiciously large raw payload (tensor bytes or a
   raw `bytes`/`strings` attribute) -- a likely embedded NPU command stream.

None of this is guaranteed to find the actual boundary marker -- the exact
schema is unknown per the notes' open questions -- but it's the concrete,
falsifiable first step the notes call for, and its output is exactly what's
needed to fill in `pulsar2_ops.py`'s heuristics with real data.

Usage:
    python inspect_axmodel.py path/to/compiled.axmodel
"""

from __future__ import annotations

import argparse
import os
import sys

import onnx
from onnx import defs

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pulsar2_ops import STANDARD_DOMAINS  # noqa: E402

# Attribute payloads larger than this are called out as a likely embedded
# NPU blob rather than an ordinary weight/constant. Chosen generously above
# typical small conv/gemm bias tensors; tune once a real file shows what
# "ordinary large weight" vs. "NPU command stream" actually looks like here.
_LARGE_ATTR_BYTES = 64 * 1024


def _standard_op_types(opset_version: int) -> set:
    types = set()
    for schema in defs.get_all_schemas_with_history():
        if schema.domain == "" and schema.since_version <= opset_version:
            types.add(schema.name)
    return types


def _attr_byte_size(attr: onnx.AttributeProto) -> int:
    if attr.HasField("t"):
        return len(attr.t.raw_data) or sum(
            len(getattr(attr.t, f))
            for f in (
                "float_data",
                "int32_data",
                "string_data",
                "int64_data",
                "double_data",
                "uint64_data",
            )
        )
    if attr.strings:
        return sum(len(s) for s in attr.strings)
    if attr.HasField("s"):
        return len(attr.s)
    return 0


def inspect(path: str) -> int:
    print(f"loading {path} with onnx.load() ...")
    model = onnx.load(path, load_external_data=False)
    print(f"  OK -- parsed as a standard ModelProto, ir_version={model.ir_version}")

    opset_version = 0
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset_version = max(opset_version, opset.version)
    print(f"  opset (default domain): {opset_version}")
    standard_ops = _standard_op_types(opset_version) if opset_version else set()

    custom_domain_nodes = []
    nonstandard_ops = []
    large_attrs = []

    def visit(graph: onnx.GraphProto, path_prefix: str) -> None:
        for node in graph.node:
            label = f"{path_prefix}{node.name or node.op_type}"
            if node.domain not in STANDARD_DOMAINS:
                custom_domain_nodes.append((label, node.op_type, node.domain))
            elif standard_ops and node.op_type not in standard_ops:
                nonstandard_ops.append((label, node.op_type))
            for attr in node.attribute:
                size = _attr_byte_size(attr)
                if size > _LARGE_ATTR_BYTES:
                    large_attrs.append((label, attr.name, size))
                if attr.HasField("g"):
                    visit(attr.g, f"{label}/{attr.name}/")
                for sub_g in attr.graphs:
                    visit(sub_g, f"{label}/{attr.name}/")

    visit(model.graph, "")

    print(f"\nnodes with a non-standard domain: {len(custom_domain_nodes)}")
    for label, op_type, domain in custom_domain_nodes[:50]:
        print(f"  {label}: op_type={op_type!r} domain={domain!r}")

    print(
        f"\nnode op_types outside the standard opset {opset_version}: "
        f"{len(nonstandard_ops)}"
    )
    for label, op_type in nonstandard_ops[:50]:
        print(f"  {label}: op_type={op_type!r}")

    print(
        f"\nattributes larger than {_LARGE_ATTR_BYTES} bytes "
        f"(candidate NPU blobs): {len(large_attrs)}"
    )
    for label, attr_name, size in sorted(large_attrs, key=lambda t: -t[2])[:50]:
        print(f"  {label}: attribute={attr_name!r} size={size} bytes")

    if not (custom_domain_nodes or nonstandard_ops or large_attrs):
        print(
            "\nNo boundary markers found by these heuristics -- either this "
            "file has no NPU subgraphs (e.g. a CPU-only/reference export), or "
            "the real scheme doesn't match what this script looks for. Time "
            "to open the file in `onnx inspect` / a hex viewer by hand."
        )
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("axmodel_path")
    args = ap.parse_args()
    return inspect(args.axmodel_path)


if __name__ == "__main__":
    sys.exit(main())
