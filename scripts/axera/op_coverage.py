#!/usr/bin/env python3
"""Per-ONNX-op coverage for the Axera AX650 NPU, and per-model op usage.

Two questions this answers without a compiler or a card:

1. **Of the ai.onnx operator set, how much can the AX650 take?** Every schema
   in the default domain is classified against `pulsar2_ops`: NPU-eligible,
   confirmed broken on real hardware, CPU-only by construction (control flow,
   sequences, strings), or simply not on the vendor's list.
2. **For a given model, which of its ops are eligible?** Node counts by op
   type, so "89% of nodes" and "the six op types blocking it" are separate
   numbers -- they usually tell very different stories.

Usage::

    op_coverage.py                          # the ai.onnx table
    op_coverage.py model.onnx [more.onnx]   # plus per-model usage
    op_coverage.py --csv coverage.csv model.onnx
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import onnx
import onnx.defs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pulsar2_ops  # noqa: E402

#: Entries of ``AX650_SUPPORTED_OPS`` that are Pulsar2-native op names rather
#: than ai.onnx ones. They are not typos; they name fused or vendor ops the
#: compiler accepts, and they will never match a node in a standard graph.
VENDOR_NATIVE = frozenset({"InverseSigmoid", "Silu", "SpatialTransformer"})

#: Classifications, most specific first.
ELIGIBLE = "npu"
BROKEN = "broken"
CPU_ONLY = "cpu-only"
UNLISTED = "unlisted"


def classify(op_type):
    """How the AX650 heuristic treats one ai.onnx op type."""
    if op_type in pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS:
        return BROKEN
    if op_type in pulsar2_ops.AX650_SUPPORTED_OPS:
        return ELIGIBLE
    if op_type in pulsar2_ops.CPU_ONLY_OPS:
        return CPU_ONLY
    return UNLISTED


def onnx_op_types():
    """Every op type in the default (ai.onnx) domain."""
    return sorted({s.name for s in onnx.defs.get_all_schemas() if s.domain == ""})


def coverage_table():
    """`{classification: [op types]}` over the whole ai.onnx operator set."""
    out = collections.defaultdict(list)
    for op in onnx_op_types():
        out[classify(op)].append(op)
    return dict(out)


def model_usage(path):
    """`(node count, {(domain, op_type): count})` for one model, without
    loading its weights -- op types do not depend on them, and these graphs
    routinely carry hundreds of megabytes of external data."""
    model = onnx.load(path, load_external_data=False)
    counts = collections.Counter(
        (node.domain or "", node.op_type) for node in model.graph.node
    )
    return sum(counts.values()), counts


def summarise(path):
    """`(total nodes, eligible nodes, {label: count} for the rest)`."""
    total, counts = model_usage(path)
    eligible, blocked = 0, collections.Counter()
    for (domain, op_type), n in counts.items():
        if domain == "" and classify(op_type) == ELIGIBLE:
            eligible += n
        else:
            blocked[f"{domain + '.' if domain else ''}{op_type}"] += n
    return total, eligible, blocked


#: Pulsar2's own fused operator names, observed in real builds. They appear in
#: a compiled model's `quant/quant_axmodel.json` as `op_<n>:<name>` and exist
#: nowhere in the ONNX graph, which is what makes them unreachable by
#: `layer_configs.op_types` -- that field matches ai.onnx names only.
FUSED_OPS = frozenset(
    {
        "onnx.FullyConnected",
        "onnx.Matmul",
        "onnx.Mul",
        "onnx.RMSNormalization",
        "onnx.LayerNormalization",
        "onnx.RotaryEmbedding",
        "onnx.Silu",
        "onnx.Gelu",
        "onnx.pre_Reshape",
        "onnx.pre_Transpose",
    }
)


def dispatch_report(build_dir):
    """What a finished build actually scheduled, from its own quant config.

    Returns `(targets, by_name)`: how many layers went to each execution
    engine, and the layers grouped by whether they kept their ONNX node name
    or were fused into one of Pulsar2's own operators.

    This is the report that would have prevented several wasted builds. A
    `layer_configs` entry keyed on `op_types` reaches only the ONNX-named
    group; the fused group can be targeted solely by `layer_names`, using the
    names printed here. Asking for a precision change on a fused operator by
    op type does not fail -- it is ignored, and the build succeeds unchanged.
    """
    path = os.path.join(build_dir, "quant", "quant_axmodel.json")
    with open(path) as handle:
        quant = json.load(handle)
    targets = collections.Counter()
    by_name = collections.defaultdict(list)
    for layer, engine in (quant.get("dispatchings") or {}).items():
        targets[engine] += 1
        fused = layer.split(":", 1)[1] if ":" in layer else None
        by_name["fused" if fused else "onnx"].append(layer)
    return targets, dict(by_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="ONNX models to inventory")
    parser.add_argument("--csv", help="write the per-op table here")
    parser.add_argument(
        "--build-dir",
        help="a finished pulsar2 output dir to report dispatch and fused-op names for",
    )
    args = parser.parse_args(argv)

    table = coverage_table()
    total_ops = sum(len(v) for v in table.values())
    print(f"ai.onnx operator set: {total_ops} op types")
    for label in (ELIGIBLE, BROKEN, CPU_ONLY, UNLISTED):
        ops = table.get(label, [])
        print(f"  {label:9s} {len(ops):4d}  {100 * len(ops) / total_ops:5.1f}%")
    extra = sorted(set(pulsar2_ops.AX650_SUPPORTED_OPS) - set(onnx_op_types()))
    print(f"  vendor-native entries (never match a standard node): {extra}")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["op_type", "classification"])
            for op in onnx_op_types():
                writer.writerow([op, classify(op)])
        print(f"wrote {args.csv}")

    if args.build_dir:
        targets, by_name = dispatch_report(args.build_dir)
        print(f"\n{args.build_dir}: {sum(targets.values())} dispatched layers")
        for engine, n in targets.most_common():
            print(f"  {engine:28s} {n:6d}")
        fused = by_name.get("fused", [])
        print(f"  reachable by op_types : {len(by_name.get('onnx', [])):6d}")
        print(f"  layer_names only      : {len(fused):6d}")
        kinds = collections.Counter(f.split(":", 1)[1] for f in fused)
        for name, n in kinds.most_common(10):
            print(f"      {name:32s} {n:5d}")

    for path in args.models:
        total, eligible, blocked = summarise(path)
        share = 100 * eligible / total if total else 0.0
        print(f"\n{os.path.basename(path)}: {total} nodes, {share:.1f}% eligible")
        for op, n in blocked.most_common(12):
            print(f"    {op:48s} {n:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
