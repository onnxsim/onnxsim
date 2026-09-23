"""Parse the vendor-compiled MatMul corpus shipped in Axera's public AX650 BSP.

``AXERA-TECH/ax650n_bsp_sdk`` carries 152 ``.axmodel`` files under
``msp/sample/ive/data/ive/matmul/matmul_models/``, compiled by Axera's own
toolchain for the IVE "matmul" sample: ``R[M, N] = X[M, K] @ Y[N, K]^T`` with two
*live*, already-quantized int8 or int16 inputs and an FP32 output. They span a
dtype x K x N x M grid under two compiler generations (``npu1``: IR 7 / opset 13;
``npu3``: IR 8 / opset 16). This module reads their graphs and compiled
``mcode`` with ``mcode.py`` and tabulates what varies. See
``docs/axera-bsp-mining.md`` for the findings.

Usage::

    bsp_matmul_corpus.py table DIR        # one row per model
    bsp_matmul_corpus.py check DIR        # mcode.check() violations per model

Downloading is left to the caller (the files are ~22 MB); ``DIR`` holds the
``.axmodel`` files with their original names.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_NAME = re.compile(
    r"^v1_matmul_(npu\d)_(s8|s16)_(\d+)_(\d+)_(\d+)_(\d+)\.axmodel(\.gz)?$"
)
_ELEM = {3: "s8", 5: "s16"}

ENGINES = {
    "npu1": {"conv": (0, 1), "teng": (2,), "cv": (3,), "dma": (4,)},
    "npu3": {
        "conv": (0, 1, 2, 3, 4, 5),
        "teng": (6, 7, 8),
        "cv": (9, 10, 11),
        "dma": (12, 13, 14),
    },
}
"""Segment index -> engine queue. ``npu1`` is the five-queue order this project
already uses for single-core builds (conv, conv, teng, cv, sdma). The ``npu3``
grouping (three cores per engine) is an inference from segment sizes, not a
decoded fact -- see ``docs/axera-bsp-mining.md``."""


def _read_model(path: str) -> onnx.ModelProto:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return onnx.load_model_from_string(f.read())
    return onnx.load(path, load_external_data=False)


def load(path: str) -> dict:
    """Graph facts plus the raw mcode, npu_params and npu_dyn_params bytes."""
    model = _read_model(path)
    (node,) = model.graph.node
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    info = json.loads(attrs["npu_graph_info"])
    (dot,) = info["dotneus"]
    inits = {i.name: bytes(i.raw_data) for i in model.graph.initializer}
    params_key = next(
        e["const_data_key"]
        for e in dot["extra_inputs"]
        if e.get("type", "DotNeuInputConst") == "DotNeuInputConst"
    )
    dyn_key = next(
        (
            e["dyn_param_expr_key"] or "npu_dyn_params"
            for e in dot["extra_inputs"]
            if e.get("type") == "DotNeuInputDynParam"
        ),
        None,
    )
    x, y = model.graph.input
    m, k = (d.dim_value for d in x.type.tensor_type.shape.dim)
    n, k2 = (d.dim_value for d in y.type.tensor_type.shape.dim)
    if k != k2:
        raise ValueError(f"{path}: X and Y disagree on K ({k} vs {k2})")
    return {
        "name": os.path.basename(path),
        "ir": model.ir_version,
        "opset": model.opset_import[0].version,
        "dtype": _ELEM[x.type.tensor_type.elem_type],
        "M": m,
        "N": n,
        "K": k,
        "mcode": inits[dot["neu_key"]],
        "params": inits[params_key],
        "dyn": inits.get(dyn_key, b"") if dyn_key else None,
    }


def gen(name: str) -> str:
    match = _NAME.match(name)
    if not match:
        raise ValueError(f"unexpected corpus file name {name!r}")
    return match.group(1)


def corpus(root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)):
        if name.endswith((".axmodel", ".axmodel.gz")):
            row = load(os.path.join(root, name))
            row["gen"] = gen(name)
            rows.append(row)
    return rows


def segment_sizes(blob: bytes) -> list[int]:
    return [length for _, length, _ in mcode.segments(blob)[1]]


def segment_bytes(blob: bytes, index: int) -> bytes:
    offset, length, _ = mcode.segments(blob)[1][index]
    return bytes(blob[offset : offset + length])


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("table", "check"):
        print(__doc__)
        return 2
    rows = corpus(argv[1])
    for r in rows:
        head = f"{r['gen']} {r['dtype']:3s} K={r['K']:3d} N={r['N']:6d} M={r['M']:6d}"
        if argv[0] == "table":
            try:
                segs = segment_sizes(r["mcode"])
            except Exception as exc:  # noqa: BLE001 -- report, keep going
                segs = f"no segment table ({exc})"
            dyn = "-" if r["dyn"] is None else r["dyn"].hex()
            print(
                f"{head} mcode={len(r['mcode']):7d} params={len(r['params']):6d}"
                f" dyn={dyn} segs={segs}"
            )
        else:
            print(f"{head} {mcode.check(r['mcode'])[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
