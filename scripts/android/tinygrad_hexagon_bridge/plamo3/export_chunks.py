#!/usr/bin/env python3
"""Export PLaMo 3 as phone-sized ONNX programs: one graph per chunk of decoder layers, plus the head.

The host keeps the embedding table and every layer's KV cache; each chunk program is one decode step over its
layers with a static cache of C slots (C+1 with the new token). Graphs are float32 with the weights as initializers.

  export_chunks.py CHECKPOINT_DIR OUT_DIR [--chunk 1] [--cache-len 15] [--layers 0-23] [--head]

The onnx-remote wire carries at most 512 MB per artifact, so one decoder layer (~392 MB fp32) per program.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "axera"))

import plamo3_onnx as O  # noqa: E402
import plamo3_reference as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--cache-len", type=int, default=15)
    ap.add_argument("--layers", default=None, help="first-last layer, default all")
    ap.add_argument(
        "--int8",
        action="store_true",
        help="store the layer matrices as per-column int8 (dequantized in the graph)",
    )
    ap.add_argument(
        "--int4",
        action="store_true",
        help="store the layer matrices as per-column int4, packed two per byte",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=0,
        help="export row-block graphs (prefill, or a padded decode step) with this many rows",
    )
    ap.add_argument(
        "--split",
        default="",
        help="column slices per matrix for row-block graphs, e.g. qkv:2,o:2,gu:8,d:4",
    )
    ap.add_argument(
        "--state",
        action="store_true",
        help="row-block graphs keep the KV caches on the device (recurrent state)",
    )
    ap.add_argument(
        "--qdir",
        default="",
        help="int4 codes and group scales per layer (<qdir>/<layer>.npz: <key>_q [out, in], <key>_s [in/G, out]), e.g. from GPTQ",
    )
    ap.add_argument(
        "--head",
        action="store_true",
        help="also export the final norm + lm_head (1.1 GB: over the wire limit)",
    )
    a = ap.parse_args()
    split = {k: int(v) for k, v in (kv.split(":") for kv in a.split.split(",") if kv)}
    qsource = None
    if a.qdir:
        import numpy as np

        cache = {}

        def qsource(li, key, B):
            if li not in cache:
                cache[li] = np.load(os.path.join(a.qdir, f"{li:02d}.npz"))
            z = cache[li]
            # codes are stored [out, in]; the graph wants B = [in, out]; scales [in/G, out] already match columns
            return z[key + "_q"].T.astype(np.int8), z[key + "_s"].astype(np.float32)

    os.makedirs(a.out, exist_ok=True)
    model = R.Plamo3Reference.from_checkpoint(a.checkpoint)
    lo, hi = (0, model.L - 1) if a.layers is None else map(int, a.layers.split("-"))
    chunks = []
    for start in range(lo, hi + 1, a.chunk):
        end = min(start + a.chunk - 1, hi)
        path = os.path.join(a.out, f"chunk_{start:02d}_{end:02d}.onnx")
        if a.rows:
            O.build_block(
                model,
                range(start, end + 1),
                path,
                cache_len=a.cache_len,
                rows=a.rows,
                int8=a.int8,
                int4=a.int4,
                split=split,
                state=a.state,
                qsource=qsource,
            )
        else:
            O.build_chunk(
                model,
                range(start, end + 1),
                path,
                cache_len=a.cache_len,
                int8=a.int8,
                int4=a.int4,
            )
        chunks.append((start, end, path))
        print("chunk", start, end, path, flush=True)
    if a.head and hi == model.L - 1:
        path = os.path.join(a.out, "head.onnx")
        O.build_head(model, path)
        print("head", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
