#!/usr/bin/env python3
"""Export nanochat-style GPT models to ONNX and simplify with onnxsim.

See model.py's module docstring for what this reimplements and why (short
version: nanochat's own attention backend, Flash Attention 3 with a
sliding-window pattern, does not trace to ONNX at all, so this is a
standalone, structurally-faithful, ONNX-exportable stand-in). nanochat has no
ONNX export path of its own to reproduce, so unlike scripts/yolo and
scripts/rfdetr (which replay an existing package's real export call), this
harness is the export path.

Models are built at nanochat's own depth-based sizing (``--depth``, see
``config_for_depth()``) with random weights by default -- the graph
*structure* onnxsim simplifies doesn't depend on the weight values, so this
stays fast and offline, same reasoning as the YOLO/RF-DETR harnesses. Pass
``--checkpoint`` to export weights trained by ``train_tiny.py`` instead.

Usage::

    python simplify_nanochat.py                        # depth=4, fast default
    python simplify_nanochat.py --depth 4 --depth 12 --depth 20
    python simplify_nanochat.py --checkpoint ckpt.pt --seq-len 64
    python simplify_nanochat.py --tiny --output-dir .   # tiny wasm-UI-sized fixture
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import onnx
import torch

import onnxsim

_NANOCHAT_DIR = os.path.dirname(os.path.abspath(__file__))
if _NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, _NANOCHAT_DIR)

from model import GPT, GPTConfig, config_for_depth  # noqa: E402

# Small enough to export/simplify/check in well under a second each, but
# still real vocab/seq-len enough to exercise the full op set (rotary,
# RMSNorm, GQA repeat_interleave, relu^2, logit softcap).
_WASM_DEMO_CONFIG = GPTConfig(
    sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=16
)


def _build_model(args, depth: int | None) -> tuple[torch.nn.Module, GPTConfig]:
    if args.checkpoint:
        # train_tiny.py's saved dict: {"model": state_dict, "config": vars(config), ...},
        # built with pad_vocab_size_to=8 -- match both to load cleanly.
        state = torch.load(args.checkpoint, map_location="cpu")
        config = GPTConfig(**state["config"])
        model = GPT(config, pad_vocab_size_to=8).eval()
        model.load_state_dict(state["model"])
        return model, config

    if args.tiny:
        config = _WASM_DEMO_CONFIG
    else:
        config = config_for_depth(
            depth, sequence_len=args.seq_len, vocab_size=args.vocab_size
        )
        if args.n_kv_head is not None:
            config.n_kv_head = args.n_kv_head

    model = GPT(config, pad_vocab_size_to=8 if args.tiny else 64).eval()
    return model, config


def run_one(args, depth_label: str, depth: int | None) -> dict:
    result: dict = {
        "depth": depth_label,
        "onnxsim_version": onnxsim.__version__,
        "onnx_version": onnx.__version__,
    }
    try:
        torch.manual_seed(0)
        model, config = _build_model(args, depth)
        result["n_embd"] = config.n_embd
        result["n_head"] = config.n_head
        result["n_kv_head"] = config.n_kv_head
        result["n_layer"] = config.n_layer
        result["params"] = sum(p.numel() for p in model.parameters())

        os.makedirs(args.output_dir, exist_ok=True)
        onnx_path = os.path.join(args.output_dir, f"nanochat_{depth_label}.onnx")
        dummy = torch.randint(0, config.vocab_size, (1, config.sequence_len), dtype=torch.long)

        t0 = time.time()
        torch.onnx.export(
            model,
            (dummy,),
            onnx_path,
            opset_version=args.opset,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamo=False,
        )
        result["export_s"] = round(time.time() - t0, 1)
        result["onnx_path"] = onnx_path

        graph = onnx.load(onnx_path)
        result["nodes_before"] = len(graph.graph.node)

        t0 = time.time()
        model_opt, check_ok = onnxsim.simplify(onnx_path, check_n=3)
        result["simplify_s"] = round(time.time() - t0, 1)
        result["check_ok"] = bool(check_ok)
        result["nodes_after"] = len(model_opt.graph.node)
        onnx.save(model_opt, onnx_path.replace(".onnx", ".sim.onnx"))
        result["status"] = "OK" if check_ok else "SIMPLIFY_CHECK_FAILED"
    except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth", type=int, action="append", dest="depths",
        help="nanochat depth d (n_layer=d, n_embd via config_for_depth); repeatable",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--vocab-size", type=int, default=2048,
        help="much smaller than nanochat's real ~32768/65536 so the demo's "
        "wte/lm_head stay small and fast",
    )
    parser.add_argument(
        "--n-kv-head", type=int, default=None,
        help="override to < n_head for GQA (config_for_depth defaults to full MHA, "
        "matching nanochat's own base_train.py)",
    )
    parser.add_argument("--checkpoint", default=None, help="train_tiny.py checkpoint to export")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", default="nanochat_onnx")
    parser.add_argument(
        "--tiny", action="store_true",
        help="tiny fixed-size config sized for committing/loading in the browser wasm UI, "
        "ignores --depth/--seq-len/--vocab-size/--n-kv-head",
    )
    args = parser.parse_args()

    if args.checkpoint and (args.depths or args.tiny):
        parser.error("--checkpoint loads its own saved config; drop --depth/--tiny")

    if args.checkpoint:
        depths: list[int | None] = [None]
        labels = ["checkpoint"]
    elif args.tiny:
        depths = [None]
        labels = ["tiny"]
    else:
        depths = args.depths or [4]
        labels = [f"d{d}" for d in depths]

    results = []
    for label, depth in zip(labels, depths):
        print(f"=== {label} ===", flush=True)
        result = run_one(args, label, depth)
        print(json.dumps(result), flush=True)
        results.append(result)

    print("\n===== SUMMARY =====")
    ok = 0
    for r in results:
        status = r.get("status")
        ok += status == "OK"
        b, a = r.get("nodes_before", "?"), r.get("nodes_after", "?")
        red = (
            f"{100 * (b - a) // b}%"
            if isinstance(b, int) and isinstance(a, int) and b
            else "?"
        )
        print(
            f"{r['depth']:10s} n_embd={str(r.get('n_embd', '?')):5s} {status:22s} "
            f"nodes {b}->{a} ({red}) check_ok={r.get('check_ok', '?')} {r.get('error', '')}"
        )
    print(f"\n{ok}/{len(results)} configs simplified with onnxsim {onnxsim.__version__}")


if __name__ == "__main__":
    main()
