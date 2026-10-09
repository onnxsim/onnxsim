"""Greedy-decode an `llm_build` LLM whose layers run on an onnx-remote AXCL worker.

    python scripts/axera/run_llm_rpc.py --host 127.0.0.1 --port 39501 \\
        --model-dir /mnt/share/smol/out --checkpoint ~/models/SmolLM2-135M \\
        --prompt "The capital of France is" --max-new-tokens 16 --compare-reference

`--model-dir` is a path on the machine the worker runs on: the directory
`pulsar2 llm_build` wrote (`llama_p128_l{i}_together.axmodel`,
`llama_post.axmodel`); the worker opens the files, this script never does.
`--checkpoint` is the Hugging Face checkpoint directory (`config.json`,
`model.safetensors`, `tokenizer.json`) and is read by this script, for the
embedding table and the float32 reference. When the worker runs on another
machine that directory has to exist here.

The worker must be up (`onnx-remote-axcl-worker --port P`, see
tools/onnx-remote/README.md); this script starts nothing and touches no
device API itself. See docs/axera-llm-rpc-decode.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CLIENT_DIR = os.path.join(_HERE, "..", "..", "tools", "onnx-remote", "python")
if _CLIENT_DIR not in sys.path:
    sys.path.insert(0, _CLIENT_DIR)

import llm_layer_loop as loop  # noqa: E402
import llm_reference as ref  # noqa: E402
import onnx_remote_client  # noqa: E402


def cache_len_from_io_info(info: dict) -> int:
    """kv_cache_len of a compiled layer: the row count of its K_cache input."""
    by_name = {e["name"]: e for e in info["inputs"]}
    entry = by_name.get("K_cache", info["inputs"][0])
    return int(entry["shape"][1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39501)
    ap.add_argument(
        "--model-dir",
        required=True,
        help="llm_build output directory, as the worker sees it",
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="HF checkpoint directory, read by this script",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument(
        "--prompt-ids",
        help="comma-separated token ids; replaces --prompt and needs no tokenizer",
    )
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument(
        "--layer-pattern",
        default=loop.LAYER_FILE_PATTERN,
        help="layer file name, {i} = layer index",
    )
    ap.add_argument("--post-name", default=loop.POST_FILE_NAME)
    ap.add_argument("--stop-at-eos", action="store_true")
    ap.add_argument(
        "--compare-reference",
        action="store_true",
        help="also run the float32 numpy reference and report token agreement and per-layer hidden-state error",
    )
    args = ap.parse_args(argv)

    model = ref.Model.load(args.checkpoint)
    tokenizer = None
    if args.prompt_ids:
        ids = [int(x) for x in args.prompt_ids.split(",")]
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(os.path.join(args.checkpoint, "tokenizer.json"))
        ids = tokenizer.encode(args.prompt).ids
    print("prompt ids", ids)

    client = onnx_remote_client.Client(args.host, args.port)
    print("worker", client.capabilities())
    directory = args.model_dir.rstrip("/")
    layer_paths = [
        f"{directory}/{args.layer_pattern.format(i=i)}" for i in range(model.L)
    ]
    post_path = f"{directory}/{args.post_name}"

    t0 = time.time()
    first = client.io_info(layer_paths[0])
    spec = loop.IOSpec.from_model(model, cache_len_from_io_info(first))
    backend = loop.RpcBackend(
        client, layer_paths=layer_paths, post_path=post_path, spec=spec
    )
    backend.preload()
    print("layer 0 io_info", first)
    print("post io_info", backend.info(post_path))
    print("%d models loaded on the worker in %.2f s" % (model.L + 1, time.time() - t0))

    host = loop.Host.from_model(model, spec)
    t0 = time.time()
    if args.compare_reference:
        result = loop.compare_decode(ids, args.max_new_tokens, backend, model)
        tokens, reference = result["tokens"], result["ref_tokens"]
    else:
        tokens = loop.decode(
            ids, args.max_new_tokens, backend, host=host, stop_at_eos=args.stop_at_eos
        )
        reference = None
    elapsed = time.time() - t0
    fed = len(ids) + max(len(tokens) - 1, 0)
    print("tokens", tokens)
    if tokenizer is not None:
        print("text", repr(tokenizer.decode(tokens)))
        if reference is not None:
            print("reference text", repr(tokenizer.decode(reference)))
    print(
        "%d worker calls, %.2f s, %.1f ms per fed token (%d layers + post)"
        % (backend.calls, elapsed, 1e3 * elapsed / max(fed, 1), model.L)
    )
    if reference is not None:
        same = tokens == reference
        print("identical to the float32 reference:", same)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
