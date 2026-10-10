"""Greedy-decode an `llm_build` LLM whose layers run on an onnx-remote AXCL worker.

    python scripts/axera/run_llm_rpc.py --host 127.0.0.1 --port 39501 \\
        --model-dir /mnt/share/smol/out --checkpoint ~/models/SmolLM2-135M \\
        --prompt "The capital of France is" --max-new-tokens 16 --compare-reference

`--model-dir` is a path on the machine the worker runs on: the directory
`pulsar2 llm_build` wrote; the worker opens the files, this script never does.
The file names (`{model_type}_p{prefill_len}_l{i}_together.axmodel`,
`{model_type}_post.axmodel`) are not assumed. They come from, in this order:
`--layer-pattern` and `--post-name`; a listing of `--local-model-dir` (a copy
of the directory on this machine; `--model-dir` itself is listed when it
exists here); or probing the worker with `io_info` for the checkpoint's
`model_type` prefix and the usual prefill lengths. Tensor sizes, the cache
length and with it the context limit are read from the worker's `io_info`.

`--checkpoint` is the Hugging Face checkpoint directory (`config.json`,
`model.safetensors`, and `tokenizer.json` unless `--prompt-ids` is given) and
is read by this script, for the embedding table and the float32 reference.
When the worker runs on another machine that directory has to exist here.

Prompt: `--prompt TEXT` is tokenized as it is; `--chat` wraps it in the
checkpoint's chat template first (`tokenizer_config.json`'s `chat_template`,
or `chat_template.jinja`) as a single user turn, plus `--system TEXT`.
Qwen3's template has a thinking switch: `--chat` renders with
`enable_thinking=False`, which ends the prompt with an empty
`<think>\\n\\n</think>\\n\\n` block so the model answers directly; `--thinking`
leaves the block out and the model writes its reasoning first (which needs
far more than 16 new tokens, and the context holds `kv_cache_len` in all).
`--prompt-ids 1,2,3` takes token ids and needs no tokenizer.

Decoding is greedy. `--stop-at-eos` stops at the checkpoint's end-of-sequence
ids (`config.json` and `generation_config.json`).

The worker must be up (`onnx-remote-axcl-worker --port P`, see
tools/onnx-remote/README.md); this script starts nothing and touches no
device API itself. See docs/axera-llm-rpc-decode.md.

When the worker advertises `resident_state`, each layer's KV caches stay in
the worker's device buffers and a layer call carries only `indices`, `input`
and `mask`; `--no-resident-kv` sends the caches in full with every call, as
before. `--compare-layers` always sets the caches explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CLIENT_DIR = os.path.join(_HERE, "..", "..", "tools", "onnx-remote", "python")
if _CLIENT_DIR not in sys.path:
    sys.path.insert(0, _CLIENT_DIR)

import llm_layer_loop as loop  # noqa: E402
import llm_reference as ref  # noqa: E402
import onnx_remote_client  # noqa: E402


# --------------------------------------------------------------------------
# Chat templates
# --------------------------------------------------------------------------
def load_chat_template(checkpoint_dir: str) -> tuple[Optional[str], dict]:
    """(template text or None, tokenizer_config.json contents)."""
    config: dict = {}
    try:
        with open(os.path.join(checkpoint_dir, "tokenizer_config.json")) as f:
            config = json.load(f)
    except OSError:
        pass
    template = config.get("chat_template")
    if isinstance(template, list):  # [{"name": "default", "template": ...}, ...]
        named = {t.get("name"): t.get("template") for t in template}
        template = named.get("default") or next(iter(named.values()), None)
    if template is None:
        try:
            with open(os.path.join(checkpoint_dir, "chat_template.jinja")) as f:
                template = f.read()
        except OSError:
            pass
    return template, config


def render_chat_by_hand(
    template: str,
    user: str,
    system: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """One user turn (plus an optional system message) and the generation prompt,
    for the two template shapes used here, without a template engine:

    - Qwen3 (ChatML with an `enable_thinking` switch): no default system
      message; with thinking disabled the prompt ends with an empty think block.
    - plain ChatML (SmolLM2-Instruct, Qwen2): a system message is always
      present, the template's own default when none is given.
    """
    if "<|im_start|>" not in template:
        raise ValueError(
            "this chat template is not ChatML; install jinja2 to render it"
        )
    turn = "<|im_start|>{role}\n{content}<|im_end|>\n"
    if "enable_thinking" in template:
        text = turn.format(role="system", content=system) if system is not None else ""
        text += turn.format(role="user", content=user) + "<|im_start|>assistant\n"
        return text if enable_thinking else text + "<think>\n\n</think>\n\n"
    if system is None:
        # the default system message is a literal in the template
        # (inside a jinja string, so its line break is the two characters backslash, n)
        m = re.search(
            r"<\|im_start\|>system(?:\\n|\n)(.*?)<\|im_end\|>", template, re.S
        )
        system = m.group(1) if m else None
    text = turn.format(role="system", content=system) if system is not None else ""
    return text + turn.format(role="user", content=user) + "<|im_start|>assistant\n"


def render_chat_jinja(
    template: str,
    user: str,
    system: Optional[str] = None,
    enable_thinking: bool = False,
    tokenizer_config: Optional[dict] = None,
) -> str:
    """The template rendered by jinja2 the way transformers' apply_chat_template does
    (sandboxed, trim_blocks, lstrip_blocks, add_generation_prompt=True)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = raise_exception
    messages = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": user})
    special = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = (tokenizer_config or {}).get(key)
        special[key] = value.get("content") if isinstance(value, dict) else value
    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **{k: v for k, v in special.items() if v is not None},
    )


def render_chat(
    checkpoint_dir: str,
    user: str,
    system: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """The checkpoint's chat template applied to one user turn. jinja2 renders the
    template itself when it is installed; otherwise the hand-written ChatML forms."""
    template, config = load_chat_template(checkpoint_dir)
    if template is None:
        raise ValueError(
            f"{checkpoint_dir} has no chat template (tokenizer_config.json 'chat_template' "
            "or chat_template.jinja): drop --chat"
        )
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return render_chat_by_hand(template, user, system, enable_thinking)
    return render_chat_jinja(template, user, system, enable_thinking, config)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def timing_summary(step_seconds, n_prompt: int) -> list[str]:
    """Lines describing per-token wall times. `step_seconds` is Host.step_seconds:
    one (total, post) pair per fed token, the prompt's tokens first."""

    def line(name, steps):
        if not steps:
            return f"{name}: no tokens"
        ms = [1e3 * total for total, _ in steps]
        return (
            "%s: %d tokens, %.1f ms/token mean, %.1f median, %.1f min, %.1f max (%.2f tokens/s)"
            % (
                name,
                len(ms),
                statistics.fmean(ms),
                statistics.median(ms),
                min(ms),
                max(ms),
                1e3 / statistics.fmean(ms),
            )
        )

    steps = list(step_seconds)
    prefill, generated = steps[:n_prompt], steps[n_prompt:]
    lines = [
        line("prefill (prompt, one token per step)", prefill),
        line("decode (generated tokens fed back)", generated),
    ]
    posts = [1e3 * post for _, post in steps if post > 0]
    layers = [1e3 * (total - post) for total, post in steps]
    if layers:
        lines.append(
            "per fed token: all layers %.1f ms mean; post model %.1f ms mean over %d calls"
            % (
                statistics.fmean(layers),
                statistics.fmean(posts) if posts else 0.0,
                len(posts),
            )
        )
        lines.append(
            "first new token after %.2f s" % sum(total for total, _ in prefill)
        )
    return lines


# --------------------------------------------------------------------------
def find_files(args, model: ref.Model, client) -> tuple[list, str, str]:
    """(layer paths, post path, how they were found), all as the worker sees them."""
    directory = args.model_dir.rstrip("/")
    if args.layer_pattern or args.post_name:
        if not (args.layer_pattern and args.post_name):
            raise SystemExit("--layer-pattern and --post-name go together")
        return (
            [f"{directory}/{args.layer_pattern.format(i=i)}" for i in range(model.L)],
            f"{directory}/{args.post_name}",
            "given on the command line",
        )
    local = args.local_model_dir or (directory if os.path.isdir(directory) else None)
    if local:
        files = loop.list_model_dir(local, args.prefix)
        how = f"listed in {local}"
    else:
        prefixes = [args.prefix] if args.prefix else [model.model_type, "llama"]

        def exists(path):
            try:
                client.io_info(path)
                return True
            except onnx_remote_client.RemoteError:
                return False

        files = loop.probe_files(exists, directory, prefixes, model.L)
        how = "probed on the worker"
    if files.num_layers != model.L:
        raise SystemExit(
            f"{files.num_layers} layer files ({files.layer_pattern}), the checkpoint has {model.L} layers"
        )
    return files.layer_paths(directory), files.post_path(directory), how


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
        "--local-model-dir",
        help="a copy of that directory on this machine; only listed, to learn the file names",
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
    ap.add_argument(
        "--chat",
        action="store_true",
        help="wrap --prompt in the checkpoint's chat template as one user turn (thinking disabled)",
    )
    ap.add_argument("--system", help="system message for --chat")
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="with --chat: render with enable_thinking=True (Qwen3 then reasons before answering)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument(
        "--prefix",
        help="file name prefix when the directory holds several models (default: the only one / model_type)",
    )
    ap.add_argument(
        "--layer-pattern",
        help="layer file name with {i} = layer index; with --post-name, skips discovery",
    )
    ap.add_argument("--post-name")
    ap.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="stop at an end-of-sequence id",
    )
    ap.add_argument(
        "--compare-reference",
        action="store_true",
        help="also run the float32 numpy reference and report token agreement and per-layer hidden-state error",
    )
    ap.add_argument(
        "--compare-layers",
        action="store_true",
        help="teacher-forced per-layer check of the prompt tokens against the reference, instead of decoding",
    )
    ap.add_argument(
        "--no-resident-kv",
        action="store_true",
        help="send both KV caches with every layer call and take the rows back (the original "
        "request form) even when the worker can keep the caches in its device buffers",
    )
    ap.add_argument(
        "--mirror-kv",
        action="store_true",
        help="with resident KV caches: also have every token's K/V rows returned, so the host's "
        "copy of the caches stays complete (diagnostics; a lost worker state is then re-uploaded "
        "instead of recomputed)",
    )
    args = ap.parse_args(argv)

    model = ref.Model.load(args.checkpoint)
    print(
        "checkpoint: %s, hidden %d, %d layers, %d heads / %d kv heads x head_dim %d, vocab %d, eos %s"
        % (
            model.model_type,
            model.H,
            model.L,
            model.nh,
            model.nkv,
            model.hd,
            model.vocab,
            list(model.eos_ids),
        )
    )
    tokenizer = None
    tokenizer_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
    if args.prompt_ids:
        if args.chat:
            raise SystemExit("--chat applies to --prompt, not to --prompt-ids")
        ids = [int(x) for x in args.prompt_ids.split(",")]
    elif tokenizer is None:
        raise SystemExit(
            f"{args.checkpoint} has no tokenizer.json: pass the prompt as --prompt-ids"
        )
    else:
        text = args.prompt
        if args.chat:
            text = render_chat(args.checkpoint, text, args.system, args.thinking)
            print("chat prompt", repr(text))
        ids = tokenizer.encode(text).ids
    if not ids or min(ids) < 0 or max(ids) >= model.vocab:
        raise SystemExit(f"prompt ids {ids} are not all in 0..{model.vocab - 1}")
    print("prompt ids", ids)

    client = onnx_remote_client.Client(args.host, args.port)
    capabilities = client.capabilities()
    print("worker", capabilities)
    t0 = time.time()
    layer_paths, post_path, how = find_files(args, model, client)
    print("model files %s: %s ... %s" % (how, layer_paths[0], post_path))
    backend = loop.RpcBackend(
        client,
        layer_paths=layer_paths,
        post_path=post_path,
        resident=False
        if args.no_resident_kv
        else bool(capabilities.get("resident_state")),
    )
    print(
        "KV caches: %s"
        % (
            "resident on the worker (a layer call sends indices, input and mask)"
            if backend.resident
            else "sent in full with every layer call"
            + (
                ""
                if args.no_resident_kv
                else " (the worker does not advertise resident_state)"
            )
        )
    )
    spec = (
        backend.spec
    )  # sizes from the worker's io_info of layer 0 and of the post model
    print("layer 0 io_info", backend.info(layer_paths[0]))
    print("post io_info", backend.info(post_path))
    print(
        "compiled: hidden %d, KV width %d, cache %d rows (context limit %d tokens), vocab %d"
        % (spec.hidden, spec.kv_dim, spec.cache_len, spec.cache_len, spec.vocab)
    )
    spec.check_model(model)
    backend.preload()
    print("%d models loaded on the worker in %.2f s" % (model.L + 1, time.time() - t0))

    host = loop.Host.from_model(model, spec)
    host.mirror_kv = args.mirror_kv
    if args.compare_layers:
        result = loop.compare_layers(ids, backend, model)
        worst = float(result["output"].max())
        print("worst teacher-forced layer output rel-L2 error: %.4f" % worst)
        return 0
    fed = len(ids) + max(args.max_new_tokens - 1, 0)
    if fed > spec.cache_len:
        raise SystemExit(
            f"{len(ids)} prompt tokens + {args.max_new_tokens} new ones feed {fed} tokens, "
            f"the compiled context holds {spec.cache_len}"
        )
    t0 = time.time()
    if args.compare_reference:
        result = loop.compare_decode(
            ids,
            args.max_new_tokens,
            backend,
            model,
            host=host,
            stop_at_eos=args.stop_at_eos,
        )
        tokens, reference = result["tokens"], result["ref_tokens"]
    else:
        tokens = loop.decode(
            ids, args.max_new_tokens, backend, host=host, stop_at_eos=args.stop_at_eos
        )
        reference = None
    elapsed = time.time() - t0
    print("tokens", tokens)
    if tokenizer is not None:
        print("text", repr(tokenizer.decode(tokens, skip_special_tokens=False)))
        if reference is not None:
            print(
                "reference text",
                repr(tokenizer.decode(reference, skip_special_tokens=False)),
            )
    if args.stop_at_eos and tokens and tokens[-1] in host.eos_ids:
        print("stopped at eos id", tokens[-1])
    print(
        "%d worker calls in %.2f s (%.2f s in layer calls, %.2f s in post calls; %d layers + post)"
        % (
            backend.calls,
            elapsed,
            backend.layer_seconds,
            backend.post_seconds,
            model.L,
        )
    )
    fed_tokens = max(len(host.step_seconds), 1)
    print(
        "tensor payload per fed token: %.0f bytes sent, %.0f received (layer and post calls)"
        % (backend.bytes_sent / fed_tokens, backend.bytes_received / fed_tokens)
    )
    if host.recoveries or backend.resident_losses:
        print(
            "resident KV state was lost %d time(s) and rebuilt; resident mode is %s"
            % (host.recoveries, "still on" if backend.resident else "off now")
        )
    for text in timing_summary(host.step_seconds, len(ids)):
        print(text)
    if reference is not None:
        same = tokens == reference
        print("identical to the float32 reference:", same)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
