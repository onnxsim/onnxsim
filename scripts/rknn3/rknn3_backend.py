#!/usr/bin/env python3
"""Wraps RKNN3-Toolkit's dedicated LLM conversion API, `rknn.api.RKNN.load_llm()`.

RKNN3-Toolkit (`airockchip/rknn3-toolkit`, targeting the RK1820/RK1828/RK3572
NPU line -- a distinct, **incompatible** package from `rknn-toolkit2`,
despite the identical ``from rknn.api import RKNN`` import path this repo's
other `scripts/rknn` harness also uses) has explicit LLM/VLM support built
around a second entry point alongside the ordinary `load_onnx()`:

    rknn.config(target_platform="rk1820", quantized_dtype="w4a16", ...)
    rknn.load_llm(model="model.onnx", config="model.config.pkl", seq_lens=[1, 128])
    rknn.build(do_quantization=True, dataset="calib.txt")
    rknn.export_rknn("model.rknn")

Unlike Rockchip's older, separate RKLLM stack (`airockchip/rknn-llm`, whose
`rkllm.api.RKLLM.load_huggingface()`/`load_gguf()` never touches ONNX at
all), `load_llm()`'s first argument **is an ONNX file** -- a fixed-shape,
KV-cache-friendly export of a plain HF causal LM (`input_ids`/
`attention_mask`/`position_ids`/`num_logits_to_keep`, `use_cache=False`),
produced by `scripts/rknn3/llm_export.py` here (a trimmed, verified port of
`airockchip/rknn3-model-zoo`'s own `causal_llm_to_onnx` reference
implementation). So, unlike RKLLM, onnxsim genuinely has a hook into this
pipeline: this harness runs onnxsim's `simplify()` on that exported ONNX and
confirms `load_llm()` + `build()` + the PC simulator still accept it and
produce the same result.

## A real, verified quirk: `load_llm()` strips the embedding lookup

Confirmed by direct reproduction against a real `Qwen2ForCausalLM` checkpoint
(`yujiepan/qwen2.5-tiny-random`, a real Qwen2.5-architecture config with tiny
dimensions): even though the exported ONNX declares `input_ids` (int64
token ids) as its first input, `load_llm()`'s own log says so explicitly --
``"The gather index 'input_ids' is from model input, but 'auto' found in
vocab, treat it as embedding!"`` -- and `rknn.inference()` then expects
**precomputed float embeddings** (`input_embeds`) as its first positional
input instead, looked up host-side from the separately exported
`.embed.bin` float16 weight file. :func:`run` does this lookup itself
before calling `inference()`.

## Fidelity tier

Same PC-simulator-only tier as `scripts/rknn/rknn_backend.py` (see that
module's docstring for the general caveats): `init_runtime()` with no
`target=` argument runs Rockchip's own PC simulator, not real RK1820/RK1828/
RK3572 hardware -- no NPU coprocessor board or `rknn3_transfer_proxy` link
was used or claimed anywhere in this file. `build(do_quantization=False)`
throughout: this checks graph-compile and single-token-prefill numerics, not
GRQ/W4A16 quantization accuracy (a separate, calibration-dataset-dependent
concern).

This module degrades gracefully: whenever `rknn-toolkit` (the RKNN3 one) or
its LLM API can't be imported, `RKNN3_AVAILABLE` is False and
:func:`unavailable_reason` explains why.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
from typing import Dict, Tuple

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, path: str):
    """Load a module straight from its file path, *not* via ``sys.path`` +
    ``import`` -- deliberately does not register it under a bare top-level
    name in ``sys.modules``.

    A real, reproduced bug this avoids: `scripts/common/ep_numerics.py` is
    reused across every `scripts/<vendor>` harness the same way the sibling
    `scripts/rknn` (rknn-toolkit2) harness imports it -- via `sys.path.
    insert(0, scripts_dir)` then `from common.ep_numerics import compare`.
    That registers onnxsim's own `scripts/common` package as
    `sys.modules["common"]`. RKNN3-Toolkit's C-extension code *also* does a
    bare `import common...` internally (its own, differently-shaped vendored
    package) -- confirmed by direct reproduction: with `scripts/` on
    `sys.path` at any point before `rknn.load_llm()` runs, it fails with
    `ModuleNotFoundError: No module named 'common.rknpu_profiler'`, because
    Python resolves `common` to the already-cached (and already
    fully-imported, `sys.path` removal after the `import` does not undo the
    cache entry) onnxsim package instead of RKNN3-Toolkit's own. Loading
    `ep_numerics.py` directly by path -- it only imports `numpy`/`onnx`
    itself, no package-relative imports -- sidesteps the collision entirely
    rather than depending on import order.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name!r} from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_module(
    "_rknn3_ep_numerics", os.path.join(_SCRIPTS_DIR, "common", "ep_numerics.py")
).compare
llm_export = _load_module("_rknn3_llm_export", os.path.join(_HERE, "llm_export.py"))

# Purely a compile-time parameter for the simulator build -- never used to
# select or contact a real device (init_runtime() with no target= always
# runs the host-CPU simulator regardless of this value).
TARGET_PLATFORM = os.environ.get("RKNN3_TARGET_PLATFORM", "rk1820")
# A real, tiny (hidden_size=8, 2 layers), publicly hosted Qwen2.5-architecture
# checkpoint (config.json's own architectures: ["Qwen2ForCausalLM"]) -- small
# enough (~10MB) to download and convert in a CI job, but a real HF checkpoint
# exercising the real export path, not a from-scratch synthetic graph.
MODEL_NAME = os.environ.get("RKNN3_LLM_MODEL", "yujiepan/qwen2.5-tiny-random")
SEQ_LEN = int(os.environ.get("RKNN3_LLM_SEQ_LEN", "16"))

RKNN3_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None

try:
    from rknn.api import RKNN  # noqa: E402

    if not hasattr(RKNN, "load_llm"):
        raise RuntimeError(
            "this 'rknn' package has no load_llm() -- looks like rknn-toolkit2, "
            "not RKNN3-Toolkit (both ship the same 'rknn.api.RKNN' import path)"
        )
    RKNN3_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without the SDK
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> str:
    return _UNAVAILABLE_REASON or "unknown"


def export_llm_artifacts(out_dir: str, seq_len: int = SEQ_LEN) -> Dict[str, str]:
    """Download :data:`MODEL_NAME` and export the ONNX + `.config.pkl` +
    `.embed.bin` triple `load_llm()` needs. Returns their paths."""
    model = llm_export.load_causal_lm(MODEL_NAME)

    onnx_path = os.path.join(out_dir, "model.onnx")
    config_path = os.path.join(out_dir, "model.config.pkl")
    embed_path = os.path.join(out_dir, "model.embed.bin")

    llm_export.export_causal_lm_to_onnx(model, onnx_path, prompt_size=seq_len)
    llm_export.export_embed_weight(model.model.embed_tokens.weight, embed_path)
    llm_export.export_llm_config(MODEL_NAME, config_path)

    return {"onnx": onnx_path, "config": config_path, "embed": embed_path}


def run(
    onnx_path: str,
    config_path: str,
    embed_path: str,
    token_ids: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Convert ``onnx_path`` through `load_llm()` and run one prefill step
    through the PC simulator, returning the last-token logits.

    Raises on any failure (unsupported graph, build error, ...) -- callers
    decide what a raised exception means for their status.
    """
    if not RKNN3_AVAILABLE:
        raise RuntimeError(unavailable_reason())

    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    hidden_size = cfg["hidden_size"]

    embeds_table = np.fromfile(embed_path, dtype=np.float16).reshape(
        cfg["vocab_size"], hidden_size
    )

    rknn = RKNN(verbose=False)
    # build() drops a `tmp/model_report.html` next to the process's current
    # directory with no path parameter to redirect it (confirmed: it landed
    # in the onnxsim repo checkout during development of this harness) --
    # run from onnx_path's own directory so it lands there instead, and gets
    # cleaned up along with the rest of that scratch directory.
    prev_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(onnx_path)) or ".")
    try:
        ret = rknn.config(target_platform=TARGET_PLATFORM)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed (ret={ret})")
        ret = rknn.load_llm(model=onnx_path, config=config_path, seq_lens=[1, seq_len])
        if ret != 0:
            raise RuntimeError(f"rknn.load_llm failed (ret={ret})")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed (ret={ret})")
        ret = rknn.init_runtime()  # no target= => PC simulator, no device
        if ret != 0:
            raise RuntimeError(f"rknn.init_runtime failed (ret={ret})")

        input_seq_len = token_ids.shape[1]
        embeds = embeds_table[token_ids].astype(np.float32)
        rope_cache = rknn.query("QUERY_ROPE_CACHE")

        inputs_embeds = np.zeros((1, seq_len, hidden_size), dtype=np.float32)
        attention_mask = np.zeros((1, seq_len), dtype=np.float32)
        inputs_embeds[:, :input_seq_len, :] = embeds
        attention_mask[:, :input_seq_len] = 1
        num_logits_to_keep = np.array([input_seq_len - 1], dtype=np.int32)

        attention_inputs, _ = rknn.kvcache_controller.generate_kvcache_control_tensors(
            input_seq_len
        )
        prefill_inputs = (
            [inputs_embeds, attention_mask, num_logits_to_keep]
            + rope_cache
            + attention_inputs[0]
        )
        data_format = ["nchw"] * len(prefill_inputs)

        logits = rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)
        if logits is None:
            raise RuntimeError("rknn.inference returned None")
        return logits[0]
    finally:
        rknn.release()
        os.chdir(prev_cwd)


def compare_logits(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-2, atol: float = 1e-3
) -> Tuple[bool, float]:
    return compare([a], [b], rtol=rtol, atol=atol)
