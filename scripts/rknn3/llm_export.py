#!/usr/bin/env python3
"""Export a Hugging Face causal-LM checkpoint to the ONNX + sidecar files
`rknn.api.RKNN.load_llm()` (RKNN3-Toolkit's LLM/VLM conversion entry point,
see ``rknn3_backend.py``'s docstring) expects.

Trimmed, self-contained port of the *plain single-segment causal-LM* export
path in `airockchip/rknn3-model-zoo`'s `py_utils/export_llm_helper.py`
(`causal_llm_to_onnx` / `export_llm_config` / `export_embed_weight`) --
verified against that real reference implementation and a real
`yujiepan/qwen2.5-tiny-random` checkpoint (a `Qwen2ForCausalLM`, the same
class real Qwen2.5 checkpoints use, just with tiny dimensions). Not vendored
wholesale: the upstream file also covers Qwen3.5's segment-wise export and a
Qwen3-ASR audio-embedding variant this harness has no use for.

## A real, verified torch/onnx-export compatibility bug

The upstream `causal_llm_to_onnx` calls plain ``torch.onnx.export(...,
dynamic_axes=...)`` with no `dynamo=` argument -- correct for the
TorchScript-based exporter that used to be `torch.onnx.export`'s default.
Reproduced directly: on a stock ``pip install torch`` today (verified
against 2.14.0), that call raises deep inside
``torch/onnx/_internal/exporter/_dynamic_shapes.py``
(``ValueError: treespec.unflatten(leaves): ...``) -- PyTorch's own
deprecation notice explains why: "Starting in PyTorch 2.9, the new
torch.export-based ONNX exporter has become the default," and that new
default does not accept a plain `dynamic_axes` dict the way the legacy
exporter did. :func:`export_causal_lm_to_onnx` passes ``dynamo=False``
explicitly to keep using the legacy exporter these dummy-tensor-traced
causal-LM graphs were written for.
"""

from __future__ import annotations

import json
import pickle
from typing import Optional

import torch


def _update_config(config, attr_names, value) -> None:
    """Recursively set ``attr_names`` on ``config`` and any nested
    ``PretrainedConfig`` (e.g. a `text_config` on a VLM wrapper config)."""
    from transformers import PretrainedConfig

    for attr in dir(config):
        if attr in attr_names:
            setattr(config, attr, value)
        elif isinstance(getattr(config, attr), PretrainedConfig):
            _update_config(getattr(config, attr), attr_names, value)


def load_causal_lm(model_path: str, dtype: torch.dtype = torch.float32):
    """Load an ``AutoModelForCausalLM`` with ``use_cache`` forced off.

    `load_llm()`'s ONNX is a single-shot forward pass with no KV-cache
    output (the KV-cache handling is RKNN3's own, added back by
    `load_llm()`/`rknn.kvcache_controller` at conversion/inference time) --
    leaving `use_cache=True` makes the traced graph return a `DynamicCache`
    object, which JIT tracing cannot flatten (confirmed by direct
    reproduction: ``RuntimeError: ... received an input of unsupported
    type: DynamicCache``).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _update_config(config, ["use_cache"], False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, config=config
    )
    model.eval()
    return model


def export_causal_lm_to_onnx(
    model, onnx_path: str, prompt_size: int = 64, dynamic_shape: bool = True
) -> None:
    """Trace ``model`` (a plain, single-segment ``AutoModelForCausalLM``,
    `use_cache` already disabled -- see :func:`load_causal_lm`) to ONNX with
    the ``input_ids``/``attention_mask``/``position_ids``/
    ``num_logits_to_keep`` input signature `load_llm()` expects."""
    in_len = prompt_size
    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask = torch.ones((1, in_len), dtype=torch.float)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    inputs = (dummy_input, attention_mask, position_ids)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    dynamic_axes = {}
    if dynamic_shape:
        dynamic_axes.update(
            {
                "input_ids": {1: "sequence"},
                "attention_mask": {1: "sequence"},
                "position_ids": {1: "sequence"},
            }
        )

    # Only keep the last token's logits -- matches load_llm()'s expected
    # decode-one-token-at-a-time shape and cuts export/compile cost.
    forward_func = model.forward
    while hasattr(forward_func, "__wrapped__"):
        forward_func = forward_func.__wrapped__
    logit_keep_key = None
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in forward_func.__code__.co_varnames:
            logit_keep_key = key
            break
    if logit_keep_key is not None:
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        insert_nones = [None] * (
            forward_func.__code__.co_varnames.index(logit_keep_key) - len(inputs) - 1
        )
        inputs = (*inputs, *insert_nones, num_logits_to_keep)
        input_names.append("num_logits_to_keep")

    output_names = ["output"]

    # See this module's docstring: dynamo=False keeps the legacy,
    # dynamic_axes-compatible TorchScript-based exporter on torch>=2.9.
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )


def export_embed_weight(weight: torch.Tensor, embed_path: str) -> None:
    """Dump the embedding table as raw float16 -- `load_llm()` strips the
    embedding-lookup node from the graph and expects host-computed
    ``input_embeds`` at inference time instead of ``input_ids`` (confirmed:
    `rknn3-toolkit`'s own `qwen2_5` example loads this file and indexes it on
    the host rather than ever passing token ids to `rknn.inference()`)."""
    weight_fp16 = weight.detach().cpu().to(torch.float16).numpy()
    with open(embed_path, "wb") as f:
        weight_fp16.tofile(f)


def _split_chat_template_prompt(chat_template: str, chat_context: dict, prompt: str):
    from jinja2 import Template

    rendered = Template(chat_template).render(**chat_context)
    prompt_prefix, prompt_postfix = rendered.split(prompt)
    system_prompt = ""
    if "system" in prompt_prefix:
        sys_idx = prompt_prefix.find("system")
        start_str = prompt_prefix[:sys_idx]
        second_start_idx = prompt_prefix.find(start_str, sys_idx)
        system_prompt = prompt_prefix[:second_start_idx]
        prompt_prefix = prompt_prefix[second_start_idx:]
    return system_prompt, prompt_prefix, prompt_postfix


def export_llm_config(
    model_path: str,
    config_path: str,
    prompt: str = "RKLLM",
    user_config: Optional[dict] = None,
) -> None:
    """Write the ``.config.pkl`` sidecar `load_llm()` reads alongside the
    ONNX file -- chat-template fragments plus `vocab_size`/`hidden_size`/the
    full HF config as JSON. Faithful port of `rknn3-model-zoo`'s
    `export_llm_config`, minus multi-modal (`text_config`) handling this
    harness's plain causal-LM path doesn't need."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    chat_context = {
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    }
    if tokenizer.chat_template is not None:
        try:
            system_prompt, prompt_prefix, prompt_postfix = _split_chat_template_prompt(
                tokenizer.chat_template, chat_context, prompt
            )
        except Exception:
            system_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            )
            prompt_prefix = "<|im_start|>user\n"
            prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"
        chat_template = tokenizer.chat_template
    else:
        system_prompt, prompt_prefix, prompt_postfix, chat_template = "", "", "", ""

    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    hf_config_json = json.dumps(config.to_dict(), default=str)
    if user_config is not None:
        hf_config_json = json.dumps({**config.to_dict(), **user_config}, default=str)

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "hf_config_json": hf_config_json,
    }
    if hasattr(config, "quantization_config"):
        llm_config["q_params"] = {
            "bits": config.quantization_config["bits"],
            "sym": config.quantization_config["sym"],
            "group_size": config.quantization_config["group_size"],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)
