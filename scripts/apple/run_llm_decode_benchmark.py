#!/usr/bin/env python3
"""Greedy-decode a prompt through a Core ML LLM and measure decode throughput.

The counterpart to `export_llm_to_coreml.py`: loads the `.mlpackage` that
script produced (plus the tokenizer files it copied alongside it), greedily
generates tokens, and reports decode tokens/second and peak resident memory --
the same two axes DeviceMark's methodology measures on-device (see
https://devicemark.github.io/methodology.html): wall-clock decode time on the
actual runtime, RSS sampled during generation, prefill excluded from the
decode-tok/s window. Not the same *protocol*, though: DeviceMark measures
warm-state (post-specialization) decode over a 128-token prompt / 256-token
decode, two trials on a settled device, on its own private on-device engine
(not Core ML -- see bench/TODO_quality_retention_eval.md); this script runs a
single short prompt/trial with no disclosed thermal settle, against a real
Core ML `.mlpackage` -- good enough to catch this exporter's own regressions,
not a number comparable to a DeviceMark board row.

This only runs where Core ML actually executes models: macOS (loading the
model calls into Apple's Core ML framework, which isn't part of coremltools
itself -- see `onnxsim/coreml_export.py`'s module docstring). It has nothing
to do with WebGPU or any browser.

Decode strategy: real KV-cache decode. The exported model accepts a growing
`past_key_values_*` cache (see export_llm_to_coreml.py); a single forward
pass processes the whole prompt once (prefill, building the initial cache),
then each new token is generated with its own single-token forward pass that
reuses the accumulated cache instead of reprocessing earlier tokens. Prefill
is timed and reported separately (as "prefill"/time-to-first-token) and
excluded from decode tok/s, matching DeviceMark's decode-only methodology.

Usage:
    python run_llm_decode_benchmark.py smollm2.mlpackage \\
        --prompt "The capital of France is" --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.proto import FeatureTypes_pb2 as _ft

_ARRAY_DTYPE = _ft.ArrayFeatureType.ArrayDataType
_DATATYPE_TO_NUMPY = {
    _ARRAY_DTYPE.FLOAT32: np.float32,
    _ARRAY_DTYPE.FLOAT16: np.float16,
    _ARRAY_DTYPE.INT32: np.int32,
    _ARRAY_DTYPE.DOUBLE: np.float64,
}


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB.

    `ru_maxrss` is bytes on macOS, KiB on Linux; there's no portable unit.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3


class CoreMLDecoder:
    """Wraps a real growing-KV-cache Core ML decoder (see export_llm_to_coreml.py)
    for greedy generation.

    `predict()` is called once for the whole prompt (prefill) and then once per
    generated token (decode), carrying the `present_*` outputs of each call
    forward as the `past_key_values_*` inputs of the next -- so each decode step
    only pays for the one new token, not the whole context.
    """

    def __init__(self, mlpackage_path: str, compute_units: str = "ALL"):
        self.mlmodel = ct.models.MLModel(
            mlpackage_path, compute_units=getattr(ct.ComputeUnit, compute_units)
        )
        spec = self.mlmodel.get_spec()

        self._input_dtypes = {}
        max_context_length = None
        for inp in spec.description.input:
            arr = inp.type.multiArrayType
            self._input_dtypes[inp.name] = _DATATYPE_TO_NUMPY.get(
                arr.dataType, np.float32
            )
            if inp.name == "attention_mask":
                max_context_length = arr.shapeRange.sizeRanges[-1].upperBound

        if not max_context_length or max_context_length <= 0:
            raise RuntimeError(
                "could not determine a max context length from the model's "
                "'attention_mask' input -- was this exported without "
                "dynamic_shapes (see export_llm_to_coreml.py)?"
            )
        self.max_context_length = max_context_length

        # The model's float interface dtype (see export_llm_to_coreml.py's
        # --io-dtype): float32 by default, float16 when the boundary conversion
        # Core ML would otherwise do on every call was exported away. Reported
        # below so a benchmark run's numbers are readable without cross-checking
        # which flags produced the .mlpackage.
        float_dtypes = {
            self._input_dtypes[i.name]
            for i in spec.description.input
            if self._input_dtypes[i.name] in (np.float16, np.float32, np.float64)
        }
        if not float_dtypes:
            self.io_dtype = "none"
        elif len(float_dtypes) == 1:
            self.io_dtype = np.dtype(float_dtypes.pop()).name
        else:
            self.io_dtype = "mixed"

        self._present_names = [
            o.name for o in spec.description.output if o.name.startswith("present_")
        ]
        (self._logits_name,) = [
            o.name for o in spec.description.output if o.name.startswith("logits")
        ]
        self._empty_past_shapes = {}
        for inp in spec.description.input:
            if inp.name.startswith("past_key_values_"):
                shape = list(inp.type.multiArrayType.shape)
                shape[2] = 0  # past_sequence_length: empty cache
                self._empty_past_shapes[inp.name] = shape

        self.prefill_seconds = 0.0
        self.reset()

    def reset(self) -> None:
        """Clear the KV cache. Call before starting a new generation."""
        self._cache = {
            name: np.zeros(shape, dtype=self._input_dtypes[name])
            for name, shape in self._empty_past_shapes.items()
        }
        self._cache_len = 0

    def _step(self, token_ids: list[int]) -> np.ndarray:
        """Run one forward pass over `token_ids` (the whole prompt for prefill, or
        a single new token for a decode step), extending the cache in place.
        Returns the logits at the last position.
        """
        n = len(token_ids)
        input_ids = np.array([token_ids], dtype=self._input_dtypes["input_ids"])
        attention_mask = np.ones(
            (1, self._cache_len + n), dtype=self._input_dtypes["attention_mask"]
        )
        position_ids = np.arange(
            self._cache_len,
            self._cache_len + n,
            dtype=self._input_dtypes["position_ids"],
        )[None, :]
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **self._cache,
        }
        out = self.mlmodel.predict(feeds)
        self._cache = {
            name.replace("present_", "past_key_values_", 1): out[name]
            for name in self._present_names
        }
        self._cache_len += n
        return out[self._logits_name][0, -1]

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None
    ):
        """Prefill on `prompt_ids`, then greedily decode up to `max_new_tokens` more
        tokens, one at a time, reusing the growing KV cache.

        Yields (token_id, seconds_for_this_step) per generated token. The first
        yielded token comes from the prefill pass (its cost is reported separately
        via `self.prefill_seconds`, so its `seconds_for_this_step` is `None`);
        every following token is a single-token decode step. Stops early on
        `eos_token_id`, or once the KV cache reaches this model's max context
        length.
        """
        self.reset()
        if len(prompt_ids) > self.max_context_length:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, exceeding this model's max "
                f"context length of {self.max_context_length} (see "
                "--max-context-length in export_llm_to_coreml.py)"
            )

        t0 = time.perf_counter()
        logits = self._step(prompt_ids)
        self.prefill_seconds = time.perf_counter() - t0

        dt = None
        generated = 0
        while max_new_tokens is None or generated < max_new_tokens:
            next_id = int(np.argmax(logits))
            yield next_id, dt
            generated += 1
            if eos_token_id is not None and next_id == eos_token_id:
                return
            if self._cache_len >= self.max_context_length:
                return

            t0 = time.perf_counter()
            logits = self._step([next_id])
            dt = time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "mlpackage", help="Path to the .mlpackage produced by export_llm_to_coreml.py"
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = ap.parse_args()

    model_dir = Path(args.mlpackage).parent
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_dir} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(
        f"Loading {args.mlpackage} (compute_units={args.compute_units}) ...", flush=True
    )
    decoder = CoreMLDecoder(args.mlpackage, compute_units=args.compute_units)
    print(f"Max context length: {decoder.max_context_length} tokens", flush=True)
    print(f"Model float interface: {decoder.io_dtype}", flush=True)

    prompt_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)", flush=True)

    generated = []
    step_times = []
    for token_id, dt in decoder.generate(
        prompt_ids, args.max_new_tokens, tokenizer.eos_token_id
    ):
        generated.append(token_id)
        if dt is not None:
            step_times.append(dt)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated ({len(generated)} tokens): {text!r}", flush=True)

    print(
        f"\nprefill: {1000 * decoder.prefill_seconds:.1f} ms ({len(prompt_ids)} tokens)",
        flush=True,
    )
    if step_times:
        total = sum(step_times)
        print(f"decode tok/s: {len(step_times) / total:.2f}", flush=True)
        print(
            f"mean decode step latency: {1000 * total / len(step_times):.1f} ms",
            flush=True,
        )
    print(f"peak RSS: {_peak_rss_mb():.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
