#!/usr/bin/env python3
"""Export a Hugging Face causal LM to a real KV-cache Core ML decoder.

Bridges a "decoder-with-past" ONNX export (produced by `optimum-onnx`,
https://github.com/huggingface/optimum-onnx) to `onnxsim.export_coreml`. Unlike
a plain static-shape export, this keeps `sequence_length` and
`past_sequence_length` as genuinely dynamic axes all the way through to the
Core ML model (via `onnxsim.export_coreml`'s `dynamic_shapes` argument, see
`onnxsim/coreml_export.py`), so the resulting `.mlpackage` is one model that
supports both:

- **prefill**: one forward pass over the whole prompt (`sequence_length` =
  prompt length, `past_sequence_length` = 0), producing the initial cache.
- **decode**: one forward pass per new token (`sequence_length` = 1),
  consuming and extending the growing cache from the previous step.

This is a real, O(1)-per-decode-step KV cache -- each step only computes the
new token's attention over the accumulated `present_*` cache from every prior
step, instead of reprocessing the whole context window every time.
`run_llm_decode_benchmark.py` is the harness that actually runs this model
and carries the cache across steps.

Only `batch_size` is pinned to a concrete value (1); `sequence_length`,
`past_sequence_length`, and the composite `past_sequence_length +
sequence_length` axis ONNX exporters emit for the attention mask are left
dynamic, bounded by `--max-context-length` (the largest prompt + generated
tokens the exported model will accept).

`optimum`'s tracer cannot trace a `sequence_length` of exactly 1 for
Llama-family models without corrupting the attention mask (see
`onnx_export_from_model`'s "sequence length of 1" check) -- this only affects
which shape is used to *trace* the model, not what shapes the exported graph
accepts at runtime, so this script traces with `sequence_length=2` and lets
`onnxsim.simplify` + the dynamic Core ML export handle the true dynamic
range, `sequence_length=1` decode steps included.

Pipeline: `optimum.exporters.onnx` (decoder-with-past, dynamic axes) -> pin
`batch_size` to 1 -> `onnxsim.simplify` -> `onnxsim.export_coreml` with
`dynamic_shapes` for `sequence_length` / `past_sequence_length` / their sum.

`--dtype` (default `fp32`) controls the precision the ONNX graph itself is
traced and computed in -- separate from Core ML's own output precision,
which defaults to float16 regardless (coremltools' `compute_precision`).
For a multi-billion-parameter model, tracing in float32 means PyTorch and
the in-progress ONNX graph can hold more than one full-size copy of the
model's weights at once (well over the model's own on-disk size); `--dtype
fp16` roughly halves that peak.

`--quantize-weights` (default `none`) applies
`coremltools.optimize.coreml.linear_quantize_weights` to the *converted*
Core ML model, replacing each weight `const` with
`constexpr_affine_dequantize`/`constexpr_blockwise_shift_scale` (int8 or
int4, per-channel, dequantized to float on the fly at compute time). See
README.md's "Theoretical ceiling" section for why this is the lever worth
pulling for *decode* specifically: a single-token decode step is
memory-bandwidth-bound (the compute per token is tiny; the entire weight
set has to be read from DRAM once per token regardless, since nothing
amortizes that read across tokens at batch size 1), so halving weight
bytes is a direct, roughly-proportional lever on decode tok/s -- unlike
Core ML's packed W8A8 *compute* path (int8 weights **and** activations, up
to ~2x compute throughput on the M4 ANE), which this flag does not use:
weight-only quantization keeps activations and compute in float, so it
doesn't touch the compute-bound side of the ceiling at all, only the
bandwidth-bound side that actually matters for decode at these model
sizes. Applied after conversion (not baked into the ONNX graph), so it's
independent of `--dtype`, which only controls tracing/export precision.
`int4` needs `minimum_deployment_target=iOS18` (coremltools refuses lower
targets for 4-bit weights); this script sets that automatically only for
`int4`, since `int8`/`none` already work at whatever target
`onnxsim.export_coreml` otherwise picks. `--minimum-deployment-target` lets a
caller force a higher target regardless of `--quantize-weights` -- needed by
`coreml-integration.yml`'s compute-plan-trace step, since `MLComputePlan`'s
per-operation device/cost APIs return no data for a model built below their
own SDK floor (see `coreml_compute_plan_trace.m`).

`--matmul-to-conv` (default off) targets the *other* side of the ceiling --
compute, not bandwidth -- for the cases where compute still matters (e.g.
prefill's whole-prompt forward pass, or a bigger batch than this pipeline
currently does). It lowers each attention/MLP projection's `MatMul` to a
1x1/pointwise `conv` (`onnxsim.coreml_export`'s `matmul_to_conv`, see
`onnxsim.export_coreml`'s docstring for exactly which shapes qualify): the
Apple Neural Engine's compute array parallelizes over convolution output
channels and its native matmul path measures well below conv1x1's throughput
on the same hardware -- see README.md's "Theoretical ceiling" section for the
measurements. This flag is a translator-level structural change (unlike
`--quantize-weights`, it can in principle affect numeric behavior through a
different-but-equivalent op sequence, though the underlying math is
identical), and it has not been measured on real Core ML yet -- it exists so
`coreml-integration.yml`'s `benchmark-decode-macos` job can actually measure
whether it helps *this* pipeline's shapes (mostly single-token decode, unlike
the deep/wide conv1x1 chains the cited measurements used) before anyone
enables it by default.

`--io-dtype fp16` (default `fp32`) changes the exported model's *interface*
rather than anything inside it: every float input and output is declared
float16, so Core ML stops converting them at the boundary on every call. An ML
Program already computes in float16 (`compute_precision`'s default), so an
fp32 interface only buys a down-conversion on the way in and an up-conversion
on the way out, over twice the bytes. This pipeline is the shape that pays
most for that: the KV cache crosses the boundary twice per decode step (in as
`past_key_values_*`, out as `present_*`) and is by far the largest thing
moving, growing with every token generated. The ANE reverse-engineering work
behind README.md's "Theoretical ceiling" numbers measures direct fp16 I/O at
roughly 37% faster than fp32 I/O over the same buffers
(https://github.com/maderix/ANE, "How It Works" step 3). Unlike
`--quantize-weights` it changes no stored value and unlike `--matmul-to-conv`
it changes no op, so it costs nothing in accuracy relative to the default --
the computation was fp16 on both sides of the flag; only whether the caller
sees the fp16 value directly or an upcast copy of it changes.
`run_llm_decode_benchmark.py` and `check_decode_parity.py` both read their
feed dtypes off the model's own spec, so they work against either interface
unchanged.

Usage:
    python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \\
        --max-context-length 512 --output smollm2.mlpackage
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import onnx


def _fix_batch_size(model: onnx.ModelProto, batch_size: int) -> None:
    """Pin the `batch_size` dim_param to a concrete value everywhere it appears.

    `sequence_length`, `past_sequence_length`, and the composite
    `past_sequence_length + sequence_length` dim_params are deliberately left
    alone -- those are the axes this script's whole point is to keep dynamic.
    """
    for vi in list(model.graph.input) + list(model.graph.output):
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_param") and d.dim_param == "batch_size":
                d.Clear()
                d.dim_value = batch_size


def export_llm_to_coreml(
    model_id: str,
    output_path: str,
    *,
    max_context_length: int = 512,
    opset: int = 17,
    convert_to: str = "mlprogram",
    dtype: str = "fp32",
    quantize_weights: str = "none",
    matmul_to_conv: bool = False,
    minimum_deployment_target: str | None = None,
    io_dtype: str = "fp32",
) -> None:
    from optimum.exporters.onnx import main_export

    with tempfile.TemporaryDirectory(prefix="onnxsim_llm_export_") as tmpdir:
        print(
            f"Exporting {model_id!r} to ONNX (decoder-with-past, dynamic shapes, "
            f"opset {opset}, dtype {dtype})...",
            flush=True,
        )
        main_export(
            model_id,
            output=tmpdir,
            task="text-generation-with-past",
            opset=opset,
            batch_size=1,
            # Only used to trace the model; the exported graph keeps
            # sequence_length/past_sequence_length dynamic regardless (see the
            # module docstring for why 2, not 1).
            sequence_length=2,
            # optimum's own PyTorch-vs-ONNX-Runtime validation pass keeps both a
            # full copy of the model and an onnxruntime session resident at once,
            # roughly doubling peak memory during export -- prohibitive for a
            # multi-billion-parameter model. onnxsim's own onnx.checker.check_model
            # call below and the Core ML conversion that follows are the
            # correctness checks this pipeline actually relies on.
            do_validation=False,
            dtype=dtype,
        )

        onnx_path = Path(tmpdir) / "model.onnx"

        print(
            "Pinning batch_size=1 (sequence_length/past_sequence_length stay dynamic)...",
            flush=True,
        )
        # `_fix_batch_size` only touches declared shape dim fields, never tensor
        # data, so this whole step never needs the model's weights loaded into
        # memory -- `load_external_data=False` leaves each initializer as just its
        # on-disk location, and re-saving keeps those references intact (same
        # directory, untouched external-data files).
        fixed_model = onnx.load(str(onnx_path), load_external_data=False)
        _fix_batch_size(fixed_model, batch_size=1)
        onnx.save(fixed_model, str(onnx_path))
        num_nodes_before = len(fixed_model.graph.node)
        del fixed_model

        # A path-based check (unlike passing a ModelProto) uses onnx's C++
        # file checker directly, with no in-memory protobuf-serialization size
        # limit to trip over on a multi-billion-parameter model.
        onnx.checker.check_model(str(onnx_path))

        print("Simplifying with onnxsim...", flush=True)
        import onnxsim

        simplified_path = Path(tmpdir) / "model_simplified.onnx"
        # Passing a *path* (not a ModelProto) with check_n=0 engages onnxsim's
        # C++ path-to-path fast route, which needs only ~1x the model's size in
        # peak memory (see bench/RESULTS_synthetic_decoder_oom.md) instead of the
        # 2+x an in-memory ModelProto round trip costs.
        _, ok = onnxsim.simplify(
            str(onnx_path), output_path=str(simplified_path), check_n=0
        )
        if not ok:
            raise RuntimeError("onnxsim.simplify() reported failure")

        # Only now load real tensor values: Core ML conversion needs them (MIL
        # constants), everything before this point only needed shapes.
        simplified = onnx.load(str(simplified_path))
        print(f"  {num_nodes_before} -> {len(simplified.graph.node)} nodes", flush=True)

        print(f"Converting to Core ML ({convert_to}), dynamic KV cache...", flush=True)
        # Captured (not saved directly to output_path) so --quantize-weights can
        # compress it in place first; export_coreml() returns the MLModel either
        # way (see its docstring), so this doesn't change behavior when
        # quantize_weights="none".
        convert_kwargs = {
            "convert_to": convert_to,
            "dynamic_shapes": {
                "sequence_length": (1, 1, max_context_length),
                "past_sequence_length": (0, 0, max_context_length - 1),
                "past_sequence_length + sequence_length": (1, 1, max_context_length),
            },
            "matmul_to_conv": matmul_to_conv,
            "io_dtype": io_dtype,
        }
        if minimum_deployment_target is not None:
            convert_kwargs["minimum_deployment_target"] = minimum_deployment_target
        elif quantize_weights == "int4":
            # coremltools' linear_quantize_weights refuses int4 below iOS18
            # ("The 4-bit quantization is supported since iOS18") -- int8 and
            # "none" work fine at whatever target onnxsim.export_coreml picks by
            # default (already high enough for the dynamic-shape RangeDim inputs
            # this export needs), so this bump is scoped to int4 only.
            convert_kwargs["minimum_deployment_target"] = "iOS18"
        mlmodel = onnxsim.export_coreml(simplified, **convert_kwargs)

        if quantize_weights != "none":
            print(
                f"Quantizing weights ({quantize_weights}, per-channel)...", flush=True
            )
            import coremltools.optimize.coreml as cto

            config = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype=quantize_weights,
                    granularity="per_channel",
                )
            )
            mlmodel = cto.linear_quantize_weights(mlmodel, config)

        mlmodel.save(output_path)

        # Carry the tokenizer along so run_llm_decode_benchmark.py can load
        # everything it needs from `output_path`'s sibling files.
        out_dir = Path(output_path).parent
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            # Recent transformers releases store a chat model's chat template
            # as this standalone file rather than (or in addition to)
            # tokenizer_config.json's own inline "chat_template" key --
            # dropping it here left AutoTokenizer.from_pretrained(out_dir)
            # loading successfully but with no chat template at all, only
            # surfacing later as "tokenizer.chat_template is not set" from
            # lm_eval_coreml_adapter.py's apply_chat_template (used by
            # run_quality_eval.py --apply-chat-template, not by
            # run_llm_decode_benchmark.py/check_decode_parity.py, which
            # build prompts without a chat template).
            "chat_template.jinja",
        ):
            src = Path(tmpdir) / name
            if src.is_file():
                shutil.copy(src, out_dir / name)

    print(f"Wrote {output_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "model_id", help="Hugging Face model id or local path (a causal LM)"
    )
    ap.add_argument(
        "--output", default="model.mlpackage", help="Output .mlpackage path"
    )
    ap.add_argument(
        "--max-context-length",
        type=int,
        default=512,
        help="Upper bound on prompt + generated tokens the exported model will accept "
        "(default: 512). Unlike a fixed-context recompute model, this only bounds the "
        "KV cache's Core ML flexible-shape range -- actual per-step compute scales with "
        "how much of that range is in use, not the bound itself.",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--format",
        choices=["mlprogram", "neuralnetwork"],
        default="mlprogram",
        dest="convert_to",
    )
    ap.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision to trace and export the ONNX graph in (default: fp32). Use "
        "fp16 for multi-billion-parameter models where tracing in float32 risks "
        "running out of memory (see the module docstring).",
    )
    ap.add_argument(
        "--quantize-weights",
        choices=["none", "int8", "int4"],
        default="none",
        help="Weight-only post-conversion quantization (default: none). Reduces "
        "weight bytes read from DRAM per decode step -- the lever that matters for "
        "decode throughput at these model sizes, not compute precision. See the "
        "module docstring and README.md's 'Theoretical ceiling' section.",
    )
    ap.add_argument(
        "--matmul-to-conv",
        action="store_true",
        help="Lower each attention/MLP projection's MatMul to a 1x1/pointwise conv "
        "instead of MIL's native matmul (default: off). The M4 ANE's compute array "
        "parallelizes over convolution output channels; see README.md's 'Theoretical "
        "ceiling' section for the measurements this is based on. Opt-in and "
        "unvalidated on real hardware -- benchmark before trusting it.",
    )
    ap.add_argument(
        "--io-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Dtype of the exported model's float inputs and outputs (default: "
        "fp32). 'fp16' matches what an ML Program already computes in, so Core ML "
        "stops converting every float tensor -- the whole KV cache, twice per "
        "decode step -- at the boundary. See the module docstring and README.md's "
        "'fp16 model interface' section.",
    )
    ap.add_argument(
        "--minimum-deployment-target",
        default=None,
        help="Force a minimum Core ML deployment target (e.g. 'iOS18'), overriding "
        "whatever onnxsim.export_coreml would otherwise pick (or the automatic iOS18 "
        "bump --quantize-weights int4 applies). Needed for e.g. MLComputePlan's "
        "per-operation compute-device/cost APIs, which return no data for models "
        "built below their SDK floor -- see coreml_compute_plan_trace.m.",
    )
    args = ap.parse_args()

    export_llm_to_coreml(
        args.model_id,
        args.output,
        max_context_length=args.max_context_length,
        opset=args.opset,
        convert_to=args.convert_to,
        dtype=args.dtype,
        quantize_weights=args.quantize_weights,
        matmul_to_conv=args.matmul_to_conv,
        minimum_deployment_target=args.minimum_deployment_target,
        io_dtype=args.io_dtype,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
