#!/usr/bin/env python3
"""Convert a BitNet b1.58 ONNX model into onnxruntime's native low-bit kernels.

The pipeline is::

    export/load .onnx -> onnxsim.simplify -> ternary rewrite -> onnxsim.simplify
                      -> onnxruntime verification

The ``onnxsim`` step before the rewrite folds away the export's rotary and shape
arithmetic, and folds ``Transpose(weight) -> MatMul`` (torch stores
``[out, in]``, ONNX MatMul wants ``[in, out]``) into the plain weight
initializers ternary detection needs -- on an export whose own constant folding
did not already do so, that is the difference between converting nothing and
converting every BitLinear. The step after the rewrite confirms onnxsim still
simplifies the converted graph, and cleans up the quantization arithmetic
introduced by ``--mode int8``.

Usage::

    # Convert an existing BitNet export.
    python convert_bitnet.py bitnet.onnx -o bitnet.ort.onnx

    # Same, using BitNet's own W1.58A8 integer arithmetic.
    python convert_bitnet.py bitnet.onnx -o bitnet.int8.onnx --mode int8

    # No checkpoint at hand: build, export and convert the reference decoder.
    python convert_bitnet.py --demo -o demo.ort.onnx

Requirements::

    pip install onnx onnxruntime onnxsim
    pip install torch          # --demo only

``onnxruntime`` is not optional: it executes the converted graph for the
numerical comparison, and it is the target of the conversion in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bitnet_ort  # noqa: E402


def _model_bytes(model: onnx.ModelProto) -> int:
    return model.ByteSize()


def _simplify(model: onnx.ModelProto, feeds, check: int):
    import onnxsim

    return onnxsim.simplify(model, check_n=check, input_data=feeds if check else None)


def convert_model(
    model: onnx.ModelProto,
    mode: str,
    block_size,
    accuracy_level: int,
    simplify: bool,
    check: int,
    batch: int,
    seq: int,
    vocab,
    benchmark: int = 0,
) -> tuple:
    """Run the full pipeline and return ``(converted_model, report)``."""
    report = {
        "mode": mode,
        "onnx_version": onnx.__version__,
        "nodes_exported": len(model.graph.node),
        "bytes_exported": _model_bytes(model),
    }
    feeds = bitnet_ort.random_feeds(model, batch=batch, seq=seq, vocab=vocab)
    reference = bitnet_ort.run(model, feeds)

    if simplify:
        import onnxsim

        report["onnxsim_version"] = onnxsim.__version__
        model, ok = _simplify(model, feeds, check)
        report["simplify_check_ok"] = bool(ok)
        report["nodes_simplified"] = len(model.graph.node)
    baseline = model

    converted, stats = bitnet_ort.convert(
        model, mode=mode, block_size=block_size, accuracy_level=accuracy_level
    )
    if mode == "nbits":
        report["accuracy_level"] = accuracy_level
    report["bitlinear_converted"] = stats.converted
    report["matmuls_left_float"] = len(stats.skipped)
    report["weight_bytes_before"] = stats.weight_bytes_before
    report["weight_bytes_after"] = stats.weight_bytes_after
    report["weight_compression"] = round(stats.compression, 2)

    # A 2-bit model converts on any onnxruntime but only *loads* on a build with
    # the 2-bit MatMulNBits kernel. Where it is missing, still write the model
    # (it is valid, and the target runtime may differ from this machine) but skip
    # everything that would need to execute it.
    runnable = mode != "nbits" or bitnet_ort.matmul_nbits_supported(2)
    report["runnable_here"] = runnable

    if simplify and stats.converted:
        converted, ok = _simplify(converted, feeds, check if runnable else 0)
        if runnable:
            report["post_simplify_check_ok"] = bool(ok)

    report["nodes_converted"] = len(converted.graph.node)
    report["bytes_converted"] = _model_bytes(converted)
    report["model_compression"] = round(
        report["bytes_exported"] / max(report["bytes_converted"], 1), 2
    )
    if not runnable:
        return converted, report

    actual = bitnet_ort.run(converted, feeds)
    report["max_relative_error"] = float(
        bitnet_ort.max_relative_error(reference, actual)
    )
    # For a causal LM the prediction itself is what has to survive conversion.
    if reference[0].ndim == 3:
        agreement = np.mean(
            np.argmax(reference[0], axis=-1) == np.argmax(actual[0], axis=-1)
        )
        report["argmax_agreement"] = float(agreement)

    if benchmark:
        float_ms = bitnet_ort.benchmark(baseline, feeds, repeats=benchmark)
        converted_ms = bitnet_ort.benchmark(converted, feeds, repeats=benchmark)
        report["float_ms"] = round(float_ms, 2)
        report["converted_ms"] = round(converted_ms, 2)
        report["speedup"] = round(float_ms / converted_ms, 2)
    return converted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="BitNet ONNX model to convert")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="build and export the reference BitNet decoder instead (needs torch)",
    )
    parser.add_argument("-o", "--output", help="where to write the converted model")
    parser.add_argument(
        "--mode",
        choices=bitnet_ort.MODES,
        default="nbits",
        help="nbits: 2-bit MatMulNBits (default). int8: BitNet W1.58A8 MatMulInteger",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="MatMulNBits block size (16/32/64/128/256); default picks the largest that fits",
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=bitnet_ort.ACCURACY_LEVEL_INT8,
        help="MatMulNBits compute type: 4 = int8 (default, BitNet's W1.58A8 and the "
        "only fast 2-bit kernel), 1 = float32 (lossless but ~35x slower)",
    )
    parser.add_argument("--no-simplify", action="store_true", help="skip onnxsim")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="time N single-threaded onnxruntime runs of the float and converted models",
    )
    parser.add_argument(
        "--check",
        type=int,
        default=1,
        help="onnxsim numerical check repetitions (0 disables)",
    )
    parser.add_argument("--batch", type=int, default=1, help="verification batch size")
    parser.add_argument(
        "--seq", type=int, default=32, help="verification sequence length"
    )
    parser.add_argument(
        "--vocab", type=int, default=None, help="override the inferred vocabulary size"
    )
    parser.add_argument(
        "--demo-layers", type=int, default=2, help="--demo: decoder layers"
    )
    parser.add_argument(
        "--demo-hidden", type=int, default=128, help="--demo: hidden size"
    )
    args = parser.parse_args()

    if args.demo == bool(args.model):
        parser.error("pass either a model path or --demo")

    if args.demo:
        import bitnet_model

        config = bitnet_model.BitNetConfig(
            hidden_size=args.demo_hidden,
            intermediate_size=args.demo_hidden * 2,
            num_hidden_layers=args.demo_layers,
            max_position_embeddings=max(128, args.seq),
        )
        path = (args.output or "bitnet_demo.onnx").replace(".onnx", ".export.onnx")
        bitnet_model.export_onnx(bitnet_model.build(config), path, seq=args.seq)
        model = onnx.load(path)
        print(f"exported reference BitNet decoder -> {path}", flush=True)
    else:
        model = onnx.load(args.model)

    converted, report = convert_model(
        model,
        mode=args.mode,
        block_size=args.block_size,
        accuracy_level=args.accuracy_level,
        simplify=not args.no_simplify,
        check=args.check,
        batch=args.batch,
        seq=args.seq,
        vocab=args.vocab,
        benchmark=args.benchmark,
    )

    if args.output:
        onnx.save(converted, args.output)
        report["output"] = args.output

    print(json.dumps(report, indent=2))
    if not report.get("runnable_here", True):
        print(
            "\nThis onnxruntime build has no 2-bit MatMulNBits kernel, so the "
            "converted model was written but not verified or benchmarked here. "
            "Upgrade onnxruntime, or use --mode int8 (standard ops only).",
            file=sys.stderr,
        )
    if not report["bitlinear_converted"]:
        print(
            "\nNo ternary weights found. Is this a BitNet export? Weights must be "
            "dequantized to {-s, 0, +s} float32 initializers (not packed).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
