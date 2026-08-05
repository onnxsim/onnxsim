"""GPU leg of the operator re-fusion survey (NVIDIA / CUDA execution provider).

The CPU survey (``bench/refusion_survey.py``) answers *"is the attention pattern
still recognisable after onnxsim?"*.  This script answers the question that
actually matters on an accelerator: **what does it cost in milliseconds?**

It builds the same variants, runs each one on the CUDA execution provider in
fp32 and fp16, and reports:

* median / p90 latency and speed-up versus the un-simplified export,
* the fused-operator census of the graph as *ONNX Runtime runs it* -- the CUDA
  EP's own ``AttentionFusion``/``SkipLayerNormFusion``/``GeluFusion`` run inside
  session creation, so this is what the GPU really executes,
* whether every variant still produces the same numbers.

Tested target: **GeForce RTX 2060** (Turing, sm75, 6 GB).  fp16 matters on this
card: Turing has fp16 tensor cores, and ORT's fused attention kernels
(``TRT fused attention``, ``MemoryEfficientAttention``) only kick in for fp16.

Setup (CUDA 12.x + cuDNN 9 wheels, no system CUDA install needed)::

    python -m venv .venv && . .venv/bin/activate
    pip install "onnxruntime-gpu" onnx numpy sympy coloredlogs packaging
    pip install .            # onnxsim, from the repo root

Run::

    # quick smoke test (tiny model, ~1 minute)
    python bench/refusion_gpu_bench.py --preset tiny

    # the real measurement (BERT-base sized, fp32 + fp16)
    python bench/refusion_gpu_bench.py --preset bert-base --json rtx2060.json

    # add the TensorRT EP too, if the wheel has it
    python bench/refusion_gpu_bench.py --preset bert-base --trt

``--json`` writes every number this prints; that file is the thing worth sharing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refusion_fixtures as fixtures  # noqa: E402
import refusion_survey as survey  # noqa: E402


def gpu_info() -> Dict[str, str]:
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            info["gpu"] = out.stdout.strip()
    except Exception:
        pass
    return info


def to_fp16(model: onnx.ModelProto) -> Optional[onnx.ModelProto]:
    """fp16 cast with fp32 graph I/O (what ORT's transformer optimizer emits).

    ``OnnxModel`` wraps the ModelProto **by reference** and converts it in
    place, so the caller's fp32 model must be copied first -- otherwise the
    fp32 leg silently ends up timing the fp16 graph too.
    """
    try:
        from onnxruntime.transformers.onnx_model import OnnxModel

        clone = onnx.ModelProto()
        clone.CopyFrom(model)
        wrapper = OnnxModel(clone)
        wrapper.convert_float_to_float16(keep_io_types=True)
        return wrapper.model
    except Exception as exc:  # pragma: no cover - version dependent
        print(f"  (fp16 conversion failed: {str(exc)[:120]})")
        return None


def timed(
    model: onnx.ModelProto,
    feed,
    providers: Sequence,
    warmup: int,
    iters: int,
) -> Dict[str, Optional[float]]:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=list(providers)
        )
    except Exception as exc:
        return {"error": str(exc)[:200]}
    used = sess.get_providers()
    names = {i.name for i in sess.get_inputs()}
    types = {i.name: i.type for i in sess.get_inputs()}
    inputs = {}
    for name, value in feed.items():
        if name not in names:
            continue
        if "float16" in types[name] and value.dtype == np.float32:
            value = value.astype(np.float16)
        inputs[name] = value
    for _ in range(warmup):
        sess.run(None, inputs)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        sess.run(None, inputs)
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    outputs = sess.run(None, inputs)
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[int(0.9 * (len(samples) - 1))],
        "min_ms": samples[0],
        "providers": used,
        "_outputs": outputs,
    }


def build_variants(name: str, use_gpu: bool) -> "Dict[str, onnx.ModelProto]":
    model, feed = fixtures.build(name)
    model_type = fixtures.MODEL_TYPE[name]
    heads = fixtures.NUM_HEADS[name]
    hidden = fixtures.HIDDEN_SIZE[name]

    variants = {"raw": model}
    print(f"  simplifying {name} ...", flush=True)
    sim, _ = survey.simplify(model)
    variants["sim"] = sim
    sim_nogemm, _ = survey.simplify(
        model, skipped_optimizers=[survey.BATCHED_GEMM_PASS]
    )
    variants["sim_nogemm"] = sim_nogemm
    # The same simplification with the fusion-hostile passes turned off, i.e.
    # what onnxsim could deliver today with a different default pass set.
    sim_safe, _ = survey.simplify(
        model, skipped_optimizers=survey.FUSION_HOSTILE_PASSES
    )
    variants["sim_safe"] = sim_safe

    print("  running ONNX Runtime transformer fusion ...", flush=True)
    for label, src in (
        ("ort(raw)", model),
        ("ort(sim)", sim),
        ("ort(sim_nogemm)", sim_nogemm),
        ("ort(sim_safe)", sim_safe),
    ):
        fused, stats = survey.ort_fuse(src, model_type, heads, hidden, use_gpu=use_gpu)
        if fused is not None:
            variants[label] = fused
        else:
            print(f"  {label}: fusion failed ({stats.get('error')})")
    return variants, feed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=[
            "bert_layer",
            "llama_layer",
            "fused_contrib_mha",
            "fused_onnx_attention",
        ],
        choices=list(fixtures.FIXTURES),
    )
    parser.add_argument("--preset", default="bert-base", choices=list(fixtures.PRESETS))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-fp16", action="store_true", help="skip the fp16 leg")
    parser.add_argument("--trt", action="store_true", help="also time the TensorRT EP")
    parser.add_argument("--cpu", action="store_true", help="run on CPU (debugging)")
    parser.add_argument("--json", help="write all measurements here")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    available = ort.get_available_providers()
    if args.cpu:
        providers = ["CPUExecutionProvider"]
    else:
        if "CUDAExecutionProvider" not in available:
            print(
                "CUDAExecutionProvider is not available.\n"
                f"available: {available}\n"
                "Install the GPU build:  pip install onnxruntime-gpu\n"
                "(or pass --cpu to run this script on the CPU)"
            )
            return 2
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    use_gpu = "CUDAExecutionProvider" in providers

    fixtures.configure(
        preset=args.preset,
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
    )

    env = gpu_info()
    env.update({"onnx": onnx.__version__, "onnxruntime": ort.__version__})
    print(json.dumps(env, indent=2))
    print(f"model size: {fixtures.describe()}")
    print(f"providers: {providers}")

    results = []
    for name in args.fixtures:
        print(f"\n### {name}")
        variants, feed = build_variants(name, use_gpu)
        reference = None
        rows = []
        legs = [("fp32", providers)]
        if args.trt and "TensorrtExecutionProvider" in available:
            legs.append(("trt", ["TensorrtExecutionProvider"] + providers))
        for label, model in variants.items():
            candidates = [(label, model, "fp32")]
            if not args.no_fp16 and use_gpu:
                half = to_fp16(model)
                if half is not None:
                    candidates.append((f"{label} fp16", half, "fp16"))
            for leg_name, leg_providers in legs:
                for variant_label, variant_model, precision in candidates:
                    tag = (
                        variant_label
                        if leg_name == "fp32"
                        else f"{variant_label} [trt]"
                    )
                    result = timed(
                        variant_model, feed, leg_providers, args.warmup, args.iters
                    )
                    outputs = result.pop("_outputs", None)
                    if reference is None and outputs is not None:
                        reference = outputs
                    diff = survey.max_abs_diff(reference, outputs) if outputs else None
                    session_ops = survey.ort_session_ops(variant_model, leg_providers)
                    rows.append(
                        {
                            "variant": tag,
                            "precision": precision,
                            "ep": leg_name,
                            "nodes": len(variant_model.graph.node),
                            # Guard against a silently-not-converted fp16 leg:
                            # a genuine fp16 graph has fp16 (dtype 10) weights.
                            "fp16_initializers": sum(
                                1
                                for init in variant_model.graph.initializer
                                if init.data_type == onnx.TensorProto.FLOAT16
                            ),
                            "fused": survey.fused_census(variant_model),
                            "session_fused": {
                                k: v
                                for k, v in session_ops.items()
                                if k in survey.FUSED_OPS or k == "error"
                            },
                            "max_abs_diff": diff,
                            **result,
                        }
                    )
                    lat = result.get("median_ms")
                    status = (
                        "%.3f ms" % lat
                        if lat
                        else "does not run: " + result.get("error", "-")[:50]
                    )
                    print(
                        f"  {tag:<24} {status:>16}"
                        f"   {survey._fmt_counts(rows[-1]['session_fused'])}"
                    )
        base = next(
            (
                r["median_ms"]
                for r in rows
                if r["variant"] == "raw" and r.get("median_ms")
            ),
            None,
        )
        if base:
            for row in rows:
                if row.get("median_ms"):
                    row["speedup_vs_raw"] = base / row["median_ms"]
        results.append({"fixture": name, "rows": rows, "env": env})

        print(
            f"\n  {'variant':<24}{'median ms':>10}{'p90 ms':>9}{'speedup':>9}  maxdiff"
        )
        for row in rows:
            if not row.get("median_ms"):
                continue
            print(
                f"  {row['variant']:<24}{row['median_ms']:>10.3f}{row['p90_ms']:>9.3f}"
                f"{row.get('speedup_vs_raw', float('nan')):>9.2f}x"
                f"  {row['max_abs_diff'] if row['max_abs_diff'] is not None else '-'}"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
