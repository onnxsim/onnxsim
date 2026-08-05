"""Survey: what happens to *fused, compute-dominant* operators (Attention and
friends) when a model goes through onnxsim.

Two questions, measured rather than argued:

1. **Re-fusion** -- after onnxsim rewrites a decomposed attention block, can the
   downstream runtime still recognise it and fuse it back into a single kernel
   (``com.microsoft.Attention`` / ``MultiHeadAttention`` / ``SkipLayerNormalization``
   / ``BiasGelu`` ...)?  If simplification makes the pattern unrecognisable, a
   "simplified" model is *slower*, because attention is where all the time goes.
2. **Preservation** -- if the model already contains fused ops (``ai.onnx``
   ``Attention``, ``com.microsoft.MultiHeadAttention``, ...), does onnxsim keep
   them intact?

For each fixture the script builds these variants and compares them:

===========================  ============================================
variant                      meaning
===========================  ============================================
``raw``                      the exported graph
``sim``                      ``onnxsim.simplify`` with default settings
``sim_nogemm``               ``simplify`` with ``fuse_matmul_add_bias_into_gemm_batched`` skipped
``ort(raw)``                 ORT transformer fusion on the exported graph (the ceiling)
``ort(sim)``                 ORT transformer fusion *after* onnxsim -- the re-fusion question
``ort(sim_nogemm)``          same, with the batched-Gemm rewrite skipped
``sim_safe``/``ort(sim_safe)``  ``--bisect`` only: simplify with the passes that were
                             destroying a fusion skipped
``sim(ort(raw))``            onnxsim on an already-fused model -- the preservation question
===========================  ============================================

and reports, per variant: node counts, the fused-op census, whether ONNX
Runtime's *own* session-level optimizer (``ORT_ENABLE_ALL``, on the selected
execution provider) fuses it, numerical agreement with ``raw``, and latency.

Usage::

    python bench/refusion_survey.py                       # CPU
    python bench/refusion_survey.py --bisect              # + name the guilty passes
    python bench/refusion_survey.py --providers CUDAExecutionProvider CPUExecutionProvider
    python bench/refusion_survey.py --fixtures bert_layer --json out.json

``bench/refusion_gpu_bench.py`` is the GPU-side driver (CUDA EP, fp16, TensorRT).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refusion_fixtures as fixtures  # noqa: E402

# Ops that mean "one kernel does the whole compute-dominant block".
FUSED_OPS = (
    # ai.onnx (opset 23+)
    "Attention",
    "RMSNormalization",
    "RotaryEmbedding",
    "LayerNormalization",
    "Gelu",
    # com.microsoft
    "MultiHeadAttention",
    "GroupQueryAttention",
    "PackedAttention",
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "SimplifiedLayerNormalization",
    "EmbedLayerNormalization",
    "BiasGelu",
    "FastGelu",
    "QuickGelu",
    "BiasSplitGelu",
    "FusedMatMul",
    "Gemm",
)

BATCHED_GEMM_PASS = "fuse_matmul_add_bias_into_gemm_batched"

# The default passes this survey found to be hostile to downstream fusion on
# transformer graphs -- see bench/RESULTS_refusion.md.  ``fuse_consecutive_
# unsqueezes`` merges the attention mask's Unsqueeze pair, which is the exact
# node sequence ORT's FusionAttention walks; the batched-Gemm rewrite turns the
# Q/K/V ``MatMul+Add`` into ``Reshape+Gemm`` (and, combined with
# ``eliminate_consecutive_idempotent_ops``, currently produces a model that does
# not load at all).
FUSION_HOSTILE_PASSES = [
    BATCHED_GEMM_PASS,
    "fuse_consecutive_unsqueezes",
]


def op_counts(model: onnx.ModelProto) -> Dict[str, int]:
    return dict(collections.Counter(n.op_type for n in model.graph.node))


def fused_census(model: onnx.ModelProto) -> Dict[str, int]:
    counts = op_counts(model)
    return {op: counts[op] for op in FUSED_OPS if counts.get(op)}


@contextlib.contextmanager
def _tmp_model(model: onnx.ModelProto):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.onnx")
        onnx.save(model, path)
        yield path


def ensure_opset_imports(model: onnx.ModelProto) -> bool:
    """Add opset imports for domains the graph uses but does not declare.

    ONNX Runtime's transformer optimizer emits ``com.microsoft`` nodes *without*
    adding the matching opset import, which makes the model invalid ONNX -- the
    onnx checker (and therefore ``onnxsim.simplify``) rejects it with
    ``No opset import for domain 'com.microsoft'``.  Returns True if the model
    had to be patched.
    """
    declared = {entry.domain for entry in model.opset_import}
    used = {node.domain for node in model.graph.node if node.domain}
    missing = used - declared
    for domain in sorted(missing):
        model.opset_import.append(onnx.helper.make_opsetid(domain, 1))
    return bool(missing)


def simplify(model: onnx.ModelProto, skipped_optimizers=None):
    import onnxsim

    simplified, ok = onnxsim.simplify(
        model, check_n=0, skipped_optimizers=skipped_optimizers
    )
    return simplified, ok


def default_optimizers() -> List[str]:
    """The onnx-optimizer passes onnxsim runs by default."""
    import onnxsim.onnxsim_cpp2py_export as C

    passes = list(C._list_optimizers())
    if BATCHED_GEMM_PASS not in passes:
        # Opted into explicitly by onnxsim (registered as PassType::Other, so it
        # is not part of onnx-optimizer's default fuse/elimination list).
        passes.append(BATCHED_GEMM_PASS)
    return passes


def find_fusion_breaking_passes(
    model: onnx.ModelProto, model_type: str, num_heads: int, hidden_size: int
) -> Dict[str, Dict[str, int]]:
    """Which optimizer pass costs the model a downstream fusion?

    Runs ``simplify`` once per default pass with only that pass skipped and
    reports the passes whose removal *increases* the number of fused ops ORT can
    recover -- i.e. the passes that were destroying a fusion.
    """

    def loadable(candidate) -> bool:
        import onnxruntime as ort

        try:
            ort.InferenceSession(
                candidate.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception:
            return False
        return True

    def census_after(skip):
        try:
            candidate, _ = simplify(model, skipped_optimizers=skip)
        except Exception:
            return None
        # A simplified model that no longer loads has lost every fusion there is.
        if not loadable(candidate):
            return {}
        try:
            recovered, _ = ort_fuse(candidate, model_type, num_heads, hidden_size)
        except Exception:
            return None
        return fused_census(recovered) if recovered is not None else None

    baseline_fused, _ = ort_fuse(model, model_type, num_heads, hidden_size)
    reference = fused_census(baseline_fused) if baseline_fused is not None else {}

    def total(census):
        # Score only against the fusions the un-simplified model achieved, so
        # incidental extra ops (a Gemm the rewrite introduced) cannot look like
        # a gain.
        return sum(min(count, census.get(op, 0)) for op, count in reference.items())

    current = census_after(None) or {}
    if total(current) >= total(reference):
        return {}

    # Greedy: some fusions are gated on others (attention fusion anchors on a
    # LayerNormalization node), so a single skipped pass may recover nothing
    # until another one is skipped too.  Keep adding the pass with the biggest
    # gain until nothing improves.
    culprits: Dict[str, Dict[str, int]] = {}
    skipped: List[str] = []
    remaining = default_optimizers()
    while total(current) < total(reference):
        best = None
        for pass_name in remaining:
            census = census_after(skipped + [pass_name])
            if census is None or total(census) <= total(current):
                continue
            if best is None or total(census) > total(best[1]):
                best = (pass_name, census)
        if best is None:
            break
        pass_name, census = best
        culprits[pass_name] = {
            op: census[op] - current.get(op, 0)
            for op in census
            if census[op] > current.get(op, 0)
        }
        skipped.append(pass_name)
        remaining = [p for p in remaining if p != pass_name]
        current = census
    return culprits


def ort_fuse(
    model: onnx.ModelProto,
    model_type: str,
    num_heads: int,
    hidden_size: int,
    use_gpu: bool = False,
    float16: bool = False,
) -> Tuple[Optional[onnx.ModelProto], Dict[str, int]]:
    """Run ONNX Runtime's transformer fusion passes (the python ones).

    ``opt_level=0`` disables ORT's C++ graph optimizations so the result
    measures *pattern recognisability* only: every fused op in the output was
    produced by a python pattern matcher walking this exact graph.
    """
    from onnxruntime.transformers import optimizer

    with _tmp_model(model) as path:
        try:
            optimized = optimizer.optimize_model(
                path,
                model_type=model_type,
                num_heads=num_heads,
                hidden_size=hidden_size,
                opt_level=0,
                use_gpu=use_gpu,
            )
        except Exception as exc:  # pragma: no cover - depends on ORT version
            return None, {"error": str(exc)[:200]}
    if float16:
        optimized.convert_float_to_float16(keep_io_types=True)
    return optimized.model, dict(optimized.get_fused_operator_statistics())


def ort_session_ops(model: onnx.ModelProto, providers: Sequence[str]) -> Dict[str, int]:
    """Op census of the model *as ONNX Runtime actually runs it*.

    Runs the session with ``ORT_ENABLE_ALL`` and dumps the optimized graph, so
    the counts include ORT's own C++ fusions (``AttentionFusion``,
    ``SkipLayerNormFusion``, ``GeluFusion``, ...) for the selected execution
    provider.  This is EP-dependent: the CUDA EP fuses more than the CPU EP.
    """
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.onnx")
        dst = os.path.join(d, "opt.onnx")
        onnx.save(model, src)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = dst
        try:
            ort.InferenceSession(src, so, providers=list(providers))
        except Exception as exc:
            return {"error": str(exc)[:200]}
        if not os.path.exists(dst):
            return {}
        return op_counts(onnx.load(dst))


def run(model: onnx.ModelProto, feed, providers: Sequence[str]):
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=list(providers))
    names = {i.name for i in sess.get_inputs()}
    return sess.run(None, {k: v for k, v in feed.items() if k in names})


def latency_ms(
    model: onnx.ModelProto,
    feed,
    providers: Sequence[str],
    warmup: int = 10,
    iters: int = 50,
) -> Optional[float]:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=list(providers)
        )
    except Exception:
        return None
    names = {i.name for i in sess.get_inputs()}
    inputs = {k: v for k, v in feed.items() if k in names}
    dtypes = {i.name: i.type for i in sess.get_inputs()}
    for name, value in list(inputs.items()):
        if "float16" in dtypes.get(name, "") and value.dtype == np.float32:
            inputs[name] = value.astype(np.float16)
    for _ in range(warmup):
        sess.run(None, inputs)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        sess.run(None, inputs)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def max_abs_diff(a, b) -> Optional[float]:
    if a is None or b is None or len(a) != len(b):
        return None
    return max(
        float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for x, y in zip(a, b)
    )


def survey_fixture(
    name: str,
    providers: Sequence[str],
    iters: int,
    do_latency: bool = True,
    float16: bool = False,
    bisect: bool = False,
) -> dict:
    model, feed = fixtures.build(name)
    model_type = fixtures.MODEL_TYPE[name]
    heads = fixtures.NUM_HEADS[name]
    hidden = fixtures.HIDDEN_SIZE[name]
    use_gpu = any("CUDA" in p or "Tensorrt" in p for p in providers)

    variants: "collections.OrderedDict[str, onnx.ModelProto]" = (
        collections.OrderedDict()
    )
    fusion_stats: Dict[str, Dict[str, int]] = {}

    variants["raw"] = model
    sim, _ = simplify(model)
    variants["sim"] = sim
    sim_nogemm, _ = simplify(model, skipped_optimizers=[BATCHED_GEMM_PASS])
    variants["sim_nogemm"] = sim_nogemm

    culprits: Dict[str, Dict[str, int]] = {}
    if bisect:
        culprits = find_fusion_breaking_passes(model, model_type, heads, hidden)
        if culprits:
            safe, _ = simplify(model, skipped_optimizers=sorted(culprits))
            variants["sim_safe"] = safe

    fusion_sources = [
        ("ort(raw)", model),
        ("ort(sim)", sim),
        ("ort(sim_nogemm)", sim_nogemm),
    ]
    if "sim_safe" in variants:
        fusion_sources.append(("ort(sim_safe)", variants["sim_safe"]))

    for label, src in fusion_sources:
        fused, stats = ort_fuse(src, model_type, heads, hidden, use_gpu, float16)
        fusion_stats[label] = stats
        if fused is not None:
            variants[label] = fused

    notes: List[str] = []
    if "ort(raw)" in variants:
        # onnxsim on an already-fused model.  ORT leaves the com.microsoft opset
        # import out of its own output, so try it as produced first and record
        # what happens before patching the model up.
        fused_model = variants["ort(raw)"]
        try:
            resim, _ = simplify(fused_model)
        except Exception as exc:
            notes.append(
                f"simplify(ort(raw)) raised as produced by ORT: {str(exc)[:120]}"
            )
            patched = onnx.ModelProto()
            patched.CopyFrom(fused_model)
            if ensure_opset_imports(patched):
                notes.append("retried after adding the missing opset import")
            try:
                resim, _ = simplify(patched)
            except Exception as exc2:
                notes.append(f"still fails: {str(exc2)[:120]}")
                resim = None
        if resim is not None:
            variants["sim(ort(raw))"] = resim

    reference = run(model, feed, providers)
    rows = []
    for label, variant in variants.items():
        try:
            outputs = run(variant, feed, providers)
            err = max_abs_diff(reference, outputs)
        except Exception as exc:
            outputs, err = None, None
            runnable = str(exc)[:160]
        else:
            runnable = "ok"
        rows.append(
            {
                "variant": label,
                "nodes": len(variant.graph.node),
                "ops": op_counts(variant),
                "fused": fused_census(variant),
                "ort_fusion_stats": fusion_stats.get(label),
                "ort_session_ops": ort_session_ops(variant, providers),
                "runs": runnable,
                "max_abs_diff": err,
                "latency_ms": latency_ms(variant, feed, providers, iters=iters)
                if do_latency
                else None,
            }
        )
    return {
        "fixture": name,
        "providers": list(providers),
        "variants": rows,
        "notes": notes,
        "fusion_breaking_passes": culprits,
    }


def _fmt_counts(counts: Optional[Dict[str, int]], keys=None) -> str:
    if counts is None:
        return "-"
    if "error" in counts:
        return f"error: {counts['error'][:60]}"
    items = [
        (k, v) for k, v in sorted(counts.items()) if v and (keys is None or k in keys)
    ]
    return ", ".join(f"{k}={v}" for k, v in items) or "-"


def print_report(result: dict) -> None:
    print()
    print("=" * 100)
    print(f"fixture: {result['fixture']}   providers: {', '.join(result['providers'])}")
    print("=" * 100)
    header = f"{'variant':<18}{'nodes':>6}  {'lat(ms)':>8}  {'maxdiff':>9}  fused ops in the graph"
    print(header)
    print("-" * 100)
    for row in result["variants"]:
        lat = f"{row['latency_ms']:.3f}" if row["latency_ms"] is not None else "-"
        diff = f"{row['max_abs_diff']:.2e}" if row["max_abs_diff"] is not None else "-"
        print(
            f"{row['variant']:<18}{row['nodes']:>6}  {lat:>8}  {diff:>9}  "
            f"{_fmt_counts(row['fused'])}"
        )
    print()
    print(
        "ONNX Runtime session-level optimization (ORT_ENABLE_ALL) -- fused ops it produces:"
    )
    for row in result["variants"]:
        print(
            f"  {row['variant']:<18}{_fmt_counts({k: v for k, v in (row['ort_session_ops'] or {}).items() if k in FUSED_OPS or k == 'error'})}"
        )
    print()
    print(
        "onnxruntime.transformers fusion statistics (opt_level=0, pattern matching only):"
    )
    for row in result["variants"]:
        if row["ort_fusion_stats"] is not None:
            print(f"  {row['variant']:<18}{_fmt_counts(row['ort_fusion_stats'])}")
    if result.get("fusion_breaking_passes"):
        print("optimizer passes that cost the model a downstream fusion:")
        for pass_name, gained in result["fusion_breaking_passes"].items():
            print(f"  {pass_name:<40}recovers {_fmt_counts(gained)}")
        print()
    for note in result.get("notes", []):
        print(f"  note: {note}")
    failures = [r for r in result["variants"] if r["runs"] != "ok"]
    for row in failures:
        print(f"  !! {row['variant']} does not run: {row['runs']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=list(fixtures.FIXTURES),
        choices=list(fixtures.FIXTURES),
    )
    parser.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"])
    parser.add_argument(
        "--preset", default="tiny", choices=list(fixtures.PRESETS), help="model size"
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument(
        "--bisect",
        action="store_true",
        help="find which optimizer pass destroys a downstream fusion "
        "(re-runs simplify once per default pass)",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="convert the ORT-fused variants to fp16 (GPU tensor cores)",
    )
    parser.add_argument("--json", help="write the full result table to this path")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    fixtures.configure(
        preset=args.preset,
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
    )
    print(f"onnx {onnx.__version__}, onnxruntime {ort.__version__}")
    print(f"available providers: {ort.get_available_providers()}")
    print(f"model size: {fixtures.describe()}")

    results = []
    for name in args.fixtures:
        result = survey_fixture(
            name,
            args.providers,
            args.iters,
            do_latency=not args.no_latency,
            float16=args.float16,
            bisect=args.bisect,
        )
        print_report(result)
        results.append(result)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
