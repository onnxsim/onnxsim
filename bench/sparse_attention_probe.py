#!/usr/bin/env python3
"""Does a *static* sparse-attention mask buy anything on ONNX Runtime?

This is the reproducible probe behind ``docs/sparse-attention-portability.md``,
which concludes that none of AngelSlim's sparse-attention algorithms port
usefully into onnxsim. The load-bearing claim in that note is empirical -- "a
fixed A-Shape/Tri-Shape mask baked into an ONNX graph does not make ONNX Runtime
do less work" -- so it is worth being able to re-run it rather than trusting a
number in a markdown file, in particular after an ONNX Runtime upgrade (the note
lists "ORT grows a real CPU block-sparse kernel" as one of the two changes that
would flip the sign of this result).

Two probes, both batch=1, causal, no KV cache reuse:

``blockmask``
    One ``com.microsoft::SparseAttention`` node run twice with identical inputs,
    differing only in its CSR block mask: full causal (every causal block
    present) vs A-Shape (sink blocks + local-window blocks). That op's *CUDA*
    kernel really does iterate only the CSR non-zeros; the question this probe
    answers is whether the kernel actually reached on the machine you are on
    does too. If it does, the A-Shape run is several times faster. Requires
    IOBinding with past/present KV bound to one buffer -- the CPU kernel
    hard-fails otherwise.

``maskadd``
    The form onnxsim could actually emit today: a decomposed
    ``MatMul``/``Mul``/``Add``/``Softmax``/``MatMul`` attention subgraph, and
    ``com.microsoft::MultiHeadAttention``, each built twice with a constant
    additive mask -- causal vs A-Shape. Runs the two sessions interleaved so
    ordering/thermal drift cannot masquerade as a difference, and (with
    ``--profile``) reports per-node kernel times, which is where the result is
    clearest: the two matmuls cost the same either way.

A-Shape is MInference's ``a_shape`` variant (StreamingLLM): keep the first
``n_init`` keys (attention sink) plus the last ``n_local`` keys relative to each
query, intersected with the causal mask. It is one of only two AngelSlim
patterns that are position-only rather than activation-dependent, hence the only
one a static graph rewrite could express at all.

Usage:
    python bench/sparse_attention_probe.py blockmask [--seq 4096] [--reps 3]
    python bench/sparse_attention_probe.py maskadd [--seq 4096] [--profile]
    python bench/sparse_attention_probe.py all
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.parser
import onnxruntime as ort

# Defaults chosen to be long enough for the quadratic term to dominate while
# still running in seconds on a 4-core CPU sandbox.
DEFAULT_SEQ = 4096
NUM_HEADS = 8
HEAD_DIM = 64
# AngelSlim's own a_shape defaults are n_init=128 / n_local=3968; n_local is
# scaled down here so the pattern is actually sparse at seq 4096 rather than
# degrading to dense (which it does whenever n_init + n_local >= k_len).
N_INIT = 128
N_LOCAL = 512
BLOCK = 64


def _a_shape_mask(seq):
    """Token-granularity A-Shape keep-mask ``(seq, seq)``.

    Mirrors AngelSlim ``algorithms/minference/reference.py::a_shape_attention``:
    ``(sink | window) & causal``, plus the self-key guard that keeps a row from
    being fully masked (which would make softmax produce NaN).
    """
    qi = np.arange(seq)[:, None]
    ki = np.arange(seq)[None, :]
    causal = ki <= qi
    keep = ((ki < N_INIT) | (ki > qi - N_LOCAL)) & causal
    return keep | (ki == qi)


def _causal_mask(seq):
    qi = np.arange(seq)[:, None]
    ki = np.arange(seq)[None, :]
    return ki <= qi


def _block_masks(seq):
    """Block-granularity causal and A-Shape masks ``(nb, nb)`` at ``BLOCK``."""
    nb = seq // BLOCK
    bi = np.arange(nb)
    causal = bi[:, None] >= bi[None, :]
    sink = -(-N_INIT // BLOCK)
    window = -(-N_LOCAL // BLOCK)
    a_shape = ((bi[None, :] < sink) | (bi[None, :] > bi[:, None] - window)) & causal
    return causal, a_shape


def _csr(mask):
    """Block mask ``(nb, nb)`` -> ORT's ``(1, nb+1)`` / ``(1, nnz)`` CSR pair."""
    rows, cols = [0], []
    for row in mask:
        idx = np.nonzero(row)[0]
        cols.extend(int(j) for j in idx)
        rows.append(len(cols))
    return np.array([rows], np.int32), np.array([cols], np.int32)


def _timed(fn, reps):
    fn()  # warm up: first run pays for arena growth and weight paging
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# probe 1: com.microsoft::SparseAttention with a CSR block mask
# ---------------------------------------------------------------------------
def _build_sparse_attention(path, seq):
    nb = seq // BLOCK
    hidden = NUM_HEADS * HEAD_DIM
    # onnx.parser cannot spell an optional-input gap or a scalar (rank-0)
    # graph input cleanly enough to be readable here, and SparseAttention's
    # signature is nine positional inputs, so this one model is built with
    # onnx.helper rather than the text format CLAUDE.md otherwise prefers.
    node = onnx.helper.make_node(
        "SparseAttention",
        [
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "block_row_indices",
            "block_col_indices",
            "total_sequence_length",
            "key_total_sequence_lengths",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        kv_num_heads=NUM_HEADS,
        sparse_block_size=BLOCK,
    )
    f32, i32 = onnx.TensorProto.FLOAT, onnx.TensorProto.INT32
    vi = onnx.helper.make_tensor_value_info
    graph = onnx.helper.make_graph(
        [node],
        "sparse_attention_probe",
        [
            vi("query", f32, [1, seq, hidden]),
            vi("key", f32, [1, seq, hidden]),
            vi("value", f32, [1, seq, hidden]),
            vi("past_key", f32, [1, NUM_HEADS, seq, HEAD_DIM]),
            vi("past_value", f32, [1, NUM_HEADS, seq, HEAD_DIM]),
            vi("block_row_indices", i32, [1, nb + 1]),
            vi("block_col_indices", i32, [1, "nnz"]),
            vi("total_sequence_length", i32, []),
            vi("key_total_sequence_lengths", i32, [1]),
        ],
        [
            vi("output", f32, [1, seq, hidden]),
            vi("present_key", f32, [1, NUM_HEADS, seq, HEAD_DIM]),
            vi("present_value", f32, [1, NUM_HEADS, seq, HEAD_DIM]),
        ],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 20),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    onnx.save(model, path)


def probe_blockmask(seq, reps, threads):
    print(f"== com.microsoft::SparseAttention, CSR block mask (block={BLOCK})")
    causal, a_shape = _block_masks(seq)
    print(
        f"   seq={seq} blocks={seq // BLOCK}  causal nnz={int(causal.sum())}"
        f"  a-shape nnz={int(a_shape.sum())}"
        f"  ({a_shape.sum() / causal.sum():.1%} of causal)"
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sparse_attention.onnx")
        _build_sparse_attention(path, seq)
        try:
            sess = _session(path, threads)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
            print(f"   no SparseAttention kernel on this build: {exc}")
            return

        rng = np.random.default_rng(0)
        hidden = NUM_HEADS * HEAD_DIM
        qkv = {
            name: rng.standard_normal((1, seq, hidden), dtype=np.float32)
            for name in ("query", "key", "value")
        }

        def run_one(mask, label):
            rows, cols = _csr(mask)
            # The CPU kernel asserts past and present KV share one buffer, so
            # the same OrtValue has to be bound as both input and output.
            zeros = np.zeros((1, NUM_HEADS, seq, HEAD_DIM), np.float32)
            past_k = ort.OrtValue.ortvalue_from_numpy(zeros)
            past_v = ort.OrtValue.ortvalue_from_numpy(zeros.copy())
            out = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((1, seq, hidden), np.float32)
            )
            binding = sess.io_binding()
            for name, arr in list(qkv.items()) + [
                ("block_row_indices", rows),
                ("block_col_indices", cols),
                ("total_sequence_length", np.array(seq, np.int32)),
                ("key_total_sequence_lengths", np.array([seq], np.int32)),
            ]:
                binding.bind_cpu_input(name, arr)
            binding.bind_ortvalue_input("past_key", past_k)
            binding.bind_ortvalue_input("past_value", past_v)
            binding.bind_ortvalue_output("present_key", past_k)
            binding.bind_ortvalue_output("present_value", past_v)
            binding.bind_ortvalue_output("output", out)
            median = _timed(lambda: sess.run_with_iobinding(binding), reps)
            print(f"   {label:22s} median {median * 1e3:8.1f} ms")
            return median, out.numpy().copy()

        t_causal, y_causal = run_one(causal, "causal block mask")
        t_sparse, y_sparse = run_one(a_shape, "A-Shape block mask")
        ratio = causal.sum() / a_shape.sum()
        print(
            f"   speedup {t_causal / t_sparse:.2f}x against a {ratio:.2f}x"
            f" reduction in non-zero blocks"
        )
        print(
            "   max |causal - a_shape| output delta:"
            f" {float(np.abs(y_causal - y_sparse).max()):.4f}"
            " (a sparse mask changes what the model computes)"
        )


# ---------------------------------------------------------------------------
# probe 2: a baked additive mask on graphs onnxsim can actually emit
# ---------------------------------------------------------------------------
_DECOMPOSED = """
<ir_version: 10, opset_import: ["" : 21]>
attn (float[1,{heads},{seq},{dim}] Q, float[1,{heads},{seq},{dim}] K,
      float[1,{heads},{seq},{dim}] V) => (float[1,{heads},{seq},{dim}] out)
{{
  Kt = Transpose<perm=[0,1,3,2]>(K)
  scores = MatMul(Q, Kt)
  scaled = Mul(scores, scale)
  masked = Add(scaled, mask)
  probs = Softmax<axis=-1>(masked)
  out = MatMul(probs, V)
}}
"""

_MHA = """
<ir_version: 10, opset_import: ["" : 21, "com.microsoft" : 1]>
attn (float[1,{seq},{hidden}] Q, float[1,{seq},{hidden}] K,
      float[1,{seq},{hidden}] V) => (float[1,{seq},{hidden}] out)
{{
  out = com.microsoft.MultiHeadAttention<num_heads={heads}>(Q, K, V, , , mask)
}}
"""


def _additive_mask(keep):
    bias = np.zeros((1, 1) + keep.shape, np.float32)
    bias[0, 0][~keep] = -np.inf
    return bias


def _build_decomposed(path, seq, keep):
    model = onnx.parser.parse_model(
        _DECOMPOSED.format(heads=NUM_HEADS, seq=seq, dim=HEAD_DIM)
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(np.float32(HEAD_DIM**-0.5), "scale"),
            onnx.numpy_helper.from_array(_additive_mask(keep), "mask"),
        ]
    )
    onnx.save(model, path)


def _build_mha(path, seq, keep):
    model = onnx.parser.parse_model(
        _MHA.format(seq=seq, hidden=NUM_HEADS * HEAD_DIM, heads=NUM_HEADS)
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(_additive_mask(keep), "mask")
    )
    onnx.save(model, path)


def _profile_by_op(path, feeds, tmp, tag, reps, threads):
    """Per-node kernel time, min over ``reps`` runs (min drops warm-up noise)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.enable_profiling = True
    opts.profile_file_prefix = os.path.join(tmp, f"prof_{tag}")
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    for _ in range(reps):
        sess.run(None, feeds)
    with open(sess.end_profiling()) as handle:
        events = json.load(handle)
    per_op = {}
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            per_op.setdefault(event["args"].get("op_name", "?"), []).append(
                event["dur"]
            )
    return {op: min(durs) / 1000.0 for op, durs in per_op.items()}


def probe_maskadd(seq, reps, threads, profile):
    causal = _causal_mask(seq)
    a_shape = _a_shape_mask(seq)
    print("== baked additive mask (what a static onnxsim rewrite could emit)")
    print(
        f"   seq={seq}  causal keeps {int(causal.sum())}"
        f"  a-shape keeps {int(a_shape.sum())}"
        f"  ({a_shape.sum() / causal.sum():.1%} of causal)"
    )
    rng = np.random.default_rng(0)
    hidden = NUM_HEADS * HEAD_DIM

    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            (
                "decomposed MatMul/Softmax/MatMul",
                _build_decomposed,
                {
                    name: rng.standard_normal(
                        (1, NUM_HEADS, seq, HEAD_DIM), dtype=np.float32
                    )
                    for name in ("Q", "K", "V")
                },
            ),
            (
                "com.microsoft::MultiHeadAttention",
                _build_mha,
                {
                    name: rng.standard_normal((1, seq, hidden), dtype=np.float32)
                    for name in ("Q", "K", "V")
                },
            ),
        ]
        for label, build, feeds in cases:
            paths = {}
            for tag, keep in (("causal", causal), ("ashape", a_shape)):
                paths[tag] = os.path.join(tmp, f"{tag}_{build.__name__}.onnx")
                build(paths[tag], seq, keep)
            try:
                sessions = {t: _session(p, threads) for t, p in paths.items()}
            except Exception as exc:  # noqa: BLE001
                print(f"   {label}: unavailable on this build: {exc}")
                continue
            # Interleaved A/B: alternate the two so drift hits both equally.
            for sess in sessions.values():
                sess.run(None, feeds)
            times = {"causal": [], "ashape": []}
            for _ in range(reps):
                for tag, sess in sessions.items():
                    start = time.perf_counter()
                    sess.run(None, feeds)
                    times[tag].append(time.perf_counter() - start)
            t_causal = float(np.median(times["causal"]))
            t_sparse = float(np.median(times["ashape"]))
            print(f"   {label}")
            print(f"     causal  median {t_causal * 1e3:8.1f} ms")
            print(f"     A-Shape median {t_sparse * 1e3:8.1f} ms")
            print(f"     A-Shape / causal = {t_sparse / t_causal:.3f}x wall time")
            if profile:
                prof = {
                    tag: _profile_by_op(p, feeds, tmp, tag, reps + 4, threads)
                    for tag, p in paths.items()
                }
                ops = sorted(set(prof["causal"]) | set(prof["ashape"]))
                for op in ops:
                    print(
                        f"     {op:14s} causal {prof['causal'].get(op, 0):7.1f} ms"
                        f"   A-Shape {prof['ashape'].get(op, 0):7.1f} ms"
                    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("probe", choices=["blockmask", "maskadd", "all"])
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)

    if args.seq % BLOCK:
        parser.error(f"--seq must be a multiple of the block size {BLOCK}")
    print(
        f"onnxruntime {ort.__version__}  providers"
        f" {ort.get_available_providers()}  threads {args.threads}"
    )
    if args.probe in ("blockmask", "all"):
        probe_blockmask(args.seq, args.reps, args.threads)
    if args.probe in ("maskadd", "all"):
        probe_maskadd(args.seq, args.reps, args.threads, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
