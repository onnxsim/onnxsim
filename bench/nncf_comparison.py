#!/usr/bin/env python3
"""Head-to-head: onnxsim vs. Intel NNCF on INT4 weight-only compression.

``docs/nncf-comparison-future-work.md`` records an *architectural* comparison
of onnxsim's quantization series against `NNCF
<https://github.com/openvinotoolkit/nncf>`_, and lists as candidate future
work item 3 the thing that comparison could not answer: "no benchmark in this
repo actually runs the same model through both onnxsim and NNCF and compares
accuracy/latency/size." This script is that benchmark.

The two tools' surfaces overlap most directly on **INT4 weight-only
compression**, so that is what is measured here: onnxsim's
:func:`onnxsim.quantize_weight_only_int4` (optionally refined by
:func:`onnxsim.apply_gptq` / :func:`onnxsim.apply_awq`, reached through
:class:`onnxsim.QuantizationConfig`'s own composable flags) against NNCF's
``compress_weights(mode=INT4_SYM, ...)``.

Two semantic differences have to be neutralized before any number here means
anything; both were established empirically against NNCF 3.3.0 while writing
this script, not assumed:

1. **NNCF keeps the last layer at INT8 by default.** NNCF classifies a
   model's final MatMul as a non-"ratio-defining" layer (the ``lm_head``
   convention) and leaves it at ``int8_asym`` per-channel unless
   ``all_layers=True`` is passed. onnxsim's ``quantize_weight_only_int4``
   quantizes *every* eligible layer unconditionally. Comparing NNCF's
   default against onnxsim's therefore compares INT4-plus-an-INT8-layer
   against INT4-everywhere -- a size and accuracy difference that has
   nothing to do with either tool's quantization quality. This script
   reports the ``all_layers=True`` configuration as the apples-to-apples
   one, and NNCF's own default alongside it (clearly labeled), since the
   default is what an NNCF user actually gets.
2. **Group size.** onnxsim's INT4 pass is fixed at ``block_size=32`` (see
   ``onnxsim/passes/weight_only_quantize_int4_matmul.h``'s ``kBlockSize``);
   NNCF's ``group_size`` defaults to 128. They are compared at a matched
   group size of 32, with NNCF's own 128 default reported separately.

With ``all_layers=True`` and ``group_size=32``, NNCF's ONNX backend emits
exactly the layout onnxsim's own pass does -- INT4 codes ``[K, N]`` plus a
float32 scale ``[K / 32, N]`` feeding a ``DequantizeLinear`` -- so the
comparison really is like-for-like.

**What is and isn't measured.** Accuracy (relative L2 of each quantized
model's output against the float model's, on shared random evaluation data)
and size (bytes of initializers actually referenced by a node -- the same
definition :func:`onnxsim.accuracy.recommend_quantization` uses internally,
so an unpruned dead float weight left behind by a quantizer is not counted
as if it shipped) and quantization wall-clock. **Inference latency is not
measured**: both tools emit the same ``DequantizeLinear``-based graph shape
here, so latency would be measuring onnxruntime's kernel, not either
quantizer, and on a synthetic model at that.

Usage::

    python bench/nncf_comparison.py                 # synthetic MLP stack
    python bench/nncf_comparison.py --layers 8 --hidden 512
    python bench/nncf_comparison.py model.onnx      # a real model

Dependencies: ``onnxsim`` (built, i.e. with its compiled extension --
``quantize_weight_only_int4`` is a C++ pass), ``nncf``, ``onnx``,
``onnxruntime``, ``numpy``. NNCF configurations are skipped with a printed
note if ``nncf`` is not importable, so the onnxsim half still runs.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

BLOCK_SIZE = 32


def _synthetic_mlp(
    layers: int = 4, hidden: int = 256, in_dim: int = 256, seed: int = 0
) -> onnx.ModelProto:
    """A plain stack of ``MatMul``s -- no activations between them, so every
    layer is INT4-eligible for both tools and the measured error is purely
    quantization error rather than a nonlinearity's own conditioning.
    """
    rng = np.random.default_rng(seed)
    inits = []
    nodes = []
    prev = "X"
    dim = in_dim
    for i in range(layers):
        out_dim = hidden
        w = (rng.standard_normal((dim, out_dim)) / np.sqrt(dim)).astype(np.float32)
        inits.append(onnx.numpy_helper.from_array(w, f"W{i}"))
        out = "Y" if i == layers - 1 else f"H{i}"
        nodes.append(onnx.helper.make_node("MatMul", [prev, f"W{i}"], [out]))
        prev, dim = out, out_dim
    graph = onnx.helper.make_graph(
        nodes,
        "mlp",
        [
            onnx.helper.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, ["b", in_dim]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, ["b", hidden]
            )
        ],
        inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )
    model.ir_version = 10
    return model


def _referenced_initializer_bytes(model: onnx.ModelProto) -> int:
    """Bytes of initializers actually referenced by some node input.

    Matches :func:`onnxsim.accuracy._initializer_nbytes`: onnxsim's
    ``quantize_*`` passes rewrite a node to point at new (smaller) quantized
    initializers but do not themselves prune the now-dead original float
    weight -- counting unreferenced initializers would make a successful
    quantization look like it grew the model.
    """
    referenced = {name for n in model.graph.node for name in n.input}
    total = 0
    for t in model.graph.initializer:
        if t.name not in referenced:
            continue
        total += (
            len(t.raw_data) if t.HasField("raw_data") else len(t.SerializeToString())
        )
    return total


def _weight_dtypes(model: onnx.ModelProto) -> Dict[str, int]:
    """Referenced-initializer element-type histogram, as ``{type name: count}``
    -- how the "NNCF left the last layer at INT8" effect shows up concretely.
    """
    referenced = {name for n in model.graph.node for name in n.input}
    hist: Dict[str, int] = {}
    for t in model.graph.initializer:
        if t.name not in referenced:
            continue
        name = onnx.TensorProto.DataType.Name(t.data_type)
        hist[name] = hist.get(name, 0) + 1
    return hist


def _run(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _rel_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(actual, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


# --------------------------------------------------------------------------- #
# The two tools' configurations
# --------------------------------------------------------------------------- #


def _onnxsim_configs(
    calibration: Sequence[Dict[str, np.ndarray]],
) -> List[Tuple[str, Callable[[onnx.ModelProto], onnx.ModelProto]]]:
    import onnxsim

    def rtn(m):
        return onnxsim.quantize_weight_only_int4(m)

    def gptq(m):
        return onnxsim.quantize(
            m,
            onnxsim.QuantizationConfig(
                scheme="weight_only",
                dtype="int4",
                gptq=True,
                calibration_data=list(calibration),
            ),
        )

    def awq(m):
        return onnxsim.quantize(
            m,
            onnxsim.QuantizationConfig(
                scheme="weight_only",
                dtype="int4",
                awq=True,
                calibration_data=list(calibration),
            ),
        )

    def awq_gptq(m):
        return onnxsim.quantize(
            m,
            onnxsim.QuantizationConfig(
                scheme="weight_only",
                dtype="int4",
                awq=True,
                gptq=True,
                calibration_data=list(calibration),
            ),
        )

    return [
        ("onnxsim int4 (RTN)", rtn),
        ("onnxsim int4 + GPTQ", gptq),
        ("onnxsim int4 + AWQ", awq),
        ("onnxsim int4 + AWQ + GPTQ", awq_gptq),
    ]


def _nncf_configs(
    calibration: Sequence[Dict[str, np.ndarray]],
) -> List[Tuple[str, Callable[[onnx.ModelProto], onnx.ModelProto]]]:
    import nncf

    mode = nncf.CompressWeightsMode.INT4_SYM

    def dataset():
        # NNCF's ONNX backend takes the same {input_name: array} feed dicts
        # onnxruntime does.
        return nncf.Dataset(list(calibration))

    def default(m):
        # NNCF exactly as an NNCF user gets it out of the box: group_size 128
        # and the final layer left at int8_asym. Not comparable to onnxsim's
        # own INT4-everywhere -- reported for reference, not as the
        # head-to-head number.
        return nncf.compress_weights(m, mode=mode)

    def matched(m):
        return nncf.compress_weights(
            m, mode=mode, group_size=BLOCK_SIZE, all_layers=True
        )

    def matched_awq(m):
        return nncf.compress_weights(
            m,
            mode=mode,
            group_size=BLOCK_SIZE,
            all_layers=True,
            awq=True,
            dataset=dataset(),
        )

    def matched_se(m):
        return nncf.compress_weights(
            m,
            mode=mode,
            group_size=BLOCK_SIZE,
            all_layers=True,
            scale_estimation=True,
            dataset=dataset(),
        )

    return [
        ("nncf int4_sym (NNCF defaults)", default),
        ("nncf int4_sym all_layers g32", matched),
        ("nncf int4_sym g32 + AWQ", matched_awq),
        ("nncf int4_sym g32 + ScaleEstimation", matched_se),
    ]


# --------------------------------------------------------------------------- #


def compare(model: onnx.ModelProto, num_eval: int = 32, seed: int = 0) -> None:
    in_name = model.graph.input[0].name
    in_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    rng = np.random.default_rng(seed)
    eval_x = {in_name: rng.standard_normal((num_eval, in_dim)).astype(np.float32)}
    calibration = [
        {in_name: rng.standard_normal((16, in_dim)).astype(np.float32)}
        for _ in range(4)
    ]

    float_y = _run(model, eval_x)
    float_bytes = _referenced_initializer_bytes(model)

    configs: List[Tuple[str, Callable[[onnx.ModelProto], onnx.ModelProto]]] = []
    try:
        configs += _onnxsim_configs(calibration)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        print(f"! skipping onnxsim configurations: {exc}")
    try:
        configs += _nncf_configs(calibration)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        print(f"! skipping NNCF configurations (pip install nncf): {exc}")

    print(f"float model: {float_bytes / 1024:.1f} KiB of referenced weights\n")
    header = f"{'configuration':<34}{'rel L2':>10}{'KiB':>10}{'x smaller':>11}{'quant s':>9}  dtypes"
    print(header)
    print("-" * len(header))

    for name, fn in configs:
        start = time.perf_counter()
        try:
            quantized = fn(model)
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"{name:<34}{'FAILED':>10}  {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - start
        try:
            y = _run(quantized, eval_x)
            err = f"{_rel_l2(float_y, y):.4f}"
        except Exception as exc:  # pragma: no cover - reported, not raised
            err = f"run:{type(exc).__name__}"
        nbytes = _referenced_initializer_bytes(quantized)
        dtypes = ",".join(
            f"{k}x{v}" for k, v in sorted(_weight_dtypes(quantized).items())
        )
        print(
            f"{name:<34}{err:>10}{nbytes / 1024:>10.1f}"
            f"{float_bytes / max(nbytes, 1):>11.2f}{elapsed:>9.2f}  {dtypes}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "model", nargs="?", help="path to a .onnx model (default: synthetic)"
    )
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-dim", type=int, default=256)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model and os.path.exists(args.model):
        model = onnx.load(args.model)
        print(f"model: {args.model}")
    else:
        model = _synthetic_mlp(
            layers=args.layers, hidden=args.hidden, in_dim=args.in_dim, seed=args.seed
        )
        print(
            f"model: synthetic MLP, {args.layers} layers, "
            f"in_dim={args.in_dim}, hidden={args.hidden}"
        )
    compare(model, num_eval=args.eval_samples, seed=args.seed)


if __name__ == "__main__":
    main()
