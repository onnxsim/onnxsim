"""Build a step segment's subgraph with Pulsar2 at 16-bit MatMul precision.

The step's 8-bit MatMul/Conv/Gemm chains are 2-7% off float on real data
(``docs/axera-step-runner.md``); the same chain built with
``quant.layer_configs`` U16 is about 240x closer, at about 2.5x the device
time. The chain is built on real tensors of the reference batch, so the
result is one axmodel per segment rather than a recalibrated template.
Segment inputs and outputs stay float32, like every other template.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import onnx
from onnx import utils

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pulsar2_docker as pd  # noqa: E402

OPS_16BIT = ("MatMul", "Conv", "Gemm")
PASSIVE_OPS = frozenset({"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Gather", "Slice", "Flatten"})
DEFAULT_IMAGE = "pulsar2:7.0-lite"


def chain_model(
    model: onnx.ModelProto,
    inputs: Sequence[str],
    outputs: Sequence[str],
    split: int = 1,
) -> tuple[onnx.ModelProto, Callable[[np.ndarray], np.ndarray] | None]:
    """The subgraph between ``inputs`` and ``outputs``, legalized the way the
    step is compiled (live-weight Conv/Gemm as MatMul) and in the opset the
    Pulsar2 frontend takes.

    A Transpose/Reshape that ends the chain makes Pulsar2 quantize the whole
    output path to 8 bits even under a 16-bit layer config (the bias Add and
    the output stay uint8). Those trailing ops are cut off and returned as a
    host function of the axmodel's output."""
    import step_calibration

    if len(outputs) != 1:
        raise ValueError("16-bit chains have one output")
    sub = utils.Extractor(model).extract_model(list(inputs), list(outputs))
    if split > 1:
        # a chain too large to compile at the step's batch: every input whose
        # leading dimension is the batch (the first input's, or the largest)
        # shrinks by ``split``; the runner calls the model ``split`` times
        batch = split_batch(sub)
        for vi in sub.graph.input:
            dims = vi.type.tensor_type.shape.dim
            if dims and dims[0].dim_value == batch:
                dims[0].dim_value = batch // split
        del sub.graph.value_info[:]
        del sub.graph.output[:]
        sub.graph.output.extend(
            onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None)
            for o in outputs
        )
    sub = onnx.shape_inference.infer_shapes(sub)
    sub = step_calibration.legalized(sub)
    del sub.opset_import[:]
    sub.opset_import.extend([onnx.helper.make_opsetid("", 13)])
    sub.ir_version = 8
    g = sub.graph
    final_shape = tuple(
        d.dim_value for d in g.output[0].type.tensor_type.shape.dim
    )
    if split > 1:
        # the runner concatenates the chunks' outputs along axis 0 first
        final_shape = (final_shape[0] * split, *final_shape[1:])
    steps: list[tuple[str, object]] = []
    by_out = {o: n for n in g.node for o in n.output}
    tail = g.output[0].name
    while tail in by_out and by_out[tail].op_type in ("Transpose", "Reshape"):
        node = by_out[tail]
        if node.op_type == "Transpose":
            perm = [a.ints for a in node.attribute if a.name == "perm"]
            steps.append(("T", tuple(perm[0]) if perm else None))
        else:
            steps.append(("R", None))
        g.node.remove(node)
        tail = node.input[0]
    if not steps:
        return sub, None
    sub = onnx.shape_inference.infer_shapes(sub)
    g = sub.graph
    vi = next(v for v in g.value_info if v.name == tail)
    del g.output[:]
    g.output.append(vi)
    order = list(reversed(steps))

    def post(y: np.ndarray) -> np.ndarray:
        for kind, perm in order:
            if kind == "T":
                y = np.transpose(y, perm)
        return np.ascontiguousarray(y).reshape(final_shape)

    # the same ops as device models: the chain's output shape, the stripped
    # steps in order, and the final shape (see ``transpose_models``)
    post.pre_shape = tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)  # type: ignore[attr-defined]
    post.steps = order  # type: ignore[attr-defined]
    post.final_shape = final_shape  # type: ignore[attr-defined]
    return sub, post


STEP_BATCH = 16
"""The training step's batch size (a fixed-batch compiled graph)."""


def split_batch(sub: onnx.ModelProto) -> int:
    """The batch a chain is split along: the step's batch, taken from the
    inputs whose leading dimension equals it."""
    return STEP_BATCH


def split_flags(sub_inputs: Sequence[onnx.ValueInfoProto], batch: int) -> list[bool]:
    return [
        bool(vi.type.tensor_type.shape.dim)
        and vi.type.tensor_type.shape.dim[0].dim_value == batch
        for vi in sub_inputs
    ]


def build_chain(
    work: str,
    tag: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
    precision: str = "U16",
    image: str = DEFAULT_IMAGE,
    timeout: int = 1800,
):
    """``pulsar2 build`` of ``sub`` calibrated on ``data`` (four identical
    samples, so MinMax sees exactly the real range)."""
    root = os.path.join(work, tag)
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    onnx.save(sub, os.path.join(root, "t.onnx"))
    inputs = []
    for name, arr in data.items():
        samples = list(arr) if isinstance(arr, list) else [arr] * 4
        pd.make_numpy_calibration_tar(os.path.join(root, f"dataset/{name}.tar"), samples)
        inputs.append(
            {
                "tensor_name": name,
                "calibration_dataset": f"./dataset/{name}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(samples),
            }
        )
    quant: dict = {
        "input_configs": inputs,
        "calibration_method": "MinMax",
        "precision_analysis": False,
    }
    if precision != "U8":
        # every op of the chain: a Concat/Add/Relu left at 8 bits after the
        # 16-bit MatMuls puts the 8-bit error back (legalized Convs)
        types = {n.op_type for n in sub.graph.node}
        ops = sorted(types - PASSIVE_OPS) or sorted(types)
        quant["layer_configs"] = [{"op_types": ops, "data_type": precision}]
        # an op type entry is not applied to a bias Add that follows the
        # MatMul (its output stayed 8-bit); select those by layer name
        adds = [n.name for n in sub.graph.node if n.op_type == "Add"]
        if adds:
            quant["layer_configs"].append(
                {"layer_names": adds, "data_type": precision}
            )
    with open(os.path.join(root, "config/c.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": quant,
                "compiler": {"check": 0},
            },
            f,
        )
    return pd.build(
        root, "t.onnx", "out", config_path="config/c.json", image=image, timeout=timeout
    )


def chain_cache_path(
    cache_dir: str,
    name: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
    precision: str,
) -> str:
    """Where ``cached_chain_axmodel`` keeps the build of this graph,
    calibration data and precision."""
    h = hashlib.sha256(sub.SerializeToString())
    h.update(precision.encode())
    for k in sorted(data):
        h.update(k.encode())
        for a in data[k] if isinstance(data[k], list) else [data[k]]:
            h.update(np.ascontiguousarray(a, np.float32).tobytes())
    return os.path.join(cache_dir, f"{name}.{h.hexdigest()[:16]}.axmodel")


def cached_chain_axmodel(
    cache_dir: str,
    work: str,
    name: str,
    sub: onnx.ModelProto,
    data: Mapping[str, np.ndarray | list[np.ndarray]],
    precision: str = "U16",
    image: str = DEFAULT_IMAGE,
    need_quant: bool = False,
) -> bytes:
    """The compiled axmodel bytes, from ``cache_dir`` when the same graph,
    calibration data and precision were built before. Every build also leaves
    its ``quant_axmodel.json`` beside the axmodel (``<axmodel>.quant.json``);
    with ``need_quant`` a cached axmodel without one is built again."""
    path = chain_cache_path(cache_dir, name, sub, data, precision)
    if os.path.exists(path) and (not need_quant or os.path.exists(path + ".quant.json")):
        with open(path, "rb") as f:
            return f.read()
    failed = path + ".failed"
    if os.path.exists(failed) and not os.environ.get("U16_RETRY_FAILED"):
        with open(failed) as f:
            raise RuntimeError(f"{name}: earlier build failed ({f.read().strip()})")
    res = build_chain(
        work,
        name,
        sub,
        data,
        precision,
        image,
        timeout=int(os.environ.get("U16_BUILD_TIMEOUT", "1800")),
    )
    if not res.success:
        os.makedirs(cache_dir, exist_ok=True)
        with open(failed, "w") as f:
            f.write((res.error or "")[-200:].replace("\n", " "))
        raise RuntimeError(f"{name}: pulsar2 build failed: {(res.error or '')[-500:]}")
    with open(res.axmodel_path, "rb") as f:
        blob = f.read()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    quant = os.path.join(os.path.dirname(res.axmodel_path), "quant", "quant_axmodel.json")
    if os.path.exists(quant):
        shutil.copyfile(quant, path + ".quant.json")
    return blob


def node_model(
    node: onnx.NodeProto, shapes: Mapping[str, Sequence[int]]
) -> onnx.ModelProto:
    """A one-node model whose every input (constants included) is a float32
    graph input, so that one build serves every node with the same operator,
    attributes and input shapes. A rank-0 input becomes ``[1]`` (Pulsar2's
    calibrator rejects rank-0 tensors; the runner reshapes the value)."""
    from onnx import TensorProto, helper

    def shape_of(t: str) -> list[int]:
        return list(shapes[t]) or [1]

    n = onnx.NodeProto()
    n.CopyFrom(node)
    n.name = node.op_type.lower()
    ins = [
        helper.make_tensor_value_info(t, TensorProto.FLOAT, shape_of(t))
        for t in node.input
    ]
    outs = [
        helper.make_tensor_value_info(t, TensorProto.FLOAT, shape_of(t))
        for t in node.output
    ]
    m = helper.make_model(
        helper.make_graph([n], "one", ins, outs),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    m.ir_version = 8
    return m


def signature_data(node: onnx.NodeProto, shapes: Mapping[str, Sequence[int]]):
    """Calibration tensors of a node's shapes. An FP32 layer does not quantize,
    so the values only have to be valid for the operator (ones)."""
    return {
        t: np.ones(list(shapes[t]) or [1], np.float32) for t in node.input if t
    }


def chain_ranges(
    sub: onnx.ModelProto, samples: Mapping[str, list[np.ndarray]]
) -> dict[str, tuple[float, float]]:
    """``{tensor: (min, max)}`` of every float tensor of ``sub`` over the
    calibration ``samples`` (a list per graph input)."""
    import step_calibration

    return step_calibration.collect_ranges(sub, samples)


def predict_scales16(
    quant_json: str | Mapping,
    ranges: Mapping[str, tuple[float, float]],
    margin: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """``{tensor: (scale, zero point)}`` Pulsar2 would assign a 16-bit chain
    whose tensors span ``ranges`` (widened by ``margin``), without building it.

    The rule, checked against native U16 builds of a bare MatMul and a forward
    Conv chain at two calibrations: a tensor that is (or shares a quantization
    with) a live MatMul operand is symmetric int16, ``scale = max|x| / 32767.5``;
    every other tensor is unsigned 16-bit over its range widened to include 0,
    ``scale = f32((hi - lo) / 65535)``, ``zero point = round(-lo / scale)``. Tensors
    that Pulsar2 marks OVERLAPPED share their dominator's quantization, taken
    over the union of the group's ranges. ``quant_json`` is a template build's
    ``quant_axmodel.json`` (path or parsed), which says which tensors are
    symmetric and which are grouped."""
    import json

    if not isinstance(quant_json, Mapping):
        with open(quant_json) as f:
            quant_json = json.load(f)
    info: dict[str, Mapping] = {}
    for per_op in quant_json["tensor_configs"].values():
        for t, v in per_op.items():
            # a BAKED tensor (a constant such as a gather mask) has its values
            # quantized into the model: its scale cannot change
            if v["state"] == "BAKED":
                continue
            if t not in info or v["state"] != "OVERLAPPED":
                info[t] = v
    group: dict[int, list[float]] = {}
    for t, v in info.items():
        if t in ranges:
            lo, hi = ranges[t]
            lo, hi = lo * margin, hi * margin
            g = group.setdefault(v["dominator"], [lo, hi])
            g[0], g[1] = min(g[0], lo), max(g[1], hi)
    out: dict[str, tuple[float, float]] = {}
    for t, v in info.items():
        if v["dominator"] not in group or v["bit_width"] != 16:
            continue
        lo, hi = group[v["dominator"]]
        if v["quant_min"] < 0:
            out[t] = (max(abs(lo), abs(hi)) / 32767.5, 0.0)
        else:
            lo0, hi0 = min(lo, 0.0), max(hi, 0.0)
            s = float(np.float32((hi0 - lo0) / 65535))
            out[t] = (s, float(round(-lo0 / s)))
    return out


def transpose_models(
    pre_shape: Sequence[int], steps: Sequence[tuple[str, object]]
) -> list[onnx.ModelProto]:
    """One-op Transpose models that apply the stripped transposes of a chain to
    its output on the device (a Reshape moves no data, so it is only a change of
    shape). A Transpose-only model is bit-exact on the NPU (measured at U8, U16
    and FP32), so it adds no error."""
    from onnx import TensorProto, helper

    models = []
    shape = list(pre_shape)
    for kind, perm in steps:
        if kind != "T":
            continue
        perm = list(perm) if perm else list(reversed(range(len(shape))))
        out = [shape[i] for i in perm]
        node = helper.make_node("Transpose", ["x"], ["y"], name="tr", perm=perm)
        m = helper.make_model(
            helper.make_graph(
                [node],
                "t",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, out)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        m.ir_version = 8
        models.append(m)
        shape = out
    return models


def expand_model(src: Sequence[int], dst: Sequence[int]) -> onnx.ModelProto:
    """A one-op Expand model (``src`` broadcast to ``dst``): data movement only,
    exact on the NPU (max error 0.0, 7.7 ms for 115 MB), so a broadcast input of
    a device segment need not be broadcast on the host."""
    from onnx import TensorProto, helper, numpy_helper

    node = helper.make_node("Expand", ["x", "shape"], ["y"], name="ex")
    m = helper.make_model(
        helper.make_graph(
            [node],
            "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(src))],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(dst))],
            initializer=[numpy_helper.from_array(np.array(dst, np.int64), "shape")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    m.ir_version = 8
    return m


def derive_ranges(
    sub: onnx.ModelProto,
    quant_json: str | Mapping,
    scales: Mapping[str, tuple[float, float]],
    measured: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Ranges for every tensor of a chain when only some are measured.

    ``scales`` are the template's ``(scale, zero point)`` per tensor, which give
    the range each was calibrated on (symmetric: +-scale * 32767.5; unsigned:
    ``-zp * scale .. (65535 - zp) * scale``). Tensors Pulsar2 groups into one
    quantization (a Transpose, Slice or Reshape passes the same values on) share
    their dominator's range, so **every member of a group with a measured tensor takes
    that measured range**. A group with no measured member keeps its template range
    scaled by how much the groups it depends on moved: a MatMul/Conv/Mul output by the
    product of its inputs' factors, an Add/Sub output by the larger, any other op by
    the larger of its inputs; a factor is the ratio of a measured group's extent
    (max |x|) to its template extent."""
    import json

    if not isinstance(quant_json, Mapping):
        with open(quant_json) as f:
            quant_json = json.load(f)
    dom: dict[str, int] = {}
    sym: dict[str, bool] = {}
    baked: set[str] = set()
    for per_op in quant_json["tensor_configs"].values():
        for t, v in per_op.items():
            if v["state"] == "BAKED":
                baked.add(t)
            elif v["bit_width"] == 16 and (t not in dom or v["state"] != "OVERLAPPED"):
                dom[t] = v["dominator"]
                sym[t] = v["quant_min"] < 0
    tmpl: dict[str, tuple[float, float]] = {}
    for t, (sc, z) in scales.items():
        if t in baked:
            continue
        tmpl[t] = (
            (-sc * 32767.5, sc * 32767.5) if sym.get(t, z == 0) else (-z * sc, (65535 - z) * sc)
        )
    ext = lambda r: max(abs(r[0]), abs(r[1]))  # noqa: E731
    group_range: dict[int, tuple[float, float]] = {}
    for t, r in measured.items():
        if t in dom:
            g = group_range.setdefault(dom[t], tuple(r))
            group_range[dom[t]] = (min(g[0], r[0]), max(g[1], r[1]))
    # An average of values in [lo, hi] lies in [lo, hi]: a ReduceMean/pool output is
    # bounded by its input's measured range (scaling the template range by the input's
    # factor is wrong: a mean does not move like the maximum).
    for node in sub.graph.node:
        if node.op_type in ("ReduceMean", "GlobalAveragePool", "AveragePool") and node.output:
            ri = group_range.get(dom.get(node.input[0]))
            if ri is not None and node.output[0] in dom and dom[node.output[0]] not in group_range:
                group_range[dom[node.output[0]]] = ri
    # An Add's unmeasured input (a MatMul's output, before the bias) moves with the
    # Add's measured output: its template range scaled by the output's factor. (An
    # interval bound, out - bias, is safe but up to 1.9x too wide, which cost
    # gradient cosine; across the forward Convs the scaled estimate is within
    # 0.72-1.13 of the real extent, and the headroom covers the undershoot.)
    for node in reversed(sub.graph.node):
        if node.op_type != "Add" or len(node.input) != 2 or not node.output:
            continue
        y, a, b = node.output[0], node.input[0], node.input[1]
        ry = group_range.get(dom.get(y))
        if ry is None or y not in tmpl or ext(tmpl[y]) <= 0:
            continue
        fy = ext(ry) / ext(tmpl[y])
        for u in (a, b):
            if u in dom and u in tmpl and dom[u] not in group_range:
                group_range[dom[u]] = (tmpl[u][0] * fy, tmpl[u][1] * fy)
    f: dict[str, float] = {}
    for t in tmpl:
        g = dom.get(t)
        if g in group_range and ext(tmpl[t]) > 0:
            f[t] = ext(group_range[g]) / ext(tmpl[t])
    mult_ops = {"MatMul", "Gemm", "Conv", "Mul"}
    for node in sub.graph.node:
        ins = [f[t] if t in f else 1.0 for t in node.input if t]
        for o in node.output:
            if o in f:
                continue
            f[o] = ins[0] * ins[1] if node.op_type in mult_ops and len(ins) >= 2 else max(ins or [1.0])
    out: dict[str, tuple[float, float]] = {}
    for t, r in tmpl.items():
        g = dom.get(t)
        if g in group_range:
            out[t] = group_range[g]
        else:
            k = f.get(t, 1.0)
            out[t] = (r[0] * k, r[1] * k)
    return out


def amax_model(numel: int) -> onnx.ModelProto:
    """A model returning ``[[max, min]]`` of a float32 tensor of ``numel``
    elements, input shape ``[numel]`` (bytes are bytes, so it reads any tensor of
    that size). The NPU reduces only the last axis, so the tensor is reshaped to
    ``[R, C]`` and reduced twice (``[R, 1]`` -> ``[1, R]`` -> ``[1, 1]``)."""
    from onnx import TensorProto, helper, numpy_helper

    c = next(d for d in range(min(numel, 4096), 0, -1) if numel % d == 0)
    r = numel // c
    init = [
        numpy_helper.from_array(np.array([r, c], np.int64), "s2d"),
        numpy_helper.from_array(np.array([1, r], np.int64), "s1r"),
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "s2d"], ["x2"], name="rs"),
        helper.make_node("ReduceMax", ["x2"], ["mx"], name="mx", keepdims=1, axes=[1]),
        helper.make_node("ReduceMin", ["x2"], ["mn"], name="mn", keepdims=1, axes=[1]),
        helper.make_node("Reshape", ["mx", "s1r"], ["mxr"], name="r1"),
        helper.make_node("Reshape", ["mn", "s1r"], ["mnr"], name="r2"),
        helper.make_node("ReduceMax", ["mxr"], ["mx2"], name="mx2", keepdims=1, axes=[1]),
        helper.make_node("ReduceMin", ["mnr"], ["mn2"], name="mn2", keepdims=1, axes=[1]),
        helper.make_node("Concat", ["mx2", "mn2"], ["y"], name="cc", axis=1),
    ]
    m = helper.make_model(
        helper.make_graph(
            nodes,
            "amax",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [numel])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            initializer=init,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    m.ir_version = 8
    return m


def amax_blob(cache_dir: str, numel: int) -> bytes:
    """The compiled amax model for ``numel`` elements (built once, cached)."""
    return cached_chain_axmodel(
        cache_dir,
        os.path.join(cache_dir, "work"),
        "amax",
        amax_model(numel),
        {"x": np.ones([numel], np.float32)},
        "FP32",
    )
