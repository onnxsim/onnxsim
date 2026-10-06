#!/usr/bin/env python3
"""Rewrite an ONNX model (typically a transformer) into operators the Allwinner NPU toolchain documents, before `compile_nbg.py`.

    npu_rewrite.py IN.onnx OUT.onnx [--no-simplify] [--no-check] [--report-only]

Acuity's ONNX importer (ONNX 1.14.0 in Allwinner's operator-support list for the A733) does not list `LayerNormalization`, `Gelu` or
`Div`, which is what a torch/transformers export of BERT, ViT or LLaMA is made of. This tool, after onnxsim has removed `Identity`/
`Constant` nodes, replaces them with listed operators:

  LayerNormalization -> ReduceMean, Sub, Mul, ReduceMean, Add(eps), Sqrt, Reciprocal, Mul (+ scale, bias)
  Gelu (exact)       -> 0.5 * x * (1 + Erf(x / sqrt(2)))       Gelu (tanh) -> 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
  Div(a, const)      -> Mul(a, 1/const)                         Div(a, b)   -> Mul(a, Reciprocal(b))     (float tensors only)

and checks the result against the original on random inputs. It reports what is still outside the documented set (for example `Einsum`,
`Trilu`, `RMSNormalization`) rather than guessing a rewrite. It does not know how Acuity handles any of this: the documented list is a
necessary condition, not a guarantee, and it has not been run through Acuity.
"""

import argparse
import sys
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

# ONNX operators in the "Onnx" table of Allwinner's NPU operator-support list v1.5 (A733 chapter), extracted from that document.
# `Div`, `Identity`, `Dropout` and `Constant` are absent from the table although the hardware list has a divide kernel and the
# importer/simplifier removes the others; they are treated as "not documented" here, and this tool still rewrites Div.
DOCUMENTED_ONNX_OPS = frozenset(
    """Abs Acos Acosh Add And ArgMax ArgMin Asin Asinh Atan Atanh AveragePool BatchNormalization BitwiseAnd BitwiseOr BitwiseXor Cast
    CastLike Ceil Celu CenterCropPad Clip Col2Im Concat ConstantOfShape Conv ConvTranspose Cos Cumsum Elu Equal Erf Exp Expand Flatten
    Floor Gather GatherElements GatherND Gemm GlobalAveragePool GlobalMaxPool Greater GreaterOrEqual GridSample GroupNormalization GRU
    HammingWindow HannWindow HardSigmoid HardSwish InstanceNormalization LeakyRelu Less LessOrEqual Log Logsoftmax LRN LSTM MatMul Max
    MaxPool MaxRoiPool Mean MeanVarianceNormalization Min Mish Mod Mul Neg NonZero OneHot Or Pad Pow Prelu QLinearConv QLinearMatMul
    QuantizeLinear Range Reciprocal ReduceL1 ReduceL2 ReduceLogSum ReduceLogSumExp ReduceMax ReduceMean ReduceMin ReduceProd ReduceSum
    ReduceSumSquare Relu Reshape Resize ReverseSequence Round ScatterElements ScatterND Selu Shape Sigmoid Sign Silu Sin Size Slice
    Softmax Softplus Softsign SpaceToDepth Split Sqrt Squeeze STFT Sub Sum Tan Tanh Tile TopK Transpose Unsqueeze Upsample Where Xor""".split()
)

LN_PRESCALE = (
    1.0 / 16
)  # keeps the squared deviations of a decomposed LayerNormalization inside fp16 range (see layer_norm)

FLOAT_TYPES = (
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
)
_NP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
}


def op_histogram(model):
    return Counter(n.op_type for n in model.graph.node)


def undocumented(model):
    return {
        op: n for op, n in op_histogram(model).items() if op not in DOCUMENTED_ONNX_OPS
    }


class _Rewriter:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self.opset = next(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 13
        )
        inferred = shape_inference.infer_shapes(model)
        self.dtype = {}
        for v in (
            list(inferred.graph.value_info)
            + list(inferred.graph.input)
            + list(inferred.graph.output)
        ):
            self.dtype[v.name] = v.type.tensor_type.elem_type
        for init in self.graph.initializer:
            self.dtype[init.name] = init.data_type
        self.inits = {i.name: i for i in self.graph.initializer}
        self.counter = 0
        self.new_nodes = []
        self.stats = Counter()

    def name(self, base, tag):
        self.counter += 1
        return f"{base}__{tag}{self.counter}"

    def const(self, base, tag, value, np_dtype):
        name = self.name(base, tag)
        self.graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=np_dtype), name)
        )
        return name

    def emit(self, op, inputs, base, tag, **attrs):
        out = self.name(base, tag)
        self.new_nodes.append(
            helper.make_node(op, inputs, [out], name=out + "_n", **attrs)
        )
        return out

    def reduce_mean(self, x, axes, base, tag):
        # `axes` became an input in opset 18 (ReduceMean-18); before that it is an attribute.
        if self.opset >= 18:
            ax = self.const(base, tag + "ax", axes, np.int64)
            return self.emit("ReduceMean", [x, ax], base, tag, keepdims=1)
        return self.emit("ReduceMean", [x], base, tag, keepdims=1, axes=list(axes))

    def layer_norm(self, node):
        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        axis, eps = attrs.get("axis", -1), attrs.get("epsilon", 1e-5)
        et = self.dtype.get(x, TensorProto.FLOAT)
        np_t = _NP.get(et, np.float32)
        base = node.name or node.output[0]
        # Normalize over axis..last. Negative axes are valid for ReduceMean as given; a positive axis would need the tensor's rank.
        if axis >= 0:
            raise ValueError(
                f"LayerNormalization {base}: positive axis {axis} needs a static rank; export with a negative axis"
            )
        axes = list(range(axis, 0))
        # The deviation is pre-scaled by s = 1/16 before it is squared, and the compensation is folded into epsilon and the reciprocal:
        #   ds = d*s, var_s = E[ds^2] = s^2 var, 1/sqrt(var_s + eps*s^2) = 1/(s*sqrt(var+eps)), so ds * that = d / sqrt(var+eps).
        # Mathematically identical, but d^2 can no longer overflow fp16 (65504): DistilBERT's residual stream has outlier channels with
        # deviations of several hundred, whose squares reach 3e5.
        mean = self.reduce_mean(x, axes, base, "mean")
        d = self.emit("Sub", [x, mean], base, "d")
        ds = self.emit("Mul", [d, self.const(base, "s", LN_PRESCALE, np_t)], base, "ds")
        sq = self.emit("Mul", [ds, ds], base, "sq")
        var = self.reduce_mean(sq, axes, base, "var")
        veps = self.emit(
            "Add",
            [var, self.const(base, "eps", eps * LN_PRESCALE**2, np_t)],
            base,
            "veps",
        )
        inv = self.emit(
            "Reciprocal", [self.emit("Sqrt", [veps], base, "std")], base, "inv"
        )
        y = self.emit(
            "Mul", [self.emit("Mul", [ds, inv], base, "n"), scale], base, "scaled"
        )
        if bias:
            self.new_nodes.append(
                helper.make_node(
                    "Add", [y, bias], [node.output[0]], name=base + "_bias"
                )
            )
        else:
            self.new_nodes[-1].output[0] = node.output[0]
        self.stats["LayerNormalization"] += 1

    def gelu(self, node):
        x = node.input[0]
        approx = next(
            (
                helper.get_attribute_value(a)
                for a in node.attribute
                if a.name == "approximate"
            ),
            b"none",
        )
        approx = approx.decode() if isinstance(approx, bytes) else approx
        np_t = _NP.get(self.dtype.get(x, TensorProto.FLOAT), np.float32)
        base = node.name or node.output[0]
        c = lambda tag, v: self.const(base, tag, v, np_t)  # noqa: E731
        if approx == "tanh":
            cube = self.emit(
                "Mul", [self.emit("Mul", [x, x], base, "x2"), x], base, "x3"
            )
            inner = self.emit(
                "Add",
                [x, self.emit("Mul", [cube, c("k", 0.044715)], base, "kx3")],
                base,
                "inner",
            )
            t = self.emit(
                "Tanh",
                [self.emit("Mul", [inner, c("s", np.sqrt(2.0 / np.pi))], base, "arg")],
                base,
                "t",
            )
        else:
            t = self.emit(
                "Erf",
                [self.emit("Mul", [x, c("r", 1.0 / np.sqrt(2.0))], base, "arg")],
                base,
                "t",
            )
        one_plus = self.emit("Add", [t, c("one", 1.0)], base, "onep")
        half_x = self.emit("Mul", [x, c("half", 0.5)], base, "hx")
        self.new_nodes.append(
            helper.make_node(
                "Mul", [half_x, one_plus], [node.output[0]], name=base + "_out"
            )
        )
        self.stats["Gelu"] += 1

    def div(self, node):
        a, b = node.input
        et = self.dtype.get(a, self.dtype.get(b))
        if et not in FLOAT_TYPES:
            self.new_nodes.append(
                node
            )  # integer Div (shape arithmetic) is not a tensor op the NPU runs
            return
        base = node.name or node.output[0]
        if b in self.inits and self.inits[b].data_type in _NP:
            arr = numpy_helper.to_array(self.inits[b])
            if not np.any(arr == 0):
                rb = self.const(
                    base,
                    "rcp",
                    (1.0 / arr.astype(np.float64)).astype(arr.dtype),
                    arr.dtype,
                )
                self.new_nodes.append(
                    helper.make_node(
                        "Mul", [a, rb], [node.output[0]], name=base + "_mul"
                    )
                )
                self.stats["Div(const)"] += 1
                return
        self.new_nodes.append(
            helper.make_node(
                "Mul",
                [a, self.emit("Reciprocal", [b], base, "rcp")],
                [node.output[0]],
                name=base + "_mul",
            )
        )
        self.stats["Div"] += 1

    def run(self):
        for node in self.graph.node:
            if node.domain not in ("", "ai.onnx"):
                self.new_nodes.append(node)
            elif node.op_type == "LayerNormalization":
                self.layer_norm(node)
            elif node.op_type == "Gelu":
                self.gelu(node)
            elif node.op_type == "Div":
                self.div(node)
            else:
                self.new_nodes.append(node)
        del self.graph.node[:]
        self.graph.node.extend(self.new_nodes)
        return self.model, self.stats


def rewrite(model):
    """Return (rewritten copy, Counter of rewrites). Does not simplify; see main()."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    return _Rewriter(model).run()


# Allwinner's list gives per-operator size limits: Softmax input/output dimensions up to 8191 (16383 on one axis), matrix multiplication
# and fully-connected operands up to 16383-1048575 depending on the axis. Which ONNX axis maps to which documented axis is not stated,
# so this is a conservative heuristic (any dimension above the common 8191) that flags candidates to check, not a verdict.
SIZE_LIMIT = 8191
SIZE_LIMITED_OPS = ("Softmax", "MatMul", "Gemm")


def large_dims(model, limit=SIZE_LIMIT):
    """[(node name, op, tensor, shape)] for Softmax/MatMul/Gemm operands or results with a dimension above `limit`."""
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for v in (
        list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    ):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    for i in inferred.graph.initializer:
        shapes[i.name] = list(i.dims)
    found = []
    for n in inferred.graph.node:
        if n.op_type not in SIZE_LIMITED_OPS:
            continue
        for t in list(n.input) + list(n.output):
            if t and any(d > limit for d in shapes.get(t, [])):
                found.append((n.name or n.output[0], n.op_type, t, shapes[t]))
    return found


def random_inputs(model, seed=0):
    rng = np.random.default_rng(seed)
    init = {i.name for i in model.graph.initializer}
    feeds = {}
    for v in model.graph.input:
        if v.name in init:
            continue
        tt = v.type.tensor_type
        shape = [d.dim_value or 1 for d in tt.shape.dim]
        if tt.elem_type in (TensorProto.INT64, TensorProto.INT32):
            feeds[v.name] = rng.integers(0, 50, shape).astype(
                np.int64 if tt.elem_type == TensorProto.INT64 else np.int32
            )
        else:
            feeds[v.name] = rng.standard_normal(shape).astype(
                _NP.get(tt.elem_type, np.float32)
            )
    return feeds


def run_model(model, feeds):
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        return ReferenceEvaluator(model).run(None, feeds)


def compare(a, b, seed=0):
    feeds = random_inputs(a, seed)
    ya, yb = run_model(a, feeds), run_model(b, feeds)
    worst_abs = max(
        float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for x, y in zip(ya, yb)
    )
    scale = max(float(np.max(np.abs(x))) for x in ya) or 1.0
    return worst_abs, worst_abs / scale


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input")
    p.add_argument("output", nargs="?")
    p.add_argument(
        "--no-simplify",
        action="store_true",
        help="skip onnxsim (needs it installed otherwise)",
    )
    p.add_argument(
        "--no-check",
        action="store_true",
        help="skip the numerical comparison against the original",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="only list operators not in the documented set",
    )
    a = p.parse_args(argv)
    original = onnx.load(a.input)
    work = original
    if not a.no_simplify:
        import onnxsim

        work, ok = onnxsim.simplify(original)
        if not ok:
            print(
                "onnxsim could not validate the simplified model; pass --no-simplify to skip it",
                file=sys.stderr,
            )
            return 1
    print(
        f"nodes: {sum(op_histogram(original).values())} original -> {sum(op_histogram(work).values())} simplified"
    )
    print(
        "not in the documented ONNX set (simplified):",
        dict(undocumented(work)) or "none",
    )
    if a.report_only:
        return 0
    rewritten, stats = rewrite(work)
    rewritten = shape_inference.infer_shapes(rewritten)
    left = undocumented(rewritten)
    print("rewrites:", dict(stats) or "none")
    print("not in the documented ONNX set (rewritten):", dict(left) or "none")
    for name, op, tensor, shape in large_dims(rewritten):
        print(
            f"warning: {op} {name}: tensor {tensor} {shape} has a dimension above {SIZE_LIMIT}; check Allwinner's per-operator size limits"
        )
    if not a.no_check:
        err, rel = compare(original, rewritten)
        print(
            f"max |original - rewritten| on random inputs: {err:.3g} ({rel:.3g} of the output range)"
        )
    if a.output:
        onnx.save(rewritten, a.output)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
