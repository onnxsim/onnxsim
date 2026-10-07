#!/usr/bin/env python3
"""Rewrite an ONNX model (typically a transformer) into operators the Allwinner NPU toolchain documents, before `compile_nbg.py`.

    npu_rewrite.py IN.onnx OUT.onnx [--no-simplify] [--no-check] [--report-only] [--norm-scaling max|none]

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

# Floor for the per-row max |x| that a decomposed norm divides by (the smallest normal fp16 number). It only guards the division: the
# result is algebraically identical for any positive floor (see _Rewriter.normalize).
NORM_FLOOR = 2.0**-14

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
    def __init__(self, model, norm_scaling="max"):
        if norm_scaling not in ("max", "none"):
            raise ValueError(
                f"norm_scaling must be 'max' or 'none', not {norm_scaling!r}"
            )
        self.norm_scaling = norm_scaling
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

    def emit(self, op, inputs, base, tag, out=None, **attrs):
        out = out or self.name(base, tag)
        self.new_nodes.append(
            helper.make_node(op, inputs, [out], name=out + "_n", **attrs)
        )
        return out

    def reduce(self, op, x, axes, base, tag):
        # `axes` became an input in opset 18 (ReduceMean/ReduceMax-18); before that it is an attribute.
        if self.opset >= 18:
            ax = self.const(base, tag + "ax", axes, np.int64)
            return self.emit(op, [x, ax], base, tag, keepdims=1)
        return self.emit(op, [x], base, tag, keepdims=1, axes=list(axes))

    def reduce_mean(self, x, axes, base, tag):
        return self.reduce("ReduceMean", x, axes, base, tag)

    def normalize(self, d, axes, eps, np_t, base, out=None):
        """d / sqrt(mean(d^2) + eps) over `axes`, without ever forming d^2 at full scale.

        Squaring a raw activation overflows fp16 (65504) once |d| exceeds 255, and real transformers exceed it: DistilBERT's residual
        stream has deviations of several hundred, and SmolLM2-135M's reach about 20 000, so the squares are 3e5 and 4e8. Instead the
        row is divided by its own max |d| first (m' = max(max|d|, floor), r = 1/m', ds = d*r):
            mean(ds^2) = mean(d^2)/m'^2,  eps/m'^2 = eps*r^2,  1/sqrt(mean(ds^2) + eps*r^2) = m'/sqrt(mean(d^2) + eps)
        so ds * that = d/sqrt(mean(d^2) + eps): exact for any m' > 0, and every squared value is at most 1."""
        c = lambda tag, v: self.const(base, tag, v, np_t)  # noqa: E731
        if self.norm_scaling == "none":
            # the textbook form, kept for A/B experiments (it overflows fp16 on models with outlier channels)
            sq = self.emit("Mul", [d, d], base, "sq")
            veps = self.emit(
                "Add",
                [self.reduce_mean(sq, axes, base, "var"), c("eps", eps)],
                base,
                "veps",
            )
            inv = self.emit(
                "Reciprocal", [self.emit("Sqrt", [veps], base, "std")], base, "inv"
            )
            return self.emit("Mul", [d, inv], base, "n", out=out)
        mx = self.reduce(
            "ReduceMax", self.emit("Abs", [d], base, "ad"), axes, base, "mx"
        )
        rm = self.emit(
            "Reciprocal",
            [self.emit("Max", [mx, c("fl", NORM_FLOOR)], base, "mxs")],
            base,
            "rm",
        )
        ds = self.emit("Mul", [d, rm], base, "ds")
        var = self.reduce_mean(
            self.emit("Mul", [ds, ds], base, "sq"), axes, base, "var"
        )
        eps_s = self.emit(
            "Mul", [self.emit("Mul", [rm, rm], base, "rr"), c("eps", eps)], base, "epss"
        )
        veps = self.emit("Add", [var, eps_s], base, "veps")
        inv = self.emit(
            "Reciprocal", [self.emit("Sqrt", [veps], base, "std")], base, "inv"
        )
        return self.emit("Mul", [ds, inv], base, "n", out=out)

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
        mean = self.reduce_mean(x, axes, base, "mean")
        d = self.emit("Sub", [x, mean], base, "d")
        y = self.emit(
            "Mul", [self.normalize(d, axes, eps, np_t, base), scale], base, "scaled"
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

    def find_rms_norms(self):
        """Match torch's RMSNorm: Pow(x, 2) -> ReduceMean(axes=-1, keepdims) -> Add(eps) -> Sqrt -> Reciprocal | Div(1, .) -> Mul(x, .).

        Every intermediate must have exactly one consumer. Returns {index of the final Mul: (x, eps, its output, indices to drop)}."""
        nodes = list(self.graph.node)
        consumers = {}
        for i, n in enumerate(nodes):
            for t in n.input:
                consumers.setdefault(t, []).append(i)
        outputs = {o.name for o in self.graph.output}

        def const_of(name):
            """Value of a tiny initializer (the pattern needs only scalars and an axes list), else None. Weights are never read."""
            init = self.inits.get(name)
            if init is None or int(np.prod(init.dims or [1])) > 16:
                return None
            try:
                return numpy_helper.to_array(init)
            except ValueError:  # a malformed tensor is just not a constant we can match
                return None

        def sole(tensor, *ops):
            idx = consumers.get(tensor, [])
            if len(idx) != 1 or tensor in outputs:
                return None
            return idx[0] if nodes[idx[0]].op_type in ops else None

        matches = {}
        for i, pw in enumerate(nodes):
            if pw.op_type != "Pow" or pw.domain not in ("", "ai.onnx"):
                continue
            exponent = const_of(pw.input[1])
            if exponent is None or exponent.size != 1 or not np.allclose(exponent, 2.0):
                continue
            x = pw.input[0]
            m = sole(pw.output[0], "ReduceMean")
            if m is None:
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in nodes[m].attribute}
            axes = attrs.get("axes")
            if axes is None and len(nodes[m].input) > 1:
                given = const_of(nodes[m].input[1])
                axes = given.tolist() if given is not None else None
            if attrs.get("keepdims", 1) != 1 or list(axes or []) != [-1]:
                continue
            e = sole(nodes[m].output[0], "Add")
            if e is None:
                continue
            others = [t for t in nodes[e].input if t != nodes[m].output[0]]
            eps = const_of(others[0]) if len(others) == 1 else None
            if eps is None or eps.size != 1:
                continue
            q = sole(nodes[e].output[0], "Sqrt")
            r = sole(nodes[q].output[0], "Reciprocal", "Div") if q is not None else None
            if r is None:
                continue
            if nodes[r].op_type == "Div":
                numerator = const_of(nodes[r].input[0])
                if (
                    numerator is None
                    or not np.allclose(numerator, 1.0)
                    or nodes[r].input[1] != nodes[q].output[0]
                ):
                    continue
            n = sole(nodes[r].output[0], "Mul")
            if n is None or x not in nodes[n].input or nodes[n].output[0] in outputs:
                continue
            matches[n] = (
                x,
                float(eps.reshape(())),
                nodes[n].output[0],
                {i, m, e, q, r},
            )
        return matches

    def run(self):
        matches = self.find_rms_norms()
        drop = set().union(*(m[3] for m in matches.values())) if matches else set()
        rename = {}
        for i, node in enumerate(list(self.graph.node)):
            if i in drop:
                continue
            for k, t in enumerate(node.input):
                if t in rename:
                    node.input[k] = rename[t]
            if i in matches:
                x, eps, final, _ = matches[i]
                np_t = _NP.get(self.dtype.get(x, TensorProto.FLOAT), np.float32)
                # the normalized value (before the weight multiply) is interior to the norm: later nodes read the new tensor
                rename[final] = self.normalize(x, [-1], eps, np_t, final)
                self.stats["RMSNorm"] += 1
            elif node.domain not in ("", "ai.onnx"):
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


def rewrite(model, norm_scaling="max"):
    """Return (rewritten copy, Counter of rewrites). Does not simplify; see main().

    norm_scaling: "max" (default) divides each normalization row by its max |x| before squaring so fp16 cannot overflow; "none" emits
    the textbook d*d form, which is only for A/B experiments."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    return _Rewriter(model, norm_scaling).run()


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
    p.add_argument(
        "--norm-scaling",
        choices=("max", "none"),
        default="max",
        help="'max' (default) keeps norm squares inside fp16; 'none' is the textbook form for A/B experiments",
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
    rewritten, stats = rewrite(work, a.norm_scaling)
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
