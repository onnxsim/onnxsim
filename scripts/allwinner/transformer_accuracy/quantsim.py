"""Emulate the Allwinner/Acuity quantization schemes on an ONNX graph with fake-quantize nodes, to measure their accuracy cost.

Schemes, named after Acuity's `--quantizer` / `--qtype`:

  uint8   asymmetric_affine: per-tensor 8-bit affine weights and activations
  pcq     perchannel_symmetric_affine: per-output-channel symmetric int8 weights, per-tensor 8-bit affine activations
  int16   dynamic_fixed_point: power-of-two scales, 16 bits
  fp16 / bf16: every tensor and weight rounded through the format

Every float tensor a node produces is fake-quantized (as Acuity quantizes the whole graph) unless it is named in `keep` (left exactly
float: the interior of a fused kernel) or `keep_fp16` (rounded through fp16: a layer kept at higher precision, as in hybrid
quantization). This simulates the schemes. It is not Acuity: its calibration algorithm, operator fusion and internal precisions may
differ, so treat results as what the scheme can do, not what a compiled NBG will do.
"""

import copy
import math
import re

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

FLOAT = TensorProto.FLOAT


# ---- calibration ---------------------------------------------------------------------------------------------------------------
def float_node_outputs(model):
    """Names of float tensors produced by nodes. The model needs value_info (run onnx.shape_inference.infer_shapes first)."""
    dtype = {
        v.name: v.type.tensor_type.elem_type
        for v in list(model.graph.value_info) + list(model.graph.output)
    }
    return [o for n in model.graph.node for o in n.output if dtype.get(o) == FLOAT]


def collect_ranges(model, batches, names):
    """Per-tensor (min, max) over the calibration batches, computed inside the graph with ReduceMin/ReduceMax.

    Returns {"minmax": ..., "ema": ...}: the overall min/max, and the mean of the per-batch min/max (Acuity's moving-average style)."""
    g = copy.deepcopy(model)
    outs = []
    for i, t in enumerate(names):
        for op, tag in (("ReduceMin", "mn"), ("ReduceMax", "mx")):
            o = f"__r_{tag}_{i}"
            g.graph.node.append(helper.make_node(op, [t], [o], keepdims=0, name=o))
            g.graph.output.append(helper.make_tensor_value_info(o, FLOAT, []))
            outs.append(o)
    sess = ort.InferenceSession(
        g.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    per_lo, per_hi = [], []
    for feeds in batches:
        r = sess.run(outs, feeds)
        per_lo.append(np.array([float(x) for x in r[0::2]]))
        per_hi.append(np.array([float(x) for x in r[1::2]]))
    lo, hi = np.min(per_lo, axis=0), np.max(per_hi, axis=0)
    mlo, mhi = np.mean(per_lo, axis=0), np.mean(per_hi, axis=0)
    return {
        "minmax": {t: (float(lo[i]), float(hi[i])) for i, t in enumerate(names)},
        "ema": {t: (float(mlo[i]), float(mhi[i])) for i, t in enumerate(names)},
    }


def collect_percentile(model, batches, names, lo_p=0.1, hi_p=99.9, max_elems=300000):
    """Per-tensor range from activation percentiles averaged over the batches (clips outliers; see the README for why that hurts)."""
    g = copy.deepcopy(model)
    for t in names:
        g.graph.output.append(helper.make_tensor_value_info(t, FLOAT, None))
    sess = ort.InferenceSession(
        g.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    acc = {t: ([], []) for t in names}
    for feeds in batches:
        for t, a in zip(names, sess.run(names, feeds)):
            flat = a.ravel()
            if flat.size > max_elems:
                flat = flat[:: flat.size // max_elems + 1]
            lo, hi = np.percentile(flat, [lo_p, hi_p])
            acc[t][0].append(lo)
            acc[t][1].append(hi)
    return {t: (float(np.mean(v[0])), float(np.mean(v[1]))) for t, v in acc.items()}


# ---- numpy fake-quantization (weights, and the reference for the graph nodes) ----------------------------------------------------
def fq_affine(x, lo, hi, bits=8):
    """Asymmetric affine fake-quantize over [lo, hi] widened to include zero."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-38:
        return np.asarray(x, np.float32).copy()
    scale = (hi - lo) / (2**bits - 1)
    zp = np.clip(np.round(-lo / scale), 0, 2**bits - 1)
    q = np.clip(np.round(x / scale) + zp, 0, 2**bits - 1)
    return ((q - zp) * scale).astype(np.float32)


def fq_sym_perchannel(x, axis, bits=8):
    """Symmetric per-channel fake-quantize along `axis`."""
    reduce_axes = tuple(i for i in range(x.ndim) if i != axis % x.ndim)
    m = np.abs(x).max(axis=reduce_axes, keepdims=True)
    scale = np.where(m > 0, m / (2 ** (bits - 1) - 1), 1.0)
    q = np.clip(np.round(x / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (q * scale).astype(np.float32)


def dfp_fl(maxabs, bits=16):
    """Fractional length of a dynamic-fixed-point format that just holds `maxabs`."""
    return (bits - 1) - math.ceil(math.log2(maxabs)) if maxabs > 0 else 0


def fq_dfp(x, bits=16):
    s = 2.0 ** dfp_fl(float(np.abs(x).max()), bits)
    q = np.clip(np.round(x * s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (q / s).astype(np.float32)


def to_half(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def to_bf16(x):
    """Round float32 to bfloat16 (nearest even) and back."""
    b = np.asarray(x, np.float32).view(np.uint32)
    return (
        ((b + ((b >> 16) & 1) + 0x7FFF) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    )


# ---- fake-quantize nodes -------------------------------------------------------------------------------------------------------
class _Builder:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def const(self, value, name):
        self.n += 1
        nm = f"__c{self.n}_{name}"
        self.inits.append(numpy_helper.from_array(np.asarray(value, np.float32), nm))
        return nm

    def op(self, op, ins, out=None, **kw):
        self.n += 1
        out = out or f"__t{self.n}"
        self.nodes.append(helper.make_node(op, ins, [out], name=f"__n{self.n}", **kw))
        return out

    def chain(self, src, dst, scheme, rng):
        """Fake-quantize tensor `src` into `dst`."""
        if scheme in ("fp16", "bf16"):
            mid = self.op(
                "Cast",
                [src],
                to=TensorProto.FLOAT16 if scheme == "fp16" else TensorProto.BFLOAT16,
            )
            self.op("Cast", [mid], dst, to=FLOAT)
        elif scheme == "int16":
            s = 2.0 ** dfp_fl(max(abs(rng[0]), abs(rng[1])), 16)
            scaled = self.op("Mul", [src, self.const(s, "s")])
            q = self.op(
                "Clip",
                [
                    self.op("Round", [scaled]),
                    self.const(-32768, "lo"),
                    self.const(32767, "hi"),
                ],
            )
            self.op("Mul", [q, self.const(1.0 / s, "inv")], dst)
        else:  # uint8 / pcq activations: 8-bit affine
            lo, hi = min(rng[0], 0.0), max(rng[1], 0.0)
            if hi - lo < 1e-38:
                self.op("Identity", [src], dst)
                return
            scale = (hi - lo) / 255.0
            zp = float(np.clip(np.round(-lo / scale), 0, 255))
            shifted = self.op(
                "Add",
                [
                    self.op("Mul", [src, self.const(1.0 / scale, "is")]),
                    self.const(zp, "zp"),
                ],
            )
            q = self.op(
                "Clip",
                [
                    self.op("Round", [shifted]),
                    self.const(0, "lo"),
                    self.const(255, "hi"),
                ],
            )
            self.op(
                "Mul",
                [self.op("Sub", [q, self.const(zp, "zp2")]), self.const(scale, "sc")],
                dst,
            )


WEIGHT_CONSUMERS = {"MatMul", "Gemm", "Conv", "Gather", "Mul", "Add", "Sub"}


def quantize_weights(model, scheme):
    """Fake-quantize, in place, the float initializers that weight-like operators consume. A Gemm bias stays float (Acuity keeps int32)."""
    consumers = {}
    for n in model.graph.node:
        for k, t in enumerate(n.input):
            consumers.setdefault(t, []).append((n, k))
    for init in model.graph.initializer:
        if init.data_type != FLOAT or init.name not in consumers:
            continue
        uses = [
            (n, k)
            for n, k in consumers[init.name]
            if n.op_type in WEIGHT_CONSUMERS and not (n.op_type == "Gemm" and k == 2)
        ]
        if not uses:
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        n, k = uses[0]
        axis = None
        if (
            scheme == "pcq"
            and k == 1
            and n.op_type in ("MatMul", "Gemm", "Conv")
            and w.ndim >= 2
        ):
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            axis = (
                0 if (n.op_type == "Conv" or (n.op_type == "Gemm" and trans_b)) else -1
            )
        if scheme == "fp16":
            q = to_half(w)
        elif scheme == "bf16":
            q = to_bf16(w)
        elif scheme == "int16":
            q = fq_dfp(w)
        elif scheme == "pcq" and axis is not None:
            q = fq_sym_perchannel(w, axis)
        else:
            q = fq_affine(w, float(w.min()), float(w.max()))
        init.CopyFrom(numpy_helper.from_array(q, init.name))


def build(model, ranges, scheme, keep=frozenset(), keep_fp16=frozenset()):
    """A fake-quantized copy of `model`. `ranges` maps tensor -> (min, max); see the module docstring for keep / keep_fp16."""
    m = copy.deepcopy(model)
    if scheme != "fp32":
        quantize_weights(m, scheme)
    b = _Builder()
    for n in m.graph.node:
        renamed = []
        for o in n.output:
            quantized = scheme != "fp32" and o in ranges and o not in keep
            renamed.append(o + "__pre" if quantized else o)
        n2 = copy.deepcopy(n)
        del n2.output[:]
        n2.output.extend(renamed)
        b.nodes.append(n2)
        for o, new in zip(n.output, renamed):
            if new != o:
                b.chain(new, o, "fp16" if o in keep_fp16 else scheme, ranges[o])
    del m.graph.node[:]
    m.graph.node.extend(b.nodes)
    m.graph.initializer.extend(b.inits)
    return m


def evaluate(model, batches):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return np.concatenate([sess.run(None, f)[0] for f in batches])


# ---- structure: which tensors a hybrid scheme leaves alone ---------------------------------------------------------------------
LN_INTERIOR = re.compile(
    r"__(mean|d|ad|mx|mxs|rm|ds|sq|var|rr|epss|veps|std|inv|n|scaled)\d+$"
)


def hybrid_sets(model):
    """Tensor sets for hybrid schemes, found from the graph (call on the npu_rewrite.py output after shape inference).

    softmax_in: the (scores + mask) tensors feeding Softmax, whose range is set by the mask fill value;
    ln_interior: the interior tensors npu_rewrite.py creates for a LayerNormalization (a fused kernel computes them internally);
    gelu_interior: the interior of an erf-GELU, between the Gemm/MatMul that feeds it and the one that consumes it;
    rms_interior: the interior of a decomposed RMSNorm (Pow(x, 2) ... up to the weight multiply);
    silu_interior: the Sigmoid output of a SiLU (h * Sigmoid(h));
    matmul_in: activation tensors feeding Gemm/MatMul, the only ones an int8 matrix engine needs in 8 bits."""
    names = set(float_node_outputs(model))
    producers = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for n in model.graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)
    softmax_in = {n.input[0] for n in model.graph.node if n.op_type == "Softmax"}
    ln_interior = {t for t in names if LN_INTERIOR.search(t)}
    matmul_in = {
        t
        for n in model.graph.node
        if n.op_type in ("Gemm", "MatMul")
        for t in n.input[:2]
        if t in names
    }
    gelu_interior = set()
    for erf in (n for n in model.graph.node if n.op_type == "Erf"):
        back, stack = set(), [erf.input[0]]
        while stack:
            x = stack.pop()
            if (
                x in back
                or x not in producers
                or producers[x].op_type in ("Gemm", "MatMul")
            ):
                continue
            back.add(x)
            stack += list(producers[x].input)
        fwd, stack = set(), [erf.output[0]]
        while stack:
            x = stack.pop()
            if x in fwd:
                continue
            fwd.add(x)
            for c in consumers.get(x, []):
                if c.op_type not in ("Gemm", "MatMul"):
                    stack += list(c.output)
        boundary = {
            x
            for x in fwd
            if any(c.op_type in ("Gemm", "MatMul") for c in consumers.get(x, []))
        }
        gelu_interior |= (back | fwd) - boundary
    # RMSNorm, as torch exports it: Pow(x, 2) -> ReduceMean -> Add(eps) -> Sqrt -> Reciprocal/Div -> Mul(x, .) -> Mul(weight, .).
    # Everything from the Pow output up to (not including) the weight multiply is the interior of a fused kernel.
    initializers = {i.name: i for i in model.graph.initializer}
    rms_interior = set()
    for pow_node in (n for n in model.graph.node if n.op_type == "Pow"):
        exponent = initializers.get(pow_node.input[1])
        is_square = exponent is not None and np.allclose(
            numpy_helper.to_array(exponent), 2.0
        )
        if not is_square or not any(
            c.op_type == "ReduceMean" for c in consumers.get(pow_node.output[0], [])
        ):
            continue
        seen, stack = set(), [pow_node.output[0]]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            for c in consumers.get(t, []):
                if c.op_type == "Mul" and any(i in initializers for i in c.input):
                    continue  # the weight multiply: its output is the norm's result and stays quantized
                stack += list(c.output)
        rms_interior |= seen
    # SiLU(h) = h * Sigmoid(h): the sigmoid output is interior to a fused swish kernel; the product leaves the kernel and stays quantized
    silu_interior = {
        s.output[0]
        for s in model.graph.node
        if s.op_type == "Sigmoid"
        and any(
            c.op_type == "Mul" and s.input[0] in c.input
            for c in consumers.get(s.output[0], [])
        )
    }
    return {
        "softmax_in": softmax_in & names,
        "ln_interior": ln_interior,
        "gelu_interior": gelu_interior & names,
        "rms_interior": rms_interior & names,
        "silu_interior": silu_interior & names,
        "matmul_in": matmul_in,
    }
