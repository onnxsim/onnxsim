"""Models and graph comparisons shared by ``test_quark_fusions.py`` (Quark-free) and
``test_quark_fusions_parity.py`` (against the real package).

The models are written in the ONNX text format and are the patterns torch (LayerNorm,
Gelu), tf2onnx / Keras (the Gelu shapes, InstanceNorm, the L2 normalization) emit,
decomposed. Graph comparison is by content: a name-independent signature of every graph
output and the multiset of the nodes, with the Q/DQ parameters as values.
"""

import hashlib

import numpy as np
import onnx
from onnx import numpy_helper, parser

F32 = np.float32


def _t(name, value, dtype=F32):
    return numpy_helper.from_array(np.array(value, dtype), name)


def _model(body, initializer=(), opset=17, shape=(2, 4, 16), out_shape=None):
    out_shape = out_shape or shape
    m = parser.parse_model(
        f"""<ir_version: 8, opset_import: ["": {opset}]>
        g (float{list(shape)} x) => (float{list(out_shape)} y) {{ {body} }}"""
    )
    m.graph.initializer.extend(initializer)
    return m


def _ops(m):
    return [n.op_type for n in m.graph.node]


def _attrs(n):
    return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}


def _run(model, x, name="x"):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {name: x})[0]


def _ln(opset=17, x="x", y="y", p="", variant="base", consts="init", eps_first=False):
    """The decomposed LayerNorm torch emits (a ``Pow``, two ``ReduceMean``);
    ``variant`` bends one place of it."""
    ax = "<axes=[-1]>" if opset < 18 else ""
    axin = f", {p}axes" if opset >= 18 else ""
    const = (
        ""
        if consts == "init"
        else f"""{p}two = Constant<value = float {{2.0}}>()
        {p}eps = Constant<value = float {{1e-5}}>()
        """
    )
    add_eps = (
        f"{p}ve = Add({p}eps, {p}v)" if eps_first else f"{p}ve = Add({p}v, {p}eps)"
    )
    sub = f"{p}d = Sub({x}, {p}m)"
    pow_in, div_in = f"{p}d", f"{p}d"
    extra = ""
    out = f"{p}n"
    mul = f"{p}sc = Mul({p}n, {p}w)"
    last = f"{y} = Add({p}sc, {p}b)"
    exponent = f"{p}two"
    if variant == "dup_sub":
        sub += f"\n{p}d2 = Sub({x}, {p}m)"
        pow_in = f"{p}d2"
    elif variant == "cast_pow":
        extra = f"{p}dc = Cast<to=1>({p}d)"
        pow_in = f"{p}dc"
    elif variant == "cast_after_div":
        extra = f"{p}nc = Cast<to=1>({p}n)"
        mul = f"{p}sc = Mul({p}nc, {p}w)"
    elif variant == "pow3":
        exponent = f"{p}three"
    elif variant == "square_by_mul":
        exponent = ""
    elif variant == "no_affine":
        mul, last = "", ""
        out = y
    elif variant == "no_bias":
        mul = f"{y} = Mul({p}n, {p}w)"
        last = ""
    elif variant == "extra_consumer":
        last = f"{p}y0 = Add({p}sc, {p}b)\n{y} = Add({p}y0, {p}d)"
    elif variant == "comm_affine":
        mul = f"{p}sc = Mul({p}w, {p}n)"
        last = f"{y} = Add({p}b, {p}sc)"
    pow_node = (
        f"{p}p = Pow({pow_in}, {exponent})"
        if exponent
        else f"{p}p = Mul({pow_in}, {pow_in})"
    )
    return f"""
        {const}{p}m = ReduceMean{ax}({x}{axin})
        {sub}
        {extra}
        {pow_node}
        {p}v = ReduceMean{ax}({p}p{axin})
        {add_eps}
        {p}s = Sqrt({p}ve)
        {out} = Div({div_in}, {p}s)
        {mul}
        {last}
    """


def _ln_inits(opset=17, p="", d=16, eps=1e-5, consts="init", w_shape=None):
    rng = np.random.default_rng(0)
    inits = [
        _t(f"{p}w", 1 + rng.random(w_shape or d)),
        _t(f"{p}b", rng.standard_normal(w_shape or d)),
        _t(f"{p}three", 3.0),
    ]
    if consts == "init":
        inits += [_t(f"{p}two", 2.0), _t(f"{p}eps", eps)]
    if opset >= 18:
        inits.append(_t(f"{p}axes", [-1], np.int64))
    return inits


def _ln_model(opset=17, eps=1e-5, consts="init", w_shape=None, **kw):
    return _model(
        _ln(opset, consts=consts, **kw),
        _ln_inits(opset, eps=eps, consts=consts, w_shape=w_shape),
        opset,
    )


_GELU = {
    # torch: x * 0.5 * (1 + erf(x / sqrt 2)), in the two orders it emits
    "torch_mul_half_first": "h = Mul(x, half)\nd = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\ny = Mul(h, a)",
    "torch_mul_half_last": "d = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\nm = Mul(x, a)\ny = Mul(m, half)",
    "torch_commuted": "h = Mul(half, x)\nd = Div(x, rt2)\ne = Erf(d)\na = Add(one, e)\ny = Mul(a, h)",
    # Keras / TensorFlow: the input is the output of an earlier node
    "keras": "r = Relu(x)\nd = Div(r, rt2)\ne = Erf(d)\na = Add(e, one)\nmh = Mul(a, half)\ny = Mul(mh, r)",
    "keras_sqrt": "r = Relu(x)\nsq = Sqrt(two)\nd = Div(r, sq)\ne = Erf(d)\na = Add(e, one)\nmh = Mul(a, half)\ny = Mul(r, mh)",
    "tf": "r = Relu(x)\nf = Mul(r, rrt2)\ne = Erf(f)\na = Add(e, one)\nmh = Mul(a, half)\ny = Mul(mh, r)",
}


_GELU_NOT = {
    # the input is a graph input: Keras / TF want it to be a node's output
    "keras_on_graph_input": "d = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\nmh = Mul(a, half)\ny = Mul(mh, x)",
    "tf_on_graph_input": "f = Mul(x, rrt2)\ne = Erf(f)\na = Add(e, one)\nmh = Mul(a, half)\ny = Mul(mh, x)",
    # the Add's result is read twice
    "shared_add": "h = Mul(x, half)\nd = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\ny0 = Mul(h, a)\ny = Add(y0, a)",
    "wrong_square": "h = Mul(x, x)\nd = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\ny = Mul(h, a)",
    # tanh form: no matcher in Quark or ONNX Runtime
    "tanh_form": "x3 = Mul(x, x)\nx3b = Mul(x3, x)\nc = Mul(x3b, k1)\ns = Add(x, c)\nt = Mul(s, k2)\nth = Tanh(t)\na = Add(th, one)\nh = Mul(x, half)\ny = Mul(h, a)",
}


def _gelu_inits():
    return [
        _t("half", 0.5),
        _t("one", 1.0),
        _t("two", 2.0),
        _t("rt2", 1.4142135623730951),
        _t("rrt2", 0.7071067690849304),
        _t("k1", 0.044715),
        _t("k2", 0.7978845608),
    ]


def _gelu_model(name, opset=20, table=_GELU):
    return _model(table[name], _gelu_inits(), opset)


_IN = """
    mean = GlobalAveragePool(x)
    d = Sub(x, mean)
    sq = Mul(d, d)
    var = GlobalAveragePool(sq)
    ve = Add(var, eps)
    sd = Sqrt(ve)
    rc = Reciprocal(sd)
    s = Mul(rc, scale)
    xs = Mul(x, s)
    ms = Mul(mean, s)
    b2 = Sub(bias, ms)
    y = Add(xs, b2)
"""


def _in_model(body=_IN, c=4, flat=False, opset=13):
    rng = np.random.default_rng(0)
    shape = (c,) if flat else (1, c, 1, 1)
    return _model(
        body,
        [
            _t("eps", 1e-3),
            _t("scale", 1 + rng.random(shape)),
            _t("bias", rng.standard_normal(shape)),
        ],
        opset,
        shape=(2, c, 6, 6),
    )


_L2 = """
    u = Unsqueeze(x, ax)
    sq = Mul(u, u)
    rs = ReduceSum<keepdims=1>(sq, rax)
    mx = Max(rs, eps)
    sr = Sqrt(mx)
    rc = Reciprocal(sr)
    y = Mul(u, rc)
"""


def _l2_model(body=_L2, rax=2):
    return _model(
        body,
        [
            _t("ax", [1], np.int64),
            _t("rax", [rax], np.int64),
            _t("eps", [1e-12]),
        ],
        13,
        shape=(3, 8),
        out_shape=(3, 1, 8),
    )


# -- comparing graphs -----------------------------------------------------------------


def _attr_key(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, onnx.TensorProto):
        v = numpy_helper.to_array(v)
    if isinstance(v, bytes):
        v = v.decode()
    if hasattr(v, "tolist"):
        v = v.tolist()
    return (a.name, tuple(v) if isinstance(v, (list, tuple)) else v)


_DEFAULT_ATTRS = {
    ("auto_pad", "NOTSET"),
    ("group", 1),
    ("ceil_mode", 0),
    ("count_include_pad", 0),
    ("storage_order", 0),
}


def _node_attrs(n):
    return tuple(
        sorted(k for k in map(_attr_key, n.attribute) if k not in _DEFAULT_ATTRS)
    )


def _uniform_qdq(model):
    """Per-axis Q/DQ parameters that are uniform become per-tensor scalars (an
    int32 bias carries one scale per element in onnxsim, one in Quark)."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    inits = {t.name: t for t in out.graph.initializer}
    for n in out.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) > 2:
            s, z = inits.get(n.input[1]), inits.get(n.input[2])
            if s is None or z is None:
                continue
            sa, za = numpy_helper.to_array(s), numpy_helper.to_array(z)
            if (
                (sa.size > 1 or za.size > 1 or sa.ndim)
                and np.all(sa == sa.flat[0])
                and np.all(za == za.flat[0])
            ):
                s.CopyFrom(
                    numpy_helper.from_array(np.array(sa.flat[0], sa.dtype), s.name)
                )
                z.CopyFrom(
                    numpy_helper.from_array(np.array(za.flat[0], za.dtype), z.name)
                )
                for a in list(n.attribute):
                    if a.name == "axis":
                        n.attribute.remove(a)
    return out


def _values(model):
    """Initializers plus Constant nodes' values (onnxsim folds a Constant into an
    initializer, Quark keeps the node)."""
    vals = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            vals[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return vals


def _arr_key(a):
    a = np.ascontiguousarray(a)
    return f"{a.dtype}{list(a.shape)}:{hashlib.md5(a.tobytes()).hexdigest()[:12]}"


def _graph_keys(model):
    """Name-independent signature of every graph output (hash of the whole
    producing sub-graph) and the multiset of local node signatures."""
    model = _uniform_qdq(model)
    vals = _values(model)
    prod = {
        o: n
        for n in model.graph.node
        if not (n.op_type == "Constant" and n.output[0] in vals)
        for o in n.output
    }
    memo = {}

    def sig(t):
        if t not in memo:
            if t == "":
                memo[t] = "-"
            elif t in vals:
                memo[t] = "init:" + _arr_key(vals[t])
            elif t in prod:
                n = prod[t]
                body = f"{n.op_type}{_node_attrs(n)}#{list(n.output).index(t)}("
                body += ",".join(sig(x) for x in n.input) + ")"
                memo[t] = hashlib.md5(body.encode()).hexdigest()[:12]
            else:
                memo[t] = "in:" + t
        return memo[t]

    outs = [sig(o.name) for o in model.graph.output]
    local = []
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] in vals:
            continue
        ins = tuple(
            "I" + _arr_key(vals[x])
            if x in vals
            else "T:" + prod[x].op_type
            if x in prod
            else "in"
            for x in n.input
        )
        local.append((n.op_type, _node_attrs(n), ins))
    return outs, local


def _graph_diff(q, m):
    """``[]`` when the two graphs are the same graph, else the nodes only one of
    them has."""
    from collections import Counter

    sq, lq = _graph_keys(q)
    sm, lm = _graph_keys(m)
    if sq == sm:
        return []
    cq, cm = Counter(lq), Counter(lm)
    return [("Quark only", x) for x in (cq - cm).elements()] + [
        ("onnxsim only", x) for x in (cm - cq).elements()
    ] or [("wiring differs", None)]


def _assert_same_graph(q, m, msg=""):
    diff = _graph_diff(q, m)
    assert not diff, f"{msg}: graphs differ: {diff}"


def _close_summary(model):
    """Op types, the activation quantizers (scale, zero point, dtype) and the
    dequantized constants of a Q/DQ graph, independent of names and order."""
    vals = _values(model)
    ops, acts, consts = [], [], []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in vals:
            zp = vals[n.input[2]]
            acts.append((str(zp.dtype), int(zp), float(vals[n.input[1]])))
        elif n.op_type == "DequantizeLinear" and n.input[0] in vals:
            q = vals[n.input[0]].astype(np.float64)
            z = vals[n.input[2]].astype(np.float64)
            s = vals[n.input[1]].astype(np.float64)
            if s.ndim:
                s = s.reshape([-1] + [1] * (q.ndim - 1)) if q.ndim else s
            consts.append(((q - z) * s, str(vals[n.input[0]].dtype)))
        elif n.op_type not in ("QuantizeLinear", "DequantizeLinear", "Constant"):
            ops.append(n.op_type)
    acts.sort(key=lambda a: (a[0], a[1], round(a[2], 3)))
    consts.sort(key=lambda c: (c[0].size, c[1], round(float(np.abs(c[0]).sum()), 1)))
    return sorted(ops), acts, consts


def _assert_close_graph(q, m, msg=""):
    oq, aq, cq = _close_summary(q)
    om, am, cm = _close_summary(m)
    assert om == oq, msg
    assert [a[:2] for a in am] == [a[:2] for a in aq], msg
    np.testing.assert_allclose(
        [a[2] for a in am], [a[2] for a in aq], rtol=1e-5, err_msg=msg
    )
    assert len(cm) == len(cq), msg
    for (xm, dm), (xq, dq) in zip(cm, cq):
        assert dm == dq, msg
        assert xm.shape == xq.shape, msg
        # int32 biases: a code or two of calibration noise
        atol = 1e-3 * max(float(np.abs(xq).max()), 1e-6) if dq == "int32" else 1e-5
        np.testing.assert_allclose(xm, xq, rtol=1e-5, atol=atol, err_msg=msg)
