"""Tests for onnxsim.shape_cost: certified memory/compute bounds under dynamic shapes.

The core property, checked against onnxruntime: for dynamic dims sampled inside the given
ranges, *every* tensor of a real run (shape, element count, bytes) lies inside its certified
bound, the liveness peak of the real run lies inside ``peak_live_bytes``, and (with every dim
fixed) the bounds collapse to the exact figures ``onnxsim.model_info`` reports.
"""

import contextlib
import io
import json
import pathlib
import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import shape_cost as SC
from onnxsim import shape_ranges as sr


def _model(body, inits=None, opset=17):
    m = parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (inits or {}).items()
    )
    onnx.checker.check_model(m)
    return m


def _f(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


# --------------------------------------------------------------------------- models


def conv_net(seed=0):
    r = np.random.default_rng(seed)
    return _model(
        """g (float[N,3,H,W] x) => (float[N,10] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          r1 = Relu(c1)
          p1 = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r1)
          c2 = Conv<pads=[1,1,1,1], strides=[2,2]>(p1, W2, B2)
          r2 = Relu(c2)
          gp = GlobalAveragePool(r2)
          fl = Flatten(gp)
          y = Gemm<transB=1>(fl, W3, B3)
        }""",
        dict(
            W1=_f(r, 8, 3, 3, 3), B1=_f(r, 8), W2=_f(r, 16, 8, 3, 3), B2=_f(r, 16),
            W3=_f(r, 10, 16), B3=_f(r, 10),
        ),
    )  # fmt: skip


def transformer_block(seed=0, d=16, heads=4):
    r = np.random.default_rng(seed)
    dh = d // heads

    def w(*s):
        return (_f(r, *s) * 0.2).astype(np.float32)

    return _model(
        f"""g (float[B,S,{d}] x) => (float[B,S,{d}] y) {{
          ln = LayerNormalization<axis=-1, epsilon=1e-5>(x, G, Bt)
          q = MatMul(ln, Wq)
          k = MatMul(ln, Wk)
          v = MatMul(ln, Wv)
          q4 = Reshape(q, shp)
          k4 = Reshape(k, shp)
          v4 = Reshape(v, shp)
          qt = Transpose<perm=[0,2,1,3]>(q4)
          kt = Transpose<perm=[0,2,3,1]>(k4)
          vt = Transpose<perm=[0,2,1,3]>(v4)
          sc = MatMul(qt, kt)
          scs = Mul(sc, scale)
          pr = Softmax<axis=-1>(scs)
          ctx = MatMul(pr, vt)
          ct = Transpose<perm=[0,2,1,3]>(ctx)
          cf = Reshape(ct, shp2)
          o = MatMul(cf, Wo)
          r1 = Add(x, o)
          h = MatMul(r1, W1)
          hd = Div(h, rt2)
          he = Erf(hd)
          hp = Add(he, one)
          hm = Mul(h, hp)
          hg = Mul(hm, half)
          h2 = MatMul(hg, W2)
          y = Add(r1, h2)
        }}""",
        dict(
            G=np.ones(d, np.float32), Bt=np.zeros(d, np.float32),
            Wq=w(d, d), Wk=w(d, d), Wv=w(d, d), Wo=w(d, d), W1=w(d, 4 * d), W2=w(4 * d, d),
            shp=np.array([0, 0, heads, dh], np.int64), shp2=np.array([0, 0, d], np.int64),
            scale=np.float32(1 / np.sqrt(dh)), rt2=np.float32(np.sqrt(2)),
            one=np.float32(1), half=np.float32(0.5),
        ),
    )  # fmt: skip


def shape_chain(seed=0):
    r = np.random.default_rng(seed)
    return _model(
        """g (float[B,S,8] x) => (float[T,3] y) {
          sh = Shape(x)
          b = Gather<axis=0>(sh, i0)
          s = Gather<axis=0>(sh, i1)
          t = Mul(b, s)
          tu = Unsqueeze(t, ax0)
          tgt = Concat<axis=0>(tu, eight)
          fl = Reshape(x, tgt)
          y = MatMul(fl, W)
        }""",
        dict(
            i0=np.int64(0), i1=np.int64(1), ax0=np.array([0], np.int64),
            eight=np.array([8], np.int64), W=_f(r, 8, 3),
        ),
    )  # fmt: skip


def nonzero_net():
    return _model(
        """g (float[N,4] x) => (int64[2,M] nz, float[1] s) {
          nz = NonZero(x)
          c = Cast<to=1>(nz)
          s0 = ReduceSum<keepdims=0>(c)
          s = Unsqueeze(s0, ax)
        }""",
        dict(ax=np.array([0], np.int64)),
    )


def compress_net(seed=0):
    r = np.random.default_rng(seed)
    return _model(
        """g (float[N,6] x, bool[N] c) => (float[K,3] y) {
          kept = Compress<axis=0>(x, c)
          rr = Relu(kept)
          y = MatMul(rr, W)
        }""",
        dict(W=_f(r, 6, 3)),
    )


def topk_net():
    return _model(
        """g (float[N,10] x, int64[1] k) => (float[N,K] v, int64[N,K] i) {
          v0, i = TopK<axis=1>(x, k)
          v = Relu(v0)
        }"""
    )


def nms_net():
    return _model(
        """g (float[1,B,4] boxes, float[1,1,B] scores) => (int64[R,3] sel) {
          sel = NonMaxSuppression(boxes, scores, mx)
        }""",
        dict(mx=np.array([5], np.int64)),
    )


# --------------------------------------------------------------------------- harness


def run_all_tensors(model, feeds):
    """Run onnxruntime exposing every node output; returns {name: ndarray}."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for o in [o for n in m.graph.node for o in n.output if o and o not in have]:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    out.update({k: np.asarray(v) for k, v in feeds.items()})
    for t in model.graph.initializer:
        out[t.name] = numpy_helper.to_array(t)
    return out


def violations(cb, vals):
    """Tensors of an actual run that escape their certified shape / byte bounds."""
    bad = []
    for name, t in cb.tensors.items():
        if name not in vals:
            continue
        a = vals[name]
        if t.shape is not None and not sr.contains_shape(t.shape, a.shape):
            bad.append((name, "shape", a.shape, sr.shape_str(t.shape)))
        if t.bytes.hi is not None and a.nbytes > t.bytes.hi:
            bad.append((name, "bytes_hi", a.nbytes, t.bytes.hi))
        if a.nbytes < t.bytes.lo:
            bad.append((name, "bytes_lo", a.nbytes, t.bytes.lo))
    return bad


def actual_peak(model, vals):
    """Independent re-implementation of model_info's liveness peak, on actual sizes."""
    g = model.graph
    weights = {t.name for t in g.initializer}
    size = {n: v.nbytes for n, v in vals.items()}
    resident = sum(size[n] for n in weights)
    last = {}
    for i, node in enumerate(g.node):
        for x in node.input:
            if x:
                last[x] = i
    for o in g.output:
        last[o.name] = len(g.node)
    live = {i.name for i in g.input if i.name not in weights}
    peak = resident + sum(size[n] for n in live)
    for i, node in enumerate(g.node):
        for o in node.output:
            if o and o not in weights:
                live.add(o)
        peak = max(peak, resident + sum(size[n] for n in live))
        for n in [n for n in live if last.get(n) == i]:
            live.discard(n)
    return peak


def check_runs(model, cb, feed_list):
    for feeds in feed_list:
        vals = run_all_tensors(model, feeds)
        assert violations(cb, vals) == []
        pk = actual_peak(model, vals)
        assert cb.peak_live_bytes.lo <= pk <= cb.peak_live_bytes.hi, (
            pk,
            cb.peak_live_bytes,
        )
        if cb.arena is not None:  # every tensor of the run fits the slot it was given
            for name, (_off, size) in cb.arena.tensor_offsets.items():
                if name in vals:
                    assert vals[name].nbytes <= size, name


# --------------------------------------------------------------------------- whole models


def test_conv_net_dynamic_batch_and_image_size():
    m = conv_net()
    cb = SC.bounds(m, {"N": (1, 4), "H": (8, 32), "W": (8, 32)})
    assert cb.complete and cb.unbounded_tensors == []
    rng = np.random.default_rng(1)
    feeds = [
        {"x": _f(rng, n, 3, h, w)}
        for n, h, w in [(1, 8, 8), (4, 32, 32), (2, 17, 23), (3, 9, 31)]
    ]
    check_runs(m, cb, feeds)
    assert cb.arena is not None and cb.arena.verified
    # the upper bound is attained at the extreme shape (no looseness for plain conv/pool/gemm)
    vals = run_all_tensors(m, feeds[1])
    assert vals["c1"].nbytes == cb.tensors["c1"].bytes.hi
    assert vals["c2"].nbytes == cb.tensors["c2"].bytes.hi


def test_transformer_block_dynamic_batch_and_sequence():
    m = transformer_block()
    cb = SC.bounds(m, {"B": (1, 4), "S": (1, 64)})
    assert cb.complete and cb.unbounded_tensors == []
    rng = np.random.default_rng(2)
    feeds = [{"x": _f(rng, b, s, 16)} for b, s in [(1, 1), (4, 64), (2, 17), (3, 33)]]
    check_runs(m, cb, feeds)
    # attention scores are quadratic in the sequence length: the bound sees it
    assert cb.tensors["sc"].bytes.hi == 4 * 4 * 64 * 64 * 4
    vals = run_all_tensors(m, feeds[1])
    assert actual_peak(m, vals) == cb.peak_live_bytes.hi  # tight at the extreme shape


def test_shape_arithmetic_chain_through_shape_gather_mul_concat_reshape():
    m = shape_chain()
    cb = SC.bounds(m, {"B": (1, 4), "S": (1, 10)})
    assert cb.complete and cb.unbounded_tensors == []
    rng = np.random.default_rng(3)
    check_runs(m, cb, [{"x": _f(rng, b, s, 8)} for b, s in [(1, 1), (4, 10), (2, 7)]])
    assert (
        cb.arena is not None and cb.arena.verified
    )  # rank-0 tensors (scalars) are planned too


def test_nonzero_count_is_bounded_by_the_element_count():
    m = nonzero_net()
    cb = SC.bounds(m, {"N": (1, 6)})
    assert cb.tensors["nz"].bytes.hi == 2 * 6 * 4 * 8  # at most numel columns of int64
    rng = np.random.default_rng(4)
    feeds = []
    for n in (1, 6, 3):
        x = _f(rng, n, 4) * (rng.random((n, 4)) > 0.5)
        feeds.append({"x": x})
    feeds.append({"x": np.zeros((2, 4), np.float32)})
    check_runs(m, cb, feeds)


def test_compress_and_topk_and_nms_counts():
    rng = np.random.default_rng(5)
    m = compress_net()
    cb = SC.bounds(m, {"N": (1, 8)})
    check_runs(
        m,
        cb,
        [{"x": _f(rng, n, 6), "c": rng.random(n) > 0.5} for n in (1, 8, 5)]
        + [{"x": _f(rng, 4, 6), "c": np.zeros(4, bool)}],
    )
    m = topk_net()
    cb = SC.bounds(m, {"N": (1, 4)}, input_ranges={"k": (1, 4)})
    assert cb.tensors["v0"].bytes.hi == 4 * 4 * 4  # K is at most 4
    check_runs(
        m,
        cb,
        [
            {"x": _f(rng, n, 10), "k": np.array([k], np.int64)}
            for n, k in [(1, 1), (4, 4), (2, 3)]
        ],
    )
    m = nms_net()
    cb = SC.bounds(m, {"B": (1, 20)})
    assert (
        cb.tensors["sel"].bytes.hi == 5 * 3 * 8
    )  # batches * classes * max_per_class rows
    feeds = []
    for n in (1, 20, 9):
        xy = rng.random((1, n, 2)) * 0.5
        wh = rng.random((1, n, 2)) * 0.5 + 0.05
        boxes = np.concatenate([xy, xy + wh], axis=2).astype(np.float32)
        feeds.append(
            {"boxes": boxes, "scores": rng.random((1, 1, n)).astype(np.float32)}
        )
    check_runs(m, cb, feeds)


# --------------------------------------------------------------------------- conventions


@pytest.mark.parametrize("n,h,w", [(2, 16, 16), (1, 8, 12), (3, 31, 17)])
def test_fixed_shapes_collapse_to_model_info(n, h, w):
    """With every dim fixed the bounds are points equal to onnxsim.model_info's figures."""
    from onnxsim.model_info import ModelInfo

    m = conv_net()
    cb = SC.bounds(m, input_shapes={"x": [n, 3, h, w]})
    mm = onnx.ModelProto()
    mm.CopyFrom(m)
    for d, v in zip(mm.graph.input[0].type.tensor_type.shape.dim, (n, 3, h, w)):
        d.ClearField("dim_param")
        d.dim_value = v
    info = ModelInfo(mm)
    assert cb.macs.lo == cb.macs.hi == int(info.macs)
    assert cb.mem_access_bytes.lo == cb.mem_access_bytes.hi == int(info.mem_access)
    assert cb.peak_live_bytes.lo == cb.peak_live_bytes.hi == int(info.memory_footprint)
    assert cb.flops == SC.Bound(2 * int(info.macs), 2 * int(info.macs))


def test_macs_are_ranges_with_the_exact_endpoints():
    m = conv_net()
    cb = SC.bounds(m, {"N": (1, 4), "H": (8, 32), "W": (8, 32)})
    lo = SC.bounds(m, input_shapes={"x": [1, 3, 8, 8]}).macs
    hi = SC.bounds(m, input_shapes={"x": [4, 3, 32, 32]}).macs
    assert cb.macs == SC.Bound(lo.lo, hi.hi)


# --------------------------------------------------------------------------- honesty


def test_a_dimension_without_a_range_is_unbounded_and_says_so():
    m = conv_net()
    cb = SC.bounds(m, {"N": (1, 4)})  # H and W have no range
    assert cb.peak_live_bytes.hi is None and cb.macs.hi is None
    assert "x" in cb.unbounded_tensors
    assert any("without a range" in n for n in cb.notes)
    # a lower bound is still stated
    assert cb.peak_live_bytes.lo > 0


def test_unknown_domain_op_leaves_its_outputs_unbounded():
    m = parser.parse_model(
        """<ir_version: 8, opset_import: ["" : 17, "com.example" : 1]>
        g (float[N,4] x) => (float[N,4] y) {
          a = Relu(x)
          b = com.example.Mystery(a)
          y = Relu(b)
        }"""
    )
    cb = SC.bounds(m, {"N": (1, 4)}, plan_arena=False)
    assert "b" in cb.unbounded_tensors and "shape unknown" in cb.tensors["b"].note
    assert "Mystery" in cb.tensors["b"].note
    assert (
        cb.peak_live_bytes.hi is None
    )  # a total that depends on an unbounded tensor is unbounded


def test_control_flow_subgraph_makes_totals_unbounded_not_undercounted():
    then_g = parser.parse_graph(
        "then_g () => (float[2] t) { t = Constant<value=float[2] {1.0, 2.0}>() }"
    )
    else_g = parser.parse_graph(
        "else_g () => (float[2] e) { e = Constant<value=float[2] {3.0, 4.0}>() }"
    )
    node = onnx.helper.make_node(
        "If", ["c"], ["y"], then_branch=then_g, else_branch=else_g
    )
    g = onnx.helper.make_graph(
        [node],
        "g",
        [onnx.helper.make_tensor_value_info("c", onnx.TensorProto.BOOL, [])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 17)])
    m.ir_version = 8
    cb = SC.bounds(m)
    assert not cb.complete and cb.macs.hi is None and cb.peak_live_bytes.hi is None
    assert cb.arena is None
    assert any("control-flow subgraph" in n for n in cb.notes)


def test_dtype_widths_drive_bytes():
    m = _model(
        """g (float16[N,4] a, int8[N,4] b) => (float16[N,4] y, int8[N,4] z) {
          y = Relu(a)
          z = Neg(b)
        }"""
    )
    cb = SC.bounds(m, {"N": (2, 5)})
    assert cb.tensors["a"].bytes == SC.Bound(2 * 4 * 2, 5 * 4 * 2)
    assert cb.tensors["b"].bytes == SC.Bound(2 * 4, 5 * 4)


# --------------------------------------------------------------------------- arena


def test_arena_plan_is_verified_and_covers_every_shape_in_range():
    m = transformer_block()
    cb = SC.bounds(m, {"B": (1, 3), "S": (1, 16)})
    assert cb.arena is not None and cb.arena.verified
    assert cb.arena.arena_bytes <= cb.arena.naive_bytes
    # every tensor's slot is at least its upper-bound size and inside the arena
    for name, (off, size) in cb.arena.tensor_offsets.items():
        assert size >= cb.tensors[name].bytes.hi
        assert off + size <= cb.arena.arena_bytes


def test_binary_op_aliasing_is_not_trusted_at_dynamic_shapes():
    """Add(a[N,8], b[M,8]) has equal byte size to its output at the upper bounds, yet at N=1,
    M=4 the input is broadcast: computing in place would overwrite it while it is still read."""
    m = _model(
        """g (float[N,8] a, float[M,8] b) => (float[Q,8] y) {
          r = Relu(a)
          s = Add(r, b)
          y = Relu(s)
        }"""
    )
    cb = SC.bounds(m, {"N": (1, 4), "M": (1, 4)})
    assert cb.arena is not None and cb.arena.verified
    off = {k: v[0] for k, v in cb.arena.tensor_offsets.items()}
    assert off["s"] != off["r"]  # the Add output was not placed on its operand
    # the verifier rejects a plan that does alias them, whoever produced it
    t = cb.tensors
    bad = {
        "a": (0, 128),
        "b": (128, 128),
        "r": (256, 128),
        "s": (256, 128),
        "y": (384, 128),
    }
    ok, why = SC._verify_plan(m, bad, 1024, t, {"a", "b"}, {"y"})
    assert not ok and "live together" in why
    good = {
        "a": (0, 128),
        "b": (128, 128),
        "r": (256, 128),
        "s": (384, 128),
        "y": (256, 128),
    }
    assert SC._verify_plan(m, good, 1024, t, {"a", "b"}, {"y"})[0]


def test_binary_aliasing_is_allowed_when_shapes_are_provably_identical():
    m = _model(
        """g (float[2,8] a, float[2,8] b) => (float[2,8] y) {
          r = Relu(a)
          y = Add(r, b)
        }"""
    )
    cb = SC.bounds(m)
    t = cb.tensors
    node = m.graph.node[1]
    assert SC._alias_allowed(node, "r", "y", t)  # static, identical shapes
    dyn = _model(
        """g (float[N,8] a, float[N,8] b) => (float[N,8] y) {
          r = Relu(a)
          y = Add(r, b)
        }"""
    )
    cd = SC.bounds(dyn, {"N": (1, 4)})
    assert not SC._alias_allowed(dyn.graph.node[1], "r", "y", cd.tensors)


def test_verify_plan_rejects_small_slots_and_overlaps():
    m = conv_net()
    cb = SC.bounds(m, {"N": (1, 2), "H": (8, 8), "W": (8, 8)})
    assert cb.arena is not None
    offs = dict(cb.arena.tensor_offsets)
    n = cb.arena.arena_bytes
    t = cb.tensors
    ins, outs = {"x"}, {"y"}
    assert SC._verify_plan(m, offs, n, t, ins, outs)[0]
    shrunk = dict(offs)
    shrunk["c1"] = (offs["c1"][0], 1)
    assert (
        "smaller than its upper bound" in SC._verify_plan(m, shrunk, n, t, ins, outs)[1]
    )
    outside = dict(offs)
    outside["c1"] = (n, offs["c1"][1])
    assert "leaves the arena" in SC._verify_plan(m, outside, n, t, ins, outs)[1]
    stacked = {k: (0, v[1]) for k, v in offs.items()}  # everything at offset 0
    assert not SC._verify_plan(m, stacked, n, t, ins, outs)[0]


# --------------------------------------------------------------------------- budget + CLI


def test_check_budget_statuses():
    m = conv_net()
    dims = {"N": (1, 4), "H": (8, 32), "W": (8, 32)}
    cb = SC.bounds(m, dims)
    big = cb.peak_live_bytes.hi + 1
    v = SC.check_budget(m, dims, memory_bytes=big, macs=cb.macs.hi)
    assert v.fits is True and all(c.status == "proved" for c in v.checks)
    v = SC.check_budget(
        m, dims, memory_bytes=cb.peak_live_bytes.hi - 1
    )  # reachable shape exceeds it?
    assert (
        v.fits is None and v.checks[0].status == "unknown"
    )  # may fit at smaller shapes
    v = SC.check_budget(m, dims, memory_bytes=cb.peak_live_bytes.lo - 1)
    assert (
        v.fits is False and v.checks[0].status == "exceeds"
    )  # even the smallest shape needs more
    v = SC.check_budget(m, {"N": (1, 4)}, macs=10**12)  # H, W unbounded
    assert v.fits is None and v.checks[0].status == "unknown"
    assert "UNKNOWN" in str(v)
    v = SC.check_budget(m, dims, arena_bytes=cb.arena.arena_bytes)
    assert v.fits is True and v.checks[0].metric == "arena_bytes"
    assert SC.check_budget(m, dims, arena_bytes=cb.arena.arena_bytes - 1).fits is None


def test_cli_json_and_text(tmp_path, capsys):
    p = tmp_path / "m.onnx"
    onnx.save(conv_net(), str(p))
    rc = SC.main(
        [
            str(p),
            "--dim",
            "N=1:4",
            "--dim",
            "H=8:32",
            "--dim",
            "W=8:32",
            "--json",
            "--tensors",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["complete"] is True
    assert out["peak_live_bytes"]["hi"] >= out["peak_live_bytes"]["lo"] > 0
    assert out["tensors"]["c1"]["bytes"]["hi"] == 4 * 8 * 32 * 32 * 4
    rc = SC.main([str(p), "--dim", "N=1:4", "--dim", "H=8:32", "--dim", "W=8:32"])
    text = capsys.readouterr().out
    assert rc == 0 and "peak live bytes" in text and "MACs" in text
    rc = SC.main(
        [
            str(p),
            "--dim",
            "N=1:4",
            "--dim",
            "H=8:32",
            "--dim",
            "W=8:32",
            "--budget-memory",
            "1",
        ]
    )
    assert rc == 1  # provably does not fit
    capsys.readouterr()


def test_dim_range_forms():
    m = conv_net()
    a = SC.bounds(m, {"N": 2, "H": (16, 16), "W": 16})
    b = SC.bounds(m, input_shapes={"x": [2, 3, 16, 16]})
    assert a.macs == b.macs and a.peak_live_bytes == b.peak_live_bytes
    c = SC.bounds(m, {"N": (1, None), "H": 8, "W": 8})  # open upper end
    assert c.peak_live_bytes.hi is None


def test_overriding_a_static_input_is_not_clamped_by_its_declared_shape():
    """ONNX infers shapes from the *declared* inputs. A model exported with static shapes and then
    analysed with an input_shapes range must not have every tensor clamped back to the export size."""
    m = _model(
        """g (float[2,3,8,8] x) => (float[2,8,8,8] y) {
          c = Conv<pads=[1,1,1,1]>(x, W)
          y = Relu(c)
        }""",
        dict(
            W=(np.random.default_rng(0).standard_normal((8, 3, 3, 3)) * 0.1).astype(
                np.float32
            )
        ),
    )
    static = SC.bounds(m)
    assert static.tensors["c"].bytes == SC.Bound(2 * 8 * 8 * 8 * 4, 2 * 8 * 8 * 8 * 4)
    dyn = SC.bounds(m, input_shapes={"x": [(1, 4), 3, (8, 16), (8, 16)]})
    assert dyn.tensors["x"].bytes.hi == 4 * 3 * 16 * 16 * 4
    assert dyn.tensors["c"].bytes == SC.Bound(1 * 8 * 8 * 8 * 4, 4 * 8 * 16 * 16 * 4)
    assert dyn.tensors["y"].bytes.hi == 4 * 8 * 16 * 16 * 4
    # and a real run at an in-range shape the export never saw stays inside the bounds
    rng = np.random.default_rng(1)
    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    for d, v in zip(m2.graph.input[0].type.tensor_type.shape.dim, ("N", 3, "H", "W")):
        d.ClearField("dim_value")
        if isinstance(v, str):
            d.dim_param = v
        else:
            d.dim_value = v
    vals = run_all_tensors(m2, {"x": _f(rng, 3, 3, 11, 13)})
    assert violations(dyn, vals) == []


# --------------------------------------------------------------------------- shape rules vs onnxruntime

_X4 = "float[N,4,H,W] x"
_DIMS4 = {"N": (1, 3), "H": (6, 20), "W": (6, 20)}


def _case(name, body, inits, outs, dims=None, feeds=None, opset=17, tight=()):
    return pytest.param(
        name, body, inits, outs, dims or _DIMS4, feeds, opset, tight, id=name
    )


def _w(*shape):
    return (np.random.default_rng(0).standard_normal(shape) * 0.1).astype(np.float32)


_RULE_CASES = [
    _case(
        "conv_stride2_dil2_pad",
        f"g ({_X4}) => (float[N,6,P,Q] y) {{ y = Conv<strides=[2,2], dilations=[2,2], pads=[1,2,1,2]>(x, W) }}",
        dict(W=_w(6, 4, 3, 3)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "conv_group",
        f"g ({_X4}) => (float[N,8,P,Q] y) {{ y = Conv<group=2, pads=[1,1,1,1]>(x, W) }}",
        dict(W=_w(8, 2, 3, 3)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "conv_same_upper",
        f'g ({_X4}) => (float[N,6,P,Q] y) {{ y = Conv<auto_pad="SAME_UPPER", strides=[2,2]>(x, W) }}',
        dict(W=_w(6, 4, 3, 3)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "conv_1d",
        "g (float[N,4,L] x) => (float[N,6,M] y) { y = Conv<strides=[3]>(x, W) }",
        dict(W=_w(6, 4, 5)),
        ["y"],
        dims={"N": (1, 3), "L": (10, 40)},
        tight=("y",),
    ),
    _case(
        "maxpool_ceil",
        f"g ({_X4}) => (float[N,4,P,Q] y) {{ y = MaxPool<kernel_shape=[3,3], strides=[2,2], ceil_mode=1>(x) }}",
        {},
        ["y"],
        tight=("y",),
    ),
    _case(
        "avgpool_pad",
        f"g ({_X4}) => (float[N,4,P,Q] y) {{ y = AveragePool<kernel_shape=[2,2], pads=[1,1,1,1]>(x) }}",
        {},
        ["y"],
        tight=("y",),
    ),
    _case(
        "convtranspose",
        f"g ({_X4}) => (float[N,5,P,Q] y) {{ y = ConvTranspose<strides=[2,2], output_padding=[1,1], pads=[1,1,1,1]>(x, W) }}",
        dict(W=_w(4, 5, 3, 3)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "resize_nearest_x2",
        f'g ({_X4}) => (float[N,4,P,Q] y) {{ y = Resize<mode="nearest">(x, roi, sc) }}',
        dict(roi=np.zeros(0, np.float32), sc=np.array([1, 1, 2, 2], np.float32)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "resize_linear_x1p5",
        f'g ({_X4}) => (float[N,4,P,Q] y) {{ y = Resize<mode="linear">(x, roi, sc) }}',
        dict(roi=np.zeros(0, np.float32), sc=np.array([1, 1, 1.5, 1.5], np.float32)),
        ["y"],
    ),
    _case(
        "resize_sizes",
        f'g ({_X4}) => (float[N,4,P,Q] y) {{ y = Resize<mode="nearest">(x, roi, sc, sz) }}',
        dict(
            roi=np.zeros(0, np.float32),
            sc=np.zeros(0, np.float32),
            sz=np.array([2, 4, 7, 9], np.int64),
        ),
        ["y"],
    ),
    _case(
        "split_sizes",
        f"g ({_X4}) => (float[N,1,H,W] a, float[N,3,H,W] b) {{ a, b = Split<axis=1>(x, sp) }}",
        dict(sp=np.array([1, 3], np.int64)),
        ["a", "b"],
        tight=("a", "b"),
    ),
    _case(
        "split_equal_dynamic_axis",
        "g (float[N,L] x) => (float[N,M] a, float[N,M] b, float[N,M] c) { a, b, c = Split<axis=1, num_outputs=3>(x) }",
        {},
        ["a", "b", "c"],
        dims={"N": (1, 3), "L": (3, 30)},
        opset=18,
    ),
    _case(
        "gemm_transA_transB",
        "g (float[K,M] a) => (float[M,5] y) { y = Gemm<transA=1, transB=1>(a, B) }",
        dict(B=_w(5, 7)),
        ["y"],
        dims={"M": (1, 6), "K": (7, 7)},
    ),
    _case(
        "batchnorm",
        f"g ({_X4}) => (float[N,4,H,W] y) {{ y = BatchNormalization(x, g, b, m, v) }}",
        dict(
            g=np.ones(4, np.float32),
            b=np.zeros(4, np.float32),
            m=np.zeros(4, np.float32),
            v=np.ones(4, np.float32),
        ),
        ["y"],
        tight=("y",),
    ),
    _case(
        "layernorm_stats",
        "g (float[N,L,8] x) => (float[N,L,8] y, float[N,L,1] mu, float[N,L,1] inv) { y, mu, inv = LayerNormalization<axis=-1>(x, g, b) }",
        dict(g=np.ones(8, np.float32), b=np.zeros(8, np.float32)),
        ["y", "mu", "inv"],
        dims={"N": (1, 3), "L": (1, 9)},
        tight=("y", "mu", "inv"),
    ),
    _case(
        "reduce_l2_keepdims0",
        f"g ({_X4}) => (float[N,4] y) {{ y = ReduceL2<axes=[2,3], keepdims=0>(x) }}",
        {},
        ["y"],
        tight=("y",),
    ),
    _case(
        "argmax",
        f"g ({_X4}) => (int64[N,H,W] y) {{ y = ArgMax<axis=1, keepdims=0>(x) }}",
        {},
        ["y"],
        tight=("y",),
    ),
    _case(
        "einsum_attention",
        'g (float[B,S,8] q, float[B,T,8] k) => (float[B,S,T] y) { y = Einsum<equation="bsd,btd->bst">(q, k) }',
        {},
        ["y"],
        dims={"B": (1, 3), "S": (1, 7), "T": (1, 9)},
        feeds="qk",
        tight=("y",),
    ),
    _case(
        "depthtospace_spacetodepth",
        "g (float[N,8,H,W] x) => (float[N,2,P,Q] d, float[N,8,H,W] r) { d = DepthToSpace<blocksize=2>(x)\n r = SpaceToDepth<blocksize=2>(d) }",
        {},
        ["d", "r"],
        dims={"N": (1, 3), "H": (3, 9), "W": (3, 9)},
        tight=("d", "r"),
    ),
    _case(
        "onehot",
        "g (int64[N] i) => (float[N,5] y) { y = OneHot<axis=-1>(i, depth, vals) }",
        dict(depth=np.array(5, np.int64), vals=np.array([0, 1], np.float32)),
        ["y"],
        dims={"N": (1, 6)},
        feeds="onehot",
        tight=("y",),
    ),
    _case(
        "scatter_gather_elements",
        "g (float[N,4] d, int64[N,2] idx, float[N,2] u) => (float[N,4] s, float[N,2] g) { s = ScatterElements<axis=1>(d, idx, u)\n g = GatherElements<axis=1>(d, idx) }",
        {},
        ["s", "g"],
        dims={"N": (1, 5)},
        feeds="scatter",
        tight=("s", "g"),
    ),
    _case(
        "pow_broadcast",
        "g (float[N,1,W] a, float[N,H,1] b) => (float[N,H,W] y) { y = Pow(a, b) }",
        {},
        ["y"],
        dims={"N": (1, 3), "H": (1, 6), "W": (1, 6)},
        feeds="pow",
    ),
    _case(
        "quant_dequant",
        f"g ({_X4}) => (float[N,4,H,W] y) {{ q = QuantizeLinear(x, s, z)\n y = DequantizeLinear(q, s, z) }}",
        dict(s=np.float32(0.1), z=np.int8(0)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "where_broadcast_mask",
        "g (float[N,1,1,W] m, float[N,4,H,W] s) => (float[N,4,H,W] y) { c = Greater(m, zero)\n y = Where(c, s, neg) }",
        dict(zero=np.float32(0), neg=np.float32(-1e9)),
        ["y", "c"],
        dims={"N": (1, 3), "H": (2, 7), "W": (2, 7)},
    ),
    _case(
        "globalmaxpool",
        f"g ({_X4}) => (float[N,4,1,1] y) {{ y = GlobalMaxPool(x) }}",
        {},
        ["y"],
        tight=("y",),
    ),
    _case(
        "dynamic_quantize_linear",
        f"g ({_X4}) => (uint8[N,4,H,W] q, float s, uint8 z) {{ q, s, z = DynamicQuantizeLinear(x) }}",
        {},
        ["q", "s", "z"],
        tight=("q",),
    ),
    _case(
        "sum_broadcast_three",
        "g (float[N,1,W] a, float[N,H,1] b, float[1,H,W] c) => (float[N,H,W] y) { y = Sum(a, b, c) }",
        {},
        ["y"],
        dims={"N": (1, 3), "H": (1, 6), "W": (1, 6)},
    ),
    _case(
        "matmulinteger",
        "g (uint8[M,6] a) => (int32[M,5] y) { y = MatMulInteger(a, B) }",
        dict(B=np.random.default_rng(1).integers(-5, 5, (6, 5)).astype(np.int8)),
        ["y"],
        dims={"M": (1, 9)},
        tight=("y",),
    ),
    _case(
        "convinteger",
        "g (uint8[N,4,H,W] x) => (int32[N,6,P,Q] y) { y = ConvInteger<strides=[2,2], pads=[1,1,1,1]>(x, W) }",
        dict(W=np.random.default_rng(2).integers(-5, 5, (6, 4, 3, 3)).astype(np.int8)),
        ["y"],
        tight=("y",),
    ),
    _case(
        "gridsample",
        "g (float[N,3,H,W] x, float[N,P,Q,2] g) => (float[N,3,P,Q] y) { y = GridSample<align_corners=1>(x, g) }",
        {},
        ["y"],
        dims={"N": (1, 3), "H": (4, 9), "W": (4, 9), "P": (2, 7), "Q": (2, 7)},
        tight=("y",),
    ),
    _case(
        "roialign",
        "g (float[N,3,H,W] x, float[R,4] rois, int64[R] bi) => (float[R,3,2,2] y) { y = RoiAlign<output_height=2, output_width=2, spatial_scale=1.0, sampling_ratio=2>(x, rois, bi) }",
        {},
        ["y"],
        dims={"N": (1, 3), "H": (8, 12), "W": (8, 12), "R": (1, 6)},
        feeds="roi",
        tight=("y",),
    ),
]


@pytest.mark.parametrize("name,body,inits,outs,dims,feeds,opset,tight", _RULE_CASES)
def test_shape_rule_against_onnxruntime(
    name, body, inits, outs, dims, feeds, opset, tight
):
    m = _model(body, inits, opset=opset)
    cb = SC.bounds(m, dims, plan_arena=False)
    for o in outs:
        assert cb.tensors[o].shape is not None, f"{name}: no shape rule for {o}"
        assert cb.tensors[o].bytes.bounded, f"{name}: {o} is unbounded"
    rng = np.random.default_rng(7)
    extremes = [
        {k: lo for k, (lo, hi) in dims.items()},
        {k: hi for k, (lo, hi) in dims.items()},
    ]
    samples = extremes + [
        {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in dims.items()}
        for _ in range(4)
    ]
    for pick in samples:
        feed = _make_feed(m, feeds, pick, rng)
        vals = run_all_tensors(m, feed)
        assert violations(cb, vals) == [], (name, pick)
        if (
            pick == extremes[1]
        ):  # at the upper corner the bound is attained for shape-exact rules
            for o in tight:
                assert vals[o].nbytes == cb.tensors[o].bytes.hi, (name, o)


def _make_feed(m, kind, pick, rng):
    """Concrete inputs for ``pick`` (a value for each named dim)."""
    feeds = {}
    for vi in m.graph.input:
        if vi.name in {t.name for t in m.graph.initializer}:
            continue
        dims = [
            d.dim_value if d.HasField("dim_value") else pick[d.dim_param]
            for d in vi.type.tensor_type.shape.dim
        ]
        et = vi.type.tensor_type.elem_type
        if et == onnx.TensorProto.UINT8:
            feeds[vi.name] = rng.integers(0, 255, size=dims).astype(np.uint8)
        elif et == onnx.TensorProto.INT64:
            hi = 5 if kind == "onehot" else (4 if kind == "scatter" else 3)
            if kind == "roi":  # RoiAlign batch indices: all rois on image 0
                feeds[vi.name] = np.zeros(dims, np.int64)
            else:
                feeds[vi.name] = rng.integers(0, hi, size=dims).astype(np.int64)
        elif et == onnx.TensorProto.FLOAT and kind == "roi" and len(dims) == 2:
            feeds[vi.name] = (
                rng.random(dims).astype(np.float32) * 6
            )  # x1,y1,x2,y2 inside the image
        elif et == onnx.TensorProto.FLOAT and vi.name == "g" and len(dims) == 4:
            feeds[vi.name] = (rng.random(dims) * 2 - 1).astype(
                np.float32
            )  # grid in [-1, 1]
        elif et == onnx.TensorProto.FLOAT and kind == "pow":
            feeds[vi.name] = (rng.random(dims) + 0.5).astype(np.float32)
        else:
            feeds[vi.name] = rng.standard_normal(dims).astype(np.float32)
    return feeds


# --------------------------------------------------------------------------- the doc stays true

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "shape-cost.md"
_BLOCK = re.compile(
    r"<!-- doctest -->\n```python\n(.*?)```\n```text\n(.*?)```", re.DOTALL
)


def test_doc_examples_print_what_the_doc_says():
    """Run every ``<!-- doctest -->`` block of docs/shape-cost.md (one shared namespace, in order)
    and compare the output quoted right after it, so the page cannot drift from the code."""
    blocks = _BLOCK.findall(_DOC.read_text())
    assert len(blocks) >= 2
    namespace = {"__name__": "__doc__"}
    for i, (code, expected) in enumerate(blocks):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"{_DOC.name}[{i}]", "exec"), namespace)
        assert out.getvalue().strip() == expected.strip(), f"doc block {i}"
