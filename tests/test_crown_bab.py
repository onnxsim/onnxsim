"""Tests for the branch-and-bound / multi-neuron extensions of onnxsim.crown.

What is checked, in order of how much each one would catch:

* Soundness against onnxruntime: sample inputs in the box, expose every intermediate
  tensor, and require each observed value inside the bounds -- for input splitting, Relu
  splitting (with and without multipliers), alpha/beta leaves and the multi-neuron cuts,
  at tiny and exhausted budgets, and on nets with Tanh units that cannot be branched on.
* The Lagrangian machinery itself, with *random* non-negative multipliers (not optimised
  ones): a Relu branch constraint and multi-neuron cuts must give a valid bound for every
  multiplier, on exactly the points of the branch.
* The hull facets of a Relu pair are valid for the true graph on millions of sampled
  points, and over a plain box there are none (a product of triangles).
* Ordering: branch and bound is never looser than the root bound and strictly tighter
  where splitting helps; ``verify_output_ranges(method="bab")`` proves a range plain
  CROWN cannot, and never proves one that real outputs violate.
* Budgets: an exhausted budget returns a sound partial result and says so.
"""

import itertools

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import crown
from onnxsim import ranges as R

HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="alpha/beta/prima need torch")

BOX = {"x": (-1.0, 1.0)}


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _f32(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _mlp(rng, act="Relu", widths=(6, 10, 8, 3)):
    body = [f"m (float[1,{widths[0]}] x) => (float[1,{widths[-1]}] y) {{"]
    inits, prev = {}, "x"
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        inits[f"W{i}"], inits[f"B{i}"] = _f32(rng, a, b), _f32(rng, b)
        last = i == len(widths) - 2
        body.append(f"  g{i} = Gemm({prev}, W{i}, B{i})")
        prev = f"g{i}"
        if not last:
            kind = act if isinstance(act, str) else act[i % len(act)]
            body.append(f"  a{i} = {kind}(g{i})")
            prev = f"a{i}"
    body.append(f"  y = Identity({prev})\n}}")
    return _model("\n".join(body), inits)


def _conv_bn_relu_net(rng):
    k = 4
    return _model(
        """
        m (float[1,3,8,8] x) => (float[1,5] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g, be, mu, var)
          r1 = Relu(b1)
          c2 = Conv<strides=[2,2], pads=[1,1,1,1]>(r1, W2, B2)
          r2 = Relu(c2)
          p = GlobalAveragePool(r2)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W3, B3)
        }""",
        dict(
            W1=_f32(rng, k, 3, 3, 3), B1=_f32(rng, k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k), mu=_f32(rng, k), var=rng.uniform(0.5, 2, k).astype(np.float32),
            W2=_f32(rng, 6, k, 3, 3), B2=_f32(rng, 6), W3=_f32(rng, 5, 6), B3=_f32(rng, 5),
        ),
    )  # fmt: skip


def _produced(model):
    return [o for n in model.graph.node for o in n.output if o]


def _session_all(model):
    """A callable ``x -> {tensor: value}`` with every intermediate tensor exposed."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    names = _produced(m)
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in names)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0].name

    def run(x):
        return dict(zip(names, sess.run(None, {inp: x})))

    return run


def _input_shape(model):
    return tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)


def _assert_encloses(model, got, rng, n=40, slack=1e-4, box=(-1.0, 1.0)):
    run = _session_all(model)
    shape = _input_shape(model)
    for _ in range(n):
        vals = run(rng.uniform(box[0], box[1], shape).astype(np.float32))
        for name, tb in got.items():
            pad = slack * (1.0 + np.abs(vals[name]))
            assert np.all(vals[name] >= tb.lo - pad) and np.all(
                vals[name] <= tb.hi + pad
            ), f"{name} escaped its bounds ({tb.method})"


def _width(tb):
    return float(np.mean(tb.hi - tb.lo))


def _within(inner, outer, tol=1e-9):
    return bool(
        np.all(inner.lo >= outer.lo - tol) and np.all(inner.hi <= outer.hi + tol)
    )


# ---- API validation ---------------------------------------------------------------


def test_argument_validation():
    model = _mlp(np.random.default_rng(0))
    with pytest.raises(ValueError, match="split must be"):
        crown.bab_bounds(model, BOX, split="diagonal")
    with pytest.raises(ValueError, match="leaf_method must be"):
        crown.bab_bounds(model, BOX, leaf_method="gamma")
    with pytest.raises(ValueError, match="budget"):
        crown.bab_bounds(model, BOX, budget=0)
    with pytest.raises(ValueError, match="multi_neuron"):
        crown.bab_bounds(model, BOX, multi_neuron=1, leaf_method="alpha")
    with pytest.raises(ValueError, match="multi_neuron"):
        crown.bab_bounds(model, BOX, multi_neuron=4, leaf_method="crown")
    with pytest.raises(ValueError, match="no range for output"):
        crown.bab_bounds(model, BOX, target={"other": (0.0, 1.0)})
    with pytest.raises(ValueError, match="method must be"):
        crown.bounds(model, BOX, method="gamma")


def test_beta_and_alpha_leaves_without_torch_raise_a_clear_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    model = _mlp(np.random.default_rng(0))
    for leaf in ("alpha", "beta"):
        with pytest.raises(ImportError):
            crown.bab_bounds(model, BOX, leaf_method=leaf)
    # CROWN leaves need no torch at all
    assert crown.bab_bounds(model, BOX, budget=5, split="input").evaluations == 5


# ---- soundness against onnxruntime ------------------------------------------------


@pytest.mark.parametrize("split", ["input", "relu", "auto"])
@pytest.mark.parametrize("budget", [1, 2, 9, 33])
def test_bab_with_crown_leaves_encloses_every_intermediate_tensor(split, budget):
    rng = np.random.default_rng(1)
    model = _mlp(rng)
    res = crown.bab_bounds(
        model, BOX, output=_produced(model), budget=budget, split=split
    )
    _assert_encloses(model, res.bounds, rng)


def test_bab_encloses_a_conv_net_for_both_split_kinds():
    rng = np.random.default_rng(2)
    model = _conv_bn_relu_net(rng)
    for split in ("input", "relu"):
        res = crown.bab_bounds(model, BOX, budget=7, split=split)
        _assert_encloses(model, res.bounds, rng, n=15)


def test_nets_with_tanh_units_fall_back_to_input_splitting_and_stay_sound():
    rng = np.random.default_rng(3)
    model = _mlp(rng, act=("Relu", "Tanh"), widths=(5, 8, 8, 2))
    res = crown.bab_bounds(
        model, {"x": (-0.5, 0.5)}, output=_produced(model), budget=17, split="auto"
    )
    assert res.split in ("relu+input", "input")  # the Tanh layer cannot be branched on
    _assert_encloses(model, res.bounds, rng, box=(-0.5, 0.5))
    only_tanh = _mlp(rng, act="Tanh", widths=(4, 6, 2))
    res = crown.bab_bounds(only_tanh, {"x": (-0.5, 0.5)}, budget=9, split="relu")
    assert res.split == "none" and res.regions == 1  # nothing to branch on: root only


def test_unbounded_inputs_are_returned_as_crown_without_splitting():
    model = _mlp(np.random.default_rng(4))
    res = crown.bab_bounds(model, None, budget=9)
    plain = crown.bounds(model, None, method="crown")
    assert res.regions == 1 and res.split == "none" and not res.exhausted
    out = res.bounds["y"]
    assert not np.any(np.isnan(out.lo)) and not np.any(np.isnan(out.hi))
    assert _within(out, plain["y"])


@needs_torch
@pytest.mark.parametrize("leaf", ["alpha", "beta"])
@pytest.mark.parametrize("split", ["input", "relu"])
def test_alpha_and_beta_leaves_enclose_every_intermediate_tensor(leaf, split):
    rng = np.random.default_rng(5)
    model = _mlp(rng)
    res = crown.bab_bounds(
        model,
        BOX,
        output=_produced(model),
        budget=9,
        split=split,
        leaf_method=leaf,
        alpha_iters=10,
    )
    _assert_encloses(model, res.bounds, rng)


@needs_torch
def test_multi_neuron_cuts_enclose_every_intermediate_tensor():
    rng = np.random.default_rng(6)
    model = _mlp(rng)
    got = crown.bounds(
        model, BOX, output=_produced(model), method="prima", alpha_iters=15
    )
    _assert_encloses(model, got, rng)
    res = crown.bab_bounds(
        model,
        BOX,
        output=_produced(model),
        budget=5,
        leaf_method="beta",
        multi_neuron=3,
        alpha_iters=8,
    )
    _assert_encloses(model, res.bounds, rng)


# ---- the Lagrangian terms, with random multipliers --------------------------------


def _unstable_relus(an):
    return [
        i
        for i, n in enumerate(an.nodes)
        if n.op_type == "Relu" and an._supported(n) and an._relax(i)["unstable"].any()
    ]


def test_branch_constraint_gives_a_valid_bound_for_every_nonnegative_multiplier():
    """Bound on the branch's own points, for random beta: the core beta-CROWN claim."""
    rng = np.random.default_rng(7)
    model = _mlp(rng)
    an = crown._Analyzer(model, BOX)
    an.refine()
    ops = crown._NumpyOps()
    run = _session_all(model)
    shape = _input_shape(model)
    samples = [rng.uniform(-1, 1, shape).astype(np.float32) for _ in range(1500)]
    runs = [run(x) for x in samples]
    idx = _unstable_relus(an)[0]
    pre = an.nodes[idx].input[0]
    flat_ok = 0
    for j in np.flatnonzero(an._relax(idx)["unstable"].reshape(-1))[:4]:
        for sign in (1, -1):
            child = an.clone()
            assert child.restrict(idx, int(j), sign)
            child.refine(force=True)
            if child.infeasible:
                continue
            pts = [r for r in runs if sign * r[pre].reshape(-1)[j] >= 0.0]
            if not pts:
                continue
            flat_ok += 1
            y = np.concatenate([r["y"] for r in pts], 0)
            spec = np.eye(3).reshape((3, 1, 3))
            for beta in (0.0, 0.05, 0.5, 5.0, 50.0):
                extra_in = np.zeros((3,) + tuple(an.ib[pre][0].shape))
                extra_in.reshape(3, -1)[:, j] = -float(sign) * beta
                lb = ops.to_numpy(
                    child._lower(ops, "y", spec, extra={"in": {idx: extra_in}})
                )
                assert np.all(y.min(0) >= lb - 1e-6), (j, sign, beta)
    assert flat_ok >= 4


def test_multi_neuron_cuts_give_a_valid_bound_for_every_nonnegative_multiplier():
    rng = np.random.default_rng(8)
    model = _mlp(rng)
    an = crown._Analyzer(model, BOX)
    an.refine()
    ops = crown._NumpyOps()
    run = _session_all(model)
    shape = _input_shape(model)
    runs = [run(rng.uniform(-1, 1, shape).astype(np.float32)) for _ in range(2000)]
    y_min = np.concatenate([r["y"] for r in runs], 0).min(0)
    idx = _unstable_relus(an)[0]
    pre = an.nodes[idx].input[0]
    unstable = np.flatnonzero(an._relax(idx)["unstable"].reshape(-1))
    used = 0
    for j1, j2 in itertools.combinations(unstable[:4], 2):
        pb = an.pair_bounds(idx, int(j1), int(j2))
        if pb is None:
            continue
        lo, hi = an.ib[pre]
        n_, d_ = crown._pair_facets(
            lo.flat[j1], hi.flat[j1], lo.flat[j2], hi.flat[j2], *pb
        )
        if not len(d_):
            continue
        used += 1
        spec = np.eye(3).reshape((3, 1, 3))
        for scale in (0.05, 0.5, 5.0):
            pi = rng.uniform(0.0, scale, (3, len(d_)))
            ein = np.zeros((3,) + tuple(lo.shape))
            eout = np.zeros_like(ein)
            ein.reshape(3, -1)[:, j1] += pi @ n_[:, 0]
            ein.reshape(3, -1)[:, j2] += pi @ n_[:, 1]
            eout.reshape(3, -1)[:, j1] += pi @ n_[:, 2]
            eout.reshape(3, -1)[:, j2] += pi @ n_[:, 3]
            extra = {"in": {idx: ein}, "out": {idx: eout}, "const": -(pi @ d_)}
            lb = ops.to_numpy(an._lower(ops, "y", spec, extra=extra))
            assert np.all(y_min >= lb - 1e-6), (j1, j2, scale)
    assert used >= 3


def test_zero_multipliers_reproduce_the_plain_bound_exactly():
    rng = np.random.default_rng(9)
    model = _mlp(rng)
    an = crown._Analyzer(model, BOX)
    an.refine()
    ops = crown._NumpyOps()
    spec = np.eye(3).reshape((3, 1, 3))
    plain = ops.to_numpy(an._lower(ops, "y", spec))
    idx = _unstable_relus(an)[0]
    z = np.zeros((3,) + tuple(an.ib[an.nodes[idx].input[0]][0].shape))
    extra = {"in": {idx: z}, "out": {idx: z.copy()}, "const": np.zeros(3)}
    np.testing.assert_array_equal(
        ops.to_numpy(an._lower(ops, "y", spec, extra=extra)), plain
    )


# ---- pair facets ------------------------------------------------------------------


def test_pair_facets_are_valid_for_the_true_relu_pair_graph():
    rng = np.random.default_rng(10)
    worst, checked, nonempty = -np.inf, 0, 0
    for _ in range(120):
        l1, l2 = -rng.uniform(0.05, 3, 2)
        u1, u2 = rng.uniform(0.05, 3, 2)
        t = rng.uniform(0.0, 0.45, 4)
        w_s, w_d = (u1 + u2) - (l1 + l2), (u1 - l2) - (l1 - u2)
        ls, us = l1 + l2 + t[0] * w_s, u1 + u2 - t[1] * w_s
        ld, ud = l1 - u2 + t[2] * w_d, u1 - l2 - t[3] * w_d
        n_, d_ = crown._pair_facets(l1, u1, l2, u2, ls, us, ld, ud)
        z1, z2 = rng.uniform(l1, u1, 6000), rng.uniform(l2, u2, 6000)
        ok = (z1 + z2 >= ls) & (z1 + z2 <= us) & (z1 - z2 >= ld) & (z1 - z2 <= ud)
        z1, z2 = z1[ok], z2[ok]
        if not len(d_) or not len(z1):
            continue
        nonempty += 1
        v = np.stack([z1, z2, np.maximum(z1, 0), np.maximum(z2, 0)], 1)
        worst = max(worst, float((v @ n_.T - d_[None, :]).max()))
        checked += len(z1)
    assert nonempty > 60 and checked > 100_000
    assert worst <= 0.0  # no true graph point violates any facet


def test_a_plain_box_has_no_coupling_facets_because_its_hull_is_a_product():
    # conv(A x B) = conv(A) x conv(B): two independent triangles cannot be cut jointly
    for box in [(-1.0, 2.0, -0.5, 1.5), (-3.0, 0.7, -0.2, 4.0)]:
        n_, d_ = crown._pair_facets(*box)
        assert len(d_) == 0


def test_correlated_neurons_do_get_coupling_facets():
    # z1 and z2 move together: the polygon is a thin diagonal strip, not the box
    n_, d_ = crown._pair_facets(-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -0.2, 0.2)
    assert len(d_) >= 4


# ---- restricting a branch and detecting empty ones --------------------------------


def _chain():
    return _model(
        """
        m (float[1,1] x) => (float[1,1] y) {
          z = Gemm(x, W1, B1)
          a = Relu(z)
          v = Gemm(a, W2, B2)
          r = Relu(v)
          y = Identity(r)
        }""",
        dict(
            W1=np.array([[1.0]], np.float32), B1=np.zeros(1, np.float32),
            W2=np.array([[1.0]], np.float32), B2=np.array([-0.3], np.float32),
        ),
    )  # fmt: skip


def test_refinement_proves_a_branch_empty_and_restrict_refuses_an_empty_cut():
    an = crown._Analyzer(_chain(), BOX)
    an.refine()
    ra, rv = (i for i, n in enumerate(an.nodes) if n.op_type == "Relu")
    assert an._relax(ra)["unstable"].any() and an._relax(rv)["unstable"].any()
    # inactive first neuron: a == 0, so v == -0.3 and "v >= 0" has no points. The cut itself
    # is accepted on the stale box; refinement then proves it empty.
    stale = an.clone()
    assert stale.restrict(ra, 0, -1) and stale.restrict(rv, 0, +1)
    stale.refine(force=True)
    assert stale.infeasible
    # after refining, restrict() sees the empty box directly
    fresh = an.clone()
    assert fresh.restrict(ra, 0, -1)
    fresh.refine(force=True)
    assert not fresh.infeasible
    assert fresh.restrict(rv, 0, +1) is False
    # the other branch is genuinely feasible
    ok = an.clone()
    assert ok.restrict(ra, 0, +1) and ok.restrict(rv, 0, +1)
    ok.refine(force=True)
    assert not ok.infeasible


def test_a_restriction_does_not_leak_into_the_analyser_it_came_from():
    an = crown._Analyzer(_chain(), BOX)
    an.refine()
    before = {k: (v[0].copy(), v[1].copy()) for k, v in an.ib.items()}
    child = an.clone()
    ra = next(i for i, n in enumerate(an.nodes) if n.op_type == "Relu")
    child.restrict(ra, 0, +1)
    child.refine(force=True)
    for k, (lo, hi) in before.items():
        np.testing.assert_array_equal(an.ib[k][0], lo)
        np.testing.assert_array_equal(an.ib[k][1], hi)


# ---- ordering and tightening ------------------------------------------------------


def test_bab_is_never_looser_than_crown_and_strictly_tighter_with_input_splitting():
    model = _mlp(np.random.default_rng(1))
    ibp = crown.bounds(model, BOX, method="ibp")["y"]
    plain = crown.bounds(model, BOX, method="crown")["y"]
    widths = [_width(plain)]
    for budget in (5, 17, 65):
        res = crown.bab_bounds(model, BOX, budget=budget, split="input")
        out = res.bounds["y"]
        assert _within(plain, ibp) and _within(out, plain)
        widths.append(_width(out))
    assert widths == sorted(widths, reverse=True)  # more budget never hurts
    assert widths[-1] < 0.7 * widths[0]


def test_bab_method_on_bounds_reports_what_it_did():
    model = _mlp(np.random.default_rng(1))
    got = crown.bounds(model, BOX, method="bab", budget=9)["y"]
    assert got.method == "bab" and got.regions >= 2 and got.exhausted
    assert _within(got, crown.bounds(model, BOX, method="crown")["y"])
    # a net with nothing to split reports plain crown and a single region
    only_tanh = _mlp(np.random.default_rng(1), act="Tanh", widths=(4, 6, 2))
    got = crown.bounds(
        only_tanh, {"x": (-0.5, 0.5)}, method="bab", budget=9, split="relu"
    )["y"]
    assert got.method == "crown" and got.regions == 1


@needs_torch
def test_ordering_ibp_crown_alpha_bab_with_alpha_leaves():
    model = _mlp(np.random.default_rng(1))
    ibp = crown.bounds(model, BOX, method="ibp")["y"]
    plain = crown.bounds(model, BOX, method="crown")["y"]
    alpha = crown.bounds(model, BOX, method="alpha")["y"]
    bab = crown.bab_bounds(
        model, BOX, budget=17, split="input", leaf_method="alpha"
    ).bounds["y"]
    assert _within(plain, ibp) and _within(alpha, plain) and _within(bab, alpha)
    assert _width(bab) < 0.9 * _width(
        alpha
    )  # splitting adds real tightness on top of alpha


@needs_torch
def test_multipliers_make_relu_splitting_converge_where_plain_splitting_stalls():
    model = _mlp(np.random.default_rng(1))
    kw = dict(budget=65, split="relu", alpha_iters=30)
    nobeta = crown.bab_bounds(model, BOX, leaf_method="alpha", **kw).bounds["y"]
    beta = crown.bab_bounds(model, BOX, leaf_method="beta", **kw).bounds["y"]
    alpha = crown.bounds(model, BOX, method="alpha")["y"]
    assert _within(nobeta, alpha) and _within(beta, nobeta)
    assert _width(beta) < 0.95 * _width(
        nobeta
    )  # the constraint in the dual is what helps


@needs_torch
def test_multi_neuron_cuts_tighten_alpha_crown():
    model = _mlp(np.random.default_rng(1))
    alpha = crown.bounds(model, BOX, method="alpha")["y"]
    prima = crown.bounds(model, BOX, method="prima", multi_neuron=4)["y"]
    assert prima.method == "prima"
    assert _within(prima, alpha)
    assert _width(prima) < 0.97 * _width(alpha)


# ---- verifying output ranges ------------------------------------------------------


def _grid_hull(model, n=81):
    run = _session_all(model)
    g = np.linspace(-1, 1, n)
    out = np.array(
        [run(np.array([[a, b]], np.float32))["y"].ravel() for a in g for b in g]
    )
    return float(out.min()), float(out.max())


def _annotated(model, lo, hi):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    R.set_range(m, "y", lo, hi)
    return m


def test_bab_proves_an_output_range_that_plain_crown_cannot():
    model = _mlp(np.random.default_rng(1), widths=(2, 8, 8, 1))
    lo, hi = _grid_hull(model)
    margin = 0.1 * (hi - lo)
    ann = _annotated(model, lo - margin, hi + margin)
    plain = crown.verify_output_ranges(ann, BOX, method="crown")["y"]
    assert not plain.proved  # CROWN's hull is ~3x the true one here
    got = crown.verify_output_ranges(ann, BOX, method="bab", budget=257, split="input")[
        "y"
    ]
    assert got.proved and got.method == "bab"
    assert lo - margin <= got.hull[0] and got.hull[1] <= hi + margin


def test_bab_never_proves_a_range_that_real_outputs_violate():
    model = _mlp(np.random.default_rng(1), widths=(2, 8, 8, 1))
    lo, hi = _grid_hull(model)
    shrink = 0.05 * (hi - lo)  # the annotation is wrong: outputs reach beyond it
    ann = _annotated(model, lo + shrink, hi - shrink)
    for split in ("input", "relu"):
        got = crown.verify_output_ranges(
            ann, BOX, method="bab", budget=129, split=split
        )["y"]
        assert not got.proved
    # ...yet the proven hull is still a sound enclosure of every real output
    assert got.hull[0] <= lo + 1e-6 and got.hull[1] >= hi - 1e-6


def test_bab_stops_as_soon_as_the_target_is_decided():
    model = _mlp(np.random.default_rng(1), widths=(2, 8, 8, 1))
    lo, hi = _grid_hull(model)
    wide = 5.0 * (hi - lo)
    easy = crown.bab_bounds(
        model, BOX, budget=257, target={"y": (lo - wide, hi + wide)}
    )
    assert easy.proved and easy.evaluations == 1 and not easy.exhausted
    m = 0.1 * (hi - lo)
    hard = crown.bab_bounds(
        model, BOX, budget=257, split="input", target={"y": (lo - m, hi + m)}
    )
    assert hard.proved and 1 < hard.evaluations < 257 and not hard.exhausted


@needs_torch
def test_beta_leaves_prove_a_range_by_relu_splitting():
    model = _mlp(np.random.default_rng(4), widths=(2, 8, 8, 1))
    lo, hi = _grid_hull(model)
    margin = 0.15 * (hi - lo)
    ann = _annotated(model, lo - margin, hi + margin)
    assert not crown.verify_output_ranges(ann, BOX, method="crown")["y"].proved
    for leaf in ("alpha", "beta"):
        res = crown.bab_bounds(
            ann,
            BOX,
            budget=129,
            split="relu",
            leaf_method=leaf,
            alpha_iters=15,
            target={"y": (lo - margin, hi + margin)},
        )
        assert res.proved, leaf


# ---- budgets ----------------------------------------------------------------------


def test_an_exhausted_budget_returns_a_sound_partial_result_and_says_so():
    rng = np.random.default_rng(1)
    model = _mlp(rng)
    root = crown.bounds(model, BOX, output=_produced(model), method="crown")
    res = crown.bab_bounds(model, BOX, output=_produced(model), budget=3, split="input")
    assert res.exhausted and res.evaluations == 3 and res.regions == 2
    assert all(_within(res.bounds[n], root[n]) for n in root)
    _assert_encloses(model, res.bounds, rng)
    assert crown.bab_bounds(model, BOX, budget=1).evaluations == 1
    # a time limit of zero stops before any split but still returns the root, soundly
    res = crown.bab_bounds(model, BOX, budget=500, time_limit=0.0)
    assert res.evaluations == 1 and res.exhausted
    assert _within(res.bounds["y"], root["y"])


def test_budget_is_a_hard_cap_and_regions_partition_the_box():
    model = _mlp(np.random.default_rng(1))
    for budget in (1, 2, 3, 4, 5, 9, 17):
        res = crown.bab_bounds(model, BOX, budget=budget, split="input")
        assert (
            res.evaluations == 1 + 2 * ((budget - 1) // 2) <= budget
        )  # a split bounds two
        # each split replaces one region by two, minus any branch proven empty
        assert res.regions == 1 + (res.evaluations - 1) // 2 - res.pruned
