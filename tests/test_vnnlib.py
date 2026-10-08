"""Tests for onnxsim.vnnlib: the VNN-LIB subset parser and the unsat / sat / unknown verdicts.

Everything here is small, deterministic and offline. The public-benchmark conformance runs live
in scripts/vnncomp_bench.py (not part of CI) and are described in docs/vnnlib-bench.md.
"""

import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import vnnlib as V


def _verify(*args, **kwargs):
    """``vnnlib.verify`` with a small attack budget: the tests need verdicts, not exhaustive searches."""
    kwargs.setdefault("attack_seconds", 0.3)
    return V.verify(*args, **kwargs)


_HDR = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n"
_HDR1 = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n"  # one output


def _net(body, init, nin=2, nout=1):
    m = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 13]> g (float[1,{nin}] x) => (float[1,{nout}] y) {{ {body} }}'
    )
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in init.items())
    onnx.checker.check_model(m)
    return m


def _f(*v):
    return np.array(v, dtype=np.float32)


# y = 2*x0 + x1 on a box in [0, 1]^2: range [0, 3]
_LINEAR = _net("y = MatMul(x, W)", {"W": _f(2.0, 1.0).reshape(2, 1)})
_BOX01 = "(assert (>= X_0 0.0))(assert (<= X_0 1.0))(assert (>= X_1 0.0))(assert (<= X_1 1.0))"
_ROOT_ENGINES = ("ibp", "zonotope", "crown")


# ---- parser ------------------------------------------------------------------


def test_input_box_and_one_output_atom():
    p = V.parse(
        _HDR
        + "(assert (>= X_0 -1))(assert (<= X_0 2))(assert (>= X_1 0))(assert (<= X_1 0.5))"
        "(assert (>= Y_0 3.5))"
    )
    assert (p.n_inputs, p.n_outputs, len(p.clauses)) == (2, 2, 1)
    c = p.clauses[0]
    assert list(c.lo) == [-1.0, 0.0] and list(c.hi) == [2.0, 0.5]
    (atom,) = c.atoms
    assert (
        list(atom.a) == [-1.0, 0.0] and atom.b == -3.5 and not atom.strict
    )  # -Y_0 <= -3.5


def test_comments_and_whitespace_are_ignored():
    p = V.parse(
        "; a comment\n" + _HDR + _BOX01 + "\n; more\n(assert   (<=  Y_0   1.0 ))"
    )
    assert len(p.clauses) == 1 and p.clauses[0].atoms[0].b == 1.0


def test_or_of_ands_gives_one_clause_per_disjunct():
    p = V.parse(
        _HDR + _BOX01 + "(assert (or (and (>= Y_0 Y_1)) (and (<= Y_0 1) (>= Y_1 2))))"
    )
    assert [len(c.atoms) for c in p.clauses] == [1, 2]
    a = p.clauses[0].atoms[0]
    assert list(a.a) == [-1.0, 1.0] and a.b == 0.0  # Y_0 >= Y_1  <=>  -Y_0 + Y_1 <= 0


def test_input_bounds_may_sit_inside_the_disjunction():
    # the shape of VNN-COMP's test_small.vnnlib: the box is part of the or/and
    p = V.parse(
        "(declare-const X_0 Real)(declare-const Y_0 Real)"
        "(assert (or (and (>= X_0 -1) (<= X_0 0) (>= Y_0 5)) (and (>= X_0 0.5) (<= X_0 1) (>= Y_0 7))))"
    )
    assert [(float(c.lo[0]), float(c.hi[0])) for c in p.clauses] == [
        (-1.0, 0.0),
        (0.5, 1.0),
    ]


@pytest.mark.parametrize(
    "expr, a, b",
    [
        ("(<= (+ Y_0 (* -1.0 Y_1)) 0.5)", [1.0, -1.0], 0.5),
        ("(<= (- Y_0 Y_1) 0.5)", [1.0, -1.0], 0.5),
        ("(<= (* 2 (+ Y_0 1)) 6)", [2.0, 0.0], 4.0),  # 2*Y_0 + 2 <= 6
        ("(<= (/ Y_0 2) 1)", [0.5, 0.0], 1.0),
        ("(<= (- Y_0) 1)", [-1.0, 0.0], 1.0),
        ("(>= 3 (+ Y_0 Y_1))", [1.0, 1.0], 3.0),  # constants may be on the left
    ],
)
def test_linear_terms_are_normalised_to_a_dot_y_le_b(expr, a, b):
    p = V.parse(_HDR + _BOX01 + f"(assert {expr})")
    (atom,) = p.clauses[0].atoms
    assert list(atom.a) == a and atom.b == pytest.approx(b)


def test_equality_fixes_an_input_and_gives_two_output_atoms():
    p = V.parse(
        _HDR
        + "(assert (= X_0 0.5))(assert (>= X_1 0))(assert (<= X_1 1))(assert (= Y_0 1))"
    )
    c = p.clauses[0]
    assert c.lo[0] == c.hi[0] == 0.5
    assert len(c.atoms) == 2


def test_strict_atoms_keep_their_strictness_for_replay():
    p = V.parse(_HDR + _BOX01 + "(assert (< Y_0 1))")
    atom = p.clauses[0].atoms[0]
    assert atom.strict
    y_edge = np.array([1.0, 0.0])
    assert not p.violated_by(np.array([0.5, 0.5]), y_edge)  # Y_0 = 1 is NOT < 1
    assert p.violated_by(np.array([0.5, 0.5]), np.array([0.99, 0.0]))


def test_an_empty_input_box_clause_is_dropped_and_constant_atoms_are_decided():
    p = V.parse(
        _HDR
        + "(assert (>= X_0 2))(assert (<= X_0 1))(assert (<= X_1 1))(assert (>= Y_0 0))"
    )
    assert p.clauses == []  # no input satisfies the box: no unsafe point
    assert V.parse(
        _HDR + _BOX01 + "(assert (<= 1 2))(assert (>= Y_0 0))"
    ).clauses  # true atom: neutral
    assert (
        V.parse(_HDR + _BOX01 + "(assert (<= 2 1))(assert (>= Y_0 0))").clauses == []
    )  # false


@pytest.mark.parametrize(
    "text, why",
    [
        ("(assert (not (>= Y_0 0)))", "unsupported formula operator"),
        ("(assert (<= X_0 Y_0))", "mixing inputs and outputs"),
        ("(assert (<= (+ X_0 X_1) 1))", "several input variables"),
        ("(assert (<= (* Y_0 Y_1) 1))", "non-linear"),
        ("(assert (<= (/ Y_0 Y_1) 1))", "division by a non-constant"),
        ("(assert (<= Z_0 1))", "unknown symbol"),
        ("(assert (=> (>= Y_0 0) (>= Y_1 0)))", "unsupported formula operator"),
        ("(assert (<= (max Y_0 Y_1) 1))", "unsupported operator"),
        ("(declare-network net)", "unsupported top-level form"),
        ("(assert (<= Y_0))", "two operands"),
    ],
)
def test_unsupported_input_fails_loudly(text, why):
    with pytest.raises(V.VnnlibError, match=why):
        V.parse(_HDR + _BOX01 + text)


def test_malformed_text_fails_loudly():
    for bad in (
        "(assert (<= Y_0 1)",
        "(assert (<= Y_0 1)))",
        "(declare-const X_0 Int)",
        "(declare-const foo Real)",
    ):
        with pytest.raises(V.VnnlibError):
            V.parse(_HDR + bad)
    with pytest.raises(V.VnnlibError, match="no inputs or no outputs"):
        V.parse("(declare-const X_0 Real)")
    with pytest.raises(V.VnnlibError, match="not declared"):
        V.parse(
            "(declare-const X_0 Real)(declare-const Y_0 Real)(assert (<= X_5 1))(assert (>= Y_0 0))"
        )


def test_parse_file_and_text_detection(tmp_path):
    text = (
        "(declare-const X_0 Real)(declare-const X_1 Real)(declare-const Y_0 Real)"
        + _BOX01
        + "(assert (>= Y_0 3.5))"
    )
    path = tmp_path / "p.vnnlib"
    path.write_text(text)
    assert V.parse_file(str(path)).n_inputs == 2
    assert (
        _verify(_LINEAR, str(path), "crown", attack=False).status == V.UNSAT
    )  # a path is accepted
    assert (
        _verify(_LINEAR, text, "crown", attack=False).status == V.UNSAT
    )  # and so is the text


def test_output_count_mismatch_is_a_clear_unsupported_not_an_engine_error():
    v = _verify(
        _LINEAR, _HDR + _BOX01 + "(assert (>= Y_0 3.5))", "crown", attack=False
    )  # 2 outputs declared
    assert (
        v.status == V.UNSUPPORTED and "2 outputs but the network produces 1" in v.detail
    )


# ---- verdicts ----------------------------------------------------------------

_SAFE = _HDR1 + _BOX01 + "(assert (>= Y_0 3.5))"  # max y = 3
_UNSAFE = _HDR1 + _BOX01 + "(assert (>= Y_0 2.5))"  # y = 3 at (1, 1)
_FLIPPED = _HDR1 + _BOX01 + "(assert (<= Y_0 3.5))"  # all of the box


@pytest.mark.parametrize("engine", _ROOT_ENGINES + ("bab",))
def test_safe_unsafe_and_the_direction_of_the_inequality(engine):
    assert _verify(_LINEAR, _SAFE, engine, attack=False).status == V.UNSAT
    v = _verify(_LINEAR, _UNSAFE, engine)
    assert v.status == V.SAT and v.counterexample is not None
    # swapping >= for <= turns the same threshold into "everything is unsafe"
    assert _verify(_LINEAR, _FLIPPED, engine).status == V.SAT


def test_find_counterexample_is_independent_of_the_bounds_and_never_claims_safety():
    cex = V.find_counterexample(_LINEAR, _UNSAFE, seconds=2.0)
    assert cex is not None
    x, y = cex
    assert y[0] >= 2.5 and np.all(x >= 0.0) and np.all(x <= 1.0)
    # the property is safe: nothing to find (None means "not found", it is not a proof)
    assert V.find_counterexample(_LINEAR, _SAFE, seconds=0.5) is None
    # an unbounded box cannot be attacked
    unbounded = _HDR1 + "(assert (>= X_0 0))(assert (>= Y_0 3.5))"
    assert V.find_counterexample(_LINEAR, unbounded, seconds=0.5) is None


def test_attack_finds_a_thin_unsafe_region_that_plain_sampling_misses():
    # y = x0 - x1 with a thin unsafe sliver |y| < 1e-4 at the box edge: (y >= -1e-4 and y <= 0.0)
    # random sampling hits a 1e-4-thin slice of a unit box with probability ~1e-4; PGD walks into it
    net = _net("y = MatMul(x, W)", {"W": _f(1.0, -1.0).reshape(2, 1)})
    prop = (
        "(declare-const X_0 Real)(declare-const X_1 Real)(declare-const Y_0 Real)"
        "(assert (>= X_0 0.5))(assert (<= X_0 1.0))(assert (>= X_1 0.5))(assert (<= X_1 1.0))"
        "(assert (<= Y_0 0.0))(assert (>= Y_0 -0.0001))"
    )
    cex = V.find_counterexample(net, prop, seconds=5.0, seed=3)
    assert cex is not None and -1e-4 <= cex[1][0] <= 0.0


def test_counterexample_replays_on_onnxruntime_and_stays_in_the_box():
    v = _verify(_LINEAR, _UNSAFE, "crown")
    x, y = v.counterexample
    sess = ort.InferenceSession(
        _LINEAR.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got = sess.run(None, {"x": x.astype(np.float32).reshape(1, 2)})[0].reshape(-1)
    assert got[0] >= 2.5 and np.allclose(
        got, y, atol=0
    )  # the stored y is the replayed output
    assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_sat_is_never_claimed_from_an_overapproximation():
    # IBP of relu(x0-x1) + relu(x1-x0) over [0,1]^2 is [0, 2] but the true max is 1:
    # "Y_0 >= 1.5" is safe, and IBP must say unknown, not sat.
    net = _net(
        "d = MatMul(x, D)\n e = MatMul(x, E)\n a = Relu(d)\n b = Relu(e)\n y = Add(a, b)",
        {"D": _f(1, -1).reshape(2, 1), "E": _f(-1, 1).reshape(2, 1)},
    )
    prop = _HDR1 + _BOX01 + "(assert (>= Y_0 1.5))"
    assert _verify(net, prop, "ibp").status == V.UNKNOWN
    assert (
        _verify(net, prop, "crown").status == V.UNSAT
    )  # the triangle relaxation closes the gap
    assert _verify(net, prop, "bab").status == V.UNSAT


def test_branch_and_bound_decides_a_region_when_ANY_atom_is_infeasible_on_it():
    # y = (x, x); unsafe iff Y_0 <= -0.1 AND Y_1 >= 0.1 -- impossible, but neither atom is infeasible on the
    # whole box [-1, 1]: only after splitting does the left half kill one atom and the right half the other.
    net = _net("y = MatMul(x, W)", {"W": _f(1, 1).reshape(1, 2)}, nin=1, nout=2)
    prop = (
        "(declare-const X_0 Real)(declare-const Y_0 Real)(declare-const Y_1 Real)"
        "(assert (>= X_0 -1))(assert (<= X_0 1))(assert (<= Y_0 -0.1))(assert (>= Y_1 0.1))"
    )
    assert (
        _verify(net, prop, "crown").status == V.UNKNOWN
    )  # no single atom is excluded at the root
    v = _verify(net, prop, "bab", budget=20)
    assert v.status == V.UNSAT and v.regions >= 2


def test_any_clause_left_open_keeps_the_property_unproven_and_a_counterexample_wins():
    prop = (
        "(declare-const X_0 Real)(declare-const X_1 Real)(declare-const Y_0 Real)"
        "(assert (>= X_0 0))(assert (<= X_0 1))(assert (>= X_1 0))(assert (<= X_1 1))"
        "(assert (or (and (>= Y_0 3.5)) (and (>= Y_0 2.9))))"  # second clause: y=3 at (1,1)
    )
    v = _verify(_LINEAR, prop, "crown")
    assert v.status == V.SAT and v.clauses == 2
    both_safe = prop.replace("(>= Y_0 2.9)", "(>= Y_0 3.2)")
    assert _verify(_LINEAR, both_safe, "crown", attack=False).status == V.UNSAT


def test_empty_unsafe_region_and_no_output_constraint():
    nothing = V.parse(
        _HDR1 + "(assert (>= X_0 2))(assert (<= X_0 1))(assert (>= Y_0 0))"
    )
    assert _verify(_LINEAR, nothing, "crown").status == V.UNSAT
    whole_box = _HDR1 + _BOX01
    assert _verify(_LINEAR, whole_box, "crown").status == V.SAT


def test_unsupported_and_unbounded_cases_are_reported_not_guessed():
    one_input = "(declare-const X_0 Real)(declare-const Y_0 Real)(assert (>= X_0 0))(assert (<= X_0 1))(assert (>= Y_0 1))"
    v = _verify(
        _LINEAR, one_input, "crown", attack=False
    )  # 1 declared input, the net takes 2
    assert v.status == V.UNSUPPORTED and "1 inputs but the network takes 2" in v.detail
    two_in = _net("a = Add(x, x)\n y = MatMul(a, W)", {"W": _f(1, 1).reshape(2, 1)})
    two_in.graph.input.append(
        onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1, 2])
    )
    assert _verify(two_in, _SAFE, "crown").status == V.UNSUPPORTED
    unbounded = _HDR1 + "(assert (>= X_0 0))(assert (>= Y_0 3.5))"
    # X_1 has no bounds at all: a valid property, but nothing can be proved (and nothing is attackable)
    assert _verify(_LINEAR, unbounded, "crown").status == V.UNKNOWN
    with pytest.raises(ValueError, match="engine must be one of"):
        _verify(_LINEAR, _SAFE, "nonsense")


def test_non_float32_coefficients_are_refused():
    prop = _HDR1 + _BOX01 + "(assert (>= (* 0.1 Y_0) 0.35))"
    assert _verify(_LINEAR, prop, "crown", attack=False).status == V.UNSUPPORTED


def test_net_output_bounds_helper_matches_the_exact_range_of_a_linear_net():
    lo, hi = V.net_output_bounds(_LINEAR, np.zeros(2), np.ones(2), "crown")
    assert (
        float(lo[0]) <= 0.0 + 1e-6
        and float(hi[0]) >= 3.0 - 1e-6
        and float(hi[0]) <= 3.0 + 1e-6
    )
    with pytest.raises(ValueError):
        V.net_output_bounds(_LINEAR, np.zeros(2), np.ones(2), "bab")


# ---- soundness property test -------------------------------------------------


def _random_relu_net(rng, nin=3, hid=6, nout=3):
    w1 = (rng.standard_normal((nin, hid)) * 0.9).astype(np.float32)
    b1 = (rng.standard_normal(hid) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((hid, hid)) * 0.9).astype(np.float32)
    b2 = (rng.standard_normal(hid) * 0.3).astype(np.float32)
    w3 = (rng.standard_normal((hid, nout)) * 0.9).astype(np.float32)
    net = _net(
        "a = MatMul(x, W1)\n b = Add(a, B1)\n r = Relu(b)\n c = MatMul(r, W2)\n d = Add(c, B2)\n s = Relu(d)\n"
        "y = MatMul(s, W3)",
        {"W1": w1, "B1": b1, "W2": w2, "B2": b2, "W3": w3},
        nin=nin,
        nout=nout,
    )

    def forward(x):
        r = np.maximum(x @ w1 + b1, 0.0)
        s = np.maximum(r @ w2 + b2, 0.0)
        return s @ w3

    return net, forward


@pytest.mark.parametrize("seed", range(10))
def test_no_engine_says_unsat_when_sampling_finds_an_unsafe_point(seed):
    rng = np.random.default_rng(seed)
    net, forward = _random_relu_net(rng)
    lo = rng.uniform(-1.0, 0.0, 3)
    hi = lo + rng.uniform(0.1, 1.0, 3)
    xs = lo + (hi - lo) * rng.random((4000, 3))
    ys = forward(xs.astype(np.float32))
    spread = ys[:, 0] - ys[:, 1]
    # thresholds around the sampled range of Y_0 - Y_1: some safe-ish, some clearly unsafe
    for c in (
        float(spread.max()) + 0.3,
        float(spread.max()) + 0.02,
        float(spread.max()) - 0.05,
        float(np.median(spread)),
    ):
        text = (
            "".join(f"(declare-const X_{i} Real)" for i in range(3))
            + "".join(f"(declare-const Y_{j} Real)" for j in range(3))
            + "".join(
                f"(assert (>= X_{i} {float(lo[i])!r}))(assert (<= X_{i} {float(hi[i])!r}))"
                for i in range(3)
            )
            + f"(assert (>= (- Y_0 Y_1) {c!r}))"
        )
        prop = V.parse(text)
        sampled_unsafe = any(prop.violated_by(x, y) for x, y in zip(xs, ys))
        verdicts = {}
        for engine in ("ibp", "zonotope", "crown", "bab"):
            verdicts[engine] = _verify(
                net, prop, engine, attack=False, budget=12, timeout=20
            ).status
            if verdicts[engine] == V.UNSAT:
                assert not sampled_unsafe, (
                    seed,
                    c,
                    engine,
                    "unsat but a sampled point is unsafe",
                )
        # tighter engines never lose a proof a looser one found
        order = ["ibp", "crown", "bab"]
        for loose, tight in zip(order, order[1:]):
            if verdicts[loose] == V.UNSAT:
                assert verdicts[tight] == V.UNSAT, (seed, c, loose, tight)


def test_sat_counterexamples_always_replay_on_the_given_network():
    rng = np.random.default_rng(99)
    net, forward = _random_relu_net(rng)
    lo, hi = np.full(3, -0.5), np.full(3, 0.5)
    ys = forward((lo + (hi - lo) * rng.random((2000, 3))).astype(np.float32))
    c = float(np.quantile(ys[:, 0] - ys[:, 1], 0.5))  # half the box is unsafe
    text = (
        "".join(f"(declare-const X_{i} Real)" for i in range(3))
        + "".join(f"(declare-const Y_{j} Real)" for j in range(3))
        + "".join(f"(assert (>= X_{i} -0.5))(assert (<= X_{i} 0.5))" for i in range(3))
        + f"(assert (>= (- Y_0 Y_1) {c!r}))"
    )
    prop = V.parse(text)
    v = _verify(net, prop, "crown")
    assert v.status == V.SAT
    x, y = v.counterexample
    assert prop.violated_by(x, y) and np.allclose(
        y, forward(x.astype(np.float32)), atol=1e-5
    )


# ---- VNN-COMP run_instance protocol ----------------------------------------------


def _instance(tmp_path, prop_text, model=_LINEAR):
    onnx_path = tmp_path / "net.onnx"
    onnx.save(model, onnx_path)
    vnn = tmp_path / "prop.vnnlib"
    vnn.write_text(prop_text)
    return str(onnx_path), str(vnn), str(tmp_path / "result.txt")


def test_run_instance_writes_the_protocol_word(tmp_path):
    onnx_path, vnn, out = _instance(tmp_path, _SAFE)
    assert V.run_instance(onnx_path, vnn, out, 30, engine="crown") == "unsat"
    with open(out) as f:
        assert f.read() == "unsat\n"


def test_run_instance_sat_writes_the_counterexample_block(tmp_path):
    onnx_path, vnn, out = _instance(tmp_path, _UNSAFE)
    assert V.run_instance(onnx_path, vnn, out, 30, engine="crown") == "sat"
    with open(out) as f:
        lines = f.read().splitlines()
    assert lines[0] == "sat" and lines[1] == "(" and lines[-1] == ")"
    pairs = [line[1:-1].split(" ") for line in lines[2:-1]]
    assert [name for name, _ in pairs] == ["X_0", "X_1", "Y_0"]
    assert all(np.isfinite(float(v)) for _, v in pairs)


def test_run_instance_maps_unsupported_to_unknown_and_failures_to_error(tmp_path):
    nonlinear = _HDR1 + _BOX01 + "(assert (<= (* Y_0 Y_0) 1.0))"
    onnx_path, vnn, out = _instance(tmp_path, nonlinear)
    assert V.run_instance(onnx_path, vnn, out, 30) == "unknown"
    assert (
        V.run_instance(onnx_path, str(tmp_path / "missing.vnnlib"), out, 30) == "error"
    )
    with open(out) as f:
        assert f.read() == "error\n"


def test_watchdog_writes_timeout_and_exits_cleanly_when_the_answer_is_late(tmp_path):
    onnx_path, vnn, out = _instance(tmp_path, _SAFE)
    code = (
        "import time\n"
        "import onnxsim.vnnlib as V\n"
        "V.verify = lambda *a, **k: time.sleep(60)\n"
        f"V.main(['run', {onnx_path!r}, {vnn!r}, {out!r}, '1'])\n"
        "raise SystemExit('watchdog did not end the process')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode()
    with open(out) as f:
        assert f.read() == "timeout\n"


def test_cli_run_subcommand(tmp_path):
    onnx_path, vnn, out = _instance(tmp_path, _UNSAFE)
    assert V.main(["run", onnx_path, vnn, out, "30", "--engine", "crown"]) == 0
    with open(out) as f:
        assert f.readline() == "sat\n"
