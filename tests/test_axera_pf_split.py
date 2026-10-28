"""Host-only checks of the split linears and Adds of the Pulsar-free decode: the
registry wiring (``pf_registry.split_keys`` / ``add_split``: a width is split only
where it is measured absent, the shipped key lists are unchanged, the missing-key
check no longer fires for a split key), the manifest trees (``pf_ksplit``: the
leaves in run order, the transforms of each leaf), the K-split grids of
``pf_generate.ksplit_grids`` (one shared output grid), and the card runner's byte
offsets (``pf_plan.Plan.chain_tree`` / ``chain_add`` on a fake card that records the
BIND commands). No build, no card, no checkpoint: the programs are fake, the grids
are numpy arrays."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import pf_generate as pg  # noqa: E402
import pf_ksplit as K  # noqa: E402
import pf_plan as P  # noqa: E402
import pf_registry as R  # noqa: E402
import pf_templates as tpl  # noqa: E402

#: the repository's manifest: Qwen3-0.6B and llama-160m have no absent widths
MAN = R.load_manifest()
Q4B = R.model_spec("qwen3_4b", MAN)
Q06 = R.model_spec("qwen3_0.6b", MAN)
LLAMA = R.model_spec("llama160m", MAN)


# ---- registry: which widths are split -------------------------------------------------
def test_shipped_models_have_no_split_at_any_layout():
    for spec in (Q06, LLAMA):
        for layout in R.LAYOUTS:
            base = R.keys_for(spec, layout)
            names, plans = R.split_keys(base, MAN)
            assert names == base and plans == {}, (spec.name, layout)


def test_qwen3_4b_absent_widths_are_split_and_the_rest_are_not():
    base = R.keys_for(Q4B, 1)
    names, plans = R.split_keys(base, MAN)
    assert set(plans) == {"k9728n2560", "add19456"}
    assert plans["k9728n2560"]["names"] == ["k4864n2560"] * 2
    assert plans["add19456"] == {"parts": 2, "width": 9728, "names": ["add9728"] * 2}
    assert "k9728n2560" not in names and "add19456" not in names
    # the other linears and Adds (measured present, or not measured) stay as they are
    for keep in ("k2560n19456", "k4096n2560", "k2560n151936", "add2560", "silu9728"):
        assert keep in names
    assert len(names) == len(set(names))  # deduplicated


def test_add_split_only_for_measured_absent_widths():
    assert R.add_split(19456, MAN)["width"] == 9728
    assert R.add_split(9728, MAN) is None  # measured present
    assert R.add_split(2560, MAN) is None  # measured present
    assert R.add_split(1024, MAN) is None  # not measured at all


def test_add_split_into_no_present_width_is_an_error():
    man = {"add_widths": {"measured": {"19456": {"present": False}}}}
    with pytest.raises(ValueError):
        R.add_split(19456, man)


def test_missing_key_check_does_not_fire_for_split_keys(tmp_path):
    """Every key of the split list is present (a signed directory each): the plan has
    nothing missing; the unsplit list asks for the absent widths and is missing them."""
    names, plans = R.split_keys(R.keys_for(Q4B, 1), MAN)
    root = str(tmp_path / "t")
    for n in names:
        d = tpl.template_dir(root, n)
        os.makedirs(d, exist_ok=True)
        sig = R.signature(n, Q4B.dims())
        R._write_sig(d, n, R.sig_id(sig), sig, Q4B)
    ch = R.plan(Q4B, 1, [R.Root(root)], {}, names=names)
    assert [c.name for c in ch if c.how == "missing"] == []
    base = R.plan(Q4B, 1, [R.Root(root)], {})
    missing = {c.name for c in base if c.how == "missing"}
    assert missing == {"k9728n2560", "add19456"}


def test_linear_split_keeps_its_plan_shape():
    p = R.linear_split(9728, 2560, {"linear_widths": {"2560": {"measured": {
        "4864": {"present": True}, "9728": {"present": False}}}}})  # fmt: skip
    assert p == {
        "parts": 2,
        "k_part": 4864,
        "names": ["k4864n2560"] * 2,
        "add": "add2560",
    }


# ---- manifest trees ------------------------------------------------------------------
def _lin(name, n=2560):
    return {"kind": "linear", "op": "Linear", "shape": [4864, n], "bytes": 1}


def _add(n):
    return {
        "kind": "add",
        "op": "Add",
        "calibration": {"a": [1.0, 0], "b": [1.0, 0], "y": [1.0, 0]},
    }


def _ksplit(k, n=2560):
    return {"kind": "ksplit", "op": "KSplitLinear", "k_part": 4864,
            "parts": [_lin(f"p{i}", n) for i in range(2)], "add": _add(n)}  # fmt: skip


def _gu_split():
    col = {"kind": "colsplit", "op": "ColumnSplitAdd", "col": [0, 9728],
           "parts": [_add(9728), _add(9728)]}  # fmt: skip
    return {"kind": "split", "op": "SplitLinear",
            "split": {"hi": _lin("hi", 19456), "lo": _lin("lo", 19456), "add": col}}  # fmt: skip


def test_ksplit_leaves_are_parts_then_add():
    assert [n for n, _ in K.leaves(_ksplit(9728), "down")] == [
        "down_p0", "down_p1", "down_add"]  # fmt: skip


def test_split_leaves_nest_the_ksplit_halves():
    e = {
        "kind": "split",
        "split": {"hi": _ksplit(9728), "lo": _ksplit(9728), "add": _add(2560)},
    }
    names = [n for n, _ in K.leaves(e, "down")]
    assert names == ["down_hi_p0", "down_hi_p1", "down_hi_add",
                     "down_lo_p0", "down_lo_p1", "down_lo_add", "down_add"]  # fmt: skip


def test_program_count_per_layer_goes_up_by_the_expected_amount():
    rest = _ksplit(9728)  # a rest layer's down: 1 program -> 3
    assert len(K.leaves(rest, "down")) == 3
    first = {
        "kind": "split",
        "split": {"hi": _lin("h"), "lo": _lin("l"), "add": _add(2560)},
    }
    assert len(K.leaves(first, "down")) == 3  # an unsplit first-set down: hi, lo, Add
    # the first set's split layer with both linears split: gu's Add into two (+1) and the
    # down's halves into K-splits (+2 each, +4): 3 + 4 -> 4 + 7
    gu = _gu_split()
    down = {
        "kind": "split",
        "split": {"hi": _ksplit(9728), "lo": _ksplit(9728), "add": _add(2560)},
    }
    assert len(K.leaves(gu, "gu")) == 4
    assert len(K.leaves(down, "down")) == 7


def test_leaf_transforms_are_the_parts_slices_and_the_split_halves():
    class Cfg:
        def __init__(self, W):
            self.W = W

        def merged_weight(self, k, li, gptq=True):
            return self.W

        def split_weights(self, w):
            q = np.round(w * 16) / 16
            return q, (w - q).astype(np.float32)

    W = np.arange(9728 * 3, dtype=np.float32).reshape(9728, 3)
    cfg = Cfg(W)
    e = {
        "kind": "split",
        "split": {"hi": _ksplit(9728), "lo": _ksplit(9728), "add": _add(2560)},
    }
    _, path = K.find(e, "down", "down_lo_p1")
    assert path == [("lo",), ("part", 1, 4864)]
    got = K.leaf_weight(cfg, "down", 0, path)
    _, r = cfg.split_weights(W)
    assert np.array_equal(got, r[4864:])
    assert K.in_ksplit(e, "down", "down_lo_p1") and not K.in_ksplit(
        e, "down", "down_add"
    )


def test_colsplit_leaves_are_the_column_adds():
    names = [n for n, _ in K.leaves(_gu_split(), "gu")]
    assert names == ["gu_hi", "gu_lo", "gu_add_0", "gu_add_1"]


# ---- K-split grids ---------------------------------------------------------------------
def test_ksplit_grids_share_one_output_grid_for_the_parts_and_their_sum():
    rng = np.random.default_rng(0)
    kp, N = 4864, 8
    w = rng.standard_normal((2 * kp, N)).astype(np.float32) * 0.05
    x = rng.standard_normal((5, 2 * kp)).astype(np.float32) * 3
    y = x @ w
    sx, sps, sy = pg.ksplit_grids(x, w, y, kp)
    assert sx == pg.rng2(x)  # both parts read the input on its whole grid
    # one grid (s, 32768) for both parts and the sum: the Add is q = q0 + q1 - 32768
    assert sps[0] == sps[1] == sy
    R = max(
        np.abs(y).max(),
        np.abs(x[:, :kp] @ w[:kp]).max(),
        np.abs(x[:, kp:] @ w[kp:]).max(),
    )  # fmt: skip
    assert sy == (float(R) / 32767.0, 32768)
    for g in (sx, *sps, sy):
        assert len(g) == 2 and isinstance(
            g[1], (int, np.integer)
        )  # (scale, zero point)


# ---- card runner: byte offsets of the split trees --------------------------------------
class _Spec:
    def __init__(self, name, nbytes):
        self.name, self.nbytes = name, nbytes


class _Prog:
    def __init__(self, id_, ins, outs):
        self.id, self.inputs, self.outputs = id_, ins, outs


class _FakeAX:
    """Answers ``BUF`` with fresh ids and ``BIND`` / ``PLAN`` with OK; keeps the log."""

    def __init__(self):
        self.n, self.log = 100, []

    def _cmd(self, c):
        self.log.append(c)
        if c.startswith("BUF "):
            self.n += 1
            return f"OK {self.n}"
        return "OK"


def _plan(progs):
    """A ``pf_plan.Plan`` without a card or a manifest: only what the chain functions use."""
    pl = object.__new__(P.Plan)
    pl.ax, pl.binds, pl.out_bufs, pl.slices, pl.lines = _FakeAX(), {}, {}, {}, []
    pl.M = progs
    pl.g = pl.h = "/x"
    return pl


def _binds(pl, prog_id):
    return pl.binds[prog_id]


def test_ksplit_parts_read_their_slice_at_a_float32_byte_offset():
    kp, N = 4864, 2560
    progs = {}
    ids = iter(range(1, 100))
    for nm in ("down_p0", "down_p1"):
        progs[("rest", "L0", nm)] = _Prog(
            next(ids), [_Spec("x", kp * P.ITEM)], [_Spec("y", N * P.ITEM)]
        )
    progs[("rest", "L0", "down_add")] = _Prog(
        next(ids), [_Spec("a", N * 4), _Spec("b", N * 4)], [_Spec("y", N * 4)]
    )
    pl = _plan(progs)
    e = _ksplit(2 * kp, N)
    e["k_part"] = kp
    xbuf = 77
    out = pl.chain_tree("rest", "L0", "down", e, {"x": (xbuf, 0)})
    b0 = _binds(pl, progs[("rest", "L0", "down_p0")].id)
    b1 = _binds(pl, progs[("rest", "L0", "down_p1")].id)
    assert b0[("I", 0)] == (xbuf, 0)
    assert b1[("I", 0)] == (
        xbuf,
        kp * 4,
    )  # the second slice starts kp elements in, 4 bytes each
    assert pl.slices[("rest", "L0", "down_p1")] == ((xbuf, 0), kp * 4, kp, 2 * kp)
    add = _binds(pl, progs[("rest", "L0", "down_add")].id)
    assert add[("I", 0)][0] != add[("I", 1)][0]  # the Add reads the two part outputs
    assert out == pl.out_bufs[("L0", "down", "y")]


def test_column_split_add_writes_one_output_at_its_column_offsets():
    W = 9728
    progs = {}
    ids = iter(range(1, 100))
    for nm in ("gu_hi", "gu_lo"):
        progs[("first", "L0", nm)] = _Prog(
            next(ids), [_Spec("x", 2560 * 4)], [_Spec("y", 2 * W * 4)]
        )
    for i in range(2):
        progs[("first", "L0", f"gu_add_{i}")] = _Prog(
            next(ids), [_Spec("a", W * 4), _Spec("b", W * 4)], [_Spec("y", W * 4)]
        )
    pl = _plan(progs)
    e = _gu_split()
    out = pl.chain_add("first", "L0", "gu", "gu_add", e["split"]["add"], 1, 2)
    b0 = _binds(pl, progs[("first", "L0", "gu_add_0")].id)
    b1 = _binds(pl, progs[("first", "L0", "gu_add_1")].id)
    assert b0[("O", 0)] == (out, 0)  # gate: the start of the shared output
    assert b1[("O", 0)] == (
        out,
        W * 4,
    )  # up: its column offset (the gate's width, in bytes)
    assert b1[("I", 0)] == (1, W * 4) and b1[("I", 1)] == (
        2,
        W * 4,
    )  # its columns of a and b
    assert pl.out_bufs[("L0", "gu", "colsplit")] == out


def test_plain_add_is_unchanged_by_the_column_split_path():
    progs = {
        ("rest", "L0", "n2_add"): _Prog(
            5, [_Spec("a", 40), _Spec("b", 40)], [_Spec("y", 40)]
        )
    }
    pl = _plan(progs)
    out = pl.chain_add("rest", "L0", "res", "n2_add", _add(10), 3, 4)
    assert _binds(pl, 5)[("I", 0)] == (3, 0) and _binds(pl, 5)[("I", 1)] == (4, 0)
    assert out == pl.out_bufs[("L0", "res", "y")]
