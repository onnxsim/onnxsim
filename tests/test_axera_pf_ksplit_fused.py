"""Host-only checks of the K-split down linear inside the fused programs of the Pulsar-free
decode (Qwen3-4B's K = 9728 is absent at N = 2560): the program graph (two K / 2 MatMuls, a
``Split`` of the input, an Add), the key names (``pf_templates.ksplit_name`` / ``needed``),
the shared grid of the parts and their Add (``pf_generate.ksplit_ranges``), the missing-key
check (an unsplit fused program with an absent width is flagged), and the program count per
layer (one program: the split is inside it). No build, no card, no checkpoint: the graphs
are built from the template text and the manifest.
"""

import os
import re
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import pf_layout as lay  # noqa: E402
import pf_registry as R  # noqa: E402
import pf_templates as tpl  # noqa: E402
import u16_retarget as ur  # noqa: E402

MAN = R.load_manifest()
Q4B = R.model_spec("qwen3_4b", MAN)
Q06 = R.model_spec("qwen3_0.6b", MAN)
LLAMA = R.model_spec("llama160m", MAN)
D4B = Q4B.dims()


def _matmuls(model):
    """(K, N) of each MatMul with a constant weight, and its input / output names."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type == "MatMul" and n.input[1] in inits:  # a constant weight
            w = inits[n.input[1]]
            out.append((int(w.shape[0]), int(w.shape[1]), n.input[0], n.output[0]))
    return out


# ---- names --------------------------------------------------------------------------
def test_shipped_names_are_never_k_split():
    kinds = re.compile(r"(alyrqkrk|alyrfk|lanormk|slanormk|slanormqkk|slanormqkrk)\d+")
    for spec in (Q06, LLAMA):
        assert not spec.ksplit_down
        for layout in R.LAYOUTS:
            for name in R.keys_for(spec, layout):
                assert kinds.fullmatch(name) is None, (spec.name, layout, name)


def test_ksplit_name_only_where_the_down_linear_is_k_equals_i():
    inter = 9728
    assert (
        tpl.ksplit_name("alyrqkr4096", inter) == "alyrqkrk4096"
    )  # the layer's own down
    assert tpl.ksplit_name("alyrf4096", inter) == "alyrfk4096"
    assert tpl.ksplit_name("lanorm9728", inter) == "lanormk9728"
    assert tpl.ksplit_name("slanorm9728", inter) == "slanormk9728"
    assert tpl.ksplit_name("slanormqkr9728", inter) == "slanormqkrk9728"
    assert tpl.ksplit_name("slanormqk9728", inter) == "slanormqkk9728"
    # the o_proj's norm (K = nh * hd = 4096) and the q / k / v norms are not the down linear
    assert tpl.ksplit_name("lanorm4096", inter) == "lanorm4096"
    assert tpl.ksplit_name("nrmqkr2560", inter) == "nrmqkr2560"
    assert tpl.ksplit_name("alanorml4096x19456", inter) == "alanorml4096x19456"


def test_qwen3_4b_layout_8_fused_keys_are_k_split():
    names = R.keys_for(Q4B, 8)
    for k in ("alyrqkrk4096", "lanormk9728", "slanormk9728", "slanormqkrk9728"):
        assert k in names, k
    for k in ("alyrqkr4096", "lanorm9728", "slanorm9728", "slanormqkr9728"):
        assert k not in names, k
    # the split layers' linears are the registry's K-split plan (unchanged)
    split, plans = R.split_keys(names, MAN)
    assert plans["k9728n2560"]["names"] == ["k4864n2560"] * 2
    assert "k9728n2560" not in split


def test_qwen3_4b_k_split_is_wired_for_layouts_2_3_5_to_9_only():
    for layout in (2, 3, 5, 6, 7, 8, 9):
        assert tpl.needed(R._Cfg(Q4B, layout)), layout
    for layout in (4, 10, 11):
        with pytest.raises(NotImplementedError):
            R.keys_for(Q4B, layout)


# ---- the program graph ---------------------------------------------------------------
def test_alyr_k_split_graph_has_two_halves_and_one_add():
    m = tpl.float_graph("alyrqkrk4096", D4B)
    onnx.checker.check_model(m)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    assert inits["w"].shape == (4864, 2560) and inits["wl"].shape == (4864, 2560)
    ks = [k for k, _, _, _ in _matmuls(m)]
    assert ks.count(4864) >= 2 and 9728 not in ks  # the down linear is gone
    splits = [n for n in m.graph.node if n.op_type == "Split"]
    down = [n for n in splits if list(n.input) == ["z"]]
    assert len(down) == 1 and list(down[0].output) == ["zk0", "zk1"]
    adds = [
        n for n in m.graph.node if n.op_type == "Add" and list(n.input) == ["bh", "bl"]
    ]
    assert len(adds) == 1 and adds[0].output[0] == "b"
    assert len(splits) >= 2  # the q / k rotation's Split stays


def test_k_split_graph_is_the_unsplit_one_with_the_halves():
    base = tpl.float_graph("alyrqkr4096", D4B)
    ks = tpl.float_graph("alyrqkrk4096", D4B)
    b = {n.output[0]: n for n in base.graph.node}
    assert "b" in b and b["b"].op_type == "MatMul"
    assert {n.op_type for n in base.graph.node} <= {
        n.op_type for n in ks.graph.node
    } | {"MatMul"}
    # the rest of the layer is the same graph: the same op count apart from the halves
    assert len(ks.graph.node) == len(base.graph.node) - 1 + 4


def test_slanormk_and_lanormk_split_only_the_down_linear():
    for name, base in (("lanormk9728", "lanorm9728"), ("slanormk9728", "slanorm9728")):
        m = tpl.float_graph(name, D4B)
        onnx.checker.check_model(m)
        ks = sorted(k for k, _, _, _ in _matmuls(m))
        assert ks == [4864, 4864], (name, ks)
        assert tpl.float_graph(base, D4B) is not None


def test_the_k_split_keeps_the_generators_of_the_unsplit_program():
    _, _, _, _, _, gens_split = tpl.graph("alyrqkrk4096", D4B)
    _, _, _, _, _, gens_lyr = tpl.graph("alyrqkr4096", D4B)
    # the six variants of the unsplit program's lanorm v0 come first (templates built from
    # them stay valid); the ``KSPLIT_ALYR_EXTRA`` ones after them (v6 on), and lanorm's v2 /
    # v3 halves are not used
    assert len(gens_split) == 6 + len(tpl.KSPLIT_ALYR_EXTRA)
    assert len(gens_lyr) == 8


# ---- the parts and their Add share one grid ------------------------------------------
def test_ksplit_ranges_put_the_parts_and_their_sum_on_one_grid():
    import pf_generate as pg

    rng = np.random.default_rng(0)
    z = rng.uniform(-3, 3, (5, 9728)).astype(np.float32)
    w = rng.uniform(-0.1, 0.1, (9728, 2560)).astype(np.float32)
    b = (z @ w).astype(np.float32)
    zr = [-1.0, 1.0]
    rg = pg.ksplit_ranges(z, w, b, zr)
    assert rg["bh"] == rg["bl"] == rg["b"]
    lo, hi = rg["b"]
    s, zp = ur.unsigned_minmax(lo, hi)
    R_ = float(max(np.abs(b).max(), np.abs(z[:, :4864] @ w[:4864]).max(),
                   np.abs(z[:, 4864:] @ w[4864:]).max()))  # fmt: skip
    assert s == pytest.approx(R_ / 32767.0, rel=1e-6)
    assert zp == 32768
    assert rg["z"] == zr  # the input keeps its own range


def test_wdown_halves_the_down_weights_by_rows():
    import pf_generate as pg

    w = np.arange(9728 * 4, dtype=np.float32).reshape(9728, 4)
    d = pg.wdown(w)
    assert d["w"].shape == (4864, 4) and d["wl"].shape == (4864, 4)
    assert np.array_equal(np.vstack([d["w"], d["wl"]]), w)


# ---- the missing-key check and the program count -------------------------------------
def _absent_k(name, dims, man):
    """The (K, N) of the linears of a fused program whose width is absent at its N."""
    out = []
    for k, n, _, _ in _matmuls(tpl.float_graph(name, dims)):
        if R.linear_split(k, n, man) is not None:
            out.append((k, n))
    return out


def test_missing_key_check_flags_the_unsplit_fused_program():
    assert _absent_k("lanorm9728", D4B, MAN) == [(9728, 2560)]
    assert _absent_k("slanormqkr9728", D4B, MAN) == [(9728, 2560)]


def test_k_split_fused_programs_have_no_absent_width():
    for name in ("alyrqkrk4096", "lanormk9728", "slanormk9728", "slanormqkrk9728"):
        assert _absent_k(name, D4B, MAN) == [], name


def test_one_program_per_rest_layer_at_layout_8():
    # the split is inside the program: layout 8 keeps one program per layer
    assert lay.LAYER_FUSED8 == ["alyr"]
    assert len(lay.LAYER_FUSED8) == 1
