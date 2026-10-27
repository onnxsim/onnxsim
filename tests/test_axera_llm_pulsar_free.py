"""Host-only checks of ``scripts/axera/llm_pulsar_free`` (the Pulsar-free U16
decode for the AX650N): the plan layout, the fake-quant evaluator, and the
whole plan wiring on synthetic models in ``FakeAX`` against the float model.

No card, no Pulsar2, no checkpoint: the models are
``llm_layer_decomposition.make_synthetic``'s random tiny llama / qwen3
layouts, and the programs float stand-ins (``pf_plan.write_float_set``).
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import llm_layer_decomposition as ld  # noqa: E402
import pf_estimate  # noqa: E402
import pf_fake_quant as fqm  # noqa: E402
import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
import pf_plan  # noqa: E402
import pf_templates as tpl  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

F32 = np.float32


def _softmax(a):
    e = np.exp(a - a.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@pytest.mark.parametrize("p", [0, 1, 5, 127])
def test_plan_mask_attends_to_the_same_rows(p):
    """The plan layout (new row in slot p) and layer_graph's (new row at S)
    give the same attention output."""
    rng = np.random.default_rng(p)
    S, d = lay.S, 8
    q = rng.standard_normal((1, d)).astype(F32)
    rows = rng.standard_normal((p + 1, 2, d)).astype(F32)  # (k, v) of positions 0..p
    Kp, Vp = np.zeros((2, S + 1, d), F32)
    Kp[: p + 1], Vp[: p + 1] = rows[:, 0], rows[:, 1]
    Kr, Vr = np.zeros((2, S + 1, d), F32)
    Kr[:p], Vr[:p] = rows[:p, 0], rows[:p, 1]
    Kr[S], Vr[S] = rows[p, 0], rows[p, 1]
    yp = _softmax(q @ Kp.T + lay.mask_plan(p)) @ Vp
    yr = _softmax(q @ Kr.T + lay.mask_ref(p)) @ Vr
    np.testing.assert_allclose(yp, yr, rtol=1e-5, atol=1e-6)


def test_qkv_layout():
    """With a per-head k norm, k_proj and k_rot are one contiguous slice."""
    parts = lay.qkv_parts(16, 128, 1024, qk_norm=True)
    o = lay.offsets(parts)
    assert o["k_rot"] == o["k_proj"] + 1024
    assert sum(n for _, n in parts) == 2 * 2048 + 3 * 1024
    assert (
        lay.layer_programs(True).index("kc") == lay.layer_programs(True).index("rk") - 1
    )
    assert "kc" not in lay.layer_programs(False)


DIMS = tpl.Dims(H=32, I=64, nh=4, nkv=2, hd=16, eps=1e-6, vocab=64)


@pytest.mark.parametrize(
    "name",
    [
        "crms32",
        "rms32",
        "rope64",
        "qkrope64",
        "qkrope32t",
        "hcrms64",
        "attn64",
        "silu64",
        "nrm32",
        "anorm32",
        "qkf64",
        "lanorm64",
        "slanorm64",
        "alanorm64",
        "alanorml64x128",
    ],
)
def test_fused_eval_float_matches_reference(name):
    """With no calibration FusedEval is the float graph (every output of a
    merged program, in graph order); with one it rounds every tensor to its
    U16 grid (close to float, not equal)."""
    _, _, ins, _, _, gens = tpl.graph(name, DIMS)
    m = tpl.float_graph(name, DIMS)
    feeds = gens[0](np.random.default_rng(0))
    wants = ReferenceEvaluator(m).run(None, feeds)
    gots = fqm.FusedEval(m, {})(**feeds)
    gots = gots if isinstance(gots, tuple) else (gots,)
    assert len(gots) == len(wants) == len(m.graph.output)
    for got, want in zip(gots, wants):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    want = wants[-1]
    res = ReferenceEvaluator(m).run(None, feeds, intermediate=True)
    cal = {}
    for k, v in res.items():
        a = np.asarray(v)
        if a.dtype.kind == "f" and k not in {i.name for i in m.graph.initializer}:
            lo, hi = min(float(a.min()), 0.0), max(float(a.max()), 0.0)
            s = (hi - lo) / 65535 or 1.0
            cal[k] = (s, int(np.rint(-lo / s)))
    got = fqm.FusedEval(m, cal)(**feeds)
    got = got[-1] if isinstance(got, tuple) else got
    assert lay.rel_err(got, want) < 1e-2


def test_fq_grid():
    a = np.array([-1.0, 0.0, 0.3, 2.0])
    np.testing.assert_allclose(fqm.fq(a, (0.5, 2)), [-1.0, 0.0, 0.5, 2.0])
    assert fqm.fq(np.array([1e9]), (1.0, 0), signed=True)[0] == 32767


#: llama-160m layer 1's residual Add in ``on2`` (layout 3, rest set): on the card
#: every element of its sum came out 1.55 codes above the rounded float sum
L1_ADD = {
    "a": (0.0013510935241356492, 57309),
    "b": (0.00011911139154108241, 30468),
    "y": (0.0013367648934945464, 57238),
}


def test_card_add_bias_and_exact_offset():
    """The card's U16 Add (fixed-point ratios, Pulsar2's offset word from the
    unrounded ratios) is off by ``add_bias`` codes on every element; with
    ``u16_retarget.exact_add_offset`` the sum is the rounded float sum."""
    import u16_retarget as ur

    c = L1_ADD
    assert fqm.add_bias(c["a"], c["b"], c["y"]) == pytest.approx(1.55, abs=0.01)
    rng = np.random.default_rng(0)
    a = fqm.fq(rng.normal(0, 3, (1, 768)), c["a"])
    b = fqm.fq(rng.normal(0, 1, (1, 768)), c["b"])
    ideal = fqm.fq(a + b, c["y"]) / c["y"][0]
    biased = fqm.card_add(a, b, c["a"], c["b"], c["y"]) / c["y"][0]
    exact = fqm.card_add(a, b, c["a"], c["b"], c["y"], exact_offset=True) / c["y"][0]
    assert np.mean(biased - ideal) == pytest.approx(1.55, abs=0.05)
    assert abs(np.mean(exact - ideal)) < 0.05
    assert np.abs(exact - ideal).max() <= 1
    # the exact offset word puts the zero points on the output's zero point
    sa, za = c["a"]
    sb, zb = c["b"]
    sy, zy = c["y"]
    n = 15 - ur.add_shift(sa / sy, sb / sy)
    off = ur.exact_add_offset(c, "a", "b", "y")
    off = off - (1 << 32) if off >= 1 << 31 else off
    big_a, big_b = round(sa / sy * 2**n), round(sb / sy * 2**n)
    assert za * big_a + zb * big_b + off == zy * 2**n
    # AddEval and FusedEval take the same model
    ad = fqm.AddEval(c, card_add=True)
    np.testing.assert_allclose(ad(a, b), (biased * sy).astype(F32))
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> g (float[1,768] a, float[1,768] b)'
        " => (float[1,768] y) { y = Add(a, b) }"
    )
    got = fqm.FusedEval(m, c, card_add=True, exact_offset=True)(a=a, b=b)
    np.testing.assert_allclose(got, (exact * sy).astype(F32))
    np.testing.assert_allclose(fqm.FusedEval(m, c)(a=a, b=b), (ideal * sy).astype(F32))


def test_split_weights_sum_to_w():
    w = np.random.default_rng(3).standard_normal((48, 32)).astype(F32)
    q, r = pf_model.Config.split_weights(w)
    np.testing.assert_allclose(q + r, w, atol=1e-6)
    assert np.abs(r).max() < np.abs(w).max() / 127
    codes = q / (np.abs(w).max(axis=0) / 127.5)
    np.testing.assert_allclose(codes, np.rint(codes), atol=1e-3)


def test_fake_ax_plan_lines(tmp_path):
    """FakeAX runs W / X / C / R lines over its buffers as the card's runner does."""
    import onnx

    ax = lay.FakeAX(str(tmp_path))
    path = str(tmp_path / "add.onnx")
    onnx.save(tpl.add_model(4), path)
    m = ax.load(path, lambda f: f["a"] + f["b"])
    assert [i.name for i in m.inputs] == ["a", "b"] and m.outputs[0].nbytes == 16
    b = int(ax._cmd("BUF 48").split()[1])
    y = int(ax._cmd("BUF 16").split()[1])
    np.arange(12, dtype=F32).tofile(tmp_path / "t" / "in.bin")
    for line in (f"BIND 0 I 0 {b} 0", f"BIND 0 I 1 {b} 16", f"BIND 0 O 0 {y} 0"):
        assert ax._cmd(line) == "OK"
    plan = tmp_path / "t" / "p.txt"
    plan.write_text(
        f"W {b} 0 48 {tmp_path}/t/in.bin\nX 0\nC {b} 32 {y} 0 16\n"
        f"R {b} 0 48 {tmp_path}/t/out.bin\n"
    )
    assert ax._cmd(f"PLAN {plan}").startswith("OK")
    out = np.fromfile(tmp_path / "t" / "out.bin", F32)
    np.testing.assert_array_equal(out[8:], [4, 6, 8, 10])


def _config(name, out, fuse=0):
    model = ld.make_synthetic(name)
    if (
        model.qk_norm
    ):  # Qwen3-like: position 0 first, shared linears, split MLP at layer 0
        return pf_model.Config(
            model,
            out=out,
            first_pos=0,
            share_linears=True,
            split_first=dict(layers=[0], keys=["gu", "down_proj"]),
            fuse=fuse,
        )
    return pf_model.Config(model, out=out, first_pos=1, fuse=fuse)


WIRING = [("qwen3_like", 0), ("llama_gqa", 0), ("llama_mha", 0)]
WIRING += [("qwen3_like", f) for f in range(1, 12)]
WIRING += [(m, f) for m in ("llama_gqa", "llama_mha") for f in (1, 2, 3, 7)]


@pytest.mark.parametrize("name,fuse", WIRING)
def test_plan_wiring_matches_float_model(name, fuse, tmp_path):
    """The plan (buffers, binds, merged-linear slices, K row rebinding, V row
    copies, the two program sets, sharing and split linears; with ``fuse``
    the merged programs and their several outputs) with float programs
    reproduces the float model step by step; so does the host chain
    pf_estimate.Chain with every program exact."""
    cfg = _config(name, str(tmp_path), fuse)
    root = str(tmp_path / "float_set")
    pf_plan.write_float_set(cfg, root)
    plan = pf_plan.Plan(lay.FakeAX(str(tmp_path / "fake")), cfg, fake=True, root=root)
    ids = [int(t) for t in np.random.default_rng(5).integers(1, cfg.vocab, 6)]
    rows = pf_plan.decode(plan, cfg, ids)
    assert len(rows) == 6
    assert max(r["rel"] for r in rows) < 1e-4
    assert all(r["top1"] == r["float_top1"] for r in rows)
    chains = {
        st: pf_estimate.Chain(cfg, st, exact=True, root=root)
        for st in ("first", "rest")
    }
    est = pf_estimate.run(cfg, ids, chains)
    assert max(r["rel"] for r in est) < 1e-4


@pytest.mark.parametrize("name", ["llama_gqa", "llama_mha"])
def test_nl_set_runs_the_newline_steps(name, tmp_path):
    """With an ``nl`` set (llama-160m's newline steps, ``pf_record.py``) the
    plan loads three sets, runs every ``nl_tokens`` step on ``nl`` (the first
    position still on ``first``) and still reproduces the float model: the
    ``nl`` programs write the same K / V rows the ``rest`` ones read."""
    cfg = _config(name, str(tmp_path), 3)
    root = str(tmp_path / "float_set")
    pf_plan.write_float_set(cfg, root)
    assert cfg.sets(root) == ("first", "rest")
    os.symlink(os.path.join(root, "rest"), os.path.join(root, "nl"))
    with (
        open(os.path.join(root, "manifest_rest.json")) as f,
        open(os.path.join(root, "manifest_nl.json"), "w") as g,
    ):
        g.write(f.read())
    assert cfg.sets(root) == ("first", "rest")  # the model has no nl_tokens
    two = pf_plan.Plan(lay.FakeAX(str(tmp_path / "f2")), cfg, fake=True, root=root)
    ids = [int(t) for t in np.random.default_rng(7).integers(1, cfg.vocab, 7)]
    nl = ids[3]
    ids[1] = ids[5] = nl
    cfg.nl_tokens = (nl,)
    assert cfg.sets(root) == ("first", "rest", "nl")
    plan = pf_plan.Plan(lay.FakeAX(str(tmp_path / "fake")), cfg, fake=True, root=root)
    assert plan.sets == ("first", "rest", "nl")
    assert plan.n_programs > two.n_programs and plan.n_runs == two.n_runs
    used = [cfg.set_of(t, p, plan.sets) for p, t in enumerate(ids) if p]
    assert used == ["first", "rest", "nl", "rest", "nl", "rest"]
    rows = pf_plan.decode(plan, cfg, ids)
    assert max(r["rel"] for r in rows) < 1e-4
    assert all(r["top1"] == r["float_top1"] for r in rows)
    chains = {st: pf_estimate.Chain(cfg, st, exact=True, root=root) for st in plan.sets}
    assert max(r["rel"] for r in pf_estimate.run(cfg, ids, chains)) < 1e-4


def test_lines_keeps_the_line_breaks(tmp_path):
    """``pf_layout.lines`` (the ``nl`` set's calibration text) keeps line
    breaks, headings, lists and code; ``prose`` joins them into one line."""
    md = tmp_path / "d.md"
    md.write_text(
        "# Title\n\n<p>html</p>\nSome [link](x.md) text.\n\n- item\n\n"
        "```py\nx = 1\n```\n"
    )
    assert lay.lines(str(md)) == (
        "# Title\n\nSome link text.\n\n- item\n\n```py\nx = 1\n```"
    )
    assert lay.prose(str(md)) == "Some link text. - item"


@pytest.mark.parametrize("name,fuse", [("qwen3_like", 5), ("llama_mha", 3)])
def test_free_running_greedy_matches_float_model(name, fuse, tmp_path):
    """``pf_plan.greedy`` (``--generate``) on the plan with float programs feeds
    its own tokens back and continues the prompt exactly as the float model's
    greedy decode does; ``compare_generation`` counts the agreement."""
    cfg = _config(name, str(tmp_path), fuse)
    root = str(tmp_path / "float_set")
    pf_plan.write_float_set(cfg, root)
    plan = pf_plan.Plan(lay.FakeAX(str(tmp_path / "fake")), cfg, fake=True, root=root)
    prompt = [int(t) for t in np.random.default_rng(6).integers(1, cfg.vocab, 3)]
    got, _ = pf_plan.greedy(lambda t, p: plan.step(t, p)[0], cfg, prompt, 5)
    ref, gaps = pf_plan.float_greedy(cfg, prompt, 5)
    assert len(got) == 5 and got == ref
    row = pf_plan.compare_generation(cfg, prompt, got, ref, gaps)
    assert row["agree"] == 5 and "diverge" not in row
    row = pf_plan.compare_generation(cfg, prompt, got[:2] + [-1] * 3, ref, gaps)
    assert row["agree"] == 2 and row["diverge"]["ref"] == ref[2]
    with pytest.raises(ValueError):
        pf_plan.greedy(None, cfg, prompt, cfg.S)


def test_template_names():
    cfg = _config("qwen3_like", ".")
    names = tpl.needed(cfg)
    assert {
        "crms32",
        "rms32",
        "qkrope64",
        "qkrope32",
        "qkrope64t",
        "hcrms64",
        "attn64",
    } <= set(names)
    assert "add128" in names and "k64n32" in names  # split gu Add; down_proj
    assert tpl.template_dir("r", "k32n64").endswith(os.path.join("linear", "k32n64"))
    assert tpl.template_dir("r", "add32").endswith(os.path.join("single", "add32"))


@pytest.mark.parametrize(
    "fuse,runs",
    [(0, 15), (1, 9), (2, 7), (3, 6), (4, 4), (5, 3), (6, 3), (7, 2), (8, 1)]
    + [(9, 1), (10, 1), (11, 1)],
)
def test_programs_per_layer(fuse, runs):
    """Executes per layer of the rest set; layer 0 also runs the first norm,
    and a split first-set ``down_proj`` keeps its own programs."""
    cfg = _config("qwen3_like", ".", fuse)
    # fuse=8: the synthetic model's L1 is its last layer (aon2g + dn)
    progs = cfg.programs("rest", "L0" if fuse == 8 else "L1")
    assert len([k for k in progs if k != "n1qk"]) == runs
    if fuse == 11:  # lm_head joins the rest set's last program
        assert cfg.programs("rest", "L1") == ["alyrfh"]
        assert cfg.programs("rest", "post") == []
        assert cfg.programs("first", "post") == ["lm_head"]
        assert cfg.programs("first", "L0") == ["n1qk", "alyrs"]
    elif fuse == 10:  # the split first-set layer 0 is one program too
        assert cfg.programs("first", "L0") == ["n1qk", "alyrs"]
        assert cfg.programs("first", "L1") == ["aon2g", "dn"]
        assert cfg.kv_producer("first", 1) == ("L0", "alyrs")
        assert cfg.programs("rest", "L1") == ["alyrf"]
    elif fuse == 9:  # the last layer is one program too
        assert cfg.programs("rest", "L0") == ["n1qk", "alyr"]
        assert cfg.programs("rest", "L1") == ["alyrf"]
        assert cfg.programs("first", "L1") == ["qkv5", "qk", "aon2g", "dn"]
        assert cfg.programs("rest", "post") == ["lm_head"]
    elif fuse == 8:
        assert cfg.programs("rest", "L0") == ["n1qk", "alyr"]
        assert cfg.programs("rest", f"L{cfg.L - 1}") == ["aon2g", "dn"]
        assert cfg.programs("first", "L0")[:3] == ["n1qk", "attn", "on2"]  # split
        assert cfg.programs("first", "L1")[:2] == ["qkv5", "qk"]
        assert cfg.kv_producer("rest", 1) == ("L0", "alyr")
        assert cfg.kv_producer("first", 1) == ("L1", "qk")
    elif fuse == 7:
        assert cfg.programs("rest", "L0") == ["n1qk", "aon2g", "dnqk"]
        assert cfg.programs("first", "L0")[:3] == ["n1qk", "attn", "on2"]  # split
        assert cfg.kv_producer("rest", 1) == ("L0", "dnqk")
        llama = _config("llama_gqa", ".", fuse)
        assert llama.programs("rest", "L1") == ["qkv5", "rq", "rk", "aon2", "gu", "dn"]
        assert llama.kv_producer("rest", 1) == ("L1", "rk")
    elif fuse >= 5:
        assert cfg.programs("rest", "L0")[0] == "n1qk"
        assert cfg.programs("first", "L1")[:2] == ["qkv5", "qk"]
        assert cfg.kv_producer("rest", 1) == ("L0", "dnqk")
        assert cfg.kv_producer("first", 1) == ("L1", "qk")
    elif fuse == 4:
        assert cfg.programs("rest", "L0")[0] == "n1q"
        assert cfg.programs("first", "L0")[-5:] == [
            "on2",
            "gu",
            "silu_f",
            "down_proj",
            "dn",
        ]  # split
        assert cfg.programs("first", "L1")[0] == "qkv5"
        assert cfg.programs("rest", f"L{cfg.L - 1}")[-1] == "dn"
    elif fuse >= 2:
        assert cfg.programs("rest", "L0")[0] == "n1"
        assert "n1" not in cfg.programs("rest", "L1")
        first = cfg.programs("first", "L0")
        assert first.index("down_proj") == first.index("dn") - 1
        assert "silu_f" in first
        assert ("silu_f" in cfg.programs("rest", "L0")) == (fuse == 2)
        assert cfg.programs("rest", "post") == ["lm_head"]


def test_merged_template_names():
    for fuse, want in ((1, {"nrm32", "anorm32", "qkf64"}), (3, {"slanorm64"})):
        names = set(tpl.needed(_config("qwen3_like", ".", fuse)))
        assert want <= names and "crms32" not in names
    assert {"lanorm64"} <= set(tpl.needed(_config("qwen3_like", ".", 2)))
    five, six = (set(tpl.needed(_config("qwen3_like", ".", f))) for f in (5, 6))
    assert {"nrmqk32", "slanormqk64"} <= five and not {"nrmqkr32"} & five
    assert {"nrmqkr32", "slanormqkr64"} <= six and not {"nrmqk32"} & six
    assert "alanorml64x128" in tpl.needed(_config("qwen3_like", ".", 7))
    assert "alanorm32" in tpl.needed(_config("llama_gqa", ".", 7))  # nh * hd = 32
    eight = set(tpl.needed(_config("qwen3_like", ".", 8)))
    seven = set(tpl.needed(_config("qwen3_like", ".", 7)))
    assert "alyrqkr64" in eight and "alyrqkr64" not in seven
    nine = set(tpl.needed(_config("qwen3_like", ".", 9)))
    assert nine - eight == {"alyrf64"}
    ten = set(tpl.needed(_config("qwen3_like", ".", 10)))
    assert ten - nine == {"alyrqkrs64"}
    eleven = set(tpl.needed(_config("qwen3_like", ".", 11)))
    assert eleven - ten == {f"alyrf64x{_config('qwen3_like', '.').vocab}"}


@pytest.mark.parametrize("name", ["nrmqkr32", "slanormqkr64"])
def test_in_program_rotation_equals_rotated_weights(name):
    """``fuse=6``'s q / k rotation by Split / Concat in the program computes
    what ``fuse=5``'s two linears of rotated weights (q_rot, k_rot) do."""
    _, body, ins, outs, inits, gens = tpl.graph(name, DIMS)
    gen, ws = gens[1]
    m6 = tpl.model_of(body, ins, outs, {**inits, **ws})
    _, body5, ins5, outs5, inits5, _ = tpl.graph(name.replace("qkr", "qk"), DIMS)
    perm = np.kron(np.eye(DIMS.nh, dtype=F32), rr.perm_matrix(DIMS.hd))
    pk = np.kron(np.eye(DIMS.nkv, dtype=F32), rr.perm_matrix(DIMS.hd))
    w5 = {**ws, "wqr": ws["wq"] @ perm, "wkr": ws["wk"] @ pk}
    m5 = tpl.model_of(body5, ins5, outs5, {**inits5, **w5})
    feeds = gen(np.random.default_rng(0))
    got = fqm.FusedEval(m6, {})(**feeds)
    for a, b, c in zip(
        got, ReferenceEvaluator(m6).run(None, feeds), fqm.FusedEval(m5, {})(**feeds)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["lyrqk64", "lyrqkr64", "alyrqkr64"])
def test_lyr_graph_is_its_parts(name):
    """``lyrqk`` computes ``lanorm`` (o_proj + residual Add + norm), the gate /
    up linears, then ``slanormqk`` (SiLU + down_proj + residual Add + norm +
    q / k / v + RoPE) on them; ``lyrqkr`` with ``slanormqkr``; ``alyrqkr``
    (``fuse=8``, the whole layer) with ``alanorm`` (the attention first).
    Every output equals the chain of the parts' float graphs, same weights."""
    ops, body, ins, outs, inits, gens = tpl.graph(name, DIMS)
    assert "Silu" in ops and list(outs) == ["x", "yq", "yk", "v"]
    assert "Mul(sg, gate)" in body  # Pulsar2 keeps the Sigmoid table and the Mul
    gen, ws = gens[1]
    w = {**inits, **ws}
    m = tpl.model_of(body, ins, outs, w)
    feeds = gen(np.random.default_rng(0))
    got = ReferenceEvaluator(m).run(None, feeds)

    def run(name, feeds, **weights):
        g = tpl.float_graph(name, DIMS)
        for i in g.graph.initializer:
            if i.name in weights:
                i.CopyFrom(onnx.numpy_helper.from_array(weights[i.name], i.name))
        names = [o.name for o in g.graph.output]
        return dict(zip(names, ReferenceEvaluator(g).run(None, feeds)))

    first = "alanorm64" if name.startswith("a") else "lanorm64"
    fins = tpl.graph(first, DIMS)[2]
    a = run(first, {t: feeds[tpl.lyr_name(t)] for t in fins}, w=w["w_1"])
    gate, up = a["y"] @ w["wg"], a["y"] @ w["wu"]
    rope = {t: feeds[t] for t in ("cq", "sq", "ck", "sk")}
    b = run(
        name.lstrip("a").replace("lyrqk", "slanormqk"),
        dict(gate=gate, up=up, a=a["x"], g=feeds["g"], **rope),
        **{k: w[k] for k in ("w", "wq", "wqr", "wk", "wkr", "wv") if k in w},
    )
    for name, v in zip(outs, got):
        np.testing.assert_allclose(v, b[name], rtol=1e-5, atol=1e-5)


def test_signed_inputs_skip_a_constant_weight():
    """A MatMul by a constant is a FullyConnected: its input stays unsigned."""
    import pf_generate

    assert pf_generate.signed_inputs(tpl.float_graph("lanorm64", DIMS)) == set()
    assert pf_generate.signed_inputs(tpl.float_graph("attn64", DIMS)) >= {"qh", "kt"}


def test_tied_ranges_follow_the_template():
    """Tensors at one template calibration share one range in the target;
    a tie that would widen a range past TIE_MAX refuses the template."""
    import pf_generate

    g = pf_generate.Generator.__new__(pf_generate.Generator)
    m = tpl.float_graph("nrm32", DIMS)
    rg = {"u": [-2.0, 3.0], "d": [-1.0, 1.5], "x": [-50.0, 60.0]}
    tq = {"u": (1e-4, 30000), "d": (1e-4, 30000), "x": (2e-3, 31000)}
    out = g._tied(rg, m, tq)
    assert out["u"] == out["d"] == [-2.0, 3.0]
    tq["x"] = (1e-4, 30000)
    with pytest.raises(ValueError, match="widen"):
        g._tied(rg, m, tq)


#: ``pf_generate._run_one``'s layer functions, by layout
LAYER_FNS = ("", "_fused", "_fused2", "_fused4", "_fused5", "_fused8")


def test_generate_patches_add_offsets(monkeypatch):
    """With ``add_offset="exact"`` (Qwen3-0.6B's default) ``pf_generate``
    patches each layer's programs as it writes them
    (``pf_add_offset.patch_entry`` in place); ``pulsar2`` (llama-160m's, and
    a synthetic ``Config``'s) keeps Pulsar2's word."""
    import types

    import pf_add_offset
    import pf_generate

    assert pf_model.MODELS["qwen3_0.6b"]["add_offset"] == "exact"
    assert pf_model.MODELS["qwen3_0.6b"]["fuse"] == 8
    assert pf_model.MODELS["llama160m"]["add_offset"] == "pulsar2"
    assert _config("qwen3_like", ".", 8).add_offset == "pulsar2"
    with pytest.raises(ValueError, match="add_offset"):
        pf_model.Config(ld.make_synthetic("qwen3_like"), add_offset="zero")

    calls = []

    def patch_entry(src, out, st, wh, k, e):
        calls.append((src, out, st, wh, k))
        e["add_offset"] = "exact"
        return 1

    def layer(st, li, man):
        man[f"L{li}"] = {"alyr": {"kind": "fused"}, "lm": {"kind": "linear"}}

    cfg = types.SimpleNamespace(fuse=8, qk_norm=True, out="/set")
    gen = types.SimpleNamespace(cfg=cfg, **{f"layer{s}": layer for s in LAYER_FNS})
    monkeypatch.setattr(pf_add_offset, "patch_entry", patch_entry)
    monkeypatch.setattr(pf_generate, "_GEN", gen)
    monkeypatch.setattr(pf_generate, "_ADD_OFFSET", "pulsar2")
    assert pf_generate._run_one(("rest", 3)) == {
        "alyr": {"kind": "fused"},
        "lm": {"kind": "linear"},
    }
    assert calls == []
    monkeypatch.setattr(pf_generate, "_ADD_OFFSET", "exact")
    entry = pf_generate._run_one(("rest", 3))
    assert entry["alyr"]["add_offset"] == "exact"
    assert calls == [
        ("/set", "/set", "rest", "L3", "alyr"),
        ("/set", "/set", "rest", "L3", "lm"),
    ]


def test_alyrf_graph_is_its_parts():
    """``alyrf`` (``fuse=9``, the last layer) computes ``alanorm``, the gate /
    up linears, then ``slanorm`` (SiLU + down_proj + residual Add + the final
    norm); ``alyrf<n>x<v>`` (``fuse=11``) also ``lm_head`` on the norm."""
    ops, body, ins, outs, inits, gens = tpl.graph("alyrf64", DIMS)
    assert list(outs) == ["x", "y"] and "cq" not in ins and len(gens) == 4
    gen, ws = gens[1]
    w = {**inits, **ws}
    feeds = gen(np.random.default_rng(0))
    got = ReferenceEvaluator(tpl.model_of(body, ins, outs, w)).run(None, feeds)

    def run(name, feeds, **weights):
        g = tpl.float_graph(name, DIMS)
        for i in g.graph.initializer:
            if i.name in weights:
                i.CopyFrom(onnx.numpy_helper.from_array(weights[i.name], i.name))
        names = [o.name for o in g.graph.output]
        return dict(zip(names, ReferenceEvaluator(g).run(None, feeds)))

    fins = tpl.graph("alanorm64", DIMS)[2]
    a = run("alanorm64", {t: feeds[tpl.lyr_name(t)] for t in fins}, w=w["w_1"])
    gate, up = a["y"] @ w["wg"], a["y"] @ w["wu"]
    b = run("slanorm64", dict(gate=gate, up=up, a=a["x"], g=feeds["g"]), w=w["w"])
    for name, v in zip(outs, got):
        np.testing.assert_allclose(v, b[name], rtol=1e-5, atol=1e-5)
    _, bh, insh, outsh, initsh, _ = tpl.graph("alyrf64x48", DIMS)
    assert list(outsh) == ["x", "y"] and outsh["y"] == [1, 48]
    wh = {**initsh, **{k: v for k, v in w.items() if k in initsh and k != "w2"}}
    yh = ReferenceEvaluator(tpl.model_of(bh, insh, outsh, wh)).run(None, feeds)[1]
    np.testing.assert_allclose(yh, got[1] @ wh["w2"], rtol=1e-4, atol=1e-5)


def test_alyrqkrs_split_linears():
    """``alyrqkrs`` (``fuse=10``) is ``alyrqkr`` with the gate, up and down
    linears each two linears and an Add: with the second parts zero it equals
    ``alyrqkr``; its variants keep ``alanorm``'s v0 half."""
    ops, body, ins, outs, inits, gens = tpl.graph("alyrqkrs64", DIMS)
    assert len(gens) == 6
    _, body8, ins8, outs8, inits8, gens8 = tpl.graph("alyrqkr64", DIMS)
    assert ins == ins8 and outs == outs8
    gen, ws = gens[1]
    z = {k: np.zeros_like(inits[k]) for k in ("wgl", "wul", "wl")}
    got = ReferenceEvaluator(tpl.model_of(body, ins, outs, {**inits, **ws, **z}))
    want = ReferenceEvaluator(tpl.model_of(body8, ins8, outs8, {**inits8, **ws}))
    feeds = gen(np.random.default_rng(1))
    for a, b in zip(got.run(None, feeds), want.run(None, feeds)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_split_override():
    spec = dict(split_first=dict(layers=[0, 1, 2], keys=["gu", "down_proj"]))
    assert pf_model.split_override(spec, {}) == spec
    got = pf_model.split_override(spec, {"LLM_PF_SPLIT_FIRST": "-"})
    assert got["split_first"] == dict(layers=[], keys=["gu", "down_proj"])
    got = pf_model.split_override(
        spec, {"LLM_PF_SPLIT_FIRST": "2", "LLM_PF_SPLIT_KEYS": "down_proj"}
    )
    assert got["split_first"] == dict(layers=[2], keys=["down_proj"])
    assert spec["split_first"]["layers"] == [0, 1, 2]  # not modified
    assert pf_model.split_override({}, {"LLM_PF_SPLIT_FIRST": "-"}) == {}
