"""Host-only checks for the second round of Pulsar-free build-time changes
(``docs/axera-llm-pulsar-free-decode.md``, "Build time"): parallel template
builds (``pf_templates.build_parallel``), the torch float step of
``pf_record.py --device`` (``pf_float_torch``), and ``pf_generate.py both``'s
schedule. No Docker, card, checkpoint or template build is needed."""

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import parser
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
for _p in (AXERA, os.path.join(AXERA, "llm_pulsar_free")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pf_generate as pg  # noqa: E402
import pf_templates as tpl  # noqa: E402
import pulsar2_docker as P  # noqa: E402


def test_build_names_are_unique_and_memory_capped(tmp_path, monkeypatch):
    """Two builds started in the same millisecond by one process get distinct
    container names; ``memory`` caps the container without swap."""
    cmds = []
    monkeypatch.setattr(P, "docker_image_available", lambda image=None: True)
    monkeypatch.setattr(P.time, "time", lambda: 1000.0)

    def run(cmd, **kw):
        cmds.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="no build")

    monkeypatch.setattr(P.subprocess, "run", run)
    for mem in ("3g", None):
        P.build(str(tmp_path), "m.onnx", "out", config_path="c.json", memory=mem)
    names = [c[c.index("--name") + 1] for c in cmds]
    assert len(set(names)) == 2
    i = cmds[0].index("--memory")
    assert cmds[0][i : i + 4] == ["--memory", "3g", "--memory-swap", "3g"]
    assert "--memory" not in cmds[1]


def _fake_units(root, specs, log, fail_once=()):
    """``tpl.Unit``s that record their start, run time and memory cap and
    write the ``result.json`` a build would."""
    lock = threading.Lock()
    state = {"n": 0, "gb": 0}
    out = []
    for name, i, gb, sec in specs:

        def run(memory=None, name=name, i=i, gb=gb):
            W = os.path.join(tpl.template_dir(root, name), f"v{i}")
            os.makedirs(W, exist_ok=True)
            cap = int(memory[:-1])
            with lock:
                first = not any(e[:2] == (name, i) for e in log)
                state["n"] += 1
                state["gb"] += cap
                log.append((name, i, memory, state["n"], state["gb"]))
            time.sleep(0.05)
            with lock:
                state["n"] -= 1
                state["gb"] -= cap
            ok = not (first and (name, i) in fail_once)
            json.dump({"ok": ok}, open(os.path.join(W, "result.json"), "w"))

        out.append(tpl.Unit(name, i, run, gb, sec))
    return out


def test_build_parallel_bounds_jobs_and_memory(tmp_path, monkeypatch):
    root = str(tmp_path)
    specs = [("nrm8", i, 3, 30) for i in range(6)] + [("attn8", 0, 9, 300)]
    specs += [("silu8", i, 2, 20) for i in range(4)]
    log = []
    fake = _fake_units(root, specs, log, fail_once={("silu8", 2)})
    monkeypatch.setattr(tpl, "units", lambda r, names, d: fake)
    bad = tpl.build_parallel(root, ["x"], None, jobs=4, mem_gb=12, stagger=0.02)
    assert bad == []
    assert log[0][:2] == ("attn8", 0)  # the longest first
    assert max(e[3] for e in log) <= 4
    assert max(e[4] for e in log) <= 12
    # the failed build ran again, alone at the whole budget
    assert log[-1][:3] == ("silu8", 2, "12g")
    assert len(log) == len(specs) + 1


def test_torch_graph_equals_the_reference_evaluator():
    """``pf_float_torch.TorchGraph`` evaluates the ops of the decode graphs as
    ``ReferenceEvaluator`` does (float32, to rounding)."""
    torch = pytest.importorskip("torch")
    import pf_float_torch as pft

    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[1, 8] x, float[4, 8] K, float[8, 8] w, float[1, 8] gn,
           float[1] eps, int64[2] shp, int64[2] shk)
            => (float y, float a, float sm)
        {
            sq = Mul(x, x)
            m = ReduceMean<axes=[-1], keepdims=1>(sq)
            e = Add(m, eps)
            r = Sqrt(e)
            d = Div(x, r)
            a = Mul(d, gn)
            q = MatMul(a, w)
            Kc = Concat<axis=0>(K, q)
            q2 = Reshape(q, shp)
            K2 = Reshape(Kc, shk)
            Kt = Transpose<perm=[1, 0]>(K2)
            sc = MatMul(q2, Kt)
            sm = Softmax<axis=-1>(sc)
            s = Sigmoid(sm)
            y = Mul(sm, s)
        }
        """
    )
    rng = np.random.default_rng(0)
    feeds = {
        "x": rng.standard_normal((1, 8)).astype(np.float32),
        "K": rng.standard_normal((4, 8)).astype(np.float32),
        "w": rng.standard_normal((8, 8)).astype(np.float32),
        "gn": rng.standard_normal((1, 8)).astype(np.float32),
        "eps": np.array([1e-6], np.float32),
        "shp": np.array([2, 4], np.int64),
        "shk": np.array([10, 4], np.int64),
    }
    want = ReferenceEvaluator(model).run(None, feeds)
    env = pft.TorchGraph(model, "cpu").run(feeds)
    host = pft._Host(env)
    for name, w in zip(("y", "a", "sm"), want):
        np.testing.assert_allclose(host[name], w, rtol=1e-5, atol=1e-6)
    assert isinstance(env["y"], torch.Tensor)


def test_ordered_manifest_is_the_serial_order():
    base = {"post": {"old": 1}, "L1": {"old": 1}, "Lx": {}}
    done = {"post": {"p": 1}, "L2": {"b": 1}, "L0": {"a": 1}}
    m = pg._ordered(base, done, ["L0", "L1", "L2", "post"])
    assert list(m) == ["post", "L1", "Lx", "L0", "L2"]
    assert m["post"] == {"p": 1} and m["L1"] == {"old": 1}


def _fake_run_one(job):
    """``pf_generate._run_one`` stand-in: a first-set job checks that its
    rest-set entry is already in ``manifest_rest.json``."""
    st, li = job
    k = "post" if li is None else f"L{li}"
    out = pg._GEN.cfg.out
    if st == "first":
        rest = json.load(open(os.path.join(out, "manifest_rest.json")))
        assert k in rest, f"first {k} ran before its rest entry was written"
    time.sleep(0.02 * (3 if li is None else 1))
    return {"prog": {"set": st, "where": k}}


def test_run_both_writes_the_serial_manifests(tmp_path, monkeypatch):
    cfg = SimpleNamespace(out=str(tmp_path), L=5, split_first={"layers": [0, 1]})
    monkeypatch.setattr(pg, "_GEN", SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(pg, "_run_one", _fake_run_one)
    pg.run_both(cfg, list(range(5)), True, 3, time.time())
    for st in ("rest", "first"):
        m = json.load(open(tmp_path / f"manifest_{st}.json"))
        assert list(m) == [f"L{i}" for i in range(5)] + ["post"]
        assert all(v["prog"]["set"] == st for v in m.values())
