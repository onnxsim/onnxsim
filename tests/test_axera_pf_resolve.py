"""Host-only checks of how ``pf_generate.py`` resolves its template keys through
the registry (``pf_registry.plan``): the order exact > by name > reuse, the
refusal of a key with no record (exit 2, the missing keys and the
``pf_registry.py build`` line on stderr), the spec lookup by name or config
path, and the spec's dims against the checkpoint's. No build, no card, no
checkpoint: the fake configs are ``pf_registry._Cfg`` objects, and the
directories are written by hand (signature files or unsigned ones)."""

import json
import os
import sys
from dataclasses import replace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import pf_generate as pg  # noqa: E402
import pf_registry as R  # noqa: E402
import pf_templates as tpl  # noqa: E402

QWEN = R.SPECS["qwen3_0.6b"]
LLAMA = R.SPECS["llama160m"]
#: Qwen3-0.6B's dims with another eps: every norm changes, attention does not
EPS = replace(QWEN, name="qwen_eps", eps=1e-05)


def _dir(root, group, name, sig_spec=None):
    """A template directory: a signature file of ``sig_spec``'s signature, or
    (``sig_spec`` None) an unsigned one."""
    d = os.path.join(root, group, name)
    os.makedirs(d, exist_ok=True)
    if sig_spec is not None:
        sig = R.signature(name, sig_spec.dims())
        R._write_sig(d, name, R.sig_id(sig), sig, sig_spec)
    return d


def _plan(spec, roots, reuse=None):
    return {c.name: c for c in R.plan(spec, 8, roots, reuse)}


def _cfg(spec=QWEN, layout=8):
    """The attributes ``pf_generate`` reads from a loaded checkpoint."""
    c = R._Cfg(spec, layout)
    c.L, c.eps, c.S, c.name = spec.layers, spec.eps, spec.S, spec.name
    c.out = "."
    return c


# ---- spec lookup ----------------------------------------------------------------------
def test_spec_by_name_and_by_config_path(tmp_path):
    """A name is a model of the manifest; a config path is matched by its dims
    to a manifest model (the manifest's own spec, its split and layout)."""
    assert R.model_spec("qwen3_4b") == R.model_specs()["qwen3_4b"]
    assert R.model_spec("llama160m").qk_norm is False
    cfg = {
        "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 28,
        "rms_norm_eps": 1e-6, "vocab_size": 151936, "model_type": "qwen3",
    }  # fmt: skip
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    got = R.model_spec(str(p))
    assert got.name == "qwen3_0.6b" and got.split_layers == (0, 1, 2)
    # Qwen3-4B's dims from its config: the manifest's qwen3_4b
    cfg4 = dict(cfg, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36)
    cfg4["num_attention_heads"] = 32
    p.write_text(json.dumps(cfg4))
    assert R.model_spec(str(p)).name == "qwen3_4b"


def test_spec_lookup_refuses_unknown_names_and_dims(tmp_path):
    with pytest.raises(ValueError, match="neither a model"):
        R.model_spec("qwen3_99b")
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 2,
        "rms_norm_eps": 1e-6, "vocab_size": 100, "num_hidden_layers": 2,
        "model_type": "llama",
    }))  # fmt: skip
    with pytest.raises(ValueError, match="match no model"):
        R.model_spec(str(p))


# ---- the resolution order -------------------------------------------------------------
def test_exact_signature_is_chosen_over_a_name_match(tmp_path):
    """An exact directory (signature file) and a same-named shared directory of
    the same signature: the exact one is the choice."""
    shared = str(tmp_path / "templates")
    _dir(shared, "fused", "nrm1024")  # shared: by name (owner qwen3_0.6b)
    over = str(tmp_path / "overlay")
    _dir(over, "fused", "nrm1024", QWEN)  # signed with the same signature
    got = _plan(QWEN, [R.Root(over), R.Root(shared)])["nrm1024"]
    assert got.how == "exact" and got.dir == os.path.join(over, "fused", "nrm1024")


def test_by_name_in_a_shared_root_needs_the_same_owner_signature(tmp_path):
    """An unsigned directory of the shared root is ``by name`` for its owner's
    dims: for Qwen3-0.6B's own signatures it is used; for another eps (its norm
    differs) it is missing, with the directory named as the reason."""
    shared = str(tmp_path / "templates")
    _dir(shared, "fused", "nrm1024")
    _dir(shared, "fused", "attn2048")
    own = _plan(QWEN, [R.Root(shared)])
    assert own["nrm1024"].how == "by name"
    assert own["attn2048"].how == "by name"
    other = _plan(EPS, [R.Root(shared)])
    assert other["attn2048"].how == "by name"  # eps is not in attention
    assert other["nrm1024"].how == "missing"
    assert shared in other["nrm1024"].reason and "signature" in other["nrm1024"].reason


def test_unsigned_outside_a_shared_root_is_not_used(tmp_path):
    """No signature file and no shared owner: no record, so no use, and the
    reason says so."""
    root = str(tmp_path / "scratch")  # not a shipped root's name
    _dir(root, "fused", "nrm1024")
    got = _plan(QWEN, [R.Root(root)])["nrm1024"]
    assert got.how == "missing" and "no signature" in got.reason
    # a declared owner makes it count
    got = _plan(QWEN, [R.Root(root, QWEN)])["nrm1024"]
    assert got.how == "by name"


def test_symlink_into_a_shared_root_is_by_name(tmp_path):
    """An overlay's symlink to a shipped directory is found through its real
    root (the shared root's name), not through the overlay's."""
    shared = str(tmp_path / "templates")
    real = _dir(shared, "fused", "nrm1024")
    over = str(tmp_path / "overlay")
    os.makedirs(os.path.join(over, "fused"))
    os.symlink(real, os.path.join(over, "fused", "nrm1024"))
    got = _plan(QWEN, [R.Root(over)])["nrm1024"]
    assert got.how == "by name" and got.dir == os.path.join(over, "fused", "nrm1024")


def test_verified_reuse_needs_an_equal_record_and_its_source(tmp_path):
    """A stored ``equal`` record of the key's signature: the source directory
    (the record's ``from``) is the choice. ``differ`` or no source: missing."""
    shared = str(tmp_path / "templates")
    _dir(shared, "fused", "nrm1024")  # Qwen3-0.6B's: the source
    sid_eps = R.sig_id(R.signature("nrm1024", EPS.dims()))
    sid_q = R.sig_id(R.signature("nrm1024", QWEN.dims()))
    rec = {"name": "nrm1024", "verdict": "equal", "from": sid_q}
    got = _plan(EPS, [R.Root(shared)], {sid_eps: rec})["nrm1024"]
    assert got.how == "reuse" and got.dir == os.path.join(shared, "fused", "nrm1024")
    rec["verdict"] = "differ"
    got = _plan(EPS, [R.Root(shared)], {sid_eps: rec})["nrm1024"]
    assert got.how == "missing" and "differ" in got.reason
    rec["verdict"] = "equal"
    got = _plan(EPS, [R.Root(str(tmp_path / "empty"))], {sid_eps: rec})["nrm1024"]
    assert got.how == "missing" and "not in the roots" in got.reason


def test_nothing_is_a_guess(tmp_path):
    """With no roots every key is missing, and no key is ``by name`` or
    ``reuse`` without its record."""
    got = R.plan(QWEN, 8, [R.Root(str(tmp_path / "none"))])
    assert [c.how for c in got] == ["missing"] * len(got) and len(got) == 20


# ---- pf_generate's resolve_templates -------------------------------------------------
def _shipped_dirs(root, names):
    """Unsigned directories of ``names`` under ``root`` (where ``pf_templates``
    puts them)."""
    for n in names:
        os.makedirs(tpl.template_dir(root, n), exist_ok=True)


def test_resolve_templates_logs_each_key_and_returns_the_directories(tmp_path, capsys):
    shared = str(tmp_path / "templates")  # a shipped root's name: owner qwen3_0.6b
    _shipped_dirs(shared, R.keys_for(QWEN, 8))
    res = pg.resolve_templates(_cfg(), shared)
    assert set(res) == set(R.keys_for(QWEN, 8))
    assert res["nrm1024"] == tpl.template_dir(shared, "nrm1024")
    out = capsys.readouterr().out
    assert "templates qwen3_0.6b layout 8" in out and "20 by name" in out
    assert "nrm1024" in out and "by name" in out


def test_missing_keys_exit_2_and_name_the_build_command(tmp_path, capsys):
    root = str(tmp_path / "overlay")
    os.makedirs(root)
    with pytest.raises(SystemExit) as e:
        pg.resolve_templates(_cfg(), root)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "20 template key(s) of qwen3_0.6b at layout 8" in err
    assert "nrm1024" in err and "k1024n151936" in err
    assert "pf_registry.py" in err and " build --model qwen3_0.6b --layout 8" in err
    assert f"--out {os.path.abspath(root)}" in err


def test_a_wrong_spec_is_refused_before_resolving(tmp_path):
    """``--spec`` must have the checkpoint's dims (here Qwen3-0.6B's config
    against Qwen3-4B's spec)."""
    with pytest.raises(SystemExit) as e:
        pg.resolve_templates(_cfg(), str(tmp_path), "qwen3_4b")
    assert "is not the checkpoint" in str(e.value.code)


def test_generator_refuses_a_key_it_was_not_given(tmp_path):
    g = pg.Generator(_cfg(), str(tmp_path))
    g.tmap = {"nrm1024": "/x/fused/nrm1024"}
    assert g.tdir("nrm1024") == "/x/fused/nrm1024"
    with pytest.raises(RuntimeError, match="not a template key"):
        g.tdir("anorm1024")


def test_llama_layout_keys_resolve_with_their_own_spec(tmp_path, capsys):
    shared = str(tmp_path / "templates_llama")
    _shipped_dirs(shared, R.keys_for(LLAMA, 3))
    res = pg.resolve_templates(_cfg(LLAMA, 3), shared, "llama160m")
    assert len(res) == len(R.keys_for(LLAMA, 3))
    assert "by name" in capsys.readouterr().out


def test_an_exact_directory_that_fails_the_table_check_is_missing(tmp_path):
    """With a ``verify`` (``pf_generate --verify-tables``), an exact directory
    whose weight-table check fails is not used: it is ``missing`` and the
    reason names the check. A passing check keeps it ``exact``; without
    ``verify`` nothing changes."""
    root = str(tmp_path / "overlay")
    d = _dir(root, "fused", "nrm1024", QWEN)
    bad = lambda dd, name: "the template's weight table is not found once"  # noqa: E731
    got = _plan(QWEN, [R.Root(root)])["nrm1024"]
    assert got.how == "exact"
    got = {c.name: c for c in R.plan(QWEN, 8, [R.Root(root)], None, bad)}["nrm1024"]
    assert got.how == "missing" and got.dir is None
    assert d in got.reason and "weight-table check" in got.reason
    ok = lambda dd, name: None  # noqa: E731
    got = {c.name: c for c in R.plan(QWEN, 8, [R.Root(root)], None, ok)}["nrm1024"]
    assert got.how == "exact"
    # the check is asked only of exact directories
    seen = []
    R.plan(QWEN, 8, [R.Root(str(tmp_path / "none"))], None,
           lambda dd, name: seen.append(name))  # fmt: skip
    assert seen == []
