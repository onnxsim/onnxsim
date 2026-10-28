"""Host-only checks of the template registry (``scripts/axera/llm_pulsar_free/
pf_registry.py``): signatures, the key list of a model at a layout, the
manifest round trip, names and aliases, and the missing / check / verified
states. No Pulsar2, no Docker, no card, no checkpoint.

The two comparisons with real shipped templates (``compare_variants``) need
the shared template roots: they are skipped unless ``LLM_PF_TEMPLATES`` and
``LLM_PF_TEMPLATES_LLAMA`` name existing directories.
"""

import json
import os
import sys
from dataclasses import replace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import pf_registry as R  # noqa: E402
import pf_templates as tpl  # noqa: E402

QWEN = R.SPECS["qwen3_0.6b"]
LLAMA = R.SPECS["llama160m"]
Q4B = R.SPECS["qwen3_4b"]

#: ``pf_templates.needed`` of Qwen3-0.6B at layout 8 (the default), pinned
QWEN_L8 = [
    "nrm1024", "anorm1024", "silu3072", "attn2048", "qkf2048", "lanorm2048",
    "lanorm3072", "slanorm3072", "lanorml2048x6144", "nrmqkr1024",
    "slanormqkr3072", "alanorml2048x6144", "alyrqkr2048", "add6144", "add1024",
    "k1024n7168", "k2048n1024", "k1024n6144", "k3072n1024", "k1024n151936",
]  # fmt: skip


def replace_dims(d, **kw):
    return tpl.Dims(**{**d.__dict__, **kw})


def _cfg_dims():
    return tpl.Dims(1024, 3072, 16, 8, 128, 1e-06, 151936, 128)


def _write_dir(root, group, name, sig=None):
    d = os.path.join(root, group, name)
    os.makedirs(d, exist_ok=True)
    if sig is not None:
        R._write_sig(d, name, R.sig_id(sig), sig, QWEN)
    return d


def test_keys_for_pins_the_shipped_key_list():
    """The key list of a shipped model is ``pf_templates.needed``'s, unchanged."""
    assert R.keys_for(QWEN, 8) == QWEN_L8


def test_keys_have_no_duplicates_at_any_layout():
    for spec in (QWEN, LLAMA, Q4B):
        for layout in range(12):
            keys = R.keys_for(spec, layout)
            assert len(keys) == len(set(keys)), (spec.name, layout)
    assert R.keys_for(LLAMA, 3) == [
        "nrm768", "anorm768", "silu3072", "attn768", "rope768", "lanorm768",
        "lanorm3072", "slanorm3072", "k768n3840", "k768n768", "k768n6144",
        "k3072n768", "k768n32000",
    ]  # fmt: skip
    assert len(R.keys_for(Q4B, 8)) == 20


def test_layout_out_of_range_is_refused():
    with pytest.raises(ValueError):
        R.keys_for(QWEN, 12)


def test_names_parse_and_round_trip():
    """Every name of a shipped set parses to a name that prints the same."""
    for spec in (QWEN, LLAMA, Q4B):
        for layout in range(12):
            for name in R.keys_for(spec, layout):
                p = R.parse(name)
                assert p.text == name, name
    assert R.parse("qkrope2048t") == R.Name("qkrope", 2048, None, True)
    assert R.parse("lanorml2048x6144") == R.Name("lanorml", 2048, 6144, False)
    assert R.parse("k1024n151936") == R.Name("linear", 1024, 151936, False)
    for bad in ("foo1024", "nrm", "k1024", "1024"):
        with pytest.raises(ValueError):
            R.parse(bad)


def test_signature_is_deterministic_and_serialisable():
    d = _cfg_dims()
    sig = R.signature("nrm1024", d)
    assert json.loads(json.dumps(sig, sort_keys=True)) == sig
    assert R.sig_id(sig) == R.sig_id(R.signature("nrm1024", _cfg_dims()))
    assert len(R.sig_id(sig)) == 16


def test_signature_fields_follow_the_dims_each_kind_reads():
    d = _cfg_dims()
    # eps is in every norm, not in rope / silu / linears; the attention needs
    # the context and the head split, not eps
    assert R.signature("nrm1024", d)["dims"] == {"eps": 1e-06}
    assert R.signature("silu3072", d)["dims"] == {}
    assert R.signature("rope2048", d)["dims"] == {}
    assert R.signature("attn2048", d)["dims"] == {
        "nh": 16, "nkv": 8, "hd": 128, "S": 128,
    }  # fmt: skip
    assert R.signature("qkf2048", d)["dims"]["hd"] == 128
    assert R.signature("k1024n6144", d)["dims"] == {}
    # the q / k norm and the graph variant
    assert R.signature("rope2048", d)["variant"] == {"qk_norm": False}
    assert R.signature("qkrope2048t", d)["variant"] == {"qk_norm": True, "alt": True}
    assert R.signature("nrmqkr1024", d)["variant"] == {"qk_norm": True, "rot": True}
    assert R.signature("alyrqkrs2048", d)["variant"]["split"] is True
    # the Pulsar2 config: the layer ops the build quantizes, the draws
    p = R.signature("nrm1024", d)["pulsar2"]
    assert p["dtype"] == "U16" and p["layer_ops"] == tpl.NORM_OPS
    assert p["draws"] == {"count": 18, "seed": 300}


def test_signature_changes_with_the_dims_it_reads():
    """Equal names, another eps or head split: another signature. The same
    eps on another model with equal dims: the same signature."""
    d = _cfg_dims()
    other_eps = tpl.Dims(1024, 3072, 16, 8, 128, 1e-05, 151936, 128)
    assert R.sig_id(R.signature("nrm1024", d)) != R.sig_id(
        R.signature("nrm1024", other_eps)
    )
    # eps is not in a silu or an attention: their signatures stay
    assert R.sig_id(R.signature("silu3072", d)) == R.sig_id(
        R.signature("silu3072", other_eps)
    )
    assert R.sig_id(R.signature("attn2048", d)) == R.sig_id(
        R.signature("attn2048", replace_dims(d, eps=1e-05))
    )
    # a head split with the same width: another signature for the q/k norm
    split = tpl.Dims(1024, 3072, 8, 8, 256, 1e-06, 151936, 128)
    assert R.sig_id(R.signature("qkf2048", d)) != R.sig_id(
        R.signature("qkf2048", split)
    )


def test_model_specs_from_config_match_the_shipped_dims():
    """``ModelSpec.from_config`` on the Hugging Face configs' fields."""
    cfg = {
        "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 28,
        "rms_norm_eps": 1e-6, "vocab_size": 151936, "model_type": "qwen3",
    }  # fmt: skip
    s = R.ModelSpec.from_config("x", cfg, split=((0, 1, 2), ("gu", "down_proj")))
    assert s.qk_norm and s.kvd == 1024 and s.dims() == _cfg_dims()
    llama = {
        "hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12,
        "num_key_value_heads": 12, "head_dim": None, "num_hidden_layers": 12,
        "rms_norm_eps": 1e-6, "vocab_size": 32000, "model_type": "llama",
    }  # fmt: skip
    s = R.ModelSpec.from_config("y", llama, layout=3)
    assert s.hd == 64 and not s.qk_norm and s.default_layout == 3


def test_manifest_round_trip(tmp_path):
    man = R.manifest_for({"qwen3_0.6b": QWEN, "llama160m": LLAMA}, layouts=range(3, 9))
    path = str(tmp_path / "m.json")
    R.dump_manifest(man, path)
    back = R.load_manifest(path)
    assert back == man
    R.dump_manifest(back, path)
    again = open(path).read()
    assert json.loads(again) == man
    assert set(back["models"]) == {"qwen3_0.6b", "llama160m"}
    # every key of every layout is in the signature table
    sids = {R.name_sig(n, QWEN)[0] for n in R.keys_for(QWEN, 8)}
    assert sids <= set(back["signatures"])


def test_committed_manifest_has_the_signatures_of_its_models():
    """``template_registry.json`` holds the signature of every key of the
    shipped models at their default layouts and of the third model at 8
    (regenerate with ``pf_registry.py manifest``)."""
    man = R.load_manifest()
    assert set(man["models"]) == set(R.SPECS)
    for spec in (QWEN, LLAMA, Q4B):
        for name in R.keys_for(spec, spec.default_layout):
            sid, sig = R.name_sig(name, spec)
            assert man["signatures"][sid]["signature"] == sig, name


def test_names_alias_their_dirs_whether_or_not_they_carry_a_signature(tmp_path):
    """A legacy directory (no ``signature.json``, found by the owner's dims)
    and a directory written by ``resolve`` both serve the same key."""
    root = str(tmp_path / "root")
    _write_dir(root, "fused", "nrm1024")  # legacy: no signature file
    own = R.Root(root, QWEN)
    found = R.scan([own])
    sid, _ = R.name_sig("nrm1024", QWEN)
    assert found[sid][0]["name"] == "nrm1024"
    assert found[sid][0]["how"].startswith("owner")
    # a directory with a signature file: found by the signature, not the owner
    root2 = str(tmp_path / "root2")
    _write_dir(root2, "fused", "anorm1024", R.signature("anorm1024", _cfg_dims()))
    found2 = R.scan([R.Root(root2)])
    sid2, _ = R.name_sig("anorm1024", QWEN)
    assert found2[sid2][0]["how"] == "signature.json"


def test_missing_and_verified_reporting(tmp_path):
    """Missing: no directory of the name. Check: the same name, another
    signature. Verified: a stored reuse record ``equal`` against a directory
    that exists. Exact: the same signature."""
    root = str(tmp_path / "root")
    _write_dir(root, "fused", "nrm1024")  # the owner's (qwen dims)
    _write_dir(root, "fused", "attn2048")
    own = [R.Root(root, QWEN)]
    found = R.scan(own)
    eps_spec = replace(QWEN, eps=1e-05)  # norms change, attn does not

    needs = {n.name: n for n in R.status(eps_spec, 8, found, {})}
    assert needs["attn2048"].state == "exact"
    assert needs["nrm1024"].state == "check"
    assert needs["nrm1024"].candidates[0]["dir"].endswith("fused/nrm1024")
    assert needs["anorm1024"].state == "missing"
    assert needs["silu3072"].state == "missing"
    s = R.summary(list(needs.values()))
    assert s["keys"] == 20 and s["exact"] == 1 and s["check"] == 1
    assert s["missing"] == 18 and s["verified"] == 0

    # a stored equal verdict for the new signature (against the owner's dir)
    sid_new, _ = R.name_sig("nrm1024", eps_spec)
    sid_old, _ = R.name_sig("nrm1024", QWEN)
    reuse = {sid_new: {"verdict": "equal", "from": sid_old, "name": "nrm1024"}}
    needs = {n.name: n for n in R.status(eps_spec, 8, found, reuse)}
    assert needs["nrm1024"].state == "verified"
    assert needs["nrm1024"].dirs[0]["name"] == "nrm1024"
    # a "differ" verdict stays a check
    reuse[sid_new]["verdict"] = "differ"
    assert {n.name: n for n in R.status(eps_spec, 8, found, reuse)}[
        "nrm1024"
    ].state == "check"


def test_no_roots_means_every_key_is_missing():
    needs = R.status(QWEN, 8, {}, {})
    assert all(n.state == "missing" for n in needs)
    assert R.summary(needs) == {
        "keys": 20, "exact": 0, "verified": 0, "check": 0, "missing": 20,
    }  # fmt: skip


def test_compare_variants_against_shipped_templates():
    """``compare_variants`` on real builds (no Pulsar2): a shipped template
    equals itself, and llama's ``lanorm3072`` (H 768) differs from Qwen's (H
    1024), a verdict the registry stores as ``differ``. Skipped unless
    ``LLM_PF_TEMPLATES`` / ``LLM_PF_TEMPLATES_LLAMA`` name built roots."""
    shared = os.environ.get("LLM_PF_TEMPLATES", "")
    llama = os.environ.get("LLM_PF_TEMPLATES_LLAMA", "")
    if not os.path.isdir(os.path.join(shared, "fused", "nrm1024", "v0", "out")):
        pytest.skip("LLM_PF_TEMPLATES has no built nrm1024")
    same = R.compare_variants(
        shared, "nrm1024", os.path.join(shared, "fused", "nrm1024")
    )
    assert same["verdict"] == "equal"
    if not os.path.isdir(os.path.join(llama, "fused", "lanorm3072", "v0", "out")):
        pytest.skip("LLM_PF_TEMPLATES_LLAMA has no built lanorm3072")
    diff = R.compare_variants(
        llama, "lanorm3072", os.path.join(shared, "fused", "lanorm3072")
    )
    assert diff["verdict"] == "differ"
