"""PLaMo 3's Pulsar-free decomposition (``llm_plamo3_decomposition``) equals the PLaMo float
reference, layer by layer and token by token, on a small random model (no checkpoint).

The layers cover both kinds: plain RoPE on the sliding layers and YaRN with the concentration on
the full-attention layers. Each token's decode step reads the reference's own keys and values of
the earlier tokens, so the per-layer check does not accumulate error.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

onnx_reference = pytest.importorskip("onnx.reference")
import llm_plamo3_decomposition as D  # noqa: E402
import plamo3_reference as R  # noqa: E402

CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 40,
    "sliding_window_pattern": 2,
    "window_size": 2048,
    "rope_theta": 1000000,
    "rope_local_theta": 10000,
    "rope_scaling_factor": 4.0,
    "rope_parameters": {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 1000000,
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
    },
}
IDS = [3, 17, 5, 29, 8, 11, 2, 35]


def _weights(seed=0):
    rng = np.random.RandomState(seed)
    H, inter, hd = CFG["hidden_size"], CFG["intermediate_size"], CFG["head_dim"]
    nh, nkv, V = (
        CFG["num_attention_heads"],
        CFG["num_key_value_heads"],
        CFG["vocab_size"],
    )

    def r(*shape, scale=0.2):
        return (rng.randn(*shape) * scale).astype(np.float32)

    t = {
        "model.embed_tokens.weight": r(V, H),
        "model.norm.weight": r(H, scale=0.1),
        "lm_head.weight": r(V, H),
    }
    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.layers.{i}."
        t[p + "pre_mixer_norm.weight"] = r(H, scale=0.1)
        t[p + "post_mixer_norm.weight"] = r(H, scale=0.1)
        t[p + "pre_mlp_norm.weight"] = r(H, scale=0.1)
        t[p + "post_mlp_norm.weight"] = r(H, scale=0.1)
        t[p + "mixer.qkv_proj.weight"] = r((nh + 2 * nkv) * hd, H)
        t[p + "mixer.o_proj.weight"] = r(H, nh * hd)
        t[p + "mixer.q_norm.weight"] = r(hd, scale=0.1)
        t[p + "mixer.k_norm.weight"] = r(hd, scale=0.1)
        t[p + "mlp.gate_up_proj.weight"] = r(2 * inter, H)
        t[p + "mlp.down_proj.weight"] = r(H, inter)
    return t


@pytest.fixture(scope="module")
def model():
    return R.Plamo3Reference(CFG, R.DictWeights(_weights()))


def _run(graph, feeds):
    return onnx_reference.ReferenceEvaluator(graph).run(None, feeds)


def test_plamo3_layers_decode_matches_reference(model):
    T = len(IDS)
    S = T - 1  # cache slots; the new token is the extra slot S
    kvd = CFG["num_key_value_heads"] * CFG["head_dim"]
    h = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for li in range(model.L):
        cap = {}
        ref_out = model.layer(li, h, capture=cap)
        k_rows = (
            cap["k"].transpose(1, 0, 2).reshape(T, kvd)
        )  # [T, nkv*hd], the keys as stored
        v_rows = cap["v"].transpose(1, 0, 2).reshape(T, kvd)
        graph = D.layer_graph(model, li, S)
        for p in range(T):
            Kc = np.zeros((S, kvd), np.float32)
            Vc = np.zeros((S, kvd), np.float32)
            Kc[:p] = k_rows[:p]
            Vc[:p] = v_rows[:p]
            mask = np.full((1, S + 1), -1e9, np.float32)
            mask[0, :p] = 0.0
            mask[0, S] = 0.0
            feeds = {
                "input": h[p : p + 1].copy(),
                "K_cache": Kc,
                "V_cache": Vc,
                "mask": mask,
                **{k: v for k, v in D.rope_inputs(model, li, p).items()},
            }
            out, k_out, v_out = _run(graph, feeds)
            np.testing.assert_allclose(out[0], ref_out[p], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(k_out[-1], k_rows[p], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(v_out[-1], v_rows[p], rtol=1e-4, atol=1e-4)
        h = ref_out


def test_plamo3_post_matches_reference_logits(model):
    h = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for li in range(model.L):
        h = model.layer(li, h)
    graph = D.post_graph(model)
    (logits,) = _run(graph, {"input": h[-1:].copy()})
    np.testing.assert_allclose(logits[0], model.logits(IDS)[-1], rtol=1e-4, atol=1e-4)
