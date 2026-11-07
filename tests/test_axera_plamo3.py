"""PLaMo 3 float reference and its ONNX graphs, on a small random model (no checkpoint, no Docker).

The reference is checked for causality. The decode graph (KV cache in and out) and the static
layer graph are checked token by token against the reference, on both kinds of layer (plain
RoPE and YaRN full attention).
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

ort = pytest.importorskip("onnxruntime")
import plamo3_onnx as O  # noqa: E402
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


IDS = [3, 17, 5, 29, 8, 11, 2, 35]


def test_reference_is_causal(model):
    full = model.logits(IDS)
    for n in (1, 3, 6):
        np.testing.assert_allclose(
            model.logits(IDS[:n]), full[:n], rtol=1e-4, atol=1e-5
        )


def test_decode_graph_matches_reference(model, tmp_path):
    path = O.build_decode(
        model, str(tmp_path / "decode.onnx"), max_pos=64, external=False
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = model.logits(IDS)
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    past = {}
    for i in range(CFG["num_hidden_layers"]):
        past[f"past_k_{i}"] = np.zeros((nkv, 0, hd), np.float32)
        past[f"past_v_{i}"] = np.zeros((nkv, 0, hd), np.float32)
    names = [o.name for o in sess.get_outputs()]
    got = []
    for pos, tok in enumerate(IDS):
        feed = dict(past)
        feed["input_ids"] = np.array([tok], np.int64)
        feed["position_ids"] = np.array([pos], np.int64)
        out = sess.run(None, feed)
        got.append(out[0][0])
        for j, name in enumerate(names[1:]):
            past[name.replace("present_", "past_")] = out[1 + j]
    got = np.stack(got)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layer", [0, 1])  # 0: plain RoPE, 1: YaRN full attention
def test_layer_graph_matches_reference(model, tmp_path, layer):
    cap = {}
    h_in = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for i in range(layer):
        h_in = model.layer(i, h_in)
    ref_out = model.layer(layer, h_in, capture=cap)
    path = O.build_layer(
        model, layer, str(tmp_path / f"layer{layer}.onnx"), cache_len=16
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    pk = np.zeros((nkv, 16, hd), np.float32)
    pv = np.zeros((nkv, 16, hd), np.float32)
    outs = []
    for pos in range(len(IDS)):
        mask = np.full((1, 17), -1000.0, np.float32)
        mask[0, :pos] = 0.0
        mask[0, 16] = 0.0
        res = sess.run(
            None,
            {
                "hidden": h_in[pos : pos + 1],
                "position_ids": np.array([pos], np.int64),
                "past_k": pk,
                "past_v": pv,
                "attn_mask": mask,
            },
        )
        outs.append(res[0][0])
        np.testing.assert_allclose(
            res[1][:, 0, :], cap["k"][:, pos, :], rtol=1e-4, atol=1e-4
        )
        pk[:, pos, :] = res[1][:, 0, :]
        pv[:, pos, :] = res[2][:, 0, :]
    np.testing.assert_allclose(np.stack(outs), ref_out, rtol=1e-4, atol=1e-4)
