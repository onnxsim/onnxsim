"""Replaying a Pulsar2 build's quantisation offline, checked without a build.

`scripts/axera/replay.py` reads `quant/quant_axmodel.json` -- the table
`pulsar2 build` writes -- and re-runs the float graph with exactly those
scales, which reproduced four real AX650N measurements to within 0.19 dB.
The tests here build the two files a real output directory would contain, so
they need neither Docker, a card, nor a committed multi-megabyte fixture.
"""

import json
import os
import sys

import numpy as np
import onnx
import onnx.parser
from onnx import numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import replay  # noqa: E402


def _policy(per_channel=False):
    return {
        "PER_TENSOR": not per_channel,
        "PER_CHANNEL": per_channel,
        "LINEAR": True,
        "EXPONENTIAL": False,
        "SYMMETRICAL": per_channel,
        "ASYMMETRICAL": not per_channel,
        "POWER_OF_2": False,
        "PER_CHANNEL_RE_GROUP": False,
        "PER_BLOCK": False,
    }


def _write_build(tmp_path, entries, fused_graph=None):
    """A minimal `out_*/quant/` directory in the shape Pulsar2 writes one.

    `entries` is `{op: {tensor: (bit_width, scale, zero_point, per_channel)}}`.
    """
    quant = tmp_path / "quant"
    quant.mkdir(parents=True, exist_ok=True)
    values, configs = {}, {}
    for op, tensors in entries.items():
        configs[op] = {}
        for tensor, (bits, scale, zero, per_channel) in tensors.items():
            h = str(abs(hash((op, tensor))) % (10**10))
            values[h] = {
                "scale": list(np.atleast_1d(scale).astype(float)),
                "zero_point": list(np.atleast_1d(zero).astype(float)),
            }
            configs[op][tensor] = {
                "bit_width": bits,
                "policy": _policy(per_channel),
                "state": "ACTIVATED",
                "hash": int(h),
                "quant_min": -(2 ** (bits - 1)) if per_channel else 0,
                "quant_max": 2 ** (bits - 1) - 1 if per_channel else 2**bits - 1,
            }
    (quant / "quant_axmodel.json").write_text(
        json.dumps(
            {
                "quant_config": {},
                "tensor_configs": configs,
                "dispatchings": {op: "AX_NPU_AX650_INT8" for op in entries},
                "values": values,
            }
        )
    )
    if fused_graph is not None:
        onnx.save(fused_graph, str(quant / "quant_axmodel.onnx"))
    return str(tmp_path)


def _conv_model(w):
    model = onnx.parser.parse_model(
        '<ir_version: 10, opset_import: ["": 17]>'
        f" g (float[1, {w.shape[1]}, 8] x) => (float[1, {w.shape[0]}, 8] y) {{"
        "   y = Conv <kernel_shape = [3], pads = [1, 1]> (x, w)"
        " }"
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    return model


def _affine(x, bits, scale, zero):
    n = 2**bits - 1
    codes = np.clip(np.rint(x / scale) + zero, 0, n)
    return ((codes - zero) * scale).astype(np.float32)


def test_scales_load_from_the_table_the_compiler_writes(tmp_path):
    build = _write_build(
        tmp_path,
        {
            "conv": {
                "x": (8, 0.031, 130.0, False),
                "w": (8, [0.004, 0.002], 0.0, True),
                "y": (16, 9.3e-05, 0.0, False),
            }
        },
    )
    scales = replay.load_scales(build)
    assert set(scales) == {"x", "w", "y"}
    assert scales["x"][0] == 8 and scales["y"][0] == 16
    assert scales["w"][3] is True and scales["x"][3] is False
    assert list(scales["w"][1]) == [0.004, 0.002]


def test_fused_away_tensors_are_dropped(tmp_path):
    """A tensor the compiler folded inside one of its own ops still has a table
    entry, but nothing on the card rounds it. Quantising it anyway invented
    6 dB of error on the Audio8 decoder."""
    fused = onnx.parser.parse_model(
        '<ir_version: 10, opset_import: ["": 17]>'
        " g (float[1, 4, 8] x) => (float[1, 4, 8] y) {"
        "   y = Relu (x)"
        " }"
    )
    build = _write_build(
        tmp_path,
        {
            "op": {
                "x": (8, 0.03, 128.0, False),
                "gone": (8, 0.02, 128.0, False),
                "y": (8, 0.01, 128.0, False),
            }
        },
        fused_graph=fused,
    )
    assert set(replay.load_scales(build)) == {"x", "y"}
    assert set(replay.load_scales(build, fused_aware=False)) == {"x", "gone", "y"}
    assert replay.surviving_edges(build) == {"x", "y"}


def test_quantising_only_drops_the_movement_ops(tmp_path):
    """`AxReshape` and friends carry codes without recomputing them. Whether
    the card rounds their output is unresolved -- the two rules bracket the one
    measurement rather than settling it -- so the strict rule is opt-in and the
    default stays on the pessimistic side."""
    fused = onnx.parser.parse_model(
        '<ir_version: 10, opset_import: ["": 17]>'
        " g (float[1, 4, 8] x) => (float[1, 8, 4] moved) {"
        "   y = Relu (x)"
        "   moved = Transpose <perm = [0, 2, 1]> (y)"
        " }"
    )
    for node in fused.graph.node:
        node.op_type = "AxQuantizedRelu" if node.op_type == "Relu" else "AxTranspose"
    build = _write_build(
        tmp_path,
        {
            "op": {
                "x": (8, 0.03, 128.0, False),
                "y": (8, 0.02, 128.0, False),
                "moved": (8, 0.01, 128.0, False),
            }
        },
        fused_graph=fused,
    )
    assert replay.surviving_edges(build) == {"x", "y", "moved"}
    assert replay.surviving_edges(build, quantising_only=True) == {"x", "y"}


def test_no_fused_graph_means_no_filter(tmp_path):
    build = _write_build(tmp_path, {"op": {"x": (8, 0.03, 128.0, False)}})
    assert replay.surviving_edges(build) is None
    assert set(replay.load_scales(build)) == {"x"}


def test_the_replay_reproduces_the_affine_quantisation_exactly(tmp_path):
    """The inserted arithmetic has to be the quantiser, not something like it:
    a replay is only worth having if it is exact."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal((6, 4, 3)).astype(np.float32) * 0.3
    model = _conv_model(w)
    x_scale, x_zp = 0.03115052543580532, 130.0
    y_scale, y_zp = 9.281877282774076e-05, 32768.0
    w_scale = np.abs(w).max(axis=(1, 2)) / 127.5
    build = _write_build(
        tmp_path,
        {
            "y": {
                "x": (8, x_scale, x_zp, False),
                "w": (8, w_scale, np.zeros(6), True),
                "y": (16, y_scale, y_zp, False),
            }
        },
    )

    x = rng.standard_normal((1, 4, 8)).astype(np.float32) * 0.5
    got = replay.replay(model, build, {"x": x})[0]

    xq = _affine(x, 8, x_scale, x_zp)
    s = w_scale.reshape(-1, 1, 1)
    wq = (np.clip(np.rint(w / s), -128, 127) * s).astype(np.float32)
    pad = np.pad(xq, ((0, 0), (0, 0), (1, 1)))
    acc = sum(
        np.einsum("oi,bil->bol", wq[:, :, k], pad[:, :, k : k + 8]) for k in range(3)
    )
    want = _affine(acc, 16, y_scale, y_zp)
    assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()


def test_widening_the_activations_stalls_on_the_int8_weights(tmp_path):
    """The replay reproduces the plateau the card shows: widen the activations
    to 16 bits and the result improves, then stops, because the weights are
    still INT8 -- which is what `precision.weight_residual_split` is for."""
    rng = np.random.default_rng(9)
    w = rng.standard_normal((6, 4, 3)).astype(np.float32) * 0.3
    model = _conv_model(w)
    x = rng.standard_normal((1, 4, 8)).astype(np.float32) * 0.5
    ort = __import__("onnxruntime")
    ref = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]
    w_scale = np.abs(w).max(axis=(1, 2)) / 127.5
    snrs = {}
    for bits in (8, 16):
        n = 2**bits - 1
        build = _write_build(
            tmp_path / f"b{bits}",
            {
                "y": {
                    "x": (bits, 2 * float(np.abs(x).max()) / n, n // 2, False),
                    "w": (8, w_scale, np.zeros(6), True),
                    "y": (bits, 2 * float(np.abs(ref).max()) / n, n // 2, False),
                }
            },
        )
        got = replay.replay(_conv_model(w), build, {"x": x})[0]
        snrs[bits] = replay.snr_db(ref, got)
    import precision

    assert 20 < snrs[8] < 60
    assert snrs[16] > snrs[8] + 3
    # and it stops at the weights: the closed form for INT8 per-channel weight
    # error is what the 16-bit activation run is left sitting on
    assert abs(snrs[16] - precision.weight_error_db(w, axis=0)) < 6


def test_a_tensor_with_no_entry_is_left_in_float(tmp_path):
    """Pulsar2 leaves shape operands and some ops in float; the replay must not
    invent a scale for them."""
    rng = np.random.default_rng(1)
    w = rng.standard_normal((6, 4, 3)).astype(np.float32) * 0.3
    build = _write_build(
        tmp_path,
        {"y": {"w": (8, np.abs(w).max(axis=(1, 2)) / 127.5, np.zeros(6), True)}},
    )
    model = _conv_model(w)
    quantised, n_act, n_w = replay.insert_qdq(model, replay.load_scales(build))
    assert (n_act, n_w) == (0, 1)
    assert [n.op_type for n in quantised.graph.node] == ["Conv"]
