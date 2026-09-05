"""Real-hardware structural analysis of the compiled `.axmodel` blobs this
project's own research identified as Axera's real "Wbt" (Weight Table --
the `npu_params` initializer) and "mcode" (the compiled command-queue
program -- the `<neu_key>`-named initializer) terms. See
scripts/axera/README.md's "Applying the new vocabulary to a real
`.axmodel`, and a real mcode-size finding" section for the full narrative
(a real 1-through-10-identical-Conv-layer sweep) this file locks a smaller
slice of in as a regression test.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _n_conv_model(n):
    """`n` sequential, identically-shaped Conv layers -- same spatial size
    throughout (`pads=[1,1,1,1]`) so every layer is a truly identical unit,
    isolating per-op growth from shape-dependent effects."""
    rng = np.random.RandomState(0)
    nodes = []
    inits = []
    prev = "x"
    for i in range(n):
        w = (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)
        wname = f"w{i}"
        inits.append(numpy_helper.from_array(w, name=wname))
        out = f"y{i}" if i < n - 1 else "y"
        nodes.append(helper.make_node("Conv", [prev, wname], [out], pads=[1, 1, 1, 1]))
        prev = out
    graph = helper.make_graph(
        nodes,
        f"g_{n}conv",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 16, 16])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, 16, 16])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _single_conv_model(cin, cout, k):
    """One `Conv(cin -> cout, kxk)` -- for shape-sweep experiments, unlike
    `_n_conv_model`'s op-count sweep."""
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin, k, k) * 0.1).astype(np.float32)
    pad = k // 2
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[pad, pad, pad, pad])],
        f"g_cin{cin}_cout{cout}_k{k}",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, cin, 16, 16])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, cout, 16, 16])],
        initializer=[numpy_helper.from_array(w, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _asym_kernel_model(kh, kw, cin=4, cout=4, insz=16):
    """One `Conv` with an asymmetric `kh x kw` kernel -- same total weight
    count regardless of orientation (`kh * kw` is the same for `(3,1)` and
    `(1,3)`), isolating orientation from raw weight data size."""
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin, kh, kw) * 0.1).astype(np.float32)
    ph, pw = kh // 2, kw // 2
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[ph, pw, ph, pw])],
        f"g_kernel_{kh}x{kw}",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, cin, insz, insz]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, cout, insz, insz]
            )
        ],
        initializer=[numpy_helper.from_array(w, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _autopad_model(use_auto_pad, cin=4, cout=4, k=3, insz=16):
    """One `Conv`, either using `auto_pad="SAME_UPPER"` or the numerically
    equivalent explicit `pads` -- to check whether auto_pad leaves any
    trace in mcode after normalization."""
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin, k, k) * 0.1).astype(np.float32)
    if use_auto_pad:
        node = helper.make_node("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER")
    else:
        node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node],
        f"g_autopad_{use_auto_pad}",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, cin, insz, insz]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, cout, insz, insz]
            )
        ],
        initializer=[numpy_helper.from_array(w, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _dilation_conv_model(dilation, pad, cin=4, cout=4, k=3, insz=16):
    """One `Conv` with a specific dilation/padding -- padding chosen by the
    caller so output shape (and thus mcode's total serialized length) stays
    identical across dilation values, isolating dilation's own encoding
    from the wholesale re-serialization a shape change triggers."""
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin, k, k) * 0.1).astype(np.float32)
    out = insz + 2 * pad - dilation * (k - 1) - 1 + 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "w"],
                ["y"],
                strides=[1, 1],
                dilations=[dilation, dilation],
                pads=[pad, pad, pad, pad],
            )
        ],
        f"g_dilation{dilation}",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, cin, insz, insz]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, cout, out, out]
            )
        ],
        initializer=[numpy_helper.from_array(w, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _two_conv_model(vary_first, dilation, pad, cin=4, mid=4, cout=4, k=3, insz=16):
    """Two chained `Conv`s with a real intermediate activation flowing
    between them (not a graph-boundary tensor) -- `vary_first` selects
    which of the two gets the dilation/padding override, the other stays
    fixed at dilation=1/pad=1."""
    rng = np.random.RandomState(0)
    w1 = (rng.randn(mid, cin, k, k) * 0.1).astype(np.float32)
    w2 = (rng.randn(cout, mid, k, k) * 0.1).astype(np.float32)
    d1, p1 = (dilation, pad) if vary_first else (1, 1)
    d2, p2 = (1, 1) if vary_first else (dilation, pad)
    mid_sz = insz + 2 * p1 - d1 * (k - 1) - 1 + 1
    out_sz = mid_sz + 2 * p2 - d2 * (k - 1) - 1 + 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w1"], ["mid"], dilations=[d1, d1], pads=[p1, p1, p1, p1]
            ),
            helper.make_node(
                "Conv", ["mid", "w2"], ["y"], dilations=[d2, d2], pads=[p2, p2, p2, p2]
            ),
        ],
        f"g_two_conv_vary{'1' if vary_first else '2'}_{dilation}",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, cin, insz, insz]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, cout, out_sz, out_sz]
            )
        ],
        initializer=[
            numpy_helper.from_array(w1, name="w1"),
            numpy_helper.from_array(w2, name="w2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _build_and_get_mcode_bytes(work_dir, model, input_shape):
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.randn(*input_shape).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)
    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error
    compiled = onnx.load(result.axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    mcode_key = info["dotneus"][0]["neu_key"]
    return bytes(inits[mcode_key].raw_data)


def _build_and_get_wbt_and_mcode_bytes(work_dir, model, input_shape):
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.randn(*input_shape).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)
    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error
    compiled = onnx.load(result.axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    dotneu = info["dotneus"][0]
    wbt_key = dotneu["extra_inputs"][0]["const_data_key"]
    mcode_key = dotneu["neu_key"]
    return bytes(inits[wbt_key].raw_data), bytes(inits[mcode_key].raw_data)


def _build_and_get_blob_sizes_for_model(work_dir, model, input_name, input_shape):
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))

    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.randn(*input_shape).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": input_name,
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)

    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error

    compiled = onnx.load(result.axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    dotneu = info["dotneus"][0]
    wbt_key = dotneu["extra_inputs"][0]["const_data_key"]
    mcode_key = dotneu["neu_key"]
    return len(inits[wbt_key].raw_data), len(inits[mcode_key].raw_data)


def _build_and_get_blob_sizes(tmp_path, n):
    return _build_and_get_blob_sizes_for_model(
        os.path.join(str(tmp_path), f"conv{n}"), _n_conv_model(n), "x", (1, 4, 16, 16)
    )


def test_wbt_and_mcode_scale_differently_with_identical_ops(tmp_path):
    """Confirmed real (see the README's full 1-10 layer sweep): Axera's
    "Wbt" (Weight Table, the `npu_params` blob) grows by an *exact* constant
    number of bytes per added identical Conv layer -- a flat,
    one-record-per-op concatenation, no compression. Axera's "mcode" (the
    `<neu_key>` compiled command-queue blob) does NOT scale linearly, but
    every observed size delta is an exact multiple of 32 bytes -- consistent
    with a 32-byte-aligned command-queue allocation unit where a variable
    (not fixed) number of units get assigned per op instance.
    """
    sizes = {n: _build_and_get_blob_sizes(tmp_path, n) for n in (2, 3, 4)}
    wbt = {n: s[0] for n, s in sizes.items()}
    mcode = {n: s[1] for n, s in sizes.items()}

    wbt_delta_1 = wbt[3] - wbt[2]
    wbt_delta_2 = wbt[4] - wbt[3]
    assert wbt_delta_1 == wbt_delta_2 > 0, (wbt, "Wbt delta should be constant")

    mcode_delta_1 = mcode[3] - mcode[2]
    mcode_delta_2 = mcode[4] - mcode[3]
    assert mcode_delta_1 % 32 == 0, (mcode, "mcode delta should be a multiple of 32")
    assert mcode_delta_2 % 32 == 0, (mcode, "mcode delta should be a multiple of 32")


def test_mcode_32_byte_unit_holds_under_shape_variation_too(tmp_path):
    """Confirmed real (see the README's cout/cin/kernel-size sweeps): the
    32-byte mcode-serialization-unit finding above isn't specific to
    *repeating* an op -- it holds just as well when a single Conv's *shape*
    changes instead. Also locks in a real, reproducible surprise: Wbt is
    NOT proportional to output-channel count -- it's identical for cout=8
    and cout=16 (same output-channel tile), confirming a real tiling
    granularity rather than a naive per-channel cost.
    """
    sizes = {
        cout: _build_and_get_blob_sizes_for_model(
            os.path.join(str(tmp_path), f"cout{cout}"),
            _single_conv_model(4, cout, 3),
            "x",
            (1, 4, 16, 16),
        )
        for cout in (8, 16)
    }
    wbt = {cout: s[0] for cout, s in sizes.items()}
    mcode = {cout: s[1] for cout, s in sizes.items()}

    assert wbt[8] == wbt[16], (
        wbt,
        "Wbt should be identical within one output-channel tile",
    )
    assert (mcode[16] - mcode[8]) % 32 == 0, (
        mcode,
        "mcode delta should be a multiple of 32",
    )


def _contiguous_diff_runs(a, b):
    assert len(a) == len(b)
    runs = []
    cur = None
    for i in range(len(a)):
        if a[i] != b[i]:
            if cur and i == cur[-1] + 1:
                cur.append(i)
            else:
                if cur:
                    runs.append(cur)
                cur = [i]
        elif cur:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return runs


def test_axquantizedconv_command_has_a_real_periodic_4x_field(tmp_path):
    """Confirmed real (see the README's "A first real crack at
    AxQuantizedConv's command encoding" section): picking a dilation pair
    whose padding is adjusted to keep output shape -- and thus mcode's
    total serialized length -- identical avoids the wholesale
    re-serialization a shape change triggers, letting a real byte-level
    diff isolate dilation's own encoding. The diff is small and
    structured, not a full rewrite, and contains a real, precisely-located
    periodic field: exactly 4 repeats of a 3-byte value, each 7 bytes
    apart. A control experiment (not repeated here for hardware-time cost,
    see the README) with double the output channels still shows exactly 4
    repeats -- ruling out "one entry per output channel" -- consistent
    with this compiler's real 4-way spatial tiling (independently visible
    in this project's own trace.json profiling as `_s0`../`_s3` sub-events).
    """
    a = _build_and_get_mcode_bytes(
        os.path.join(str(tmp_path), "dilation2"),
        _dilation_conv_model(2, 2),
        (1, 4, 16, 16),
    )
    b = _build_and_get_mcode_bytes(
        os.path.join(str(tmp_path), "dilation3"),
        _dilation_conv_model(3, 3),
        (1, 4, 16, 16),
    )
    assert len(a) == len(b), (
        "this dilation/padding pair should serialize to the same length"
    )

    runs = _contiguous_diff_runs(a, b)
    three_byte_runs = [r for r in runs if len(r) == 3]
    strides = [
        three_byte_runs[i + 1][0] - three_byte_runs[i][0]
        for i in range(len(three_byte_runs) - 1)
    ]

    assert len(three_byte_runs) == 4, (
        len(three_byte_runs),
        "expected exactly 4 repeats",
    )
    assert all(s == 7 for s in strides), (strides, "expected a constant 7-byte stride")


def test_downstream_conv_dilation_perturbs_upstream_conv_bytes(tmp_path):
    """Confirmed real (see the README's "Chaining two real convs" section):
    a genuinely new, previously-unknown cross-op coupling. Two chained
    `Conv`s at the one shape (cin=cout=mid=4) confirmed to give
    same-length pairs; varying only the *second* conv's dilation, with the
    *first* conv's own attributes completely untouched, still perturbs
    bytes within the first ~800 bytes of mcode -- the same relative region
    the single-op experiments already showed holds the first conv's own
    per-op command template. An op's encoding is not independent of what
    happens downstream of it, even at fixed total mcode length.
    """
    a = _build_and_get_mcode_bytes(
        os.path.join(str(tmp_path), "vary2_d2"),
        _two_conv_model(vary_first=False, dilation=2, pad=2),
        (1, 4, 16, 16),
    )
    b = _build_and_get_mcode_bytes(
        os.path.join(str(tmp_path), "vary2_d3"),
        _two_conv_model(vary_first=False, dilation=3, pad=3),
        (1, 4, 16, 16),
    )
    assert len(a) == len(b), (
        "this dilation/padding pair should serialize to the same length"
    )

    UPSTREAM_REGION = 800
    upstream_diffs = sum(1 for i in range(UPSTREAM_REGION) if a[i] != b[i])
    assert upstream_diffs > 0, (
        "expected the unchanged first Conv's own bytes to still be perturbed "
        "by a downstream-only dilation change"
    )


def test_wbt_is_deterministic_mcode_has_small_bounded_nondeterminism(tmp_path):
    """Confirmed real (see the README's "Is .axmodel deterministic?"
    section): rebuilding the *identical* model/config is not fully
    reproducible. Wbt (npu_params) is byte-identical across rebuilds every
    time tested; mcode is not, but the non-determinism is small (a
    handful of bytes) and bounded (same total length every time), not
    pervasive. This underpins every other differential-analysis test in
    this file -- a same-length pair with more than a token handful of
    byte differences is real signal, not noise, but this test exists to
    catch it if that ever stops being true (e.g. a toolchain regression
    that makes mcode non-determinism much larger or Wbt non-deterministic
    at all).
    """
    model = _dilation_conv_model(2, 2)
    wbt_a, mcode_a = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "run1"), model, (1, 4, 16, 16)
    )
    wbt_b, mcode_b = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "run2"), model, (1, 4, 16, 16)
    )

    assert wbt_a == wbt_b, "Wbt should be byte-identical across identical rebuilds"

    assert len(mcode_a) == len(mcode_b), (
        "mcode should serialize to the same length across identical rebuilds"
    )
    mcode_diff_count = sum(1 for i in range(len(mcode_a)) if mcode_a[i] != mcode_b[i])
    assert mcode_diff_count < 50, (
        mcode_diff_count,
        "expected only a small, bounded amount of run-to-run mcode noise",
    )


def test_mcode_nondeterminism_is_a_label_permutation_not_metadata(tmp_path):
    """Confirmed real (see the README's "Where does the non-determinism
    actually come from" section): the noisy positions found by rebuilding
    an identical two-Conv model always carry the *same multiset* of
    values across independent rebuilds -- only which position gets which
    value changes. That's the signature of a small set of interchangeable
    labels (plausibly per-tile job/resource IDs) being assigned to
    equivalent slots in a non-deterministic order (e.g. unordered-
    container iteration order), not embedded metadata like a timestamp or
    build ID -- real metadata could never coincidentally reproduce the
    exact same value set across independent builds run at different
    times, only a fixed label set being reshuffled could.
    """
    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    _, mcode_a = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "run1"), model, (1, 4, 16, 16)
    )
    _, mcode_b = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "run2"), model, (1, 4, 16, 16)
    )
    assert len(mcode_a) == len(mcode_b)

    noisy = [i for i in range(len(mcode_a)) if mcode_a[i] != mcode_b[i]]
    assert noisy, "expected the known small amount of run-to-run mcode noise"

    multiset_a = sorted(mcode_a[i] for i in noisy)
    multiset_b = sorted(mcode_b[i] for i in noisy)
    assert multiset_a == multiset_b, (
        (multiset_a, multiset_b),
        "expected the same multiset of values at the noisy positions, just reordered",
    )


def _build_axmodel(work_dir, model, input_shape):
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.randn(*input_shape).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)
    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error
    return result.axmodel_path


def test_mcode_nondeterminism_does_not_change_real_device_output(tmp_path):
    """Confirmed real (see the README's "Following up on determinism"
    section): the mcode byte-level non-determinism above is functionally
    harmless. Three independent rebuilds of the identical two-Conv model
    (each with different mcode bytes, due to the confirmed label-
    permutation noise) all produce bit-identical output on the real
    AX650N for the same input -- whichever arbitrary label a slot gets
    internally, the hardware executes the same computation.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    paths = [
        _build_axmodel(os.path.join(str(tmp_path), f"run{i}"), model, (1, 4, 16, 16))
        for i in range(3)
    ]

    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 16, 16).astype(np.float32)

    outputs = []
    for path in paths:
        dev = pulsar2_docker.run_on_device_with_inputs(path, {"x": x.tobytes()})
        assert not dev.error, dev.error
        outputs.append(np.frombuffer(dev.outputs[0], dtype=np.float32))

    for out in outputs[1:]:
        assert np.array_equal(outputs[0], out), "expected bit-identical device output"


def test_kernel_orientation_changes_mcode_size_despite_equal_weight_count(tmp_path):
    """Confirmed real (see the README's "Two more Conv attributes tried"
    section): a `3x1` and a `1x3` kernel hold the exact same number of
    weight values (`cin*cout*3*1 == cin*cout*1*3`), so Wbt comes out the
    same size either way -- but mcode does not. A real, new asymmetry:
    the compiler encodes a "tall" and a "wide" kernel of identical size
    differently, plausibly due to a real difference in how it scans/tiles
    the input by row vs. column.
    """
    wbt_h, mcode_h = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "k3x1"), _asym_kernel_model(3, 1), (1, 4, 16, 16)
    )
    wbt_w, mcode_w = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "k1x3"), _asym_kernel_model(1, 3), (1, 4, 16, 16)
    )
    assert len(wbt_h) == len(wbt_w), "same weight count should give the same Wbt size"
    assert len(mcode_h) != len(mcode_w), (
        "expected kernel orientation to produce a real mcode size difference"
    )


def test_autopad_normalizes_with_no_signal_beyond_known_noise(tmp_path):
    """Confirmed real (see the README's "Two more Conv attributes tried"
    section): `auto_pad="SAME_UPPER"` vs. the numerically-equivalent
    explicit `pads` first looked like a real signal (a same-length pair
    with a 4-byte diff at a location not seen before), but rebuilding the
    identical `auto_pad=NOTSET` config alone, twice, reproduced a nearly
    identical diff -- confirming it was this project's second encounter
    with non-deterministic label noise, not a real auto_pad-specific
    encoding. auto_pad appears to fully normalize away before
    quantization. This test locks in the *correct*, determinism-checked
    conclusion: the auto_pad-vs-explicit diff should be no bigger than
    what an identical rebuild alone already produces.
    """
    same_model = _autopad_model(use_auto_pad=True)
    explicit_model = _autopad_model(use_auto_pad=False)

    _, mcode_same = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "same"), same_model, (1, 4, 16, 16)
    )
    _, mcode_explicit_a = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "explicit_a"), explicit_model, (1, 4, 16, 16)
    )
    _, mcode_explicit_b = _build_and_get_wbt_and_mcode_bytes(
        os.path.join(str(tmp_path), "explicit_b"), explicit_model, (1, 4, 16, 16)
    )
    assert len(mcode_same) == len(mcode_explicit_a) == len(mcode_explicit_b)

    cross_diff = sum(
        1 for i in range(len(mcode_same)) if mcode_same[i] != mcode_explicit_a[i]
    )
    noise_diff = sum(
        1
        for i in range(len(mcode_explicit_a))
        if mcode_explicit_a[i] != mcode_explicit_b[i]
    )
    assert cross_diff <= noise_diff + 2, (
        (cross_diff, noise_diff),
        "auto_pad-vs-explicit diff should not exceed identical-rebuild noise "
        "by more than a token amount",
    )
