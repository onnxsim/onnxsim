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
import re
import struct
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, parser

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
    # 4 rebuilds, not 2: with only 2, the non-deterministic label
    # assignment occasionally lands identically by chance, giving zero
    # differing positions and nothing to test (a real flake hit in CI --
    # the same 2-build sample-size pitfall the frankenstein-splice test
    # below fixed). Noisy positions are taken as the union against build 0.
    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    mcodes = [
        _build_and_get_wbt_and_mcode_bytes(
            os.path.join(str(tmp_path), f"run{i}"), model, (1, 4, 16, 16)
        )[1]
        for i in range(4)
    ]
    assert len({len(m) for m in mcodes}) == 1

    noisy = [i for i in range(len(mcodes[0])) if len({m[i] for m in mcodes}) > 1]
    assert noisy, "expected the known small amount of run-to-run mcode noise"

    multiset_0 = sorted(mcodes[0][i] for i in noisy)
    for n, other in enumerate(mcodes[1:], start=1):
        multiset_n = sorted(other[i] for i in noisy)
        assert multiset_0 == multiset_n, (
            (n, multiset_0, multiset_n),
            "expected the same multiset of values at the noisy positions, "
            "just reordered",
        )


def _build_axmodel(work_dir, model, input_shape, profile=False):
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
        work_dir, "model.onnx", "output", config_path="config/cfg.json", profile=profile
    )
    assert result.success, result.error
    if profile:
        return result.axmodel_path, result.trace_path
    return result.axmodel_path


def _mcode_from_axmodel(axmodel_path):
    compiled = onnx.load(axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    mcode_key = info["dotneus"][0]["neu_key"]
    return bytes(inits[mcode_key].raw_data)


def _mcode_key(compiled):
    """The `neu_key` initializer name holding a compiled model's mcode --
    shared by tests that hand-patch mcode bytes and need to resave."""
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    return info["dotneus"][0]["neu_key"]


def _params_key(compiled):
    """The `npu_params` initializer name holding a compiled model's Wbt --
    the hand-patching counterpart to `_mcode_key`."""
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    return info["dotneus"][0]["extra_inputs"][0]["const_data_key"]


def _conv_with_bias_model(bias_vals, cin=4, cout=4, k=3, insz=16):
    """One `Conv(x, w, b)` with an explicit, distinctly non-uniform bias --
    used for hand-patching Wbt's per-channel requantization scale, where a
    uniform bias would make every channel's own decoded value coincide and
    hide per-channel indexing bugs."""
    pad = k // 2
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": 17]> '
        f"agraph (float[1,{cin},{insz},{insz}] x) => (float[1,{cout},{insz},{insz}] y) "
        f"{{ y = Conv<pads=[{pad},{pad},{pad},{pad}]>(x, w, b) }}"
    )
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin, k, k) * 0.1).astype(np.float32)
    b = np.array(bias_vals, dtype=np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, name="w"))
    model.graph.initializer.append(numpy_helper.from_array(b, name="b"))
    onnx.checker.check_model(model)
    return model


def _build_real_resnet18d(work_dir):
    """Fetch and really build `resnet18d_Opset18` the same way
    `convert_onnxmodelzoo.py` does (single-image-classifier config,
    synthetic calibration tar). Returns `(axmodel_path, mcode_key,
    mcode_bytes)`; shared by the real-resnet18d hand-patching tests."""
    import convert_onnxmodelzoo  # also puts model_zoo on sys.path
    import model_zoo

    model = onnx.load(model_zoo.fetch_model("resnet18d_Opset18"))
    tensor_name = convert_onnxmodelzoo._single_image_input(model)
    assert tensor_name is not None

    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    pulsar2_docker.make_synthetic_calibration_tar(
        os.path.join(work_dir, "dataset", "calib.tar")
    )
    onnx.save(model, os.path.join(work_dir, "model", "resnet18d.onnx"))
    result = pulsar2_docker.build(
        work_dir,
        "model/resnet18d.onnx",
        "output/resnet18d",
        tensor_name=tensor_name,
        mean=convert_onnxmodelzoo._DEFAULT_MEAN,
        std=convert_onnxmodelzoo._DEFAULT_STD,
        calibration_dataset_rel_path="dataset/calib.tar",
    )
    assert result.success, result.error
    compiled = onnx.load(result.axmodel_path)
    key = _mcode_key(compiled)
    mcode = bytes({i.name: i for i in compiled.graph.initializer}[key].raw_data)
    assert len(mcode) == 49080, len(mcode)
    return result.axmodel_path, key, mcode


def _run_retry_once(axmodel_path, x):
    """`run_on_device_with_inputs`, retried once on a `0x8030070C` fault.

    Confirmed real (see the README's "the fault is transient after a
    burst" note): immediately after a run of deliberately-faulting
    models, the runtime can transiently reject a *valid* model with the
    same `0x8030070C` -- it self-clears on the next attempt. A retry
    cleanly separates the two: a genuinely faulting byte faults on every
    attempt (verified many times), so a second fault is a real one, while
    a transient recovers. Never retries any other error."""
    dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, {"x": x.tobytes()})
    if dev.error and "0x8030070C" in dev.error:
        dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, {"x": x.tobytes()})
    return dev


def _gemm_model(transb, m=1, k=16, n=8):
    """One `Gemm(x, w, b)` -- `transb` selects whether `w` is stored as
    `[k,n]` (transB=0) or `[n,k]` (transB=1), same logical matrix either
    way."""
    rng = np.random.RandomState(0)
    w_shape = (n, k) if transb else (k, n)
    w = (rng.randn(*w_shape) * 0.1).astype(np.float32)
    b = (rng.randn(n) * 0.1).astype(np.float32)
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], transB=transb)
    graph = helper.make_graph(
        [node],
        f"g_gemm_transb{transb}",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [m, n])],
        initializer=[
            numpy_helper.from_array(w, name="w"),
            numpy_helper.from_array(b, name="b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _grouped_conv_model(groups, cin=4, cout=4, k=3, insz=16):
    """One `Conv` with the `group` attribute set -- `groups=1` is an
    ordinary dense conv, `groups>1` splits input/output channels into
    independent groups (`groups=cin=cout` is a full depthwise conv). Text
    form used per this repo's model-building convention -- see
    scripts/axera/README.md's "`Conv`'s `group` attribute" section."""
    pad = k // 2
    model = parser.parse_model(
        f'<ir_version: 10, opset_import: ["": 17]> '
        f"agraph (float[1,{cin},{insz},{insz}] x) => (float[1,{cout},{insz},{insz}] y) "
        f"{{ y = Conv<pads=[{pad},{pad},{pad},{pad}], group={groups}>(x, w) }}"
    )
    rng = np.random.RandomState(0)
    w = (rng.randn(cout, cin // groups, k, k) * 0.1).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, name="w"))
    onnx.checker.check_model(model)
    return model


def _maxpool_model(k, stride, pad, ceil_mode=0, cin=4, insz=16):
    import math

    if ceil_mode:
        out = math.ceil((insz + 2 * pad - k) / stride) + 1
    else:
        out = math.floor((insz + 2 * pad - k) / stride) + 1
    node = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        kernel_shape=[k, k],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        ceil_mode=ceil_mode,
    )
    graph = helper.make_graph(
        [node],
        f"g_maxpool_k{k}_s{stride}_p{pad}_ceil{ceil_mode}",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, cin, insz, insz]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [1, cin, out, out]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


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


def test_maxpool_ceil_mode_is_a_third_confirmed_false_lead(tmp_path):
    """Confirmed real (see the README's "Expanding past Conv" section):
    MaxPool's `ceil_mode=0` vs `ceil_mode=1`, on a shape where both give
    the identical output size, produces a same-length pair with a few
    differing bytes -- but rebuilding `ceil_mode=0` alone, twice,
    reproduces the same diff. This is the same non-deterministic label
    noise confirmed on Conv/auto_pad, now shown on a completely different
    op type: mcode's non-determinism is a global property, not tied to
    any one op or model.
    """
    model_off = _maxpool_model(2, 2, 0, ceil_mode=0)
    model_on = _maxpool_model(2, 2, 0, ceil_mode=1)

    mcode_off_a = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "off_a"), model_off, (1, 4, 16, 16))
    )
    mcode_off_b = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "off_b"), model_off, (1, 4, 16, 16))
    )
    mcode_on = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "on"), model_on, (1, 4, 16, 16))
    )
    assert len(mcode_off_a) == len(mcode_off_b) == len(mcode_on)

    noise_diff = sum(
        1 for i in range(len(mcode_off_a)) if mcode_off_a[i] != mcode_off_b[i]
    )
    cross_diff = sum(
        1 for i in range(len(mcode_off_a)) if mcode_off_a[i] != mcode_on[i]
    )
    assert cross_diff <= noise_diff + 2, (
        (cross_diff, noise_diff),
        "ceil_mode-vs-off diff should not exceed identical-rebuild noise "
        "by more than a token amount",
    )


def test_non_mac_ops_schedule_on_teng2_not_conv_engines(tmp_path):
    """Confirmed real via --profile (see the README's "Expanding past
    Conv" section): AxMaxPool, the real residual AxQuantizedAdd, and
    AxQuantizedGlobAvgPool all schedule on the `teng2` engine, never on
    `conv0`/`conv1` (reserved for AxQuantizedConv/Gemm MAC work) --
    extending the same finding already confirmed for AxQuantizedNormalize
    in this project's resnet18d profiling to three more real primitive
    families.
    """
    model = _maxpool_model(2, 2, 0)
    _, trace_path = _build_axmodel(
        os.path.join(str(tmp_path), "maxpool_profiled"),
        model,
        (1, 4, 16, 16),
        profile=True,
    )
    trace = json.load(open(trace_path))
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    maxpool_events = [
        e for e in events if e.get("ph") == "X" and "AxMaxPool" in e.get("name", "")
    ]
    assert maxpool_events, "expected at least one AxMaxPool event in the trace"
    for e in maxpool_events:
        assert e["tid"] == "teng2", (e, "expected AxMaxPool to schedule on teng2")


def test_gemm_schedules_on_conv_engine_like_conv_does(tmp_path):
    """Confirmed real via --profile (see the README's "Gemm joins the MAC
    engines" section): Gemm schedules on `conv1`, joining Conv in the
    MAC-engine category rather than teng2's non-MAC group.
    """
    model = _gemm_model(transb=0)
    _, trace_path = _build_axmodel(
        os.path.join(str(tmp_path), "gemm_profiled"), model, (1, 16), profile=True
    )
    trace = json.load(open(trace_path))
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    gemm_events = [
        e
        for e in events
        if e.get("ph") == "X" and re.fullmatch(r"y_\d+_\d+", e.get("name", ""))
    ]
    assert gemm_events, "expected at least one Gemm output event in the trace"
    for e in gemm_events:
        assert e["tid"] in ("conv0", "conv1"), (
            e,
            "expected Gemm to schedule on a conv engine",
        )


def test_gemm_transb_produces_a_real_substantial_signal_beyond_noise(tmp_path):
    """Confirmed real (see the README's "Gemm joins the MAC engines"
    section): this Gemm shape has the same small, known non-deterministic
    noise as Conv/MaxPool (confirmed here by rebuilding `transb=0` twice --
    an *initial* single rebuild pair happened to show zero diffs, which
    turned out to be a lucky draw, not a real "this shape is noise-free"
    property; corrected after a second rebuild pair showed the familiar
    ~6-byte noise). Even accounting for that, transB produces a real,
    substantial signal far beyond noise scale: dozens of differing bytes,
    including a large contiguous block, at a completely different scale
    than the small periodic fields found for Conv's attributes.
    """
    model_a = _gemm_model(transb=0)
    model_b = _gemm_model(transb=1)

    mcode_a1 = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "a1"), model_a, (1, 16))
    )
    mcode_a2 = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "a2"), model_a, (1, 16))
    )
    mcode_b = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "b"), model_b, (1, 16))
    )
    assert len(mcode_a1) == len(mcode_a2) == len(mcode_b)

    noise_diff = sum(1 for i in range(len(mcode_a1)) if mcode_a1[i] != mcode_a2[i])
    transb_diff = sum(1 for i in range(len(mcode_a1)) if mcode_a1[i] != mcode_b[i])
    assert transb_diff > noise_diff + 50, (
        (transb_diff, noise_diff),
        "expected a real, substantial transB signal well beyond noise scale",
    )


def test_grouped_conv_splits_across_two_mac_engines_dense_does_not(tmp_path):
    """Confirmed real via --profile (see the README's "Conv's group
    attribute" section): a dense Conv (group=1) schedules its 3 sub-events
    on a single MAC engine (conv1), while a grouped Conv -- both a partial
    grouping (group=2) and a full depthwise one (group=4, cin=cout=4) --
    schedules 6 sub-events split evenly across *both* conv0 and conv1.
    This is a binary split on "is this Conv grouped at all," confirmed
    stable across independent rebuilds of each config: group=2 and group=4
    produce the identical 6-event, both-engines pattern rather than a
    count that scales with the number of groups.
    """

    def conv_engines(groups):
        _, trace_path = _build_axmodel(
            os.path.join(str(tmp_path), f"groups{groups}"),
            _grouped_conv_model(groups=groups),
            (1, 4, 16, 16),
            profile=True,
        )
        trace = json.load(open(trace_path))
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        conv_events = [
            e
            for e in events
            if e.get("ph") == "X" and "AxQuantizedConv" in e.get("name", "")
        ]
        return len(conv_events), sorted({e.get("tid") for e in conv_events})

    assert conv_engines(groups=1) == (3, ["conv1"])
    for groups in (2, 4):
        assert conv_engines(groups=groups) == (6, ["conv0", "conv1"]), groups


def test_dense_conv_two_engine_split_is_a_channel_count_threshold(tmp_path):
    """Confirmed real via --profile (see the README's "Does the two-engine
    split transfer to resnet18d itself?" section): the single-vs-two-engine
    split above is not really about grouping -- a dense (group=1) Conv
    crosses into the two-engine regime once its channel count passes a
    real, sharp threshold. Confirmed stable across independent rebuilds at
    both ends: cin=cout<=4 stays on a single engine (conv1); cin=cout>=5
    splits across both conv0 and conv1. Every real resnet18d layer (64 to
    512 channels) sits far on the two-engine side of this threshold,
    explaining why a real profiled resnet18d build shows all 15 of its
    distinct Conv ops on both engines despite having no grouped convs at
    all.
    """

    def conv_engines(channels):
        _, trace_path = _build_axmodel(
            os.path.join(str(tmp_path), f"ch{channels}"),
            _grouped_conv_model(groups=1, cin=channels, cout=channels),
            (1, channels, 16, 16),
            profile=True,
        )
        trace = json.load(open(trace_path))
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        conv_events = [
            e
            for e in events
            if e.get("ph") == "X" and "AxQuantizedConv" in e.get("name", "")
        ]
        return len(conv_events), sorted({e.get("tid") for e in conv_events})

    assert conv_engines(channels=4) == (3, ["conv1"])
    assert conv_engines(channels=5) == (6, ["conv0", "conv1"])


def test_gemm_two_engine_split_threshold_differs_from_conv(tmp_path):
    """Confirmed real via --profile (see the README's "Gemm has the same
    two-regime split, but at a much higher, distinct threshold" section):
    Gemm(k=n=128) (16,384 weight elements) stays on a single engine while
    Gemm(k=n=256) (65,536 elements) splits across both conv0 and conv1 --
    confirmed stable across independent rebuilds at both ends. This is a
    real threshold, but at a much larger, roughly-square shape than
    Conv's tiny 4-vs-5-channel cutover -- neither op's threshold reduces
    to the other's formula.
    """

    def gemm_engines(k, n):
        _, trace_path = _build_axmodel(
            os.path.join(str(tmp_path), f"k{k}_n{n}"),
            _gemm_model(transb=0, k=k, n=n),
            (1, k),
            profile=True,
        )
        trace = json.load(open(trace_path))
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        gemm_events = [
            e
            for e in events
            if e.get("ph") == "X" and re.fullmatch(r"y_\d+_\d+", e.get("name", ""))
        ]
        return len(gemm_events), sorted({e.get("tid") for e in gemm_events})

    assert gemm_engines(k=128, n=128) == (3, ["conv1"])
    assert gemm_engines(k=256, n=256) == (6, ["conv0", "conv1"])


def test_spliced_frankenstein_noise_bytes_run_correctly_on_device(tmp_path):
    """Confirmed real (see the README's "Beyond passive diffing" section):
    building the identical two-Conv model several times gives real mcode
    blobs differing only at the known small set of non-deterministic noise
    positions. Splicing build 0's mcode with differing bytes cycled in
    from the other builds produces a byte sequence that is *not* identical
    to any single real build (a genuinely novel combination the real
    compiler never produced as a whole) -- yet it loads and runs on the
    real AX650N with bit-identical output to the unpatched original. This
    is proof by construction, not correlation, that the noise zone is a
    truly swappable, functionally inert label.

    Uses 5 rebuilds and mixes noisy positions across all of them (not just
    one other build) specifically to avoid a real edge case hit during
    development: with too few rebuilds, sometimes only a single position
    actually varies, and splicing just that one position reproduces
    another real build's mcode byte-for-byte rather than a novel one.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    paths = [
        _build_axmodel(os.path.join(str(tmp_path), f"run{i}"), model, (1, 4, 16, 16))
        for i in range(5)
    ]
    compiled = [onnx.load(p) for p in paths]
    keys = [_mcode_key(c) for c in compiled]
    assert len(set(keys)) == 1, "expected the same neu_key across identical rebuilds"
    key = keys[0]
    mcodes = [
        bytes({i.name: i for i in c.graph.initializer}[key].raw_data) for c in compiled
    ]
    assert len(set(len(m) for m in mcodes)) == 1

    noisy = [i for i in range(len(mcodes[0])) if len(set(m[i] for m in mcodes)) > 1]
    assert noisy, "expected the known small amount of run-to-run mcode noise"

    frank = bytearray(mcodes[0])
    for n, pos in enumerate(noisy):
        frank[pos] = mcodes[1 + (n % (len(mcodes) - 1))][pos]
    frank = bytes(frank)
    assert frank not in mcodes, "expected a genuinely novel combination"

    inits = {i.name: i for i in compiled[0].graph.initializer}
    inits[key].raw_data = frank
    frank_path = os.path.join(str(tmp_path), "frankenstein.axmodel")
    onnx.save(compiled[0], frank_path)

    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 16, 16).astype(np.float32)
    dev_orig = pulsar2_docker.run_on_device_with_inputs(paths[0], {"x": x.tobytes()})
    dev_frank = pulsar2_docker.run_on_device_with_inputs(frank_path, {"x": x.tobytes()})
    assert not dev_orig.error, dev_orig.error
    assert not dev_frank.error, dev_frank.error
    out_orig = np.frombuffer(dev_orig.outputs[0], dtype=np.float32)
    out_frank = np.frombuffer(dev_frank.outputs[0], dtype=np.float32)
    assert np.array_equal(out_orig, out_frank)


def test_bit_flip_in_opaque_mcode_region_is_sometimes_inert_sometimes_faults(tmp_path):
    """Confirmed real (see the README's "Beyond passive diffing" section):
    flipping all 8 bits of a single byte, at offsets in the still-opaque
    majority of mcode (not the header/footer, not the known noise zone),
    splits cleanly into two real, reproducible outcomes. Offset 400 is
    completely inert (bit-identical device output). Offset 700 reliably
    faults the runtime with the same real error every time -- confirming
    part of that opaque region is genuinely load-bearing structural data
    (plausibly a checksum/opcode/address-range check), without decoding
    a single new bit.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    path = _build_axmodel(os.path.join(str(tmp_path), "base"), model, (1, 4, 16, 16))
    compiled = onnx.load(path)
    key = _mcode_key(compiled)
    mcode = bytes({i.name: i for i in compiled.graph.initializer}[key].raw_data)

    def flip_and_run(offset, x):
        patched = bytearray(mcode)
        patched[offset] ^= 0xFF
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"flip_{offset}.axmodel")
        onnx.save(c, p)
        return _run_retry_once(p, x)

    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 16, 16).astype(np.float32)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    dev_inert = flip_and_run(400, x)
    assert not dev_inert.error, dev_inert.error
    out_inert = np.frombuffer(dev_inert.outputs[0], dtype=np.float32)
    assert np.array_equal(out_base, out_inert)

    dev_fault = flip_and_run(700, x)
    assert dev_fault.error and "0x8030070C" in dev_fault.error, dev_fault.error


def test_patching_decoded_requant_scale_isolates_to_one_channel(tmp_path):
    """Confirmed real (see the README's "Hand-patching the decoded Wbt
    requantization scale" section): halving `Conv`'s per-channel
    requantization scale `M_channel` (Wbt's small ~1e-3-magnitude
    per-channel float32 array, present in 4 identical repeated copies) at
    only channel 0's 4 copies produces bit-identical device output for
    channels 1-3, but visibly reshapes channel 0's own output (narrower
    spread, fewer unique values -- the signature of compressing the int8
    requantization range around a fixed zero-point, not simply halving
    the final float result). Confirmed stable across independent
    rebuilds. This test locks in the causal isolation (other channels
    untouched) and the qualitative reshaping (not a naive linear scale),
    without depending on brittle exact-offset assumptions beyond this
    specific, confirmed model configuration.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    model = _conv_with_bias_model([0, 5, 10, 15])
    path = _build_axmodel(os.path.join(str(tmp_path), "base"), model, (1, 4, 16, 16))
    compiled = onnx.load(path)
    key = _params_key(compiled)
    wbt = bytes({i.name: i for i in compiled.graph.initializer}[key].raw_data)

    # channel 0's requant-scale value, confirmed present at these 4
    # identical repeated offsets for this exact model configuration.
    repeat_offsets = [1216, 1232, 1248, 1264]
    values = [struct.unpack("<f", wbt[o : o + 4])[0] for o in repeat_offsets]
    assert len(set(values)) == 1, "expected the 4 repeated copies to agree"
    assert 1e-4 < values[0] < 1e-2, (values, "expected a small requant-scale value")

    patched = bytearray(wbt)
    for off in repeat_offsets:
        v = struct.unpack("<f", patched[off : off + 4])[0]
        patched[off : off + 4] = struct.pack("<f", v * 0.5)
    compiled_patched = onnx.load(path)
    {i.name: i for i in compiled_patched.graph.initializer}[key].raw_data = bytes(
        patched
    )
    patched_path = os.path.join(str(tmp_path), "patched.axmodel")
    onnx.save(compiled_patched, patched_path)

    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 16, 16).astype(np.float32)
    dev_base = _run_retry_once(path, x)
    dev_patch = pulsar2_docker.run_on_device_with_inputs(
        patched_path, {"x": x.tobytes()}
    )
    assert not dev_base.error, dev_base.error
    assert not dev_patch.error, dev_patch.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32).reshape(
        1, 4, 16, 16
    )
    out_patch = np.frombuffer(dev_patch.outputs[0], dtype=np.float32).reshape(
        1, 4, 16, 16
    )

    for c in (1, 2, 3):
        assert np.array_equal(out_base[0, c], out_patch[0, c]), (
            c,
            "expected untouched channels to be bit-identical",
        )

    base0, patch0 = out_base[0, 0], out_patch[0, 0]
    assert not np.array_equal(base0, patch0)
    assert patch0.std() < base0.std(), (
        base0.std(),
        patch0.std(),
        "expected halving the requant scale to narrow channel 0's output spread",
    )
    assert len(np.unique(patch0)) < len(np.unique(base0))


def test_bit_flip_probe_on_real_resnet18d_has_three_outcome_classes(tmp_path):
    """Confirmed real (see the README's "The bit-flip probe on the real
    resnet18d mcode" section): on the real, unmodified resnet18d_Opset18
    mcode (49,080 bytes), flipping a single byte lands in one of THREE
    deterministic classes -- not the two the tiny two-Conv model showed:

    - FAULT: the runtime rejects it with 0x8030070C (structural check).
    - DIFFERENT: it runs, but the real 1000-class logits change -- the
      first bytes this project found whose effect on actual computation
      is directly observable, most of them changing the predicted class.
    - identical: it runs, bit-identical output (inert).

    A 41-offset sweep measured 61% / 22% / 17%. This locks in two
    representative offsets per class. Heavier than this file's other
    tests (fetches the real model, one real Docker build) -- the same
    class of work as the dormant self-hosted `pulsar2-docker-convert` CI
    job, which is where it is meant to run.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    work_dir = str(tmp_path)

    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)
    assert out_base.shape == (1000,)

    def flip_and_run(offset):
        patched = bytearray(mcode)
        patched[offset] ^= 0xFF
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(work_dir, f"flip_{offset}.axmodel")
        onnx.save(c, p)
        return _run_retry_once(p, x)

    for off in (300, 1500):
        dev = flip_and_run(off)
        assert dev.error and "0x8030070C" in dev.error, (off, dev.error)

    for off in (12300, 26700):
        dev = flip_and_run(off)
        assert not dev.error, (off, dev.error)
        assert np.array_equal(
            np.frombuffer(dev.outputs[0], dtype=np.float32), out_base
        ), off

    for off in (9900, 32700):
        dev = flip_and_run(off)
        assert not dev.error, (off, dev.error)
        out = np.frombuffer(dev.outputs[0], dtype=np.float32)
        assert not np.array_equal(out, out_base), off
        assert out.argmax() != out_base.argmax(), (
            off,
            "expected the predicted class to change",
        )


def test_resnet18d_live_byte_neighborhoods_share_a_template_signature(tmp_path):
    """Confirmed real (see the README's "Bisecting the live bytes'
    neighbors" section): flipping each byte in a 17-byte window around
    two of the real resnet18d mcode's output-changing offsets, 15,600
    bytes apart, gives the byte-for-byte identical fault/inert/different
    signature `FF==D=FDDDDDF=FF=` -- the repeated-command-template
    structure seen from the hardware's side, with the same internal field
    layout. Within it, two distinct bytes (X-4 and X-1) are functionally
    interchangeable: corrupting either yields the bit-identical full
    1000-logit output. And every one of the 8 bits of byte 9900 is live
    (all run, all change the output).
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))

    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    def flip_and_run(offset, mask):
        patched = bytearray(mcode)
        patched[offset] ^= mask
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"flip_{offset}_{mask:02x}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        if dev.error:
            assert "0x8030070C" in dev.error, (offset, mask, dev.error)
            return "F", None
        out = np.frombuffer(dev.outputs[0], dtype=np.float32)
        return ("=" if np.array_equal(out, out_base) else "D"), out

    outputs = {}
    signatures = {}
    for center in (32700, 48300):
        sig = ""
        for off in range(center - 8, center + 9):
            cls, out = flip_and_run(off, 0xFF)
            sig += cls
            outputs[off] = out
        signatures[center] = sig

    assert signatures[32700] == signatures[48300] == "FF==D=FDDDDDF=FF=", signatures

    for center in (32700, 48300):
        a, b = outputs[center - 4], outputs[center - 1]
        assert a is not None and b is not None
        assert not np.array_equal(a, out_base)
        assert np.array_equal(a, b), (
            center,
            "expected X-4 and X-1 to be interchangeable",
        )

    for bit in range(8):
        cls, _ = flip_and_run(9900, 1 << bit)
        assert cls == "D", (bit, cls, "expected every bit of byte 9900 to be live")


def test_resnet18d_template_has_a_gate_a_dead_nibble_and_a_checked_msb(tmp_path):
    """Confirmed real (see the README's "Inside one repeated template"
    section): in the `FF==D=FDDDDDF=FF=` template of the real resnet18d
    mcode, bytes X-4 and X-1 are a *gate*, not a value -- flipping X-4,
    X-1, or both at once lands in the byte-identical output (a double
    flip neither cancels nor compounds). Byte X-1 is half dead, half
    gate: each low-nibble bit trips that same state, every high-nibble
    bit is inert. And bit 7 of X+3 is the one single-bit flip that
    faults the runtime's validator, while its other bits are live.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))

    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    def run(edits, tag):
        patched = bytearray(mcode)
        for off, mask in edits:
            patched[off] ^= mask
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"edit_{tag}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        if dev.error:
            assert "0x8030070C" in dev.error, (tag, dev.error)
            return None
        return np.frombuffer(dev.outputs[0], dtype=np.float32)

    gate_state = {}
    for X in (32700, 48300):
        a = run([(X - 4, 0xFF)], f"{X}_a")
        b = run([(X - 1, 0xFF)], f"{X}_b")
        ab = run([(X - 4, 0xFF), (X - 1, 0xFF)], f"{X}_ab")
        assert a is not None and b is not None and ab is not None
        assert not np.array_equal(a, out_base), X
        assert np.array_equal(a, b) and np.array_equal(a, ab), (
            X,
            "expected X-4, X-1, and both together to land in one gate state",
        )
        gate_state[X] = a

    # X-1 at 32700: low nibble is the gate, high nibble is dead.
    for bit in range(4):
        out = run([(32699, 1 << bit)], f"32699_b{bit}")
        assert out is not None and np.array_equal(out, gate_state[32700]), bit
    for bit in range(4, 8):
        out = run([(32699, 1 << bit)], f"32699_b{bit}")
        assert out is not None and np.array_equal(out, out_base), (
            bit,
            "expected the high nibble of X-1 to be inert",
        )

    # The MSB of X+3 is the one single-bit flip that faults.
    assert run([(32703, 0x80)], "32703_b7") is None
    assert run([(32703, 0x01)], "32703_b0") is not None


def test_resnet18d_output_changing_bytes_are_mostly_input_independent(tmp_path):
    """Confirmed real (see the README's "What the output-changing bytes
    are" section): running the same flipped real-resnet18d model against
    different inputs discriminates control-like bytes from data-like
    ones. Offsets 9900 and 48300 are control-like: the flip lands the
    classifier on the same wrong class (567 and 834) with a near-identical
    delta regardless of input. Offset 43500 is data-like: the shifted
    class moves with the input and the delta stays small. 8 of the 9
    output-changing offsets behaved like the former.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    inputs = {
        seed: np.random.RandomState(seed).randint(
            0, 256, size=(1, 224, 224, 3), dtype=np.uint8
        )
        for seed in (42, 7)
    }
    base = {}
    for seed, x in inputs.items():
        dev = _run_retry_once(path, x)
        assert not dev.error, (seed, dev.error)
        base[seed] = np.frombuffer(dev.outputs[0], dtype=np.float32)

    def flipped_argmax(offset, seed):
        patched = bytearray(mcode)
        patched[offset] ^= 0xFF
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"flip_{offset}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, inputs[seed])
        assert not dev.error, (offset, seed, dev.error)
        out = np.frombuffer(dev.outputs[0], dtype=np.float32)
        assert not np.array_equal(out, base[seed]), (offset, seed)
        return int(out.argmax())

    for offset, wrong_class in ((9900, 567), (48300, 834)):
        assert flipped_argmax(offset, 42) == flipped_argmax(offset, 7) == wrong_class, (
            offset,
            "expected a control-like byte to shift to the same class on every input",
        )

    assert flipped_argmax(43500, 42) != flipped_argmax(43500, 7), (
        "expected the data-like byte's shifted class to move with the input"
    )


def test_resnet18d_data_like_byte_sits_in_a_gated_template_and_is_sign_like(tmp_path):
    """Confirmed real (see the README's "The one data-like byte, probed"
    section): the data-like byte 43500 sits in a third template that
    shares the recurring gate structure -- flipping X-4 (43496) and X-1
    (43499) yields the bit-identical output. Its own per-bit behavior is
    only partly bit-weighted (bit 7 moves the class, bits 0-3 do not),
    and it shows the signature of a small *signed* value: flipping all 8
    bits changes the output less than flipping bit 7 alone. Every bit is
    live.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    x = np.random.RandomState(42).randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    def flip(offset, mask):
        patched = bytearray(mcode)
        patched[offset] ^= mask
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"flip_{offset}_{mask:02x}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        assert not dev.error, (offset, mask, dev.error)
        return np.frombuffer(dev.outputs[0], dtype=np.float32)

    # The gate: X-4 and X-1 are interchangeable, and neither is the baseline.
    gate_a, gate_b = flip(43496, 0xFF), flip(43499, 0xFF)
    assert not np.array_equal(gate_a, out_base)
    assert np.array_equal(gate_a, gate_b), "expected X-4 and X-1 to trip one state"

    # Every bit of 43500 is live; bit 7 moves the class, bit 0 does not.
    outs = {bit: flip(43500, 1 << bit) for bit in range(8)}
    for bit, out in outs.items():
        assert not np.array_equal(out, out_base), (bit, "expected every bit to be live")
    assert outs[7].argmax() != out_base.argmax()
    assert outs[0].argmax() == out_base.argmax()

    # Sign-like: flipping all 8 bits perturbs less than flipping bit 7 alone.
    all_bits = flip(43500, 0xFF)
    assert np.max(np.abs(all_bits - out_base)) < np.max(np.abs(outs[7] - out_base))


def _word_stream_stats(mcode):
    """Local, deterministic structure stats for an mcode blob read as a
    stream of 4-byte words (see the README's "32-bit-word instruction
    stream" section): the word indices holding an `a1 00 xx yy` header,
    how many headers are immediately followed by another header, the
    fraction of even gaps between consecutive headers, and a Counter of
    the `xx yy` header suffixes. No Docker or device needed beyond
    producing the blob."""
    from collections import Counter

    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    headers = [i for i, w in enumerate(words) if w[0] == 0xA1 and w[1] == 0x00]
    adjacent = sum(
        1
        for i in headers
        if i + 1 < len(words) and words[i + 1][0] == 0xA1 and words[i + 1][1] == 0x00
    )
    gaps = [b - a for a, b in zip(headers, headers[1:])]
    even_frac = sum(1 for g in gaps if g % 2 == 0) / len(gaps) if gaps else 0.0
    suffixes = Counter(words[i][2:].hex(" ") for i in headers)
    return headers, adjacent, even_frac, suffixes


def test_mcode_is_a_word_stream_with_never_adjacent_headers_tiny_model(tmp_path):
    """Confirmed real (see the README's "32-bit-word instruction stream"
    section), on the tiny two-Conv model: the mcode blob is 4-byte
    aligned, holds `a1 00 xx yy` header words at word alignment, no
    header is ever immediately followed by another (the [header][operand]
    structure), and the gaps between headers are dominantly even. Needs
    Docker for the build but no device.
    """
    model = _two_conv_model(vary_first=True, dilation=2, pad=2)
    mcode = _mcode_from_axmodel(
        _build_axmodel(os.path.join(str(tmp_path), "tiny"), model, (1, 4, 16, 16))
    )
    assert len(mcode) % 4 == 0
    headers, adjacent, even_frac, _ = _word_stream_stats(mcode)
    assert len(headers) >= 40, len(headers)
    assert adjacent == 0, adjacent
    assert even_frac >= 0.8, even_frac


def test_resnet18d_mcode_word_stream_counts(tmp_path):
    """Confirmed real (see the README's "32-bit-word instruction stream"
    section), on the real resnet18d blob: exactly 1,197 word-aligned
    `a1 00` headers, none adjacent to another, 98% even gaps, and the two
    probed instruction kinds `40 02` (control-like) and `50 03`
    (data-like) occurring exactly as often as each other. Needs Docker
    for the build but no device.
    """
    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    assert len(mcode) % 4 == 0
    headers, adjacent, even_frac, suffixes = _word_stream_stats(mcode)
    assert len(headers) == 1197, len(headers)
    assert adjacent == 0, adjacent
    assert even_frac >= 0.95, even_frac
    assert suffixes["40 02"] == suffixes["50 03"] == 181, (
        suffixes["40 02"],
        suffixes["50 03"],
    )
    assert suffixes.most_common(1)[0] == ("50 01", 724), suffixes.most_common(3)


def test_resnet18d_40_02_operands_are_wbt_offsets_paired_with_50_03(tmp_path):
    """Confirmed real (see the README's "`40 02` operands are weight-table
    offsets" section), all local byte analysis: `40 02` and `50 03`
    headers are paired one-to-one, canonically 8 words (one 32-byte unit)
    apart; the 181 `40 02` operands are all distinct and span exactly the
    Wbt (the largest lands within 0.3% of `npu_params`'s size); `50 03`
    operands stop at about a quarter of that; and the most common kind,
    `50 01`, takes only four distinct operand values across 724 uses.
    Needs Docker for the build but no device.
    """
    from collections import Counter

    path, _, mcode = _build_real_resnet18d(str(tmp_path))
    compiled = onnx.load(path)
    wbt_size = len(
        bytes(
            {i.name: i for i in compiled.graph.initializer}[
                _params_key(compiled)
            ].raw_data
        )
    )

    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    kind = {i: w[2:].hex(" ") for i, w in enumerate(words) if w[:2] == b"\xa1\x00"}
    by_kind = {}
    for i, k in kind.items():
        by_kind.setdefault(k, []).append(i)
    operand = lambda i: int.from_bytes(words[i + 1], "little")  # noqa: E731

    i4002, i5003 = by_kind["40 02"], by_kind["50 03"]
    assert len(i4002) == len(i5003) == 181
    dist = Counter(
        j - max(i for i in i4002 if i < j) for j in i5003 if any(i < j for i in i4002)
    )
    assert dist.most_common(1)[0][0] == 8 and dist[8] >= 160, dist.most_common(3)

    ops4002 = [operand(i) for i in i4002]
    assert len(set(ops4002)) == 181
    assert max(ops4002) <= wbt_size
    assert max(ops4002) >= 0.99 * wbt_size, (max(ops4002), wbt_size)

    ops5003 = [operand(i) for i in i5003]
    assert max(ops5003) < 0.35 * wbt_size, (max(ops5003), wbt_size)

    assert len({operand(i) for i in by_kind["50 01"]}) <= 4
