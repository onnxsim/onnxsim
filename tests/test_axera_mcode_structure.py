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


def test_resnet18d_40_02_operand_is_live_for_any_value_with_no_bounds_check(tmp_path):
    """Confirmed real (see the README's "Patching a `40 02` operand on the
    device" section): overwriting instance 32700's whole 4-byte operand
    with another instance's valid offset, with 0, or with a value 1 MB
    past the Wbt's end all run without fault and all change the output
    -- the operand is live and causal for any value, and the runtime
    does not bounds-check it (the 0x8030070C validator guards header
    words, never operand values). The past-end read is deterministic.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    compiled = onnx.load(path)
    wbt_size = len(
        bytes(
            {i.name: i for i in compiled.graph.initializer}[
                _params_key(compiled)
            ].raw_data
        )
    )
    x = np.random.RandomState(42).randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    a, b = 32700, 48300  # the two probed `40 02` instances' operand words
    op_b = int.from_bytes(mcode[b : b + 4], "little")
    assert op_b <= wbt_size

    def run_with_operand(value, tag):
        patched = bytearray(mcode)
        patched[a : a + 4] = struct.pack("<I", value)
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"op_{tag}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        assert not dev.error, (tag, value, dev.error)
        out = np.frombuffer(dev.outputs[0], dtype=np.float32)
        assert not np.array_equal(out, out_base), (tag, value)
        return out

    run_with_operand(op_b, "foreign")
    run_with_operand(0, "zero")
    past = wbt_size + 0x100000
    first, second = run_with_operand(past, "past1"), run_with_operand(past, "past2")
    assert np.array_equal(first, second), (
        "expected the past-end read to be deterministic"
    )


def _header_fields(mcode):
    """Every `a1 00 xx yy` header in an mcode blob read as 4-byte words,
    as `(xx, yy, following 32-bit LE value)` -- see the README's
    "`a1 00 xx yy` is a field write" section."""
    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    return [
        (w[2], w[3], int.from_bytes(words[i + 1], "little"))
        for i, w in enumerate(words)
        if w[:2] == b"\xa1\x00" and i + 1 < len(words)
    ]


def test_mcode_headers_are_field_writes_with_a_shared_map_across_models(tmp_path):
    """Confirmed real (see the README's "`a1 00 xx yy` is a field write"
    section), on both the tiny two-Conv blob and the real resnet18d
    blob: `xx` is a multiple of 0x10 in every single header (a 16-byte
    granular field offset, not an opcode); bank 0x02's `xx` ladder is
    shared by both models; and the `50 01` field takes the identical
    one-hot operand set {bit 0, 8, 20, 24} in both. Needs Docker for the
    builds but no device.
    """
    tiny = _mcode_from_axmodel(
        _build_axmodel(
            os.path.join(str(tmp_path), "tiny"),
            _two_conv_model(vary_first=True, dilation=2, pad=2),
            (1, 4, 16, 16),
        )
    )
    _, _, r18 = _build_real_resnet18d(str(tmp_path))
    one_hot = {0x1, 0x100, 0x100000, 0x1000000}
    shared_bank2_ladder = {0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0}

    for name, mcode in (("tiny", tiny), ("resnet18d", r18)):
        fields = _header_fields(mcode)
        assert len(fields) >= 40, (name, len(fields))
        assert all(xx % 0x10 == 0 for xx, _, _ in fields), name
        assert {v for xx, yy, v in fields if (xx, yy) == (0x50, 0x01)} == one_hot, name
        assert shared_bank2_ladder <= {xx for xx, yy, _ in fields if yy == 0x02}, name

    assert (
        sum(1 for xx, yy, _ in _header_fields(r18) if (xx, yy) == (0x40, 0x02)) == 181
    )


def test_resnet18d_three_per_op_fields_are_91_percent_of_all_field_writes(tmp_path):
    """Confirmed real (see the README's "Typing the field map" section),
    all local: resnet18d writes 68 distinct fields, but 57 of them fewer
    than three times (one-time setup), and just three per-op fields --
    `50 01` (flag), `40 02` (Wbt offset), `50 03` (the ~3.1 MB address
    region) -- account for 1,086 of the 1,197 writes, 91%. Needs Docker
    for the build but no device.
    """
    from collections import Counter

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    fields = _header_fields(mcode)
    writes = Counter((xx, yy) for xx, yy, _ in fields)
    assert len(writes) >= 60, len(writes)
    assert sum(1 for n in writes.values() if n < 3) >= 50, writes.most_common(12)

    hot = writes[(0x50, 0x01)] + writes[(0x40, 0x02)] + writes[(0x50, 0x03)]
    assert hot >= 0.9 * len(fields), (hot, len(fields))
    assert len({v for xx, yy, v in fields if (xx, yy) == (0x50, 0x03)}) >= 170


def test_resnet18d_50_03_operands_are_a_64_byte_aligned_tile_arena(tmp_path):
    """Confirmed real (see the README's "`50 03` is not whole activation
    tensors" section), all local: the 176 distinct `50 03` operands are
    64-byte aligned (175 of 176 exact multiples of 64), span ~3.1 MB, and
    their consecutive deltas are tile-sized, not activation-tensor-sized
    -- the largest resnet18d activation is 802,816 bytes and the span is
    neither that nor the sum of all intermediates. Needs Docker for the
    build but no device.
    """
    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    ops = sorted({v for xx, yy, v in _header_fields(mcode) if (xx, yy) == (0x50, 0x03)})
    assert len(ops) >= 170, len(ops)
    assert sum(1 for o in ops if o % 64 == 0) >= len(ops) - 1, (
        "expected 64-byte alignment"
    )
    assert 3_000_000 < max(ops) < 3_300_000, max(ops)
    deltas = [b - a for a, b in zip(ops, ops[1:])]
    assert max(deltas) < 802_816, "expected tile-sized deltas, not tensor-sized ones"
    assert min(o for o in ops if o) == 37_120


def _flag_sequence_per_op(mcode):
    """For each pair of consecutive `40 02` (Wbt-offset) writes, the
    ordered tuple of `50 01` values written between them -- see the
    README's "`50 01` is a per-op four-step sequence" section."""
    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    hdrs = [
        (i, w[2], w[3], int.from_bytes(words[i + 1], "little"))
        for i, w in enumerate(words)
        if w[:2] == b"\xa1\x00" and i + 1 < len(words)
    ]
    cuts = [i for i, xx, yy, _ in hdrs if (xx, yy) == (0x40, 0x02)]
    flags = [(i, v) for i, xx, yy, v in hdrs if (xx, yy) == (0x50, 0x01)]
    return [tuple(v for i, v in flags if a < i < b) for a, b in zip(cuts, cuts[1:])]


def test_50_01_is_a_fixed_four_step_sequence_per_op_in_both_models(tmp_path):
    """Confirmed real (see the README's "`50 01` is a per-op four-step
    sequence" section), all local: every op writes `50 01` exactly four
    times in the fixed order bit8, bit0, bit20, bit24 -- 180 of 180
    inter-op segments in resnet18d, and the tiny two-Conv model's single
    op writes the same four in the same order. Not an engine selector.
    Needs Docker for the builds but no device.
    """
    order = (0x100, 0x1, 0x100000, 0x1000000)

    tiny = _mcode_from_axmodel(
        _build_axmodel(
            os.path.join(str(tmp_path), "tiny"),
            _two_conv_model(vary_first=True, dilation=2, pad=2),
            (1, 4, 16, 16),
        )
    )
    tiny_flags = [v for xx, yy, v in _header_fields(tiny) if (xx, yy) == (0x50, 0x01)]
    assert tuple(tiny_flags) == order, tiny_flags

    _, _, r18 = _build_real_resnet18d(str(tmp_path))
    segments = _flag_sequence_per_op(r18)
    assert len(segments) == 180, len(segments)
    assert all(seg == order for seg in segments), (
        sum(1 for seg in segments if seg != order),
        "expected every op to write the same four-step sequence",
    )


def test_resnet18d_50_01_steps_three_required_one_optional_on_device(tmp_path):
    """Confirmed real on the device (see the README's "The four `50 01`
    steps on the device" section), on two independent ops: zeroing the
    `bit8` step faults (`0x8030070C`), zeroing the `bit24` step leaves
    the output bit-identical, swapping `bit8` and `bit0` leaves it
    bit-identical, and zeroing all four runs with a changed output. Also
    the concrete counter-example to "the validator never faults on an
    operand value": these are operand values, and one of them faults.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    hdrs = [
        (i, w[2], w[3], int.from_bytes(words[i + 1], "little"))
        for i, w in enumerate(words)
        if w[:2] == b"\xa1\x00" and i + 1 < len(words)
    ]
    cuts = [i for i, xx, yy, _ in hdrs if (xx, yy) == (0x40, 0x02)]
    op_start = 8174  # the op whose 40 02 write sits at byte 32696
    assert op_start in cuts
    op_end = min(i for i in cuts if i > op_start)
    steps = [
        (i, v)
        for i, xx, yy, v in hdrs
        if (xx, yy) == (0x50, 0x01) and op_start < i < op_end
    ]
    assert [v for _, v in steps] == [0x100, 0x1, 0x100000, 0x1000000], steps

    x = np.random.RandomState(42).randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    def run_variant(edits, tag):
        patched = bytearray(mcode)
        for word_index, value in edits:
            patched[(word_index + 1) * 4 : (word_index + 2) * 4] = struct.pack(
                "<I", value
            )
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"steps_{tag}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        if dev.error:
            assert "0x8030070C" in dev.error, (tag, dev.error)
            return None
        return np.frombuffer(dev.outputs[0], dtype=np.float32)

    (w8, v8), (w0, v0), _, (w24, _) = steps
    assert run_variant([(w8, 0)], "skip_bit8") is None, (
        "expected skipping bit8 to fault"
    )
    out = run_variant([(w24, 0)], "skip_bit24")
    assert out is not None and np.array_equal(out, out_base), (
        "expected bit24 to be optional"
    )
    out = run_variant([(w8, v0), (w0, v8)], "swap_bit8_bit0")
    assert out is not None and np.array_equal(out, out_base), (
        "expected bit8/bit0 order-free"
    )
    out = run_variant([(w, 0) for w, _ in steps], "all_zero")
    assert out is not None and not np.array_equal(out, out_base), (
        "expected an absent step set to run with a changed result, not fault"
    )


def test_resnet18d_50_01_step_set_dispatches_the_op_and_bit24_is_inert_model_wide(
    tmp_path,
):
    """Confirmed real on the device (see the README's "The `50 01` step set
    is the op's dispatch" section): with all four `50 01` steps of one op
    zeroed, the model runs with a changed result, and additionally
    pointing that op's Wbt offset at another instance's weights gives the
    byte-identical changed result -- the op no longer executes at all.
    And zeroing `bit24` in all 181 ops at once leaves the whole output
    bit-identical to the unpatched baseline.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    words = [mcode[i : i + 4] for i in range(0, len(mcode) - 3, 4)]
    hdrs = [
        (i, w[2], w[3], int.from_bytes(words[i + 1], "little"))
        for i, w in enumerate(words)
        if w[:2] == b"\xa1\x00" and i + 1 < len(words)
    ]
    cuts = [i for i, xx, yy, _ in hdrs if (xx, yy) == (0x40, 0x02)]
    op_start = 8174
    assert op_start in cuts
    op_end = min(i for i in cuts if i > op_start)
    steps = [
        i for i, xx, yy, _ in hdrs if (xx, yy) == (0x50, 0x01) and op_start < i < op_end
    ]
    assert len(steps) == 4
    other_offset = int.from_bytes(
        mcode[48300:48304], "little"
    )  # a different op's 40 02 value
    bit24_writes = [
        i for i, xx, yy, v in hdrs if (xx, yy) == (0x50, 0x01) and v == 0x1000000
    ]
    assert len(bit24_writes) == 181

    x = np.random.RandomState(42).randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error
    out_base = np.frombuffer(dev_base.outputs[0], dtype=np.float32)

    def run_variant(edits, tag):
        patched = bytearray(mcode)
        for word_index, value in edits:
            patched[(word_index + 1) * 4 : (word_index + 2) * 4] = struct.pack(
                "<I", value
            )
        c = onnx.load(path)
        {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
        p = os.path.join(str(tmp_path), f"dispatch_{tag}.axmodel")
        onnx.save(c, p)
        dev = _run_retry_once(p, x)
        assert not dev.error, (tag, dev.error)
        return np.frombuffer(dev.outputs[0], dtype=np.float32)

    no_steps = run_variant([(w, 0) for w in steps], "no_steps")
    assert not np.array_equal(no_steps, out_base)
    no_steps_other_weights = run_variant(
        [(w, 0) for w in steps] + [(op_start, other_offset)], "no_steps_other_weights"
    )
    assert np.array_equal(no_steps, no_steps_other_weights), (
        "expected an op with no step set to be skipped, so its weight offset is never read"
    )

    all_bit24_zero = run_variant([(w, 0) for w in bit24_writes], "all_bit24_zero")
    assert np.array_equal(all_bit24_zero, out_base), (
        "expected bit24 to be inert for correctness across every op"
    )


_VERBS = {0xA1, 0xA2, 0xA3, 0xA8, 0xA9}


def _verb_headers(mcode, start=328):
    """Every verb header word in the phase-0 bulk of an mcode blob -- see
    the README's "Five verbs, a readable per-op program" section. Returns
    `(word_index, verb, xx, yy, operand)` for each `XX 00 <mult of 0x10>
    yy` word whose first byte is a known verb, indexing words from
    `start` (the preamble before it has variable-length slots)."""
    words = [mcode[i : i + 4] for i in range(start, len(mcode) - 3, 4)]
    return [
        (k, w[0], w[2], w[3], int.from_bytes(words[k + 1], "little"))
        for k, w in enumerate(words)
        if w[0] in _VERBS and w[1] == 0 and w[2] % 0x10 == 0 and k + 1 < len(words)
    ]


def test_resnet18d_bulk_is_a_two_word_stream_of_five_verbs_with_a_per_op_program(
    tmp_path,
):
    """Confirmed real (see the README's "Five verbs, a readable per-op
    program" section), all local: with all five verbs counted, 97.5% of
    consecutive headers in the bulk are exactly two words apart; `a9 00
    00` and both `a8` selectors occur exactly once per op (181); and 64
    of 180 ops are exactly one eleven-instruction template. Also locks in
    the header-declared arena size 0x2ff000 bounding every `50 03`
    address. Needs Docker for the build but no device.
    """
    from collections import Counter

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    hdrs = _verb_headers(mcode)
    assert len(hdrs) >= 2200, len(hdrs)

    gaps = Counter(b[0] - a[0] for a, b in zip(hdrs, hdrs[1:]))
    assert gaps[2] / sum(gaps.values()) >= 0.95, gaps.most_common(5)

    by_verb_sel = Counter((v, xx, yy) for _, v, xx, yy, _ in hdrs)
    assert by_verb_sel[(0xA9, 0x00, 0x00)] == 181
    assert by_verb_sel[(0xA8, 0x30, 0x02)] == by_verb_sel[(0xA8, 0x40, 0x03)] == 181
    assert by_verb_sel[(0xA2, 0x00, 0x00)] >= 350
    assert by_verb_sel[(0xA3, 0x00, 0x00)] >= 180

    cuts = [k for k, v, xx, yy, _ in hdrs if (v, xx, yy) == (0xA1, 0x40, 0x02)]
    template = "a1:5001 a1:5001 a8:4003 a1:5003 a1:5001 a3:0000 a1:5001 a9:0000 a2:0000 a8:3002"
    patterns = Counter(
        " ".join(f"{v:02x}:{xx:02x}{yy:02x}" for k, v, xx, yy, _ in hdrs if a < k < b)
        for a, b in zip(cuts, cuts[1:])
    )
    assert patterns[template] >= 60, patterns.most_common(2)

    arena = int.from_bytes(mcode[76:80], "little")
    assert arena == 0x2FF000, hex(arena)
    assert int.from_bytes(mcode[72:76], "little") == 4096
    max_50_03 = max(
        op for _, v, xx, yy, op in hdrs if (v, xx, yy) == (0xA1, 0x50, 0x03)
    )
    assert max_50_03 < arena and arena - max_50_03 < 40_000, (
        hex(max_50_03),
        hex(arena),
    )


def _build_real_zoo_model(work_dir, name):
    """Fetch and really build any single-image-input onnxmodelzoo model
    the same way `convert_onnxmodelzoo.py` does. Returns `(axmodel_path,
    mcode_key, mcode_bytes, wbt_bytes)`. The resnet18d-specific
    `_build_real_resnet18d` above predates this and keeps its exact-size
    assertion; new multi-model tests should use this one."""
    import convert_onnxmodelzoo  # also puts model_zoo on sys.path
    import model_zoo

    model = onnx.load(model_zoo.fetch_model(name))
    tensor_name = convert_onnxmodelzoo._single_image_input(model)
    assert tensor_name is not None, name

    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    pulsar2_docker.make_synthetic_calibration_tar(
        os.path.join(work_dir, "dataset", "calib.tar")
    )
    onnx.save(model, os.path.join(work_dir, "model", f"{name}.onnx"))
    result = pulsar2_docker.build(
        work_dir,
        f"model/{name}.onnx",
        f"output/{name}",
        tensor_name=tensor_name,
        mean=convert_onnxmodelzoo._DEFAULT_MEAN,
        std=convert_onnxmodelzoo._DEFAULT_STD,
        calibration_dataset_rel_path="dataset/calib.tar",
    )
    assert result.success, (name, result.error)
    compiled = onnx.load(result.axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    key = _mcode_key(compiled)
    return (
        result.axmodel_path,
        key,
        bytes(inits[key].raw_data),
        bytes(inits[_params_key(compiled)].raw_data),
    )


def test_mcode_field_map_generalizes_to_mnasnet_small(tmp_path):
    """Confirmed real (see the README's "The whole map generalizes to a
    second real architecture" section), all local on a real
    `mnasnet_small_Opset17` build: the same header constants (4096,
    0x2ff000 arena), the same five verbs, one `40 02`/`50 03`/`a9`/`a8`
    write each per op with the op count simply changing (133), four
    `50 01` writes per op in the same one-hot set and fixed order, `40 02`
    spanning its (very different) Wbt, and the dominant per-op template.
    Needs Docker for the build but no device.
    """
    from collections import Counter

    _, _, mcode, wbt = _build_real_zoo_model(str(tmp_path), "mnasnet_small_Opset17")
    assert int.from_bytes(mcode[72:76], "little") == 4096
    assert int.from_bytes(mcode[76:80], "little") == 0x2FF000, (
        "arena size is a platform constant"
    )

    hdrs = _verb_headers(mcode)
    by = Counter((v, xx, yy) for _, v, xx, yy, _ in hdrs)
    n_ops = by[(0xA1, 0x40, 0x02)]
    assert 100 <= n_ops <= 160, n_ops
    for sel in (
        (0xA1, 0x50, 0x03),
        (0xA9, 0x00, 0x00),
        (0xA8, 0x30, 0x02),
        (0xA8, 0x40, 0x03),
    ):
        assert by[sel] == n_ops, (sel, by[sel], n_ops)
    assert by[(0xA1, 0x50, 0x01)] == 4 * n_ops
    assert {op for _, v, xx, yy, op in hdrs if (v, xx, yy) == (0xA1, 0x50, 0x01)} == {
        0x1,
        0x100,
        0x100000,
        0x1000000,
    }

    ops4002 = [op for _, v, xx, yy, op in hdrs if (v, xx, yy) == (0xA1, 0x40, 0x02)]
    assert len(set(ops4002)) == n_ops
    assert 0.95 * len(wbt) <= max(ops4002) <= len(wbt), (max(ops4002), len(wbt))
    max_50_03 = max(
        op for _, v, xx, yy, op in hdrs if (v, xx, yy) == (0xA1, 0x50, 0x03)
    )
    assert max_50_03 < 0x2FF000 and 0x2FF000 - max_50_03 < 40_000, hex(max_50_03)

    cuts = [k for k, v, xx, yy, _ in hdrs if (v, xx, yy) == (0xA1, 0x40, 0x02)]
    flags = [(k, op) for k, v, xx, yy, op in hdrs if (v, xx, yy) == (0xA1, 0x50, 0x01)]
    order = (0x100, 0x1, 0x100000, 0x1000000)
    assert all(
        tuple(op for k, op in flags if a < k < b) == order
        for a, b in zip(cuts, cuts[1:])
    ), "expected every op to write the same four-step sequence"
    template = "a1:5001 a1:5001 a8:4003 a1:5003 a1:5001 a3:0000 a1:5001 a9:0000 a2:0000 a8:3002"
    patterns = Counter(
        " ".join(f"{v:02x}:{xx:02x}{yy:02x}" for k, v, xx, yy, _ in hdrs if a < k < b)
        for a, b in zip(cuts, cuts[1:])
    )
    assert patterns.most_common(1)[0][0] == template, patterns.most_common(2)


def _verb_free_regions(mcode, min_words=8):
    """Gaps between consecutive verb headers larger than a [header][operand]
    pair, as `(extra_words)` per gap -- the verb-free regions of the README's
    "Correction: the instruction runs are two-word, but they are only 37% of
    the bulk" section."""
    hdrs = _verb_headers(mcode)
    return [b[0] - a[0] - 2 for a, b in zip(hdrs, hdrs[1:]) if b[0] - a[0] >= min_words]


def test_verb_runs_are_a_minority_of_the_bulk_and_the_prologue_is_shared(tmp_path):
    """Confirmed real (see the README's "Correction: the instruction runs
    are two-word, but they are only 37% of the bulk" section), all local
    on real resnet18d and mnasnet_small builds: the five-verb pairs cover
    ~37% / ~19% of the bulk words; the rest sits in 44 / 127 verb-free
    regions; both prologues carry the compact `00 xx 84 vv` ladder (a
    short-form write with no verb byte); and the two real classifiers
    share a byte-identical prologue of 150+ bytes from the first field
    write. Needs Docker for the builds but no device.
    """
    _, _, r18 = _build_real_resnet18d(str(tmp_path))
    _, _, mnas, _ = _build_real_zoo_model(str(tmp_path), "mnasnet_small_Opset17")

    for name, mcode, cover_lo, cover_hi, min_regions in (
        ("resnet18d", r18, 0.30, 0.45, 40),
        ("mnasnet", mnas, 0.15, 0.25, 120),
    ):
        n_words = (len(mcode) - 328) // 4
        coverage = 2 * len(_verb_headers(mcode)) / n_words
        assert cover_lo <= coverage <= cover_hi, (name, coverage)
        assert len(_verb_free_regions(mcode)) >= min_regions, (
            name,
            len(_verb_free_regions(mcode)),
        )
        seg = mcode[297 : 297 + 120]
        ladder = [
            seg[i + 1]
            for i in range(len(seg) - 3)
            if seg[i] == 0 and seg[i + 2] == 0x84 and seg[i + 1] % 0x10 == 0
        ]
        assert ladder == [0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0], (name, ladder)

    shared = 0
    while (
        297 + shared < min(len(r18), len(mnas))
        and r18[297 + shared] == mnas[297 + shared]
    ):
        shared += 1
    assert shared >= 150, shared


def _walk_variable_length(mcode, start=297, end=None):
    """Greedy variable-length walk over an mcode blob's bulk with the three
    known instruction forms (8-byte verb, 7-byte verb when the next header
    lands at +7, 4-byte compact `00 xx 84 vv` write), stepping one unknown
    byte otherwise -- see the README's "Second correction: the verb-free
    regions are the same instruction stream, drifted off the 4-byte grid"
    section. Returns `(explained_bytes, total_bytes, Counter of forms)`."""
    from collections import Counter

    end = len(mcode) - 252 if end is None else end

    def is_verb(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] in _VERBS
            and mcode[i + 1] == 0
            and mcode[i + 2] % 0x10 == 0
        )

    def is_compact(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] == 0
            and mcode[i + 2] == 0x84
            and mcode[i + 1] % 0x10 == 0
        )

    forms, explained, i = Counter(), 0, start
    while i < end:
        if is_verb(i):
            n = (
                7
                if (
                    not (is_verb(i + 8) or is_compact(i + 8))
                    and (is_verb(i + 7) or is_compact(i + 7))
                )
                else 8
            )
            forms[f"verb{n}"] += 1
        elif is_compact(i):
            n = 4
            forms["compact4"] += 1
        else:
            n = 1
            forms["unknown"] += 1
        explained += n if n > 1 else 0
        i += n
    return explained, end - start, forms


def test_verb_free_regions_are_drifted_instructions_and_a_walker_beats_the_grid(
    tmp_path,
):
    """Confirmed real (see the README's "Second correction" section), all
    local on real resnet18d and mnasnet_small builds: the "verb-free
    regions" hold verb-shaped words at byte phases 1-3 (>= 100 in
    resnet18d) and hundreds of compact `00 xx 84 vv` writes, and a
    variable-length walker knowing only three forms explains more of the
    bulk than the fixed 4-byte grid on both models, recovering 7-byte and
    compact instructions the grid cannot see. Needs Docker for the builds
    but no device.
    """
    _, _, r18 = _build_real_resnet18d(str(tmp_path))
    _, _, mnas, _ = _build_real_zoo_model(str(tmp_path), "mnasnet_small_Opset17")

    for name, mcode in (("resnet18d", r18), ("mnasnet", mnas)):
        explained, total, forms = _walk_variable_length(mcode)
        grid = 8 * len(_verb_headers(mcode)) / total
        assert explained / total > grid + 0.05, (name, explained / total, grid)
        assert forms["verb7"] > 0 and forms["compact4"] > 300, (name, dict(forms))
        assert explained / total < 0.6, (
            name,
            "walker should not claim near-complete coverage",
        )

    hdrs = _verb_headers(r18)
    off_phase = 0
    for a, b in zip(hdrs, hdrs[1:]):
        if b[0] - a[0] >= 8:
            lo, hi = 328 + 4 * (a[0] + 2), 328 + 4 * b[0]
            off_phase += sum(
                1
                for i in range(lo, hi - 3)
                if (i - 328) % 4 != 0
                and r18[i] in _VERBS
                and r18[i + 1] == 0
                and r18[i + 2] % 0x10 == 0
            )
    assert off_phase >= 100, off_phase


def test_resnet18d_verb_free_regions_fault_like_instructions_on_device(tmp_path):
    """Confirmed real on the device (see the README's "The regions fault
    like instructions on the device" section): flipping single bytes at
    the start, middle and end of the largest "verb-free regions" of the
    real resnet18d mcode mostly faults with 0x8030070C (24 of 30 in the
    ten largest; 10 of 12 in the four largest used here on one build, 7
    of 12 on a fresh rebuild -- per-flip outcomes vary between builds, the
    majority does not) -- the answer of a validated instruction stream,
    not of a data table, which would give wrong values or nothing.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    path, key, mcode = _build_real_resnet18d(str(tmp_path))
    hdrs = _verb_headers(mcode)
    regions = sorted(
        (
            (328 + 4 * (a[0] + 2), 4 * (b[0] - a[0] - 2))
            for a, b in zip(hdrs, hdrs[1:])
            if b[0] - a[0] >= 8
        ),
        key=lambda r: -r[1],
    )[:4]
    assert regions and regions[0][1] >= 2000, regions

    x = np.random.RandomState(42).randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    dev_base = _run_retry_once(path, x)
    assert not dev_base.error, dev_base.error

    faults = 0
    for start, length in regions:
        for off in (start + 8, start + length // 2, start + length - 8):
            patched = bytearray(mcode)
            patched[off] ^= 0xFF
            c = onnx.load(path)
            {i.name: i for i in c.graph.initializer}[key].raw_data = bytes(patched)
            p = os.path.join(str(tmp_path), f"region_{off}.axmodel")
            onnx.save(c, p)
            dev = _run_retry_once(p, x)
            if dev.error:
                assert "0x8030070C" in dev.error, (off, dev.error)
                faults += 1
    assert faults >= 7, (
        faults,
        "expected a majority of flips inside the regions to fault",
    )


def _permutation_ratio(non_verb_bytes, pattern, shuffles=10, seed=0):
    """Observed count of `pattern(b, i)` over the bytes vs. its mean count
    over byte-shuffled copies (same histogram, order destroyed) -- the
    chance baseline the README's "Third correction" section shows is the
    right one for positional patterns."""
    import random

    rng = random.Random(seed)

    def count(b):
        return sum(1 for i in range(len(b)) if pattern(b, i))

    observed = count(non_verb_bytes)
    nulls = []
    for _ in range(shuffles):
        s = bytearray(non_verb_bytes)
        rng.shuffle(s)
        nulls.append(count(bytes(s)))
    return observed / (sum(nulls) / len(nulls))


def test_short_bank_tagged_forms_are_far_above_a_permutation_null(tmp_path):
    """Confirmed real (see the README's "Third correction" section), all
    local: against a permutation null (the non-verb bytes shuffled, same
    histogram), the 4-byte `00 ?? TT ??` form, the prologue ladder
    `00 x0 84 ??`, and the prefix + tag == 0x84 rule are each far above
    chance on the real resnet18d blob -- unlike an independence estimate,
    which understated them. Deterministic (fixed seed). Needs Docker for
    the build but no device.
    """
    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    tags = set(range(0x81, 0x85))

    def is_verb(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] in _VERBS
            and mcode[i + 1] == 0
            and mcode[i + 2] % 0x10 == 0
        )

    non_verb = bytearray()
    i, end = 297, len(mcode) - 252
    while i < end:
        if is_verb(i):
            i += 8
        else:
            non_verb.append(mcode[i])
            i += 1
    non_verb = bytes(non_verb)
    assert len(non_verb) > 20_000

    w4 = _permutation_ratio(
        non_verb, lambda b, i: i + 3 < len(b) and b[i] == 0 and b[i + 2] in tags
    )
    ladder = _permutation_ratio(
        non_verb,
        lambda b, i: i + 3 < len(b)
        and b[i] == 0
        and b[i + 1] % 0x10 == 0
        and b[i + 2] == 0x84,
    )
    complement = _permutation_ratio(
        non_verb,
        lambda b, i: i + 3 < len(b)
        and b[i] <= 3
        and i + 2 + b[i] < len(b)
        and b[i + 2 + b[i]] == 0x84 - b[i],
    )
    assert w4 >= 2.5, w4
    assert ladder >= 5.0, ladder
    assert complement >= 4.0, complement

    # The width rule: for prefix p, the tag byte is enriched at offset
    # exactly p + 2 and nowhere else nearby -- the unit is p + 4 bytes long.
    for prefix in range(4):
        ratios = {
            k: _permutation_ratio(
                non_verb,
                lambda b, i, k=k, p=prefix: i + k < len(b)
                and b[i] == p
                and b[i + k] in tags,
                shuffles=5,
            )
            for k in range(1, 6)
        }
        peak = max(ratios, key=ratios.get)
        assert peak == prefix + 2, (prefix, ratios)
        assert ratios[peak] >= 2.0, (prefix, ratios)
        assert all(r <= 1.8 for k, r in ratios.items() if k != peak), (prefix, ratios)


_WIDE_TAGS = frozenset(
    {0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x89, 0x8A, 0x8B, 0x8C, 0x8D}
    | {0x94, 0x95, 0x9B, 0x9C, 0x9D, 0x9F}
)
"""The 17 short-unit tag bytes that beat a shuffled *explained-bytes* null
by >= 2x in both real models -- the README's "Fourth correction" section.
Superseded by `_ALL_TAGS`: the fifth correction's conditional-parity test
showed the rejected tags were false rejections of rare forms."""

_ALL_TAGS = frozenset(range(0x81, 0xA0))
"""Every byte below the verb range: the full short-unit tag set. For every
tag the byte after it is even in ~100% of real units against ~60% shuffled
-- see the README's "Fifth correction" section."""


def _tokenize_mcode(
    mcode,
    start=297,
    end=None,
    tags=None,
    pmax=3,
    bare=False,
    extra_byte_tags=frozenset(),
):
    """Tokenize an mcode blob's bulk with every validated form -- the 8/7-byte
    verb instructions, the width-rule short units `[p][p+1 bytes][tag]
    [register]` (p+4 bytes) and, with `bare=True`, the payload-less 2-byte
    `[tag][register]` pair -- stepping one unknown byte otherwise. Units
    whose tag is in `extra_byte_tags` take one extra trailing byte (the
    sixth correction: tag 0x9f). Returns `(byte_offset, kind, a, b, c)`
    tuples: kind 'V' (a=verb, b=xx, c=yy), 'S' (a=prefix, b=tag, c=first
    payload byte), 'B' (a=tag, b=register) or '?' (a=byte). The defaults
    (tags 0x81..0x84, p <= 3, no bare pairs, no extra bytes, stop 252 bytes
    before the end) are the original narrow rule; `tags=_ALL_TAGS, pmax=4,
    bare=True, extra_byte_tags={0x9F}` is the corrected one. See the
    README's "The layout that explains all of it" and "Fourth" .. "Sixth
    correction" sections."""
    tags = set(range(0x81, 0x85)) if tags is None else set(tags)
    extra_byte_tags = set(extra_byte_tags)
    end = len(mcode) - 252 if end is None else end

    def is_verb(i):
        return (
            i + 3 < len(mcode)
            and mcode[i] in _VERBS
            and mcode[i + 1] == 0
            and mcode[i + 2] % 0x10 == 0
        )

    def short_len(i):
        if i >= len(mcode):
            return 0
        p = mcode[i]
        if p <= pmax and i + p + 3 < len(mcode) and mcode[i + p + 2] in tags:
            return p + 4 + (1 if mcode[i + p + 2] in extra_byte_tags else 0)
        return 0

    out, i = [], start
    while i < end:
        if is_verb(i):
            n = (
                7
                if (
                    not (is_verb(i + 8) or short_len(i + 8))
                    and (is_verb(i + 7) or short_len(i + 7))
                )
                else 8
            )
            out.append((i, "V", mcode[i], mcode[i + 2], mcode[i + 3]))
        elif short_len(i):
            n = short_len(i)
            out.append((i, "S", mcode[i], mcode[i + n - 2], mcode[i + 1]))
        elif bare and i + 1 < end and mcode[i] in tags and mcode[i + 1] % 2 == 0:
            n = 2 + (1 if mcode[i] in extra_byte_tags else 0)
            out.append((i, "B", mcode[i], mcode[i + 1], 0))
        else:
            n = 1
            out.append((i, "?", mcode[i], 0, 0))
        i += n
    return out


def test_mcode_layout_is_two_config_blocks_then_clean_op_programs(tmp_path):
    """Confirmed real (see the README's "The layout that explains all of
    it" section), all local on real resnet18d and mnasnet_small builds:
    splitting the token stream at every `a1 40 02` write, the op-program
    region (the last 180 / 132 segments) contains no short unit and no
    unknown byte -- each op is exactly the verb template -- while at least
    90% of all short units and unknown bytes sit in the first two
    segments, the configuration blocks. Needs Docker for the builds but
    no device.
    """
    _, _, r18 = _build_real_resnet18d(str(tmp_path))
    _, _, mnas, _ = _build_real_zoo_model(str(tmp_path), "mnasnet_small_Opset17")

    for name, mcode, min_clean_ops in (("resnet18d", r18, 178), ("mnasnet", mnas, 130)):
        toks = _tokenize_mcode(mcode)
        cuts = [k for k, t in enumerate(toks) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
        assert len(cuts) >= min_clean_ops + 2, (name, len(cuts))
        segments = [toks[a:b] for a, b in zip(cuts, cuts[1:])]

        def noise(seg):
            return sum(1 for t in seg if t[1] in "S?")

        clean = sum(1 for seg in segments[2:] if noise(seg) == 0)
        assert clean >= min_clean_ops, (name, clean, len(segments))

        total_noise = sum(noise(seg) for seg in segments) + noise(toks[cuts[-1] :])
        front_noise = noise(segments[0]) + noise(segments[1])
        assert front_noise >= 0.9 * total_noise, (name, front_noise, total_noise)
        assert toks[cuts[2]][0] > 20_000, (
            name,
            "expected the op region to start after the config blocks",
        )


def test_resnet18d_config_block_b_residue_is_structured_and_headed(tmp_path):
    """Confirmed real (see the README's "Into config block B" section),
    all local on a real resnet18d build: within the larger configuration
    block, the width rule extends to prefix 4 (tag at offset 6, >= 2x a
    shuffled null); the undecoded residue left after removing every
    validated instruction has a best internal period that beats its own
    shuffle by >= 2x (real local structure, not fixed records); and its
    most common 2-byte pair is an `XX 00` header-like pair from the
    candidate second family. Deterministic (fixed seed). Needs Docker for
    the build but no device.
    """
    import random
    from collections import Counter

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    toks = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(toks) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    lo, hi = toks[cuts[1]][0], toks[cuts[2]][0]
    block = mcode[lo:hi]
    assert 15_000 <= len(block) <= 25_000, len(block)

    rng = random.Random(0)
    tags = set(range(0x81, 0x85))

    def count_p4(b):
        return sum(1 for i in range(len(b) - 7) if b[i] == 4 and b[i + 6] in tags)

    shuffled = bytearray(block)
    rng.shuffle(shuffled)
    observed, null = count_p4(block), count_p4(bytes(shuffled))
    assert observed >= 40 and null > 0 and observed / null >= 2.0, (observed, null)

    residue = bytes(mcode[i] for i, kind, *_ in toks if kind == "?" and lo <= i < hi)
    assert 8_000 <= len(residue) <= 14_000, len(residue)

    def best_period(b):
        best = 0.0
        for q in range(2, 65):
            best = max(
                best,
                sum(1 for i in range(q, len(b)) if b[i] == b[i - q]) / (len(b) - q),
            )
        return best

    res_shuffled = bytearray(residue)
    rng.shuffle(res_shuffled)
    assert best_period(residue) >= 2.0 * best_period(bytes(res_shuffled))

    top_pair = Counter(residue[i : i + 2] for i in range(len(residue) - 1)).most_common(
        1
    )[0][0]
    assert top_pair[1] == 0 and top_pair[0] in {0x23, 0x16, 0x03, 0x04, 0x01, 0x00}, (
        top_pair.hex()
    )


def test_resnet18d_short_form_tags_are_seventeen_wide_and_trail_a_register(tmp_path):
    """Confirmed real (see the README's "Fourth correction" section), on a
    fresh resnet18d build with a fixed-seed shuffled null of config block B:
    the 17-tag width rule explains most of the block where the 4-tag rule
    explains under half; the byte after the tag is even in >= 99.5% of
    units (a 2-byte-granular register) while the first payload byte is at
    the background rate; `23` is a prefix byte sitting directly before
    ordinary units; the payload-less `[tag][register]` pair is real (its
    second byte is even in >= 99% of 2-byte tag-led residue runs -- the
    fifth correction's claim, replacing the fourth's "null-level" verdict
    that rested on the wrong null); and blocks A and B share a 158-byte
    prologue. No device.
    """
    import random

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    narrow = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(narrow) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    a_lo, lo, hi = narrow[cuts[0]][0], narrow[cuts[1]][0], narrow[cuts[2]][0]
    assert mcode[a_lo : a_lo + 158] == mcode[lo : lo + 158]
    block = mcode[lo:hi]
    shuffled = bytearray(block)
    random.Random(0).shuffle(shuffled)
    shuffled = bytes(shuffled)

    def explained(blob, tags, pmax):
        toks = _tokenize_mcode(blob, start=0, end=len(blob), tags=tags, pmax=pmax)
        return 1 - sum(1 for t in toks if t[1] == "?") / len(blob), toks

    e_narrow, _ = explained(block, None, 3)
    e_wide, toks = explained(block, _WIDE_TAGS, 4)
    e_null, null_toks = explained(shuffled, _WIDE_TAGS, 4)
    assert e_narrow < 0.5 < 0.7 < e_wide and e_wide / e_null >= 3.0, (
        e_narrow,
        e_wide,
        e_null,
    )

    units = [(o, p) for o, k, p, *_ in toks if k == "S"]
    assert len(units) > 2000, len(units)
    trailing_even = sum(1 for o, p in units if block[o + p + 3] % 2 == 0) / len(units)
    payload_even = sum(1 for o, p in units if block[o + 1] % 2 == 0) / len(units)
    assert trailing_even >= 0.995 and payload_even <= 0.75, (
        trailing_even,
        payload_even,
    )

    starts = {o for o, _ in units}
    prefixed = sum(
        1 for o, k, a, *_ in toks if k == "?" and a == 0x23 and o + 1 in starts
    )
    assert prefixed >= 100, prefixed

    def bare_pairs(blob, toks):
        runs, last = [], None
        for o, k, *_ in toks:
            if k == "?":
                if last == o:
                    runs[-1][1] = o + 1
                else:
                    runs.append([o, o + 1])
                last = o + 1
        pairs = [blob[a + 1] for a, b in runs if b - a == 2 and blob[a] in _ALL_TAGS]
        return len(pairs), sum(1 for x in pairs if x % 2 == 0)

    n_real, even_real = bare_pairs(block, toks)
    assert n_real >= 100 and even_real >= 0.99 * n_real, (n_real, even_real)
    n_null, even_null = bare_pairs(shuffled, null_toks)
    assert even_null <= 0.85 * n_null + 2, (n_null, even_null)


def test_resnet18d_every_tag_and_the_bare_pair_pass_the_parity_null(tmp_path):
    """Confirmed real (see the README's "Fifth correction" section), on a
    fresh resnet18d build with a fixed-seed shuffled null of config block
    B: with every byte in 0x81..0x9f admitted as a tag, the trailing byte
    is even in >= 95% of units for *every* tag with n >= 20 (~60% when the
    block is shuffled); with bare `[tag][register]` pairs admitted too the
    block is >= 90% explained; `e1 XX` pairs have odd XX in every case;
    and the residue is dominated by 1-byte prefixes that sit directly
    before a unit. No device.
    """
    import random
    from collections import defaultdict

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    narrow = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(narrow) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    lo, hi = narrow[cuts[1]][0], narrow[cuts[2]][0]
    block = mcode[lo:hi]
    shuffled = bytearray(block)
    random.Random(0).shuffle(shuffled)
    shuffled = bytes(shuffled)

    def per_tag_even(blob):
        toks = _tokenize_mcode(blob, start=0, end=len(blob), tags=_ALL_TAGS, pmax=4)
        per = defaultdict(lambda: [0, 0])
        for o, k, p, tag, _ in toks:
            if k == "S":
                per[tag][0] += 1
                per[tag][1] += blob[o + p + 3] % 2 == 0
        return per

    real, null = per_tag_even(block), per_tag_even(shuffled)
    tested = [t for t, (n, _) in real.items() if n >= 20]
    assert len(tested) >= 15 and {0x87, 0x88, 0x8E, 0x92} <= set(tested), tested
    for t in tested:
        n, ev = real[t]
        # 0x9c sits at 47/48 on this build -- one miss, well above the ~60% null.
        assert ev >= 0.95 * n, (hex(t), n, ev)
    null_even = sum(ev for n, ev in null.values()) / sum(n for n, ev in null.values())
    assert null_even <= 0.8, null_even

    toks = _tokenize_mcode(
        block, start=0, end=len(block), tags=_ALL_TAGS, pmax=4, bare=True
    )
    explained = 1 - sum(1 for t in toks if t[1] == "?") / len(block)
    assert explained >= 0.9, explained
    assert sum(1 for t in toks if t[1] == "B") >= 500

    runs, last = [], None
    for o, k, *_ in toks:
        if k == "?":
            if last == o:
                runs[-1][1] = o + 1
            else:
                runs.append([o, o + 1])
            last = o + 1

    # `e1 XX` as a 2-byte residue run: XX is odd (README: 13/14, 48/48, 7/7).
    e1 = [block[a + 1] for a, b in runs if b - a == 2 and block[a] == 0xE1]
    assert len(e1) >= 10 and sum(x % 2 for x in e1) >= 0.9 * len(e1), e1

    starts = {o for o, k, *_ in toks if k != "?"}
    singles = [a for a, b in runs if b - a == 1]
    assert len(singles) >= 0.75 * len(runs), (len(singles), len(runs))
    assert sum(1 for a in singles if a + 1 in starts) >= 0.9 * len(singles)


def test_resnet18d_tag_9f_units_carry_one_extra_byte(tmp_path):
    """Confirmed real (see the README's "Sixth correction" section), on a
    fresh resnet18d build: a unit whose tag is 0x9f is followed by exactly
    one leftover byte in >= 85% of cases while every other tag with n >= 30
    is followed by one in <= 10%; that byte is below 0x40 in >= 95% of
    cases; and admitting the extra byte (`extra_byte_tags={0x9f}`) takes
    config block B to >= 95% explained while a shuffled copy stays under
    55%. No device.
    """
    import random
    from collections import Counter, defaultdict

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    narrow = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(narrow) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    lo, hi = narrow[cuts[1]][0], narrow[cuts[2]][0]
    block = mcode[lo:hi]

    toks = _tokenize_mcode(
        block, start=0, end=len(block), tags=_ALL_TAGS, pmax=4, bare=True
    )
    follow = defaultdict(Counter)
    extra = Counter()
    for j, t in enumerate(toks[:-1]):
        if t[1] not in ("S", "B"):
            continue
        tag = t[3] if t[1] == "S" else t[2]
        nxt = toks[j + 1]
        one = nxt[1] == "?" and (j + 2 >= len(toks) or toks[j + 2][1] != "?")
        follow[tag][one] += 1
        if one and tag == 0x9F:
            extra[block[nxt[0]]] += 1
    n9 = sum(follow[0x9F].values())
    assert n9 >= 300 and follow[0x9F][True] >= 0.85 * n9, dict(follow[0x9F])
    for tag, c in follow.items():
        if tag != 0x9F and sum(c.values()) >= 30:
            assert c[True] <= 0.1 * sum(c.values()), (hex(tag), dict(c))
    assert sum(c for v, c in extra.items() if v < 0x40) >= 0.95 * sum(extra.values())

    def explained(blob):
        toks = _tokenize_mcode(
            blob,
            start=0,
            end=len(blob),
            tags=_ALL_TAGS,
            pmax=4,
            bare=True,
            extra_byte_tags={0x9F},
        )
        return 1 - sum(1 for t in toks if t[1] == "?") / len(blob)

    shuffled = bytearray(block)
    random.Random(0).shuffle(shuffled)
    e_real, e_null = explained(block), explained(bytes(shuffled))
    assert e_real >= 0.95 and e_null <= 0.55, (e_real, e_null)
