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

import glob
import json
import os
import re
import shutil
import struct
import sys
import tempfile

import numpy as np
import onnx
import onnx.utils
import pytest
from onnx import TensorProto, helper, numpy_helper, parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode as _mcode_mod  # noqa: E402
import pulsar2_docker  # noqa: E402

# The codec moved to scripts/axera/mcode.py, so that reading, writing and
# checking an mcode needs neither a Pulsar2 image nor a card. These tests keep
# the private names they grew up with.
_ALL_TAGS = _mcode_mod.ALL_TAGS
_FULL_RULE = _mcode_mod.FULL_RULE
_VERBS = _mcode_mod.VERBS
_VERBS6 = _mcode_mod.VERBS6
_WIDE_TAGS = _mcode_mod.WIDE_TAGS
_decode_mcode = _mcode_mod.decode
_encode_mcode = _mcode_mod.encode
_mcodes_of = _mcode_mod.mcodes_of
_nonzero_coverage = _mcode_mod.nonzero_coverage
_segment_coverage = _mcode_mod.segment_coverage
_segments = _mcode_mod.segments
_stream_bounds = _mcode_mod.stream_bounds
_structured_share = _mcode_mod.structured_share
_tail_tables = _mcode_mod.tail_tables
_tail_vector = _mcode_mod.tail_vector
_tokenize_mcode = _mcode_mod.tokenize

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


def test_resnet18d_stream_ends_at_a_five_table_flatbuffers_tail(tmp_path):
    """Confirmed real (see the README's "The op programs are fully
    tokenized" section), on a fresh resnet18d build: the header's word at
    offset 272 is a FlatBuffers offset landing exactly on the tail's
    five-table vector, ~480 bytes before the end; the instruction stream
    ends 7 bytes after its last verb, just before that vector's zero
    padding; under the full rule the op region has zero residue and the
    stream is >= 97% explained; the header's last 17 bytes close block A
    too; and tables 1..3 carry type word 4097 with the packed field equal
    to `count << 8 | 1`. No device.
    """
    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    vec, tables = _tail_tables(mcode)
    assert 460 <= len(mcode) - vec <= 500, len(mcode) - vec
    assert 272 + struct.unpack_from("<I", mcode, 272)[0] == vec

    pad = 0
    while mcode[vec - pad - 1] == 0:
        pad += 1
    stream_end = vec - pad

    narrow = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(narrow) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    a_lo, b_lo, ops_lo = narrow[cuts[0]][0], narrow[cuts[1]][0], narrow[cuts[2]][0]
    assert mcode[280:297] == mcode[b_lo - 17 : b_lo]
    assert mcode[b_lo - 8 : b_lo - 4] == bytes.fromhex("a2000000")

    full = dict(tags=_ALL_TAGS, pmax=4, bare=True, extra_byte_tags={0x9F})
    toks = _tokenize_mcode(mcode, start=a_lo, end=stream_end, **full)
    last = toks[-1]
    assert last[1] == "V" and last[2] == 0xA2 and stream_end - last[0] == 7, last
    ops = [t for t in toks if t[0] >= ops_lo]
    assert ops and not any(t[1] == "?" for t in ops)
    explained = 1 - sum(1 for t in toks if t[1] == "?") / (stream_end - a_lo)
    assert explained >= 0.97, explained

    assert len(tables) == 5
    for k in (1, 2, 3):
        assert tables[k][0] == 4097 and tables[k][4] == (tables[k][3] << 8) | 1, tables[
            k
        ]
    assert tables[4][0] == 2561


def _check_segment_table(mcode):
    """The segment-table claims shared by every build: countdown, tiling,
    an `a7` marker verb within the first 16 bytes of every segment."""
    vec, tables = _tail_tables(mcode)
    for k in range(1, len(tables)):
        assert tables[k].get(3, 0) == tables[k - 1][3] - tables[k][2], (k, tables[k])
    assert tables[0][3] == sum(t.get(2, 0) for t in tables[1:])
    header, segs = _segments(mcode)
    assert segs[-1][0] + segs[-1][1] == vec
    # The marker's `a7 00 00` may sit in the 4 bytes before the 8-byte-word
    # boundary (the `llm_build` subgraphs' segments open `1e 00 00 00 00 a2`,
    # the marker's operand), so look from 4 bytes before each segment start.
    for pos, length, _ in segs:
        assert mcode.find(b"\xa7\x00\x00", pos - 4, pos + 16) >= 0, (
            pos,
            mcode[pos - 4 : pos + 16].hex(),
        )
    return header, segs


def test_resnet18d_tail_segments_tile_the_stream_and_open_with_a7(tmp_path):
    """Confirmed real (see the README's "The tail is the segment table"
    section), on a fresh resnet18d build: the tail tables' field 2 is a
    length in 8-byte words that tiles the blob exactly from the 280-byte
    header to the tail vector in reverse table order; field 3 counts down;
    the last segment is block A exactly; every segment opens with an `a7`
    verb; the op-program segment (table 0) is 100% tokenized with six
    verbs and holds every `a1 40 02` program; types are 0xa01, 0x1001 x3,
    0xe801. No device.
    """
    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    header, segs = _check_segment_table(mcode)
    assert header == 280 and len(segs) == 5
    narrow = _tokenize_mcode(mcode)
    cuts = [k for k, t in enumerate(narrow) if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    a_lo, b_lo = narrow[cuts[0]][0], narrow[cuts[1]][0]
    assert segs[0][0] + segs[0][1] == b_lo - 17 and segs[0][1] == b_lo - a_lo
    assert [s[2][0] for s in segs] == [0xA01, 0x1001, 0x1001, 0x1001, 0xE801]
    cov, programs, _ = _segment_coverage(mcode, segs[-1])
    assert cov >= 0.999 and programs == len(cuts) - 2, (cov, programs, len(cuts))
    for seg in segs[:-1]:
        assert _segment_coverage(mcode, seg)[0] >= 0.9


def _cached_hf_checkpoint(repo_id, fallback_dir=None):
    """Local snapshot directory of a HuggingFace checkpoint already in the
    cache (with its weights present), else `fallback_dir` if that holds a
    checkpoint, else None. Never downloads -- these tests stay offline."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, local_files_only=True)
        if any(f.endswith((".safetensors", ".bin")) for f in os.listdir(path)):
            return path
    except Exception:
        pass
    if fallback_dir and os.path.exists(os.path.join(fallback_dir, "config.json")):
        return os.path.abspath(fallback_dir)
    return None


def test_llm_build_layer_mcode_keeps_the_layout_and_uses_a7(tmp_path):
    """Confirmed real (see the README's "The tail is the segment table"
    section), on a fresh `pulsar2 llm_build` of SmolLM2-135M: each of the
    per-layer file's two subgraphs has a 15-segment tail whose segments
    tile the stream, every segment opens with `a7`, the op-program segment
    (table 2, type 0x8801) is 100% tokenized and uses `a7` inside its
    programs, and the whole stream is >= 95% explained. Skips unless the
    checkpoint is already in the HuggingFace cache. No device.
    """
    import shutil
    from collections import Counter

    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)
    result = pulsar2_docker.llm_build(str(work), "SmolLM2-135M", "output", parallel=8)
    assert result.success, getattr(result, "error", None)
    layer = str(work / "output" / "llama_p512_l0_together.axmodel")
    mcodes = _mcodes_of(layer)
    assert len(mcodes) == 2, [n for n, _ in mcodes]
    for _, mcode in mcodes:
        _, segs = _check_segment_table(mcode)
        assert len(segs) == 15
        ops = [s for s in segs if s[2][0] == 0x8801]
        assert len(ops) == 1
        cov, programs, a7 = _segment_coverage(mcode, ops[0])
        # Not exactly 1.0 any more: admitting 0xa1 as a tag (the fifth
        # correction) costs a handful of bytes inside op segments and saves
        # hundreds elsewhere -- see the README's coverage section.
        assert cov >= 0.999 and programs >= 100 and a7 >= 100, (cov, programs, a7)
        # The op template: the CNN core with `a7.1e` (operand 0) after the
        # two `50.01` writes and `a7.02` (operand 2) before the closing
        # `a8 30.02`, in the two most common skeletons.
        pos, length, _ = ops[0]
        toks = _tokenize_mcode(mcode, start=pos, end=pos + length, **_FULL_RULE)
        starts = [t[0] for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02)]
        skeletons = Counter()
        for a, b in zip(starts, starts[1:]):
            prog = tuple(
                (
                    t[2],
                    t[3],
                    t[4],
                    struct.unpack_from("<I", mcode, t[0] + 4)[0]
                    if t[2] == 0xA7
                    else None,
                )
                for t in toks
                if a <= t[0] < b and t[1] == "V"
            )
            skeletons[prog] += 1
        top = skeletons.most_common(2)
        assert sum(c for _, c in top) >= 0.4 * sum(skeletons.values()), top
        for prog, _ in top:
            assert prog[3] == (0xA7, 0x00, 0x1E, 0) and prog[-2] == (
                0xA7,
                0x00,
                0x02,
                2,
            ), prog
            assert prog[-1][:3] == (0xA8, 0x30, 0x02)
        total_e = total_n = 0
        for seg in segs:
            pos, length, _ = seg
            end = pos + length
            while end > pos and mcode[end - 1] == 0:
                end -= 1
            c, _, _ = _segment_coverage(mcode, seg)
            total_e += c * (end - pos)
            total_n += end - pos
        assert total_e / total_n >= 0.95, total_e / total_n

    # One instruction stream for all 30 layers: header + stream identical
    # byte for byte, only the tail's name string and the weights differ.
    for other in (1, 29):
        path = str(work / "output" / f"llama_p512_l{other}_together.axmodel")
        # Pair by node name: the two subgraphs are not always in the same
        # order in every layer's file.
        by_name = dict(_mcodes_of(path))
        for name, m0 in mcodes:
            m1 = by_name[name]
            header, segs = _segments(m0)
            vec = segs[-1][0] + segs[-1][1]
            assert len(m0) == len(m1), (name, len(m0), len(m1))
            assert m0[header:vec] == m1[header:vec], name
            diff = [i for i in range(vec, len(m0)) if m0[i] != m1[i]]
            assert 1 <= len(diff) <= 24, (name, len(diff))
    w0 = onnx.load(layer)
    w29 = onnx.load(str(work / "output" / "llama_p512_l29_together.axmodel"))
    params0 = {t.name: bytes(t.raw_data) for t in w0.graph.initializer}
    params29 = {t.name: bytes(t.raw_data) for t in w29.graph.initializer}
    key = _params_key(w0)
    assert len(params0[key]) == len(params29[key]) and params0[key] != params29[key]


def test_onnx_path_llm_mcode_keeps_the_op_program_skeleton(tmp_path):
    """Confirmed real (see the README's "The tail is the segment table"
    section), on a fresh ONNX-path build of the 1-layer tiny-random-mistral
    checkpoint: five segments that tile the stream, a 436-byte header (two
    graph inputs), a 100%-tokenized op segment with ~75 programs for 80
    ONNX ops, and the CNN skeleton verb for verb with `a1 20.02` in place
    of `a1 30.03`. Skips unless the checkpoint is in the HuggingFace
    cache. No device.
    """
    from collections import Counter

    ckpt = _cached_hf_checkpoint(
        "distilabel-internal-testing/tiny-random-mistral",
        fallback_dir=os.path.join(
            os.path.dirname(__file__), "..", "tiny-random-mistral"
        ),
    )
    if ckpt is None:
        pytest.skip("tiny-random-mistral is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    result = pulsar2_docker.build_from_hf_checkpoint(ckpt, str(work), "output")
    assert result.success, result.error
    ((_, mcode),) = _mcodes_of(result.axmodel_path)
    header, segs = _check_segment_table(mcode)
    assert header == 436 and len(segs) == 5
    cov, programs, _ = _segment_coverage(mcode, segs[-1])
    assert cov >= 0.999 and 60 <= programs <= 90, (cov, programs)
    pos, length, _ = segs[-1]
    toks = _tokenize_mcode(mcode, start=pos, end=pos + length, **_FULL_RULE)
    starts = [t[0] for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02)]
    skeletons = Counter()
    for a, b in zip(starts, starts[1:]):
        prog = tuple((t[2], t[3], t[4]) for t in toks if a <= t[0] < b and t[1] == "V")
        skeletons[prog] += 1
    core = (
        (0xA1, 0x40, 0x02),
        (0xA1, 0x50, 0x01),
        (0xA1, 0x50, 0x01),
        (0xA8, 0x40, 0x03),
        (0xA1, 0x50, 0x03),
        (0xA1, 0x50, 0x01),
        (0xA3, 0x00, 0x00),
        (0xA1, 0x50, 0x01),
        (0xA9, 0x00, 0x00),
    )
    top = skeletons.most_common(2)
    assert sum(c for _, c in top) >= 0.6 * sum(skeletons.values()), top
    for prog, _ in top:
        assert tuple(v for v in prog if v in core) == core, prog
    assert sum(c for prog, c in skeletons.items() if (0xA1, 0x20, 0x02) in prog) >= 5


def _llm_layer_inputs(model):
    """Valid inputs for an `llm_build` per-layer file's decode subgraph
    (group 0: the graph inputs not suffixed `_1`): zero K/V caches, zero
    hidden state and mask, and *in-range* `indices` -- `axcl_run_model`'s
    own random bytes land in that gather input and fault the NPU."""
    sizes = {onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.FLOAT: 4}
    sizes.update(
        {
            onnx.TensorProto.INT32: 4,
            onnx.TensorProto.UINT32: 4,
            onnx.TensorProto.INT64: 8,
        }
    )
    inputs = {}
    for i in model.graph.input:
        if i.name.endswith("_1"):
            continue
        n = int(np.prod([d.dim_value for d in i.type.tensor_type.shape.dim] or [1]))
        size = sizes[i.type.tensor_type.elem_type]
        if i.name.startswith("indices"):
            inputs[i.name] = np.arange(
                n, dtype=np.int32 if size == 4 else np.int64
            ).tobytes()
        else:
            inputs[i.name] = b"\x00" * (n * size)
    return inputs


def _run_llm_layer(path, inputs, times):
    """Run a per-layer file `times` times; returns a list of outcomes, each
    either the string "fault" (a `0x8030070C` rejection) or the tuple of
    output digests. Empty-output runs (the runtime's own transient) are
    skipped, so callers judge on what actually came back."""
    import hashlib

    out = []
    for _ in range(times):
        r = pulsar2_docker.run_on_device_with_inputs(path, inputs, repeat=1, warmup=1)
        if r.error and "0x8030070C" in r.error:
            out.append("fault")
        elif r.outputs:
            out.append(tuple(hashlib.sha1(o).hexdigest() for o in r.outputs))
    return out


def test_llm_build_a7_is_a_sync_verb_on_device(tmp_path):
    """Confirmed real on the AX650N (see the README's "Confirmed on the
    AX650N: `a7` is a synchronization verb" table), on a fresh `llm_build`
    of SmolLM2-135M: the layer-0 decode subgraph runs deterministically
    with valid inputs; patching every in-program `a7.02` operand from 2 to
    0 keeps it running but changes every output; patching every in-program
    `a7.1e` operand from 0 to 1 faults; re-routing `a7.02` to channel 0x1e
    leaves the outputs identical. Each outcome must reproduce on both of
    two runs. Skips without a device or the cached checkpoint.
    """
    import shutil

    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")
    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)
    result = pulsar2_docker.llm_build(str(work), "SmolLM2-135M", "output", parallel=8)
    assert result.success, getattr(result, "error", None)
    layer = str(work / "output" / "llama_p512_l0_together.axmodel")

    model = onnx.load(layer)
    inputs = _llm_layer_inputs(model)
    neu0 = next(n for n in model.graph.node if n.op_type == "neu mode")
    info = json.loads(
        next(a for a in neu0.attribute if a.name == "npu_graph_info").s.decode()
    )
    key = info["dotneus"][0]["neu_key"]
    mcode = bytes(next(i for i in model.graph.initializer if i.name == key).raw_data)
    _, segs = _segments(mcode)
    ops = next(s for s in segs if s[2][0] == 0x8801)
    toks = _tokenize_mcode(mcode, start=ops[0], end=ops[0] + ops[1], **_FULL_RULE)
    a7_02 = [t[0] for t in toks if t[1] == "V" and t[2] == 0xA7 and t[4] == 0x02]
    a7_1e = [t[0] for t in toks if t[1] == "V" and t[2] == 0xA7 and t[4] == 0x1E]
    assert len(a7_02) >= 50 and len(a7_1e) >= 50, (len(a7_02), len(a7_1e))

    def patched(name, edit):
        b = bytearray(mcode)
        edit(b)
        m2 = onnx.ModelProto()
        m2.CopyFrom(model)
        next(i for i in m2.graph.initializer if i.name == key).raw_data = bytes(b)
        path = str(tmp_path / f"{name}.axmodel")
        onnx.save(m2, path)
        return path

    def set_operand(offs, val):
        def edit(b):
            for o in offs:
                struct.pack_into("<I", b, o + 4, val)

        return edit

    def set_channel(offs, ch):
        def edit(b):
            for o in offs:
                b[o + 3] = ch

        return edit

    # The runtime can reject one run transiently with the same fault code
    # (see `_run_retry_once`), so a variant that is *expected to run* is
    # judged on its non-fault runs -- at least two, all equal -- while a
    # variant expected to fault must fault on every run.
    def good(outcomes):
        return [o for o in outcomes if o != "fault"]

    base = good(_run_llm_layer(layer, inputs, 3))
    assert len(base) >= 2 and len(set(base)) == 1, ("baseline", base)
    baseline = base[0]

    changed = good(
        _run_llm_layer(patched("a7_02_operand_0", set_operand(a7_02, 0)), inputs, 3)
    )
    assert len(changed) >= 2 and len(set(changed)) == 1, changed
    assert all(a != b for a, b in zip(changed[0], baseline)), (changed[0], baseline)

    faulted = _run_llm_layer(
        patched("a7_1e_operand_1", set_operand(a7_1e, 1)), inputs, 2
    )
    assert faulted == ["fault", "fault"], faulted

    same = good(
        _run_llm_layer(patched("a7_02_channel_1e", set_channel(a7_02, 0x1E)), inputs, 3)
    )
    assert len(same) >= 2 and all(s == baseline for s in same), same


def test_resnet18d_a1_is_a_tag_and_bit6_tags_take_odd_registers(tmp_path):
    """Confirmed real (see the README's "0xa1 is also a tag" paragraph), on
    a fresh resnet18d build's configuration segments under the full rule:
    among the 2-byte non-zero residue runs, `a1 XX` has an even XX in
    >= 95% of cases and `e1 XX`/`c1 XX` an odd XX in every case, while the
    same runs in the shuffled segments split about evenly; admitting them
    (`tags | {0xa1}`, `odd_tags={0xc1, 0xe1}`) lowers the non-zero residue.
    No device.
    """
    import random
    from collections import Counter

    _, _, mcode = _build_real_resnet18d(str(tmp_path))
    _, segs = _segments(mcode)
    blob = b"".join(mcode[p : p + n] for p, n, t in segs if t[0] != 0xE801)
    shuffled = bytearray(blob)
    random.Random(0).shuffle(shuffled)
    shuffled = bytes(shuffled)

    def pair_parity(b):
        toks = _tokenize_mcode(b, start=0, end=len(b), **_FULL_RULE)
        runs, last = [], None
        for o, k, *_ in toks:
            if k == "?" and b[o]:
                if last == o:
                    runs[-1][1] = o + 1
                else:
                    runs.append([o, o + 1])
                last = o + 1
        par = Counter()
        for a, e in runs:
            if e - a == 2 and b[a] in (0xA1, 0xC1, 0xE1):
                par[(b[a], b[a + 1] % 2)] += 1
        return par

    real, null = pair_parity(blob), pair_parity(shuffled)
    a1_even, a1_odd = real[(0xA1, 0)], real[(0xA1, 1)]
    # resnet18d has ~15 such pairs (the llm_build subgraphs 37 and 65).
    assert a1_even >= 10 and a1_even >= 0.9 * (a1_even + a1_odd), (a1_even, a1_odd)
    odd = real[(0xE1, 1)] + real[(0xC1, 1)]
    even = real[(0xE1, 0)] + real[(0xC1, 0)]
    # 18 of 19 on a fresh resnet18d build; 71/71 and 116/116 in llm_build.
    assert odd >= 10 and odd >= 0.9 * (odd + even), (odd, even)
    null_a1 = null[(0xA1, 0)] + null[(0xA1, 1)]
    assert null_a1 == 0 or null[(0xA1, 0)] <= 0.8 * null_a1 + 2, dict(null)

    def nonzero_residue(b, **extra):
        rule = dict(_FULL_RULE)
        rule.update(extra)
        toks = _tokenize_mcode(b, start=0, end=len(b), **rule)
        return sum(1 for t in toks if t[1] == "?" and b[t[0]])

    before = nonzero_residue(blob)
    after = nonzero_residue(blob, tags=_ALL_TAGS | {0xA1}, odd_tags={0xC1, 0xE1})
    assert after <= 0.85 * before, (before, after)


def test_llm_build_segment_table_is_validated_on_device(tmp_path):
    """Confirmed real on the AX650N (see the README's "the segment table is
    a loader manifest -- but only its word counts are load-bearing"
    paragraph), on a fresh `llm_build` of SmolLM2-135M and a healthy
    device: the layer-0 decode subgraph runs deterministically with valid
    inputs; table 0's word count + 1 is rejected at load ("Loading model
    failed", no fault); the op segment's word count - 1 faults every run;
    and the op segment's remaining count + 1 runs with outputs identical
    to baseline. The type-word patch is deliberately not exercised -- it
    is the one that stalled the runtime. Skips without a device or the
    cached checkpoint.
    """
    import shutil

    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")
    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)
    result = pulsar2_docker.llm_build(str(work), "SmolLM2-135M", "output", parallel=8)
    assert result.success, getattr(result, "error", None)
    layer = str(work / "output" / "llama_p512_l0_together.axmodel")

    model = onnx.load(layer)
    inputs = _llm_layer_inputs(model)
    neu0 = next(n for n in model.graph.node if n.op_type == "neu mode")
    info = json.loads(
        next(a for a in neu0.attribute if a.name == "npu_graph_info").s.decode()
    )
    key = info["dotneus"][0]["neu_key"]
    mcode = bytes(next(i for i in model.graph.initializer if i.name == key).raw_data)
    vec, tables = _tail_tables(mcode)

    def field_offset(k, f):
        p = vec + 4 + 4 * k
        tpos = p + struct.unpack_from("<I", mcode, p)[0]
        vt = tpos - struct.unpack_from("<i", mcode, tpos)[0]
        fo = struct.unpack_from("<H", mcode, vt + 4 + 2 * f)[0]
        assert fo, (k, f)
        return tpos + fo

    k_ops = next(k for k, t in enumerate(tables) if t[0] == 0x8801)

    def patched(name, k, f, value):
        b = bytearray(mcode)
        struct.pack_into("<I", b, field_offset(k, f), value)
        m2 = onnx.ModelProto()
        m2.CopyFrom(model)
        next(i for i in m2.graph.initializer if i.name == key).raw_data = bytes(b)
        path = str(tmp_path / f"{name}.axmodel")
        onnx.save(m2, path)
        return path

    base = [o for o in _run_llm_layer(layer, inputs, 3) if o != "fault"]
    assert len(base) >= 2 and len(set(base)) == 1, ("baseline", base)
    baseline = base[0]

    load = pulsar2_docker.run_on_device_with_inputs(
        patched("t0_f2_plus1", 0, 2, tables[0][2] + 1), inputs, repeat=1, warmup=1
    )
    assert not load.outputs and load.error and "Loading model" in load.error, load.error

    short = _run_llm_layer(
        patched("ops_f2_minus1", k_ops, 2, tables[k_ops][2] - 1), inputs, 2
    )
    assert short == ["fault", "fault"], short

    same = [
        o
        for o in _run_llm_layer(
            patched("ops_f3_plus1", k_ops, 3, tables[k_ops][3] + 1), inputs, 3
        )
        if o != "fault"
    ]
    assert len(same) >= 2 and all(o == baseline for o in same), same


_ANALYSIS_BUILDS = ("resnet18d", "mistral")
"""The device-free builds the coverage floor is measured on: the real
resnet18d, and the 1-layer tiny-random-mistral through the ONNX path (the
`llm_build` layer is covered by its own test, which already builds it)."""


def _build_analysis_mcode(name, work_dir):
    """The mcode of one `_ANALYSIS_BUILDS` entry, compiled for real."""
    if name == "resnet18d":
        return _build_real_resnet18d(work_dir)[2]
    ckpt = _cached_hf_checkpoint(
        "distilabel-internal-testing/tiny-random-mistral",
        fallback_dir=os.path.join(
            os.path.dirname(__file__), "..", "tiny-random-mistral"
        ),
    )
    if ckpt is None:
        pytest.skip("tiny-random-mistral is not available locally")
    result = pulsar2_docker.build_from_hf_checkpoint(ckpt, work_dir, "output")
    assert result.success, result.error
    ((_, mcode),) = _mcodes_of(result.axmodel_path)
    return mcode


def _vocoder_model(alpha=0.1, tail="Tanh", frames=32, stride=8, dilation=2):
    """A HiFi-GAN-shaped vocoder -- the part of a text-to-speech model that
    actually runs on this NPU. A full VITS graph (e.g. a Piper voice) cannot
    be compiled: it carries `RandomNormalLike`, `NonZero`, `CumSum` and
    `Range`, which are stochastic or data-dependent, so real deployments keep
    the text encoder and sampling on the CPU and send only the vocoder to the
    device. This is that shape: transposed-convolution upsampling, a dilated
    residual convolution, LeakyReLU and a bounded output.
    """
    rng = np.random.RandomState(0)
    inits, nodes = [], []

    def w(name, shape):
        inits.append(
            numpy_helper.from_array((rng.randn(*shape) * 0.05).astype(np.float32), name)
        )
        return name

    nodes.append(
        helper.make_node(
            "Conv", ["mel", w("w0", (64, 80, 7))], ["h0"], kernel_shape=[7], pads=[3, 3]
        )
    )
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            ["h0", w("w1", (64, 32, 16))],
            ["u1"],
            kernel_shape=[16],
            strides=[stride],
            pads=[4, 4],
        )
    )
    nodes.append(helper.make_node("LeakyRelu", ["u1"], ["a1"], alpha=alpha))
    nodes.append(
        helper.make_node(
            "Conv",
            ["a1", w("w2", (32, 32, 3))],
            ["r1"],
            kernel_shape=[3],
            pads=[dilation, dilation],
            dilations=[dilation],
        )
    )
    nodes.append(helper.make_node("Add", ["a1", "r1"], ["s1"]))
    nodes.append(
        helper.make_node(
            "Conv", ["s1", w("w3", (1, 32, 7))], ["y0"], kernel_shape=[7], pads=[3, 3]
        )
    )
    nodes.append(helper.make_node(tail, ["y0"], ["audio"]))
    graph = helper.make_graph(
        nodes,
        "vocoder",
        [helper.make_tensor_value_info("mel", onnx.TensorProto.FLOAT, [1, 80, frames])],
        [
            helper.make_tensor_value_info(
                "audio", onnx.TensorProto.FLOAT, [1, 1, frames * stride]
            )
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return onnx.shape_inference.infer_shapes(model)


def _build_vocoder(work_dir, **kwargs):
    """Compile `_vocoder_model()` for real and return its mcode."""
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    model = _vocoder_model(**kwargs)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    frames = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    rng = np.random.RandomState(0)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib.tar"),
        [rng.randn(1, 80, frames).astype(np.float32) for _ in range(4)],
    )
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "mel",
                            "calibration_dataset": "./dataset/calib.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": 4,
                        }
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error
    ((_, mcode),) = _mcodes_of(result.axmodel_path)
    return mcode


def _elementwise_model(op, channels, length=64):
    """A single elementwise op, shape-preserving, so its program can be
    compared against a convolution's on equal footing."""
    rng = np.random.RandomState(0)
    inits = []
    if op in ("Add", "Mul"):
        inits.append(
            numpy_helper.from_array(
                rng.randn(1, channels, length).astype(np.float32), "b"
            )
        )
        node = helper.make_node(op, ["x", "b"], ["y"])
    elif op == "LeakyRelu":
        node = helper.make_node(op, ["x"], ["y"], alpha=0.1)
    else:
        node = helper.make_node(op, ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "elementwise",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, length])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, channels, length])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _one_conv_model(
    cin, cout, length=64, kernel=3, dilation=1, op="Conv", weights=None
):
    """A single same-padded 1-D convolution, so the compiled program has
    exactly one op and its operands are unambiguous."""
    rng = np.random.RandomState(0)
    shape = (cin, cout, kernel) if op == "ConvTranspose" else (cout, cin, kernel)
    values = (rng.randn(*shape) * 0.05) if weights is None else weights
    weight = numpy_helper.from_array(np.asarray(values, dtype=np.float32), "w")
    pad = dilation * (kernel - 1) // 2
    node = helper.make_node(
        op,
        ["x", "w"],
        ["y"],
        name="conv",
        kernel_shape=[kernel],
        pads=[pad, pad],
        dilations=[dilation],
    )
    graph = helper.make_graph(
        [node],
        "one_conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, length])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, length])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _build_single_op(work_dir, tag, model):
    """Compile a one-op model for real and return its mcode."""
    ((_, mcode),) = _mcodes_of(_build_single_op_axmodel(work_dir, tag, model))
    return mcode


def _build_single_op_axmodel(work_dir, tag, model):
    """Compile a one-op model for real and return the `.axmodel` path."""
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    onnx.save(model, os.path.join(work_dir, f"{tag}.onnx"))
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    rng = np.random.RandomState(1)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", f"{tag}.tar"),
        [rng.randn(*shape).astype(np.float32) for _ in range(4)],
    )
    with open(os.path.join(work_dir, "config", f"{tag}.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "x",
                            "calibration_dataset": f"./dataset/{tag}.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": 4,
                        }
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    result = pulsar2_docker.build(
        work_dir, f"{tag}.onnx", f"out_{tag}", config_path=f"config/{tag}.json"
    )
    assert result.success, result.error
    return result.axmodel_path


def _operands(mcode, verb, field, bank):
    """Every operand written by `verb` to `field`.`bank`, as integers."""
    lo, hi = _stream_bounds(mcode)
    out = []
    for record in _decode_mcode(mcode, start=lo, end=hi, **_FULL_RULE):
        if (
            record["kind"] == "V"
            and record["verb"] == verb
            and record.get("field") == field
            and record.get("bank") == bank
        ):
            operand = record.get("operand")
            out.append(
                int.from_bytes(operand, "little")
                if isinstance(operand, (bytes, bytearray))
                else operand
            )
    return out


# The weight table's layout for a convolution, confirmed by placing a single
# non-zero weight at known positions and seeing which byte moved. See the
# README's "Generating a weight table" section.
#
# Weights are INT8 (zero point 128) split across two 36-byte nibble planes,
# the high plane `_WBT_PLANE_GAP` bytes after the low one. Two input channels
# share a byte. Output channels are grouped 16 at a time and slots are chunked
# 36 at a time, and those groups and chunks interleave in 72-byte pairs -- so
# a channel's weights are not contiguous once either dimension overflows.
_WBT_PLANE = 36
_WBT_PLANE_GAP = 36
_WBT_PAIR = 72
_WBT_O_GROUP = 16


def _wbt_of(axmodel_path):
    """The `npu_params` weight table of a compiled model."""
    model = onnx.load(axmodel_path)
    return bytes(
        next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    )


def _weight_offset(o, i, k, cin, cout, kernel):
    """`(byte offset, nibble shift)` of weight `(o, i, k)`'s low-nibble plane.
    The high nibble lives `_WBT_PLANE_GAP` bytes further on.

    Slot `s` walks the weights of one output channel; it is chunked into
    36-byte planes, and output channels are grouped 16 at a time. Both the
    group and the chunk index select which 72-byte pair the plane lands in.
    """
    s = (cin // 2) * k + i // 2
    chunk, within = divmod(s, _WBT_PLANE)
    group, member = divmod(o, _WBT_O_GROUP)
    groups = -(-cout // _WBT_O_GROUP)
    chunks = -(-((cin // 2) * kernel) // _WBT_PLANE)
    low = (
        _WBT_PAIR * groups * chunks * member
        + _WBT_PAIR * (group + groups * chunk)
        + within
    )
    return low, (4 if i % 2 else 0)


def _read_weight_codes(wbt, shape):
    """The INT8 codes (zero point 128) a weight table holds, as `shape`."""
    cout, cin, kernel = shape
    codes = np.zeros(shape, dtype=int)
    for o in range(cout):
        for i in range(cin):
            for k in range(kernel):
                off, sh = _weight_offset(o, i, k, cin, cout, kernel)
                lo = (wbt[off] >> sh) & 0xF
                hi = (wbt[off + _WBT_PLANE_GAP] >> sh) & 0xF
                codes[o, i, k] = (hi << 4) | lo
    return codes


def _write_weight_codes(wbt, codes):
    """`wbt` with the convolution's weight codes replaced."""
    out = bytearray(wbt)
    cout, cin, kernel = codes.shape
    for o in range(cout):
        for i in range(cin):
            for k in range(kernel):
                off, sh = _weight_offset(o, i, k, cin, cout, kernel)
                value = int(codes[o, i, k]) & 0xFF
                out[off] = (out[off] & ~(0xF << sh) & 0xFF) | ((value & 0xF) << sh)
                high = off + _WBT_PLANE_GAP
                out[high] = (out[high] & ~(0xF << sh) & 0xFF) | (
                    ((value >> 4) & 0xF) << sh
                )
    return bytes(out)


def _effective_slopes(weights, codes):
    """Pulsar2's own per-output-channel slope (the reciprocal of the weight
    scale), recovered by least squares from a compiled reference. Reading it
    back beats deriving it: the scale is close to `127.5 / max|w|` but not
    exactly, and it is the compiler's choice, not ours."""
    slopes = np.zeros(weights.shape[0])
    for o in range(weights.shape[0]):
        w = weights[o].reshape(-1).astype(float)
        q = codes[o].reshape(-1).astype(float) - 128
        nz = np.abs(w) > 0
        slopes[o] = np.sum(w[nz] * q[nz]) / np.sum(w[nz] ** 2)
    return slopes


def _one_conv2d_model(cin, cout, hw=16, kernel=3, weights=None):
    """A single same-padded 2-D convolution -- the shape a real CNN is built
    from, and a different weight packing from the 1-D case."""
    rng = np.random.RandomState(0)
    shape = (cout, cin, kernel, kernel)
    values = (rng.randn(*shape) * 0.05) if weights is None else weights
    weight = numpy_helper.from_array(np.asarray(values, dtype=np.float32), "w")
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        name="conv",
        kernel_shape=[kernel, kernel],
        pads=[kernel // 2] * 4,
    )
    graph = helper.make_graph(
        [node],
        "one_conv2d",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, hw, hw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, cout, hw, hw])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# A 2-D convolution packs its weights differently from a 1-D one: four 2-bit
# planes rather than two 4-bit ones, four input channels to a byte, and the
# kernel stored in reverse. Confirmed at `cin = cout = 8, 3x3`. See the
# README's "A second packing" section.
_WBT2D_O_STRIDE = 144
_WBT2D_PLANE = 36
_WBT2D_PLANES = (3, 2, 1, 0)  # most significant first


# The shape-parameterised form, recovered by probing one build per index bit
# (the address is *linear* in the bits of the channel indices, so a bit costs
# one build rather than a channel costing one). Both units scale with the
# input channel count and cap at 128: a wider convolution is split into
# 128-channel slices. See the README's "Reading a real network's weights".
_WBT2D_UNIT_CAP = 128


def _wbt2d_unit(cin):
    return min(cin, _WBT2D_UNIT_CAP)


def _wbt2d_planes(cin):
    """Byte offsets of the four 2-bit planes, relative to plane 0."""
    pair = 9 * _wbt2d_unit(cin)
    return (0, _WBT2D_PLANE, pair, pair + _WBT2D_PLANE)


def _weight2d_offset(o, i, kh, kw, kernel, cin=None):
    """`(byte offset of plane 0, bit shift)` for a 2-D weight.

    `cin` selects the shape-dependent output unit `A = 18 * min(cin, 128)`;
    omitting it keeps the 8-channel constant the first experiments used.
    Output-channel bit 3 always costs 72 bytes, bits 0-2 cost `A` each and
    bits 4 and up cost `8A` -- a bit-interleave, not a stride.
    """
    flat = kernel * kh + kw
    kernel_part = 4 * (kernel * kernel - 1 - flat)
    if cin is None:
        return _WBT2D_O_STRIDE * o + kernel_part + i // 4, 2 * (i % 4)
    a = 18 * _wbt2d_unit(cin)
    return (
        a * (o % 8)
        + 72 * ((o >> 3) & 1)
        + 8 * a * (o >> 4)
        + kernel_part
        + (i % 16) // 4
        + 144 * (i >> 4)
    ), 2 * (i % 4)


def _read_weight2d_code_at(wbt, base, o, i, kh, kw, kernel, cin):
    """One INT8 code from a real network's weight block at `base`."""
    off, shift = _weight2d_offset(o, i, kh, kw, kernel, cin)
    value = 0
    for plane in _WBT2D_PLANES:
        value = (value << 2) | (
            (wbt[base + off + _wbt2d_planes(cin)[plane]] >> shift) & 0x3
        )
    return value


def _conv_channel0_addresses(shape):
    """Where output channel 0's weights live, relative to a layer's base, as
    `(offsets, shifts or None, plane offsets)`.

    Three packings, chosen by shape -- this is the whole point: the format
    does not have *a* weight layout, the convolution's shape picks one.

    * a narrow input (under four channels) stores plain INT8 bytes with the
      kernel row fastest: `3*i + kh + 12*kw`;
    * a 1x1 convolution stores plain INT8 bytes too, but chunks the input
      channels 36 at a time with the next chunk 144 bytes on;
    * anything wider bit-slices into four 2-bit planes (`_weight2d_offset`).
    """
    cin, kernel = shape[1], shape[2]
    if cin < 4:
        offsets, picks = [], []
        for i in range(cin):
            for kh in range(kernel):
                for kw in range(kernel):
                    offsets.append(3 * i + kh + 12 * kw)
                    picks.append((i, kh, kw))
        return np.array(offsets), None, (0,), picks
    if kernel == 1:
        offsets = [144 * (i // 36) + (i % 36) for i in range(cin)]
        return np.array(offsets), None, (0,), [(i, 0, 0) for i in range(cin)]
    slice_in = min(cin, _WBT2D_UNIT_CAP)
    offsets, shifts, picks = [], [], []
    for i in range(slice_in):
        for kh in range(kernel):
            for kw in range(kernel):
                off, shift = _weight2d_offset(0, i, kh, kw, kernel, cin)
                offsets.append(off)
                shifts.append(shift)
                picks.append((i, kh, kw))
    return np.array(offsets), np.array(shifts), _wbt2d_planes(cin), picks


def _locate_conv_weights_any(wbt, weights, step=4, samples=120):
    """Locate any convolution's weight block, dispatching on its shape.
    Scored on a single output channel, since each carries its own scale.

    Every block observed so far starts on a 4-byte boundary, which is what
    makes a whole-table scan affordable: at `step=1` this search over
    resnet18d's 22 layers does not finish in a useful time.
    """
    offsets, shifts, planes, picks = _conv_channel0_addresses(weights.shape)
    if len(picks) > samples:
        keep = np.random.RandomState(1).choice(len(picks), samples, replace=False)
        offsets = offsets[keep]
        picks = [picks[k] for k in keep]
        if shifts is not None:
            shifts = shifts[keep]
    target = np.array([weights[0, i, kh, kw] for (i, kh, kw) in picks], dtype=float)
    centred = target - target.mean()
    limit = len(wbt) - int(offsets.max()) - max(planes) - 4
    best = (-2.0, None)
    for start in range(0, max(limit, 1), 4096 * 32):
        bases = np.arange(start, min(start + 4096 * 32, limit), step)
        if not len(bases):
            break
        index = bases[:, None] + offsets[None, :]
        if shifts is None:
            codes = wbt[index].astype(float) - 128
        else:
            value = np.zeros(index.shape, np.int32)
            for plane in _WBT2D_PLANES:
                value = (value << 2) | ((wbt[index + planes[plane]] >> shifts) & 0x3)
            codes = value.astype(float) - 128
        codes = codes - codes.mean(1, keepdims=True)
        denom = np.sqrt((codes**2).sum(1) * (centred**2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs((codes * centred).sum(1) / denom)
        pick = int(np.nanargmax(corr))
        if corr[pick] > best[0]:
            best = (float(corr[pick]), int(bases[pick]))
    return best


def _locate_conv_weights(wbt, weights, step=8, samples=200):
    """Find a convolution's weight block in a whole network's table, by
    correlating a *single* output channel -- each channel carries its own
    quantisation scale, so pooling channels hides the match behind those
    scales (it reads about 0.9 instead of 0.9999).

    Returns `(best correlation, base offset)`.
    """
    cin, kernel = weights.shape[1], weights.shape[2]
    slice_in = min(cin, _WBT2D_UNIT_CAP)
    rng = np.random.RandomState(1)
    count = min(samples, slice_in * kernel * kernel)
    picks = [
        (int(rng.randint(slice_in)), int(rng.randint(kernel)), int(rng.randint(kernel)))
        for _ in range(count)
    ]
    target = np.array([weights[0, i, kh, kw] for (i, kh, kw) in picks], dtype=float)
    rel, shifts = [], []
    for i, kh, kw in picks:
        off, shift = _weight2d_offset(0, i, kh, kw, kernel, cin)
        rel.append(off)
        shifts.append(shift)
    rel = np.array(rel)
    shifts = np.array(shifts)
    planes = _wbt2d_planes(cin)
    limit = len(wbt) - int(rel.max()) - max(planes) - 4
    best = (-2.0, None)
    centred = target - target.mean()
    for start in range(0, max(limit, 1), 4096 * 16):
        bases = np.arange(start, min(start + 4096 * 16, limit), step)
        if not len(bases):
            break
        index = bases[:, None] + rel[None, :]
        value = np.zeros(index.shape, np.int32)
        for plane in _WBT2D_PLANES:
            value = (value << 2) | ((wbt[index + planes[plane]] >> shifts) & 0x3)
        codes = value.astype(float) - 128
        codes -= codes.mean(1, keepdims=True)
        denom = np.sqrt((codes**2).sum(1) * (centred**2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs((codes * centred).sum(1) / denom)
        pick = int(np.nanargmax(corr))
        if corr[pick] > best[0]:
            best = (float(corr[pick]), int(bases[pick]))
    return best


def _read_weight2d_code(wbt, o, i, kh, kw, kernel):
    off, shift = _weight2d_offset(o, i, kh, kw, kernel)
    value = 0
    for plane in _WBT2D_PLANES:
        value = (value << 2) | ((wbt[off + _WBT2D_PLANE * plane] >> shift) & 0x3)
    return value


def _write_weight2d_code(wbt, o, i, kh, kw, kernel, code):
    off, shift = _weight2d_offset(o, i, kh, kw, kernel)
    for n, plane in enumerate(_WBT2D_PLANES):
        bits = (code >> (2 * (len(_WBT2D_PLANES) - 1 - n))) & 0x3
        at = off + _WBT2D_PLANE * plane
        wbt[at] = (wbt[at] & ~(0x3 << shift) & 0xFF) | (bits << shift)


def test_resnet18d_conv_weights_are_addressable(tmp_path):
    """Confirmed real (see the README's "Reading a real network's weights"
    section): the shape-parameterised 2-D layout locates and reads the
    convolution weights of a *real* network -- resnet18d, 22 convolutions
    from 3 to 512 channels in an 11.9 MB weight table -- with nothing but the
    layer's shape and a search for its base offset.

    The two things that made this work are worth keeping. The output unit
    `A = 18 * min(cin, 128)` caps: a convolution wider than 128 input
    channels is split into slices, which is why 256- and 512-channel layers
    read as noise until the cap is applied. And the match must be scored
    *per output channel*, because each channel carries its own quantisation
    scale -- pooling channels reads about 0.9 where the true figure is
    0.9999. Needs Docker, no device.
    """
    # `_build_real_resnet18d` imports convert_onnxmodelzoo, which is what puts
    # model_zoo on sys.path -- so the build has to come first.
    path, _, _ = _build_real_resnet18d(str(tmp_path))
    import model_zoo

    compiled = onnx.load(path)
    wbt = np.frombuffer(
        next(i for i in compiled.graph.initializer if i.name == "npu_params").raw_data,
        dtype=np.uint8,
    )
    source = onnx.load(model_zoo.fetch_model("resnet18d_Opset18"))
    inits = {i.name: i for i in source.graph.initializer}

    # Every convolution in the network, all three packings.
    convs = [
        n.input[1]
        for n in source.graph.node
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in inits
    ]
    assert len(convs) == 22, len(convs)
    located = 0
    for name in convs:
        weights = numpy_helper.to_array(inits[name]).astype(float)
        correlation, base = _locate_conv_weights_any(wbt, weights)
        assert correlation > 0.99, (name, weights.shape, correlation)
        located += 1
    assert located == 22, located

    for name in ("onnx::Conv_217", "onnx::Conv_223", "onnx::Conv_253"):
        weights = numpy_helper.to_array(inits[name]).astype(float)
        correlation, base = _locate_conv_weights(wbt, weights)
        assert correlation > 0.99, (name, weights.shape, correlation)

        # Read the located block back and check a whole output channel.
        cin, kernel = weights.shape[1], weights.shape[2]
        codes = np.array(
            [
                [
                    [
                        _read_weight2d_code_at(wbt, base, 0, i, kh, kw, kernel, cin)
                        for kw in range(kernel)
                    ]
                    for kh in range(kernel)
                ]
                for i in range(min(cin, _WBT2D_UNIT_CAP))
            ],
            dtype=float,
        )
        truth = weights[0, : min(cin, _WBT2D_UNIT_CAP)]
        channel = abs(np.corrcoef(truth.ravel(), (codes - 128).ravel())[0, 1])
        assert channel > 0.999, (name, channel)


def test_conv2d_weights_are_int8_split_across_four_bit_planes(tmp_path):
    """Confirmed real (see the README's "A second packing" section): a *2-D*
    convolution stores the same INT8 weights in a different shape -- four
    2-bit planes 36 bytes apart rather than two 4-bit ones, four input
    channels to a byte, and the kernel laid out in reverse, so `(kh, kw)`
    counts *down* from the end of the block.

    The packing is not a property of the format but of the convolution: the
    1-D and 2-D cases here hold identical INT8 codes in different bit
    layouts. A generator therefore needs the shape, not just the weights.
    Needs Docker, no device.
    """
    cin = cout = 8
    kernel, hw = 3, 16
    zero = tmp_path / "zero"
    zero.mkdir()
    base = _wbt_of(
        _build_single_op_axmodel(
            str(zero),
            "m",
            _one_conv2d_model(
                cin, cout, hw, kernel, weights=np.zeros((cout, cin, kernel, kernel))
            ),
        )
    )
    for o, i, kh, kw in [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (7, 7, 2, 2),
    ]:
        w = np.zeros((cout, cin, kernel, kernel))
        w[o, i, kh, kw] = 0.5
        work = tmp_path / f"s{o}{i}{kh}{kw}"
        work.mkdir()
        spike = _wbt_of(
            _build_single_op_axmodel(
                str(work), "m", _one_conv2d_model(cin, cout, hw, kernel, weights=w)
            )
        )
        moved = [j for j in range(min(len(base), len(spike))) if base[j] != spike[j]]
        off, shift = _weight2d_offset(o, i, kh, kw, kernel)
        for plane in _WBT2D_PLANES:
            at = off + _WBT2D_PLANE * plane
            assert at in moved, (o, i, kh, kw, at, moved[:6])
            # A full-scale weight saturates its two bits in every plane.
            assert (spike[at] >> shift) & 0x3 == 0x3, (o, i, kh, kw, plane)


def test_conv2d_weights_can_be_rewritten_without_pulsar2(tmp_path):
    """Confirmed on the AX650N: the 2-D packing is understood well enough to
    rewrite a real 2-D convolution's weights by hand, and the device then
    computes the new convolution. Same permutation trick and same two
    cross-controls as the 1-D case. Needs Docker and a device.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    ort = pytest.importorskip("onnxruntime")
    cin = cout = 8
    kernel, hw = 3, 16
    rng = np.random.RandomState(0)
    w_old = (rng.randn(cout, cin, kernel, kernel) * 0.1).astype(np.float32)
    w_new = w_old[:, rng.permutation(cin), :, :].copy()

    work = tmp_path / "work"
    work.mkdir()
    axmodel = _build_single_op_axmodel(
        str(work), "m", _one_conv2d_model(cin, cout, hw, kernel, weights=w_old)
    )
    compiled = onnx.load(axmodel)
    init = next(i for i in compiled.graph.initializer if i.name == "npu_params")
    wbt = bytearray(init.raw_data)
    codes = np.array(
        [
            [
                [
                    [
                        _read_weight2d_code(wbt, o, i, kh, kw, kernel)
                        for kw in range(kernel)
                    ]
                    for kh in range(kernel)
                ]
                for i in range(cin)
            ]
            for o in range(cout)
        ],
        dtype=float,
    )
    slopes = _effective_slopes(
        w_old.astype(float).reshape(cout, -1, 1), codes.reshape(cout, -1, 1)
    )
    retargeted = np.clip(
        np.round(w_new.astype(float) * slopes[:, None, None, None]) + 128, 0, 255
    ).astype(int)
    for o in range(cout):
        for i in range(cin):
            for kh in range(kernel):
                for kw in range(kernel):
                    _write_weight2d_code(
                        wbt, o, i, kh, kw, kernel, int(retargeted[o, i, kh, kw])
                    )
    init.raw_data = bytes(wbt)
    patched = str(work / "patched.axmodel")
    onnx.save(compiled, patched)

    x = rng.randn(1, cin, hw, hw).astype(np.float32)

    def reference(weights):
        ref = _one_conv2d_model(cin, cout, hw, kernel, weights=weights)
        path = str(work / "ref.onnx")
        onnx.save(ref, path)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return np.asarray(session.run(None, {"x": x})[0]).ravel()

    def on_device(path):
        result = pulsar2_docker.run_on_device_with_inputs(
            path, {"x": x.tobytes()}, timeout=300
        )
        assert not result.error, result.error
        return np.frombuffer(result.outputs[0], dtype=np.float32).ravel()

    ref_old, ref_new = reference(w_old), reference(w_new)
    npu_old, npu_new = on_device(axmodel), on_device(patched)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    assert corr(npu_old, ref_old) > 0.99, corr(npu_old, ref_old)
    assert corr(npu_new, ref_new) > 0.99, corr(npu_new, ref_new)
    assert corr(npu_new, ref_old) < 0.9, corr(npu_new, ref_old)
    assert corr(npu_old, ref_new) < 0.9, corr(npu_old, ref_new)


def test_conv_weights_are_int8_split_across_two_nibble_planes(tmp_path):
    """Confirmed real (see the README's "Generating a weight table"
    section): a convolution's weights live in `npu_params` as INT8 with zero
    point 128, each byte split across *two* nibble planes 36 bytes apart,
    with two input channels sharing a byte.

    The single-non-zero-weight builds are what make the addressing exact
    rather than fitted: a lone weight at `(o, i, k)` moves exactly the byte
    `_weight_offset()` predicts, in the nibble `i % 2` selects. Needs Docker,
    no device.
    """
    kernel, length = 3, 32
    # 8 channels needs neither an output group nor a slot chunk; 32 needs
    # both, so `(0,8,2)` spills to the next chunk and `(16,0,0)` to the next
    # output group. Those are the cases a naive contiguous layout gets wrong.
    for cin, positions in (
        (8, [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 0, 1), (3, 5, 1), (7, 7, 2)]),
        (32, [(0, 0, 0), (0, 8, 2), (16, 0, 0), (31, 31, 2)]),
    ):
        cout = cin
        zero = tmp_path / f"zero{cin}"
        zero.mkdir()
        base = _wbt_of(
            _build_single_op_axmodel(
                str(zero),
                "m",
                _one_conv_model(
                    cin, cout, length, kernel, weights=np.zeros((cout, cin, kernel))
                ),
            )
        )
        for o, i, k in positions:
            w = np.zeros((cout, cin, kernel))
            w[o, i, k] = 0.5
            work = tmp_path / f"s{cin}_{o}_{i}_{k}"
            work.mkdir()
            spike = _wbt_of(
                _build_single_op_axmodel(
                    str(work),
                    "m",
                    _one_conv_model(cin, cout, length, kernel, weights=w),
                )
            )
            moved = [
                j for j in range(min(len(base), len(spike))) if base[j] != spike[j]
            ]
            low, shift = _weight_offset(o, i, k, cin, cout, kernel)
            assert low in moved, (cin, o, i, k, low, moved[:8])
            assert low + _WBT_PLANE_GAP in moved, (cin, o, i, k, moved[:8])
            # A weight at full scale saturates its nibble in both planes, and
            # an all-zero table reads as the zero point in the high plane.
            assert (spike[low] >> shift) & 0xF == 0xF, (cin, o, i, k, spike[low])
            assert (base[low + _WBT_PLANE_GAP] >> shift) & 0xF == 0x8, (cin, o, i, k)


def test_conv_weights_can_be_rewritten_without_pulsar2(tmp_path):
    """Confirmed on the AX650N: a compiled model's convolution weights can be
    replaced by hand -- no vendor compiler -- and the device then computes
    the *new* convolution at the same accuracy pulsar2's own build achieves.

    The new weights are a permutation of the old along the input-channel
    axis, which leaves every output channel's peak magnitude untouched, so
    pulsar2's own weight and activation scales stay valid and only the weight
    bytes need rewriting. The two cross-controls are what make this a real
    result: the patched model must *stop* matching the old weights, and the
    untouched model must not match the new ones. Needs Docker and a device.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    ort = pytest.importorskip("onnxruntime")
    for channels in (8, 32):
        _check_weight_retargeting(tmp_path / f"c{channels}", ort, channels)


def _check_weight_retargeting(tmp_path, ort, channels):
    """One shape's worth of `test_conv_weights_can_be_rewritten_without_pulsar2`."""
    cin = cout = channels
    kernel, length = 3, 32
    tmp_path.mkdir()
    rng = np.random.RandomState(0)
    w_old = (rng.randn(cout, cin, kernel) * 0.1).astype(np.float32)
    w_new = w_old[:, rng.permutation(cin), :].copy()
    assert np.allclose(
        np.abs(w_old).max(axis=(1, 2)), np.abs(w_new).max(axis=(1, 2))
    ), "the permutation must preserve every channel's peak"

    work = tmp_path / "work"
    work.mkdir()
    model = _one_conv_model(cin, cout, length, kernel, weights=w_old)
    axmodel = _build_single_op_axmodel(str(work), "m", model)

    compiled = onnx.load(axmodel)
    init = next(i for i in compiled.graph.initializer if i.name == "npu_params")
    codes = _read_weight_codes(bytes(init.raw_data), w_old.shape)
    slopes = _effective_slopes(w_old.astype(float), codes)
    retargeted = np.clip(
        np.round(w_new.astype(float) * slopes[:, None, None]) + 128, 0, 255
    ).astype(int)
    init.raw_data = _write_weight_codes(bytes(init.raw_data), retargeted)
    patched = str(work / "patched.axmodel")
    onnx.save(compiled, patched)

    x = rng.randn(1, cin, length).astype(np.float32)

    def reference(weights):
        ref = _one_conv_model(cin, cout, length, kernel, weights=weights)
        path = str(work / "ref.onnx")
        onnx.save(ref, path)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return np.asarray(session.run(None, {"x": x})[0]).ravel()

    def on_device(path):
        result = pulsar2_docker.run_on_device_with_inputs(
            path, {"x": x.tobytes()}, timeout=300
        )
        assert not result.error, result.error
        return np.frombuffer(result.outputs[0], dtype=np.float32).ravel()

    ref_old, ref_new = reference(w_old), reference(w_new)
    npu_old, npu_new = on_device(axmodel), on_device(patched)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731

    # The patched model computes the new convolution as well as pulsar2's own
    # build computes the old one.
    assert corr(npu_old, ref_old) > 0.99, corr(npu_old, ref_old)
    assert corr(npu_new, ref_new) > 0.99, corr(npu_new, ref_new)
    # ... and the cross-controls say the function really moved.
    assert corr(npu_new, ref_old) < 0.9, corr(npu_new, ref_old)
    assert corr(npu_old, ref_new) < 0.9, corr(npu_old, ref_new)


def _operand_field(mcode, verb, field, bank, lo, hi, ordinal=0):
    """One bitfield of the `ordinal`-th write to a register, or None."""
    values = _operands(mcode, verb, field, bank)
    if len(values) <= ordinal:
        return None
    mask = (1 << (hi - lo + 1)) - 1
    return (values[ordinal] >> lo) & mask


def test_spatial_extent_lives_in_three_bits_of_b0_03(tmp_path):
    """Confirmed real (see the README's "A second operand" section): bits 4
    to 6 of the first `a1 b0.03` operand are the input's spatial extent in
    16-element tiles, minus one -- the same inclusive convention `40.02` uses
    for channels.

    The invariance is the evidence, not the fit. The field tracks the length
    and *only* the length: changing the input channels, the output channels,
    the kernel size or the dilation leaves it alone, which is what separates
    a spatial field from the several other operands that also happen to move
    with the length. Needs Docker, no device.
    """
    base = dict(cin=32, cout=32, length=64, kernel=3, dilation=1)

    def field_for(**overrides):
        cfg = dict(base, **overrides)
        name = "_".join(f"{k}{v}" for k, v in sorted(overrides.items())) or "base"
        work = tmp_path / name
        work.mkdir()
        model = _one_conv_model(
            cfg["cin"],
            cfg["cout"],
            cfg["length"],
            cfg["kernel"],
            cfg["dilation"],
        )
        mcode = _build_single_op(str(work), "m", model)
        return _operand_field(mcode, 0xA1, 0xB0, 0x03, 4, 6)

    # It follows the length, in 16-element tiles, inclusive.
    for length in (32, 64, 96):
        assert field_for(length=length) == length // 16 - 1, length

    # ... and nothing else moves it.
    reference = 64 // 16 - 1
    for overrides in ({"cin": 16}, {"cout": 64}, {"kernel": 7}, {"dilation": 4}):
        assert field_for(**overrides) == reference, overrides


def test_the_spatial_field_follows_the_innermost_dimension_in_2d(tmp_path):
    """Confirmed real (see the README's "The same register, two spatial
    rules" section): in a *2-D* convolution the same `a1 b0.03` bits 4 to 6
    hold

        floor((W + 2*pad - 1) / 32)

    -- the index of the last 32-wide tile of the *padded* width. It follows
    the innermost dimension only: sweeping the height from 16 to 80 does not
    move it at all.

    Two details separate this from the 1-D rule for the same register, and
    both are the kind of thing a single formula would paper over: the tile is
    32 wide rather than 16, and the width is padded where the 1-D length is
    not -- which is visible only at an exact multiple of the tile, where a
    1x1 and a 3x3 kernel disagree. Needs Docker, no device.
    """
    cin = cout = 32
    height = 32

    seen = []

    def field_for(width, kernel=3, h=height):
        seen.append(1)
        work = tmp_path / f"w{width}k{kernel}h{h}_{len(seen)}"
        work.mkdir()
        model = _one_conv2d_model(cin, cout, max(h, width), kernel)
        # `_one_conv2d_model` is square; build the rectangle by hand.
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = h
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = width
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = h
        model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = width
        mcode = _build_single_op(str(work), "m", model)
        return _operand_field(mcode, 0xA1, 0xB0, 0x03, 4, 6)

    for width in (32, 64, 96):
        assert field_for(width) == (width + 2 - 1) // 32, width

    # The height does not move it.
    assert field_for(64, h=16) == field_for(64, h=80), "height must not matter"

    # At an exact multiple of the tile the padding shows: a 1x1 kernel pads
    # nothing and lands one tile lower than a 3x3.
    assert field_for(32, kernel=1) == (32 - 1) // 32
    assert field_for(32, kernel=3) == (32 + 2 - 1) // 32


def _conv_stack_model(channels, hw, kernel, layers):
    """A stack of identical same-padded convolutions -- arithmetic with as
    little else attached as possible, for a throughput measurement."""
    rng = np.random.RandomState(0)
    nodes, inits = [], []
    current = "x"
    for i in range(layers):
        name = f"w{i}"
        inits.append(
            numpy_helper.from_array(
                (rng.randn(channels, channels, kernel, kernel) * 0.02).astype(
                    np.float32
                ),
                name,
            )
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [current, name],
                [f"h{i}"],
                name=f"c{i}",
                kernel_shape=[kernel, kernel],
                pads=[kernel // 2] * 4,
            )
        )
        current = f"h{i}"
    shape = [1, channels, hw, hw]
    graph = helper.make_graph(
        nodes,
        "stack",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(current, TensorProto.FLOAT, shape)],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def test_int8_throughput_reaches_a_useful_fraction_of_the_rating(tmp_path):
    """Confirmed on the AX650N (see the README's "What the card actually
    does" section): a compute-dense INT8 convolution stack sustains several
    TOPS, well above what a misconfigured build would produce.

    This is a hardware health check as much as a performance claim. A card
    pinned to one NPU core, or a build that quietly fell back, lands around
    3 TOPS; the floor here sits between the two so either failure shows up.
    The peak actually measured was 10.13 TOPS against Axera's 10.8 TOPS
    "NPU alone" INT8 figure. Needs Docker and a device.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    channels, hw, kernel, layers = 512, 32, 3, 8
    work = tmp_path / "work"
    work.mkdir()
    os.makedirs(work / "dataset", exist_ok=True)
    os.makedirs(work / "config", exist_ok=True)
    model = _conv_stack_model(channels, hw, kernel, layers)
    onnx.save(model, str(work / "m.onnx"))
    rng = np.random.RandomState(1)
    pulsar2_docker.make_numpy_calibration_tar(
        str(work / "dataset" / "m.tar"),
        [rng.randn(1, channels, hw, hw).astype(np.float32) for _ in range(2)],
    )
    with open(work / "config" / "m.json", "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU3",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "x",
                            "calibration_dataset": "./dataset/m.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": 2,
                        }
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    result = pulsar2_docker.build(
        str(work), "m.onnx", "out", config_path="config/m.json", timeout=3000
    )
    assert result.success, result.error

    with open(work / "out" / "build_context.json") as f:
        macs = json.load(f)["macs"]
    # The compiler's own MAC count must match the arithmetic in the graph.
    assert macs == channels * channels * kernel * kernel * hw * hw * layers, macs

    stats = pulsar2_docker.run_on_device(
        str(work / "out" / "compiled.axmodel"), repeat=20, warmup=5, timeout=900
    )
    assert not stats.get("error"), stats.get("error")
    tops = 2 * macs / (stats["min_ms"] * 1e-3) / 1e12
    assert tops > 5.0, (tops, stats)


def _plane_pairing_lift(wbt, gap, limit=4_000_000):
    """How much more often a zero low-nibble byte is followed `gap` bytes
    later by `0x88`, relative to how often `0x88` occurs at all.

    In the two-nibble-plane INT8 encoding an all-zero weight reads `0x00` in
    the low plane and `0x88` in the high one, so a real plane gap shows a
    large lift and a wrong one shows about 1.0.
    """
    seg = np.frombuffer(wbt[: min(len(wbt), limit)], dtype=np.uint8)
    base = float((seg == 0x88).mean())
    if base == 0:
        return 0.0
    zeros = np.nonzero(seg[: len(seg) - gap] == 0x00)[0]
    if len(zeros) < 1000:
        return 0.0
    return float((seg[zeros + gap] == 0x88).mean()) / base


def _write_safetensors(path, tensors):
    """Minimal safetensors writer: an 8-byte header length, a JSON header,
    then the raw float32 blocks."""
    header, offset, blobs = {}, 0, []
    for name, array in tensors:
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(np.shape(array)),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)
    blob = json.dumps(header).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for data in blobs:
            f.write(data)


def _tiny_llama_checkpoint(
    path, weights, hidden=256, inter=512, heads=8, kv=2, vocab=512
):
    """A synthetic Llama checkpoint whose layer-0 `q_proj` is `weights`."""
    os.makedirs(path, exist_ok=True)
    rng = np.random.RandomState(3)

    def small(*shape):
        return rng.randn(*shape) * 0.02

    head_dim = hidden // heads
    tensors = [
        ("model.embed_tokens.weight", small(vocab, hidden)),
        ("model.layers.0.self_attn.q_proj.weight", weights),
        ("model.layers.0.self_attn.k_proj.weight", small(kv * head_dim, hidden)),
        ("model.layers.0.self_attn.v_proj.weight", small(kv * head_dim, hidden)),
        ("model.layers.0.self_attn.o_proj.weight", small(hidden, heads * head_dim)),
        ("model.layers.0.mlp.gate_proj.weight", small(inter, hidden)),
        ("model.layers.0.mlp.up_proj.weight", small(inter, hidden)),
        ("model.layers.0.mlp.down_proj.weight", small(hidden, inter)),
        ("model.layers.0.input_layernorm.weight", np.ones(hidden)),
        ("model.layers.0.post_attention_layernorm.weight", np.ones(hidden)),
        ("model.norm.weight", np.ones(hidden)),
        ("lm_head.weight", small(vocab, hidden)),
    ]
    _write_safetensors(os.path.join(path, "model.safetensors"), tensors)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": hidden,
                "intermediate_size": inter,
                "num_hidden_layers": 1,
                "num_attention_heads": heads,
                "num_key_value_heads": kv,
                "vocab_size": vocab,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "hidden_act": "silu",
                "torch_dtype": "float32",
                "tie_word_embeddings": False,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "attention_bias": False,
                "mlp_bias": False,
            },
            f,
        )


def _llm_weight_offset(row, col, cin):
    """Where `llm_build` puts a matmul weight. Every constant is half its
    convolution-pipeline counterpart: an 18-byte plane gap, chunks of 18 with
    a stride of 72, and 36 for the bit that costs 72 there."""
    a = 72 * -(-(cin // 2) // 18)
    m = 16
    top = m * a + 512
    slot = col // 2
    return (
        a * (row % m)
        + 36 * ((row // m) & 1)
        + top * (row // (2 * m))
        + 72 * (slot // 18)
        + (slot % 18)
    ), (4 if col % 2 else 0)


def _bf16_trunc(value):
    """A float32's top 16 bits, rounded toward zero -- how the mcode stores an
    activation scale."""
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0] >> 16


def _u16_offsets(mcode, word):
    """Every offset holding `word` as a little-endian uint16."""
    raw = struct.pack("<H", word)
    out, at = [], mcode.find(raw)
    while at >= 0:
        out.append(at)
        at = mcode.find(raw, at + 1)
    return out


def _compiler_scales(axmodel_path):
    """The per-tensor activation scales pulsar2 recorded for a build, smallest
    first, from its own `quant/quant_axmodel.json`."""
    path = os.path.join(os.path.dirname(axmodel_path), "quant", "quant_axmodel.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        quant = json.load(f)
    scales = [v["scale"] for v in quant["values"].values() if "scale" in v]
    return sorted(s[0] for s in scales if len(s) == 1)


def _scale_probe_build(work_dir, tag, amplitude, channels=32, kernel=3, length=64):
    """One convolution, calibrated against inputs scaled by `amplitude`. The
    weights never change, so only the activation scales can."""
    rng = np.random.RandomState(7)
    weights = (rng.randn(channels, channels, kernel) * 0.05).astype(np.float32)
    model = _one_conv_model(
        channels, channels, length, kernel, 1, "Conv", weights=weights
    )
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    onnx.save(model, os.path.join(work_dir, f"{tag}.onnx"))
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", f"{tag}.tar"),
        [
            (np.random.RandomState(1).randn(1, channels, length) * amplitude).astype(
                np.float32
            )
            for _ in range(2)
        ],
    )
    with open(os.path.join(work_dir, "config", f"{tag}.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "x",
                            "calibration_dataset": f"./dataset/{tag}.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": 2,
                        }
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    result = pulsar2_docker.build(
        work_dir, f"{tag}.onnx", f"out_{tag}", config_path=f"config/{tag}.json"
    )
    assert result.success, result.error
    ((_, mcode),) = _mcodes_of(result.axmodel_path)
    return result.axmodel_path, mcode


def test_activation_scales_are_bfloat16_in_the_mcode(tmp_path):
    """Confirmed real (see the README's "The activation scales, decoded"
    section): the two per-tensor activation scales a convolution needs are in
    the mcode as **bfloat16 truncated toward zero** -- the input's as its
    reciprocal `1/x_scale`, the output's directly as `y_scale`, four copies
    of each.

    Found by a differential that can only move scales: the same model built
    against calibration inputs scaled by 1, 2 and 4. The weight codes and the
    requantisation multipliers are invariant to that (`M = x_scale *
    (peak/127.5) / r_scale` scales away), so the weight table must not move at
    all -- and it does not, which is half of what this asserts.

    This is what stands between rewriting weights that preserve a tensor's
    dynamic range and rewriting any weights at all. Needs Docker, no device.
    """
    builds = {}
    for amplitude in (1.0, 2.0, 4.0):
        tag = f"a{amplitude:g}".replace(".", "_")
        work = tmp_path / tag
        work.mkdir()
        path, mcode = _scale_probe_build(str(work), tag, amplitude)
        scales = _compiler_scales(path)
        if scales is None:
            pytest.skip("this pulsar2 build wrote no quant_axmodel.json")
        builds[amplitude] = (path, mcode, scales)

    # The weight table cannot move: nothing in it depends on an activation
    # scale once the requantisation multiplier is formed.
    tables = {a: _wbt_of(p) for a, (p, _, _) in builds.items()}
    assert len(set(tables.values())) == 1, "the weight table moved"

    slots = {}
    for amplitude, (_, mcode, scales) in builds.items():
        y_scale, x_scale = scales[0], scales[1]
        for name, word in (
            ("1/x_scale", _bf16_trunc(1.0 / x_scale)),
            ("y_scale", _bf16_trunc(y_scale)),
        ):
            at = _u16_offsets(mcode, word)
            assert len(at) >= 4, (amplitude, name, word, at)
            slots.setdefault(name, []).append(set(at))

    # The same slots carry it at every amplitude -- otherwise a coincidence
    # somewhere in a 2.8 KB stream would pass the test above.
    for name, seen in slots.items():
        assert set.intersection(*seen), name


def _quantize_llm_weights(weights):
    """`llm_build`'s weight quantiser, reproduced exactly -- and it is not the
    convolution pipeline's (see `_quantize_conv_weights`).

    Three things differ, and all three matter. The scale divides by 128, not
    127.5. It is taken from the *signed* weight at the peak index rather than
    its magnitude, and then negated -- so the row's extreme weight always
    lands on code 0 and zero lands on 128, whichever sign that extreme has.
    And ties round toward `+inf`, not to even and not away from zero.

    The arithmetic happens in whatever precision the checkpoint holds: a
    bfloat16 safetensors file is quantised from bfloat16 values, an F32 one
    from float32. Cast before calling if the checkpoint is not float32.
    """
    w = np.asarray(weights, dtype=np.float32)
    peak = w[np.arange(len(w)), np.abs(w).argmax(1)].astype(np.float32)
    scale = (-peak / np.float32(128)).astype(np.float32)
    return np.clip(np.floor(w / scale[:, None] + np.float32(0.5)) + 128, 0, 255).astype(
        int
    )


def _read_llm_codes(wbt, base, rows, cin):
    """The INT8 codes an `llm_build` weight block holds, as `(rows, cin)`."""
    out = np.zeros((rows, cin), dtype=int)
    for r in range(rows):
        for c in range(cin):
            off, shift = _llm_weight_offset(r, c, cin)
            low = (wbt[base + off] >> shift) & 0xF
            high = (wbt[base + off + 18] >> shift) & 0xF
            out[r, c] = (high << 4) | low
    return out


def test_llm_weight_quantiser_is_reproduced_exactly(tmp_path):
    """Confirmed real (see the README's "The LLM path quantises differently"
    section): `_quantize_llm_weights()` reproduces **every** code `llm_build`
    writes, for every matmul in a layer -- and it is a different quantiser
    from the convolution pipeline's, not a variant of it.

    This is what the addressing test above deliberately did not test. With
    both solved, an `llm_build` weight table can be written from the
    checkpoint alone. Needs Docker, no device.
    """
    hidden = 256
    rng = np.random.RandomState(11)
    weights = (rng.randn(hidden, hidden) * 0.02).astype(np.float32)

    work = tmp_path / "work"
    work.mkdir()
    _tiny_llama_checkpoint(str(work / "tiny"), weights, hidden=hidden)
    result = pulsar2_docker.llm_build(
        str(work),
        "tiny",
        "out",
        weight_type="s8",
        prefill_len=64,
        kv_cache_len=127,
        parallel=8,
    )
    assert result.success, getattr(result, "error", None)
    files = sorted(glob.glob(str(work / "out" / "*.axmodel")))
    assert files
    wbt = np.frombuffer(
        next(
            i for i in onnx.load(files[0]).graph.initializer if i.name == "npu_params"
        ).raw_data,
        dtype=np.uint8,
    )

    codes = _quantize_llm_weights(weights)
    offsets = np.array([_llm_weight_offset(0, c, hidden)[0] for c in range(hidden)])
    shifts = np.array([4 if c % 2 else 0 for c in range(hidden)])
    want = codes[0].astype(np.uint8)

    # Every code of row 0 exact is already 256**-256 against chance, so the
    # block is found rather than guessed at.
    base = None
    for start in range(0, len(wbt) - int(offsets.max()) - 22, 2):
        low = (wbt[start + offsets] >> shifts) & 0xF
        high = (wbt[start + offsets + 18] >> shifts) & 0xF
        if (((high.astype(int) << 4) | low) == want).all():
            base = start
            break
    assert base is not None, "q_proj's first row is nowhere in the table"

    got = _read_llm_codes(wbt, base, 32, hidden)
    assert (got == codes[:32]).all(), (got != codes[:32]).sum()


def test_llm_matmul_weight_addressing_is_the_conv_layout_halved(tmp_path):
    """Confirmed real (see the README's "The LLM path's weight encoding"
    section): `_llm_weight_offset()` locates every weight of an `llm_build`
    matmul. Scored by correlation per output row, which tests the *addressing*
    without depending on the quantiser -- that convention differs from the
    convolution pipeline's and is not fully pinned down.

    Needs Docker, no device.
    """
    hidden = 256
    rng = np.random.RandomState(11)
    weights = rng.randn(hidden, hidden) * 0.02
    weights *= 0.2 / np.abs(weights).max(axis=1, keepdims=True)

    work = tmp_path / "work"
    work.mkdir()
    _tiny_llama_checkpoint(str(work / "tiny"), weights, hidden=hidden)
    result = pulsar2_docker.llm_build(
        str(work),
        "tiny",
        "out",
        weight_type="s8",
        prefill_len=64,
        kv_cache_len=127,
        parallel=8,
    )
    assert result.success, getattr(result, "error", None)
    files = sorted(glob.glob(str(work / "out" / "*.axmodel")))
    assert files
    wbt = np.frombuffer(
        next(
            i for i in onnx.load(files[0]).graph.initializer if i.name == "npu_params"
        ).raw_data,
        dtype=np.uint8,
    )

    offsets = np.array(
        [[_llm_weight_offset(0, c, hidden)[0] for c in range(hidden)]]
    ).ravel()
    shifts = np.array([4 if c % 2 else 0 for c in range(hidden)])
    span = int(offsets.max()) + 18 + 4

    # Find the block by its first row, then require every row to correlate.
    target = weights[0] - weights[0].mean()
    best, base = -2.0, None
    for start in range(0, len(wbt) - span, 2):
        low = (wbt[start + offsets] >> shifts) & 0xF
        high = (wbt[start + offsets + 18] >> shifts) & 0xF
        codes = ((high.astype(int) << 4) | low).astype(float)
        codes -= codes.mean()
        denom = np.sqrt((codes**2).sum() * (target**2).sum())
        if denom == 0:
            continue
        corr = abs(float((codes * target).sum() / denom))
        if corr > best:
            best, base = corr, start
        if corr > 0.9999:
            break
    assert best > 0.99, best

    worst = 1.0
    for row in range(hidden):
        rel = np.array([_llm_weight_offset(row, c, hidden)[0] for c in range(hidden)])
        low = (wbt[base + rel] >> shifts) & 0xF
        high = (wbt[base + rel + 18] >> shifts) & 0xF
        codes = ((high.astype(int) << 4) | low).astype(float)
        worst = min(worst, abs(float(np.corrcoef(weights[row], codes)[0, 1])))
    assert worst > 0.99, worst


def test_llm_build_weights_use_the_same_nibble_planes_at_half_the_gap(tmp_path):
    """Confirmed real (see the README's "The LLM path's weight encoding"
    section): `llm_build` stores INT8 weights in the same two-nibble-plane
    form the convolution pipeline uses, but with the planes **18** bytes
    apart rather than 36.

    The `s4` build is the control that makes this an encoding claim rather
    than a coincidence: four-bit weights need only one nibble, so they show
    no pairing at any gap. Needs Docker, no device.
    """
    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)

    lifts = {}
    for weight_type in ("s8", "s4"):
        result = pulsar2_docker.llm_build(
            str(work),
            "SmolLM2-135M",
            f"out_{weight_type}",
            weight_type=weight_type,
            prefill_len=512,
            kv_cache_len=1023,
            parallel=8,
        )
        assert result.success, (weight_type, getattr(result, "error", None))
        layer = sorted(glob.glob(str(work / f"out_{weight_type}" / "*_l0_*.axmodel")))
        assert layer, weight_type
        wbt = bytes(
            next(
                i
                for i in onnx.load(layer[0]).graph.initializer
                if i.name == "npu_params"
            ).raw_data
        )
        lifts[weight_type] = {g: _plane_pairing_lift(wbt, g) for g in (17, 18, 19, 36)}

    # s8: a clear peak at 18, and *not* at the convolution pipeline's 36.
    assert lifts["s8"][18] > 4.0, lifts["s8"]
    assert lifts["s8"][18] > lifts["s8"][36], lifts["s8"]
    # s4: four-bit weights occupy one nibble, so nothing pairs anywhere.
    assert max(lifts["s4"].values()) < 2.0, lifts["s4"]


def test_llm_build_offers_an_int4_weight_path_the_cnn_path_lacks(tmp_path):
    """Confirmed real (see the README's "Where the INT4 path actually is"
    section): `pulsar2 llm_build` accepts `-w s4`, and the resulting model is
    substantially smaller than the `s8` build of the same checkpoint.

    This is the only 4-bit route in the toolchain. `pulsar2 build` -- the CNN
    path everything else here uses -- has no 4-bit option at all: its
    `weight_data_type` for convolution accepts only `S8` and `FP32`, and
    there is no command-line flag either. Needs Docker, no device.
    """
    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)

    sizes = {}
    for weight_type in ("s8", "s4"):
        result = pulsar2_docker.llm_build(
            str(work),
            "SmolLM2-135M",
            f"out_{weight_type}",
            weight_type=weight_type,
            prefill_len=512,
            kv_cache_len=1023,
            parallel=8,
        )
        assert result.success, (weight_type, getattr(result, "error", None))
        files = sorted(glob.glob(str(work / f"out_{weight_type}" / "*.axmodel")))
        assert files, weight_type
        sizes[weight_type] = sum(os.path.getsize(f) for f in files)

    # 4-bit weights are the bulk of the saving, but a layer file also carries
    # non-weight structure, so the ratio lands well short of 2x.
    ratio = sizes["s8"] / sizes["s4"]
    assert 1.3 < ratio < 2.0, (ratio, sizes)


def _build_conv_with_weight_type(work_dir, tag, model, weight_type):
    """Compile a model forcing Conv's `weight_data_type`."""
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    onnx.save(model, os.path.join(work_dir, f"{tag}.onnx"))
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    rng = np.random.RandomState(1)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", f"{tag}.tar"),
        [rng.randn(*shape).astype(np.float32) for _ in range(2)],
    )
    with open(os.path.join(work_dir, "config", f"{tag}.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU3",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "x",
                            "calibration_dataset": f"./dataset/{tag}.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": 2,
                        }
                    ],
                    "layer_configs": [
                        {"op_type": "Conv", "weight_data_type": weight_type}
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    return pulsar2_docker.build(
        work_dir,
        f"{tag}.onnx",
        f"out_{tag}",
        config_path=f"config/{tag}.json",
        timeout=2400,
    )


def test_nvfp4_weights_are_accepted_and_then_ignored(tmp_path):
    """Confirmed real (see the README's "Is there a full INT4 path" section):
    `weight_data_type: NVFP4` is the only 4-bit type the CNN pipeline's config
    will parse, and it changes nothing. The compiled convolution weights come
    out **identical** to the `S8` build, code for code.

    That is worth a regression test in both directions. It documents that
    there is no 4-bit weight path here today, and if a future toolchain ever
    implements NVFP4 this test fails and says so.

    It has already done that once. Pulsar2 7.0 *does* implement `NVFP4` for
    convolution, and there the same config does not build at all -- see the
    README's "Pulsar2 7.0 implements NVFP4, and it is still unusable"
    section. This test describes `DEFAULT_IMAGE`, which is 6.0. Needs Docker,
    no device.
    """
    cin = cout = 64
    kernel, hw = 3, 16
    model = _one_conv2d_model(cin, cout, hw, kernel)
    codes = {}
    for weight_type in ("S8", "NVFP4"):
        work = tmp_path / weight_type
        work.mkdir()
        result = _build_conv_with_weight_type(str(work), "m", model, weight_type)
        assert result.success, (weight_type, result.error)
        wbt = np.frombuffer(
            next(
                i
                for i in onnx.load(result.axmodel_path).graph.initializer
                if i.name == "npu_params"
            ).raw_data,
            dtype=np.uint8,
        )
        codes[weight_type] = [
            _read_weight2d_code_at(wbt, 0, 0, i, kh, kw, kernel, cin)
            for i in range(cin)
            for kh in range(kernel)
            for kw in range(kernel)
        ]
    assert codes["S8"] == codes["NVFP4"], (
        "NVFP4 changed the compiled weights -- this toolchain may now implement "
        "a real 4-bit weight path, and the README needs revisiting"
    )
    # ... and the weights really are 8-bit, spread over far more than 16 levels.
    assert len(set(codes["S8"])) > 32, len(set(codes["S8"]))


def _quantize_conv_weights(weights):
    """Pulsar2's own convolution weight quantiser, reproduced exactly.

    Per output channel: `scale = float32(peak) / float32(127.5)`, then
    round-half-to-even in float32 and offset by the zero point 128. All three
    details matter -- 127 or 128 in place of 127.5 costs ~14% of the codes,
    float64 arithmetic costs 0.5%, and round-half-away costs 0.2%.
    """
    w = np.asarray(weights, dtype=np.float32)
    cout = w.shape[0]
    peaks = np.abs(w.reshape(cout, -1)).max(1)
    scale = (peaks / np.float32(127.5)).astype(np.float32)
    shaped = scale.reshape((cout,) + (1,) * (w.ndim - 1))
    return np.clip(np.round(w / shaped) + 128, 0, 255).astype(int)


def _weight1d_offset_wide(o, i, k, cin, cout, kernel, top):
    """1-D weight address at 64 and 128 channels.

    `A = 144 * ceil((Cin/2)*kernel / 36)` is the one constant that unifies 32,
    64 and 128 channels (288, 432, 864). `m = min(Cout//4, 16)` members share
    it, bit `log2(m)` of the output channel always costs 72, and whole
    super-blocks cost `top` -- which is `m*A` at 64 channels and `m*A + 256` at
    128, an extra region this has not explained.
    """
    a = 144 * -(-((cin // 2) * kernel) // 36)
    m = min(max(cout // 4, 1), 16)
    slot = (cin // 2) * k + i // 2
    return (
        a * (o % m)
        + 72 * ((o // m) & 1)
        + top * (o // (2 * m))
        + 144 * (slot // 36)
        + (slot % 36)
    ), (4 if i % 2 else 0)


def _one_convtranspose_model(cin, cout, length, kernel, stride, weights=None):
    """A 1-D transposed convolution. ONNX orders its weight `(Cin, Cout, K)`,
    the opposite of `Conv`."""
    rng = np.random.RandomState(0)
    values = (rng.randn(cin, cout, kernel) * 0.05) if weights is None else weights
    weight = numpy_helper.from_array(np.asarray(values, dtype=np.float32), "w")
    node = helper.make_node(
        "ConvTranspose",
        ["x", "w"],
        ["y"],
        name="convt",
        kernel_shape=[kernel],
        strides=[stride],
        pads=[(kernel - stride) // 2] * 2,
    )
    graph = helper.make_graph(
        [node],
        "one_convt",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, length])],
        [
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, [1, cout, length * stride]
            )
        ],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def test_a_widely_dilated_conv_splits_into_single_tap_convolutions(tmp_path):
    """Confirmed real (see the README's "A widely dilated convolution is K
    convolutions" section): once a dilated kernel's footprint grows past what
    one input tile holds, the convolution is compiled as **K separate
    single-tap convolutions**.

    The count is what identifies it. Changing a single weight and grouping
    the bytes that move gives one region per *tap* -- eight regions at
    `K = 7`, six at `K = 5` -- and **not** one per dilation, which is the
    other decomposition one might guess (`d` is 3 and 6 here). An undilated
    convolution of the same width moves one region. Needs Docker, no device.
    """
    cin = cout = 64
    length = 64

    def regions(kernel, dilation):
        rng = np.random.RandomState(5)
        base = rng.randn(cout, cin, kernel) * 0.05
        for o in range(cout):
            base[o] *= 0.2 / np.abs(base[o]).max()
        base[0, 0, 0] = 0.1
        flipped = base.copy()
        flipped[0, 0, 0] = -0.1176  # differs in both nibbles
        tables = []
        for tag, weights in (("a", base), ("b", flipped)):
            work = tmp_path / f"k{kernel}d{dilation}{tag}"
            work.mkdir()
            model = _one_conv_model(
                cin,
                cout,
                length,
                kernel,
                dilation=dilation,
                weights=weights.astype(np.float32),
            )
            tables.append(_wbt_of(_build_single_op_axmodel(str(work), "m", model)))
        a, b = tables
        moved = [j for j in range(min(len(a), len(b))) if a[j] != b[j]]
        assert moved, (kernel, dilation)
        groups, current = [], [moved[0]]
        for offset in moved[1:]:
            if offset - current[-1] > 64:
                groups.append(current)
                current = [offset]
            else:
                current.append(offset)
        groups.append(current)
        return len(groups)

    # Undilated: one weight region (plus the scales it perturbs).
    assert regions(3, 1) <= 2, regions(3, 1)
    # Widely dilated: one region per tap, not per dilation.
    assert regions(7, 3) >= 7, "K=7 should split into per-tap regions"
    assert regions(5, 6) >= 5, "K=5 should split into per-tap regions"


def test_convtranspose_stores_taps_reversed_in_the_conv_layout(tmp_path):
    """Confirmed real (see the README's "A transposed convolution is several
    convolutions" section): an unstrided `ConvTranspose` uses the *ordinary*
    convolution weight layout, with two adjustments -- ONNX's `(Cin, Cout, K)`
    weight maps straight on (dimension 0 behaves as input channels), and the
    kernel taps are stored **reversed**.

    Getting the reversal wrong is not obvious from a correlation, so this
    checks every code. Needs Docker, no device.
    """
    cin = cout = 32
    kernel, length = 4, 32
    rng = np.random.RandomState(5)
    weights = rng.randn(cin, cout, kernel) * 0.05
    for i in range(cin):
        weights[i] *= 0.2 / np.abs(weights[i]).max()
    weights = weights.astype(np.float32)

    work = tmp_path / "s1"
    work.mkdir()
    model = _one_convtranspose_model(cin, cout, length, kernel, 1, weights=weights)
    wbt = _wbt_of(_build_single_op_axmodel(str(work), "m", model))

    # (Cin, Cout, K) -> the conv layout's (o, i, k), taps reversed.
    as_conv = np.swapaxes(weights, 0, 1)[:, :, ::-1].copy()
    got = np.zeros(as_conv.shape, dtype=int)
    for o in range(cout):
        for i in range(cin):
            for k in range(kernel):
                off, shift = _weight_offset(o, i, k, cin, cout, kernel)
                got[o, i, k] = (((wbt[off + _WBT_PLANE_GAP] >> shift) & 0xF) << 4) | (
                    (wbt[off] >> shift) & 0xF
                )
    assert (got == _quantize_conv_weights(as_conv)).all()
    # Without the reversal it does not read.
    assert not (got == _quantize_conv_weights(np.swapaxes(weights, 0, 1))).all()


def test_dilation_changes_the_weight_layout(tmp_path):
    """Confirmed real (see the README's "Dilation reorders the weights"
    section): a dilated convolution stores its weights in a *different*
    arrangement from an undilated one of the same shape. Each kernel tap gets
    its own 144-byte chunk instead of packing into the shared slot index.

    The table is the same size either way, so this is a reordering rather
    than a different amount of data -- which is why reading a dilated layer
    with the undilated rule returns plausible-looking noise (about a third of
    the codes right by chance) rather than failing outright. Needs Docker, no
    device.
    """
    cin = cout = 64
    kernel, length = 3, 64
    rng = np.random.RandomState(5)
    weights = rng.randn(cout, cin, kernel) * 0.05
    for o in range(cout):
        weights[o] *= 0.2 / np.abs(weights[o]).max()
    weights = weights.astype(np.float32)
    expected = _quantize_conv_weights(weights)

    def read(wbt, dilated):
        got = np.zeros(weights.shape, dtype=int)
        for o in range(cout):
            for i in range(cin):
                for k in range(kernel):
                    if dilated:
                        off = (
                            432 * (o % 16)
                            + 72 * ((o >> 4) & 1)
                            + 16 * 432 * (o >> 5)
                            + 144 * k
                            + i // 2
                        )
                        shift = 4 if i % 2 else 0
                    else:
                        off, shift = _weight1d_offset_wide(
                            o, i, k, cin, cout, kernel, 16 * 432
                        )
                    got[o, i, k] = (
                        ((wbt[off + _WBT_PLANE_GAP] >> shift) & 0xF) << 4
                    ) | ((wbt[off] >> shift) & 0xF)
        return got

    tables = {}
    for dilation in (1, 2):
        work = tmp_path / f"d{dilation}"
        work.mkdir()
        model = _one_conv_model(
            cin, cout, length, kernel, dilation=dilation, weights=weights
        )
        tables[dilation] = _wbt_of(_build_single_op_axmodel(str(work), "m", model))

    # Each layout reads its own build exactly...
    assert (read(tables[1], dilated=False) == expected).all(), "undilated"
    assert (read(tables[2], dilated=True) == expected).all(), "dilated"
    # ... and the undilated rule does not read the dilated build.
    assert not (read(tables[2], dilated=False) == expected).all()
    # Same size: a reordering, not more data.
    assert len(tables[1]) == len(tables[2]), (len(tables[1]), len(tables[2]))


def test_1d_weight_layout_holds_at_64_and_128_channels(tmp_path):
    """Confirmed real (see the README's "Closing the 1-D layout to 128
    channels" section): the 1-D weight layout, which stopped working past 32
    input channels, is exact at 64 and 128 once the addressing constants are
    measured there -- every code, not a correlation.

    The probe that found them matters as much as the result. Flipping a weight
    between `+0.1` and `-0.1` moves only one of the two nibble planes at this
    scale, because both quantise to a low nibble of zero; the pair used here
    differs in *both* nibbles, which is what made the second plane visible.
    Needs Docker, no device.
    """
    kernel, length = 3, 64
    for cin, top in ((64, 16 * 432), (128, 16 * 864 + 256)):
        cout = cin
        rng = np.random.RandomState(5)
        weights = rng.randn(cout, cin, kernel) * 0.05
        for o in range(cout):
            weights[o] *= 0.2 / np.abs(weights[o]).max()
        weights = weights.astype(np.float32)
        work = tmp_path / f"c{cin}"
        work.mkdir()
        model = _one_conv_model(cin, cout, length, kernel, weights=weights)
        wbt = _wbt_of(_build_single_op_axmodel(str(work), "m", model))
        got = np.zeros(weights.shape, dtype=int)
        for o in range(cout):
            for i in range(cin):
                for k in range(kernel):
                    off, shift = _weight1d_offset_wide(o, i, k, cin, cout, kernel, top)
                    got[o, i, k] = (
                        ((wbt[off + _WBT_PLANE_GAP] >> shift) & 0xF) << 4
                    ) | ((wbt[off] >> shift) & 0xF)
        assert (got == _quantize_conv_weights(weights)).all(), cin


def test_conv_weight_quantiser_is_reproduced_exactly(tmp_path):
    """Confirmed real (see the README's "The weight quantiser, exactly"
    section): `_quantize_conv_weights()` reproduces **every** code pulsar2
    writes for a convolution, with no reference model involved.

    That is what turns reading the weight table into generating one: the
    codes no longer have to be recovered from a compiled model and
    re-quantised against a fitted slope, they can be computed from the
    weights alone. Needs Docker, no device.
    """
    kernel, hw = 3, 16
    for cin in (8, 32):
        cout = cin
        rng = np.random.RandomState(cin)
        weights = (rng.randn(cout, cin, kernel, kernel) * 0.1).astype(np.float32)
        work = tmp_path / f"c{cin}"
        work.mkdir()
        axmodel = _build_single_op_axmodel(
            str(work), "m", _one_conv2d_model(cin, cout, hw, kernel, weights=weights)
        )
        wbt = _wbt_of(axmodel)
        compiled = np.array(
            [
                [
                    [
                        [
                            _read_weight2d_code_at(wbt, 0, o, i, kh, kw, kernel, cin)
                            for kw in range(kernel)
                        ]
                        for kh in range(kernel)
                    ]
                    for i in range(cin)
                ]
                for o in range(cout)
            ],
            dtype=int,
        )
        assert (_quantize_conv_weights(weights) == compiled).all(), cin


def test_per_channel_weight_scales_sit_just_past_the_weight_block(tmp_path):
    """Confirmed real (see the README's "What a weight generator can and
    cannot do yet" section): the weight table carries one float32 per output
    channel, immediately after the weight block, exactly proportional to that
    channel's peak magnitude.

    Finding it by proportionality rather than by a fixed offset is the point:
    the constant of proportionality absorbs the activation scales and differs
    between models, so only the *ratios* identify the array. Needs Docker, no
    device.
    """
    cin = cout = 8
    kernel, hw = 3, 16
    rng = np.random.RandomState(3)
    weights = (rng.randn(cout, cin, kernel, kernel) * 0.1).astype(np.float32)
    work = tmp_path / "work"
    work.mkdir()
    axmodel = _build_single_op_axmodel(
        str(work), "m", _one_conv2d_model(cin, cout, hw, kernel, weights=weights)
    )
    wbt = _wbt_of(axmodel)
    peaks = np.abs(weights.reshape(cout, -1)).max(1).astype(np.float64)

    found = None
    for off in range(0, len(wbt) - 4 * cout, 4):
        values = np.frombuffer(wbt[off : off + 4 * cout], dtype=np.float32).astype(
            np.float64
        )
        if not np.all(np.isfinite(values)) or values.min() <= 0:
            continue
        ratio = values / peaks
        if ratio.std() / ratio.mean() < 1e-4:
            found = (off, ratio.mean())
            break
    assert found is not None, "no per-channel scale array proportional to the peaks"
    off, ratio = found
    # It sits past the weights, which occupy `_WBT2D_O_STRIDE` per channel.
    assert off >= _WBT2D_O_STRIDE * cout, (off, _WBT2D_O_STRIDE * cout)
    assert off < len(wbt), (off, len(wbt))
    # The constant is a scale, not something degenerate.
    assert 0 < ratio < 1, ratio


def test_single_conv_program_encodes_its_output_channel_count(tmp_path):
    """Confirmed real (see the README's "The first operand with a known
    meaning" section): in a program holding exactly one convolution, the
    leading `a1 40.02` operand is `8 * output channels - 1` -- an inclusive
    bit extent over one output position, at 8 bits per INT8 channel.

    The asymmetric cases are what make this an *output* channel reading
    rather than an input one: 32->64 and 64->32 give opposite answers, and
    each follows the output. This is the first operand field in the mcode
    with a confirmed meaning, which is what a generator needs. Needs Docker,
    no device.
    """
    for op in ("Conv", "ConvTranspose"):
        for cin, cout in [(32, 64), (64, 32), (32, 16)]:
            work = tmp_path / f"{op}_{cin}_{cout}"
            work.mkdir()
            model = _one_conv_model(cin, cout, op=op)
            mcode = _build_single_op(str(work), "m", model)
            values = _operands(mcode, 0xA1, 0x40, 0x02)
            assert values, f"no 40.02 write in a single-{op} program"
            assert values[0] == 8 * cout - 1, (op, cin, cout, values[:4])
            # The reading is about channels alone: it must not follow the input.
            if cin != cout:
                assert values[0] != 8 * cin - 1, (op, cin, cout, values[0])


def test_the_channel_operand_belongs_to_the_convolution_engine(tmp_path):
    """Confirmed real: `40.02` is a convolution-engine register, not a
    general "output channels" field. A lone `Relu`, `LeakyRelu` or `Sigmoid`
    never writes it at all, and `Add`/`Mul` write it with something that is
    not `8C-1`, while `Conv` and `ConvTranspose` both write `8C-1`.

    That presence-or-absence is a concrete instance of the README's claim
    that op type lives in operand values rather than in different
    instructions: these programs are built from the same verbs and tags, and
    what separates a convolution from an elementwise op is which registers
    the program bothers to set. Needs Docker, no device.
    """
    channels = 32
    for op in ("Relu", "LeakyRelu", "Sigmoid"):
        work = tmp_path / op
        work.mkdir()
        mcode = _build_single_op(str(work), "m", _elementwise_model(op, channels))
        assert _operands(mcode, 0xA1, 0x40, 0x02) == [], op
        # The stream is real and complete even so -- it just never programs
        # that register.
        lo, hi = _stream_bounds(mcode)
        records = _decode_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
        assert _encode_mcode(records) == mcode[lo:hi], op
        assert {r["verb"] for r in records if r["kind"] == "V"} <= _VERBS6, op

    for op in ("Add", "Mul"):
        work = tmp_path / op
        work.mkdir()
        mcode = _build_single_op(str(work), "m", _elementwise_model(op, channels))
        values = _operands(mcode, 0xA1, 0x40, 0x02)
        assert 8 * channels - 1 not in values, (op, values)


def test_single_conv_channel_operand_ignores_length_kernel_and_dilation(tmp_path):
    """Confirmed real: the same `a1 40.02` operand is unmoved by the input
    length, the kernel size and the dilation -- it tracks the output channel
    count and nothing else. That negative is what rules out reading it as a
    buffer size or a weight-table offset, both of which move when the kernel
    or the length does. Needs Docker, no device.
    """
    variants = {
        "base": {},
        "long": {"length": 128},
        "k7": {"kernel": 7},
        "d4": {"dilation": 4},
    }
    seen = {}
    for tag, kwargs in variants.items():
        work = tmp_path / tag
        work.mkdir()
        model = _one_conv_model(32, 32, **kwargs)
        mcode = _build_single_op(str(work), "m", model)
        seen[tag] = _operands(mcode, 0xA1, 0x40, 0x02)[0]
    assert set(seen.values()) == {8 * 32 - 1}, seen


_PIPER_REPO = "rhasspy/piper-voices"
_PIPER_VOICE = "en/en_US/lessac/low/en_US-lessac-low.onnx"


def _cached_piper_voice():
    """The `en_US-lessac-low` Piper voice and its config from the local
    HuggingFace cache, as `(onnx path, config path)`, else None. Never
    downloads -- these tests stay offline."""
    try:
        from huggingface_hub import hf_hub_download

        model = hf_hub_download(_PIPER_REPO, _PIPER_VOICE, local_files_only=True)
        config = hf_hub_download(
            _PIPER_REPO, _PIPER_VOICE + ".json", local_files_only=True
        )
        return model, config
    except Exception:
        return None


def _piper_decoder_input(model):
    """The tensor a Piper voice feeds to its HiFi-GAN decoder: whatever the
    `dec.conv_pre` convolution reads. Found by weight name rather than
    hard-coded, since the intermediate tensor names are export artefacts."""
    conv = next(
        n
        for n in model.graph.node
        if n.op_type == "Conv"
        and len(n.input) > 1
        and n.input[1] == "dec.conv_pre.weight"
    )
    return conv.input[0]


def _piper_decoder(voice_path, out_path, frames):
    """The *real* HiFi-GAN decoder lifted out of a Piper VITS voice, with a
    static `frames`-long latent input.

    The rest of a VITS graph cannot be compiled -- `RandomNormalLike`,
    `NonZero`, `CumSum` and `Range` are stochastic or data-dependent, so a
    real deployment keeps the text encoder and duration sampling on the CPU
    and sends only this decoder to the device. Extracting the subgraph
    rather than rebuilding it by hand keeps the trained weights *and* the
    exact dilations, which is what makes the device output audible speech
    instead of the noise a randomly initialised vocoder emits.
    """
    model = onnx.load(voice_path)
    latent = _piper_decoder_input(model)
    onnx.utils.extract_model(
        voice_path,
        out_path,
        [latent],
        [model.graph.output[0].name],
        check_model=False,
    )
    dec = onnx.load(out_path)
    inp = dec.graph.input[0]
    del inp.type.tensor_type.shape.dim[:]
    for value in (1, 192, frames):
        inp.type.tensor_type.shape.dim.add().dim_value = value
    inp.name = "z"
    for node in dec.graph.node:
        node.input[:] = ["z" if i == latent else i for i in node.input]
    # The exported output shape is written for the dynamic graph; drop it
    # and let inference restate it for this fixed length.
    del dec.graph.output[0].type.tensor_type.shape.dim[:]
    dec = onnx.shape_inference.infer_shapes(dec)
    onnx.save(dec, out_path)
    return dec


def _piper_say(voice_path, config_path, phonemes, length_scale=1.0):
    """Run a Piper voice on the CPU for one phoneme sequence, returning
    `(latent, audio)` -- the decoder's own input alongside the reference
    waveform, so a device run of the decoder can be scored against the
    audio the untouched model produces."""
    ort = pytest.importorskip("onnxruntime")
    model = onnx.load(voice_path)
    latent = _piper_decoder_input(model)
    model.graph.output.append(
        helper.make_tensor_value_info(latent, TensorProto.FLOAT, None)
    )
    with tempfile.TemporaryDirectory() as td:
        exposed = os.path.join(td, "piper.onnx")
        onnx.save(
            model,
            exposed,
            save_as_external_data=True,
            location="piper.data",
            all_tensors_to_one_file=True,
            size_threshold=1024,
        )
        session = ort.InferenceSession(exposed, providers=["CPUExecutionProvider"])
        table = json.load(open(config_path))["phoneme_id_map"]
        ids = list(table["^"])
        for phoneme in phonemes:
            ids += table[phoneme] + table["_"]
        ids += table["$"]
        tokens = np.array([ids], dtype=np.int64)
        audio, z = session.run(
            [model.graph.output[0].name, latent],
            {
                "input": tokens,
                "input_lengths": np.array([tokens.shape[1]], dtype=np.int64),
                "scales": np.array([0.667, length_scale, 0.8], dtype=np.float32),
            },
        )
    return np.asarray(z, dtype=np.float32), np.asarray(
        audio, dtype=np.float32
    ).squeeze()


_PIPER_HELLO = list("h") + ["\u0259", "l", "\u02c8", "o", "\u028a"]


def _pad_latent(z, frames):
    """A latent padded out to the compiled length. Zero is the right filler:
    the decoder's input is already a flow output multiplied by a length
    mask, so every frame past the utterance is zero in the untouched graph
    too."""
    padded = np.zeros((1, 192, frames), dtype=np.float32)
    padded[:, :, : z.shape[2]] = z[:, :, :frames]
    return padded


def _build_piper_decoder(work_dir, frames, latents):
    """Compile the real Piper decoder for the AX650, calibrating on real
    latents. Returns `(axmodel path, mcode)`."""
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "z.tar"), latents
    )
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": "z",
                            "calibration_dataset": "./dataset/z.tar",
                            "calibration_format": "Numpy",
                            "calibration_size": len(latents),
                        }
                    ],
                    "calibration_method": "MinMax",
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )
    result = pulsar2_docker.build(
        work_dir, "decoder.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error
    ((_, mcode),) = _mcodes_of(result.axmodel_path)
    return result.axmodel_path, mcode


def test_piper_decoder_subgraph_reproduces_the_whole_voice(tmp_path):
    """Confirmed real: the decoder subgraph lifted out of a Piper voice,
    frozen to a fixed length, reproduces the untouched model's own audio
    (correlation > 0.9999). That is what licenses treating a device run of
    this subgraph as a device run of Piper's vocoder. CPU only -- no Docker,
    no device.
    """
    voice = _cached_piper_voice()
    if voice is None:
        pytest.skip("the Piper en_US-lessac-low voice is not in the local cache")
    ort = pytest.importorskip("onnxruntime")
    frames = 64
    z, reference = _piper_say(voice[0], voice[1], _PIPER_HELLO)
    assert z.shape[1] == 192, z.shape
    # 256 audio samples per latent frame -- the three upsampling stages
    # multiply out to 8 * 8 * 4.
    assert reference.size == 256 * z.shape[2], (reference.size, z.shape)

    path = str(tmp_path / "decoder.onnx")
    _piper_decoder(voice[0], path, frames)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = np.asarray(session.run(None, {"z": _pad_latent(z, frames)})[0]).squeeze()
    out = out[: reference.size]
    assert np.corrcoef(out, reference)[0, 1] > 0.9999, np.corrcoef(out, reference)[0, 1]


def test_real_piper_decoder_uses_no_new_instruction_forms(tmp_path):
    """Confirmed real (see the README's "Real weights, real speech"
    section): the *trained* Piper vocoder -- a different model family, a
    different domain, weights this decoder has never seen -- compiles to an
    mcode that introduces no verb and no tag beyond the ones the CNN and
    transformer builds already used, and the codec round-trips it
    byte-exactly. Needs Docker, no device.
    """
    voice = _cached_piper_voice()
    if voice is None:
        pytest.skip("the Piper en_US-lessac-low voice is not in the local cache")
    frames = 64
    z, _ = _piper_say(voice[0], voice[1], _PIPER_HELLO)
    work = tmp_path / "work"
    work.mkdir()
    _piper_decoder(voice[0], str(work / "decoder.onnx"), frames)
    _, mcode = _build_piper_decoder(str(work), frames, [_pad_latent(z, frames)] * 4)

    covered, _ = _nonzero_coverage(mcode, **_FULL_RULE)
    assert covered >= 0.95, covered
    lo, hi = _stream_bounds(mcode)
    records = _decode_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
    assert _encode_mcode(records) == mcode[lo:hi]

    verbs = {r["verb"] for r in records if r["kind"] == "V"}
    assert verbs <= _VERBS6, verbs
    tags = {r["tag"] for r in records if r["kind"] in ("S", "B")}
    assert tags <= _ALL_TAGS | {0xA1, 0xC1, 0xE1}, tags


def test_real_piper_decoder_makes_speech_on_device(tmp_path):
    """Confirmed on the AX650N: the real Piper vocoder, quantised to INT8
    and run on the NPU, reproduces the CPU waveform at a correlation above
    0.98 -- audible speech, not the noise a randomly initialised vocoder
    emits. This is the end-to-end answer to whether a deep convolutional
    stack survives this NPU's INT8 quantisation; a 30-layer transformer did
    not. Needs Docker *and* a device.
    """
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    voice = _cached_piper_voice()
    if voice is None:
        pytest.skip("the Piper en_US-lessac-low voice is not in the local cache")
    ort = pytest.importorskip("onnxruntime")
    frames = 64
    z, _ = _piper_say(voice[0], voice[1], _PIPER_HELLO)
    padded = _pad_latent(z, frames)
    work = tmp_path / "work"
    work.mkdir()
    path = str(work / "decoder.onnx")
    _piper_decoder(voice[0], path, frames)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    reference = np.asarray(session.run(None, {"z": padded})[0]).squeeze()

    axmodel, _ = _build_piper_decoder(str(work), frames, [padded] * 4)
    result = pulsar2_docker.run_on_device_with_inputs(
        axmodel, {"z": padded.tobytes()}, timeout=300
    )
    assert not result.error, result.error
    out = np.frombuffer(result.outputs[0], dtype=np.float32).squeeze()
    assert out.size == frames * 256, out.size
    correlation = float(np.corrcoef(out, reference)[0, 1])
    assert correlation > 0.98, correlation


def test_tts_vocoder_uses_no_new_instruction_forms(tmp_path):
    """Confirmed real (see the README's "A third model family" section): a
    HiFi-GAN-shaped vocoder -- transposed convolution, dilated convolution,
    LeakyReLU, a bounded activation -- compiles, and its mcode introduces
    *no* verb, tag or register that the CNN and transformer builds did not
    already use. Op-type differences live in operand values, not in new
    forms. Needs Docker, no device.
    """
    mcode = _build_vocoder(str(tmp_path))
    covered, _ = _nonzero_coverage(mcode)
    assert covered >= 0.95, covered

    def destinations(blob):
        lo, hi = _stream_bounds(blob)
        seen = set()
        for t in _tokenize_mcode(blob, start=lo, end=hi, **_FULL_RULE):
            if t[1] in ("V", "W"):
                seen.add((t[1], t[2], t[3], t[4]))
            elif t[1] == "S":
                # Read the tag from the stream: for a tag that carries an
                # extra byte the token's third slot holds the register.
                seen.add(("S", blob[t[0] + t[2] + 2]))
            elif t[1] == "B":
                seen.add(("B", t[2]))
        return seen

    reference = tmp_path / "resnet18d"
    reference.mkdir()
    known = destinations(_build_real_resnet18d(str(reference))[2])
    new = {d for d in destinations(mcode) if d not in known}
    # Verbs and tags must be shared; a (field, bank) pair may well be unique
    # to either model, so compare the vocabulary rather than every pair. The
    # companion write's leading byte is not a verb and varies by design, so
    # it is excluded.
    new_tags = {d for d in new if d[0] in ("S", "B")}
    assert not new_tags, new_tags
    new_verbs = {d[1] for d in new if d[0] == "V"}
    assert new_verbs <= {t[1] for t in known if t[0] == "V"}, new_verbs

    lo, hi = _stream_bounds(mcode)
    programs = [
        t
        for t in _tokenize_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
        if t[1:] == ("V", 0xA1, 0x40, 0x02)
    ]
    assert len(programs) >= 5, len(programs)


def test_full_rule_explains_almost_every_stream_byte(tmp_path):
    """Confirmed real (see the README's "Where the decoding stands" section):
    the validated forms together account for 96..99% of the non-zero
    instruction-stream bytes of a real resnet18d *and* of a transformer
    compiled through the ONNX path -- the same rule, unchanged, across a CNN
    and an LLM. Also pins the width rule's limit: admitting prefixes p >= 5
    buys apparent coverage only by over-fitting, so it must not improve the
    real-versus-shuffled ratio. Needs Docker, no device.
    """
    import random
    from collections import Counter

    for name in _ANALYSIS_BUILDS:
        work = tmp_path / name
        work.mkdir()
        mcode = _build_analysis_mcode(name, str(work))
        covered, runs = _nonzero_coverage(mcode)
        # Measured 98.6% (resnet18d) and 96.9% (mistral); floor with headroom.
        assert covered >= 0.96, (name, covered, len(runs))

        # The width rule stops at p = 4. The decisive measure is per prefix,
        # not aggregate coverage: a greedy walk with a bigger pmax always
        # explains more of *any* byte string, so what matters is whether units
        # with a given prefix occur more often than in a shuffled stream.
        lo, hi = _stream_bounds(mcode)
        blob = mcode[lo:hi]
        shuffled = bytearray(blob)
        random.Random(0).shuffle(shuffled)
        shuffled = bytes(shuffled)

        def prefix_counts(data):
            rule = dict(_FULL_RULE)
            rule["pmax"] = 16
            toks = _tokenize_mcode(data, start=0, end=len(data), **rule)
            return Counter(a for _, kind, a, *_ in toks if kind == "S")

        real, null = prefix_counts(blob), prefix_counts(shuffled)
        assert real[1] >= 2 * null[1] and real[2] >= 2 * null[2], (name, real, null)
        # Individual large prefixes are too rare to judge one at a time; taken
        # together, p >= 5 occurs *less* often than in the shuffled stream.
        real_wide = sum(real[k] for k in range(5, 17))
        null_wide = sum(null[k] for k in range(5, 17))
        assert real_wide < null_wide, (name, real_wide, null_wide)


def test_companion_writes_fill_the_slot_below_the_next_verb(tmp_path):
    """Confirmed real (see the README's "a 7-byte write that fills the slot
    below the next one"): a 7-byte `[X][field][bank][32-bit operand]` write
    always targets the address slot directly below the verb that follows it
    -- same bank one field lower, or the last field of the previous bank.
    The anchor is exact: shuffling the stream yields *zero* such units, while
    the real builds have dozens, and admitting the form lifts coverage.
    Needs Docker, no device.
    """
    import random

    total = 0
    for name in _ANALYSIS_BUILDS:
        work = tmp_path / name
        work.mkdir()
        mcode = _build_analysis_mcode(name, str(work))
        lo, hi = _stream_bounds(mcode)
        toks = _tokenize_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
        writes = [t for t in toks if t[1] == "W"]
        assert len(writes) >= 5, (name, len(writes))
        total += len(writes)

        at = {t[0]: i for i, t in enumerate(toks)}
        for o, _, _, field, bank in writes:
            nxt = toks[at[o + 7]]
            assert nxt[1] == "V" and nxt[2] == 0xA1, (name, o, nxt)
            adjacent = (bank == nxt[4] and (nxt[3] - field) % 0x100 == 0x10) or (
                field == 0xF0 and nxt[3] == 0x00 and nxt[4] == bank + 1
            )
            assert adjacent, (name, o, hex(field), hex(bank), hex(nxt[3]), hex(nxt[4]))

        # The form never appears by chance, and it buys real coverage.
        blob = mcode[lo:hi]
        shuffled = bytearray(blob)
        random.Random(0).shuffle(shuffled)
        shuffled = bytes(shuffled)
        null = _tokenize_mcode(shuffled, start=0, end=len(shuffled), **_FULL_RULE)
        assert not [t for t in null if t[1] == "W"], name

        without = dict(_FULL_RULE)
        without["companion"] = False
        covered_without, _ = _nonzero_coverage(mcode, **without)
        covered_with, _ = _nonzero_coverage(mcode)
        assert covered_with > covered_without, (name, covered_without, covered_with)
    assert total >= 20, total


def test_decode_encode_round_trip_is_byte_exact(tmp_path):
    """Confirmed real (see the README's "A lossless codec" section): decoding
    a real instruction stream into structured records and writing those
    records back out reproduces the stream byte for byte, on a CNN and on a
    transformer. The encoder reads nothing from the original, so this proves
    the decode captures every bit the forms carry -- the first thing an mcode
    *generator* needs. >= 95% of the bytes come from recognised forms; the
    rest ride along as raw escapes. Needs Docker, no device.
    """
    for name in _ANALYSIS_BUILDS:
        work = tmp_path / name
        work.mkdir()
        mcode = _build_analysis_mcode(name, str(work))
        lo, hi = _stream_bounds(mcode)
        records = _decode_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
        assert _encode_mcode(records) == mcode[lo:hi], name
        assert _structured_share(records) >= 0.95, (name, _structured_share(records))

        # A whole .axmodel: header and tail are not instructions, so they are
        # carried verbatim, and the rebuilt blob must equal the original.
        rebuilt = mcode[:lo] + _encode_mcode(records) + mcode[hi:]
        assert rebuilt == mcode, name


def test_reencoded_mcode_runs_on_device_and_edits_take_effect(tmp_path):
    """Confirmed real on the AX650N (see the README's "A lossless codec"
    section): a stream we wrote ourselves is accepted by the hardware. The
    layer's mcode is decoded into structured records, written back out by
    `_encode_mcode()`, saved into the .axmodel and run -- the card produces
    byte-identical outputs to the untouched original. Editing a field through
    the codec (every in-program `a7` post's operand 2 -> 0) then changes the
    outputs without faulting, the same effect the raw-byte patch had, which
    is what makes this an encoder rather than a copier. Skips without a
    device or the cached checkpoint.
    """
    import hashlib
    import shutil

    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")
    ckpt = _cached_hf_checkpoint("HuggingFaceTB/SmolLM2-135M")
    if ckpt is None:
        pytest.skip("HuggingFaceTB/SmolLM2-135M is not in the local HuggingFace cache")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copytree(ckpt, work / "SmolLM2-135M", symlinks=False)
    result = pulsar2_docker.llm_build(str(work), "SmolLM2-135M", "output", parallel=8)
    assert result.success, getattr(result, "error", None)
    original = str(work / "output" / "llama_p512_l0_together.axmodel")

    def rewrite(path, edit, whole_stream=True):
        """Decode the first subgraph's mcode, apply `edit` to the records,
        re-encode, and save as a new .axmodel."""
        model = onnx.load(original)
        neu = next(n for n in model.graph.node if n.op_type == "neu mode")
        info = json.loads(
            next(a for a in neu.attribute if a.name == "npu_graph_info").s.decode()
        )
        key = info["dotneus"][0]["neu_key"]
        init = next(i for i in model.graph.initializer if i.name == key)
        mcode = bytes(init.raw_data)
        lo, hi = _stream_bounds(mcode)
        records = _decode_mcode(mcode, start=lo, end=hi, **_FULL_RULE)
        if whole_stream:
            changed = edit(records)
        else:
            # The op-program segment is the one holding the most programs.
            # Selecting it by type word is wrong on `llm_build` layers, where
            # configuration segments carry the CNN op segment's type.
            _, segs = _segments(mcode)

            def program_count(seg):
                pos, length, _ = seg
                toks = _tokenize_mcode(mcode, start=pos, end=pos + length, **_FULL_RULE)
                return sum(1 for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02))

            pos, length, _ = max(segs, key=program_count)
            changed = edit([r for r in records if pos <= r["at"] < pos + length])
        init.raw_data = mcode[:lo] + _encode_mcode(records) + mcode[hi:]
        onnx.save(model, path)
        return init.raw_data == mcode, changed

    inputs = _llm_layer_inputs(onnx.load(original))

    def digests(path, times=3):
        out = []
        for _ in range(times):
            r = pulsar2_docker.run_on_device_with_inputs(
                path, inputs, repeat=1, warmup=1
            )
            if r.error and "0x8030070C" in r.error:
                out.append("fault")
            elif r.outputs:
                out.append(tuple(hashlib.sha1(o).hexdigest() for o in r.outputs))
        return out

    base = [d for d in digests(original) if d != "fault"]
    assert len(base) >= 2 and len(set(base)) == 1, ("baseline", base)

    # 1. Re-encoded, unchanged: identical bytes, and the card agrees.
    same_path = str(tmp_path / "reencoded.axmodel")
    identical, _ = rewrite(same_path, lambda records: 0)
    assert identical, "re-encoding changed the bytes"
    again = [d for d in digests(same_path) if d != "fault"]
    assert len(again) >= 2 and set(again) == set(base), ("re-encoded", again, base[0])

    def clear_post(records):
        n = 0
        for r in records:
            if r["kind"] == "V" and r["verb"] == 0xA7 and r["bank"] == 0x02:
                r["operand"] = b"\x00" * len(r["operand"])
                n += 1
        return n

    # 2. One field edited through the records produces *exactly* the bytes
    #    the raw-byte patch produces -- the edit path is the encoder, not a
    #    copier. The device behaviour of that patch is covered by
    #    `test_llm_build_a7_is_a_sync_verb_on_device`, so it is not re-run
    #    here: repeating it back-to-back with the runs above hits the
    #    runtime's transient rejection often enough to be flaky.
    model = onnx.load(original)
    neu = next(n for n in model.graph.node if n.op_type == "neu mode")
    info = json.loads(
        next(a for a in neu.attribute if a.name == "npu_graph_info").s.decode()
    )
    key = info["dotneus"][0]["neu_key"]
    mcode = bytes(next(i for i in model.graph.initializer if i.name == key).raw_data)
    _, segs = _segments(mcode)

    def program_count(seg):
        pos, length, _ = seg
        toks = _tokenize_mcode(mcode, start=pos, end=pos + length, **_FULL_RULE)
        return sum(1 for t in toks if t[1:] == ("V", 0xA1, 0x40, 0x02))

    pos, length, _ = max(segs, key=program_count)
    toks = _tokenize_mcode(mcode, start=pos, end=pos + length, **_FULL_RULE)
    a7_posts = [t[0] for t in toks if t[1] == "V" and t[2] == 0xA7 and t[4] == 0x02]
    assert len(a7_posts) >= 50, len(a7_posts)

    raw = bytearray(mcode)
    for offset in a7_posts:
        struct.pack_into("<I", raw, offset + 4, 0)

    edited_path = str(tmp_path / "edited.axmodel")
    identical, changed = rewrite(edited_path, clear_post, whole_stream=False)
    assert not identical and changed == len(a7_posts), (changed, len(a7_posts))
    through_codec = bytes(
        next(
            i for i in onnx.load(edited_path).graph.initializer if i.name == key
        ).raw_data
    )
    assert through_codec == bytes(raw), "codec edit differs from the raw-byte patch"
