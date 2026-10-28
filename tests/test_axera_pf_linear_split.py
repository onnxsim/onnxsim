"""Host-only checks of the linear weight-table width rule of the template registry
(``pf_registry.linear_split`` / ``safe_widths``), the failed-build skip in ``scan``, and the
presence check (``linear_emit.table_in_params``, ``linear_table_present``). No Pulsar2, no
card, no checkpoint: the compiled model is a small ``npu_params`` holder written here.
"""

import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import linear_emit as le  # noqa: E402
import pf_registry as R  # noqa: E402

MAN = {
    "linear_widths": {
        "2560": {
            "measured": {
                "3072": {"present": True}, "4096": {"present": True},
                "4864": {"present": True}, "5456": {"present": False},
                "5888": {"present": True}, "5952": {"present": False},
                "6144": {"present": False},
                "9728": {"present": False},
            },
        },
    },
}  # fmt: skip


def test_safe_widths_are_the_measured_present_ones():
    assert R.safe_widths(MAN, 2560) == [3072, 4096, 4864, 5888]
    assert R.safe_widths(MAN, 1024) == []


def test_width_not_measured_present_is_split_into_equal_measured_parts():
    p = R.linear_split(9728, 2560, MAN)
    assert p == {
        "parts": 2,
        "k_part": 4864,
        "names": ["k4864n2560"] * 2,
        "add": "add2560",
    }
    # 6144 is measured absent; its smallest equal split into present widths is 2 x 3072
    assert R.linear_split(6144, 2560, MAN)["k_part"] == 3072


def test_absence_is_not_monotone_in_k():
    # 5456 is absent but 5888 (above it) is present: 5888 is used as measured, 5456 is not
    assert R.linear_split(5888, 2560, MAN) is None
    with pytest.raises(ValueError):
        R.linear_split(
            5456, 2560, MAN
        )  # 2 x 2728 and 4 x 1364 are not measured present
    # a larger width that passed does not stand in for an unmeasured smaller one either:
    # 5000 (unmeasured) is not used, and no equal split of it into present parts exists
    with pytest.raises(ValueError):
        R.linear_split(5000, 2560, MAN)
    # an unmeasured width above a present one is not assumed safe: 2 x 5888 is measured
    # present, so 11776 is planned as 2 x 5888 rather than used as is
    assert R.linear_split(11776, 2560, MAN)["k_part"] == 5888


def test_measured_present_and_unmeasured_N_are_not_split():
    assert R.linear_split(4864, 2560, MAN) is None
    assert R.linear_split(3072, 1024, MAN) is None  # output width not measured at all


def test_qwen3_4b_down_projection_is_planned_as_two_halves():
    man = {"linear_widths": MAN["linear_widths"]}
    lines = R.split_report(R.SPECS["qwen3_4b"], 8, man)
    assert "k9728n2560: split 2 x k4864n2560 + add2560" in lines


def _root_with(tmp_path, ok):
    import json

    d = tmp_path / "linear" / "k4864n2560"
    d.mkdir(parents=True)
    (d / "signature.json").write_text(json.dumps({"sig_id": "abc"}))
    if ok is not None:
        (d / "result.json").write_text(json.dumps({"ok": ok}))
    return R.Root(str(tmp_path), None)


def test_scan_skips_a_template_whose_build_failed(tmp_path):
    ok_root, none_root, bad_root = tmp_path / "ok", tmp_path / "none", tmp_path / "bad"
    for d in (ok_root, none_root, bad_root):
        d.mkdir()
    assert "abc" in R.scan([_root_with(ok_root, True)])  # a passing build is a template
    assert "abc" in R.scan([_root_with(none_root, None)])  # no result file: a template
    assert "abc" not in R.scan([_root_with(bad_root, False)])  # a failed build is not


def test_table_in_params_counts_exact_occurrences():
    assert le.table_in_params(b"xxabcxx", b"abc")
    assert not le.table_in_params(b"abcabc", b"abc")  # two copies: not a single table
    assert not le.table_in_params(b"xyz", b"abc")
    assert not le.table_in_params(b"abc", b"")


def test_linear_table_present_on_a_compiled_holder(tmp_path):
    k, n = 64, 32
    rng = np.random.default_rng(3)
    w = rng.uniform(-0.1, 0.1, (k, n)).astype(np.float32)
    s_x, zp_x, s_y, zp_y = 0.01, 128, 0.02, 130
    table = le.linear_params_u16(b"", w, s_x, zp_x, s_y, zp_y, table_only=True)
    # a compiled-model holder: the npu_params initializer carries raw bytes, which the text
    # format cannot spell (CLAUDE.md's exception), so it is built with onnx.helper
    tail = b"\x00" * 40
    blob = table + tail
    t = onnx.TensorProto(name="npu_params", data_type=onnx.TensorProto.UINT8,
                         dims=[len(blob)], raw_data=blob)  # fmt: skip
    g = onnx.helper.make_graph([], "c", [], [], initializer=[t])
    path = str(tmp_path / "compiled.axmodel")
    onnx.save(onnx.helper.make_model(g), path)
    assert le.linear_table_present(path, w, s_x, zp_x, s_y, zp_y)
    assert not le.linear_table_present(path, w * 2.0, s_x, zp_x, s_y, zp_y)
