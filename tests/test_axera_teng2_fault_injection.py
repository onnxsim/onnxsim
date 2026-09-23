"""Regression coverage for the teng2 fault-injection harness's pure logic
(patching, offset selection, classification) -- no Docker/device required."""

import os
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import teng2_fault_injection as tfi  # noqa: E402


def test_load_reference_and_segment2_bounds():
    model, mc = tfi.load_reference()
    assert model.graph.node[0].op_type == "neu mode"
    start, end = tfi.segment2_bounds(mc)
    # From the earlier interactive check: segment 2 is [380, 1788), content
    # runs through byte 1785 (two trailing zero padding bytes).
    assert start == 380
    assert end == 1786


def test_candidate_offsets_are_within_segment2_and_sorted():
    _, mc = tfi.load_reference()
    _, segs = tfi.mcode.segments(mc)
    pos, length, _ = segs[2]
    offsets = tfi.candidate_offsets(mc)
    assert offsets == sorted(offsets)
    assert len(offsets) > 100
    # within the segment's real (untrimmed) span; see candidates()'s
    # docstring for why one boundary record is excluded rather than clipped
    assert all(pos <= o < pos + length for o in offsets)


def test_candidates_test_every_byte_of_v_records():
    _, mc = tfi.load_reference()
    cands = tfi.candidates(mc)
    v_cands = [c for c in cands if c["kind"] == "V"]
    assert v_cands, "reference should contain at least one V record"
    by_record: dict[int, list[dict]] = {}
    for c in v_cands:
        by_record.setdefault(c["record_at"], []).append(c)
    # every V record contributes verb/unit/reg_lo/reg_hi plus 3-4 value bytes
    for record_at, group in by_record.items():
        assert len(group) in (7, 8), (record_at, len(group))
        assert len(group) == len(set(c["offset"] for c in group))
        roles = {c["role"] for c in group}
        assert {"verb", "unit", "reg_lo", "reg_hi"} <= roles
        assert all(c["offset"] == record_at + i for i, c in enumerate(group))
        # every candidate in a V record shares the same 16-bit register
        assert len({c["register"] for c in group}) == 1


def test_non_v_records_get_no_register():
    _, mc = tfi.load_reference()
    cands = tfi.candidates(mc)
    assert all(c["register"] is None for c in cands if c["kind"] != "V")


def test_patch_byte_changes_only_the_target_offset():
    mc = bytes(range(10))
    patched = tfi.patch_byte(mc, 3, 0xFF)
    assert patched[3] == 0xFF
    assert patched[:3] == mc[:3]
    assert patched[4:] == mc[4:]


def test_patch_byte_xor_roundtrips():
    mc = bytes([0x12, 0x34, 0x56])
    once = tfi.patch_byte(mc, 1, mc[1] ^ 0xFF)
    twice = tfi.patch_byte(once, 1, once[1] ^ 0xFF)
    assert twice == mc


def test_classify_run_inert():
    control = [b"\x00\x01\x02\x03"]
    result = {"outputs": [b"\x00\x01\x02\x03"], "error": None}
    assert tfi.classify_run(control, result) == "INERT"


def test_classify_run_functional():
    control = [b"\x00\x01\x02\x03"]
    result = {"outputs": [b"\x00\x01\x02\x04"], "error": None}
    assert tfi.classify_run(control, result) == "FUNCTIONAL"


def test_classify_run_fault():
    control = [b"\x00\x01\x02\x03"]
    result = {"outputs": None, "error": "Run model failed{0x8030070C}"}
    assert tfi.classify_run(control, result) == "FAULT"


def test_classify_run_other_error():
    control = [b"\x00\x01\x02\x03"]
    result = {"outputs": None, "error": "timeout"}
    assert tfi.classify_run(control, result) == "ERROR"


def test_max_abs_diff():
    import numpy as np

    a = [np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()]
    b = [np.array([1.0, 2.0, 5.5], dtype=np.float32).tobytes()]
    assert abs(tfi.max_abs_diff(a, b) - 2.5) < 1e-6


def test_max_abs_diff_none_on_mismatched_length():
    assert tfi.max_abs_diff([b"\x00"], None) is None


def test_new_fault_events_empty_without_npu_fault_log():
    # Regardless of whether the (unmerged, optional) npu_fault_log module is
    # importable in this environment, an empty/unchanged log yields no events.
    assert tfi.new_fault_events("", "") == []


def test_new_fault_events_diffs_a_growing_log():
    if tfi.npu_fault_log is None:
        return  # module not available in this environment; nothing to check
    prefix = "2026-09-23 kernel: [NPU][Error][npu_check_error_id 1]: EU[0] error\n"
    suffix = (
        "2026-09-23 kernel: [NPU][Error][npu_warp_check_error_id 1]: "
        "CV EU[6]: AXI0 Read Response Error.\n"
        "2026-09-23 kernel: [NPU][Error][irq_get_err_code 1]: npu hard error, please reset npu\n"
    )
    events = tfi.new_fault_events(prefix, prefix + suffix)
    assert len(events) == 1
    assert events[0].causes == [("CV", 6, "AXI0 Read Response Error")]


def test_new_fault_events_treats_non_suffix_as_all_new():
    if tfi.npu_fault_log is None:
        return
    old = "2026-09-23 kernel: some old rotated-out content\n"
    new = (
        "2026-09-23 kernel: [NPU][Error][npu_warp_check_error_id 1]: "
        "CONV EU[0]: Iftr Rdma Ch7 Cmd Error.\n"
        "2026-09-23 kernel: [NPU][Error][irq_get_err_code 1]: npu hard error, please reset npu\n"
    )
    events = tfi.new_fault_events(old, new)
    assert len(events) == 1
    assert events[0].causes == [("CONV", 0, "Iftr Rdma Ch7 Cmd Error")]


def test_event_summary_none_on_no_events():
    assert tfi.event_summary([]) is None


def test_event_summary_classifies_address_vs_trigger_causes():
    if tfi.npu_fault_log is None:
        return
    addr_events = tfi.new_fault_events(
        "",
        "2026-09-23 kernel: [NPU][Error][npu_warp_check_error_id 1]: "
        "CV EU[6]: AXI0 Read Response Error.\n"
        "2026-09-23 kernel: [NPU][Error][irq_get_err_code 1]: npu hard error, please reset npu\n",
    )
    summary = tfi.event_summary(addr_events)
    assert summary["cause_class"] == "address"
    assert summary["engine"] == "CV"
    assert summary["eu"] == 6

    cmd_events = tfi.new_fault_events(
        "",
        "2026-09-23 kernel: [NPU][Error][npu_potato_check_error_id 1]: "
        "CONV EU[0]: Iftr Rdma Ch7 Cmd Error.\n"
        "2026-09-23 kernel: [NPU][Error][irq_get_err_code 1]: npu hard error, please reset npu\n",
    )
    summary = tfi.event_summary(cmd_events)
    assert summary["cause_class"] == "trigger_or_cmd"


def test_committed_partial_sweep_is_well_formed():
    """The real, device-collected 27-candidate partial sweep committed
    alongside this test (docs/axera-teng2-fault-injection-relu.md's
    "Results" section) -- confirms the fixture stays loadable and its
    entries match this module's own record schema, without needing Docker
    or a device."""
    import json

    fixture = os.path.join(
        _AXERA_DIR, "fixtures", "teng2_fault_injection", "relu_sweep_partial.json"
    )
    with open(fixture) as f:
        data = json.load(f)
    assert len(data) == 27
    for key, entry in data.items():
        assert entry["offset"] == int(key)
        assert entry["classification"] in ("INERT", "FAULT", "FUNCTIONAL", "ERROR")
    # every candidate but the last (offset 406) passed its health check; that
    # last one failing twice is exactly what stopped the sweep -- see
    # docs/axera-teng2-fault-injection-relu.md's "Results" section.
    unhealthy = [k for k, e in data.items() if not e["health_ok"]]
    assert unhealthy == ["406"]
    faults = [e for e in data.values() if e["classification"] == "FAULT"]
    causes = {
        e["fault_log"]["cause"]
        for e in faults
        if e.get("fault_log") and e["fault_log"].get("cause")
    }
    assert "AXI0 Read Response Error" in causes


def test_fixed_input_is_deterministic_and_in_range():
    a = tfi.fixed_input(seed=1)
    b = tfi.fixed_input(seed=1)
    assert (a == b).all()
    assert a.shape == (1, 64, 56, 56)
    assert (-0.8 <= a).all() and (a <= 0.8).all()
