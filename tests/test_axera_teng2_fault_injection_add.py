"""Regression coverage for the Add teng2/cv3 fault-injection byte map. No
Docker/device required -- exercises the register-decode logic and the
committed sweep results, not the live device sweep itself."""

import json
import os
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
from teng2_fault_injection_add import (  # noqa: E402
    _FIXTURES,
    classify,
    describe_offset,
    fault_causes,
    load_reference,
)


def _records():
    _, mc, _header, _segs = load_reference()
    return mc, mcode.decode(mc, start=297, end=None, **mcode.FULL_RULE)


def test_reference_fixture_has_the_expected_segments():
    _, _mc, header, segs = load_reference()
    assert header == 340
    sizes = {i: ln for i, (_o, ln, _t) in enumerate(segs)}
    assert sizes[2] == 1632  # teng2
    assert sizes[3] == 256  # cv3


def test_results_fixture_covers_all_of_cv3_and_a_prefix_of_teng2():
    results = json.load(open(os.path.join(_FIXTURES, "results.json")))
    cv3 = {k for k in results if k.startswith("cv3:")}
    teng2 = {k for k in results if k.startswith("teng2:")}
    assert len(cv3) == 256
    offs = sorted(int(k.split(":")[1]) for k in cv3)
    assert offs == list(range(2068, 2068 + 256))
    assert len(teng2) > 0
    for k in teng2:
        off = int(k.split(":")[1])
        assert 436 <= off < 436 + 1632


def test_results_fixture_has_every_classification():
    results = json.load(open(os.path.join(_FIXTURES, "results.json")))
    kinds = {v.split("_")[0] for v in results.values()}
    assert kinds >= {"IDENTICAL", "FAULT", "DIFFERENT"}


def test_describe_offset_finds_the_known_functional_register_writes():
    mc, records = _records()
    # cv3 off=2102: value32 byte 1 of a write to reg=0x0170 (found by the
    # real sweep and independently reproducible from the committed fixture's
    # own mcode bytes, not from the sweep log).
    desc = describe_offset(records, 2102)
    assert "V-record" in desc
    assert "reg=0x0170" in desc
    assert "value32 byte 1" in desc


def test_describe_offset_finds_a_register_address_byte():
    mc, records = _records()
    # teng2 off=467/468: the LOW/HIGH byte of the register address itself
    # (not the value), for the record at 465.
    lo = describe_offset(records, 467)
    hi = describe_offset(records, 468)
    assert "register LOW byte" in lo and "reg=0x03d0" in lo
    assert "register HIGH byte" in hi and "reg=0x03d0" in hi


def test_describe_offset_handles_a_raw_or_short_unit_byte():
    mc, records = _records()
    desc = describe_offset(records, 2106)
    assert "compressed short unit" in desc


def test_classify_distinguishes_fault_identical_different():
    control = {"y": b"\x00\x01"}
    assert classify(None, "Run model failed{0x8030070C}") == "FAULT"
    assert classify({"y": b"\x00\x01"}, "", control) == "IDENTICAL"
    assert classify({"y": b"\x00\x02"}, "", control) == "DIFFERENT"
    assert classify(None, "some other error").startswith("ERROR")


def test_fault_causes_splits_address_like_from_structure_like():
    fault_events = json.load(open(os.path.join(_FIXTURES, "fault_events.json")))
    causes = fault_causes(fault_events)
    assert len(causes) > 0
    addr = [k for k, c in causes.items() if any("AXI" in x for x in c)]
    struct_ = [
        k
        for k, c in causes.items()
        if any("Undefine" in x or "Retrigger" in x for x in c)
    ]
    assert addr and struct_
    # These two sets should be real classes, not the same offsets relabeled.
    assert set(addr) != set(struct_)
