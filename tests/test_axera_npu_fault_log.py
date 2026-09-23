"""Parsing of the AX650 NPU driver's fault diagnostics (device syslog)."""

import gzip
import os
import sys

import onnx

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
from npu_fault_log import parse  # noqa: E402

_P = "2026-09-23 07:04:58 kernel: [132125.5] [C5][NPU][E][13490] [NPU][Error]"

# Line formats copied from a real `axcl-smi log` dump (AX650, firmware V3.6.5).
_ERROR_BURST = [
    f"{_P}[npu_check_error_id 821]: EU[6] error",
    f"{_P}[npu_irq_err_handle 880]: sync manager INTR_ERROR_ID is 0x40",
    f"{_P}[npu_warp_check_error_id 455]: CV EU[6]: AXI0 Read Response Error.",
    f"{_P}[npu_warp_clear_interrupt 499]: CV EU[6] CTRL_INT_VEC is 0xf001",
    f"{_P}[print_vnpu_ocm_1k_contents 2041]: vnpu 0 1k ocm data offset[0x0320]: 00100020",
    f"{_P}[irq_get_err_code 640]: npu hard error, please reset npu",
]
_HANG_BURST = [
    f"{_P}[irq_get_err_code 624]: Timeout waiting for NPU to finish running",
    f"{_P}[irq_get_err_code 625]: wait time is 10401 ms",
    f"{_P}[npu_dev_show_eu_status 1029]: EU[0] q_head=4 q_paddr=0xa493902",
    f"{_P}[npu_dev_show_eu_status 1030]: EU[0] Run CMD: cmd_l:0x00000000, cmd_h:0x00000000",
    f"{_P}[npu_dev_show_eu_status 1029]: EU[1] q_head=3 q_paddr=0xa493900",
    f"{_P}[npu_dev_show_eu_status 1030]: EU[1] Run CMD: cmd_l:0x000000a2, cmd_h:0x00300023",
    f"{_P}[npu_dev_show_eu_status 1029]: EU[2] q_head=0 q_paddr=0x0",
    f"{_P}[npu_dev_show_eu_status 1030]: EU[2] Run CMD: cmd_l:0x00000000, cmd_h:0x00000000",
    f"{_P}[npu_dev_show_all_jobs 1127]: VNPU[0] VIRTUAL_ADDR[0]: JOB Done id=1",
    f"{_P}[irq_get_err_code 640]: npu hard error, please reset npu",
    "2026-09-23 07:05:01 kernel: unrelated line",
]


def test_error_burst_names_engine_cause_and_vector():
    (event,) = parse(_ERROR_BURST)
    assert not event.timeout
    assert event.error_eus == [6]
    assert event.sync_error_ids == [0x40]
    assert event.causes == [("CV", 6, "AXI0 Read Response Error")]
    assert event.int_vecs == [("CV", 6, 0xF001)]


def test_hang_snapshot_decodes_executing_command():
    (event,) = parse(_HANG_BURST)
    assert event.timeout
    active = {s.eu: s for s in event.active_eus()}
    assert sorted(active) == [0, 1]
    assert active[1].q_head == 3
    assert active[1].cmd == bytes.fromhex("a200000023003000")
    assert active[1].verb == 0xA2


def test_consecutive_bursts_are_separate_events():
    assert len(parse(_ERROR_BURST + _HANG_BURST)) == 2


def test_command_words_share_mcode_verb_record_layout():
    # The driver's (cmd_l, cmd_h) pair is the same 8 bytes mcode.py decodes as a
    # verb record: verb, field, bank, pad, then a 4-byte operand.
    path = os.path.join(
        _AXERA_DIR, "fixtures", "dma_tiles", "relu_1x64x56x56.axmodel.gz"
    )
    with gzip.open(path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    blob = next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )
    verbs = [r for r in mcode.decode(blob, **mcode.FULL_RULE) if r["kind"] == "V"]
    record = blob[verbs[1]["at"] : verbs[1]["at"] + 8]
    assert record[:4] == bytes(
        [verbs[1]["verb"], verbs[1]["field"], verbs[1]["bank"], 0]
    )
    assert record[4:] == verbs[1]["operand"]
    (event,) = parse(_HANG_BURST)
    cmd = event.eu_states[1].cmd
    assert cmd[:4] == bytes([0xA2, 0, 0, 0]) and len(cmd) == len(record)
