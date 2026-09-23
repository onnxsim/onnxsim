"""Parse the AX650 NPU driver's fault diagnostics out of a device syslog.

When a model faults or hangs, the card's NPU kernel driver writes engine-level
diagnostics to the device kernel log, which ``axcl-smi log -o DIR`` dumps (the
tarball's ``AXSyslog/syslog/*.log``). Nothing in the host-side AXCL runtime
reports them; ``axcl_run_model`` only prints ``Run model failed{0x8030070C}``.
See ``docs/axera-axcl-runtime-mining.md``.

Line formats handled (the prefix before ``[NPU][Error]`` is ignored):

* ``[npu_check_error_id ...]: EU[6] error`` -- the faulting execution unit.
* ``[npu_irq_err_handle ...]: sync manager INTR_ERROR_ID is 0x40`` -- a bitmask,
  bit ``n`` set for EU ``n`` (0x1 = EU0, 0x2 = EU1, 0x40 = EU6 in the data seen).
* ``[npu_warp_check_error_id ...]: CV EU[6]: AXI0 Read Response Error.`` and
  ``[npu_potato_check_error_id ...]: CONV EU[0]: Iftr Rdma Ch7 Cmd Error.`` --
  the engine name and a decoded cause.
* ``[npu_*_clear_interrupt ...]: CV EU[6] CTRL_INT_VEC is 0xf001`` -- the raw
  interrupt vector for that cause.
* ``[irq_get_err_code ...]: Timeout waiting for NPU to finish running`` followed
  by ``[npu_dev_show_eu_status ...]: EU[9] q_head=205 q_paddr=0x...`` and
  ``EU[9] Run CMD: cmd_l:0x000000a2, cmd_h:0x00300013`` -- a hang snapshot. The
  command pair is the executing instruction as two little-endian words: the
  same 8 bytes ``mcode.py`` decodes as a ``V`` record (verb, field, bank, pad,
  then a 4-byte operand).

Usage::

    npu_fault_log.py SYSLOG_FILE            # one line per fault event
"""

from __future__ import annotations

import re
import struct
import sys
from dataclasses import dataclass, field

_BODY = re.compile(r"\[NPU\]\[Error\]\[(\w+) \d+\]: (.*)$")
_EU_ERROR = re.compile(r"^EU\[(\d+)\] error$")
_SYNC_ID = re.compile(r"^sync manager INTR_ERROR_ID is (0x[0-9a-fA-F]+)$")
_CAUSE = re.compile(r"^(\w+) EU\[(\d+)\]: (.+?)\.?$")
_INT_VEC = re.compile(r"^(\w+) EU\[(\d+)\] CTRL_INT_VEC is (0x[0-9a-fA-F]+)$")
_QUEUE = re.compile(r"^EU\[(\d+)\] q_head=(\d+) q_paddr=(0x[0-9a-fA-F]+)$")
_CMD = re.compile(
    r"^EU\[(\d+)\] Run CMD: cmd_l:(0x[0-9a-fA-F]+), cmd_h:(0x[0-9a-fA-F]+)$"
)
_TIMEOUT = "Timeout waiting for NPU to finish running"
_HARD = "npu hard error, please reset npu"


@dataclass
class EuState:
    """One execution unit's queue position and executing command in a hang dump."""

    eu: int
    q_head: int
    q_paddr: int
    cmd: bytes = b""

    @property
    def verb(self) -> int | None:
        return self.cmd[0] if self.cmd else None


@dataclass
class FaultEvent:
    """One driver report, from its first error line to ``npu hard error``."""

    timeout: bool = False
    error_eus: list[int] = field(default_factory=list)
    sync_error_ids: list[int] = field(default_factory=list)
    causes: list[tuple[str, int, str]] = field(default_factory=list)
    int_vecs: list[tuple[str, int, int]] = field(default_factory=list)
    eu_states: dict[int, EuState] = field(default_factory=dict)

    def active_eus(self) -> list[EuState]:
        """EUs whose queue had advanced (nonzero head) when the snapshot was taken."""
        return [s for s in self.eu_states.values() if s.q_head]


def _command_bytes(cmd_l: str, cmd_h: str) -> bytes:
    return struct.pack("<II", int(cmd_l, 16), int(cmd_h, 16))


def parse(lines) -> list[FaultEvent]:
    """Group the NPU error lines of a device syslog into fault events."""
    events: list[FaultEvent] = []
    current: FaultEvent | None = None
    for line in lines:
        match = _BODY.search(line.rstrip("\n"))
        if not match:
            continue
        func, text = match.group(1), match.group(2).strip()
        if func == "print_vnpu_ocm_1k_contents":
            continue
        if current is None:
            current = FaultEvent()
        if text == _TIMEOUT:
            current.timeout = True
        elif m := _EU_ERROR.match(text):
            current.error_eus.append(int(m.group(1)))
        elif m := _SYNC_ID.match(text):
            current.sync_error_ids.append(int(m.group(1), 16))
        elif m := _INT_VEC.match(text):
            current.int_vecs.append((m.group(1), int(m.group(2)), int(m.group(3), 16)))
        elif m := _QUEUE.match(text):
            eu = int(m.group(1))
            current.eu_states[eu] = EuState(eu, int(m.group(2)), int(m.group(3), 16))
        elif m := _CMD.match(text):
            eu = int(m.group(1))
            state = current.eu_states.setdefault(eu, EuState(eu, 0, 0))
            state.cmd = _command_bytes(m.group(2), m.group(3))
        elif m := _CAUSE.match(text):
            current.causes.append((m.group(1), int(m.group(2)), m.group(3)))
        if text == _HARD:
            events.append(current)
            current = None
    if current is not None:
        events.append(current)
    return events


def describe(event: FaultEvent) -> str:
    parts = ["timeout" if event.timeout else "error"]
    parts += [f"{name} EU{eu}: {cause}" for name, eu, cause in event.causes]
    parts += [f"sync_id=0x{i:x}" for i in event.sync_error_ids]
    for state in event.active_eus():
        parts.append(f"EU{state.eu} q_head={state.q_head} cmd={state.cmd.hex(' ')}")
    return "; ".join(parts)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    with open(argv[0], errors="replace") as f:
        for event in parse(f):
            print(describe(event))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
