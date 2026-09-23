"""Single-byte fault injection on a compiled ``Add``'s ``teng2``/``cv3``
segments, on real AX8850 hardware -- the causal-intervention technique
``README.md``'s ``M_channel`` decode used, applied to the compute
microprogram for the first time (a companion to a parallel Relu-side sweep).

Every byte of the compute segment (``teng2``, ``segments()[2]``) and the
second-input segment (``cv3``, ``segments()[3]``) of a reference
``Add(x[1,16,8,8], z[1,16,8,8])`` build is XORed with ``0xFF`` one at a time,
on its own copy of the reference, and run on device against fixed, real,
in-range inputs. Each offset is classified against a control run of the
unpatched reference:

* ``IDENTICAL`` -- byte-identical output; the byte is inert (or the flip
  cancels out).
* ``FAULT`` -- the runtime rejects the model, ``[ERROR] Run model
  failed{0x8030070C}``; retried once (a real fault faults every time; a rare
  transient recovers -- see ``README.md``'s "caveat on the fault class").
* ``DIFFERENT`` -- the model runs to completion but the output changes: a
  live, computational byte.

For every ``FAULT``, the device's own kernel syslog is pulled in the same
locked critical section as the triggering run (``axcl-smi log -t 16``,
decimal 16 == the doc's ``0x10`` mask -- see ``docs/axera-axcl-runtime-mining.md``
and ``scripts/axera/npu_fault_log.py``) and diffed against the previous
pull, so each fault gets an execution unit and a decoded cause, not just a
runtime error code. Live collection used ``scripts/axera/npu_fault_log.py``
(from a concurrent, not-yet-merged fork's PR #1821 at the time this ran) to
parse and describe new syslog lines; the committed ``fault_events.json``
fixture already holds those described strings, so ``report()`` below needs
no dependency on that module.

Findings (details and the full byte map: ``docs/axera-teng2-fault-injection-add.md``):

* ``cv3`` (256 bytes, fully swept): 143 ``IDENTICAL``, 109 ``FAULT``, 4
  ``DIFFERENT``. Its faults were independently confirmed (by a concurrent
  fork mining the same device log) to land on **CV EU6**.
* ``teng2`` (1,632 bytes; 45 swept, offsets 436-480, before the shared
  device's contention made a full sweep impractical in one session): 15
  ``IDENTICAL``, 24 ``FAULT``, 6 ``DIFFERENT`` -- a far higher functional
  rate than ``cv3``, consistent with it being the op-defining compute
  region. Its faults land on **TENG EU9**, splitting into two decoded causes:
  ``AXI0 Read Response Error`` (an address-like byte -- corrupting it makes
  the engine read the wrong memory) and ``Undefine Error``/``Retrigger
  error`` (a structural/opcode-like byte -- corrupting it makes the engine
  reject the instruction itself outright). One fault snapshot's ``EU9``
  executing-command bytes directly showed the injected ``0xFF`` byte inside
  the live instruction word, confirming the offset-to-register mapping
  end to end.
* Every ``DIFFERENT`` byte was matched back to ``mcode.py``'s own record
  decode: several land inside a ``V`` record's 32-bit value at a specific
  register address (``reg=0x0170``, ``0x03b0``, ``0x03c0``), one lands in
  the register address itself (``reg=0x03d0``'s low/high byte -- corrupting
  the address still ran, just against a different, still-valid register),
  one is the verb byte itself, and several land inside the compressed short
  units (``S``/``B``/``Q`` forms) this project decoded separately from verb
  records.

This is a *characterization* result: a per-byte map of what's live, what
faults, and (for a sample of ``teng2``) which register or which fault class
each functional byte belongs to. It is not a predictor or an emitter --
knowing a byte is "the low byte of a 32-bit write to register 0x03d0" does
not yet say what value that register should hold for a new shape.

Usage::

    teng2_fault_injection_add.py sweep cv3|teng2 WORK_DIR    # real device sweep
    teng2_fault_injection_add.py report                       # byte map from
                                                                # the committed fixtures
"""

from __future__ import annotations

import contextlib
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

_FIXTURES = os.path.join(_HERE, "fixtures", "teng2_fault_injection_add")
_REF_GZ = os.path.join(_FIXTURES, "add_1x16x8x8_ref.axmodel.gz")

_VM = "axcl-vm"
_BIN = "/usr/bin/axcl/axcl_run_model"
_LOCK_PATH = "/tmp/axcl-device.lock"
_FAULT_RE = re.compile(r"0x8030070C|failed\{0x8030070C\}", re.I)


def _lxc(*args: str, t: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["lxc", *args], capture_output=True, text=True, timeout=t)


@contextlib.contextmanager
def device_lock():
    """Hold the shared AX8850 lock for exactly one device interaction. Other
    forks share this device; a whole sweep must never hold it continuously."""
    fh = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _run_on_device_nolock(axmodel_path: str, inputs: dict, timeout: int = 90):
    """``{tensor_name: bytes} -> (outputs_dict_or_None, log)``. Caller must
    hold ``device_lock()``."""
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in", "0")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        for name, data in inputs.items():
            open(os.path.join(in_dir, f"{name}.bin"), "wb").write(data)
        open(os.path.join(td, "list.txt"), "w").write("0\n")
        made = _lxc("exec", _VM, "--", "mktemp", "-d", "/tmp/axcl.XXXXXX")
        if made.returncode != 0:
            return None, "mktemp failed: " + made.stdout + made.stderr
        remote = made.stdout.strip()
        try:
            _lxc("file", "push", axmodel_path, f"{_VM}{remote}/m.axmodel", t=60)
            _lxc("file", "push", "-r", os.path.join(td, "in"), f"{_VM}{remote}/", t=60)
            _lxc(
                "file",
                "push",
                os.path.join(td, "list.txt"),
                f"{_VM}{remote}/list.txt",
                t=30,
            )
            _lxc("exec", _VM, "--", "mkdir", "-p", f"{remote}/out")
            proc = _lxc(
                "exec",
                _VM,
                "--",
                _BIN,
                "-m",
                f"{remote}/m.axmodel",
                "-i",
                f"{remote}/in",
                "-o",
                f"{remote}/out",
                "-l",
                f"{remote}/list.txt",
                "-r",
                "1",
                "-w",
                "0",
                t=90,
            )
            log = proc.stdout + proc.stderr
            local_out = os.path.join(td, "o")
            os.makedirs(local_out, exist_ok=True)
            _lxc("file", "pull", "-r", f"{_VM}{remote}/out", local_out, t=60)
            outs = {}
            for p in sorted(glob.glob(os.path.join(local_out, "out", "0", "*.bin"))):
                outs[os.path.basename(p)[:-4]] = open(p, "rb").read()
            return (outs or None), log
        finally:
            with contextlib.suppress(Exception):
                _lxc("exec", _VM, "--", "rm", "-rf", remote, t=30)


def run_on_device(axmodel_path: str, inputs: dict, timeout: int = 90):
    with device_lock():
        return _run_on_device_nolock(axmodel_path, inputs, timeout=timeout)


def classify(outs, log: str, control_outs=None) -> str:
    if outs is None:
        return "FAULT" if _FAULT_RE.search(log) else "ERROR:" + log[-300:]
    if control_outs is not None:
        return (
            "DIFFERENT"
            if any(outs[k] != control_outs.get(k) for k in outs)
            else "IDENTICAL"
        )
    return "OK"


def dump_syslog_nolock(dest_dir: str, tag: str) -> str | None:
    """``axcl-smi log -t 16 -o <remote>``, pulled and extracted under
    ``dest_dir/tag``. Returns the extracted ``*.log`` path, or ``None``.
    Caller must hold ``device_lock()``."""
    made = _lxc("exec", _VM, "--", "mktemp", "-d", "/tmp/axlog.XXXXXX")
    if made.returncode != 0:
        return None
    remote = made.stdout.strip()
    local_dir = os.path.join(dest_dir, tag)
    os.makedirs(local_dir, exist_ok=True)
    try:
        r = _lxc(
            "exec",
            _VM,
            "--",
            "/usr/bin/axcl/axcl-smi",
            "log",
            "-t",
            "16",
            "-o",
            remote,
            t=60,
        )
        if r.returncode != 0:
            return None
        m = re.search(r"log dump finished: (\S+\.tar\.gz)", r.stdout + r.stderr)
        if not m:
            return None
        remote_tar = m.group(1)
        local_tar = os.path.join(local_dir, os.path.basename(remote_tar))
        if _lxc("file", "pull", f"{_VM}{remote_tar}", local_tar, t=60).returncode != 0:
            return None
    finally:
        with contextlib.suppress(Exception):
            _lxc("exec", _VM, "--", "rm", "-rf", remote, t=30)
    with tarfile.open(local_tar) as tf:
        tf.extractall(local_dir)
    logs = glob.glob(os.path.join(local_dir, "AXSyslog", "syslog", "*.log"))
    return sorted(logs)[-1] if logs else None


def run_and_maybe_capture_log(
    axmodel_path, inputs, log_dir, tag, control_outs, timeout=90
):
    """One locked critical section: run, classify, and -- only on ``FAULT`` --
    dump the device syslog immediately, in the same lock, so no other fork's
    activity lands between the fault and the dump."""
    with device_lock():
        outs, log = _run_on_device_nolock(axmodel_path, inputs, timeout=timeout)
        cls = classify(outs, log, control_outs)
        syslog_path = dump_syslog_nolock(log_dir, tag) if cls == "FAULT" else None
        return cls, log, syslog_path


def _fixed_inputs(seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.8, 0.8, (1, 16, 8, 8)).astype(np.float32)
    z = rng.uniform(-0.8, 0.8, (1, 16, 8, 8)).astype(np.float32)
    return x, z


def load_reference(path: str | None = None):
    """The reference ``Add`` model (from the committed fixture unless a path
    is given), its mcode bytes, and its ``(header, segments)``."""
    if path is None:
        import gzip

        path = os.path.join(tempfile.mkdtemp(), "ref.axmodel")
        with gzip.open(_REF_GZ, "rb") as f, open(path, "wb") as out:
            out.write(f.read())
    model = onnx.load(path, load_external_data=False)
    mc = bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )
    header, segs = mcode.segments(mc)
    return model, mc, header, segs


def record_at(records, off: int):
    """The decoded ``mcode.decode()`` record covering byte offset ``off``, or
    ``None`` (a raw/unclassified byte)."""
    for r in records:
        at = r["at"]
        if r["kind"] == "V":
            length = 8 if len(r["operand"]) == 4 else 7
        elif r["kind"] == "W":
            length = 7
        elif r["kind"] == "S":
            length = 1 + r["p"] + 1 + 1 + 1 + len(r["extra"])
        elif r["kind"] == "B":
            length = 2 + len(r["extra"])
        else:
            length = None
        if length is not None and at <= off < at + length:
            return r
    return None


def describe_offset(records, off: int) -> str:
    """Human-readable role of byte ``off`` within its ``mcode.decode()``
    record: which register (for a ``V``/``W`` write) and which part of it,
    or which compressed short-unit form."""
    r = record_at(records, off)
    if r is None:
        return "raw/unclassified"
    at, rel = r["at"], off - r["at"]
    if r["kind"] == "V":
        reg = r["field"] | (r["bank"] << 8)
        if rel == 0:
            return f"V-record @{at}: verb byte itself (verb=0x{r['verb']:02x}, reg=0x{reg:04x})"
        if rel == 1:
            return f"V-record @{at}: unit byte (0x00 for AX650; reg=0x{reg:04x})"
        if rel == 2:
            return f"V-record @{at}: register LOW byte (field; reg=0x{reg:04x})"
        if rel == 3:
            return f"V-record @{at}: register HIGH byte (bank; reg=0x{reg:04x})"
        return (
            f"V-record @{at}: value32 byte {rel - 4} (LE) of write to reg=0x{reg:04x}"
        )
    if r["kind"] == "W":
        reg = r["field"] | (r["bank"] << 8)
        return f"W-record @{at} (companion write, reg=0x{reg:04x}), rel byte {rel}"
    return f"{r['kind']}-record @{at} (compressed short unit), rel byte {rel}"


def fault_causes(fault_events: dict[str, list[str]]) -> dict[str, set[str]]:
    """Per ``"label:offset"`` key, the set of decoded fault causes seen
    across its capture attempts (``AXI* Response Error``, ``Undefine
    Error:*``, ``Retrigger error``, ``timeout``)."""
    causes = {}
    pattern = re.compile(
        r"(AXI\d Read Response Error|AXI\d Write Response Error|Undefine Error:\w+|Retrigger error|timeout)"
    )
    for key, events in fault_events.items():
        found = set()
        for e in events:
            found.update(pattern.findall(e))
        causes[key] = found
    return causes


def report(
    results_path: str = None, fault_events_path: str = None, ref_path: str = None
) -> None:
    results_path = results_path or os.path.join(_FIXTURES, "results.json")
    fault_events_path = fault_events_path or os.path.join(
        _FIXTURES, "fault_events.json"
    )
    results = json.load(open(results_path))
    fault_events = json.load(open(fault_events_path))
    _, mc, header, segs = load_reference(ref_path)
    records = mcode.decode(mc, start=297, end=None, **mcode.FULL_RULE)

    print("Segment layout:")
    names = {2: "teng2", 3: "cv3", 4: "sdma4"}
    for i, (o, ln, _t) in enumerate(segs):
        print(f"  seg[{i}] ({names.get(i, '?')}) off={o} len={ln}")

    import collections

    counts = collections.defaultdict(collections.Counter)
    for key, cls in results.items():
        label = key.split(":")[0]
        counts[label][cls.split("_")[0]] += 1
    print("\nCounts by classification:")
    for label, c in counts.items():
        print(f"  {label}: {dict(c)}")

    print("\nFunctional (DIFFERENT) bytes:")
    for key, cls in sorted(
        results.items(),
        key=lambda kv: (kv[0].split(":")[0], int(kv[0].split(":")[1].split("_")[0])),
    ):
        if "DIFFERENT" in cls:
            label, off_s = key.split(":")
            off = int(off_s.split("_")[0])
            print(
                f"  [{label}] off={off} cls={cls}\n      {describe_offset(records, off)}"
            )

    causes = fault_causes(fault_events)
    addr = sum(1 for c in causes.values() if any("AXI" in x for x in c))
    struct_ = sum(
        1
        for c in causes.values()
        if any("Undefine" in x or "Retrigger" in x for x in c)
    )
    timeout_ = sum(1 for c in causes.values() if any(x == "timeout" for x in c))
    print(
        f"\nFault causes captured for {len(causes)} offsets: address-like(AXI)={addr} structure-like(Undefine/Retrigger)={struct_} timeout={timeout_}"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "report":
        report()
        return 0
    print(
        "Only 'report' is runnable without device access; see the module docstring for the sweep methodology."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
