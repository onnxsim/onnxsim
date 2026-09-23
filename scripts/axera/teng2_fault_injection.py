"""Single-byte fault injection on a compiled ``teng2`` (elementwise compute)
segment, run directly on the AX650N -- the causal-intervention technique
``README.md``'s "bit-flip probe" sections used on real Conv-heavy mcode,
applied here for the first time to the compute segment of a standalone
elementwise op.

Unlike every prior ``teng2`` decode attempt in this project (which all
compared *different* compiler builds across a shape sweep), nothing here
gets rebuilt: one already-verified reference is patched byte by byte and
re-run on real hardware, so any change in behaviour is caused by the patch
alone, not by anything the compiler could have chosen differently on a
fresh build.

``classify_run`` sorts each patched offset into one of three buckets,
matching the README's own vocabulary:

* ``INERT`` -- output bit-identical to the unpatched control. Likely
  padding, or a byte the runtime never reads for this op.
* ``FAULT`` -- the runtime rejects the model (``0x8030070C``), the same
  graceful, deterministic rejection the README's Conv probes hit. Likely a
  validity-checked structural byte (an opcode, a length, a checksum-like
  field).
* ``FUNCTIONAL`` -- the model runs to completion but the output changes.
  This is a genuinely live, computational byte -- the class this project
  has never before produced for ``teng2`` specifically.

Usage::

    teng2_fault_injection.py sweep OUT.json [--limit N] [--resume]
    teng2_fault_injection.py finesweep OUT.json OFFSET [--values 0x01,0x02,...]
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

try:
    import npu_fault_log  # noqa: E402  -- read-only, from PR #1821 if present
except ImportError:  # pragma: no cover -- module not merged yet in this worktree
    npu_fault_log = None

_REFERENCE = os.path.join(_HERE, "fixtures", "dma_tiles", "relu_1x64x56x56.axmodel.gz")
_MCODE_NAME = "subgraph_npu_0_b1_neu"
_LOCK = "/tmp/axcl-device.lock"
_FAULT_MARKER = "0x8030070C"
_AXCL_BIN = "/usr/bin/axcl/axcl_run_model"
_AXCL_SMI = "/usr/bin/axcl/axcl-smi"
_AXCL_LXD_VM = os.environ.get("AXCL_LXD_VM", "axcl-vm")
"""Routes every device call through the LXD VM by default (matching this
project's established convention for anything that deliberately induces
faults): the out-of-tree AXCL host driver has crashed and wedged the host
before, but inside a KVM guest the same fault only kills the guest, which
`lxc restart --force` recovers without a host reboot. Set `AXCL_LXD_VM=`
(empty) to run against the host binary directly instead."""


def _lxc(*cmd: str, t: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["lxc", *cmd], capture_output=True, text=True, timeout=t)


def _run_on_device_with_inputs(
    axmodel_path: str, inputs: dict, *, timeout: int = 120
) -> dict:
    """Minimal, self-contained re-implementation of
    ``pulsar2_docker.run_on_device_with_inputs`` (including its
    `AXCL_LXD_VM` routing) for this module's one use case (single-input,
    single-output, ``-r 1 -w 0``). Written directly against
    `axcl_run_model`'s confirmed ``-i/-o/-l`` contract rather than importing
    `pulsar2_docker`, which pulls in `onnxsim` and needs a built C++
    extension this worktree does not have -- not needed here, since no
    Pulsar2 build happens in this file, only device runs of an
    already-compiled model."""
    import glob
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in", "0")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        for name, data in inputs.items():
            with open(os.path.join(in_dir, f"{name}.bin"), "wb") as f:
                f.write(data)
        list_path = os.path.join(td, "list.txt")
        with open(list_path, "w") as f:
            f.write("0\n")

        if not _AXCL_LXD_VM:
            if not os.path.exists(_AXCL_BIN):
                return {"outputs": None, "error": "axcl_run_model not found"}
            try:
                proc = subprocess.run(
                    [
                        _AXCL_BIN,
                        "-m",
                        axmodel_path,
                        "-i",
                        os.path.join(td, "in"),
                        "-o",
                        out_dir,
                        "-l",
                        list_path,
                        "-r",
                        "1",
                        "-w",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {"outputs": None, "error": "timeout"}
            log = proc.stdout + proc.stderr
        else:
            vm = _AXCL_LXD_VM
            made = _lxc("exec", vm, "--", "mktemp", "-d", "/tmp/axcl.XXXXXX")
            if made.returncode != 0:
                return {
                    "outputs": None,
                    "error": "lxc mktemp failed: " + made.stdout + made.stderr,
                }
            remote = made.stdout.strip()
            try:
                res = _lxc(
                    "file", "push", axmodel_path, f"{vm}{remote}/probe.axmodel", t=120
                )
                if res.returncode != 0:
                    return {
                        "outputs": None,
                        "error": "push model failed: " + res.stdout + res.stderr,
                    }
                res = _lxc(
                    "file", "push", "-r", os.path.join(td, "in"), f"{vm}{remote}/", t=60
                )
                if res.returncode != 0:
                    return {
                        "outputs": None,
                        "error": "push inputs failed: " + res.stdout + res.stderr,
                    }
                res = _lxc("file", "push", list_path, f"{vm}{remote}/list.txt", t=30)
                if res.returncode != 0:
                    return {
                        "outputs": None,
                        "error": "push list failed: " + res.stdout + res.stderr,
                    }
                res = _lxc("exec", vm, "--", "mkdir", "-p", f"{remote}/out")
                if res.returncode != 0:
                    return {
                        "outputs": None,
                        "error": "mkdir out failed: " + res.stdout + res.stderr,
                    }
                try:
                    proc = _lxc(
                        "exec",
                        vm,
                        "--",
                        _AXCL_BIN,
                        "-m",
                        f"{remote}/probe.axmodel",
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
                        t=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return {"outputs": None, "error": "timeout"}
                log = proc.stdout + proc.stderr
                # `lxc file pull -r` recreates the directory inside the
                # target, so the empty `out_dir` made above would put the
                # pulled files a level too deep for the glob below -- the
                # exact bug `pulsar2_docker._invoke_axcl`'s own docstring
                # warns about and works around; copied here.
                if os.path.isdir(out_dir) and not os.listdir(out_dir):
                    os.rmdir(out_dir)
                pulled = _lxc("file", "pull", "-r", f"{vm}{remote}/out", td, t=120)
                if pulled.returncode != 0:
                    log += "\npull out failed: " + pulled.stdout + pulled.stderr
            finally:
                try:
                    _lxc("exec", vm, "--", "rm", "-rf", remote, t=60)
                except subprocess.TimeoutExpired:
                    pass

        outs = []
        for p in sorted(glob.glob(os.path.join(out_dir, "0", "*.bin"))):
            with open(p, "rb") as f:
                outs.append(f.read())
        if not outs:
            return {"outputs": None, "error": log[-2000:]}
        return {"outputs": outs, "error": None}


def _collect_syslog_text(timeout: int = 60) -> str:
    """The device kernel log's NPU-error lines, via ``axcl-smi log -t 0x10``
    (syslog only -- see ``docs/axera-axcl-runtime-mining.md``). Call this
    *inside* the same lock window as the faulting run it explains, before
    releasing the device lock, so no other process's fault lands in between.
    Returns the concatenated text of every ``*.log`` file the dump contains
    (there may be several rotated logs); empty string on any failure (a
    missing ``axcl-smi``, a dump this project's tooling can't parse, etc.)
    -- a missing log should never abort a sweep that would otherwise
    succeed, only leave that one offset's fault unexplained."""
    import glob
    import tarfile
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as td:
            if not _AXCL_LXD_VM:
                out_dir = os.path.join(td, "log")
                proc = subprocess.run(
                    [_AXCL_SMI, "log", "-t", "16", "-o", out_dir],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    return ""
                search_root = out_dir
            else:
                vm = _AXCL_LXD_VM
                made = _lxc("exec", vm, "--", "mktemp", "-d", "/tmp/axcllog.XXXXXX")
                if made.returncode != 0:
                    return ""
                remote = made.stdout.strip()
                try:
                    proc = _lxc(
                        "exec",
                        vm,
                        "--",
                        _AXCL_SMI,
                        "log",
                        "-t",
                        "16",
                        "-o",
                        f"{remote}/log",
                        t=timeout,
                    )
                    if proc.returncode != 0:
                        return ""
                    local_log = os.path.join(td, "log")
                    pulled = _lxc("file", "pull", "-r", f"{vm}{remote}/log", td, t=120)
                    if pulled.returncode != 0 or not os.path.isdir(local_log):
                        return ""
                    search_root = local_log
                finally:
                    try:
                        _lxc("exec", vm, "--", "rm", "-rf", remote, t=30)
                    except subprocess.TimeoutExpired:
                        pass

            # `-o DIR` may write a tarball into DIR, or extracted files
            # directly -- handle both rather than assume one.
            for tar_path in glob.glob(
                os.path.join(search_root, "**", "*.tar*"), recursive=True
            ):
                try:
                    with tarfile.open(tar_path) as tf:
                        tf.extractall(search_root, filter="data")  # noqa: S202 -- our own device dump
                except tarfile.TarError:
                    continue

            chunks = []
            for log_path in sorted(
                glob.glob(os.path.join(search_root, "**", "*.log"), recursive=True)
            ):
                try:
                    with open(log_path, errors="replace") as f:
                        chunks.append(f.read())
                except OSError:
                    continue
            return "\n".join(chunks)
    except (subprocess.TimeoutExpired, OSError):
        return ""


def new_fault_events(previous_text: str, current_text: str) -> list:
    """``npu_fault_log.FaultEvent`` objects present in ``current_text`` but
    not ``previous_text`` -- a plain suffix diff (the device log is
    append-only under normal operation; if the suffix relationship doesn't
    hold, e.g. after log rotation, the whole current text is treated as
    new rather than silently attributing nothing)."""
    if npu_fault_log is None:
        return []
    if current_text.startswith(previous_text):
        new_text = current_text[len(previous_text) :]
    else:
        new_text = current_text
    return npu_fault_log.parse(new_text.splitlines())


def event_summary(events: list) -> dict | None:
    """The most useful fields of the last (most recent) fault event, as a
    JSON-friendly dict: engine name, EU number, decoded cause, and, for a
    hang, the executing command's bytes -- an AXI response error is strong
    evidence the patched byte is part of an address operand, a trigger/cmd
    error suggests an opcode/length/structure byte
    (``docs/axera-axcl-runtime-mining.md``). ``None`` if no events (the log
    dump failed, wasn't available, or genuinely had nothing new)."""
    if not events:
        return None
    event = events[-1]
    out: dict = {"timeout": event.timeout}
    if event.causes:
        name, eu, cause = event.causes[-1]
        out["engine"] = name
        out["eu"] = eu
        out["cause"] = cause
        out["cause_class"] = (
            "address" if "Response Error" in cause else "trigger_or_cmd"
        )
    active = event.active_eus()
    if active:
        out["active_eus"] = [
            {"eu": s.eu, "q_head": s.q_head, "cmd": s.cmd.hex(" ")} for s in active
        ]
    return out


def _seed_syslog_baseline() -> str:
    """A fresh device-log snapshot, taken (locked) right before a sweep's own
    patched runs start -- the baseline `new_fault_events` diffs against, so
    the sweep's first fault is attributed to its own run, not to whatever
    fork or prior sweep last faulted this shared device."""
    fd = os.open(_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _flock(fd)
        return _collect_syslog_text()
    finally:
        os.close(fd)


def load_reference() -> tuple[onnx.ModelProto, bytes]:
    """The Relu(x[1,64,56,56]) reference model and its raw mcode bytes."""
    with gzip.open(_REFERENCE, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    inits = {i.name: i for i in model.graph.initializer}
    return model, bytes(inits[_MCODE_NAME].raw_data)


def segment2_bounds(mc: bytes) -> tuple[int, int]:
    """``(start, end)`` absolute offsets of segment 2 (the compute segment),
    trimmed to its real (non-padding) content."""
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[2]
    end = pos + length
    while end > pos and mc[end - 1] == 0:
        end -= 1
    return pos, end


def candidates(mc: bytes) -> list[dict]:
    """Candidate offsets to probe, each carrying its structural role.

    For a ``V`` record (an 8-byte, or 7-byte, register write -- per
    ``docs/axera-pulsar-v1-mining.md``'s decode of the layout as ``[verb]
    [unit][reg_lo][reg_hi][value32]``, where ``mcode.py``'s ``field``+``bank``
    bytes are the 16-bit register address and ``operand`` is the 32-bit
    value), every byte is tested individually so a ``FUNCTIONAL`` result can
    be reported as *which register, which byte of its value* rather than a
    bare offset. For every other record kind (the compressed ``S``/``B``/etc.
    short units, whose schema PR #1818 did not decode), one representative
    byte -- the record's own first byte -- gives broad, reproducible coverage
    of that structural position class; see the module docstring for the
    follow-up (finer) sweep on anything this flags ``FUNCTIONAL``.
    """
    # Decode against the segment's real (untrimmed) end, not the
    # trailing-zero-trimmed one `segment2_bounds` reports: trimming before
    # decoding can clip a genuine record whose own trailing bytes happen to
    # be zero (a live register write with value 0 is not padding).
    _, segs = mcode.segments(mc)
    pos, length, _ = segs[2]
    start, end = pos, pos + length
    records = mcode.decode(mc, start=start, end=end, **mcode.FULL_RULE)
    out = []
    for r in records:
        at, kind = r["at"], r["kind"]
        if kind == "V":
            if not r["operand"]:
                # The one record straddling this segment's own end: a `V`
                # token whose length-disambiguation peeked past `end` into
                # the next segment, truncating its own operand to nothing.
                # Per mcode.py's own boundary rule ("the verb can start just
                # before the word boundary its segment begins on"), this is
                # really the *next* segment's opening `a7` marker, not a
                # genuine teng2 instruction -- skip it rather than probe
                # meaningless bytes.
                continue
            register = (r["bank"] << 8) | r["field"]
            roles = [("verb", 0), ("unit", 1), ("reg_lo", 2), ("reg_hi", 3)]
            roles += [(f"value_byte{i}", 4 + i) for i in range(len(r["operand"]))]
            for role, delta in roles:
                out.append(
                    {
                        "offset": at + delta,
                        "record_at": at,
                        "kind": kind,
                        "role": role,
                        "register": register,
                    }
                )
        else:
            out.append(
                {
                    "offset": at,
                    "record_at": at,
                    "kind": kind,
                    "role": "first_byte",
                    "register": None,
                }
            )
    return out


def candidate_offsets(mc: bytes) -> list[int]:
    """Bare offsets from :func:`candidates`, for callers that don't need the
    structural-role metadata (e.g. a quick coverage count)."""
    return [c["offset"] for c in candidates(mc)]


def patch_byte(mc: bytes, offset: int, value: int) -> bytes:
    patched = bytearray(mc)
    patched[offset] = value & 0xFF
    return bytes(patched)


def _model_with_mcode(model: onnx.ModelProto, mc: bytes) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for init in out.graph.initializer:
        if init.name == _MCODE_NAME:
            init.raw_data = mc
    return out


def fixed_input(shape=(1, 64, 56, 56), seed: int = 0) -> np.ndarray:
    """The one fixed, real, in-range input every run in a sweep uses, so
    outputs are directly comparable across patches."""
    return np.random.RandomState(seed).uniform(-0.8, 0.8, shape).astype(np.float32)


def run_model(model: onnx.ModelProto, x: np.ndarray, *, tmp_dir: str) -> dict:
    """Run one model on the AX650N and return ``{"outputs": [...] | None,
    "error": str | None}``. Holds the shared device lock only for this one
    call (not the whole sweep), so other forks' device runs can interleave
    between patches rather than queue behind an entire sweep."""
    path = os.path.join(tmp_dir, "probe.axmodel")
    onnx.save(model, path)
    fd = os.open(_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _flock(fd)
        return _run_on_device_with_inputs(path, {"x": x.tobytes()})
    finally:
        os.close(fd)


def run_model_and_collect_fault_log(
    model: onnx.ModelProto, x: np.ndarray, *, tmp_dir: str, prev_syslog_text: str
) -> tuple[dict, list, str]:
    """Like :func:`run_model`, but on a ``FAULT`` result, also collects the
    device's kernel log (``docs/axera-axcl-runtime-mining.md``) *before*
    releasing the device lock, so the dump can only contain this run's own
    fault, not one from a concurrent fork's sweep that started as soon as
    the lock was released. Returns ``(result, new_fault_events,
    updated_syslog_text)`` -- pass the returned text back in as
    ``prev_syslog_text`` on the next call to keep the diff running."""
    path = os.path.join(tmp_dir, "probe.axmodel")
    onnx.save(model, path)
    fd = os.open(_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _flock(fd)
        result = _run_on_device_with_inputs(path, {"x": x.tobytes()})
        if result["outputs"] is not None or _FAULT_MARKER not in (
            result["error"] or ""
        ):
            return result, [], prev_syslog_text
        current_text = _collect_syslog_text()
        events = new_fault_events(prev_syslog_text, current_text)
        return result, events, (current_text or prev_syslog_text)
    finally:
        os.close(fd)


def classify_run(control_outputs: list[bytes], result: dict) -> str:
    """``"INERT"``, ``"FAULT"``, ``"FUNCTIONAL"``, or ``"ERROR"`` (a run that
    failed for a reason other than the known fault code -- reported, not
    silently folded into ``FAULT``)."""
    if result["outputs"] is None:
        err = result["error"] or ""
        return "FAULT" if _FAULT_MARKER in err else "ERROR"
    if result["outputs"] == control_outputs:
        return "INERT"
    return "FUNCTIONAL"


def max_abs_diff(a: list[bytes], b: list[bytes]) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    diffs = [
        float(np.abs(np.frombuffer(x, np.float32) - np.frombuffer(y, np.float32)).max())
        for x, y in zip(a, b)
    ]
    return max(diffs) if diffs else None


def health_check(
    model: onnx.ModelProto, x: np.ndarray, control_outputs: list[bytes], tmp_dir: str
) -> bool:
    """A clean control run after every patch -- confirms the device is still
    healthy and the unpatched model still behaves, per this project's
    established fault-injection discipline. Retries once on a mismatch:
    ``README.md`` documents a rare (~1-in-200-run), non-reproducible
    ``0x8030070C`` fault on a genuinely valid, unpatched model -- a
    transient recovers on retry, while a real problem (a wedged device, or
    a patch whose damage survived past its own run) reproduces on the
    second attempt too, which is what actually halts the sweep."""
    result = run_model(model, x, tmp_dir=tmp_dir)
    if result["outputs"] == control_outputs:
        return True
    result = run_model(model, x, tmp_dir=tmp_dir)
    return result["outputs"] == control_outputs


def _flock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX)


def sweep(out_path: str, *, limit: int | None = None, resume: bool = False) -> None:
    import tempfile

    model, mc = load_reference()
    cands = candidates(mc)
    if limit is not None:
        cands = cands[:limit]

    results: dict[str, dict] = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    x = fixed_input()

    def _once():
        with tempfile.TemporaryDirectory() as td:
            control = run_model(model, x, tmp_dir=td)
            if control["outputs"] is None:
                raise RuntimeError(f"control run failed: {control['error']!r}")
            control_outputs = control["outputs"]
            syslog_text = _seed_syslog_baseline()
            for cand in cands:
                offset = cand["offset"]
                key = str(offset)
                if key in results:
                    continue
                original = mc[offset]
                patched = patch_byte(mc, offset, original ^ 0xFF)
                patched_model = _model_with_mcode(model, patched)
                result, events, syslog_text = run_model_and_collect_fault_log(
                    patched_model, x, tmp_dir=td, prev_syslog_text=syslog_text
                )
                cls = classify_run(control_outputs, result)
                entry = {
                    "offset": offset,
                    "record_at": cand["record_at"],
                    "kind": cand["kind"],
                    "role": cand["role"],
                    "register": cand["register"],
                    "original": original,
                    "classification": cls,
                }
                if cls == "FUNCTIONAL":
                    entry["max_abs_diff"] = max_abs_diff(
                        control_outputs, result["outputs"]
                    )
                elif cls == "ERROR":
                    entry["error"] = result["error"]
                if cls == "FAULT":
                    entry["fault_log"] = event_summary(events)
                results[key] = entry
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                healthy = health_check(model, x, control_outputs, td)
                results[key]["health_ok"] = healthy
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                if not healthy:
                    raise RuntimeError(
                        f"device unhealthy after offset {offset}; stopping sweep"
                    )

    _once()


def finesweep(out_path: str, offset: int, values: list[int]) -> None:
    """Try several distinct patched values at one offset (not just the one
    XOR 0xFF flip), to characterize a ``FUNCTIONAL`` byte's value/output
    relationship."""
    import tempfile

    model, mc = load_reference()
    x = fixed_input()
    results: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    def _once():
        with tempfile.TemporaryDirectory() as td:
            control = run_model(model, x, tmp_dir=td)
            control_outputs = control["outputs"]
            for value in values:
                key = f"{offset}:{value:#04x}"
                if key in results:
                    continue
                patched = patch_byte(mc, offset, value)
                patched_model = _model_with_mcode(model, patched)
                result = run_model(patched_model, x, tmp_dir=td)
                cls = classify_run(control_outputs, result)
                entry = {"offset": offset, "value": value, "classification": cls}
                if cls == "FUNCTIONAL":
                    entry["max_abs_diff"] = max_abs_diff(
                        control_outputs, result["outputs"]
                    )
                results[key] = entry
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                healthy = health_check(model, x, control_outputs, td)
                if not healthy:
                    raise RuntimeError(f"device unhealthy after {key}; stopping")

    _once()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "sweep":
        out_path = argv[1]
        limit = None
        resume = "--resume" in argv
        for a in argv[2:]:
            if a.startswith("--limit"):
                limit = (
                    int(a.split("=")[1]) if "=" in a else int(argv[argv.index(a) + 1])
                )
        sweep(out_path, limit=limit, resume=resume)
        return 0
    if argv[0] == "finesweep":
        out_path, offset = argv[1], int(argv[2])
        values = [0x00, 0x01, 0x02, 0x40, 0x7F, 0x80, 0xC0, 0xFE]
        for a in argv[3:]:
            if a.startswith("--values"):
                values = [int(v, 0) for v in a.split("=")[1].split(",")]
        finesweep(out_path, offset, values)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
