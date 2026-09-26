"""A persistent AXCL session: load many ``.axmodel`` files once, run them many
times, without one ``axcl_run_model`` process (and one ``lxc file push``)
per call.

The device side is ``vm/axcl_batch_runner.c``, a line-protocol runner built
inside the LXD guest (``build_runner``). Tensors travel as raw files on the
guest's virtiofs share (``/mnt/share``; the host directory is the
``share`` disk device of ``AXCL_LXD_VM``), so a 50 MB activation costs a page
cache write, not a pipe copy.

The session holds ``/tmp/axcl-device.lock`` for its whole lifetime, so other
agents' device runs queue behind it rather than interleaving.

    with AXSession() as s:
        m = s.load(axmodel_bytes)
        (y,) = s.run(m, [x])
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER_SRC = os.path.join(_HERE, "vm", "axcl_batch_runner.c")
LOCK = "/tmp/axcl-device.lock"
_PROTOCOL = frozenset({"READY", "OK", "ERR", "IN", "OUT", "END"})

# axclrtEngineDataType
_DTYPES = {
    3: np.int8,
    4: np.uint8,
    5: np.int16,
    6: np.uint16,
    7: np.int32,
    8: np.uint32,
    9: np.int64,
    13: np.float16,
    15: np.float32,
}


class DeviceError(RuntimeError):
    pass


class DeviceStall(DeviceError):
    """The runner stopped answering: treat the card as wedged (recover it with
    the full guest module reload, scripts/axera/vm/README.md)."""


class DeviceUnhealthy(DeviceStall):
    """The native health-check model returned wrong values: results since the
    last passing check are void."""


@dataclass
class IOSpec:
    name: str
    nbytes: int
    dtype: type
    shape: tuple[int, ...]
    elem_type: int = 0


@dataclass
class Model:
    id: int
    path: str
    inputs: list[IOSpec] = field(default_factory=list)
    outputs: list[IOSpec] = field(default_factory=list)


def _validate_schedule(model: Model, schedule: dict) -> None:
    """Check that a Pulsar-free schedule matches the loaded AX model IO."""
    if schedule.get("schema_version") != 1:
        raise DeviceError("unsupported or missing schedule schema_version")
    try:
        memory_size = int(schedule["memory_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise DeviceError("schedule has no valid memory plan") from error
    allocations = schedule.get("allocations")
    if not isinstance(allocations, list):
        raise DeviceError("schedule has no allocation list")
    allocation_by_name = {}
    allocation_ranges = []
    for allocation in allocations:
        try:
            name = allocation["name"]
            offset = int(allocation["offset"])
            nbytes = int(allocation["nbytes"])
            first_kernel = int(allocation["first_kernel"])
            last_kernel = int(allocation["last_kernel"])
        except (KeyError, TypeError, ValueError) as error:
            raise DeviceError("schedule contains a malformed allocation") from error
        if (
            not isinstance(name, str)
            or offset < 0
            or nbytes <= 0
            or offset % 64
            or first_kernel < 0
            or first_kernel > last_kernel
            or offset + nbytes > memory_size
        ):
            raise DeviceError(
                f"schedule allocation is outside its memory plan: {allocation}"
            )
        if name in allocation_by_name:
            raise DeviceError(f"schedule has duplicate allocation {name!r}")
        allocation_by_name[name] = allocation
        allocation_ranges.append(
            (name, offset, offset + nbytes, first_kernel, last_kernel)
        )
    kernels = schedule.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise DeviceError("schedule has no executable kernels")
    if any(
        first >= len(kernels) or last >= len(kernels)
        for _, _, _, first, last in allocation_ranges
    ):
        raise DeviceError("schedule allocation lifetime exceeds kernel list")
    for index, left in enumerate(allocation_ranges):
        for right in allocation_ranges[index + 1 :]:
            live = left[3] <= right[4] and right[3] <= left[4]
            overlaps = left[1] < right[2] and right[1] < left[2]
            if live and overlaps:
                raise DeviceError(
                    f"schedule allocations overlap while live: {left[0]!r}, {right[0]!r}"
                )
    kernel_names = [kernel.get("name") for kernel in kernels]
    if any(not isinstance(name, str) or not name for name in kernel_names):
        raise DeviceError("schedule contains a malformed kernel")
    if len(set(kernel_names)) != len(kernel_names):
        raise DeviceError("schedule contains duplicate kernel names")
    kernel_buffers = set()
    for kernel in kernels:
        inputs = kernel.get("inputs")
        output = kernel.get("output")
        if (
            not isinstance(inputs, list | tuple)
            or not isinstance(output, str)
            or not output
            or any(not isinstance(name, str) or not name for name in inputs)
        ):
            raise DeviceError(f"schedule contains a malformed kernel {kernel!r}")
        kernel_buffers.update(inputs)
        kernel_buffers.add(output)
    missing_buffers = kernel_buffers - set(allocation_by_name)
    if missing_buffers:
        raise DeviceError(
            "schedule has no allocation for kernel buffer(s): "
            + ", ".join(sorted(missing_buffers))
        )
    unused_buffers = set(allocation_by_name) - kernel_buffers
    if unused_buffers:
        raise DeviceError(
            "schedule contains allocation(s) unused by kernels: "
            + ", ".join(sorted(unused_buffers))
        )
    kernel_index = {name: index for index, name in enumerate(kernel_names)}
    dependencies = schedule.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise DeviceError("schedule dependencies must be a list")
    for dependency in dependencies:
        if (
            not isinstance(dependency, list | tuple)
            or len(dependency) != 2
            or dependency[0] not in kernel_index
            or dependency[1] not in kernel_index
            or kernel_index[dependency[0]] >= kernel_index[dependency[1]]
        ):
            raise DeviceError(f"schedule contains an invalid dependency {dependency!r}")
    for kind, specs in (("inputs", model.inputs), ("outputs", model.outputs)):
        entries = schedule.get(kind)
        if not isinstance(entries, list) or len(entries) != len(specs):
            raise DeviceError(
                f"schedule {kind} count does not match model ({len(entries or [])} != {len(specs)})"
            )
        for index, (spec, entry) in enumerate(zip(specs, entries)):
            expected = (
                entry.get("name"),
                tuple(entry.get("shape", ())),
                int(entry.get("elem_type", -1)),
                int(entry.get("nbytes", -1)),
            )
            actual = (spec.name, spec.shape, spec.elem_type, spec.nbytes)
            if expected != actual:
                raise DeviceError(
                    f"schedule {kind}[{index}] does not match model: "
                    f"scheduled={expected}, loaded={actual}"
                )
            allocation = allocation_by_name.get(spec.name)
            if allocation is None:
                raise DeviceError(
                    f"schedule has no allocation for model {kind[:-1]} {spec.name!r}"
                )
    if memory_size <= 0:
        raise DeviceError("schedule has no positive memory plan")


def _vm_share(vm: str, device: str = "share") -> tuple[str, str]:
    """``(host source, guest path)`` of the VM's ``share`` disk device."""

    def get(key: str) -> str:
        return subprocess.run(
            ["lxc", "config", "device", "get", vm, device, key],
            capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip

    src, path = get("source"), get("path")
    if not src or not path:
        raise DeviceError(f"{vm} has no {device!r} disk device")
    return src, path


class AXSession:
    def __init__(
        self,
        vm: str | None = None,
        subdir: str = "step_runner",
        timeout: float = 120.0,
        lock: bool = True,
    ):
        self.vm = vm or os.environ.get("AXCL_LXD_VM") or "axcl-vm"
        host_share, guest_share = _vm_share(self.vm)
        self.host_dir = os.path.join(host_share, subdir)
        self.guest_dir = f"{guest_share}/{subdir}"
        self.timeout = timeout
        self._lock_wanted = lock
        self._lock_fd = None
        self._proc = None
        self._lines: queue.Queue = queue.Queue()
        self._seq = itertools.count()
        self.exec_us = 0
        self.runs = 0
        self._resident_inputs: dict[int, set[int]] = {}

    # -- lifecycle ---------------------------------------------------------
    def build_runner(self) -> None:
        os.makedirs(self.host_dir, exist_ok=True)
        shutil.copy(RUNNER_SRC, os.path.join(self.host_dir, "axcl_batch_runner.c"))
        res = subprocess.run(
            [
                "lxc", "exec", self.vm, "--", "gcc", "-O2", "-o",
                f"{self.guest_dir}/axrun", f"{self.guest_dir}/axcl_batch_runner.c",
                "-I/usr/include/axcl", "-L/usr/lib/axcl", "-laxcl_rt", "-laxcl_sys",
                "-Wl,-rpath,/usr/lib/axcl",
            ],
            capture_output=True, text=True,
        )  # fmt: skip
        if res.returncode:
            raise DeviceError("runner build failed: " + res.stdout + res.stderr)

    def __enter__(self) -> AXSession:
        os.makedirs(os.path.join(self.host_dir, "t"), exist_ok=True)
        if not os.path.exists(os.path.join(self.host_dir, "axrun")):
            self.build_runner()
        if self._lock_wanted:
            import fcntl  # POSIX-only; imported here so this module (and its tests) import on Windows

            self._lock_fd = open(LOCK, "w")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        self._proc = subprocess.Popen(
            ["lxc", "exec", self.vm, "--", f"{self.guest_dir}/axrun"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )  # fmt: skip
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._line(60)
        if not ready.startswith("READY"):
            self.close()
            raise DeviceError(f"runner did not start: {ready}")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(30)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._lock_fd is not None:
            import fcntl

            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    def _pump(self) -> None:
        # The AXCL runtime logs to stdout too; keep only protocol lines.
        for line in self._proc.stdout:
            if line.split(" ", 1)[0].strip() in _PROTOCOL:
                self._lines.put(line.rstrip("\n"))
        self._lines.put(None)

    def _line(self, timeout: float | None = None) -> str:
        try:
            line = self._lines.get(timeout=timeout or self.timeout)
        except queue.Empty:
            raise DeviceStall("runner timed out") from None
        if line is None:
            raise DeviceStall("runner exited")
        return line

    def _cmd(self, text: str) -> str:
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()
        line = self._line()
        if line.startswith("ERR"):
            raise DeviceError(line)
        return line

    # -- models ------------------------------------------------------------
    def load(self, model: bytes | str, schedule_path: str | None = None) -> Model:
        """Load an AX model, optionally enforcing its Pulsar-free schedule."""
        n = next(self._seq)
        name = f"m{n}.axmodel"
        dst = os.path.join(self.host_dir, name)
        if isinstance(model, (bytes, bytearray)):
            with open(dst, "wb") as f:
                f.write(model)
        else:
            shutil.copy(model, dst)
        head = self._cmd(f"LOAD {self.guest_dir}/{name}")
        m = Model(id=int(head.split()[1]), path=dst)
        while (line := self._line()) != "END":
            kind, _, tname, nbytes, dt, *dims = line.split()
            spec = IOSpec(
                tname, int(nbytes), _DTYPES.get(int(dt), np.uint8),
                tuple(int(d) for d in dims), int(dt),
            )  # fmt: skip
            (m.inputs if kind == "IN" else m.outputs).append(spec)
        if schedule_path is not None:
            try:
                with open(schedule_path, encoding="utf-8") as stream:
                    _validate_schedule(m, json.load(stream))
            except Exception:
                self._cmd(f"UNLOAD {m.id}")
                try:
                    os.remove(dst)
                except OSError:
                    pass
                raise
        return m

    def unload(self, m: Model) -> None:
        self._cmd(f"UNLOAD {m.id}")
        self._resident_inputs.pop(m.id, None)
        try:
            os.remove(m.path)
        except OSError:
            pass

    def run(
        self,
        m: Model,
        inputs: list[np.ndarray],
        resident_pairs: tuple[tuple[int, int], ...] = (),
    ) -> list[np.ndarray]:
        """Execute a model, optionally retaining input/output pairs on device.

        After the first run, each resident output is copied directly into its
        input buffer on the device and later host uploads for that input are
        skipped. Outputs are still copied back for compatibility and metrics.
        """
        if len(inputs) != len(m.inputs):
            raise ValueError(f"model takes {len(m.inputs)} inputs, got {len(inputs)}")
        pairs = tuple(resident_pairs)
        if any(
            not (0 <= i < len(m.inputs) and 0 <= o < len(m.outputs))
            for i, o in pairs
        ):
            raise ValueError(f"invalid resident pairs {pairs}")
        resident = self._resident_inputs.setdefault(m.id, set())
        ins = []
        for k, (x, spec) in enumerate(zip(inputs, m.inputs)):
            buf = np.ascontiguousarray(x, dtype=spec.dtype).tobytes()
            if len(buf) != spec.nbytes:
                raise ValueError(
                    f"input {spec.name}: {len(buf)} bytes, model wants {spec.nbytes}"
                )
            p = f"t/i{k}.bin"
            if k not in resident:
                with open(os.path.join(self.host_dir, p), "wb") as f:
                    f.write(buf)
            ins.append(f"{self.guest_dir}/{p}")
        outs = [f"t/o{k}.bin" for k in range(len(m.outputs))]
        if pairs:
            pair_args = " ".join(f"{i} {o}" for i, o in pairs)
            line = self._cmd(
                f"RUNR {m.id} {len(ins)} {' '.join(ins)} {len(pairs)} {pair_args} "
                f"{len(outs)} "
                + " ".join(f"{self.guest_dir}/{p}" for p in outs)
            )
        else:
            line = self._cmd(
                f"RUN {m.id} {len(ins)} {' '.join(ins)} {len(outs)} "
                + " ".join(f"{self.guest_dir}/{p}" for p in outs)
            )
        self.exec_us += int(line.split()[1])
        self.runs += 1
        res = []
        for p, spec in zip(outs, m.outputs):
            a = np.fromfile(os.path.join(self.host_dir, p), dtype=spec.dtype)
            res.append(a.reshape(spec.shape) if spec.shape else a)
        for i, _ in pairs:
            resident.add(i)
        return res


HEALTH_MODEL = os.path.join(
    _HERE, "fixtures", "elementwise_scale_emit", "relu_16x512x7x7_x128_y128.axmodel.gz"
)
HEALTH_SCALE, HEALTH_ZP = 0.007843137718737125, 128  # that template's calibration


def health_check(s: AXSession) -> float:
    """Run the native (unpatched) ``x128,y128`` Relu template on [-1, 1] data
    and return its worst error in LSBs; raise ``DeviceError`` above 1 LSB (a
    wedged runtime returns garbage or stalls)."""
    import gzip

    with gzip.open(HEALTH_MODEL, "rb") as f:
        m = s.load(f.read())
    try:
        spec = m.inputs[0]
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, spec.shape).astype(np.float32)
        (y,) = s.run(m, [x])
        q = np.clip(np.rint(x / np.float32(HEALTH_SCALE)) + HEALTH_ZP, 0, 255)
        want = np.maximum((q - HEALTH_ZP) * np.float32(HEALTH_SCALE), 0)
        lsb = float(np.abs(y.astype(np.float32) - want).max() / HEALTH_SCALE)
        if lsb > 1.01:
            raise DeviceUnhealthy(f"health check: native Relu off by {lsb:.2f} LSB")
        return lsb
    finally:
        s.unload(m)


if __name__ == "__main__":  # smoke: python axcl_session.py model.axmodel
    import sys

    with AXSession() as s:
        t0 = time.time()
        mm = s.load(sys.argv[1])
        print(json.dumps({"load_s": time.time() - t0, "in": [vars(i) for i in mm.inputs],
                          "out": [vars(o) for o in mm.outputs]}, default=str))  # fmt: skip
        xs = [np.random.rand(*i.shape).astype(i.dtype) for i in mm.inputs]
        t0 = time.time()
        ys = s.run(mm, xs)
        print("run_s", time.time() - t0, "exec_us", s.exec_us, [y.shape for y in ys])
