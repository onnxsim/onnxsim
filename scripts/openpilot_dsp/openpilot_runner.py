#!/usr/bin/env python3
"""Usable openpilot model runner backed by tinygrad DSP codegen.

The runner keeps recurrent driving-model state as device tensors between
frames. This avoids the NumPy round trip used by the accuracy harness and is
the execution shape intended to replace an SNPE model handle.

Examples::

    DEV=DSP MOCKDSP=1 python openpilot_runner.py dm_model.onnx inputs_seg8.npz out.npz
    DEV=DSP MOCKDSP=1 python openpilot_runner.py driving.onnx inputs_seg8.npz out.npz \
      --kind driving --frames 100 --warmup 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from tinygrad_runner import TinygradSession  # noqa: E402

ACTION_T = np.array([[0.25, 0.55]], np.float32)


def _load_tuning(path: str | None, target: str) -> dict[str, str]:
    """Load the fastest candidate from a dsp_autotune cache, if supplied."""

    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    rows = {}
    for key, value in data.items():
        if key.startswith(f"{target}:") and isinstance(value, dict):
            rows.update(value)
    correct = [row for row in rows.values() if row.get("correct") and row.get("env")]
    if not correct:
        raise ValueError(f"no correct tuning result for {target} in {path}")
    best = min(correct, key=lambda row: row["seconds_at_1ghz"])
    return dict(best["env"])


class OpenpilotRunner:
    """Persistent frame runner for one driving or driver-monitoring model."""

    def __init__(self, model: str | Path, kind: str, device: str = "DSP", target: str = "snapdragon845", tuning: str | None = None):
        if kind not in ("driving", "dm"):
            raise ValueError(f"unknown openpilot model kind: {kind}")
        os.environ.update(_load_tuning(tuning, target))
        self.kind = kind
        self.session = TinygradSession(model, device, target)
        self.device = device
        self._state = None
        self._fixed = None

    def _tensor(self, value):
        return self.session._tensor(value, device=self.device)

    def _init_driving(self, road):
        inputs = {item.name: item for item in self.session.get_inputs()}
        dtype = lambda name: np.float16 if "float16" in inputs[name].type else np.float32
        self._state = {
            name: self._tensor(
                np.zeros(
                    value_shape(inputs[name]),
                    dtype=np.uint8 if "uint8" in inputs[name].type else dtype(name),
                )
            )
            for name in inputs
            if name.startswith("state_")
        }
        self._fixed = {
            "desire": self._tensor(np.zeros(value_shape(inputs["desire"]), dtype=dtype("desire"))),
            "traffic_convention": self._tensor(np.array([[1, 0]], dtype=dtype("traffic_convention"))),
            "action_t": self._tensor(ACTION_T.astype(dtype("action_t"))),
        }

    def run_driving(self, road: np.ndarray) -> np.ndarray:
        if self.kind != "driving":
            raise ValueError("run_driving requires kind='driving'")
        if self._state is None:
            self._init_driving(road)
        outputs = []
        for frame in road:
            result = self.session.run_tensors({"new_img": self._tensor(frame), **self._state, **self._fixed})
            self._state = {name: result["next_" + name] for name in self._state}
            outputs.append(result["outputs"].numpy().astype(np.float32)[0])
        return np.stack(outputs)

    def run_dm(self, driver: np.ndarray, calib=(0.0, 0.164, 0.005)) -> np.ndarray:
        if self.kind != "dm":
            raise ValueError("run_dm requires kind='dm'")
        inputs = {item.name: item for item in self.session.get_inputs()}
        ctype = np.float16 if "float16" in inputs["calib"].type else np.float32
        c = self._tensor(np.array([calib], dtype=ctype))
        outputs = []
        for frame in driver:
            result = self.session.run_tensors({"input_img": self._tensor(frame), "calib": c})
            outputs.append(result["outputs"].numpy().astype(np.float32)[0])
        return np.stack(outputs)


def value_shape(value) -> tuple[int, ...]:
    """Resolve the static openpilot input shapes used by the frame runner."""

    shape = []
    for dim in value.shape if hasattr(value, "shape") else ():
        if isinstance(dim, int):
            shape.append(dim)
        else:
            raise ValueError(f"dynamic model input is unsupported: {value.name}")
    return tuple(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("inputs", help="inputs_seg*.npz from prepare_inputs.py")
    parser.add_argument("outputs")
    parser.add_argument("--kind", choices=["driving", "dm"], required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("DEV", "DSP"))
    parser.add_argument("--target", default="snapdragon845")
    parser.add_argument("--tuning", help="dsp_autotune JSON cache")
    parser.add_argument("--calib", default="-0.00028,0.16415,0.00528")
    args = parser.parse_args()

    data = np.load(args.inputs)
    runner = OpenpilotRunner(args.model, args.kind, args.device, args.target, args.tuning)
    frames = min(args.frames, len(data["road"] if args.kind == "driving" else data["driver"]))
    inputs = data["road"][:frames] if args.kind == "driving" else data["driver"][:frames]
    if args.warmup:
        warm = inputs[: min(args.warmup, frames)]
        if args.kind == "driving":
            runner.run_driving(warm)
        else:
            runner.run_dm(warm, tuple(float(x) for x in args.calib.split(",")[-3:]))
        runner = OpenpilotRunner(args.model, args.kind, args.device, args.target, args.tuning)
    start = time.perf_counter()
    result = (
        runner.run_driving(inputs)
        if args.kind == "driving"
        else runner.run_dm(inputs, tuple(float(x) for x in args.calib.split(",")[-3:]))
    )
    elapsed = time.perf_counter() - start
    np.savez_compressed(args.outputs, outputs=result)
    print(json.dumps({"frames": frames, "seconds": elapsed, "ms_per_frame": elapsed * 1000 / frames, "shape": result.shape}))


if __name__ == "__main__":
    main()
