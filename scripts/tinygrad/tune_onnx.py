#!/usr/bin/env python3
"""Compare isolated tinygrad BEAM configurations on static float ONNX models.

Times synchronized TinyJit replay with inputs resident on the device. Checks
three distinct inputs against single-thread ONNX Runtime, including JIT replay.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import onnxruntime as ort
    from tinygrad import Device, Tensor, TinyJit
    from tinygrad.nn.onnx import OnnxRunner

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    feeds = []
    for _ in range(3):
        feed = {}
        for inp in session.get_inputs():
            if inp.type != "tensor(float)" or any(not isinstance(d, int) or d <= 0 for d in inp.shape):
                raise ValueError(f"requires static float32 inputs: {inp.name} {inp.type} {inp.shape}")
            feed[inp.name] = rng.normal(size=inp.shape).astype(np.float32)
        feeds.append(feed)
    runner = OnnxRunner(args.model)
    names = [o.name for o in session.get_outputs()]

    @TinyJit
    def step(*inputs):
        outputs = runner(dict(zip(feeds[0], inputs)))
        return tuple(outputs[n].realize() for n in names)

    device = Device[args.device]
    setup = time.perf_counter()
    max_error = 0.0
    for feed in feeds:
        tensors = tuple(Tensor(v, device=args.device).realize() for v in feed.values())
        got = step(*tensors)
        for actual, expected in zip(got, session.run(names, feed)):
            actual = actual.numpy()
            if actual.shape != expected.shape:
                raise ValueError(f"output shape mismatch: {actual.shape} != {expected.shape}")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, equal_nan=False)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    device.synchronize()
    setup_s = time.perf_counter() - setup
    for _ in range(3):
        step(*tensors)
        device.synchronize()
    samples = []
    for _ in range(args.runs):
        device.synchronize()
        start = time.perf_counter()
        step(*tensors)
        device.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    ort_samples = []
    for _ in range(3):
        session.run(names, feeds[-1])
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(names, feeds[-1])
        ort_samples.append((time.perf_counter() - start) * 1000)
    return dict(model=str(Path(args.model).resolve()),
                model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                platform=platform.platform(), device=args.device,
                beam=int(os.environ["BEAM"]), tinygrad=importlib.metadata.version("tinygrad"),
                onnxruntime=ort.__version__, max_abs_error=max_error, trials=len(feeds),
                setup_s=setup_s, median_ms=statistics.median(samples), min_ms=min(samples),
                samples_ms=samples, ort_single_thread_median_ms=statistics.median(ort_samples),
                timing="synchronized resident-input TinyJit replay; ORT uses host arrays")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--beam", default="0,2")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    results = []
    for beam in map(int, args.beam.split(",")):
        env = dict(os.environ, DEV=args.device, BEAM=str(beam))
        command = [sys.executable, str(Path(__file__).resolve()), args.model,
                   "--device", args.device, "--runs", str(args.runs), "--worker"]
        try:
            process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
            if process.returncode:
                raise RuntimeError(process.stderr[-6000:])
            result = json.loads(process.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            result = dict(model=args.model, device=args.device, beam=beam, error=str(error))
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in result for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
