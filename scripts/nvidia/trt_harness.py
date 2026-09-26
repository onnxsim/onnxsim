"""Build and time ONNX models with the real TensorRT builder (Jetson / NVIDIA GPU).

Deliberately depends only on ``tensorrt``, ``onnx`` and ``numpy`` -- not on
``onnxsim`` -- because JetPack's TensorRT Python bindings are built for the
system Python (3.10 on JetPack 6), while onnxsim itself needs Python >= 3.11.
Models are exchanged as ``.onnx`` files: ``qdq_pairs.py`` (run under the onnxsim
interpreter) writes ``<name>.orig.onnx`` / ``<name>.sim.onnx`` pairs, and this
script (run under the TensorRT interpreter) consumes them.

    python trt_harness.py build model.onnx [--fp16] [--int8]
    python trt_harness.py compare DIR [--fp16] [--int8]
    python trt_harness.py profile model.engine [--iters 50]

CUDA is reached through ``libcudart`` via ctypes so no pycuda / cuda-python
install is needed.
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.ERROR)


class Cudart:
    """Minimal ctypes wrapper over the handful of runtime calls we need."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so.12")

    def check(self, err):
        if err != 0:
            self.lib.cudaGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"CUDA error {err}: {self.lib.cudaGetErrorString(err).decode()}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr

    def free(self, ptr):
        self.check(self.lib.cudaFree(ptr))

    def memcpy_htod(self, dst, host):
        self.check(self.lib.cudaMemcpy(dst, host.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(host.nbytes), 1))

    def memcpy_dtoh(self, host, src):
        self.check(self.lib.cudaMemcpy(host.ctypes.data_as(ctypes.c_void_p), src,
                                       ctypes.c_size_t(host.nbytes), 2))

    def sync(self):
        self.check(self.lib.cudaDeviceSynchronize())


def build_engine(onnx_path, fp16=False, int8=False, workspace_mb=1024):
    """Parse + build. Returns (serialized_engine | None, info dict)."""
    info = {"onnx": str(onnx_path), "fp16": fp16, "int8": int8}
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        info["error"] = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        return None, info
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # Explicit Q/DQ needs no calibrator, but a graph without any Q/DQ nodes does
    # (TensorRT fails with "no scaling factors"), so only enable INT8 when present.
    has_qdq = any(n.op_type in ("QuantizeLinear", "DequantizeLinear")
                  for n in onnx.load(str(onnx_path), load_external_data=False).graph.node)
    info["int8_effective"] = bool(int8 and has_qdq)
    if int8 and has_qdq:
        config.set_flag(trt.BuilderFlag.INT8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    info["build_s"] = round(time.perf_counter() - t0, 2)
    if blob is None:
        info["error"] = "build_serialized_network returned None"
        return None, info
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(bytes(blob))
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    info["layers"] = [_layer_summary(l) for l in layers]
    info["n_layers"] = len(layers)
    info["n_int8_layers"] = sum("Int8" in json.dumps(l) or "int8" in json.dumps(l).lower()
                                for l in layers)
    info["engine_bytes"] = len(bytes(blob))
    return bytes(blob), info


def _layer_summary(layer):
    if isinstance(layer, str):
        return {"name": layer}
    return {"name": layer.get("Name"), "type": layer.get("LayerType"),
            "tactic": layer.get("TacticName"),
            "in": [t.get("Format/Datatype") for t in layer.get("Inputs", [])],
            "out": [t.get("Format/Datatype") for t in layer.get("Outputs", [])]}


def run_engine(blob, feeds=None, iters=50, warmup=10, seed=0):
    """Run the engine on random (or given) inputs. Returns (outputs, mean_ms)."""
    cuda = Cudart()
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()
    rng = np.random.default_rng(seed)
    bufs, host_out = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        bufs[name] = cuda.malloc(nbytes)
        ctx.set_tensor_address(name, bufs[name].value)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            x = (feeds or {}).get(name)
            if x is None:
                x = rng.standard_normal(shape).astype(dtype)
            cuda.memcpy_htod(bufs[name], np.ascontiguousarray(x))
        else:
            host_out[name] = np.empty(shape, dtype=dtype)
    stream = ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)))
    for _ in range(warmup):
        ctx.execute_async_v3(stream.value)
    cuda.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.value)
    cuda.sync()
    mean_ms = (time.perf_counter() - t0) / iters * 1e3
    for name, h in host_out.items():
        cuda.memcpy_dtoh(h, bufs[name])
    for p in bufs.values():
        cuda.free(p)
    return host_out, mean_ms


def _cmd_build(args):
    blob, info = build_engine(args.onnx, args.fp16, args.int8)
    if args.json:
        print(json.dumps(info, indent=1))
    else:
        for k, v in info.items():
            if k != "layers":
                print(f"{k}: {v}")
        for l in info.get("layers", []):
            print("  ", l)
    return 0 if blob else 1


def _cmd_compare(args):
    """For every ``X.orig.onnx``/``X.sim.onnx`` pair: build both, check the
    engine outputs agree on identical inputs, report layers/latency."""
    rc = 0
    for orig in sorted(Path(args.dir).glob("*.orig.onnx")):
        sim = orig.with_name(orig.name.replace(".orig.", ".sim."))
        name = orig.name[: -len(".orig.onnx")]
        res = {}
        for tag, path in (("orig", orig), ("sim", sim)):
            blob, info = build_engine(path, args.fp16, args.int8)
            if blob is None:
                res[tag] = {"error": info.get("error")}
                continue
            # Same seed -> same random inputs for both models (identical input specs).
            outs, ms = run_engine(blob, iters=args.iters)
            res[tag] = {"layers": info["n_layers"], "ms": round(ms, 4), "outs": outs}
        if "error" in res["orig"] or "error" in res["sim"]:
            print(f"{name}: BUILD FAILED  orig={res['orig'].get('error')}  sim={res['sim'].get('error')}")
            rc = 1
            continue
        diff = max(float(np.max(np.abs(res["orig"]["outs"][k] - res["sim"]["outs"][k])))
                   for k in res["orig"]["outs"])
        print(f"{name}: layers {res['orig']['layers']}->{res['sim']['layers']}  "
              f"ms {res['orig']['ms']}->{res['sim']['ms']}  max|diff|={diff:.3g}")
    return rc


def _cmd_profile(args):
    """Measure a serialized engine and emit a runner-friendly JSON record."""
    blob = Path(args.engine).read_bytes()
    outputs, mean_ms = run_engine(blob, iters=args.iters, warmup=args.warmup)
    result = {
        "engine": str(args.engine),
        "iterations": args.iters,
        "warmup": args.warmup,
        "mean_ms": round(mean_ms, 4),
        "outputs": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in outputs.items()
        },
    }
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"engine: {args.engine}\nmean_ms: {result['mean_ms']}\n"
              f"iterations: {args.iters}\nwarmup: {args.warmup}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", _cmd_build), ("compare", _cmd_compare),
                     ("profile", _cmd_profile)):
        sp = sub.add_parser(name)
        sp.add_argument("onnx" if name == "build" else
                        "dir" if name == "compare" else "engine")
        sp.add_argument("--fp16", action="store_true")
        sp.add_argument("--int8", action="store_true")
        sp.add_argument("--iters", type=int, default=50)
        sp.add_argument("--warmup", type=int, default=10)
        sp.add_argument("--json", action="store_true")
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
