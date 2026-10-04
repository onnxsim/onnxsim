#!/usr/bin/env python3
"""Minimal RKNN Runtime runner for a NanoPC-T6 (RK3588) board.

This file is intentionally dependency-free apart from ``numpy`` and is copied
to the target.  The FriendlyElec Ubuntu 24.04 image ships ``librknnrt.so``
(2.3.0) but no RKNN-Lite Python package, so ctypes is used for the small,
stable runtime API -- the same approach as ``luckfox_rknn_runner.py``.

Differences from the Luckfox/RV1106 runner, all verified on a connected
NanoPC-T6 (RK3588, Linux 6.1.141, FriendlyElec Ubuntu 24.04.5, RKNPU driver
v0.9.8):

* **The device node is a DRM render node, not ``/dev/rknpu``.**  This image's
  RKNPU driver registers as a DRM device (``dmesg``: ``[drm] Initialized rknpu
  0.9.8 20240828 for fdab0000.npu on minor 1``), so ``/dev/dri/card1`` and
  ``/dev/dri/renderD129`` are the NPU's nodes and ``/dev/rknpu`` does not
  exist.  ``librknnrt.so`` 2.3.0 finds it by itself -- confirmed by reading
  its strings, which contain both ``/dev/dri/%s``/``%s/renderD%d`` and the
  legacy ``/dev/rknpu`` fallback -- so no device path needs to be passed in.
  ``librknnrt.so`` is discovered at ``/usr/lib/librknnrt.so``.
* **Multiple inputs and outputs are supported.**  The RV1106 image only
  needed the single-input/single-output case; the original-vs-simplified
  correctness check needs every output dumped to a file, and models can take
  more than one input, so both counts are queried rather than assumed.
* **Input data is read from a file** instead of being left as zeros, so the
  runner can consume the exact feeds used for the ONNX Runtime reference.
* Outputs are written as raw float32 (or float16/int8, matching the
  attribute) into ``--output-dir``, which is what makes the real-NPU-vs-ORT
  numeric comparison possible.

Example on the board::

    python3 nanopc_rknn_runner.py model.rknn --input-feeds feeds.npz \\
        --output-dir outputs

The model must already have been compiled on the host with
``rknn.config(target_platform="rk3588")`` and ``rknn.export_rknn()``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time

import numpy as np

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 4
RKNN_QUERY_PERF_RUN = 5

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4

RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1

_RKNN_SUCCESS = 0

# Rockchip's ``rknn_soc_id`` enum, as documented in librknnrt's public header.
# Only the ids that can plausibly be reported by ``rknn_get_device_properties``
# are listed; anything else is reported as ``unknown(<id>)``.
_SOC_NAMES = {
    0: "RKNN_SOC_RK1808",
    1: "RKNN_SOC_RK1808A",
    2: "RKNN_SOC_RK3588",
    3: "RKNN_SOC_RK3588A",
    4: "RKNN_SOC_RK3576",
    5: "RKNN_SOC_RK3566",
    6: "RKNN_SOC_RK3568",
    7: "RKNN_SOC_RK3562",
    8: "RKNN_SOC_RK1106",
    9: "RKNN_SOC_RV1103",
    10: "RKNN_SOC_RV1106B",
    11: "RKNN_SOC_RV1126B",
    12: "RKNN_SOC_RV1109B",
    13: "RKNN_SOC_RV1116B",
}


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("_padding", ctypes.c_uint8 * 3),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through_attr", ctypes.c_uint8),
        ("_padding2", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnTensorMem(ctypes.Structure):
    # ARM EABI aligns uint64_t to 8 bytes; ctypes would otherwise apply the
    # host ABI's alignment and shift every field after phys_addr.
    _pack_ = 8
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("_padding", ctypes.c_uint8 * 2),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


def _fail(api: str, ret: int) -> None:
    if ret != _RKNN_SUCCESS:
        raise RuntimeError(f"{api} failed with {ret}")


def _declare(lib: ctypes.CDLL) -> None:
    lib.rknn_init.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.rknn_init.restype = ctypes.c_int
    lib.rknn_query.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.rknn_query.restype = ctypes.c_int
    lib.rknn_create_mem.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.rknn_create_mem.restype = ctypes.POINTER(RknnTensorMem)
    lib.rknn_set_io_mem.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(RknnTensorMem),
        ctypes.POINTER(RknnTensorAttr),
    ]
    lib.rknn_set_io_mem.restype = ctypes.c_int
    lib.rknn_destroy_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(RknnTensorMem)]
    lib.rknn_destroy_mem.restype = ctypes.c_int
    lib.rknn_run.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_run.restype = ctypes.c_int
    lib.rknn_destroy.argtypes = [ctypes.c_uint32]
    lib.rknn_destroy.restype = ctypes.c_int


def _device_info(lib: ctypes.CDLL) -> dict:
    """Report what the NPU actually is, from ``rknn_get_device_properties``.

    Verified on the connected NanoPC-T6: this ``librknnrt.so`` (2.3.0) exports
    **no** ``rknn_get_sdk_version`` symbol (unlike the newer RKNN-Lite builds),
    so the runtime version comes from the library's own embedded string instead
    (see :func:`_library_version`). The struct below follows Rockchip's
    documented ``rknn_device_properties``, whose first field is the
    ``rknn_soc_id`` **enum**, not a string -- reading it as ``char[32]``
    returned ``'\\x02'``, which is in fact ``RKNN_SOC_RK3588 == 2`` and thus
    confirms the layout bug and the device in one go.
    """

    class RknnDeviceProperties(ctypes.Structure):
        _fields_ = [
            ("soc_id", ctypes.c_int),
            ("cores", ctypes.c_uint32),
            ("core_clk", ctypes.c_uint32 * 4),
            ("npu_freq", ctypes.c_uint32),
            ("ram_size", ctypes.c_uint64),
            ("spi_flash_size", ctypes.c_uint64),
            ("psram_size", ctypes.c_uint64),
        ]

    info: dict = {}
    if hasattr(lib, "rknn_get_device_properties"):
        lib.rknn_get_device_properties.argtypes = [
            ctypes.POINTER(RknnDeviceProperties), ctypes.c_uint32
        ]
        lib.rknn_get_device_properties.restype = ctypes.c_int
        props = RknnDeviceProperties()
        if (
            lib.rknn_get_device_properties(ctypes.byref(props), ctypes.sizeof(props))
            == _RKNN_SUCCESS
        ):
            info["soc"] = _SOC_NAMES.get(props.soc_id, f"unknown({props.soc_id})")
            info["soc_id"] = int(props.soc_id)
            info["cores"] = int(props.cores)
            info["npu_freq"] = int(props.npu_freq)
            info["ram_size"] = int(props.ram_size)
    return info


def _library_version(library: str) -> str:
    """Read ``librknnrt version: X.Y.Z`` straight out of the shipped binary.

    The 2.3.0 library exports no ``rknn_get_sdk_version``, so the only
    way to report which runtime actually executed the model is to look for
    the version string it embeds.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["strings", library], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return "unknown"
    match = re.search(r"librknnrt version: ([0-9][0-9.]*)", out)
    return match.group(1) if match else "unknown"


def _fmt_to_str(fmt: int) -> str:
    return {RKNN_TENSOR_NCHW: "nchw", RKNN_TENSOR_NHWC: "nhwc"}.get(fmt, str(fmt))


def _type_to_str(tensor_type: int) -> str:
    return {
        RKNN_TENSOR_FLOAT32: "float32",
        RKNN_TENSOR_FLOAT16: "float16",
        RKNN_TENSOR_INT8: "int8",
        RKNN_TENSOR_UINT8: "uint8",
        RKNN_TENSOR_INT16: "int16",
    }.get(tensor_type, str(tensor_type))


_TYPE_BYTES = {
    RKNN_TENSOR_FLOAT32: 4,
    RKNN_TENSOR_FLOAT16: 2,
    RKNN_TENSOR_INT8: 1,
    RKNN_TENSOR_UINT8: 1,
    RKNN_TENSOR_INT16: 2,
}


def _np_dtype(tensor_type: int) -> np.dtype:
    return np.dtype(
        {
            RKNN_TENSOR_FLOAT32: np.float32,
            RKNN_TENSOR_FLOAT16: np.float16,
            RKNN_TENSOR_INT8: np.int8,
            RKNN_TENSOR_UINT8: np.uint8,
            RKNN_TENSOR_INT16: np.int16,
        }.get(tensor_type, np.uint8)
    )


def _float_buffer_size(attr: RknnTensorAttr) -> int:
    """Bytes a ``want_float`` buffer needs for ``attr``.

    ``rknn_build()`` rewrites these models' default I/O dtype to int8 (it warns
    about this at build time), so ``attr.type``/``attr.size`` describe an 8-bit
    view: ``size_with_stride`` is that view's **stride-padded** extent, and the
    runtime's float32 conversion wants the same padding at 4 bytes per element.
    Sizing from ``n_elems`` alone silently under-allocates whenever the model's
    layout is padded -- verified on the connected NanoPC-T6, where
    ``sigmoid_mul_swish``'s input is ``[1,12,12,3]`` int8 with
    ``size=432`` but ``size_with_stride=576``::

        E RKNN: rknn_set_io_mem, input memory size(1728) < model input size(2304)

    (1728 = 432*4, 2304 = 576*4). ``conv_bn_relu``, whose input has no padding
    (``size_with_stride == size``), happens to work either way, which is why
    this only shows up on some models.
    """
    base_bytes = _TYPE_BYTES.get(attr.type, 4)
    padded = attr.size_with_stride or attr.size or (attr.n_elems * base_bytes)
    return int(padded) * 4 // base_bytes


def _attr_dims(attr: RknnTensorAttr) -> list[int]:
    return [int(attr.dims[i]) for i in range(min(attr.n_dims, 16))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="compiled .rknn model")
    ap.add_argument(
        "--input-feeds",
        default=None,
        help="npz file of input arrays, applied in the model's input order",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="directory to write each output as <index>.bin plus outputs.json",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--library", default="/usr/lib/librknnrt.so")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model = f.read()
    # Keep the backing buffer alive for the whole runtime lifetime.
    model_buf = ctypes.create_string_buffer(model)

    lib = ctypes.CDLL(args.library)
    _declare(lib)

    feeds = np.load(args.input_feeds) if args.input_feeds else {}

    ctx = ctypes.c_uint32(0)
    _fail("rknn_init", lib.rknn_init(ctypes.byref(ctx), model_buf, len(model), 0, None))

    input_mems: list = []
    output_mems: list = []
    try:
        io_num = RknnInputOutputNum()
        _fail(
            "rknn_query(IN_OUT_NUM)",
            lib.rknn_query(
                ctx,
                RKNN_QUERY_IN_OUT_NUM,
                ctypes.byref(io_num),
                ctypes.sizeof(io_num),
            ),
        )
        if io_num.n_input == 0:
            raise RuntimeError("model reports no inputs")

        # Inputs: allocate first, then bind, so the driver has the full set
        # before it validates the graph's expected input list.
        input_specs = []
        for i in range(io_num.n_input):
            attr = RknnTensorAttr(index=i)
            _fail(
                f"rknn_query(input_attr[{i}])",
                lib.rknn_query(
                    ctx,
                    RKNN_QUERY_INPUT_ATTR,
                    ctypes.byref(attr),
                    ctypes.sizeof(attr),
                ),
            )
            input_specs.append(attr)

        for i, attr in enumerate(input_specs):
            # The buffer must be sized for the *float32* view of the tensor's
            # stride-padded extent, not its bare element count -- see
            # _float_buffer_size for the verified failure this avoids.
            mem_size = _float_buffer_size(attr)
            mem = lib.rknn_create_mem(ctx, mem_size)
            if not mem:
                raise RuntimeError(f"rknn_create_mem returned null for input {i}")
            input_mems.append(mem)
            ctypes.memset(mem.contents.virt_addr, 0, mem_size)

            # want_float=1 asks the runtime to convert the quantized internal
            # buffer to float on write, so the feeds can be plain float arrays
            # regardless of the model's internal quantization.
            attr.pass_through = 0
            attr.type = RKNN_TENSOR_FLOAT32
            if len(_attr_dims(attr)) == 4:
                attr.fmt = RKNN_TENSOR_NHWC
            _fail(
                f"rknn_set_io_mem(input[{i}])",
                lib.rknn_set_io_mem(ctx, mem, ctypes.byref(attr)),
            )

        output_specs = []
        for i in range(io_num.n_output):
            attr = RknnTensorAttr(index=i)
            _fail(
                f"rknn_query(output_attr[{i}])",
                lib.rknn_query(
                    ctx,
                    RKNN_QUERY_OUTPUT_ATTR,
                    ctypes.byref(attr),
                    ctypes.sizeof(attr),
                ),
            )
            output_specs.append(attr)

        for i, attr in enumerate(output_specs):
            # Same float32-view sizing as the inputs, for the same reason.
            mem_size = _float_buffer_size(attr)
            mem = lib.rknn_create_mem(ctx, mem_size)
            if not mem:
                raise RuntimeError(f"rknn_create_mem returned null for output {i}")
            output_mems.append(mem)
            ctypes.memset(mem.contents.virt_addr, 0, mem_size)
            attr.pass_through = 0
            attr.type = RKNN_TENSOR_FLOAT32
            _fail(
                f"rknn_set_io_mem(output[{i}])",
                lib.rknn_set_io_mem(ctx, mem, ctypes.byref(attr)),
            )

        applied = []
        for i, attr in enumerate(input_specs):
            name = attr.name.decode(errors="replace")
            source = "zeros"
            if name in feeds:
                array = np.ascontiguousarray(feeds[name], dtype=np.float32)
                source = "feed"
            else:
                # Fall back to positional order when the .rknn metadata carries
                # no usable name (the runtime preserves export-time names, but
                # a hand-built model may not).
                keys = list(feeds.keys())
                if i < len(keys):
                    name = keys[i]
                    array = np.ascontiguousarray(feeds[name], dtype=np.float32)
                    source = "feed[positional]"
                elif feeds:
                    # Real feeds were supplied but none of them matched this
                    # input: silently benchmarking zeros would be misleading.
                    raise RuntimeError(
                        f"no feed for input {i} ({attr.name!r}); "
                        f"available: {sorted(feeds)}"
                    )
                else:
                    # No --input-feeds at all: a pure latency benchmark, matching
                    # the Luckfox runner's zero-filled default.
                    array = np.zeros(_attr_dims(attr), dtype=np.float32)
            _write_input(lib, ctx, input_mems[i], array, attr)
            applied.append(
                {
                    "index": i,
                    "name": name,
                    "source": source,
                    "shape": list(array.shape),
                    "model_shape": _attr_dims(attr),
                    "model_fmt": _fmt_to_str(attr.fmt),
                }
            )

        for _ in range(args.warmup):
            _fail("rknn_run", lib.rknn_run(ctx, None))

        samples = []
        for _ in range(args.iterations):
            t0 = time.perf_counter_ns()
            _fail("rknn_run", lib.rknn_run(ctx, None))
            samples.append((time.perf_counter_ns() - t0) / 1e6)

        outputs = []
        for i, (attr, mem) in enumerate(zip(output_specs, output_mems)):
            dtype = _np_dtype(attr.type)
            count = attr.n_elems
            buf = (ctypes.c_uint8 * mem.contents.size).from_address(
                mem.contents.virt_addr
            )
            array = np.frombuffer(bytes(bytearray(buf)), dtype=dtype, count=count)
            array = array.reshape(_attr_dims(attr)) if attr.n_dims else array
            entry = {
                "index": i,
                "name": attr.name.decode(errors="replace"),
                "shape": list(array.shape),
                "dtype": str(dtype),
                "model_fmt": _fmt_to_str(attr.fmt),
            }
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                path = os.path.join(args.output_dir, f"{i}.bin")
                with open(path, "wb") as f:
                    f.write(array.astype(np.float32).tobytes())
                entry["path"] = path
            outputs.append(entry)

        result = {
            "model": os.path.basename(args.model),
            "so": "rk3588",
            "librknnrt_version": _library_version(args.library),
            "device": _device_info(lib),
            "n_input": int(io_num.n_input),
            "n_output": int(io_num.n_output),
            "inputs": applied,
            "outputs": outputs,
            "iterations": len(samples),
            "latency_ms": {
                "min": min(samples),
                "mean": statistics.mean(samples),
                "p50": statistics.median(samples),
                "p95": _percentile(samples, 0.95),
                "max": max(samples),
            },
        }
        if args.output_dir:
            with open(os.path.join(args.output_dir, "outputs.json"), "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)
        print(json.dumps(result, sort_keys=True))
    finally:
        for mem in output_mems:
            lib.rknn_destroy_mem(ctx, mem)
        for mem in input_mems:
            lib.rknn_destroy_mem(ctx, mem)
        lib.rknn_destroy(ctx)
    return 0


def _write_input(lib, ctx, mem, array: np.ndarray, attr: RknnTensorAttr) -> None:
    """Copy ``array`` into ``mem``, repacking NCHW <-> NHWC when the model's
    input layout differs from the feed's (a rank-4 model's internal format is
    NHWC even though the ONNX graph is NCHW)."""
    expected = _attr_dims(attr)
    if array.shape != tuple(expected):
        # Compare element counts and total bytes rather than guessing a layout:
        # a mismatch here is a real feed/shape bug worth failing loudly on.
        if array.size != int(attr.n_elems):
            raise RuntimeError(
                f"feed for input {attr.name.decode(errors='replace')!r} has shape "
                f"{list(array.shape)} ({array.size} elements) but the model wants "
                f"{expected} ({attr.n_elems} elements)"
            )
    if len(expected) == 4 and attr.fmt == RKNN_TENSOR_NHWC:
        n, c, h, w = expected
        if array.size == n * c * h * w:
            # Assume the feed is NCHW (every ONNX-native model is) unless it
            # already matches the model's NHWC dims.
            if tuple(array.shape) != (n, c, h, w):
                array = np.ascontiguousarray(np.transpose(array, (0, 2, 3, 1)))
            else:
                array = np.ascontiguousarray(array)
    payload = np.ascontiguousarray(array, dtype=np.float32)
    size = _float_buffer_size(attr)
    if payload.size > attr.n_elems:
        raise RuntimeError(
            f"feed for {attr.name.decode(errors='replace')!r} has {payload.size} "
            f"elements but the model input holds {attr.n_elems}"
        )
    # Zero the padding region beyond the tensor so no stale/uninitialized bytes
    # reach the NPU.
    ctypes.memset(mem.contents.virt_addr, 0, size)
    ctypes.memmove(mem.contents.virt_addr, payload.ctypes.data, payload.nbytes)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]


if __name__ == "__main__":
    raise SystemExit(main())