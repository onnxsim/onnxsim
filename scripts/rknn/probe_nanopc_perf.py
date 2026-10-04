#!/usr/bin/env python3
"""Probe whether librknnrt's rknn_query exposes the perf-detail codes on-device.

Rockchip's public header defines two profiling query codes that the host
toolkit's internal RKNNBase also calls (``get_run_perf_detail_len`` /
``get_run_perf_detail``):

    RKNN_QUERY_PERF_DETAIL = 4   -- full per-layer breakdown for the last run
    RKNN_QUERY_PERF_RUN    = 5   -- length of the above, in bytes

This probes them on a real board after an actual ``rknn_run``, since the
symbols ``rknn_query`` is confirmed present but the *codes* are only useful if
the runtime implements them.
"""

import ctypes
import sys

sys.path.insert(0, "/tmp/onnxsim-rknn-prof")

RKNN_QUERY_PERF_DETAIL = 4
RKNN_QUERY_PERF_RUN = 5

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16), ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32), ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32), ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32), ("fl", ctypes.c_int8),
        ("_padding", ctypes.c_uint8 * 3), ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float), ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through_attr", ctypes.c_uint8), ("_padding2", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnTensorMem(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("virt_addr", ctypes.c_void_p), ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32), ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


model = open(sys.argv[1], "rb").read()
buf = ctypes.create_string_buffer(model)
lib = ctypes.CDLL("/usr/lib/librknnrt.so")
lib.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
lib.rknn_query.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                           ctypes.c_uint32]
lib.rknn_create_mem.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
lib.rknn_create_mem.restype = ctypes.POINTER(RknnTensorMem)
lib.rknn_set_io_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(RknnTensorMem),
                                ctypes.POINTER(RknnTensorAttr)]
lib.rknn_run.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
lib.rknn_destroy.argtypes = [ctypes.c_uint32]
lib.rknn_destroy_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(RknnTensorMem)]

ctx = ctypes.c_uint32(0)
print("rknn_init ->", lib.rknn_init(ctypes.byref(ctx), buf, len(model), 0, None))

num = RknnInputOutputNum()
lib.rknn_query(ctx, 0, ctypes.byref(num), ctypes.sizeof(num))
print(f"n_input={num.n_input} n_output={num.n_output}")

in_mems, out_mems = [], []
for i in range(num.n_input):
    a = RknnTensorAttr(index=i)
    lib.rknn_query(ctx, 1, ctypes.byref(a), ctypes.sizeof(a))
    base = {0: 4, 1: 2}.get(a.type, 1)
    size = (a.size_with_stride or a.size) * 4 // base
    m = lib.rknn_create_mem(ctx, size)
    in_mems.append(m)
    ctypes.memset(m.contents.virt_addr, 0, size)
    a.pass_through = 0
    a.type = RKNN_TENSOR_FLOAT32
    if a.n_dims == 4:
        a.fmt = RKNN_TENSOR_NHWC
    print(f"set_io_mem(in[{i}]) ->", lib.rknn_set_io_mem(ctx, m, ctypes.byref(a)))
for i in range(num.n_output):
    a = RknnTensorAttr(index=i)
    lib.rknn_query(ctx, 2, ctypes.byref(a), ctypes.sizeof(a))
    base = {0: 4, 1: 2}.get(a.type, 1)
    size = (a.size_with_stride or a.size) * 4 // base
    m = lib.rknn_create_mem(ctx, size)
    out_mems.append(m)
    ctypes.memset(m.contents.virt_addr, 0, size)
    a.pass_through = 0
    a.type = RKNN_TENSOR_FLOAT32
    print(f"set_io_mem(out[{i}]) ->", lib.rknn_set_io_mem(ctx, m, ctypes.byref(a)))

print("rknn_run ->", lib.rknn_run(ctx, None))
print("rknn_run ->", lib.rknn_run(ctx, None))

print()
print("=== RKNN_QUERY_PERF_RUN (5) at increasing buffer sizes ===")
# The runtime's own guard is "info_len < sizeof(rknn_perf_run)", so a too-small
# buffer is rejected before the command is even interpreted. Try the real
# struct sizes so a -5 means "unsupported", not "buffer too small".
class RknnPerfRun(ctypes.Structure):
    _fields_ = [("n_common", ctypes.c_uint32), ("n_subcore", ctypes.c_uint32)]


class RknnPerfDetail(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("index", ctypes.c_uint32),
        ("name", ctypes.c_char * 256), ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32),
    ]


print(f"sizeof(rknn_perf_run)={ctypes.sizeof(RknnPerfRun)} "
      f"sizeof(rknn_perf_detail)={ctypes.sizeof(RknnPerfDetail)}")

for label, cmd, struct in (
    ("PERF_RUN", RKNN_QUERY_PERF_RUN, RknnPerfRun),
    ("PERF_DETAIL", RKNN_QUERY_PERF_DETAIL, RknnPerfDetail),
):
    probe = struct()
    rc = lib.rknn_query(ctx, cmd, ctypes.byref(probe), ctypes.sizeof(probe))
    print(f"{label}: rc={rc} n_common={probe.n_common if cmd == RKNN_QUERY_PERF_RUN else '-'}")

print()
print("=== is ANY cmd beyond the documented range accepted? (range guard) ===")
for cmd in (0, 4, 5, 6, 20, 100):
    probe = ctypes.c_uint32(0)
    rc = lib.rknn_query(ctx, cmd, ctypes.byref(probe), ctypes.sizeof(probe))
    print(f"cmd={cmd}: rc={rc}")

for m in out_mems + in_mems:
    lib.rknn_destroy_mem(ctx, m)
lib.rknn_destroy(ctx)