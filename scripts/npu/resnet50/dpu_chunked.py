"""Conv and Gemm on the Intel NPU DPU, through OpenVINO-compiled template blobs.

Each conv is split into sub-convs that fit one DPU launch (chunks of up to 256 x 256 channels, limited by weight count and
input/output size; 3x3 convs at 14x14 or smaller use cin * cout <= 8192). Every sub-conv is a template compiled by OpenVINO's
compile_tool (zero-bias, same shape); its constant buffer is overwritten with the real weights using the forward layout:

    groups of 16 output channels; per group a 128-slot table is reserved up front (160 + 128 * G);
    weight(o, i, kh, kw) = 160 + 128 * G + (o // 16) * 16 * R + (o % 16) * R + (kh * k + kw) * cin + i,
    R = roundup(cin * k * k, 16).

The bias is NOT in the template. It is added on the host after the input-channel partial sums are combined.
Verified on Intel NPU 5020 (OpenVINO 2026.4) against onnxruntime for all 23 ResNet50 conv shapes and the classifier
Gemm (relative error about 0.0005). The layout was measured on this device; it is not a documented format.

Requirements: OpenVINO Python (for the in-process launch) and OpenVINO's compile_tool. Set OPENVINO_COMPILE_TOOL to the
compile_tool binary if it is not on PATH. An NPU driver with a level-zero loader that supports the device is needed.
"""
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

WLIM = 147456   # weights per launch: 128 x 128 x 3 x 3 worked, larger did not
IMAX = 200704   # input or output elements per launch for 3x3: 256 x 28 x 28 worked; 128 x 56 x 56 did not
IMAX_1X1 = 401408   # input or output elements per launch for 1x1: 128 x 56 x 56 and 512 x 28 x 28 worked

_CACHE = os.path.join(tempfile.gettempdir(), "onnxsim_dpu_templates")
_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


def pick(cout, cin, k, hin, pad, stride):
    """Largest (ci, co) chunk that passes the launch limits (3x3 at 14x14 or smaller: the cin * cout limit above)."""
    hout = (hin + 2 * pad - k) // stride + 1
    if k > 1 and hin <= 14:
        # 3x3 at 14x14 or 7x7: chunks with cin * cout <= 8192 passed ((128, 64) and (64, 128)); 128 x 128 and larger did not.
        cands = [(min(ci, cin), min(co, cout)) for ci, co in ((128, 64), (64, 128), (64, 64))]
        return max(cands, key=lambda c: c[0] * c[1])
    best = (64, 64)
    for ci in (256, 128, 64):
        for co in (256, 128, 64):
            ci_, co_ = min(ci, cin), min(co, cout)
            imax = IMAX_1X1 if k == 1 else IMAX
            if co_ * ci_ * k * k <= WLIM and ci_ * hin * hin <= imax and co_ * hout * hout <= imax:
                if ci_ * co_ > best[0] * best[1]:
                    best = (ci_, co_)
    return best


def _compile_template(cin, cout, k, hin, pad, stride):
    """Compile a zero-weight template with compile_tool. Cached on disk by shape."""
    os.makedirs(_CACHE, exist_ok=True)
    name = "t_%d_%d_%d_%d_%d_%d" % (cin, cout, k, hin, pad, stride)
    blob_path = os.path.join(_CACHE, name + ".blob")
    if os.path.exists(blob_path):
        return open(blob_path, "rb").read()
    hout = (hin + 2 * pad - k) // stride + 1
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
    w = np.ones((cout, cin, k, k), np.float16) * 0.01
    g = helper.make_graph([helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k, k], pads=[pad] * 4, strides=[stride, stride])],
                          "c", [x], [y], initializer=[numpy_helper.from_array(w, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx_path = os.path.join(_CACHE, name + ".onnx")
    onnx.save(m, onnx_path)
    tool = os.environ.get("OPENVINO_COMPILE_TOOL") or shutil.which("compile_tool")
    if not tool:
        raise RuntimeError("compile_tool not found: set OPENVINO_COMPILE_TOOL")
    r = subprocess.run([tool, "-m", onnx_path, "-d", "NPU", "-o", blob_path, "-ip", "FP16", "-op", "FP16"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile_tool failed: " + r.stderr[-300:])
    return open(blob_path, "rb").read()


def _pack_weights(blob, w, cin, cout, k):
    """Write w [cout, cin, k, k] into the template's buffer.Constant section (forward layout, vectorized)."""
    data = bytearray(blob)
    so = struct.unpack_from("<Q", data, 0x28)[0]
    n = struct.unpack_from("<H", data, 0x3C)[0]
    si = struct.unpack_from("<H", data, 0x3E)[0]
    H = [struct.unpack_from("<IIQQQQIIQQ", data, so + 64 * i) for i in range(n)]
    strtab = H[si][4]
    names = [data[strtab + h[0]:].split(b"\0")[0].decode(errors="replace") for h in H]
    off, size = next((h[4], h[5]) for h, nm in zip(H, names) if nm == "buffer.Constant.0.constant")
    buf = np.frombuffer(bytes(data[off:off + size]), dtype=np.uint16).copy()
    G = -(-cout // 16)
    R = -(-(cin * k * k) // 16) * 16
    o = np.arange(cout)
    base = 160 + 128 * G + (o // 16) * 16 * R + (o % 16) * R
    idx = base[:, None, None] + np.arange(k * k)[None, None, :] * cin + np.arange(cin)[None, :, None]
    vals = np.ascontiguousarray(w.astype(np.float16)).reshape(cout, cin, k * k).view(np.uint16)
    buf[idx.ravel()] = vals.ravel()
    data[off:off + size] = buf.tobytes()
    return bytes(data)


def _launch(blob, x):
    """One DPU launch in-process. Retries transient device faults on a shared NPU."""
    last = None
    for _ in range(5):
        try:
            cm = _core().import_model(blob, "NPU")
            out = next(iter(cm.create_infer_request().infer({0: x}).values()))
            return np.asarray(out, dtype=np.float32).ravel()
        except Exception as e:  # transient fence/reset on the shared NPU
            last = e
    raise RuntimeError("NPU launch failed: %s" % last)


def conv(W, X, pad, stride):
    """Conv of X [1, cin, h, h] (float) with W [cout, cin, k, k] (float), no bias. Returns [1, cout, hout, hout] float32."""
    cout, cin, k, _ = W.shape
    hin = X.shape[2]
    hout = (hin + 2 * pad - k) // stride + 1
    CI, CO = pick(cout, cin, k, hin, pad, stride)
    out = np.zeros((1, cout, hout, hout), np.float32)
    for co in range(0, cout, CO):
        for ci in range(0, cin, CI):
            ko, ki = min(CO, cout - co), min(CI, cin - ci)
            blob = _pack_weights(_compile_template(ki, ko, k, hin, pad, stride), W[co:co + ko, ci:ci + ki], ki, ko, k)
            part = _launch(blob, X[:, ci:ci + ki].astype(np.float16))
            out[:, co:co + ko] += part.reshape(1, ko, hout, hout)
    return out


def gemm(W, x, bias=None):
    """y = x @ W^T + bias with W [N, K] and x [1, K], as a 1x1 conv on a 1x1 spatial input. Returns [1, N] float32."""
    N, K = W.shape
    y = conv(W[:, :, None, None], x.reshape(1, K, 1, 1), 0, 1).reshape(1, N)
    return y + (bias.reshape(1, N) if bias is not None else 0)
