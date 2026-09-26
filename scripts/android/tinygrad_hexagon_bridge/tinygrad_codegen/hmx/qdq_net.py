"""A static QDQ ONNX (onnxsim full_qdq + quantized_io, e.g. the hmx_gemm runner's ResNet-18) through tinygrad onto the DSP as
one program -- a thin driver over the tinygrad fork (onnxsim/tinygrad hvx-hmx-qdq), which does all of it:
  tinygrad/nn/onnx_qdq.py            the QDQ lowering OnnxRunner uses on DSP with HMX=1 (grid-form HMX convs with the fused
                                     exact requant, the exact QLinearAdd, MaxPool; qdq_emulate = ORT CPU's semantics)
  tinygrad/runtime/support/dsp_graph.py   capture every kernel of one inference, emit them as one C program, run it on
                                     hexagon-sim, build the FastRPC skel + Android client (one call per inference)

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 HMX_VTCM_KB=4096 PYTHONPATH=<tinygrad> \\
    python qdq_net.py model.onnx outdir [--sim input.bin [ort_ref.bin]] [--skel]
--sim: the whole program on hexagon-sim, vs qdq_emulate (and ORT CPU's output); per-call pcycles in outdir/sim_profile.txt.
--skel: build outdir/tg_hmx_rpc.so + outdir/client (HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN, NDK_CLANG); run it with
run_graph.sh (under the phone lock).
"""
import sys, pathlib
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.nn.onnx_qdq import qdq_emulate
from tinygrad.runtime.support import dsp_graph

if __name__ == "__main__":
  model, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
  runner = OnnxRunner(model)
  net = runner._qdq_grid()
  if net is None: sys.exit("the model isn't covered by tinygrad's QDQ grid lowering (DEBUG=1 names the node)")
  x = Tensor.empty(1, net.xin.h, net.xin.w, net.xin.c, dtype=dtypes.uint8).realize()
  for c in net.consts: c.realize()
  out: list = []
  calls, bufs = dsp_graph.capture(lambda: out.append(runner({net.in_name: x})[runner.graph_outputs[0]]))
  info = dsp_graph.emit(outdir, calls, bufs, x.uop.buffer, out[0].uop.buffer)
  kinds: dict[str, int] = {}
  for _, src, _, _ in calls:
    k = "hmx" if "__hmx_i8_mac" in src else "qadd" if "__hmx_qadd_chunk" in src else "other"
    kinds[k] = kinds.get(k, 0) + 1
  print(f"{info['calls']} kernel calls {kinds} ({len(info['kernels'])} distinct), {info['nbuf']} buffers: constants "
        f"{info['blob'] / 1e6:.2f} MB, scratch {info['scratch'] / 1e6:.2f} MB -> {outdir}")
  if "--sim" in sys.argv:
    i = sys.argv.index("--sim")
    xq = np.fromfile(sys.argv[i + 1], np.uint8).reshape(1, net.xin.h, net.xin.w, net.xin.c)
    ref = qdq_emulate(net, xq)
    got, cyc = dsp_graph.run_sim(outdir, xq.tobytes(), ref.nbytes)
    got = np.frombuffer(got, np.uint8).reshape(ref.shape)
    bad = int((got != ref).sum())
    print(f"hexagon-sim: {bad}/{ref.size} mismatches vs ORT's semantics; {cyc} pcycles/inference ({'PASS' if bad == 0 else 'FAIL'})")
    if len(sys.argv) > i + 2 and not sys.argv[i + 2].startswith("--"):
      ort = np.fromfile(sys.argv[i + 2], np.uint8)
      print(f"  vs ORT CPU's output: {int((got.ravel() != ort).sum())}/{ort.size} mismatches")
  if "--skel" in sys.argv:
    so, client = dsp_graph.build_skel(outdir)
    print(f"built {so} {client}")
