"""ORT's QLinearAdd through tinygrad (ops_dsp.hmx_qlinear_add, a custom kernel: tinygrad float ops reassociate the adds, so
the op can't be spelled as them and stay exact) on hexagon-sim and MOCKDSP, exact against numpy's fp32 in ORT's order:
  y = clamp(rne(rb*b + (ra*a + fixed)), 0, 255),  ra = sa/sy, rb = sb/sy, fixed = zy - (ra*za + rb*zb)   (all fp32)

  DEV=DSP MOCKDSP=1 CC=clang-19 PYTHONPATH=<tinygrad> python hmxsim_add.py N [--ties] [--ref]
--ties: power-of-two scale ratios, so exact .5 ties (the scalar-fix path) are common.
"""
import sys, os, tempfile, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (capture hooks + run_sim)
from tinygrad import Tensor  # noqa: E402
from tinygrad.runtime.ops_dsp import hmx_qlinear_add  # noqa: E402

f32 = np.float32

def consts(sa, sb, sy, za, zb, zy):
  ra, rb = f32(sa) / f32(sy), f32(sb) / f32(sy)
  fixed = f32(zy) - (ra * f32(za) + rb * f32(zb))
  return ra, rb, fixed

def reference(A, B, ra, rb, fixed):
  v = rb * B.astype(f32) + (ra * A.astype(f32) + fixed)   # numpy fp32: each op rounded, in this order
  return np.clip(np.rint(v), 0, 255).astype(np.uint8)

if __name__ == "__main__":
  n = int(sys.argv[1])
  rng = np.random.default_rng(4)
  A = rng.integers(0, 256, n, dtype=np.uint8); B = rng.integers(0, 256, n, dtype=np.uint8)
  if "--ties" in sys.argv: ra, rb, fixed = consts(0.5, 0.25, 1.0, 3, 0, 1)
  else: ra, rb, fixed = consts(0.0371, 0.0529, 0.0613, 118, 131, 7)
  ref = reference(A, B, ra, rb, fixed)
  hmx_qlinear_add(Tensor(A), Tensor(B), float(ra), float(rb), float(fixed)).realize()
  assert len(hmxsim._calls) == 1, f"expected one kernel, got {len(hmxsim._calls)}"
  src, bufs = hmxsim._calls[0]
  with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.cache")) as d:
    out1, c1 = hmxsim.run_sim(src, bufs, 1, pathlib.Path(d))
    _, c2 = hmxsim.run_sim(src, bufs, 2, pathlib.Path(d))
  out = np.frombuffer(out1, np.uint8)[:n]
  bad = int((out != ref).sum())
  v = (ra * A.astype(np.float64) + fixed) + rb * B.astype(np.float64)
  near = int((np.abs(v - np.floor(v) - 0.5) < 1e-4).sum())
  print(f"QLinearAdd {n}: hexagon-sim {bad}/{n} mismatches vs ORT's fp32 order ({near} within 1e-4 of a .5); {c2 - c1} Pcycles/call "
        f"({(c2 - c1) / n:.3f}/byte) ({'PASS' if bad == 0 else 'FAIL'})")
  if "--ref" in sys.argv:
    os.environ["HMXSIM_RUN_REF"] = "1"
    o = hmx_qlinear_add(Tensor(A), Tensor(B), float(ra), float(rb), float(fixed)).numpy()
    print(f"  MOCKDSP scalar reference: {int((o != ref).sum())} mismatches")
