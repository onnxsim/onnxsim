"""hmxsim.py for a QDQ int8 3x3 (stride 1 or 2) convolution in tinygrad's grid form, requantization fused: NHWC, the padded image
flattened to pixels at row stride Wp = W + 2 (plus a tail), output computed on the same grid (2 garbage columns per row, cropped
by the consumer) so the output and input pixel strides match and tinygrad merges them into one axis for the int8 TensorCore:
  A(p, dy, dx, c) = x[s*p + dy*Wp + dx, c]   (two dilated 1-D windows over the flat pixel axis: movement ops only)
Stride 2 keeps the output grid's row stride at Wp (so s*p stays affine): output row i reads input rows 2i + dy, and about half
of each output grid row (columns >= W/2) is garbage -- 2x the MACs of the conv, but no phase-split copy.
  y(p, n) = clamp(rne(fp32(fp32(sum A * w + b) * m)) + zy, lo, 255)
Checked exactly against numpy on hexagon-sim --mhmx 1 and MOCKDSP. Needs TC_OPT=1 (two reduce axes: dy, and dx*C + c).

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 CC=clang-19 PYTHONPATH=<tinygrad> python hmxsim_conv.py H W C N [--stride 2] [--relu] [--int32] [--noties] [--ref]
"""
import sys, os, tempfile, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (capture hooks + run_sim)
from hmxsim_rq import reference  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402

def grid(H, W, st=1):
  # (Wp, output grid pixels padded to 64, input pixels needed): output row i at s*i*Wp... row stride Wp for both
  Wp = W + 2; P = (H // st) * Wp; P64 = (P + 63) // 64 * 64
  return Wp, P64, st * (P64 - 1) + 2 * Wp + 3

def conv3x3_grid(x:Tensor, w:Tensor, Wp:int, P64:int, st:int=1) -> Tensor:
  # x (L, C) uint8 flat padded image, w (3, 3, C, N) int8 -> int32 (P64, N) on the output grid
  C, N = x.shape[1], w.shape[3]
  v = x.permute(1, 0)._pool((3,), 1, 1)                           # (C, L - 2, 3): dx
  v = v.permute(0, 2, 1)._pool((3,), st, Wp)                      # (C, 3, P', 3): dy, stride st over the grid
  v = v.shrink(((0, C), (0, 3), (0, P64), (0, 3))).permute(2, 3, 1, 0)   # (P64, dy, dx, C)
  return (v.reshape(P64, 1, 3, 3, C).cast(dtypes.int32) * w.permute(3, 0, 1, 2).reshape(1, N, 3, 3, C).cast(dtypes.int32)).sum((2, 3, 4))

def layer(X, Wt, b, m, zy, lo, Wp, P64, st=1, int32=False):
  acc = conv3x3_grid(Tensor(X), Tensor(Wt), Wp, P64, st) + Tensor(b)
  if int32: return acc
  return ((acc.cast(dtypes.float32) * Tensor(m)).round() + float(zy)).clip(lo, 255).cast(dtypes.uint8)

if __name__ == "__main__":
  H, W, C, N = (int(a) for a in sys.argv[1:5])
  st = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 1
  Wp, P64, L = grid(H, W, st)
  rng = np.random.default_rng(2)
  X = rng.integers(0, 256, (L, C), dtype=np.uint8); Wt = rng.integers(-128, 128, (3, 3, C, N), dtype=np.int8)
  b = rng.integers(-20000, 20000, N, dtype=np.int32); zy = 131; lo = zy if "--relu" in sys.argv else 0
  A = np.stack([X[dy * Wp + dx: dy * Wp + dx + st * (P64 - 1) + 1: st] for dy in range(3) for dx in range(3)], 1).reshape(P64, 9 * C)
  Wk = Wt.reshape(9 * C, N)
  acc0 = A.astype(np.int64) @ Wk.astype(np.int64) + b
  m = (rng.uniform(0.5, 2.0, N) * 50 / (np.abs(acc0).max(axis=0) + 1)).astype(np.float32); m[::8] = m[::8] if "--noties" in sys.argv else np.float32(2.0 ** -8)
  ref = reference(A, Wk, b, m, zy, lo)
  i32 = "--int32" in sys.argv  # the accumulator (+ bias) out instead: isolates the requantization's cost
  if i32: ref = (acc0).astype(np.int32)
  layer(X, Wt, b, m, zy, lo, Wp, P64, st, i32).realize()
  assert len(hmxsim._calls) == 1, f"expected one HMX kernel, got {len(hmxsim._calls)}"
  src, bufs = hmxsim._calls[0]
  assert "__hmx_i8_mac" in src and (i32 or "__hmx_rq4(" in src or "__hmx_rq1(" in src), "not lowered to HMX + fused requant"
  with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.cache")) as d:
    out1, c1 = hmxsim.run_sim(src, bufs, 1, pathlib.Path(d))
    _, c2 = hmxsim.run_sim(src, bufs, 2, pathlib.Path(d))
  out = np.frombuffer(out1, np.int32 if i32 else np.uint8)[:P64 * N].reshape(P64, N)
  bad = int((out != ref).sum())
  print(f"3x3/{st} {H}x{W}x{C}->{N} (grid {P64} px, K = 9 x {C}): hexagon-sim HMX {bad}/{P64 * N} mismatches vs ORT's formula; "
        f"{c2 - c1} Pcycles/call ({'PASS' if bad == 0 else 'FAIL'})")
  if "--ref" in sys.argv:
    os.environ["HMXSIM_RUN_REF"] = "1"
    o = layer(X, Wt, b, m, zy, lo, Wp, P64, st, i32).numpy()
    print(f"  MOCKDSP scalar reference: {int((o != ref).sum())} mismatches")
