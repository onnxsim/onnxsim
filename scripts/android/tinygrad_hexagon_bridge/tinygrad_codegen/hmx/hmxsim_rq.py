"""hmxsim.py for a QDQ int8 layer with the requantization fused into the HMX kernel: uint8 x int8 -> int32 on the int8 ":cm"
TensorCore, then ORT's QLinearConv/QLinearMatMul output formula per column,
  y = clamp(rne(fp32(fp32(acc + b) * m)) + zy, lo, 255)       (lo = zy under a fused Relu, else 0)
which tinygrad spells ((acc + b).float() * m).round() + zy).clip(lo, 255).cast(uint8) and the DSP renderer lowers to HVX (__hmx_rq4:
four output rows per accumulator load, one predicated store per row). Checked exactly against numpy's IEEE fp32 on hexagon-sim --mhmx 1 and MOCKDSP.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad> python hmxsim_rq.py M K N [--relu] [--nobias] [--noties] [--ref]
"""
import sys, os, tempfile, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (capture hooks + run_sim)
from tinygrad import Tensor, dtypes  # noqa: E402

def layer(A, W, b, m, zy, lo):
  acc = Tensor(A).matmul(Tensor(W), dtype=dtypes.int32)
  if b is not None: acc = acc + Tensor(b)
  return ((acc.cast(dtypes.float32) * Tensor(m)).round() + float(zy)).clip(lo, 255).cast(dtypes.uint8)

def reference(A, W, b, m, zy, lo):
  acc = A.astype(np.int64) @ W.astype(np.int64) + (0 if b is None else b)
  v = acc.astype(np.float32) * m  # fp32(acc) rounds to nearest even, the product is one IEEE fp32 multiply
  return (np.rint(np.clip(v, np.float32(lo - zy), np.float32(255 - zy))) + zy).astype(np.uint8)

if __name__ == "__main__":
  M, K, N = (int(x) for x in sys.argv[1:4])
  rng = np.random.default_rng(1)
  A = rng.integers(0, 256, (M, K), dtype=np.uint8); W = rng.integers(-128, 128, (K, N), dtype=np.int8)
  b = None if "--nobias" in sys.argv else rng.integers(-20000, 20000, N, dtype=np.int32)
  zy = 131; lo = zy if "--relu" in sys.argv else 0
  # output scales from the accumulator's actual spread per column (A isn't zero-mean), putting most outputs inside [0, 255]
  # and some saturating; every 8th column a power of two, where exact .5 ties happen
  acc0 = A.astype(np.int64) @ W.astype(np.int64) + (0 if b is None else b)
  m = (rng.uniform(0.5, 2.0, N) * 50 / (np.abs(acc0).max(axis=0) + 1)).astype(np.float32)
  if "--noties" not in sys.argv: m[::8] = np.float32(2.0 ** -8)  # --noties: arbitrary scales only, as ORT QDQ gives
  layer(A, W, b, m, zy, lo).realize()
  assert len(hmxsim._calls) == 1, f"expected one HMX kernel, got {len(hmxsim._calls)}"
  src, bufs = hmxsim._calls[0]
  assert "__hmx_rq4(" in src or "__hmx_rq1(" in src, "requantization not fused"
  ref = reference(A, W, b, m, zy, lo)
  acc = A.astype(np.int64) @ W.astype(np.int64) + (0 if b is None else b)
  ties = int((((acc * 2 ** -8) % 1) == 0.5)[:, ::8].sum())
  with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.cache")) as d:
    out1, c1 = hmxsim.run_sim(src, bufs, 1, pathlib.Path(d))
    _, c2 = hmxsim.run_sim(src, bufs, 2, pathlib.Path(d))
  out = np.frombuffer(out1, np.uint8)[:M*N].reshape(M, N)
  bad = int((out != ref).sum())
  if bad and os.environ.get("RQ_DEBUG"): print(out[0, :40]); print(ref[0, :40]); print(out[1, :8], ref[1, :8]); open(os.path.expanduser("~/.cache/rq_k.c"), "w").write(src)
  print(f"{M}x{K}x{N} int8 + requant (zy {zy}, lo {lo}, bias {b is not None}): hexagon-sim HMX {bad}/{M*N} mismatches vs "
        f"ORT's formula ({ties} exact .5 ties, {int((ref == 255).sum() + (ref == lo).sum())} clamped); {c2 - c1} Pcycles/call "
        f"({'PASS' if bad == 0 else 'FAIL'})")
  if "--ref" in sys.argv:
    os.environ["HMXSIM_RUN_REF"] = "1"
    o = layer(A, W, b, m, zy, lo).numpy()
    print(f"  MOCKDSP scalar reference: {int((o != ref).sum())} mismatches")
