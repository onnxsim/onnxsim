"""hmxsim.py for the int8 ":cm" TensorCore (hexagon_hmx_i8): a uint8 x int8 -> int32 tinygrad matmul captured from MOCKDSP,
rebuilt with hexagon-clang -mv69 -mhmx and run on hexagon-sim --mhmx 1 with real data, compared exactly against numpy.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad> python hmxsim_i8.py M K N [--ref]
"""
import sys, os, tempfile, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (capture hooks + run_sim)
from tinygrad import Tensor, dtypes  # noqa: E402

if __name__ == "__main__":
  M, K, N = (int(x) for x in sys.argv[1:4])
  rng = np.random.default_rng(1)
  A = rng.integers(0, 256, (M, K), dtype=np.uint8); B = rng.integers(-128, 128, (K, N), dtype=np.int8)
  Tensor(A).matmul(Tensor(B), dtype=dtypes.int32).realize()
  assert len(hmxsim._calls) == 1, f"expected one HMX kernel, got {len(hmxsim._calls)}"
  src, bufs = hmxsim._calls[0]
  ref = A.astype(np.int64) @ B.astype(np.int64)
  with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.cache")) as d:
    out1, c1 = hmxsim.run_sim(src, bufs, 1, pathlib.Path(d))
    _, c2 = hmxsim.run_sim(src, bufs, 2, pathlib.Path(d))
  out = np.frombuffer(out1, np.int32)[:M*N].reshape(M, N)
  bad = int((out != ref).sum())
  cyc = c2 - c1
  print(f"{M}x{K}x{N} int8: hexagon-sim HMX {bad}/{M*N} mismatches vs exact int32; {cyc} Pcycles/call, "
        f"{M*K*N/max(cyc,1):.1f} MAC/cycle ({'PASS' if bad == 0 else 'FAIL'})")
  if "--ref" in sys.argv:
    os.environ["HMXSIM_RUN_REF"] = "1"
    o = Tensor(A).matmul(Tensor(B), dtype=dtypes.int32).numpy()
    print(f"  MOCKDSP scalar reference: {int((o != ref).sum())} mismatches")
