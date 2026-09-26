"""One tinygrad fp16 matmul on the HMX TensorCore, captured and replayed on hexagon-sim (emit.py), vs numpy:
  python tc_test.py <out bundle> B M K N      (B: batch; K = 32 is the single-K-tile case, no reduce loop)"""
import sys
from pathlib import Path

import capture
import numpy as np
from tinygrad import Tensor, dtypes

out, (B, M, K, N) = Path(sys.argv[1]), map(int, sys.argv[2:6])
rng = np.random.default_rng(0)
a = (rng.standard_normal((B, M, K)) * 0.5).astype(np.float16)
b = (rng.standard_normal((B, K, N)) * 0.5).astype(np.float16)
capture.start(run=False)
y = Tensor(a).matmul(Tensor(b), dtype=dtypes.half).realize()
calls = capture.stop()
capture.save(calls, out, out_addr=capture.addr_of(y))
ref = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float16).astype(np.float32)
ref.reshape(B * M, N).tofile(out / "ref.bin")
print(f"{len(calls)} call(s), HMX: {['__hmx_begin' in c.src for c in calls]} -> {out}")
