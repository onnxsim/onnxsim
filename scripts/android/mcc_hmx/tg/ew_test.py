"""Element-wise float kernels (tanh GELU, softmax) through tinygrad's DSP codegen, captured and replayed on hexagon-sim
(emit.py) vs numpy:  python ew_test.py <out bundle> gelu|softmax"""
import sys
from pathlib import Path

import capture
import numpy as np
from tinygrad import Tensor

out, op = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(0)
if op == "gelu":
    x = (rng.standard_normal((64, 512)) * 2).astype(np.float32)
    ref = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    f = lambda t: t.gelu(approximate="tanh")  # noqa: E731
else:
    x = (rng.standard_normal((64, 225)) * 3).astype(np.float32)
    e = np.exp(x - x.max(-1, keepdims=True))
    ref = e / e.sum(-1, keepdims=True)
    f = lambda t: t.softmax(-1)  # noqa: E731
capture.start(run=False)
y = f(Tensor(x)).realize()
calls = capture.stop()
capture.save(calls, out, out_addr=capture.addr_of(y))
ref.astype(np.float32).tofile(out / "ref.bin")
print(f"{op}: {len(calls)} call(s); vector helpers used: {['__tg_exp2_f' in c.src or '__tg_recip_f' in c.src for c in calls]}")
