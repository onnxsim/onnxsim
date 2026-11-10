"""On-card check of run_plan on a model with two outputs: both are bound to device
buffers, read back in order, and must equal the host run of the same model.

Environment:
  ONNX_REMOTE_HOST / ONNX_REMOTE_PORT   the worker (default 127.0.0.1:39501)
  ONNX_REMOTE_MULTI                     the two-output model on the worker: one
                                        [1, N] float32 input, two [1, N] float32 outputs
  ONNX_REMOTE_N                         N (default 2560)
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)
import onnx_remote_client as C  # noqa: E402

HOST = os.environ.get("ONNX_REMOTE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ONNX_REMOTE_PORT", "39501"))
MODEL = os.environ["ONNX_REMOTE_MULTI"]
N = int(os.environ.get("ONNX_REMOTE_N", "2560"))
NB = 4 * N


def main() -> int:
    c = C.Client(HOST, PORT)
    x = np.random.default_rng(3).uniform(-1, 1, (1, N)).astype(np.float32)
    ref = c.run_path(MODEL, [x])
    plan = f"""model 1 {MODEL}
buffer 1 {NB}
buffer 2 {NB}
buffer 3 {NB}
write 1 0 {NB} 0
bind 1 I 0 1 0
bind 1 O 0 2 0
bind 1 O 1 3 0
exec 1
read 2 0 {NB}
read 3 0 {NB}
free
"""
    r = c.run_plan(plan, [x])
    ok1 = np.array_equal(r.outputs[0].view(np.float32).reshape(1, N), ref[0])
    ok2 = np.array_equal(r.outputs[1].view(np.float32).reshape(1, N), ref[1])
    print("output 0 == host:", ok1, "| output 1 == host:", ok2)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
