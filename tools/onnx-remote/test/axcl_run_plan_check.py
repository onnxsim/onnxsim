"""On-card check of the AXCL worker's run_plan (see README, "run_plan").

Needs a running onnx-remote-axcl-worker and a compiled float32-boundary model
on the worker's filesystem. Environment:
  ONNX_REMOTE_HOST / ONNX_REMOTE_PORT   the worker (default 127.0.0.1:39501)
  ONNX_REMOTE_MODEL                      path of the model on the worker
  ONNX_REMOTE_IN_ELEMS                   its input (and output) element count, float32
The model must take one [1, N] float32 input and produce one [1, N] float32 output
(for example a 2560x2560 linear). Exits non-zero on any mismatch.
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
MODEL = os.environ["ONNX_REMOTE_MODEL"]
N = int(os.environ.get("ONNX_REMOTE_IN_ELEMS", "2560"))
NB = 4 * N


def main() -> int:
    c = C.Client(HOST, PORT)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, N)).astype(np.float32)
    x2 = rng.uniform(-1, 1, (1, N)).astype(np.float32)

    ref1 = c.run_path(MODEL, [x])[0]
    ref2 = c.run_path(MODEL, [ref1])[0]
    ref3 = c.run_path(MODEL, [c.run_path(MODEL, [x2])[0]])[0]

    plan = f"""model 1 {MODEL}
buffer 1 {NB}
buffer 2 {NB}
buffer 3 {NB}
write 1 0 {NB} 0
bind 1 I 0 1 0
bind 1 O 0 2 0
exec 1
bind 1 I 0 2 0
bind 1 O 0 3 0
exec 1
read 3 0 {NB}
"""
    r = c.run_plan(plan, [x])
    sid = int(r.artifact_id)
    ok1 = np.array_equal(r.outputs[0].view(np.float32).reshape(1, N), ref2)
    print("plan 1 chained == host round trips:", ok1)

    p2 = f"""write 1 0 {NB} 0
bind 1 I 0 1 0
bind 1 O 0 2 0
exec 1
bind 1 I 0 2 0
bind 1 O 0 3 0
exec 1
read 3 0 {NB}
free
"""
    r2 = c.run_plan(p2, [x2], session=sid)
    ok2 = np.array_equal(r2.outputs[0].view(np.float32).reshape(1, N), ref3)
    print("plan 2 on session", sid, "== host round trips:", ok2)

    try:
        c.run_plan("buffer 9 0\n", [])
        ok3 = False
        print("bad directive: accepted (unexpected)")
    except C.RemoteError as e:
        ok3 = "line 1" in str(e)
        print("bad directive refused:", e)
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
