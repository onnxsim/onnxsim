"""On-card check of run_plan on the split-K down projection (the Pulsar-free
PLaMo decode's shape): one [1, 4K] input is copied on the card into four
K-slices, each slice runs a [K, N] linear, and a two-input Add sums the partials.
The plan's result must equal the same chain run through the worker one program
at a time (host round trips).

Environment:
  ONNX_REMOTE_HOST / ONNX_REMOTE_PORT   the worker (default 127.0.0.1:39501)
  ONNX_REMOTE_PARTIAL                   the [K, N] linear, float32 boundary (on the worker)
  ONNX_REMOTE_ADD                       the two-input Add of [1, N] float32 (on the worker)
  ONNX_REMOTE_K                         K of the linear (default 2560)
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
PARTIAL = os.environ["ONNX_REMOTE_PARTIAL"]
ADD = os.environ["ONNX_REMOTE_ADD"]
K = int(os.environ.get("ONNX_REMOTE_K", "2560"))
PARTS = 4
NB = 4 * K  # one [1, K] float32 slice (and one [1, N] partial, N = K here)
XB = PARTS * NB


def main() -> int:
    c = C.Client(HOST, PORT)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (1, PARTS * K)).astype(np.float32)

    # host reference: one program per round trip
    parts = [
        c.run_path(PARTIAL, [np.ascontiguousarray(x[:, i * K : (i + 1) * K])])[0]
        for i in range(PARTS)
    ]
    s = c.run_path(ADD, [parts[0], parts[1]])[0]
    s = c.run_path(ADD, [s, parts[2]])[0]
    ref = c.run_path(ADD, [s, parts[3]])[0]

    # plan: models 1..4 are the partials, 5 the Add (reused for each sum)
    lines = [f"model {i} {PARTIAL}" for i in range(1, PARTS + 1)]
    lines.append(f"model 5 {ADD}")
    lines.append(f"buffer 1 {XB}")  # the whole input
    for i in range(PARTS):  # partial i: input 10+i, output 20+i
        lines += [f"buffer {10 + i} {NB}", f"buffer {20 + i} {NB}"]
    lines += [f"buffer 30 {NB}", f"buffer 31 {NB}", f"buffer 32 {NB}"]  # sums
    lines.append(f"write 1 0 {XB} 0")
    for i in range(PARTS):
        lines += [
            f"copy {10 + i} 0 1 {i * NB} {NB}",
            f"bind {i + 1} I 0 {10 + i} 0",
            f"bind {i + 1} O 0 {20 + i} 0",
            f"exec {i + 1}",
        ]
    # s1 = p0 + p1 ; s2 = s1 + p2 ; s3 = s2 + p3
    lines += [
        "bind 5 I 0 20 0",
        "bind 5 I 1 21 0",
        "bind 5 O 0 30 0",
        "exec 5",
        "bind 5 I 0 30 0",
        "bind 5 I 1 22 0",
        "bind 5 O 0 31 0",
        "exec 5",
        "bind 5 I 0 31 0",
        "bind 5 I 1 23 0",
        "bind 5 O 0 32 0",
        "exec 5",
        f"read 32 0 {NB}",
        "free",
    ]
    r = c.run_plan("\n".join(lines) + "\n", [x])
    got = r.outputs[0].view(np.float32).reshape(1, K)
    ok = np.array_equal(got, ref)
    print(
        "split-K plan == host chain:",
        ok,
        "| max |diff|",
        float(np.max(np.abs(got - ref))),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
