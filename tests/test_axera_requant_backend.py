"""Checks of the requant's column-solver backends and the plain-rounding fallback.

``gpu_gptq.GpuGptq`` (WebGPU) is compared with the numpy column loop on random data.
That test needs the optional ``wgpu`` package and a WebGPU adapter and is skipped
without them. The fallback test needs neither.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import llm_int4_requant as rq  # noqa: E402


def _problem(rows=64, cin=160, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(rows, cin) * 0.05
    x = rng.randn(4000, cin) * (1 + rng.rand(cin) * 3)
    h = (x.T @ x) / len(x)
    hd = h.copy()
    damping = 0.01
    hinv = np.linalg.inv(hd + damping * np.mean(np.diag(hd)) * np.eye(cin))
    u = np.linalg.cholesky((hinv + hinv.T) / 2).T
    s = np.abs(w).max(axis=1) / 7.5
    return w, hd, u, s


def test_keep_better_prefers_plain_when_gptq_is_worse():
    w, h, _, _ = _problem(seed=1)
    lo, hi = rq.CODE_RANGE["B"]
    plain = rq._as_codes(*rq.quantize_plain(w.astype(np.float32), "B"))
    # deliberately bad GPTQ codes: every code zero
    bad = (np.zeros(w.shape, np.int8), plain[1])
    chosen = rq._keep_better(w.astype(np.float32), bad, h, "B")
    np.testing.assert_array_equal(chosen[0], plain[0])


def test_keep_better_keeps_gptq_when_it_is_better():
    w, h, u, s = _problem(seed=2)
    lo, hi = rq.CODE_RANGE["B"]
    q = rq._gptq_codes(w, u, s, lo, hi)
    good = (q.astype(np.int8), s.astype(np.float32))
    chosen = rq._keep_better(w.astype(np.float32), good, h, "B")
    assert chosen is good


def test_backend_default_is_numpy():
    assert rq.CODE_BACKEND == "numpy"


@pytest.mark.parametrize("seed", [0, 1])
def test_webgpu_codes_match_numpy(seed):
    pytest.importorskip("wgpu")
    from gpu_gptq import GpuGptq

    try:
        gpu = GpuGptq()
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no WebGPU adapter: {exc!r}")
    w, h, u, s = _problem(rows=96, cin=300, seed=seed)
    lo, hi = rq.CODE_RANGE["B"]
    cpu = rq._gptq_codes(w, u, s, lo, hi)
    got = gpu.codes(w, u, s, lo, hi).astype(np.float64)
    # float32 on the GPU, float64 on the CPU: codes agree except at rounding ties
    assert float(np.mean(got != cpu)) < 0.005

    def herr(q):
        d = w - q * s[:, None]
        return float(np.sum(d @ h * d))

    assert abs(herr(got) - herr(cpu)) / herr(cpu) < 1e-3
