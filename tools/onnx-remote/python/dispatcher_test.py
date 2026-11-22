"""Tests for model_dispatcher against the real onnx-remote compiler service and mock runner.

Set ONNX_REMOTE_BIN to a build directory that contains onnx-remote-compiler and
onnx-remote-mock-runner (for example ``build/onnx-remote``). Without it the
live tests are skipped; the routing-only tests still run.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_dispatcher import Endpoint, ModelDispatcher, NoRunnerError  # noqa: E402
from onnx_remote_client import Client  # noqa: E402

MODEL = b"not-really-onnx: the copy compiler does not parse the model"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(ep: Endpoint, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{ep.name} exited with {proc.returncode}")
        try:
            # A plain connect works for every service; the compiler rejects capabilities requests.
            with socket.create_connection((ep.host, ep.port), timeout=1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"{ep.name} did not start")


@pytest.fixture(scope="module")
def bin_dir():
    value = os.environ.get("ONNX_REMOTE_BIN")
    if not value or not (Path(value) / "onnx-remote-compiler").exists():
        pytest.skip("set ONNX_REMOTE_BIN to a build directory with the onnx-remote binaries")
    return Path(value)


@pytest.fixture(scope="module")
def services(bin_dir, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    procs = []
    compiler = Endpoint("compiler", "127.0.0.1", _free_port())
    runner = Endpoint("mock-runner", "127.0.0.1", _free_port())
    p1 = subprocess.Popen(
        [str(bin_dir / "onnx-remote-compiler"), "--port", str(compiler.port), "--cache-dir", str(cache / "c"),
         "--target", "test", "--compiler-id", "copy-1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(p1)
    p2 = subprocess.Popen(
        [str(bin_dir / "onnx-remote-mock-runner"), "--port", str(runner.port), "--cache-dir", str(cache / "r")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(p2)
    try:
        _wait_ready(compiler, p1)
        _wait_ready(runner, p2)
        yield {"compiler": compiler, "runner": runner}
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=5)


def test_compiles_loads_and_runs_on_a_compiled_runner(services):
    d = ModelDispatcher([services["runner"]], [services["compiler"]])
    x = np.arange(8, dtype=np.float32).reshape(1, 8)
    (y,) = d.run(MODEL, [x])
    # The mock runner returns its input unchanged, so the dispatched result must equal it.
    np.testing.assert_array_equal(y, x)


def test_second_run_reuses_the_compiled_artifact(services):
    d = ModelDispatcher([services["runner"]], [services["compiler"]])
    x = np.ones((1, 4), dtype=np.float32)
    d.run(MODEL, [x])
    d.run(MODEL, [x])
    assert len(d._artifacts) == 1  # one compile for the model, not one per call
    runner = d.probe()[0]
    assert len(runner.loaded) == 1


def test_unreachable_runner_is_skipped(services):
    dead = Endpoint("dead", "127.0.0.1", _free_port())
    d = ModelDispatcher([dead, services["runner"]], [services["compiler"]])
    x = np.zeros((1, 2), dtype=np.float32)
    (y,) = d.run(MODEL, [x])
    np.testing.assert_array_equal(y, x)
    assert [r.endpoint.name for r in d.probe()] == ["mock-runner"]


def test_no_reachable_runner_raises(services):
    dead = Endpoint("dead", "127.0.0.1", _free_port())
    d = ModelDispatcher([dead], [services["compiler"]])
    with pytest.raises(NoRunnerError):
        d.run(MODEL, [np.zeros((1, 2), dtype=np.float32)])


def test_no_compiler_and_no_graph_runner_raises(services):
    # The mock runner has no subgraph/graph execution, so without a compiler nothing can run the model.
    d = ModelDispatcher([services["runner"]], compilers=())
    with pytest.raises(NoRunnerError):
        d.run(MODEL, [np.zeros((1, 2), dtype=np.float32)])


def test_all_compilers_unreachable_raises(services):
    dead = Endpoint("dead-compiler", "127.0.0.1", _free_port())
    d = ModelDispatcher([services["runner"]], [dead])
    with pytest.raises(NoRunnerError, match="no compiler answered"):
        d.run(MODEL, [np.zeros((1, 2), dtype=np.float32)])


def test_routing_prefers_compiled_runner_when_a_compiler_is_configured():
    # Routing only: a fake probe result, no sockets.
    from model_dispatcher import _Runner

    d = ModelDispatcher([Endpoint("g", "x", 1), Endpoint("c", "x", 2)], [Endpoint("comp", "x", 3)])
    graph = _Runner(Endpoint("g", "x", 1), frozenset({"subgraph"}), {"graph_execution": True})
    compiled = _Runner(Endpoint("c", "x", 2), frozenset({"load_compiled", "run_compiled"}), {})
    d._runners = [graph, compiled]
    assert d.choose() is compiled
    d._compilers = []
    assert d.choose() is graph
