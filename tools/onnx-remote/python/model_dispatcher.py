"""Route an ONNX model to an onnx-remote runner, compiling it first when needed.

The dispatcher knows two kinds of endpoints:

* compilers: ``onnx-remote-compiler`` services. They answer ``compile`` with an
  artifact and an artifact_id.
* runners: workers that report their operations in ``capabilities``. A runner
  that lists ``load_compiled`` and ``run_compiled`` runs compiled artifacts; a
  runner that reports ``graph_execution`` runs ONNX graphs directly with
  ``subgraph``.

For each call the dispatcher:

1. probes the runners (once, cached) and keeps the reachable ones,
2. picks the first runner that can run the model: a compiled-artifact runner
   when a compiler is configured, otherwise a graph-execution runner,
3. compiles the model on the first compiler that answers (artifacts are cached
   per (compiler, model hash)),
4. loads the artifact on the runner once and runs it, sending the artifact
   bytes again only when the runner reports it lost them.

It only uses the client in ``onnx_remote_client.py``, so it needs numpy and
nothing else.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from onnx_remote_client import Client, RemoteError, Response, PROFILING_OFF

LOG = logging.getLogger("onnx_remote.dispatch")

ModelLike = Union[str, Path, bytes]


@dataclasses.dataclass(frozen=True)
class Endpoint:
    name: str
    host: str
    port: int

    def client(self) -> Client:
        return Client(self.host, self.port)


class NoRunnerError(RuntimeError):
    """No reachable runner can run the model."""


@dataclasses.dataclass
class _Runner:
    endpoint: Endpoint
    ops: frozenset
    manifest: dict
    loaded: set = dataclasses.field(default_factory=set)  # artifact_ids already loaded here

    @property
    def runs_compiled(self) -> bool:
        return {"load_compiled", "run_compiled"} <= self.ops

    @property
    def runs_graphs(self) -> bool:
        return bool(self.manifest.get("graph_execution")) and "subgraph" in self.ops


class ModelDispatcher:
    def __init__(
        self,
        runners: Sequence[Endpoint],
        compilers: Sequence[Endpoint] = (),
        *,
        client_factory: Callable[[Endpoint], Client] = lambda e: e.client(),
        profiling: int = PROFILING_OFF,
    ) -> None:
        self._runner_endpoints = list(runners)
        self._compilers = list(compilers)
        self._client_factory = client_factory
        self._profiling = profiling
        self._runners: Optional[List[_Runner]] = None
        self._artifacts: Dict[Tuple[str, str], Tuple[str, bytes]] = {}  # (compiler, digest) -> (artifact_id, bytes)

    # -- probing -----------------------------------------------------------

    def probe(self) -> List[_Runner]:
        """Ask every runner for its capabilities; unreachable runners are dropped."""
        if self._runners is None:
            found: List[_Runner] = []
            for ep in self._runner_endpoints:
                try:
                    manifest = json.loads(self._client_factory(ep).request("capabilities").manifest)
                except (OSError, RemoteError, ValueError) as exc:
                    LOG.warning("runner %s unavailable: %s", ep.name, exc)
                    continue
                found.append(_Runner(ep, frozenset(manifest.get("supported_ops", [])), manifest))
            self._runners = found
        return self._runners

    def reprobe(self) -> None:
        """Forget the cached probe; the next call asks every runner again."""
        self._runners = None

    # -- routing -----------------------------------------------------------

    def choose(self) -> _Runner:
        runners = self.probe()
        if self._compilers:
            for r in runners:
                if r.runs_compiled:
                    return r
        for r in runners:
            if r.runs_graphs:
                return r
        raise NoRunnerError(
            "no reachable runner can run this model "
            f"(compilers configured: {bool(self._compilers)}, runners probed: {len(runners)})"
        )

    # -- execution ---------------------------------------------------------

    def run(self, model: ModelLike, inputs: Sequence) -> List:
        """Run an ONNX model on the best available runner and return its outputs."""
        model_bytes = _read_model(model)
        runner = self.choose()
        if runner.runs_compiled:
            return self._run_compiled(runner, model_bytes, inputs)
        LOG.info("running graph directly on %s", runner.endpoint.name)
        return self._client_factory(runner.endpoint).request(
            "subgraph", inputs, model=model_bytes, profiling=self._profiling
        ).outputs

    def _run_compiled(self, runner: _Runner, model_bytes: bytes, inputs: Sequence) -> List:
        artifact_id, artifact = self._compile(model_bytes)
        client = self._client_factory(runner.endpoint)
        if artifact_id not in runner.loaded:
            client.load_compiled(artifact_id, artifact)
            runner.loaded.add(artifact_id)
        try:
            response = client.request(
                "run_compiled", inputs, artifact_id=artifact_id, profiling=self._profiling
            )
        except RemoteError as exc:
            # The runner restarted or evicted the artifact. Reloading is safe, so do it once
            # and retry; a second failure is raised as is.
            LOG.info("runner %s rejected %s (%s); reloading once", runner.endpoint.name, artifact_id, exc)
            client.load_compiled(artifact_id, artifact)
            runner.loaded.add(artifact_id)
            response = client.request(
                "run_compiled", inputs, artifact_id=artifact_id, profiling=self._profiling
            )
        return response.outputs

    def _compile(self, model_bytes: bytes) -> Tuple[str, bytes]:
        digest = hashlib.sha256(model_bytes).hexdigest()
        errors: List[str] = []
        for ep in self._compilers:
            key = (ep.name, digest)
            if key in self._artifacts:
                return self._artifacts[key]
            try:
                response: Response = self._client_factory(ep).request("compile", model=model_bytes)
            except (OSError, RemoteError) as exc:
                errors.append(f"{ep.name}: {exc}")
                continue
            self._artifacts[key] = (response.artifact_id, response.artifact)
            return self._artifacts[key]
        raise NoRunnerError("no compiler answered: " + "; ".join(errors))


def _read_model(model: ModelLike) -> bytes:
    if isinstance(model, bytes):
        return model
    return Path(model).read_bytes()
