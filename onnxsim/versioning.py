"""Versioned ONNX graphs: hash-referenced graph text, reproducible test cases, and
equivalence checks recorded in a manifest.

The graph is stored as ONNX text (the form ``onnx.parser`` accepts) with every
initializer replaced by a *reference*: a marker comment carrying the tensor's
name, element type, shape and content digest. Weights are never written into
the text. They live in ``.onnx`` files the user supplies, and
:class:`TensorIndex` finds them by digest when a text file is loaded back.

Marker lines start with ``# onnxsim-init``. ``onnx.parser`` ignores comments, so
the file remains valid ONNX text on its own.

Test inputs are either *generated* from a seed (regenerated on demand and checked
against a recorded digest) or *supplied* as an ``.npz`` file (checked the same
way). :func:`check_equivalent` runs a base and a candidate graph on every case and
reports the largest output difference. :func:`record_step` appends the result to
a JSON manifest, one entry per change.

Limitations of this first version:

* Only top-level graph initializers are referenced. Initializers inside subgraphs
  and local functions are kept as they are printed by ``onnx.printer``.
* Reproducibility of generated inputs depends on the NumPy release; the digest
  check reports a mismatch instead of silently using different bytes.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, numpy_helper, parser, printer

INIT_MARKER = "# onnxsim-init "
MANIFEST_VERSION = 1


class ReproducibilityError(RuntimeError):
    """A regenerated or loaded test input does not match its recorded digest."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def tensor_digest(array: np.ndarray) -> str:
    """Content digest of a tensor: its numpy dtype, shape and little-endian bytes.

    The name is deliberately not part of the digest, so the same weight is found
    whichever file or initializer name carries it.
    """
    arr = np.asarray(array)
    if arr.dtype.kind in "OSU":
        raise TypeError(f"cannot digest a tensor of dtype {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<"))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str};{list(arr.shape)};".encode())
    h.update(arr.tobytes())
    return "sha256:" + h.hexdigest()


def file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _tensor_type_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type)


class TensorIndex:
    """Digest -> array lookup over the initializers of user-supplied models."""

    def __init__(self) -> None:
        self._by_digest: Dict[str, np.ndarray] = {}

    def add_model(self, model: ModelProto) -> None:
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            self._by_digest.setdefault(tensor_digest(arr), arr)

    def add_file(self, path: str) -> None:
        self.add_model(onnx.load(path, load_external_data=True))

    def get(self, digest: str) -> np.ndarray:
        try:
            return self._by_digest[digest]
        except KeyError:
            raise KeyError(f"no indexed initializer has digest {digest}") from None

    def __len__(self) -> int:
        return len(self._by_digest)


def graph_text(model: ModelProto) -> str:
    """Canonical text of ``model`` with its initializers replaced by references.

    The output is deterministic for a given model, so its SHA-256 is a stable
    identity for the graph (see :func:`graph_hash`).
    """
    stripped = ModelProto()
    stripped.CopyFrom(model)
    refs: List[str] = []
    for init in stripped.graph.initializer:
        arr = numpy_helper.to_array(init)
        refs.append(
            INIT_MARKER
            + json.dumps(
                {
                    "name": init.name,
                    "elem_type": _tensor_type_name(init.data_type),
                    "dims": list(arr.shape),
                    "digest": tensor_digest(arr),
                },
                sort_keys=True,
            )
        )
    del stripped.graph.initializer[:]
    body = printer.to_text(stripped)
    return "\n".join(refs + [body]) + "\n"


def graph_hash(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def load_graph_text(text: str, index: TensorIndex) -> ModelProto:
    """Rebuild a model from :func:`graph_text` output, resolving each reference."""
    model = parser.parse_model(text)
    for line in text.splitlines():
        if not line.startswith(INIT_MARKER):
            continue
        ref = json.loads(line[len(INIT_MARKER) :])
        arr = index.get(ref["digest"])
        if list(arr.shape) != ref["dims"]:
            raise ReproducibilityError(
                f"initializer {ref['name']}: indexed shape {list(arr.shape)} "
                f"differs from the recorded {ref['dims']}"
            )
        init = numpy_helper.from_array(arr, name=ref["name"])
        if _tensor_type_name(init.data_type) != ref["elem_type"]:
            raise ReproducibilityError(
                f"initializer {ref['name']}: element type {_tensor_type_name(init.data_type)} "
                f"differs from the recorded {ref['elem_type']}"
            )
        model.graph.initializer.append(init)
    return model


def save_snapshot(model: ModelProto, path: str) -> str:
    """Write ``model`` with its weights to ``path`` and return the file's digest."""
    onnx.save(model, path)
    return file_digest(path)


@dataclass(frozen=True)
class TensorSpec:
    """How to draw one graph input: dtype, shape, and the range for random values."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    low: float = 0.0
    high: float = 1.0


@dataclass(frozen=True)
class GeneratedCase:
    """One test input drawn from a seeded PCG64 generator."""

    name: str
    seed: int
    specs: Tuple[TensorSpec, ...]
    digests: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class SuppliedCase:
    """One test input read from an ``.npz`` file the user provides."""

    name: str
    path: str
    digests: Dict[str, str]


Case = Union[GeneratedCase, SuppliedCase]


def generate_feeds(case: GeneratedCase) -> Dict[str, np.ndarray]:
    """Draw the feeds for ``case``. Draw order follows ``specs``, so it is part of the case."""
    rng = np.random.Generator(np.random.PCG64(case.seed))
    feeds: Dict[str, np.ndarray] = {}
    for spec in case.specs:
        dt = np.dtype(spec.dtype)
        if np.issubdtype(dt, np.floating):
            arr = rng.uniform(spec.low, spec.high, size=spec.shape).astype(dt)
        elif np.issubdtype(dt, np.integer):
            arr = rng.integers(int(spec.low), int(spec.high), size=spec.shape, dtype=dt)
        elif dt == np.bool_:
            arr = rng.integers(0, 2, size=spec.shape).astype(bool)
        else:
            raise TypeError(f"cannot generate inputs of dtype {dt}")
        feeds[spec.name] = arr
    return feeds


def resolve_feeds(case: Case) -> Dict[str, np.ndarray]:
    """Feeds for ``case``, checked against the digests recorded with it."""
    if isinstance(case, GeneratedCase):
        feeds = generate_feeds(case)
        recorded = case.digests or {}
    else:
        with np.load(case.path, allow_pickle=False) as npz:
            feeds = {k: npz[k] for k in npz.files}
        recorded = case.digests
    for name, digest in recorded.items():
        if name not in feeds:
            raise ReproducibilityError(f"case {case.name}: input {name} is missing")
        actual = tensor_digest(feeds[name])
        if actual != digest:
            raise ReproducibilityError(
                f"case {case.name}: input {name} has digest {actual}, recorded {digest}"
            )
    return feeds


def case_to_dict(case: Case) -> dict:
    if isinstance(case, GeneratedCase):
        return {
            "name": case.name,
            "kind": "generated",
            "seed": case.seed,
            "specs": [
                {
                    "name": s.name,
                    "dtype": s.dtype,
                    "shape": list(s.shape),
                    "low": s.low,
                    "high": s.high,
                }
                for s in case.specs
            ],
            "digests": case.digests or {},
        }
    return {
        "name": case.name,
        "kind": "supplied",
        "path": os.path.basename(case.path),
        "digests": case.digests,
    }


def case_set_id(cases: Sequence[Case]) -> str:
    """Identity of a test set: a digest over its cases, in order."""
    payload = json.dumps([case_to_dict(c) for c in cases], sort_keys=True).encode(
        "utf-8"
    )
    return _sha256_bytes(payload)


def executor_name() -> str:
    try:
        import onnxruntime as ort

        return f"onnxruntime {ort.__version__}"
    except ImportError:
        return f"onnx.reference {onnx.__version__}"


def run_model(model: ModelProto, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        return ReferenceEvaluator(model).run(None, feeds)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


@dataclass(frozen=True)
class CaseReport:
    case: str
    max_abs_diff: float
    max_rel_diff: float
    ok: bool


def check_equivalent(
    base: ModelProto,
    candidate: ModelProto,
    cases: Sequence[Case],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> List[CaseReport]:
    """Run ``base`` and ``candidate`` on every case and compare their outputs.

    A case passes when every output has the same shape and each element satisfies
    ``|candidate - base| <= atol + rtol * |base|``. Output names must match.
    """
    reports: List[CaseReport] = []
    base_names = [o.name for o in base.graph.output]
    cand_names = [o.name for o in candidate.graph.output]
    if base_names != cand_names:
        raise ValueError(f"output names differ: {base_names} vs {cand_names}")
    for case in cases:
        feeds = resolve_feeds(case)
        ref = run_model(base, feeds)
        got = run_model(candidate, feeds)
        max_abs = 0.0
        max_rel = 0.0
        ok = True
        for r, g in zip(ref, got):
            if r.shape != g.shape:
                ok = False
                max_abs = float("inf")
                continue
            r64 = r.astype(np.float64)
            g64 = g.astype(np.float64)
            diff = np.abs(g64 - r64)
            denom = np.maximum(np.abs(r64), np.finfo(np.float64).tiny)
            max_abs = max(max_abs, float(diff.max(initial=0.0)))
            max_rel = max(max_rel, float((diff / denom).max(initial=0.0)))
            if not np.all(diff <= atol + rtol * np.abs(r64)):
                ok = False
        reports.append(CaseReport(case.name, max_abs, max_rel, ok))
    return reports


def load_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {"version": MANIFEST_VERSION, "steps": []}
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: unsupported manifest version {manifest.get('version')}"
        )
    return manifest


def record_step(
    path: str,
    *,
    label: str,
    command: str,
    base_graph: str,
    output_graph: str,
    cases: Sequence[Case],
    reports: Sequence[CaseReport],
    executor: str,
) -> dict:
    """Append one verified change to the manifest at ``path`` and return the entry."""
    manifest = load_manifest(path)
    entry = {
        "label": label,
        "command": command,
        "base_graph": base_graph,
        "output_graph": output_graph,
        "executor": executor,
        "test_set": case_set_id(cases),
        "cases": [case_to_dict(c) for c in cases],
        "verdict": "pass" if all(r.ok for r in reports) else "fail",
        "reports": [
            {
                "case": r.case,
                "max_abs_diff": r.max_abs_diff,
                "max_rel_diff": r.max_rel_diff,
                "ok": r.ok,
            }
            for r in reports
        ],
    }
    manifest["steps"].append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return entry
