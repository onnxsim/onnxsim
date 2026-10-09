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
import heapq
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper, parser, printer

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
    """Result of one case. ``ok`` means no compared output mismatched.

    ``status`` is ``"fail"`` on a mismatch, ``"partial"`` when some outputs were
    skipped (see :func:`run_partial`), and ``"pass"`` otherwise.
    """

    case: str
    max_abs_diff: float
    max_rel_diff: float
    ok: bool
    status: str = "pass"
    skipped: Tuple[str, ...] = ()
    failed_nodes: Tuple[str, ...] = ()


# A backend runs one node on its input arrays (None for an omitted optional input)
# and returns its outputs. It signals an unsupported op by raising.
Backend = Callable[[NodeProto, List[Optional[np.ndarray]]], List[np.ndarray]]


@dataclass(frozen=True)
class NodeRun:
    label: str
    op_type: str
    status: str  # "ran", "failed" or "skipped" (an input was never produced)
    error: str = ""


@dataclass(frozen=True)
class PartialRun:
    """Graph outputs of a partial run. A ``None`` output is not judged: it is
    missing, or it depends on a value that was filled in at random."""

    outputs: Dict[str, Optional[np.ndarray]]
    nodes: Tuple[NodeRun, ...]

    @property
    def skipped(self) -> Tuple[str, ...]:
        return tuple(name for name, v in self.outputs.items() if v is None)

    @property
    def failed_nodes(self) -> Tuple[str, ...]:
        return tuple(n.label for n in self.nodes if n.status == "failed")


def _tensor_sizes(
    base: ModelProto, feeds: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Value of every node output of ``base``, from the reference evaluator.

    These only give shape and dtype to random values for ops a backend cannot run.
    """
    from onnx.reference import ReferenceEvaluator

    names = [o for n in base.graph.node for o in n.output if o]
    if not names:
        return {}
    return dict(zip(names, ReferenceEvaluator(base).run(names, feeds)))


def _random_like(template: np.ndarray, name: str, seed: int) -> np.ndarray:
    """Random values with the template's shape and dtype, reproducible from (seed, name)."""
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "little")))
    dt = template.dtype
    if dt.kind == "f":
        return rng.uniform(-1.0, 1.0, template.shape).astype(dt)
    if dt.kind in "iu":
        return rng.integers(0, 8, template.shape).astype(dt)
    if dt.kind == "b":
        return rng.integers(0, 2, template.shape).astype(bool)
    raise TypeError(f"cannot fill {name} with random values of dtype {dt}")


def run_partial(
    model: ModelProto,
    feeds: Dict[str, np.ndarray],
    backend: Backend,
    sizes: Dict[str, np.ndarray],
    seed: int = 0,
) -> PartialRun:
    """Run ``model`` node by node on ``backend``, continuing past ops it cannot run.

    A node that raises has each output filled with seeded random values, shaped
    like ``sizes[name]``, when that size is known. Every value computed from a
    random one is marked as such, and so are the graph outputs that depend on it.
    Nodes whose inputs were never produced are skipped.
    """
    env: Dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    env.update(feeds)
    fake: Set[str] = set()  # tensors whose value is random, or computed from one
    runs: List[NodeRun] = []
    for node in model.graph.node:
        label = _node_label(node)
        names = [t for t in node.input if t]
        if any(t not in env for t in names):
            runs.append(
                NodeRun(label, node.op_type, "skipped", "an input was not produced")
            )
            continue
        inputs = [env[t] if t else None for t in node.input]
        tainted = any(t in fake for t in names)
        try:
            outs = backend(node, inputs)
        except Exception as e:  # any backend failure, including unsupported ops
            runs.append(
                NodeRun(label, node.op_type, "failed", f"{type(e).__name__}: {e}")
            )
            for o in node.output:
                if o and o in sizes:
                    env[o] = _random_like(sizes[o], o, seed)
                    fake.add(o)
            continue
        runs.append(NodeRun(label, node.op_type, "ran"))
        for o, v in zip(node.output, outs):
            if o:
                env[o] = np.asarray(v)
                if tainted:
                    fake.add(o)
    outputs: Dict[str, Optional[np.ndarray]] = {}
    for out in model.graph.output:
        name = out.name
        outputs[name] = None if name not in env or name in fake else env[name]
    return PartialRun(outputs, tuple(runs))


def check_equivalent(
    base: ModelProto,
    candidate: ModelProto,
    cases: Sequence[Case],
    atol: float = 1e-5,
    rtol: float = 1e-4,
    backend: Optional[Backend] = None,
) -> List[CaseReport]:
    """Run ``base`` and ``candidate`` on every case and compare their outputs.

    A case passes when every output has the same shape and each element satisfies
    ``|candidate - base| <= atol + rtol * |base|``. Output names must match.

    With ``backend``, the candidate runs on that backend via :func:`run_partial`,
    so unsupported ops don't stop the check. Outputs it cannot judge are reported
    as ``skipped`` and the case as ``"partial"``. Random fills use the case's seed
    for generated cases and 0 otherwise.
    """
    reports: List[CaseReport] = []
    base_names = [o.name for o in base.graph.output]
    cand_names = [o.name for o in candidate.graph.output]
    if base_names != cand_names:
        raise ValueError(f"output names differ: {base_names} vs {cand_names}")
    for case in cases:
        feeds = resolve_feeds(case)
        ref = run_model(base, feeds)
        if backend is None:
            got: List[Optional[np.ndarray]] = list(run_model(candidate, feeds))
            failed_nodes: Tuple[str, ...] = ()
        else:
            seed = case.seed if isinstance(case, GeneratedCase) else 0
            partial = run_partial(
                candidate, feeds, backend, _tensor_sizes(base, feeds), seed
            )
            got = [partial.outputs[n] for n in cand_names]
            failed_nodes = partial.failed_nodes
        max_abs = 0.0
        max_rel = 0.0
        ok = True
        skipped: List[str] = []
        for name, r, g in zip(base_names, ref, got):
            if g is None:
                skipped.append(name)
                continue
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
        status = "fail" if not ok else ("partial" if skipped else "pass")
        reports.append(
            CaseReport(
                case.name,
                max_abs,
                max_rel,
                ok,
                status=status,
                skipped=tuple(skipped),
                failed_nodes=failed_nodes,
            )
        )
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


def _verdict(reports: Sequence[CaseReport]) -> str:
    if not all(r.ok for r in reports):
        return "fail"
    return "partial" if any(r.status == "partial" for r in reports) else "pass"


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
    culprit: Optional[dict] = None,
) -> dict:
    """Append one verified change to the manifest at ``path`` and return the entry.

    ``culprit`` is the bisect result for a failing step (see :func:`bisect_failure`),
    stored as plain data so the manifest stays JSON.
    """
    manifest = load_manifest(path)
    entry: dict = {
        "label": label,
        "command": command,
        "base_graph": base_graph,
        "output_graph": output_graph,
        "executor": executor,
        "test_set": case_set_id(cases),
        "cases": [case_to_dict(c) for c in cases],
        "verdict": _verdict(reports),
        "reports": [
            {
                "case": r.case,
                "max_abs_diff": r.max_abs_diff,
                "max_rel_diff": r.max_rel_diff,
                "ok": r.ok,
                "status": r.status,
                "skipped": list(r.skipped),
                "failed_nodes": list(r.failed_nodes),
            }
            for r in reports
        ],
    }
    if culprit is not None:
        entry["culprit"] = culprit
    manifest["steps"].append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return entry


class BisectError(RuntimeError):
    """The failing change cannot be bisected: an intermediate state is not a valid graph."""


@dataclass(frozen=True)
class ChangeUnit:
    """A connected group of changes, applied or reverted together.

    ``nodes`` labels each changed node as ``name`` or ``op:first_output``.
    ``blocks`` names the fusion blocks those changes fall in, such as ``Conv+Relu``;
    under a fusion preset, changes in the same block always share one unit. The
    ``_apply`` field holds the indices needed to build a state and is kept out of
    equality and the repr.
    """

    index: int
    nodes: Tuple[str, ...]
    initializers: Tuple[str, ...]
    blocks: Tuple[str, ...]
    _apply: Tuple[frozenset, Tuple[int, ...], frozenset] = field(
        repr=False, compare=False
    )


@dataclass(frozen=True)
class BisectResult:
    culprit: ChangeUnit
    units: Tuple[ChangeUnit, ...]
    evaluations: int


def _node_key(node) -> tuple:
    """Content identity of a node, ignoring its name."""
    attrs = tuple(
        a.SerializeToString(deterministic=True)
        for a in sorted(node.attribute, key=lambda a: a.name)
    )
    return (node.domain, node.op_type, tuple(node.input), tuple(node.output), attrs)


def _node_label(node) -> str:
    return node.name or f"{node.op_type}:{node.output[0] if node.output else ''}"


def _find(parent: List[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _match_nodes(
    bn: Sequence[NodeProto], cn: Sequence[NodeProto]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Pair candidate nodes with the base nodes that produce the same first output.

    Returns ``(matched, inverse)``: candidate index -> base index, and the reverse.
    """
    first_out: Dict[str, int] = {}
    for i, n in enumerate(bn):
        if n.output:
            first_out.setdefault(n.output[0], i)
    matched: Dict[int, int] = {}
    for j, n in enumerate(cn):
        if n.output and n.output[0] in first_out:
            matched[j] = first_out[n.output[0]]
    return matched, {i: j for j, i in matched.items()}


# Fusion presets as (head ops, tail ops). A head starts a block, and a tail joins
# the block when it consumes the block's output and nothing else does, so Conv+Relu
# or MatMul+Add form one block, the way an accelerator's fused kernel would.
_ACTIVATION_TAILS = (
    "Relu",
    "Clip",
    "Sigmoid",
    "Tanh",
    "HardSigmoid",
    "HardSwish",
    "LeakyRelu",
    "Elu",
    "Gelu",
    "Add",
    "Mul",
    "BatchNormalization",
)
FUSION_PRESETS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "node": ((), ()),
    "conv-act": (("Conv", "ConvTranspose"), _ACTIVATION_TAILS),
    "matmul-act": (("MatMul", "Gemm"), _ACTIVATION_TAILS),
    "default": (("Conv", "ConvTranspose", "MatMul", "Gemm"), _ACTIVATION_TAILS),
}


def _fusion_blocks(
    nodes: Sequence[NodeProto], outputs: Sequence[str], fusion: str
) -> List[int]:
    """Block id of each node of a topologically ordered graph under ``fusion``.

    A node's block id is the index of the block's first node. A graph output is
    never fused past, since its value has to exist on its own.
    """
    if fusion not in FUSION_PRESETS:
        raise ValueError(
            f"unknown fusion preset {fusion!r}; choose from {sorted(FUSION_PRESETS)}"
        )
    heads, tails = FUSION_PRESETS[fusion]
    uses: Dict[str, int] = {o: 1 for o in outputs}
    for n in nodes:
        for t in n.input:
            if t:
                uses[t] = uses.get(t, 0) + 1
    block_of: List[int] = []
    open_out: Dict[str, int] = {}  # tensor -> block that can still grow through it
    for idx, n in enumerate(nodes):
        first = n.input[0] if n.input else ""
        joined = n.op_type in tails and first in open_out and uses.get(first, 0) == 1
        bid = open_out.pop(first) if joined else idx
        block_of.append(bid)
        if n.output and n.output[0] and (joined or n.op_type in heads):
            open_out[n.output[0]] = bid
    return block_of


def _plan(base: ModelProto, cand: ModelProto, fusion: str) -> List[ChangeUnit]:
    """Split the difference between ``base`` and ``cand`` into units in apply order."""
    bn = list(base.graph.node)
    cn = list(cand.graph.node)
    matched, inverse = _match_nodes(bn, cn)
    block_of = _fusion_blocks(bn, [o.name for o in base.graph.output], fusion)
    block_of_tensor = {o: block_of[i] for i, n in enumerate(bn) for o in n.output if o}
    members: Dict[int, List[int]] = {}
    for i, b in enumerate(block_of):
        members.setdefault(b, []).append(i)

    diff_cand = [
        j
        for j in range(len(cn))
        if j not in matched or _node_key(cn[j]) != _node_key(bn[matched[j]])
    ]
    diff_base = [i for i in range(len(bn)) if i not in inverse]
    diff_base += [matched[j] for j in diff_cand if j in matched]

    base_inits = {t.name: t for t in base.graph.initializer}
    cand_inits = {t.name: t for t in cand.graph.initializer}
    init_diff = sorted(
        name
        for name in set(base_inits) | set(cand_inits)
        if name not in base_inits
        or name not in cand_inits
        or tensor_digest(numpy_helper.to_array(base_inits[name]))
        != tensor_digest(numpy_helper.to_array(cand_inits[name]))
    )

    # Items: (kind, ref, produced, consumed, position, block key). A block key is
    # ("block", id) for a node that belongs to a fusion block, ("cand", j) for an
    # added node outside any base block, and None for an initializer.
    items: List[tuple] = []
    for i in diff_base:
        items.append(
            (
                "base",
                i,
                bn[i].output,
                bn[i].input,
                inverse.get(i, len(cn) + i),
                ("block", block_of[i]),
            )
        )
    for j in diff_cand:
        if j in matched:
            key = ("block", block_of[matched[j]])
        else:
            key = next(
                (
                    ("block", block_of_tensor[t])
                    for t in cn[j].input
                    if t in block_of_tensor
                ),
                ("cand", j),
            )
        items.append(("cand", j, cn[j].output, cn[j].input, j, key))
    for name in init_diff:
        items.append(("init", name, [name], [], -1, None))

    parent = list(range(len(items)))
    producers: Dict[str, List[int]] = {}
    for k, item in enumerate(items):
        for t in item[2]:
            if t:
                producers.setdefault(t, []).append(k)
    for k, item in enumerate(items):
        for t in item[3]:
            for p in producers.get(t, []):
                parent[_find(parent, k)] = _find(parent, p)
    for ks in producers.values():
        for p in ks[1:]:
            parent[_find(parent, p)] = _find(parent, ks[0])
    first_in_key: Dict[tuple, int] = {}
    for k, item in enumerate(items):
        if item[5] is not None:
            if item[5] in first_in_key:
                parent[_find(parent, k)] = _find(parent, first_in_key[item[5]])
            else:
                first_in_key[item[5]] = k

    groups: Dict[int, List[int]] = {}
    for k in range(len(items)):
        groups.setdefault(_find(parent, k), []).append(k)
    ordered = sorted(groups.values(), key=lambda g: min(items[k][4] for k in g))

    units = []
    for idx, group in enumerate(ordered):
        base_rm: Set[int] = set()
        cand_add: Set[int] = set()
        inits: Set[str] = set()
        labels: List[str] = []
        init_names: List[str] = []
        blocks: List[str] = []
        block_ids: Set[int] = set()
        for k in sorted(group, key=lambda k: items[k][4]):
            kind, ref, key = items[k][0], items[k][1], items[k][5]
            if kind == "base":
                base_rm.add(ref)
                labels.append(_node_label(bn[ref]))
            elif kind == "cand":
                cand_add.add(ref)
                labels.append(_node_label(cn[ref]))
            else:
                assert isinstance(ref, str)
                inits.add(ref)
                init_names.append(ref)
            if key is not None:
                if key[0] == "block":
                    block_ids.add(key[1])
                    label = "+".join(bn[i].op_type for i in members[key[1]])
                else:
                    label = _node_label(cn[key[1]])
                if label not in blocks:
                    blocks.append(label)
        # A fused block runs as one kernel, so applying any change in it swaps in
        # every node of the block from the candidate, not only the changed ones.
        for b in block_ids:
            for i in members[b]:
                base_rm.add(i)
                if i in inverse:
                    cand_add.add(inverse[i])
        units.append(
            ChangeUnit(
                index=idx,
                nodes=tuple(labels),
                initializers=tuple(init_names),
                blocks=tuple(blocks),
                _apply=(
                    frozenset(base_rm),
                    tuple(sorted(cand_add)),
                    frozenset(inits),
                ),
            )
        )
    return units


class _Invalid(Exception):
    pass


def _topo_order(
    pool: List[Tuple[int, NodeProto]], avail: set, outputs: Sequence[str]
) -> List[NodeProto]:
    """Order ``pool`` so every node comes after the producers of its inputs."""
    producers: Dict[str, int] = {}
    for idx, (_, node) in enumerate(pool):
        for t in node.output:
            if t:
                if t in producers:
                    raise _Invalid(f"tensor {t} is produced twice")
                producers[t] = idx
    indeg = [0] * len(pool)
    consumers: Dict[int, List[int]] = {}
    for idx, (_, node) in enumerate(pool):
        deps = set()
        for t in node.input:
            if not t or t in avail:
                continue
            if t not in producers:
                raise _Invalid(
                    f"node {_node_label(node)} reads {t}, which nothing produces"
                )
            deps.add(producers[t])
        indeg[idx] = len(deps)
        for d in deps:
            consumers.setdefault(d, []).append(idx)
    heap = [(pool[i][0], i) for i in range(len(pool)) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(pool[i][1])
        for c in consumers.get(i, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, (pool[c][0], c))
    if len(order) != len(pool):
        raise _Invalid("the graph has a cycle")
    for o in outputs:
        if o not in avail and o not in producers:
            raise _Invalid(f"graph output {o} is not produced")
    return order


def _state(
    base: ModelProto,
    cand: ModelProto,
    header: ModelProto,
    units: Sequence[ChangeUnit],
    applied: int,
    inverse: Dict[int, int],
) -> ModelProto:
    """The base graph with the first ``applied`` units replaced by their candidate versions."""
    bn = list(base.graph.node)
    cn = list(cand.graph.node)
    removed: set = set()
    added: List[int] = []
    inits = {t.name: t for t in base.graph.initializer}
    cand_inits = {t.name: t for t in cand.graph.initializer}
    for unit in units[:applied]:
        base_rm, cand_add, names = unit._apply
        removed |= base_rm
        added.extend(cand_add)
        for name in names:
            if name in cand_inits:
                inits[name] = cand_inits[name]
            else:
                inits.pop(name, None)
    pool = [
        (inverse.get(i, len(cn) + i), bn[i]) for i in range(len(bn)) if i not in removed
    ]
    pool += [(j, cn[j]) for j in added]
    avail = {i.name for i in base.graph.input} | set(inits)
    outputs = [o.name for o in base.graph.output]
    order = _topo_order(pool, avail, outputs)
    state = ModelProto()
    state.CopyFrom(header)
    state.graph.node.extend(order)
    state.graph.initializer.extend(inits.values())
    return state


def bisect_failure(
    base: ModelProto,
    candidate: ModelProto,
    cases: Sequence[Case],
    atol: float = 1e-5,
    rtol: float = 1e-4,
    fusion: str = "default",
    backend: Optional[Backend] = None,
) -> Optional[BisectResult]:
    """Find the single modified subgraph that makes ``candidate`` fail against ``base``.

    Returns ``None`` when the candidate passes every case. Otherwise the change is
    split into units, and binary search over the apply order finds the first
    prefix that fails. Its last unit is the culprit.

    A unit is a connected group of modified nodes and initializers, where changes
    in the same fusion block are always grouped. ``fusion`` picks the block
    preset: ``"default"`` fuses Conv or MatMul/Gemm heads with their activation
    and elementwise tails, ``"conv-act"`` and ``"matmul-act"`` cover one head kind
    each, and ``"node"`` gives one block per node.

    The search assumes that once a prefix fails, every longer prefix also fails;
    a non-monotone change can point at a unit that is not the real cause, so the
    result should be read with that in mind.
    """
    if fusion not in FUSION_PRESETS:
        raise ValueError(
            f"unknown fusion preset {fusion!r}; choose from {sorted(FUSION_PRESETS)}"
        )
    if all(r.ok for r in check_equivalent(base, candidate, cases, atol, rtol, backend)):
        return None
    units = _plan(base, candidate, fusion)
    if not units:
        raise BisectError(
            "the candidate fails but differs from the base in no node or initializer"
        )
    _, inverse = _match_nodes(list(base.graph.node), list(candidate.graph.node))
    header = ModelProto()
    header.CopyFrom(base)
    header.graph.ClearField("node")
    header.graph.ClearField("initializer")
    header.graph.ClearField("value_info")

    lo, hi = 0, len(units)  # prefix lo passes (base), prefix hi fails (candidate)
    evaluations = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            state = _state(base, candidate, header, units, mid, inverse)
        except _Invalid as e:
            raise BisectError(
                f"prefix of {mid} units is not a valid graph: {e}"
            ) from None
        evaluations += 1
        if all(r.ok for r in check_equivalent(base, state, cases, atol, rtol, backend)):
            lo = mid
        else:
            hi = mid
    return BisectResult(
        culprit=units[hi - 1], units=tuple(units), evaluations=evaluations
    )
