"""Version and validation of the versioning manifest (``manifest.json``).

The schema is ``docs/schemas/versioning-manifest.v1.schema.json``. This module checks
the same rules in plain Python, so reading and writing a manifest needs no extra
dependency. The tests check the two against each other with ``jsonschema``.

Version 1 is the only version. A change that an older reader could misread bumps it.
"""

import math
import re
from typing import Any, Callable, Iterable, Mapping

MANIFEST_VERSION = 1

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERDICTS = ("pass", "partial", "fail")
_CASE_KINDS = ("generated", "supplied", "onnx_test_data")


def _fail(where: str, message: str) -> None:
    raise ValueError(f"{where}: {message}")


def _object(
    value: Any, where: str, required: Iterable[str], optional: Iterable[str] = ()
) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(where, "must be an object")
    required = set(required)
    allowed = required | set(optional)
    missing = sorted(required - set(value))
    if missing:
        _fail(where, f"missing {', '.join(missing)}")
    extra = sorted(set(value) - allowed)
    if extra:
        _fail(where, f"unexpected {', '.join(extra)}")
    return value


def _list(value: Any, where: str, item: Callable[[Any, str], None]) -> None:
    if not isinstance(value, list):
        _fail(where, "must be an array")
    for i, v in enumerate(value):
        item(v, f"{where}[{i}]")


def _str(value: Any, where: str) -> None:
    if not isinstance(value, str):
        _fail(where, "must be a string")


def _bool(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        _fail(where, "must be true or false")


def _int(value: Any, where: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(where, f"must be an integer >= {minimum}")


def _number(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "must be a number")
    if not math.isfinite(value):
        _fail(where, "must be finite; non-finite values are stored as null")


def _sha(value: Any, where: str) -> None:
    if not isinstance(value, str) or not _SHA256.match(value):
        _fail(where, "must be a sha256:<64 hex digits> digest")


def _str_list(value: Any, where: str) -> None:
    _list(value, where, _str)


def _digests(value: Any, where: str) -> None:
    if not isinstance(value, Mapping):
        _fail(where, "must be an object")
    for name, digest in value.items():
        _sha(digest, f"{where}.{name}")


def _enum(value: Any, where: str, choices: tuple) -> None:
    if value not in choices:
        _fail(where, f"must be one of {', '.join(choices)}")


def _spec(value: Any, where: str) -> None:
    _object(value, where, required=("name", "dtype", "shape", "low", "high"))
    _str(value["name"], f"{where}.name")
    _str(value["dtype"], f"{where}.dtype")
    _list(value["shape"], f"{where}.shape", lambda v, w: _int(v, w))
    _number(value["low"], f"{where}.low")
    _number(value["high"], f"{where}.high")


def _case(value: Any, where: str) -> None:
    _object(
        value,
        where,
        required=("name", "kind"),
        optional=("seed", "specs", "digests", "path", "directory", "test_set"),
    )
    _str(value["name"], f"{where}.name")
    kind = value["kind"]
    _enum(kind, f"{where}.kind", _CASE_KINDS)
    if kind == "generated":
        _object(value, where, required=("name", "kind", "seed", "specs", "digests"))
        _int(value["seed"], f"{where}.seed", minimum=-(2**63))
        _list(value["specs"], f"{where}.specs", _spec)
        _digests(value["digests"], f"{where}.digests")
    elif kind == "supplied":
        _object(value, where, required=("name", "kind", "path", "digests"))
        _str(value["path"], f"{where}.path")
        _digests(value["digests"], f"{where}.digests")
    else:
        _object(value, where, required=("name", "kind", "directory", "test_set"))
        _str(value["directory"], f"{where}.directory")
        _int(value["test_set"], f"{where}.test_set")


def _report(value: Any, where: str) -> None:
    _object(
        value,
        where,
        required=(
            "case",
            "max_abs_diff",
            "max_rel_diff",
            "ok",
            "status",
            "skipped",
            "failed_nodes",
        ),
    )
    _str(value["case"], f"{where}.case")
    for key in ("max_abs_diff", "max_rel_diff"):
        if value[key] is not None:
            _number(value[key], f"{where}.{key}")
    _bool(value["ok"], f"{where}.ok")
    _enum(value["status"], f"{where}.status", _VERDICTS)
    _str_list(value["skipped"], f"{where}.skipped")
    _str_list(value["failed_nodes"], f"{where}.failed_nodes")


def _culprit(value: Any, where: str) -> None:
    _object(
        value,
        where,
        required=("fusion", "nodes", "blocks", "initializers", "evaluations"),
    )
    _str(value["fusion"], f"{where}.fusion")
    for key in ("nodes", "blocks", "initializers"):
        _str_list(value[key], f"{where}.{key}")
    _int(value["evaluations"], f"{where}.evaluations")


def _step(value: Any, where: str) -> None:
    _object(
        value,
        where,
        required=(
            "label",
            "command",
            "base_graph",
            "output_graph",
            "executor",
            "test_set",
            "cases",
            "verdict",
            "reports",
        ),
        optional=("culprit",),
    )
    for key in ("label", "command", "executor"):
        _str(value[key], f"{where}.{key}")
    for key in ("base_graph", "output_graph", "test_set"):
        _sha(value[key], f"{where}.{key}")
    _list(value["cases"], f"{where}.cases", _case)
    _enum(value["verdict"], f"{where}.verdict", _VERDICTS)
    _list(value["reports"], f"{where}.reports", _report)
    if "culprit" in value:
        _culprit(value["culprit"], f"{where}.culprit")


def _build(value: Any, where: str) -> None:
    _object(
        value,
        where,
        required=(
            "output",
            "file_digest",
            "source_graph",
            "simplify",
            "simplify_checked",
            "onnxsim_version",
            "executor",
        ),
    )
    _str(value["output"], f"{where}.output")
    _sha(value["file_digest"], f"{where}.file_digest")
    _sha(value["source_graph"], f"{where}.source_graph")
    if value["simplify"] is not None and not isinstance(value["simplify"], Mapping):
        _fail(f"{where}.simplify", "must be an object or null")
    if value["simplify_checked"] is not None:
        _bool(value["simplify_checked"], f"{where}.simplify_checked")
    _str(value["onnxsim_version"], f"{where}.onnxsim_version")
    _str(value["executor"], f"{where}.executor")


def validate_manifest(manifest: Any) -> None:
    """Raise ``ValueError`` naming the first place a manifest breaks version 1."""
    _object(
        manifest, "manifest", required=("version", "steps"), optional=("base", "builds")
    )
    version = manifest["version"]
    if isinstance(version, bool) or version != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest version {version!r}; this code reads version {MANIFEST_VERSION}"
        )
    if "base" in manifest:
        base = _object(
            manifest["base"], "base", required=("file", "file_digest", "graph")
        )
        _str(base["file"], "base.file")
        _sha(base["file_digest"], "base.file_digest")
        _sha(base["graph"], "base.graph")
    _list(manifest["steps"], "steps", _step)
    if "builds" in manifest:
        _list(manifest["builds"], "builds", _build)


def finite_or_none(value: float) -> float | None:
    """A report difference as stored: the number, or None when it is not finite."""
    return float(value) if math.isfinite(value) else None
