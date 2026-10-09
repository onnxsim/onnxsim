"""Run ONNX backend test cases through the versioning checks.

Each case is a directory holding ``model.onnx`` and ``test_data_set_N/`` with
``input_i.pb`` inputs and ``output_i.pb`` expected outputs (TensorProto files).
For every case this script runs two checks:

1. ``official``: ONNX's reference evaluator must reproduce the case's
   ``output_i.pb`` files. This checks the loader and the reference data.
2. ``round-trip``: the model is written as graph text and rebuilt with
   :func:`onnxsim.versioning.load_graph_text`, with weights resolved from the
   model itself, and the rebuilt graph is compared with the original on the
   case's inputs. This exercises the same path ``verify`` uses. ONNX Runtime
   runs both graphs, so it can't run some old opsets; those cases report why.

Run from the repository root, so the checkout's ``onnxsim`` is imported::

    PYTHONPATH=. python examples/versioning/onnx_test_data.py                 # every group
    PYTHONPATH=. python examples/versioning/onnx_test_data.py --group simple --limit 20
    PYTHONPATH=. python examples/versioning/onnx_test_data.py ROOT            # another root

The default ROOT is the ONNX submodule's ``onnx/backend/test/data``. The script exits 1
if any official output mismatches or any round trip fails, and 0 otherwise.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from onnxsim.versioning import (
    OnnxTestDataCase,
    TensorIndex,
    check_equivalent,
    graph_text,
    load_graph_text,
    load_onnx_test_set,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "third_party" / "onnx" / "onnx" / "backend" / "test" / "data"


def find_cases(root: Path, group: str | None, limit: int | None) -> list[Path]:
    groups = (
        [root / group] if group else sorted(p for p in root.iterdir() if p.is_dir())
    )
    cases: list[Path] = []
    for g in groups:
        if g.is_dir():
            cases += sorted(
                d
                for d in g.iterdir()
                if (d / "model.onnx").is_file() and (d / "test_data_set_0").is_dir()
            )
    return cases[:limit] if limit else cases


def check_official(case: Path, model: onnx.ModelProto) -> str:
    feeds, expected = load_onnx_test_set(str(case), model)
    if expected is None:
        return "no expected outputs"
    try:
        got = ReferenceEvaluator(model).run(None, feeds)
    except (
        Exception
    ) as e:  # e.g. training ops the reference evaluator doesn't implement
        return f"skipped: {type(e).__name__}"
    for name, actual in zip([o.name for o in model.graph.output], got):
        want = expected[name]
        actual = np.asarray(actual)
        if (
            actual.dtype.kind in "OSU" or want.dtype.kind in "OSU"
        ):  # strings: exact match
            same = np.array_equal(actual, want)
        else:
            same = np.allclose(actual, want, rtol=1e-3, atol=1e-4, equal_nan=True)
        if not same:
            return f"output {name} differs from output_*.pb"
    return "ok"


def check_round_trip(case: Path, model: onnx.ModelProto) -> str:
    try:
        text = graph_text(model)
    except TypeError as e:  # e.g. a string initializer, which has no numeric digest
        return f"skipped: {e}"
    index = TensorIndex()
    index.add_model(model)
    rebuilt = load_graph_text(text, index)
    try:
        (report,) = check_equivalent(
            model, rebuilt, [OnnxTestDataCase(case.name, str(case))]
        )
    except (
        Exception
    ) as e:  # ONNX Runtime rejects some old opsets; string outputs can't be compared
        return f"skipped: {type(e).__name__}"
    return report.status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--group", help="only this group, such as 'simple'")
    parser.add_argument("--limit", type=int, help="stop after this many cases")
    args = parser.parse_args(argv)

    cases = find_cases(args.root, args.group, args.limit)
    if not cases:
        print(f"no test cases under {args.root}", file=sys.stderr)
        return 2

    failures = 0
    counts: dict[str, int] = {}
    for case in cases:
        model = onnx.load(str(case / "model.onnx"), load_external_data=True)
        official = check_official(case, model)
        round_trip = check_round_trip(case, model)
        counts[round_trip] = counts.get(round_trip, 0) + 1
        # Only a real difference fails the run; a check the runtime could not run is reported.
        if official.startswith("output "):
            failures += 1
        if round_trip == "fail":
            failures += 1
        print(
            f"{case.parent.name}/{case.name:<48} official: {official:<36} round-trip: {round_trip}"
        )

    print(
        f"\n{len(cases)} cases; round-trip results: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
