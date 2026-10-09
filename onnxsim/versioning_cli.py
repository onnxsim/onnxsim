"""Command line for the versioning workflow.

    python -m onnxsim.versioning_cli init ORIGINAL.onnx PROJECT_DIR
    python -m onnxsim.versioning_cli status PROJECT_DIR
    python -m onnxsim.versioning_cli build PROJECT_DIR -o OUT.onnx [-w WEIGHTS.onnx ...]
    python -m onnxsim.versioning_cli verify PROJECT_DIR --cases CASES.json [--label ...]

``verify`` exits 0 when every case passes, 1 when a case fails (the culprit is
printed and recorded), and 2 on usage or data errors. See
:mod:`onnxsim.versioning_project` for what a project directory holds.
"""

import argparse
import importlib
import sys
from typing import List, Optional

from onnxsim.versioning import FUSION_PRESETS, graph_hash, save_snapshot
from onnxsim.versioning_project import (
    build_project,
    init_project,
    load_cases,
    project_status,
    verify_project,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def _cmd_init(args) -> int:
    manifest = init_project(args.original, args.directory)
    print(f"initialized {args.directory}: base graph {manifest['base']['graph']}")
    return EXIT_PASS


def _cmd_status(args) -> int:
    status = project_status(args.directory)
    print(f"base file:  {status['base']['file']}")
    print(f"base graph: {status['base']['graph']}")
    print(f"graph now:  {status['graph']}")
    print(f"steps:      {len(status['steps'])}")
    for i, step in enumerate(status["steps"], 1):
        print(f"  {i}. {step['label'] or '(unlabeled)'}: {step['verdict']}")
    return EXIT_PASS


def _cmd_build(args) -> int:
    _, candidate, text = build_project(args.directory, args.weights)
    digest = save_snapshot(candidate, args.output)
    print(f"wrote {args.output} (file {digest})")
    print(f"graph: {graph_hash(text)}")
    return EXIT_PASS


def _load_backend(spec: str):
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise ValueError("--backend takes MODULE:FUNCTION")
    return getattr(importlib.import_module(module), attr)


def _cmd_verify(args) -> int:
    cases = load_cases(args.cases)
    backend = _load_backend(args.backend) if args.backend else None
    result = verify_project(
        args.directory,
        cases,
        weights=args.weights,
        label=args.label,
        command=args.command_text
        or " ".join(["python -m onnxsim.versioning_cli", *args.argv]),
        atol=args.atol,
        rtol=args.rtol,
        fusion=args.fusion,
        record=not args.no_record,
        backend=backend,
    )
    print(f"{'case':<24} {'max abs':>12} {'max rel':>12}  result")
    for r in result.reports:
        label = {"pass": "pass", "partial": "partial", "fail": "FAIL"}[r.status]
        print(f"{r.case:<24} {r.max_abs_diff:>12.4g} {r.max_rel_diff:>12.4g}  {label}")
        if r.failed_nodes:
            print(
                f"  ops that failed on the backend (filled at random): {', '.join(r.failed_nodes)}"
            )
        if r.skipped:
            print(f"  outputs not judged: {', '.join(r.skipped)}")
    if result.passed:
        partial = any(r.status == "partial" for r in result.reports)
        print(f"verdict: {'partial' if partial else 'pass'}")
        return EXIT_PASS
    print("verdict: fail")
    if result.culprit is not None:
        c = result.culprit
        print(f"culprit: unit {c.index} (fusion={args.fusion})")
        print(f"  blocks:       {', '.join(c.blocks) or '-'}")
        print(f"  nodes:        {', '.join(c.nodes) or '-'}")
        print(f"  initializers: {', '.join(c.initializers) or '-'}")
    return EXIT_FAIL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m onnxsim.versioning_cli",
        description="Version an ONNX graph as text, verify changes, and bisect failures.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("init", help="create a project from an original .onnx file")
    p.add_argument("original", help="the original .onnx file")
    p.add_argument("directory", help="project directory to create")
    p.set_defaults(func=_cmd_init)

    p = sub.add_parser("status", help="show the base file, graph hash and steps")
    p.add_argument("directory")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("build", help="write the project's graph as a .onnx file")
    p.add_argument("directory")
    p.add_argument("-o", "--output", required=True, help="the .onnx file to write")
    p.add_argument(
        "-w",
        "--weights",
        action="append",
        default=[],
        metavar="FILE",
        help="extra .onnx file to take weights from (repeatable)",
    )
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("verify", help="check the graph against its base on test cases")
    p.add_argument("directory")
    p.add_argument("--cases", required=True, help="JSON file listing the test cases")
    p.add_argument(
        "-w",
        "--weights",
        action="append",
        default=[],
        metavar="FILE",
        help="extra .onnx file to take weights from (repeatable)",
    )
    p.add_argument("--label", default="", help="name for the recorded step")
    p.add_argument(
        "--command",
        dest="command_text",
        default="",
        help="command text stored with the step (default: this invocation)",
    )
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument(
        "--fusion",
        choices=sorted(FUSION_PRESETS),
        default="default",
        help="fusion preset used to group units when bisecting a failure",
    )
    p.add_argument(
        "--backend",
        metavar="MODULE:FUNCTION",
        help="run the candidate on this backend, continuing past ops it cannot run",
    )
    p.add_argument(
        "--no-record",
        action="store_true",
        help="check without appending a step to the manifest",
    )
    p.set_defaults(func=_cmd_verify)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(argv)
    args.argv = argv
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
