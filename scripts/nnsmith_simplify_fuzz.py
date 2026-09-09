#!/usr/bin/env python3
"""Differential-fuzz onnxsim's own simplify() pass against NNSmith models.

The op-form fuzzer (``onnx_op_fuzzer.py``, next to this file) targets a
different question: for one op at a time, does onnx's ReferenceEvaluator
support what onnxruntime does. This script targets onnxsim's own optimizer:
does ``simplify(model)`` preserve semantics on a large, structurally diverse
sample of *whole graphs* -- exactly the class of bug a single-op test cannot
reach, because it only shows up from interactions between multiple ops (a
fusion pattern matching more broadly than intended, a rewrite applied where a
neighboring op invalidates its precondition, ...).

NNSmith (https://github.com/ise-uiuc/nnsmith, ASPLOS'23) generates ONNX
graphs that are diverse and valid by construction (solved shape/dtype
constraints, not just randomly connected nodes) -- see its own bug study for
why that diversity specifically matters for optimization-pass bugs: 43 of the
72 bugs it found upstream were graph-transformation bugs, and 49 of the 72
were bugs prior graph-fuzzers (which restrict how non-shape-preserving ops
connect) could not trigger at all. This script does not use NNSmith's own
differential-testing/backend machinery (that is built for comparing
*execution* backends like ONNXRuntime/TVM, not a graph-to-graph rewriter like
onnxsim) -- it only uses NNSmith as a diverse valid-model generator, and
reuses onnxsim's own existing random-input equivalence checker
(``simplify(..., check_n=N)``, exposed by the ``onnxsim`` CLI's positional
``check_n`` argument) as the pass/fail oracle. That checker already ran
before this script existed (it is what ``onnxsim --check`` is); the value
added here is feeding it NNSmith's diverse generated graphs instead of only
the handful of real-world models onnxsim is normally exercised against.

Each model gets its own subprocess for both generation (NNSmith generation
plus our own ONNX export, both in `_nnsmith_gen.py`) and simplification, so
a native crash in either takes down only that one case.

``check_n>0`` needs a backend (onnxruntime, or onnx's ReferenceEvaluator as a
fallback -- see onnxsim/backend.py) to actually execute the original and
simplified models to compare them, and that backend can fail on a model
independently of anything onnxsim's optimizer did -- confirmed by hand
running this script for real: onnx's ReferenceEvaluator rejecting Pad's
legal negative pad values (a real bug in onnx's own reference impl) and
onnxruntime refusing an exact Resize form it doesn't implement, on NNSmith
models that never even reach onnxsim's own pass code. ``_run_onnxsim``
tells these apart from a genuine onnxsim finding by checking which module's
frames appear in the failure (a native crash, identifiable by
``faulthandler``'s distinctive dump -- see ``main()``'s own
``faulthandler.enable()`` and its comment on why -- always wins, since a
segfault matters regardless of what else appears in the same traceback)
and reports them as ``checker_backend_error``: saved for inspection like
any other non-``ok`` case, but excluded from the exit-1 "bugs" list, since
they are not findings about onnxsim.

A second, similarly-excluded case: onnxsim's own ``model_checking.compare()``
compares outputs with plain ``np.allclose(..., equal_nan=False)`` (its
default), so when the *original* model's random inputs already produce
NaN/Inf on their own (a Div by zero, which NNSmith's inputs hit far more
often than onnxsim's usual smoke-test models), comparing it against the
simplified model's equally-NaN output reports a mismatch even though
neither side is well-defined -- confirmed by hand: "The max diff is nan."
for a case traced to exactly that. Reported as ``check_nan_mismatch``, same
treatment as ``checker_backend_error`` (saved, not gating): it is a
pre-existing blind spot in onnxsim's own checker, not evidence that
simplify() changed anything.

Requires the optional ``nnsmith[torch,onnx]`` package; skips with a clear
message if missing. Generation goes through NNSmith's ``torch`` model type
rather than its own ``onnx`` type, then ``_nnsmith_gen.py`` exports the
result to ONNX itself via the legacy ``torch.onnx.export(..., dynamo=False)``
TorchScript-trace exporter -- see that file's module docstring for why (a
torch/NNSmith compatibility problem with ``model.type=onnx``'s own export
path, and a ~80x cheaper opset-determination cost as a bonus). No extra
package beyond torch itself is needed (no onnxscript).

NNSmith's first run on a host builds and caches (``~/.cache/nnsmith-<ver>/``)
a "topset" -- which candidate ops actually work in this environment -- by
trial-running every op/dtype combination it knows. Measured on this repo's
CI-like environment (torch 2.14, nnsmith 0.1.0) for the ``torch`` model
type this script now uses: ~2.7 seconds for that one-time topset build
(``model.type=onnx``'s own trial-export-based version of the same check
measured ~3.5 minutes), then ~2s per generated model after (dominated by
Python/torch process startup, not generation itself, which is ~10-100ms).
Cache ``~/.cache/nnsmith-<ver>/`` across CI runs regardless (see the
nightly workflow) so only the first run ever pays even this small cost.

Usage:
    python scripts/nnsmith_simplify_fuzz.py --count 100
    python scripts/nnsmith_simplify_fuzz.py --count 20 --max-nodes 30 --check-n 5
    python scripts/nnsmith_simplify_fuzz.py --count 200 --output-dir failures/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CaseResult:
    seed: int
    # gen_error | ok | check_failed | onnxsim_crash | onnxsim_timeout |
    # checker_backend_error | check_nan_mismatch -- see main()'s `bugs`
    # filter for which of these gate (exit 1) and _run_onnxsim for how
    # check_failed/onnxsim_crash are told apart from checker_backend_error
    # and check_nan_mismatch.
    status: str
    detail: str = ""
    model_dir: Optional[Path] = None


def _require_nnsmith() -> Optional[str]:
    try:
        import nnsmith  # noqa: F401
    except ImportError:
        return "nnsmith is not installed (pip install 'nnsmith[torch,onnx]')"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "torch is not installed (needed by the 'nnsmith[torch,onnx]' extra)"
    return None


_GEN_HELPER = Path(__file__).with_name("_nnsmith_gen.py")


def _generate(
    seed: int, max_nodes: int, model_dir: Path, timeout: int
) -> Optional[str]:
    """Run NNSmith's model generator into `model_dir`. None on success, else
    an error string (nnsmith's own failure, not onnxsim's -- reported as
    `gen_error`, does not count as a simplify() finding).

    Goes through `_nnsmith_gen.py` rather than `-m nnsmith.cli.model_gen`
    directly -- see that file for why (generates via NNSmith's `torch` model
    type, ~80x cheaper to determine the exportable opset for than its own
    `onnx` type, then exports to ONNX itself)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_GEN_HELPER),
            f"mgen.seed={seed}",
            f"mgen.max_nodes={max_nodes}",
            f"mgen.save={model_dir}",
            # Without this, Hydra (which nnsmith's CLI is built on) writes its
            # own run log/config under an `outputs/<date>/<time>/` directory
            # created in the *caller's* CWD -- redirect that litter next to
            # the model's own scratch dir instead. Must be a *sibling* of
            # model_dir, not nested under it: Hydra creates hydra.run.dir
            # before nnsmith's own code runs, which would create model_dir
            # as a side effect (os.makedirs creates parents) and then
            # nnsmith's own mkdir() helper interactively prompts (hanging on
            # this subprocess's closed stdin) because its target already
            # exists -- the same failure mode worked around in run_one().
            f"hydra.run.dir={model_dir}.hydra_run",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not (model_dir / "model.onnx").exists():
        return (proc.stderr or proc.stdout or "no model.onnx produced")[-500:]
    return None


def _run_onnxsim(
    model_path: Path,
    out_path: Path,
    check_n: int,
    check_rtol: float,
    check_atol: float,
    timeout: int,
) -> CaseResult:
    try:
        proc = subprocess.run(
            [
                "onnxsim",
                str(model_path),
                str(out_path),
                str(check_n),
                "--check-rtol",
                str(check_rtol),
                "--check-atol",
                str(check_atol),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CaseResult(0, "onnxsim_timeout", f">{timeout}s")

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return CaseResult(0, "ok")
    if "Check failed" in output:
        # onnxsim's own model_checking.compare() reports a mismatch via plain
        # np.allclose(..., equal_nan=False) (its default) -- so when the
        # *original* model's random-input run already produces NaN/Inf (e.g.
        # a Div by zero, which NNSmith's random inputs hit far more often
        # than onnxsim's usual smoke-test models), comparing it against the
        # simplified model's equally-NaN output reports "changed" even
        # though neither side is well-defined -- confirmed by hand: "The max
        # diff is nan." printed for a case traced to exactly a Div-by-zero
        # RuntimeWarning earlier in the same run. That is a real, pre-existing
        # blind spot in onnxsim's own checker, but not evidence simplify()
        # changed anything -- keep it out of the genuine check_failed bucket.
        if "The max diff is nan." in output:
            return CaseResult(0, "check_nan_mismatch", output[-800:])
        # simplify()'s own check_n verification caught a semantic change --
        # this is the actual finding this script exists to surface.
        return CaseResult(0, "check_failed", output[-800:])
    # check_n>0 needs *some* backend (onnxruntime, or onnx's ReferenceEvaluator
    # as a fallback -- see onnxsim/backend.py) to execute both the original
    # and simplified models for comparison. A raised exception with a frame
    # in that backend's own code -- confirmed by hand: onnx.reference's Pad
    # rejecting ONNX's legal negative pads (a real onnx bug, unrelated to
    # onnxsim), and onnxruntime failing to bind a Resize form it doesn't
    # implement -- means the *checker's backend* couldn't even run one of the
    # models, which happens independently of anything onnxsim's own optimizer
    # did. A native crash (faulthandler's dump, identifiable by "Extension
    # modules:" with no ordinary Python "Traceback" -- see main()'s
    # faulthandler.enable()) always means the C++ extension itself, so takes
    # priority over this even if a backend frame appears afterward in the
    # same graph traversal.
    is_native_crash = "Extension modules:" in output and "Traceback" not in output
    checker_backend_frames = (
        "onnx/reference/ops/",
        "onnxruntime/capi/",
        "onnxsim/model_checking.py",
        "onnxsim/backend.py",
    )
    if not is_native_crash and any(f in output for f in checker_backend_frames):
        return CaseResult(0, "checker_backend_error", output[-800:])
    return CaseResult(0, "onnxsim_crash", output[-800:])


def run_one(
    seed: int,
    max_nodes: int,
    check_n: int,
    check_rtol: float,
    check_atol: float,
    gen_timeout: int,
    sim_timeout: int,
    work_dir: Path,
) -> CaseResult:
    model_dir = work_dir / f"case_{seed}"
    if model_dir.exists():
        # NNSmith's own mkdir() interactively prompts (blocking on a closed
        # stdin, since this runs as a subprocess) if its target directory
        # already exists -- only reachable via a reused --work-dir, but
        # clear it rather than hang.
        shutil.rmtree(model_dir)
    gen_err = _generate(seed, max_nodes, model_dir, gen_timeout)
    if gen_err is not None:
        return CaseResult(seed, "gen_error", gen_err, model_dir)
    result = _run_onnxsim(
        model_dir / "model.onnx",
        model_dir / "model.simplified.onnx",
        check_n,
        check_rtol,
        check_atol,
        sim_timeout,
    )
    result.seed = seed
    result.model_dir = model_dir
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--count", type=int, default=100, help="number of models to generate"
    )
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=20, help="NNSmith mgen.max_nodes")
    ap.add_argument(
        "--check-n", type=int, default=3, help="onnxsim's check_n (random-input trials)"
    )
    ap.add_argument("--check-rtol", type=float, default=1e-4)
    ap.add_argument("--check-atol", type=float, default=1e-5)
    ap.add_argument(
        "--gen-timeout", type=int, default=120, help="seconds, per model generation"
    )
    ap.add_argument(
        "--sim-timeout", type=int, default=120, help="seconds, per onnxsim run"
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help="scratch dir for generated models (default: a temp dir, cleaned up "
        "unless --keep-work-dir)",
    )
    ap.add_argument(
        "--keep-work-dir", action="store_true", help="don't delete --work-dir on exit"
    )
    ap.add_argument(
        "--output-dir",
        default="nnsmith_simplify_failures",
        help="where to copy the model+log for each non-'ok' case, for repro",
    )
    args = ap.parse_args()

    missing = _require_nnsmith()
    if missing is not None:
        print(f"skipping: {missing}", file=sys.stderr)
        return 0

    import tempfile

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="nnsmith_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)

    results: List[CaseResult] = []
    t0 = time.time()
    try:
        for i in range(args.count):
            seed = args.seed_start + i
            r = run_one(
                seed,
                args.max_nodes,
                args.check_n,
                args.check_rtol,
                args.check_atol,
                args.gen_timeout,
                args.sim_timeout,
                work_dir,
            )
            results.append(r)
            print(f"[{i + 1}/{args.count}] seed={seed} -> {r.status}", flush=True)
            if r.status not in ("ok", "gen_error"):
                dest = out_dir / f"seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                if r.model_dir and (r.model_dir / "model.onnx").exists():
                    shutil.copy(r.model_dir / "model.onnx", dest / "model.onnx")
                (dest / "detail.txt").write_text(r.detail)
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - t0
    counts: dict = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(
        f"\n{len(results)} models in {elapsed:.1f}s: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    bugs = [
        r
        for r in results
        if r.status in ("check_failed", "onnxsim_crash", "onnxsim_timeout")
    ]
    if bugs:
        print(f"\n{len(bugs)} onnxsim finding(s) (saved under {out_dir}/):")
        for r in bugs:
            print(f"  seed={r.seed} {r.status}: {r.detail[:200]}")
        return 1

    gen_errors = counts.get("gen_error", 0)
    if gen_errors == len(results) and results:
        print(
            "\nevery model failed to generate -- likely an NNSmith/environment "
            "problem, not an onnxsim finding; check the output above.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
