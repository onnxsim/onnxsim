# A worktree can silently import the wrong `onnxsim` -- confirmed, fixed, audited

Found while building the Whisper training case (`docs/axera-on-device-training-
handoff.md`'s memory-heavy case): a script under `scripts/axera/`, run from an
isolated `git worktree`, can silently `import onnxsim` from a *different*
checkout than the one it lives in. This records the confirmed mechanism, the
fix, and an audit of whether any of this project's own merged work was
actually affected.

## The mechanism, confirmed by direct repro

`onnxsim` here is `pip install -e .`-editable, which registers a
`sys.meta_path` finder (`__editable___onnxsim_*_finder.py`) via
`sys.meta_path.append(_EditableFinder)` -- **appended**, so it is tried only
after the builtin `PathFinder` (which searches `sys.path` in order) fails to
find `onnxsim` anywhere.

Running a script directly (`python3 scripts/axera/foo.py`) puts that script's
own directory -- `scripts/axera`, not the repository root -- at `sys.path[0]`.
`scripts/axera` has no `onnxsim` package inside it, and no other `sys.path`
entry has a real one either (it is editable-installed, not copied into
site-packages), so `PathFinder` always fails and control falls through to
`_EditableFinder`, which unconditionally maps `onnxsim` to the path recorded
at install time -- the main checkout -- **regardless of which worktree's
script was actually invoked.**

Confirmed empirically, not just reasoned about: two worktrees, each given a
distinguishable marker line appended to `onnxsim/__init__.py`. Invoking a
worktree's own `scripts/axera/<script>.py` directly crashed on the *main
checkout's* marker every time, before the fix; after the fix, it crashes on
its own worktree's marker instead.

**`pytest`/`python3 -m ...` are not affected**, and this was also confirmed
by repro, not assumed. `-m`/`-c` invocations put the current working
directory (not a script's own directory) at `sys.path[0]`; a fork that `cd`s
into its own worktree before running `pytest tests/...` (the standard,
overwhelmingly common pattern in this project's own history) already has that
worktree's repository root -- which *does* contain its own `onnxsim/` --
first on `sys.path`, so `PathFinder` finds and uses it directly and
`_EditableFinder` is never consulted. The hazard is specific to **directly
executing a `scripts/axera/*.py` file**, independent of `cwd`.

## The fix, and a correction to the fix

`scripts/axera/_local_import.py` gained `ensure_repo_onnxsim()`. The first
version inserted this checkout's own repository root at the front of
`sys.path`, so `PathFinder` would resolve the top-level `onnxsim` package
directly from this checkout's source tree. Re-ran the two-worktree repro
after that version and it worked: a script invoked from worktree B correctly
crashed on worktree B's own marker.

**That version broke CI on every PR merged after it** --
`ModuleNotFoundError: No module named 'onnxsim.onnxsim_cpp2py_export'` in the
`Pulsar2 compatibility check` job, on PRs (#1352, #1353) that never touched
the code path it broke. Cause: once `PathFinder` resolves the top-level
package from a plain checkout, `onnxsim.__path__` becomes just that
checkout's `onnxsim/` directory -- and the compiled extension is a build
artifact, not a tracked source file, so it isn't there. Every later
submodule import (`onnxsim.onnxsim_cpp2py_export` included) searches only
that `__path__`, never falling back to the `sys.meta_path` finder that used
to serve it. The CI job's own workflow comment already documented exactly
this risk ("the repo root contains the `onnxsim/` source dir (no compiled
extension), which would shadow the installed wheel") and works around it by
running from `runner.temp`, not the checkout -- a `cwd`-based protection the
`sys.path`-inserting fix bypassed entirely, since it doesn't look at `cwd`.

**Corrected**: import `onnxsim` first, however it would normally resolve (so
whatever already-working resolution -- the editable install's finder in a
worktree, the properly-installed wheel in CI -- gets to run and discover
where the real compiled extension lives), then *prepend* this checkout's own
`onnxsim/` directory to the already-resolved package's `__path__`, rather
than replacing `sys.path`-level resolution. A submodule search now finds
this checkout's own copy first if it has one, and falls through to the
original entry (which does have the extension) if it doesn't.

**This corrected version is safe, and confirmed weaker than first
described.** Re-running the original two-worktree repro after the merge-based
fix found a real limitation: `onnxsim/__init__.py` eagerly imports the large
majority of the package (`graph_grad` included, transitively, through other
eagerly-imported modules) as part of running `import onnxsim` itself -- before
this function ever gets to touch `__path__`. Those submodules are already
cached in `sys.modules`, unaffected by the later `__path__` prepend. Directly
confirmed: a worktree-invoked script calling the fixed function still
received the *main checkout's* `graph_grad.py`, not its own. The corrected
version reliably stops the CI regression above and correctly redirects any
submodule that genuinely isn't already cached by the time it runs, but does
**not** reliably isolate `graph_grad` specifically -- the one submodule this
whole investigation started because of. `_local_import.py`'s own docstring
now says so plainly and points at `fresh()` (already used for axera-local
`models`/`worker`) as the tool that actually guarantees isolation for one
named submodule, for anyone who needs that guarantee rather than
best-effort coverage.

Given the audit below already found zero confirmed impact from the original
hazard, stopping here -- safe, and better than nothing, rather than building
a `sys.meta_path`-finder-based fix that intercepts `onnxsim`'s own spec
before `__init__.py` runs -- was the right amount of engineering for a risk
already shown to be low-impact in practice.

**Not yet applied to `scripts/axera/build_whisper_train_step.py`** (added by
the still-open Whisper training-case PR) -- that file imports `onnxsim`
indirectly through `build_resident_train_step.py`, which this fix does cover,
but whoever merges that PR should double check it inherits the fix cleanly
rather than assuming so.

## Audit: did this affect any of this project's own merged work?

Checked every PR from this session's on-device-training work (from the first
resident-weight speedup PR through the Qwen-Drive feasibility check) against
two questions: did it modify a file under `onnxsim/` (the shared, importable
package the hazard mechanism specifically corrupts -- as opposed to
`scripts/axera/*.py`, which is never itself the thing silently substituted),
and if so, how was it verified.

**Only one PR modified `onnxsim/*.py`: #1341** (`onnxsim/graph_grad.py` --
the `Split` backward rule and its `_MULTI_OUTPUT_RULES` table). Its own PR
description's test plan is exclusively `pytest tests/...` invocations -- the
confirmed-safe pattern. Re-ran that same test list fresh, against current
master, in a clean worktree: **482 passed, 2 skipped** (a subset of its
originally-reported "492 passed" -- this run omitted
`test_formal_verify_grad_*`, not otherwise different). No discrepancy found.
**Verdict: fine**, and was never actually exposed to this hazard in the first
place, independent of the re-verification.

**Every other session PR that touched `onnxsim/` files touched none** --
scripts/axera/build_resident_train_step.py`, `legalize.py`, `resident_runner.c`,
docs, and new test files, but not the shared package itself. Nothing else
met the audit's own risk criterion.

**A real but lower-severity, unconfirmed-impact residual risk, worth
naming rather than hiding**: `build_resident_train_step.py` (and its Whisper
counterpart) *were* run directly as scripts (`python3
build_resident_train_step.py ...`) in essentially every hardware PR in this
thread, to produce the ONNX graph that was then compiled and measured on
real hardware -- and, before this fix, any such run would have resolved
`onnxsim.graph_grad`/`onnxsim.qat_graph`/`onnxsim.compile_training` from
whatever the shared main checkout happened to have checked out at that
moment, not necessarily a clean "current master". Concretely observed during
this very audit: the main checkout was sitting on a stale, already-merged
branch (`axera-batch-resident-step`) with an unrelated 300-line uncommitted
diff, left over from some earlier fork -- exactly the kind of drift that
makes "whatever the main checkout happens to have" an unreliable stand-in
for "current master". This did not corrupt any specific PR's own diff (none
of those hardware PRs edited `onnxsim/` themselves, so there was no
worktree-local change to lose), but it does mean the *exact* version of
`graph_grad.py`/`qat_graph.py` used to build any given hardware PR's graph
was never guaranteed to be the one its own branch point implied. No evidence
was found that this actually produced a wrong result anywhere (none of the
CNN/audio training-step graphs in this thread depend on anything that
changed in `onnxsim/graph_grad.py` across the relevant time window -- #1341's
`Split` rule is audio-architecture-specific and unused by any resnet18/
resnet50/Whisper graph), but it could not be ruled out by code-reading alone
either. Flagged here rather than asserted safe.

## What this doesn't cover

- No hardware was touched by this investigation or its audit -- a claim that
  specifically needed a `resident_runner` timing/memory measurement to
  re-verify (rather than a `pytest`/`onnxsim.simplify()`-level correctness
  check) was not re-run here.
- The still-open Whisper (#1351) and Qwen-Drive (#1350, merged) work was not
  independently re-verified beyond confirming neither touched `onnxsim/`.
