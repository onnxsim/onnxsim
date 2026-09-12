#!/usr/bin/env python3
"""Fixes a real cross-vendor test collision on the bare names "models"/"worker".

``scripts/qualcomm``, ``scripts/intel``, ``scripts/amd``, ``scripts/apple``,
and ``scripts/axera`` each carry their own ``models.py`` (and the first four
also carry their own ``worker.py``), and each vendor's test file
(``tests/test_*_compat.py``) does the same dance: prepend its own
``scripts/<vendor>`` dir to ``sys.path``, then ``import models`` (or ``from
worker import check``) by its bare name. That is fine as long as each
vendor's test file runs in its own process -- but the default test suite
(``pytest tests/``) collects *all* of them into one process, and Python
caches imported modules by bare name in ``sys.modules``: whichever vendor's
``models.py`` gets imported *first* (at collection time, even if that
vendor's tests are then skipped -- the module-level ``import models``
statement still runs) stays cached under the name ``"models"`` for the rest
of the process. Every other vendor's later ``import models`` silently reuses
that first one instead of its own.

This went unnoticed because every pre-existing vendor ``models.py`` is a
thin, functionally-identical alias for the same ``scripts/common/
synthetic_models.py`` suite -- getting "the wrong vendor's models.py" changed
nothing observable. It became a real, visible bug the moment
``scripts/axera/models.py`` added something the others don't have
(``axera_npu_compiled_leaf``): whichever vendor test file collected first
poisoned ``sys.modules["models"]``, and axera's own `models.build(...)` and
`worker.check(...)` calls (both reached in-process, not via a subprocess)
silently ran against a different vendor's module instead.

:func:`fresh` forces a real import from a specific directory regardless of
what is already cached under that bare name -- and regardless of ``sys.path``
order. An earlier version of this function re-imported by bare name
(``importlib.import_module(name)``) after evicting a wrong cache entry, which
is not enough on its own once more than one vendor directory needs to be on
``sys.path`` at the same time (e.g. every caller of :func:`fresh` also needs
this module's own directory on ``sys.path`` to reach ``_local_import``
itself): ``import_module`` still resolves the bare name against ``sys.path``
in order and can land back on a *different* directory's same-named file,
independent of which ``directory`` was actually asked for. Loading directly
from ``directory`` by file path removes that ambiguity entirely.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


def ensure_repo_onnxsim() -> None:
    """Make ``onnxsim``'s *pure-Python* submodules resolve to this checkout's
    own ``onnxsim/`` first, without breaking the compiled extension.

    A real, confirmed hazard for any script here run from an isolated ``git
    worktree``: invoked directly (``python3 scripts/axera/foo.py``), a
    script's ``sys.path[0]`` is its own ``scripts/axera`` directory, which
    has no ``onnxsim`` package in it -- so the normal ``sys.path`` search
    (``importlib.machinery.PathFinder``, tried first) finds nothing and
    falls through to the editable install's own finder
    (``sys.meta_path.append``ed, so tried *after* ``PathFinder`` -- checked
    directly in the generated ``__editable___onnxsim_*_finder.py``), which
    unconditionally maps ``onnxsim`` to the path recorded at ``pip install
    -e .`` time -- the main checkout, regardless of which worktree's script
    actually asked. A worktree editing ``onnxsim/*.py`` and "host-verifying"
    the change by running one of these scripts is therefore silently
    exercising the *main checkout's* code, not its own, unless something
    puts this worktree's own repo root ahead of that fallback first.

    **Must merge into the package's own ``__path__``, not replace resolution
    on ``sys.path``.** An earlier version inserted the repo root at
    ``sys.path[0]``, which makes ``importlib.machinery.PathFinder`` resolve
    the top-level ``onnxsim`` package directly from this checkout's source
    tree -- and once a package is found that way, every submodule import
    (``onnxsim.onnxsim_cpp2py_export`` included) searches only *that*
    package's own ``__path__``, never consulting ``sys.meta_path`` again.
    The compiled extension is a build artifact, not a tracked source file --
    it does not live in a plain checkout's ``onnxsim/`` directory the way
    CI's `axera-integration.yml` `pulsar2-compat` job's own comment already
    documents ("the repo root contains the `onnxsim/` source dir (no
    compiled extension), which would shadow the installed wheel"). That job
    deliberately runs from `runner.temp` to avoid exactly this -- and the
    ``sys.path``-replacing version of this function broke that protection
    anyway, since it does not look at ``cwd`` at all (confirmed: PR #1352
    green, then this exact failure on every PR after it,
    ``ModuleNotFoundError: No module named 'onnxsim.onnxsim_cpp2py_export'``
    from `pulsar2-compat`'s `calibration.py` import).

    The fix: import ``onnxsim`` first, however it would normally resolve
    (the editable install's finder in a worktree, the properly-installed
    wheel in CI) -- this is what discovers where the *real* compiled
    extension lives. Then prepend this checkout's own ``onnxsim/`` directory
    to the now-resolved package's ``__path__`` (not to ``sys.path``), so a
    submodule search that reaches this checkout's directory finds its own
    copy first, and one that doesn't (``onnxsim_cpp2py_export`` -- a plain
    checkout never has the compiled extension) falls through to wherever
    ``__path__``'s original entry already pointed, exactly as if this
    function had never run.

    **Known incomplete, confirmed by testing rather than assumed fixed**:
    ``onnxsim/__init__.py`` eagerly imports the large majority of the
    package's own submodules (``graph_grad`` included, transitively, via
    other eagerly-imported modules like ``onnxsim.lora``) as part of running
    ``import onnxsim`` itself -- before this function ever gets a chance to
    touch ``__path__``. Those submodules are already bound in
    ``sys.modules`` by the time the ``__path__`` prepend happens, so a later
    ``import onnxsim.graph_grad`` returns the *already-resolved* module,
    unaffected by this function -- confirmed directly: a worktree-invoked
    script calling this still received the main checkout's
    ``graph_grad.py``, not its own. This function therefore reliably
    prevents the compiled-extension breakage above, and correctly redirects
    any submodule that genuinely isn't already cached by the time it runs,
    but does **not** reliably give a specific, heavily-imported submodule
    (``graph_grad`` chief among them) worktree isolation. For a guaranteed
    fix on one specific submodule, use :func:`fresh` instead -- e.g.
    ``fresh("graph_grad", os.path.join(repo_root, "onnxsim"))`` -- the same
    tool this module already uses for axera-local modules, which sidesteps
    ``sys.modules``/``__path__`` entirely rather than trying to out-race
    ``onnxsim/__init__.py``'s own eager imports.

    Call this before any ``from onnxsim import ...``, as early in the
    script as possible. A bare ``import onnxsim`` already happened by the
    time this returns; that's required, not a side effect to avoid.
    """
    import onnxsim as _onnxsim

    repo_onnxsim_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "onnxsim",
    )
    if repo_onnxsim_dir not in _onnxsim.__path__:
        _onnxsim.__path__.insert(0, repo_onnxsim_dir)


def fresh(name: str, directory: str) -> ModuleType:
    """Import ``<directory>/<name>.py`` as module ``name``, bypassing both
    ``sys.modules`` caching and ``sys.path`` search-order ambiguity -- the
    result is always the file at that exact path, never a same-named module
    some other directory on ``sys.path`` happens to also provide.
    """
    path = os.path.join(directory, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name!r} from {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
