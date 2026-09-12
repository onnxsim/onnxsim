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
    """Make ``import onnxsim`` resolve to *this checkout's own* ``onnxsim``,
    not whatever ``onnxsim`` happens to be editable-installed globally.

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
    puts this worktree's own repo root ahead of that fallback on
    ``sys.path`` first -- which is exactly what this function does.

    Call this before any ``import onnxsim`` / ``from onnxsim import ...``,
    as early in the script as possible.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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
