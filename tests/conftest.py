"""Pytest configuration for the test suite.

The ``--durations=10`` flag (set as a default in ``pyproject.toml``) makes pytest
print the slowest tests to the terminal. When the suite runs on GitHub Actions we
also mirror that list into the job's step summary, so the timings show up on the
run's summary page instead of being buried in the raw logs.

This file lives in ``tests/`` rather than the repo root on purpose: pytest's
default ``prepend`` import mode inserts a conftest's directory onto ``sys.path``.
A root conftest would put the repo root there, so ``import onnxsim`` would resolve
to the source tree (which has no compiled ``onnxsim_cpp2py_export`` extension)
and shadow the installed wheel during ``cibuildwheel`` tests. Keeping it under
``tests/`` only adds ``tests/`` to the path, which is already there for
collection, so the installed package is imported unchanged.
"""

import os
import pathlib
import re
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FETCH = _ROOT / "scripts" / "android" / "fetch_hand_kernels.sh"

# The hand-written Hexagon kernel headers (HMX block/gemm/qconv, the ResNet runner, MSDA, the RPN family's
# NMS/proposal-decode/TopK, RoiAlign fp32+u8, the FPN layout, and the HMX runtime) live in the onnxsim/tinygrad
# fork's test/external/dsp/hand/, where they are the oracle that tinygrad's generated kernels are checked
# against. scripts/android/ links them in rather than keeping a second copy -- see
# scripts/android/fetch_hand_kernels.sh.
#
# This is done here, at collection time, rather than in a workflow step, because the tests that compile these
# drivers (test_msda_hvx.py, test_hexagon_tinygrad.py) also run in the wheel jobs, which know nothing about
# the hexagon setup. Doing it only in hexagon-tinygrad.yml left them failing with, for example,
# "msda_io.h:15:10: fatal error: 'msda_shape.h' file not found".
_ANDROID = _ROOT / "scripts" / "android"
if not (_ANDROID / "msda_hvx" / "msda_shape.h").exists() and _FETCH.exists():
    try:
        subprocess.run([str(_FETCH)], check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        print(
            f"warning: could not link the hand kernels ({e}); tests that #include one will fail to compile",
            file=sys.stderr,
        )

# Filename patterns for the "axera" marker: Axera Pulsar2/AXCL NPU backend
# tests. test_voyager_sdk_patterns.py doesn't follow the test_axera_*/
# test_axelera_* prefix but covers the same Axelera Voyager SDK surface as
# test_axelera_voyager_*.py, so it's matched by substring instead of prefix.
_AXERA_PREFIX_RE = re.compile(r"^test_(axera|axelera|pulsar2)_")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by filename so CI can select/deselect whole categories.

    Registered in pyproject.toml's ``markers`` list. There's no practical way
    to hand-annotate every quantization/pruning/Axera test file (hundreds of
    them, across many contributors and algorithms), and their names already
    encode the category reliably, so this infers the marker from the
    filename instead of requiring every test to carry an explicit
    ``@pytest.mark.*`` decorator.

    .github/workflows/build-and-test.yml uses ``-m "not axera and not
    quantization and not pruning"`` to skip these on every build_wheels
    leg and the Windows cross-test job -- they exercise onnxsim's own
    algorithm/format logic, not OS/arch/interpreter-dependent behavior, so
    running them on every one of those legs just adds CI time without
    adding coverage. They instead run exactly once, in that workflow's
    dedicated ``test_axera_quantization_pruning`` job.
    """
    for item in items:
        name = os.path.basename(str(item.fspath)).lower()
        if _AXERA_PREFIX_RE.match(name) or "voyager" in name:
            item.add_marker("axera")
        if "quant" in name or "gguf" in name:
            item.add_marker("quantization")
        if "prun" in name:
            item.add_marker("pruning")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Append the slowest-test table to the GitHub Actions step summary.

    Does nothing outside CI (when ``GITHUB_STEP_SUMMARY`` is unset) or when
    ``--durations`` is disabled, so local runs are unaffected.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    # Under pytest-xdist this hook also fires on each worker, which only sees its
    # own shard of the tests. Write once, from the controller, using the fully
    # aggregated stats (workers set ``config.workerinput``; the controller never
    # does).
    if hasattr(config, "workerinput"):
        return

    durations = config.getoption("durations", default=None)
    if not durations:
        return

    durations_min = config.getoption("durations_min", default=None)
    if durations_min is None:
        # Matches pytest's own default threshold for the terminal report.
        durations_min = 0.005

    # Gather every phase report (setup/call/teardown) that carries a duration,
    # mirroring how pytest itself builds the "slowest durations" list.
    reports = [
        rep
        for replist in terminalreporter.stats.values()
        for rep in replist
        if hasattr(rep, "duration") and getattr(rep, "when", None) is not None
    ]
    if not reports:
        return

    reports.sort(key=lambda rep: rep.duration, reverse=True)
    if durations > 0:
        reports = reports[:durations]
    reports = [rep for rep in reports if rep.duration >= durations_min]
    if not reports:
        return

    lines = [
        f"### ⏱️ Slowest {len(reports)} test durations",
        "",
        "| Duration (s) | Phase | Test |",
        "| ---: | :--- | :--- |",
    ]
    for rep in reports:
        lines.append(f"| {rep.duration:.2f} | {rep.when} | `{rep.nodeid}` |")
    lines.append("")

    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
