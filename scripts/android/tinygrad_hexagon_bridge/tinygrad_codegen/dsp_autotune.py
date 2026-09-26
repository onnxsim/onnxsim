#!/usr/bin/env python3
"""Search tinygrad DSP codegen settings for the Snapdragon 845 V65 profile.

Each candidate runs in a fresh subprocess. Correctness is checked with the
V65-compatible QEMU path; timing uses Hexagon-sim's V68 pipeline proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hexagon_target import configure_tinygrad_environment, get_target

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "ci" / "codegen_check.py"


@dataclass(frozen=True)
class Candidate:
    name: str
    env: tuple[tuple[str, str], ...]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.env)


def default_candidates() -> tuple[Candidate, ...]:
    """Search loop quality, prefetch distance, plus the scalar baseline."""

    return tuple(
        Candidate(
            f"beam{beam}-pf{prefetch}",
            (("BEAM", str(beam)), ("HVX_PREFETCH", str(prefetch))),
        )
        for beam in (0, 1, 2, 4)
        for prefetch in (1024, 2048, 4096)
    ) + (Candidate("noopt", (("NOOPT", "1"),)),)


def _run(mode: str, target: str, candidate: Candidate) -> dict:
    env = dict(os.environ)
    env.update(configure_tinygrad_environment(get_target(target)))
    env.update(candidate.environment)
    result = subprocess.run(
        [sys.executable, CHECK, mode],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode:
        output = result.stdout + result.stderr
        raise RuntimeError(f"{candidate.name} failed:\n{output[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def search(
    mode: str = "hexsim",
    target: str = "snapdragon845",
    candidates: tuple[Candidate, ...] | None = None,
    cache_path: str | Path | None = None,
) -> dict:
    """Measure candidates and return the fastest correct one."""

    candidates = candidates or default_candidates()
    cache_file = Path(cache_path) if cache_path else None
    cache = (
        json.loads(cache_file.read_text())
        if cache_file and cache_file.exists()
        else {}
    )
    key = f"{target}:{mode}:" + ",".join(c.name for c in candidates)
    rows = cache.get(key, {})
    for candidate in candidates:
        if candidate.name in rows:
            continue
        correctness = _run("ops", target, candidate)
        exact = all(
            correctness[name]["exact"] for name in ("add", "maxpool", "requant")
        )
        if not exact:
            rows[candidate.name] = {"correct": False, "correctness": correctness}
            continue
        timing = _run(mode, target, candidate)
        rows[candidate.name] = {
            "correct": True,
            "seconds_at_1ghz": timing["seconds_at_1ghz"],
            "env": candidate.environment,
        }
        print(candidate.name, rows[candidate.name], flush=True)
    correct = {name: row for name, row in rows.items() if row.get("correct")}
    if not correct:
        raise RuntimeError("no correct DSP codegen candidate")
    best = min(correct, key=lambda name: correct[name]["seconds_at_1ghz"])
    result = {"target": target, "mode": mode, "best": best, "candidates": rows}
    if cache_file:
        cache[key] = rows
        cache_file.write_text(json.dumps(cache, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="snapdragon845")
    parser.add_argument("--mode", default="hexsim", choices=["hexsim"])
    parser.add_argument("--cache", default="v65-tinygrad-tuning.json")
    args = parser.parse_args()
    print(json.dumps(search(args.mode, args.target, cache_path=args.cache), indent=2))


if __name__ == "__main__":
    main()
