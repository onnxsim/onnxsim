"""Hexagon code-generation profiles used by the tinygrad DSP checks."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HexagonTarget:
    """Compiler settings for a concrete Hexagon target."""

    name: str
    mcpu: str
    hvx: str
    # The open-access simulator does not model V65. V68 is the lowest
    # available pipeline proxy; lowering remains V65 through HVX_ARCH.
    simulator_arch: str = "v68"
    hvx_length: str = "128b"

    @property
    def clang_flags(self) -> tuple[str, ...]:
        return (
            "--target=hexagon",
            f"-mcpu={self.mcpu}",
            f"-mhvx={self.hvx}",
            f"-mhvx-length={self.hvx_length}",
        )


# Snapdragon 845 contains Hexagon 685, whose HVX ISA is V65.  Do not use the
# V69/V73 profile used by the newer HMX phone tests for this target.
SNAPDRAGON_845 = HexagonTarget("snapdragon845", "hexagonv65", "v65")
TARGETS = {SNAPDRAGON_845.name: SNAPDRAGON_845}
ALIASES = {
    "845": "snapdragon845",
    "sd845": "snapdragon845",
    "snapdragon-845": "snapdragon845",
}


def get_target(name: str | None = None) -> HexagonTarget:
    requested = (
        name or os.environ.get("TINYGRAD_HEXAGON_TARGET", "snapdragon845")
    ).lower()
    requested = ALIASES.get(requested, requested)
    try:
        return TARGETS[requested]
    except KeyError as exc:
        choices = ", ".join(TARGETS)
        raise ValueError(
            f"unknown Hexagon target {requested!r}; choose from {choices}"
        ) from exc


def configure_tinygrad_environment(target: HexagonTarget | None = None) -> dict[str, str]:
    """Return the target settings consumed by tinygrad's DSP renderer."""

    target = target or get_target()
    return {
        "TINYGRAD_HEXAGON_TARGET": target.name,
        "HVX_ARCH": target.hvx,
        "HEXSIM_ARCH": target.simulator_arch,
    }
