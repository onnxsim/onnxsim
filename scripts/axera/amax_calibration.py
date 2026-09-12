#!/usr/bin/env python3
"""Choosing multi-phase calibration-swap targets from a real trajectory,
instead of a hand-picked round-number ratio.

Every multi-phase calibration-swap result in this project so far (PRs #1355,
#1356, #1359) picked its phases' calibration magnitudes by hand -- a 100x or
10,000x jump chosen because it matched a round number, not because it was
derived from how a real gradient/weight trajectory actually moves.
`docs/axera-on-device-training-handoff.md`'s own "what this doesn't prove"
section names the gap directly: *"How many phases a genuinely long run needs,
and where to place the boundaries, is open... a real schedule would likely
want smaller, more numerous steps, chosen from the actual gradient-decay
curve of a real training run rather than picked by hand."*

This module is that: an *algorithm*, not a mechanism. NVIDIA TransformerEngine
(github.com/NVIDIA/TransformerEngine) tracks a rolling history of each
tensor's per-iteration absolute-max value ("amax history") and derives its
next FP8 scale from a statistic over that window (the max of the recent
history, in TE's default "delayed scaling" recipe) rather than the current
iteration's raw value alone -- robust to one noisy/atypical step, and cheap
to compute. **The mechanism TE built this for -- a runtime-adjustable scale
with no recompile -- is confirmed not portable here** (see
`docs/transformerengine-low-precision-survey.md`): Pulsar2 bakes scale/
zero-point into the compiled binary at build time, full stop. What *is*
portable is the algorithm for choosing a scale target, applied to picking
each multi-phase-swap phase's calibration data instead of a per-iteration
runtime scale.

Feed this a real per-step trajectory of a tensor's values (from a host-side
float reference training loop, e.g. `onnxruntime` run repeatedly the way
`docs/axera-on-device-training-handoff.md`'s Whisper section already did to
find the 7e-5..1.3e-4 gradient-absmax trajectory by hand) and it derives:

* :func:`next_phase_calibration_target` -- the calibration magnitude the
  *next* phase should be built for, from a rolling amax statistic over the
  most recent window rather than the single latest value or a hand-picked
  ratio.
* :func:`phase_boundaries` -- where, across a longer trajectory, the amax has
  drifted far enough from the currently-active phase's own calibration target
  that a swap would plausibly be needed -- a real, data-derived schedule
  instead of guessing "phase N should end around step X."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class AmaxHistory:
    """A rolling window of a tensor's per-step absolute-max value.

    Mirrors TransformerEngine's own `amax_history` buffer (one row per
    tracked tensor, one column per recent iteration) for a single tensor --
    this project has one tensor of interest at a time (whichever trainable
    weight's own gradient/update the current phase schedule is built around),
    not TE's whole-model batch of them, so a flat rolling window is enough.
    """

    window: int = 16
    _values: List[float] = field(default_factory=list)

    def push(self, tensor: np.ndarray) -> float:
        """Record one step's tensor and return its own amax."""
        amax = float(np.abs(tensor).max()) if tensor.size else 0.0
        self._values.append(amax)
        if len(self._values) > self.window:
            self._values.pop(0)
        return amax

    def amax(self) -> float:
        """The rolling-window statistic TE's default "delayed scaling"
        recipe uses: the max over the window, not the latest value alone --
        robust to one atypically-small step making a phase look "done"
        early, or one atypically-large step (e.g. a fresh-batch outlier)
        making it look like it needs to end sooner than it really does."""
        if not self._values:
            return 0.0
        return max(self._values)

    def __len__(self) -> int:
        return len(self._values)


def next_phase_calibration_target(
    trajectory: Sequence[np.ndarray], window: int = 16, margin: float = 2.0
) -> float:
    """The calibration magnitude to build the *next* multi-phase-swap phase
    for, from a real per-step trajectory -- not a hand-picked ratio.

    :param trajectory: real per-step snapshots of the tensor whose scale the
            next phase needs to cover (e.g. the trainable weight's own
            output-state tensor, since that's what this project's in-graph
            SGD update pipeline actually quantises -- see
            `docs/transformerengine-low-precision-survey.md`'s finding on
            why that specific choice, not a separately-output gradient
            tensor, is what makes Whisper's case a catastrophic-cancellation
            problem rather than an ordinary underflow one).
    :param window: how many of the most recent steps to track -- TE's own
            default is int8/fp8-recipe-dependent (commonly 16-1024
            iterations in real training runs); this project's phases are
            measured in the low thousands of steps at most before the
            existing ceiling bites, so a small window is the right default
            here, not TE's own.
    :param margin: safety factor over the rolling amax, matching the spirit
            of TE's own headroom over its tracked history (never calibrate
            *exactly* to the tightest value seen, since the next phase's own
            steps will see values this window hasn't yet).

    Returns the rolling-window amax times ``margin`` -- feed this as the
    magnitude passed to `make_training_calib.make_work_dir`'s existing
    `real_data=`/`weight_scale=` calibration-construction, in place of a
    hand-picked round number.
    """
    hist = AmaxHistory(window=window)
    for step in trajectory:
        hist.push(np.asarray(step))
    return hist.amax() * margin


def phase_boundaries(
    trajectory: Sequence[np.ndarray],
    initial_target: float,
    window: int = 16,
    margin: float = 2.0,
    drift_ratio: float = 4.0,
) -> List[int]:
    """Where a longer real trajectory drifts far enough from the
    currently-active phase's own calibration target that a swap would
    plausibly be needed -- a real, data-derived schedule, answering
    `docs/axera-on-device-training-handoff.md`'s own open question ("how
    many phases a genuinely long run needs, and where to place the
    boundaries") from real trajectory data rather than a guess.

    A phase's own calibration covers roughly `[target / K, target]` for some
    hardware-specific dynamic-range factor `K` this function does not know
    (it depends on the tensor's own quantised bit width and the *other*
    values sharing that tensor's range -- not something derivable from the
    trajectory alone). What the trajectory *does* tell you: how many
    steps until the rolling amax has moved by `drift_ratio` from the
    boundary's own starting point, which is the same "has this shrunk enough
    that the calibrated range is now badly mismatched" question the
    gradient-death heuristics elsewhere in this project (`finetune.LossScaler`'s
    `zero_fraction`, `mp_calib_swap_auto_runner.c`'s live death-detection)
    answer empirically on real hardware. This function answers it in advance,
    from a host-side float reference, so a phase schedule can be planned
    before ever compiling anything -- a real, complementary planning tool.

    :param initial_target: the first phase's own calibration target (e.g.
            from :func:`next_phase_calibration_target` over the trajectory's
            own first `window` steps, or a value already known from an
            existing compile).
    :param drift_ratio: how far the rolling amax must fall (or rise) from
            the currently-active phase's own target before a boundary is
            recorded -- matches this project's own measured multi-phase
            ratios (PR #1355/#1356 used 100x/10,000x jumps; a real schedule
            derived here will typically want several smaller boundaries
            instead of one large jump, hence a much smaller default here
            than either of those hand-picked values).

    Returns a list of step indices (into `trajectory`) where a phase
    boundary is placed -- empty if the trajectory never drifts far enough
    from `initial_target` to need one.
    """
    hist = AmaxHistory(window=window)
    boundaries: List[int] = []
    active_target = initial_target
    for i, step in enumerate(trajectory):
        hist.push(np.asarray(step))
        if len(hist) < window:
            continue
        current = hist.amax() * margin
        if active_target <= 0:
            continue
        ratio = max(current / active_target, active_target / current)
        if ratio >= drift_ratio:
            boundaries.append(i)
            active_target = current
    return boundaries
