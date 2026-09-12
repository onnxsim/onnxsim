"""`scripts/axera/amax_calibration.py`: choosing multi-phase calibration-swap
targets from a real trajectory (a TransformerEngine-style rolling amax
history) instead of a hand-picked round-number ratio.

No hardware, no ONNX graph construction -- this is pure host-side numpy
logic over a synthetic trajectory standing in for a real float-reference
training run.
"""

import os
import sys

import numpy as np

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import amax_calibration as ac  # noqa: E402


def test_amax_history_tracks_the_rolling_max_not_the_latest_value():
    """One atypically-small step should not make the tracked amax collapse
    early -- the whole point of using max-over-window rather than the
    latest value, per TE's own "delayed scaling" design."""
    hist = ac.AmaxHistory(window=4)
    for v in [1.0, 1.0, 1.0, 0.001]:
        hist.push(np.array([v]))
    assert hist.amax() == 1.0

    # once the large values fall out of the window, the tracked amax does
    # follow the real decay -- it is a rolling statistic, not a monotonic
    # high-water mark
    hist.push(np.array([0.001]))
    hist.push(np.array([0.001]))
    hist.push(np.array([0.001]))
    assert hist.amax() == 0.001


def test_next_phase_target_uses_rolling_max_with_margin():
    rng = np.random.default_rng(0)
    trajectory = [rng.normal(scale=0.01, size=4) for _ in range(20)]
    trajectory[10] = np.array([5.0, 0.0, 0.0, 0.0])  # one outlier step

    target = ac.next_phase_calibration_target(trajectory, window=16, margin=2.0)
    # the window at the end of the trajectory has already rolled past the
    # outlier (window=16 over 20 steps means steps 0..3 are out of range,
    # the outlier at index 10 is still inside it) -- confirm it is picked up
    assert target == 5.0 * 2.0


def test_next_phase_target_ignores_an_outlier_once_it_rolls_out_of_window():
    rng = np.random.default_rng(1)
    trajectory = [rng.normal(scale=0.01, size=4) for _ in range(5)]
    trajectory[0] = np.array([5.0, 0.0, 0.0, 0.0])
    trajectory += [rng.normal(scale=0.01, size=4) for _ in range(20)]

    target = ac.next_phase_calibration_target(trajectory, window=8, margin=1.0)
    assert target < 1.0  # the early outlier has long since rolled out


def test_phase_boundaries_finds_a_real_decay_transition():
    """A trajectory that genuinely shrinks by 100x partway through should
    produce a boundary near the transition, the same shape of event the
    real Whisper/multi-phase-probe trajectories in this project's own
    handoff doc show."""
    rng = np.random.default_rng(2)
    early = [rng.normal(scale=0.05, size=4) for _ in range(30)]
    late = [rng.normal(scale=0.0005, size=4) for _ in range(30)]
    trajectory = early + late

    initial_target = ac.next_phase_calibration_target(
        trajectory[:16], window=16, margin=2.0
    )
    boundaries = ac.phase_boundaries(
        trajectory, initial_target, window=8, margin=2.0, drift_ratio=4.0
    )
    assert boundaries, "expected at least one boundary for a real 100x decay"
    # the transition happens at index 30; a window=8 rolling statistic
    # detects it once the window is mostly past the transition, not exactly
    # at it -- assert it lands in a sensible neighbourhood, not exactly 30
    assert 28 <= boundaries[0] <= 45


def test_phase_boundaries_empty_for_a_flat_trajectory():
    """Whisper's own real gradient trajectory (docs/axera-on-device-training-
    handoff.md: "stays essentially flat around 7e-5 to 1.3e-4 throughout")
    should not produce spurious boundaries -- confirming this tool would
    have correctly told that investigation "one phase is enough, the
    problem is not a decaying-scale one" rather than recommending a
    multi-phase schedule that would not have helped."""
    rng = np.random.default_rng(3)
    trajectory = [1e-4 + rng.normal(scale=1e-6, size=4) for _ in range(50)]
    initial_target = ac.next_phase_calibration_target(
        trajectory[:16], window=16, margin=2.0
    )
    boundaries = ac.phase_boundaries(
        trajectory, initial_target, window=8, margin=2.0, drift_ratio=4.0
    )
    assert boundaries == []
