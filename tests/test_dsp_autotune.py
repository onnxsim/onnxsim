import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "android" / "tinygrad_hexagon_bridge" / "tinygrad_codegen"
sys.path.insert(0, str(TARGET))

from dsp_autotune import Candidate, default_candidates  # noqa: E402


def test_dsp_autotune_candidates_cover_beam_and_scalar_baselines():
    candidates = default_candidates()
    assert len(candidates) == 13
    assert candidates[0].name == "beam0-pf1024"
    assert candidates[5].name == "beam1-pf4096"
    assert candidates[-1].name == "noopt"
    assert candidates[0].environment == {"BEAM": "0", "HVX_PREFETCH": "1024"}
    assert Candidate("x", (("BEAM", "2"),)).environment == {"BEAM": "2"}
