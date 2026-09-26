import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "openpilot_dsp" / "openpilot_runner.py"


def test_openpilot_runner_help_is_available_without_tinygrad():
    result = subprocess.run(
        [sys.executable, RUNNER, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "openpilot_runner.py" in result.stdout
    assert "--warmup" in result.stdout
