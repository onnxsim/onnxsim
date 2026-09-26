import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "openpilot_dsp" / "tinygrad_runner.py"


def test_tinygrad_runner_help_does_not_require_tinygrad():
    result = subprocess.run(
        [sys.executable, RUNNER, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ONNX input names" in result.stdout
    assert "--target" in result.stdout
