"""scripts/allwinner/compile_nbg.py drives Acuity's `pegasus` to turn an ONNX model into an NBG for the Allwinner NPU.

Acuity is a proprietary toolkit that is not available in CI, so these tests put a fake `pegasus` (and a stub for the
acuitylib-based inputmeta step) on ACUITY_PATH. They check what the script is responsible for -- the exact step sequence and flags
the Allwinner awnpu_model_zoo scripts use, the per-SoC `--optimize` target, calibration data hand-off, the manifest, and the error
paths -- not the NBG Acuity would produce.
"""

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import parser

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "allwinner" / "compile_nbg.py"
)
spec = importlib.util.spec_from_file_location("compile_nbg", SCRIPT)
compile_nbg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compile_nbg)

FAKE_PEGASUS = """#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if os.environ.get("FAKE_FAIL") == args[0]:
    print("E: simulated " + args[0] + " failure")
    sys.exit(3)
if args[0] == "quantize":
    Path(args[args.index("--model-quantize") + 1]).write_text("quant")
if args[0] == "export":
    out = Path(args[args.index("--output-path") + 1])
    nbg = out.parent.with_name(out.parent.name + "_nbg_unify")
    nbg.mkdir(parents=True)
    (nbg / "network_binary.nb").write_bytes(b"VPMN-fake-nbg")
"""

FAKE_INPUTMETA = """#!{python}
import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(["inputmeta"] + sys.argv[2:]) + "\\n")
"""


def _model(body, opset=13):
    return parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')


CONV = """
g (float[1, 3, 8, 8] x) => (float[1, 4, 8, 8] y) <float[4, 3, 3, 3] W = {0.1}> {
  y = Conv<pads = [1, 1, 1, 1]>(x, W)
}"""
DYNAMIC = """
g (float[N, 3, 8, 8] x) => (float[N, 3, 8, 8] y) {
  y = Relu(x)
}"""


def _executable(path, text):
    path.write_text(text.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    acuity = tmp_path / "acuity"
    acuity.mkdir()
    _executable(acuity / "pegasus", FAKE_PEGASUS)
    stub = _executable(tmp_path / "inputmeta_stub", FAKE_INPUTMETA)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("ACUITY_PATH", str(acuity))
    monkeypatch.setenv("VIV_SDK", "/fake/vivante_sdk")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("AW_ACUITY_PYTHON", str(stub))
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    monkeypatch.delenv("AW_NPU_DOCKER_IMAGE", raising=False)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return calls


def _compile(tmp_path, body, *extra):
    src = tmp_path / "m.onnx"
    onnx.save(_model(body), src)
    out, manifest = tmp_path / "m.nb", tmp_path / "m.json"
    rc = compile_nbg.main([str(src), str(out), str(manifest), "--no-simplify", *extra])
    return rc, out, manifest


def _calib(tmp_path, n=3):
    d = tmp_path / "calib"
    d.mkdir()
    for i in range(n):
        np.save(d / f"{i}.npy", np.full((1, 3, 8, 8), i, np.float32))
    return d


def test_float_pipeline_runs_the_zoo_steps_for_a733(tmp_path, toolkit):
    rc, out, manifest = _compile(tmp_path, CONV, "--quant", "float")
    assert rc == 0
    assert out.read_bytes() == b"VPMN-fake-nbg"
    steps = toolkit()
    assert [s[0] if s[0] != "generate" else f"generate {s[1]}" for s in steps] == [
        "import",
        "generate inputmeta",
        "generate postprocess-file",
        "inputmeta",
        "export",
    ]
    assert steps[0][:2] == ["import", "onnx"]
    export = steps[-1]
    assert export[:2] == ["export", "ovxlib"]
    assert "--pack-nbg-unify" in export
    assert export[export.index("--optimize") + 1] == "VIP9000NANODI_PLUS_PID0X1000003B"
    assert export[export.index("--viv-sdk") + 1] == "/fake/vivante_sdk"
    assert export[export.index("--dtype") + 1] == "float"
    assert "--model-quantize" not in export


def test_manifest_describes_target_and_io(tmp_path, toolkit):
    rc, _, manifest = _compile(
        tmp_path,
        CONV,
        "--quant",
        "float",
        "--platform",
        "t527",
        "--compiler-id",
        "acuity-6.6.1",
    )
    assert rc == 0
    m = json.loads(manifest.read_text())
    assert m["target"] == {
        "backend": "viplite",
        "device": "t527",
        "optimize": "VIP9000NANOSI_PLUS_PID0X10000016",
    }
    assert m["artifact"]["format"] == "nbg"
    assert m["compiler"]["id"] == "acuity-6.6.1"
    assert m["io"]["inputs"] == [
        {"name": "x", "dtype": "float32", "shape": [1, 3, 8, 8]}
    ]
    assert m["io"]["outputs"][0]["shape"] == [1, 4, 8, 8]


def test_pcq_quantization_uses_calibration_samples(tmp_path, toolkit):
    rc, _, manifest = _compile(tmp_path, CONV, "--calib-dir", str(_calib(tmp_path)))
    assert rc == 0
    steps = {s[0]: s for s in toolkit() if s[0] in ("quantize", "export", "inputmeta")}
    q = steps["quantize"]
    assert q[q.index("--quantizer") + 1] == "perchannel_symmetric_affine"
    assert q[q.index("--qtype") + 1] == "int8"
    assert q[q.index("--iterations") + 1] == "3"
    assert "--dataset" in steps["inputmeta"]
    export = steps["export"]
    assert export[export.index("--dtype") + 1] == "quantized"
    assert export[export.index("--model-quantize") + 1] == "model_pcq.quantize"
    assert json.loads(manifest.read_text())["quantization"] == {
        "mode": "pcq",
        "calibration_samples": 3,
    }


@pytest.mark.parametrize(
    "quant,quantizer,qtype",
    [
        ("uint8", "asymmetric_affine", "uint8"),
        ("int16", "dynamic_fixed_point", "int16"),
        ("bf16", "qbfloat16", "qbfloat16"),
    ],
)
def test_quantizer_per_mode(tmp_path, toolkit, quant, quantizer, qtype):
    rc, _, _ = _compile(
        tmp_path, CONV, "--quant", quant, "--calib-dir", str(_calib(tmp_path, 2))
    )
    assert rc == 0
    q = next(s for s in toolkit() if s[0] == "quantize")
    assert q[q.index("--quantizer") + 1] == quantizer
    assert q[q.index("--qtype") + 1] == qtype


def test_calib_count_caps_samples(tmp_path, toolkit):
    rc, _, _ = _compile(
        tmp_path, CONV, "--calib-dir", str(_calib(tmp_path, 5)), "--calib-count", "2"
    )
    assert rc == 0
    q = next(s for s in toolkit() if s[0] == "quantize")
    assert q[q.index("--iterations") + 1] == "2"


def test_quantizing_without_calibration_is_an_error(tmp_path, toolkit, capsys):
    rc, out, _ = _compile(tmp_path, CONV)
    assert rc == 1
    assert "--calib-dir" in capsys.readouterr().err
    assert not out.exists()


def test_dynamic_input_shape_needs_an_override(tmp_path, toolkit, capsys):
    rc, _, _ = _compile(tmp_path, DYNAMIC, "--quant", "float")
    assert rc == 1
    assert "--input-shape x:" in capsys.readouterr().err
    rc, out, manifest = _compile(
        tmp_path, DYNAMIC, "--quant", "float", "--input-shape", "x:1,3,8,8"
    )
    assert rc == 0
    assert json.loads(manifest.read_text())["io"]["inputs"][0]["shape"] == [1, 3, 8, 8]


def test_unknown_platform(tmp_path, toolkit, capsys):
    rc, _, _ = _compile(tmp_path, CONV, "--quant", "float", "--platform", "nope")
    assert rc == 1
    assert "unknown --platform" in capsys.readouterr().err


def test_missing_toolkit_environment(tmp_path, toolkit, monkeypatch, capsys):
    monkeypatch.delenv("ACUITY_PATH")
    rc, _, _ = _compile(tmp_path, CONV, "--quant", "float")
    assert rc == 1
    assert "ACUITY_PATH is not set" in capsys.readouterr().err


@pytest.mark.parametrize("failing_step", ["import", "quantize", "export"])
def test_toolkit_failure_surfaces_the_log_tail(
    tmp_path, toolkit, monkeypatch, capsys, failing_step
):
    monkeypatch.setenv("FAKE_FAIL", failing_step)
    rc, out, _ = _compile(tmp_path, CONV, "--calib-dir", str(_calib(tmp_path)))
    assert rc == 1
    err = capsys.readouterr().err
    assert f"simulated {failing_step} failure" in err and "exit 3" in err
    assert not out.exists()


def test_every_platform_has_an_optimize_target():
    # The values come from Allwinner's pegasus_export_ovx_nbg.sh; a typo here makes Acuity target the wrong NPU core.
    assert compile_nbg.PLATFORM_OPTIMIZE["a733"] == "VIP9000NANODI_PLUS_PID0X1000003B"
    assert (
        compile_nbg.PLATFORM_OPTIMIZE["t736"] == compile_nbg.PLATFORM_OPTIMIZE["a733"]
    )
    assert compile_nbg.PLATFORM_OPTIMIZE["v853"] == "VIP9000PICO_PID0XEE"
    assert os.path.exists(SCRIPT.with_name("acuity_inputmeta.py"))
