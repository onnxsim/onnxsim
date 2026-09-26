"""Phone-free checks of the HVX multi-scale deformable attention kernel (``scripts/android/msda_hvx``).

* ``msda_ref.msda_reference`` (the kernel's contract) equals a verbatim copy of mmcv's
  ``multi_scale_deformable_attn_pytorch`` on mmcv's own interface, and BEVFormer's glue
  (``bevformer_tiny/msda_hvx/split.msda_fused``) equals the model's own ``msda_rank5`` math for
  its TSA (2-frame queue mean) and SCA (camera visibility average) calls.
* ``msda_kernel.h``'s scalar body (``msda_host_check.c``, host C compiler) matches the torch
  reference on synthetic model-shaped cases -- RT-DETR-r18's decoder (box references, 3 levels),
  both BEVFormer calls, mmcv's plain locations -- with edge cases mixed in: points off the map,
  coordinates on pixel centers / borders, invisible maps, a query no map sees; each with fp32 and
  with uint8 value maps (one scale / zero point per tensor, as the HTP emits).
* The HVX qf32 body (``msda_sim.c``) on ``hexagon-sim`` for the same cases, only when
  ``HEXAGON_TOOLS`` points at a Hexagon toolchain (qemu can't decode HVX float).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ANDROID = Path(__file__).resolve().parents[1] / "scripts" / "android"
CORE = ANDROID / "msda_hvx"
BEV = ANDROID / "vision_models" / "bevformer_tiny"

# The hand-written kernel headers live in the onnxsim/tinygrad fork and are linked into scripts/android/
# by scripts/android/fetch_hand_kernels.sh, which tests/conftest.py runs at collection time. That script
# is not present in a built wheel (scripts/ is not shipped), so under cibuildwheel -- which runs
# `pytest {project}/tests` against the installed tree -- the header cannot be materialised. Skip rather
# than fail there: the driver would not compile, and a missing header is a packaging artefact, not a
# kernel regression.
needs_hand_kernels = pytest.mark.skipif(
    not (CORE / "msda_shape.h").exists(),
    reason="the hand MSDA kernel headers are not linked in (run scripts/android/fetch_hand_kernels.sh)",
)

for _p in (ANDROID, BEV, BEV / "msda_hvx", CORE):
    sys.path.insert(0, str(_p))

import model as M  # noqa: E402
import msda_ref  # noqa: E402
import split  # noqa: E402

# Don't leak these generic module names to the rest of the session: other tests import their own
# ``model`` (e.g. tests/test_nanochat.py's ``from model import GPT``). ``M``/``msda_ref``/``split``
# keep their references; ANDROID stays on the path for the lazy ``hexagon_sim_harness`` import.
for _name in ("model", "split", "msda_ref"):
    sys.modules.pop(_name, None)
for _p in (BEV, BEV / "msda_hvx", CORE):
    sys.path.remove(str(_p))

KINDS = ["rtdetr_decoder", "bevformer_tsa", "bevformer_sca", "loc_small"]


def test_reference_matches_mmcv():
    value, levels, loc, attw, _, _, _ = msda_ref.synthetic("loc_small")
    m = loc.shape[1]
    ref = msda_ref.mmcv_msda_pytorch(
        value.reshape(1, -1, m, value.shape[-1] // m),
        levels,
        loc[:, :, 0][None],
        attw[:, :, 0][None],
    )[0]
    got = msda_ref.msda_reference(value, levels, loc, attw)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_bevformer_glue_matches_model_math():
    # TSA: 2-frame queue, per-frame offsets, mean
    g = torch.Generator().manual_seed(0)
    q, hw = 64, (50, 50)
    value = torch.randn(2, hw[0] * hw[1], 256, generator=g)
    ref = torch.rand(2, q, 1, 2, generator=g)
    off = torch.randn(q, 8, 2, 4, 2, generator=g) * 2
    attw = torch.softmax(torch.randn(q, 8, 2, 4, generator=g), -1)
    loc = ref.reshape(2, q, 1, 1, 2) + off.permute(2, 0, 1, 3, 4) / torch.tensor(
        [50.0, 50.0]
    )
    want = M.msda_rank5(
        value.reshape(2, -1, 8, 32), hw, loc, attw.permute(2, 0, 1, 3)
    ).mean(0)
    torch.testing.assert_close(
        split.msda_fused(value, hw, ref, off, attw), want, rtol=1e-5, atol=1e-5
    )
    # SCA: 6 cameras, shared offsets, point p on pillar anchor p % 4, visibility average
    hw = (15, 25)
    value = torch.randn(6, hw[0] * hw[1], 256, generator=g)
    ref = torch.rand(6, q, 4, 2, generator=g) * 1.4 - 0.2
    off = torch.randn(q, 8, 1, 8, 2, generator=g) * 2
    attw = torch.softmax(torch.randn(q, 8, 1, 8, generator=g), -1)
    vis = (torch.rand(6, q, generator=g) < 0.3).to(torch.uint8)
    loc = ref[:, :, torch.arange(8) % 4][:, :, None] + off[:, :, 0][
        None
    ] / torch.tensor([25.0, 15.0])
    per_cam = M.msda_rank5(
        value.reshape(6, -1, 8, 32), hw, loc, attw[:, :, 0][None].expand(6, -1, -1, -1)
    )
    v = vis.to(value.dtype)[..., None]
    want = (per_cam * v).sum(0) / v.sum(0).clamp(min=1.0)
    torch.testing.assert_close(
        split.msda_fused(value, hw, ref, off, attw, vis), want, rtol=1e-5, atol=1e-5
    )


def _cases(tmp_path: Path) -> list[str]:
    out = []
    for kind in KINDS:
        value, levels, loc, attw, mode, ref, vis = msda_ref.synthetic(kind)
        y = msda_ref.msda_reference(value, levels, loc, attw, mode, ref, vis)
        msda_ref.save_case(tmp_path / kind, value, levels, loc, attw, y, mode, ref, vis)
        vq = msda_ref.quantize(value)
        yq = msda_ref.msda_reference(
            msda_ref.dequantize(*vq), levels, loc, attw, mode, ref, vis
        )
        msda_ref.save_case(
            tmp_path / f"{kind}_u8", value, levels, loc, attw, yq, mode, ref, vis, vq=vq
        )
        out += [str(tmp_path / kind), str(tmp_path / f"{kind}_u8")]
    return out


def _plain_env():
    # The sanitizer CI job runs pytest with LD_PRELOAD=libasan/LSan; a host `cc` (and the
    # checker it builds) inheriting that exits non-zero on LeakSanitizer's own reports.
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_PRELOAD", "LSAN_OPTIONS", "ASAN_OPTIONS")
    }


@needs_hand_kernels
def test_scalar_body_on_host(tmp_path):
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        pytest.skip("no host C compiler")
    exe = tmp_path / "msda_host_check"
    subprocess.run(
        [cc, "-O2", "-o", str(exe), str(CORE / "msda_host_check.c"), "-lm"],
        check=True,
        env=_plain_env(),
    )
    out = subprocess.run(
        [str(exe), *_cases(tmp_path)], capture_output=True, text=True, env=_plain_env()
    )
    assert out.returncode == 0 and out.stdout.rstrip().endswith("PASS"), (
        out.stdout + out.stderr
    )


@needs_hand_kernels
def test_hvx_body_on_hexagon_sim(tmp_path):
    import hexagon_sim_harness as harness

    tools = harness.tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    elf = tmp_path / "msda_sim.elf"
    clang = str(tools / "bin" / "hexagon-clang")
    subprocess.run(
        [
            clang,
            "-mv69",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            str(CORE / "msda_sim.c"),
            "-o",
            str(elf),
            "-lm",
            "-lhexagon",
        ],
        check=True,
    )
    env = dict(os.environ)
    harness._ensure_ncurses5(env, tools / "bin" / "hexagon-sim", tmp_path)
    for case in _cases(tmp_path):
        cmd = [
            str(tools / "bin" / "hexagon-sim"),
            "-mv69",
            "--simulated_returnval",
            str(elf),
            "--",
            case,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        assert (
            out.returncode == 0 and "body: hvx" in out.stdout and "\nPASS" in out.stdout
        ), out.stdout[-2000:] + out.stderr[-2000:]
