"""Phone-free CI for the tinygrad-based Hexagon stack (scripts/android/tinygrad_hexagon_bridge/).

Two halves, both skipped cleanly when their tools are missing (so the normal test suite is unaffected):

1. tinygrad's own DSP codegen (the onnxsim/tinygrad fork's HVX codegen): plain Tensor code for add, maxpool
   and requantize, run under qemu-hexagon (MOCKDSP=1) and compared byte-exact with NumPy; the vector types
   in the rendered C; a guard against the float-max source blow-up fixed in tinygrad fccd84cfa; and a
   hexagon-sim --timing cycle ratio (vectorized vs NOOPT=1 scalar). Needs a tinygrad checkout with the fork's
   DSP backend on TINYGRAD_PATH (or importable), an LLVM clang with the Hexagon target (HEXAGON_CLANG,
   default clang-19/clang) plus ld.lld, qemu-hexagon, and the Hexagon toolchain (HEXAGON_TOOLS) for
   hexagon-sim.
2. The hand-written HVX kernels (nms/, topk/, proposal_decode/, rpn_fused/, roialign_fast/): their host-C
   checks and freestanding qemu-hexagon builds, on synthetic data with ONNX Runtime (or, for proposal
   decode, the kernel's own reference path) as the reference -- CI never downloads Mask R-CNN.

Run by .github/workflows/hexagon-tinygrad.yml.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BRIDGE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "android"
    / "tinygrad_hexagon_bridge"
)
CI_DIR = BRIDGE / "ci"
TIMEOUT = 1800


# ----------------------------------------------------------------------------------------- tools


def _run(cmd, cwd=None, env=None, timeout=TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ok(res: subprocess.CompletedProcess, what: str) -> str:
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"{what} failed (exit {res.returncode}):\n{out[-4000:]}"
    return out


@functools.lru_cache(maxsize=None)
def hexagon_tools() -> Path | None:
    root = os.environ.get("HEXAGON_TOOLS") or os.environ.get("HEXAGON_TOOLCHAIN")
    if root and (Path(root) / "bin" / "hexagon-sim").exists():
        return Path(root)
    return None


@functools.lru_cache(maxsize=None)
def qemu() -> str | None:
    return shutil.which("qemu-hexagon-static") or shutil.which("qemu-hexagon")


@functools.lru_cache(maxsize=None)
def clang() -> str | None:
    """An LLVM clang that can build freestanding hexagonv65/v73 binaries (the toolchain's own hexagon-clang
    dropped v65, which tinygrad's MOCKDSP and some qemu harnesses target)."""
    if not shutil.which("ld.lld"):
        return None
    cands = [
        os.environ.get("HEXAGON_CLANG"),
        "clang-19",
        "clang-20",
        "clang-18",
        "clang",
    ]
    for c in filter(None, cands):
        path = shutil.which(c)
        if not path:
            continue
        probe = _run(
            [
                path,
                "--target=hexagon",
                "-mcpu=hexagonv65",
                "-mhvx=v65",
                "-x",
                "c",
                "-c",
                "-o",
                os.devnull,
                "-",
            ],
            timeout=60,
        )
        if probe.returncode == 0:
            return path
    return None


@functools.lru_cache(maxsize=None)
def tinygrad_env() -> dict | None:
    env = dict(os.environ)
    tp = os.environ.get("TINYGRAD_PATH")
    if tp:
        env["PYTHONPATH"] = tp + os.pathsep + env.get("PYTHONPATH", "")
    probe = _run(
        [
            sys.executable,
            "-c",
            "import tinygrad.runtime.ops_dsp as d; d.HexagonSimRenderer",
        ],
        env=env,
    )
    return env if probe.returncode == 0 else None


needs_qemu = pytest.mark.skipif(
    qemu() is None or clang() is None or hexagon_tools() is None,
    reason="needs qemu-hexagon, an LLVM clang with Hexagon + ld.lld, and the Hexagon toolchain's C headers",
)
needs_host_cc = pytest.mark.skipif(
    clang() is None, reason="needs an LLVM clang with Hexagon + ld.lld"
)


def hexagon_build(src: Path, out: Path, *flags: str, includes=(), extra=()) -> Path:
    cmd = [
        clang(),
        "--target=hexagon",
        *flags,
        "-O2",
        "-static",
        "-nostdlib",
        "-ffreestanding",
        "-fuse-ld=lld",
    ]
    # the kernels' <string.h>/<math.h> come from the toolchain's own target headers: upstream clang has no
    # Hexagon sysroot and would otherwise fall back to the host's glibc headers (which only happen to work
    # on some machines)
    cmd += ["-isystem", hexagon_tools() / "target" / "hexagon" / "include"]
    cmd += [f"-I{i}" for i in includes] + ["-o", out, src, *extra]
    _ok(_run(cmd), f"hexagon build of {src.name}")
    return out


def host_build(srcs, out: Path, *flags: str, includes=()) -> Path:
    cmd = [
        clang(),
        "-O2",
        *flags,
        *[f"-I{i}" for i in includes],
        "-o",
        out,
        *srcs,
        "-lm",
    ]
    _ok(_run(cmd), f"host build of {out.name}")
    return out


def run_qemu(binary: Path, *args, cwd=None) -> str:
    return _ok(_run([qemu(), binary, *args], cwd=cwd), f"qemu {binary.name}")


# ------------------------------------------------------------------------- 1. tinygrad codegen

needs_tinygrad = pytest.mark.skipif(
    tinygrad_env() is None or qemu() is None or clang() is None,
    reason="needs the onnxsim/tinygrad fork (DSP backend), qemu-hexagon and an LLVM clang with Hexagon",
)


def codegen(mode: str, **extra_env) -> dict:
    env = dict(tinygrad_env())
    env.setdefault("CC", clang() or "clang")
    if hexagon_tools():
        env.setdefault("HEXAGON_TOOLS", str(hexagon_tools()))
        env.setdefault("HEXAGON_TOOLCHAIN", str(hexagon_tools()))
    env.update(extra_env)
    out = _ok(
        _run([sys.executable, CI_DIR / "codegen_check.py", mode], env=env),
        f"codegen_check.py {mode}",
    )
    return json.loads(out.strip().splitlines()[-1])


@needs_tinygrad
def test_codegen_plain_tensor_ops_exact_and_vectorized():
    res = codegen("ops")
    for op in ("add", "maxpool", "requant"):
        assert res[op]["exact"], f"{op}: generated kernel differs from NumPy under qemu"
    # add: full HVX width (the codegen upcasts to several 128-byte vectors per iteration)
    assert res["add"]["vector_bytes"] >= 128, res
    # maxpool: vectorized, but only at the 32-byte channel-block width; full width needs a stride-2
    # deinterleave that isn't built yet (see tinygrad_hexagon_bridge/README.md, tinygrad codegen section)
    assert res["maxpool"]["vector_bytes"] >= 32, res
    # requantize: the 64-bit products stay scalar, but its loads/stores are HVX-width vectors
    assert res["requant"]["vector_bytes"] >= 128, res


@needs_tinygrad
def test_codegen_float_max_chain_renders_linearly():
    """Regression guard for tinygrad fccd84cfa: float MAX used to re-evaluate its operands, doubling the rendered
    source per op in a chain (one test grew past 6 GB and helped OOM-kill a dev box)."""
    sizes = {int(k): v for k, v in codegen("maxchain")["sizes"].items()}
    assert sizes[16] < 3 * sizes[8], (
        sizes
    )  # linear growth is ~2x; the exponential bug was ~256x
    assert sizes[8] < 3 * sizes[4], sizes


@needs_tinygrad
def test_codegen_vrmpy_gemv():
    """The LLM decode GEMV (u8 activations x s8 weights, M=1) reaches the vrmpy TensorCore as vrmpybusv, with one
    128-byte weight vector per vrmpy and a vector accumulator (onnxsim/tinygrad#5; scripts/android/llm_tinygrad)."""
    res = codegen("gemv")
    assert res["exact"], "generated GEMV differs from NumPy under qemu"
    assert res["vrmpybusv"], res
    assert res["weight_vector_load"], res
    assert res["vector_accumulator"], res


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
def test_codegen_hmx_fp16_matmul_bit_exact():
    """The HMX TensorCore (onnxsim/tinygrad#6, HMX=1): an fp16 matmul is bit-exact against the TC's rounding model both
    as MOCKDSP's scalar reference (qemu) and as real HMX on hexagon-sim -mv69 --mhmx 1 (scripts/android/
    tinygrad_hexagon_bridge/tinygrad_codegen/hmx)."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        HMX="1",
        DEV="DSP",
        MOCKDSP="1",
        TC="1",
        HVX_ARCH="v69",
    )
    hmx = CI_DIR.parent / "tinygrad_codegen" / "hmx"
    out = _ok(
        _run([sys.executable, hmx / "hmxsim.py", "64", "64", "64", "--ref"], env=env),
        "hmxsim.py 64 64 64",
    )
    assert "hexagon-sim HMX 0/4096 bit mismatches" in out and "(PASS)" in out, out
    assert "MOCKDSP scalar reference: 0 mismatches" in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
def test_codegen_hmx_int8_matmul_exact():
    """The int8 ":cm" TensorCore (hexagon_hmx_i8): a uint8 x int8 -> int32 matmul with VTCM-resident tiles, the quad /
    :deep weight path (4 N tiles, 2 accumulators per weight load pair) and the exact int32 accumulator read back as four
    byte planes -- exact against numpy int64 both on hexagon-sim --mhmx 1 and as MOCKDSP's scalar reference."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        HMX="1",
        DEV="DSP",
        MOCKDSP="1",
        TC="1",
        HVX_ARCH="v69",
    )
    hmx = CI_DIR.parent / "tinygrad_codegen" / "hmx"
    out = _ok(
        _run(
            [sys.executable, hmx / "hmxsim_i8.py", "128", "256", "256", "--ref"],
            env=env,
        ),
        "hmxsim_i8.py 128 256 256",
    )
    assert "hexagon-sim HMX 0/32768 mismatches" in out and "(PASS)" in out, out
    assert "MOCKDSP scalar reference: 0 mismatches" in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
@pytest.mark.parametrize("flags", [[], ["--relu", "--nobias"]])
def test_codegen_hmx_int8_requant_exact(flags):
    """A QDQ int8 layer with ORT's requantization fused into the HMX kernel: clip(round((acc + b).float() * m) + zy, lo,
    255).cast(uint8) lowers to integer HVX (__hmx_rq4) that emulates ORT's two fp32 roundings -- exact against numpy's
    IEEE fp32, exact .5 ties included, on hexagon-sim and as MOCKDSP's scalar reference."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        HMX="1",
        DEV="DSP",
        MOCKDSP="1",
        TC="1",
        HVX_ARCH="v69",
    )
    hmx = CI_DIR.parent / "tinygrad_codegen" / "hmx"
    out = _ok(
        _run(
            [
                sys.executable,
                hmx / "hmxsim_rq.py",
                "128",
                "256",
                "256",
                *flags,
                "--ref",
            ],
            env=env,
        ),
        "hmxsim_rq.py 128 256 256",
    )
    assert "hexagon-sim HMX 0/32768 mismatches" in out and "(PASS)" in out, out
    assert "MOCKDSP scalar reference: 0 mismatches" in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
@pytest.mark.parametrize(
    "shape",
    [["16", "16", "64", "64"], ["8", "16", "128", "128", "--stride", "2", "--relu"]],
)
def test_codegen_hmx_int8_conv3x3_exact(shape):
    """A QDQ 3x3 conv (stride 1 and 2) in grid form -- the padded NHWC image flattened, taps as flat offsets, two reduce
    loops (dy, dx*C + c) under TC_OPT=1 -- with the requantization fused, exact against ORT's formula on hexagon-sim and
    MOCKDSP. The 128-channel case goes through the four-block activation pack."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        HMX="1",
        DEV="DSP",
        MOCKDSP="1",
        TC="1",
        TC_OPT="1",
        HVX_ARCH="v69",
    )
    hmx = CI_DIR.parent / "tinygrad_codegen" / "hmx"
    out = _ok(
        _run([sys.executable, hmx / "hmxsim_conv.py", *shape, "--ref"], env=env),
        "hmxsim_conv.py",
    )
    assert "mismatches vs ORT's formula" in out and "(PASS)" in out, out
    assert "MOCKDSP scalar reference: 0 mismatches" in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
@pytest.mark.parametrize("flags", [[], ["--ties"]])
def test_codegen_qlinear_add_exact(flags):
    """ORT's QLinearAdd (rne(rb*b + (ra*a + fixed)) in separate fp32 ops) through tinygrad's hmx_qlinear_add custom kernel:
    exact against numpy's fp32 in that order on hexagon-sim and MOCKDSP, also when many lanes are exact .5 ties."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        DEV="DSP",
        MOCKDSP="1",
        HVX_ARCH="v69",
    )
    hmx = CI_DIR.parent / "tinygrad_codegen" / "hmx"
    out = _ok(
        _run(
            [sys.executable, hmx / "hmxsim_add.py", "25088", *flags, "--ref"], env=env
        ),
        "hmxsim_add.py",
    )
    assert "hexagon-sim 0/25088 mismatches" in out and "(PASS)" in out, out
    assert "MOCKDSP scalar reference: 0 mismatches" in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
def test_hand_hmx_kernels_are_tinygrad_oracles():
    """The hand-written HMX kernels (hmx_gemm fp16 GEMM, the QDQ-exact 1x1 / 3x3 convs) now live in the tinygrad fork as test
    oracles (test/external/dsp/hand): each is run next to tinygrad's lowering of the same op on hexagon-sim, bit-exact."""
    env = dict(tinygrad_env())
    env.update(
        CC=clang() or "clang",
        HEXAGON_TOOLS=str(hexagon_tools()),
        HMX="1",
        DEV="DSP",
        MOCKDSP="1",
        TC="1",
        HVX_ARCH="v69",
    )
    root = _run(
        [
            sys.executable,
            "-c",
            "import tinygrad, os; print(os.path.dirname(os.path.dirname(tinygrad.__file__)))",
        ],
        env=env,
    )
    tg = Path(root.stdout.strip())
    if not (tg / "test/external/dsp/hand").is_dir():
        pytest.skip(
            f"the tinygrad at {tg} has no test/external/dsp/hand (not the fork's checkout)"
        )
    out = _ok(
        _run(
            [sys.executable, "-m", "pytest", "-q", "-s", "test/external/dsp/hand"],
            env=env,
            cwd=tg,
        ),
        "tinygrad test/external/dsp/hand",
    )
    assert " passed" in out and "failed" not in out and " skipped" not in out, out


@needs_tinygrad
@pytest.mark.skipif(
    hexagon_tools() is None,
    reason="needs the Hexagon toolchain's hexagon-sim (HEXAGON_TOOLS)",
)
def test_codegen_hexsim_vectorized_add_beats_scalar():
    """hexagon-sim --timing cycles: the default (HVX) codegen vs NOOPT=1 (scalar). Measured ~50x on toolchain
    19.0.04; only a generous ratio is asserted so simulator versions can't make this flaky."""
    vec = codegen("hexsim")["seconds_at_1ghz"]
    scalar = codegen("hexsim", NOOPT="1")["seconds_at_1ghz"]
    assert vec > 0 and scalar > 0
    assert scalar / vec >= 8, (
        f"vectorized {vec:.3g}s vs scalar {scalar:.3g}s (at 1 GHz)"
    )


@pytest.mark.skip(
    reason="reserved: tinygrad qfloat lowering (stage 2 of the HVX codegen) adds its tests here when it lands"
)
def test_codegen_qfloat_float_ops():
    """Slot for the qfloat lowering (float vector math on V69+ HVX, which has no IEEE fp32). Being built on the
    onnxsim/tinygrad fork; its sigmoid / RoiAlign-blend checks go here, in this file's style."""


# ------------------------------------------------------------------ 2. hand-written HVX kernels


@pytest.fixture(scope="module")
def nms_data(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("nms")
    _ok(
        _run(
            [
                sys.executable,
                BRIDGE / "nms" / "gen_nms_stress_data.py",
                out,
                "--calls",
                "12",
                "--seed",
                "1",
            ]
        ),
        "gen_nms_stress_data.py",
    )
    return out


@needs_host_cc
def test_nms_host_check(nms_data, tmp_path):
    exe = host_build(
        [BRIDGE / "nms" / "nms_host_check.c"], tmp_path / "nmshost", "-ffp-contract=off"
    )
    out = _ok(_run([exe, nms_data]), "nms_host_check")
    assert "PASS" in out


@needs_qemu
def test_nms_qemu(nms_data, tmp_path):
    exe = hexagon_build(
        BRIDGE / "nms" / "nms_qemu.c",
        tmp_path / "nmsq",
        "-mcpu=hexagonv65",
        "-ffp-contract=off",
    )
    for group in ("level", "class"):
        run_qemu(exe, f"{nms_data}/{group}")


@needs_host_cc
def test_topk_host_stress(tmp_path):
    exe = host_build([BRIDGE / "topk" / "topk_host_check.c"], tmp_path / "topkhost")
    out = _ok(_run([exe, "120"]), "topk_host_check stress")
    assert "PASS" in out


def _topk_case(rng, n: int, k: int, distinct: int):
    # dequantized-int8-like scores: few distinct values, many ties (ORT: value desc, then lower index)
    x = (rng.integers(-distinct // 2, distinct // 2 + 1, n) * 0.0625).astype(np.float32)
    x[x == 0] = 0.0  # no -0.0: keep the reference's tie order unambiguous
    order = np.lexsort((np.arange(n), -x))[:k]
    return x, x[order], order.astype(np.int64)


@needs_qemu
def test_topk_qemu(tmp_path):
    exe = hexagon_build(
        BRIDGE / "topk" / "topk_qemu.c",
        tmp_path / "topkq",
        "-mcpu=hexagonv73",
        "-mhvx=v73",
        "-mhvx-length=128b",
        extra=[
            CI_DIR / "hexagon_divrt.c"
        ],  # topk_qemu.c has no integer-divide helpers of its own
    )
    rng = np.random.default_rng(7)
    # the real calls' shapes (per-level pre-NMS selects, then the post-NMS ones), with synthetic scores
    for i, (n, k, distinct) in enumerate(
        [
            (10200, 1000, 213),
            (2550, 1000, 51),
            (663, 663, 90),
            (1465, 1000, 120),
            (106, 100, 60),
        ]
    ):
        x, vals, idx = _topk_case(rng, n, k, distinct)
        paths = [tmp_path / f"c{i}_{s}.bin" for s in ("x", "vals", "idx")]
        for arr, p in zip((x, vals, idx), paths):
            arr.tofile(p)
        res = _run([qemu(), exe, *paths, n, k])
        verdict = dict(
            line.split()[0:3:2]
            for line in res.stdout.splitlines()
            if "survivors=" in line
        )
        # vec-reduce_or-mask is kept on purpose as a reproducer of a qemu 8.2 HVX bug (MISMATCH under qemu,
        # EXACT on the phone; tinygrad_hexagon_bridge/README.md, TopK section), so the harness exits 1 by
        # design whenever it trips. The variants the kernel actually ships must be exact.
        for v in ("vec-rot", "vec-reduce_or", "scalar"):
            assert verdict.get(v) == "EXACT", (
                f"call {i} (n={n}, k={k}): {v}\n{res.stdout}{res.stderr}"
            )


def _roialign_model(c, h, w, r, oh, sr, scale):
    from onnx import parser

    return parser.parse_model(
        f"""<ir_version: 7, opset_import: ["" : 12]>
        roialign (float[1,{c},{h},{w}] X, float[{r},4] rois, int64[{r}] bidx) => (float[{r},{c},{oh},{oh}] Y) {{
            Y = RoiAlign <mode = "avg", output_height = {oh}, output_width = {oh}, sampling_ratio = {sr},
                          spatial_scale = {scale!r}> (X, rois, bidx)
        }}"""
    )


@pytest.fixture(scope="module")
def roialign_data(tmp_path_factory):
    ort = pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("roialign")
    rng = np.random.default_rng(3)
    calls = []
    # (H, W, C, R, OH, sr, 1/scale): box-head-like 7x7 and mask-head-like 14x14, C a multiple of 32 (<= 256)
    for i, (h, w, c, r, oh, sr, inv) in enumerate(
        [(24, 30, 64, 13, 7, 2, 4), (12, 15, 256, 5, 14, 2, 8), (7, 9, 32, 9, 7, 2, 16)]
    ):
        scale = 1.0 / inv
        feat = rng.standard_normal((1, c, h, w)).astype(np.float32)
        img_h, img_w = (
            h * inv,
            w * inv,
        )  # boxes in image coordinates, some partly outside
        x1 = rng.uniform(-8, img_w, r)
        y1 = rng.uniform(-8, img_h, r)
        rois = np.stack(
            [
                x1,
                y1,
                x1 + rng.uniform(0, img_w / 2, r),
                y1 + rng.uniform(0, img_h / 2, r),
            ],
            1,
        )
        rois = rois.astype(np.float32)
        sess = ort.InferenceSession(
            _roialign_model(c, h, w, r, oh, sr, scale).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        y = sess.run(None, {"X": feat, "rois": rois, "bidx": np.zeros(r, np.int64)})[0]
        np.ascontiguousarray(feat[0].transpose(1, 2, 0)).tofile(
            out / f"call{i}_feat.bin"
        )
        rois.tofile(out / f"call{i}_rois.bin")
        np.ascontiguousarray(y.transpose(0, 2, 3, 1)).tofile(out / f"call{i}_ref.bin")
        calls.append((h, w, c, r, oh, sr, inv))
    (out / "calls.txt").write_text(
        "".join(
            f"{h} {w} {c} {r} {oh} {oh} {sr} {float(np.float32(1.0 / inv))!r}\n"
            for h, w, c, r, oh, sr, inv in calls
        )
    )
    return out, calls


@needs_host_cc
def test_roialign_host_check(roialign_data, tmp_path):
    data, _ = roialign_data
    exe = host_build(
        [BRIDGE / "roialign_fast" / "roialign_host_check.c"], tmp_path / "roihost"
    )
    out = _ok(_run([exe, data]), "roialign_host_check")
    assert "PASS" in out


@needs_qemu
def test_roialign_qemu(roialign_data, tmp_path):
    data, calls = roialign_data
    # -mhvx=v65: qemu 8.2 can't decode the qfloat HVX ops a v68+ build emits (the real kernel's float path
    # is only checked on the phone)
    exe = hexagon_build(
        BRIDGE / "roialign_fast" / "roialign_qemu.c",
        tmp_path / "roiq",
        "-mcpu=hexagonv65",
        "-mhvx=v65",
        "-mhvx-length=128b",
    )
    for i, (h, w, c, r, oh, sr, inv) in enumerate(calls):
        files = [data / f"call{i}_{s}.bin" for s in ("feat", "rois", "ref")]
        run_qemu(exe, *files, h, w, c, r, oh, oh, sr, inv)


@pytest.fixture(scope="module")
def proposal_decode_data(tmp_path_factory):
    if clang() is None:
        pytest.skip("needs an LLVM clang with Hexagon + ld.lld")
    out = tmp_path_factory.mktemp("pd")
    exe = host_build(
        [CI_DIR / "pd_selfcheck.c"],
        out / "pd_selfcheck",
        "-ffp-contract=off",
        includes=[BRIDGE / "proposal_decode"],
    )
    log = _ok(_run([exe, out]), "pd_selfcheck")
    return out, log


def test_proposal_decode_selfcheck(proposal_decode_data):
    """No captured model in CI: both delta sources, the reference (division) path and both fast paths, and the
    off-grid fallback must agree bit for bit on synthetic grid anchors and random uint8 deltas."""
    _, log = proposal_decode_data
    assert "PASS" in log


@needs_qemu
def test_proposal_decode_qemu(proposal_decode_data, tmp_path):
    """The hexagon build under qemu vs the host reference path's output, per level."""
    data, _ = proposal_decode_data
    exe = hexagon_build(
        BRIDGE / "proposal_decode" / "pd_qemu.c",
        tmp_path / "pdq",
        "-mcpu=hexagonv73",
        "-ffp-contract=off",
    )
    for lvl, line in enumerate((data / "levels.txt").read_text().splitlines()):
        a, k, h, w = line.split()[:4]
        files = [
            data / f"l{lvl}_{s}.bin"
            for s in ("params", "anchors", "idx", "deltas", "nchw_q", "ref")
        ]
        run_qemu(exe, *files, a, k, h, w)


@needs_host_cc
def test_rpn_fused_builds(tmp_path):
    """Build-only: rpn_fused composes the TopK, proposal-decode and NMS kernels, each checked above. Running its
    checks needs ORT captures of every intermediate of the real rest.onnx span (5 levels of model constants
    plus per-image TopK/decode/NMS/merge outputs), which CI doesn't download."""
    host_build(
        [BRIDGE / "rpn_fused" / "rpn_host_check.c"],
        tmp_path / "rpnhost",
        "-ffp-contract=off",
        "-Wno-unused-function",
    )
    if hexagon_tools() is not None:
        hexagon_build(
            BRIDGE / "rpn_fused" / "rpn_qemu.c",
            tmp_path / "rpnq",
            "-mcpu=hexagonv73",
            "-ffp-contract=off",
            "-Wno-unused-function",
        )
