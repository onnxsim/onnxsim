#!/usr/bin/env python3
"""Does the Snapdragon HTP's fp16 mode overflow where fp16 storage would? Minimal graphs with inputs whose squares exceed 65504.

    python fp16_probe.py --work DIR

Builds four tiny float32 models, runs each on the phone with the QNN EP option enable_htp_fp16_precision=1 (strict HTP, no CPU
fallback) and prints the HTP output next to the exact fp32 result:
  square        y = x*x                      a squared tensor that must be stored: 3.6e5 is not representable in fp16
  mean_square   y = mean(x*x)                the squares are consumed by a reduction
  norm_naive    LayerNorm decomposed as d*d  (npu_rewrite.py --norm-scaling none)
  norm_scaled   the same, rescaled by the row max before squaring (the default)
Inputs are standard normal except one channel at 600 per row (squared deviation ~3e5, as in DistilBERT's residual stream).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def models(npu):
    from onnx import numpy_helper, parser

    def make(body, inits=()):
        m = parser.parse_model(f'<ir_version: 9, opset_import: ["": 17]> {body}')
        m.graph.initializer.extend(inits)
        return m

    norm = make(
        "g (float[1, 4, 768] x) => (float[1, 4, 768] y) { y = LayerNormalization<axis = -1, epsilon = 1e-12>(x, s) }",
        [numpy_helper.from_array(np.ones(768, np.float32), "s")],
    )
    return {
        "square": (
            make("g (float[1, 4, 768] x) => (float[1, 4, 768] y) { y = Mul(x, x) }"),
            (1, 4, 768),
        ),
        "mean_square": (
            make(
                "g (float[1, 4, 768] x) => (float[1, 4, 1] y) { p = Mul(x, x) y = ReduceMean<axes = [-1], keepdims = 1>(p) }"
            ),
            (1, 4, 1),
        ),
        "norm_naive": (npu.rewrite(norm, "none")[0], (1, 4, 768)),
        "norm_scaled": (npu.rewrite(norm, "max")[0], (1, 4, 768)),
    }


def expected(name, x):
    x = x.astype(np.float64)
    if name == "square":
        return x * x
    if name == "mean_square":
        return (x * x).mean(-1, keepdims=True)
    d = x - x.mean(-1, keepdims=True)
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True, type=Path)
    a = p.parse_args()
    import onnx

    run = load(HERE / "run_phone.py", "run_phone")
    npu = load(REPO / "scripts" / "allwinner" / "npu_rewrite.py", "npu_rewrite")
    probe = a.work / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 8
    x = rng.standard_normal((n, 1, 4, 768)).astype(np.float32)
    x[..., 0] = 600.0  # one outlier channel per row
    x.tofile(probe / "x.bin")
    (probe / "x.manifest").write_text(f"x f32 x.bin {n},1,4,768\n")
    print(
        f"input: {n} rows of 768, one channel = 600 -> raw squares up to {600.0**2:.3g} (fp16 max 65504)\n"
    )
    print(f"{'model':13} {'HTP output vs exact fp32':>60}")
    for name, (model, shape) in models(npu).items():
        path = probe / f"{name}.onnx"
        onnx.save(model, path)
        script = (
            run.push_script(
                Path("/mnt/data/cache/claude-work/htp/qnn_eval"),
                path,
                [probe / "x.bin", probe / "x.manifest"],
            )
            + f'\nadb -s {run.SERIAL} shell "cd {run.REMOTE} && QNN_PERF=burst QNN_EXTRA=enable_htp_fp16_precision=1 WARMUP=1 LD_LIBRARY_PATH={run.REMOTE} '
            + f"ADSP_LIBRARY_PATH='{run.REMOTE};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
            + f'./qnn_eval {name}.onnx x.manifest htp probe_{name} 2>&1"\n'
            + f"adb -s {run.SERIAL} pull -q {run.REMOTE}/probe_{name}_o0.bin {probe / (name + '.out')} 2>/dev/null || true\n"
        )
        out = run.sh(script)
        if "PASS" not in out:
            print(f"{name:13} did not run on the HTP: {out.strip()[-200:]}")
            continue
        got = np.fromfile(probe / f"{name}.out", np.float32).reshape((n,) + shape)
        want = expected(name, x.reshape(n, 4, 768)).reshape((n,) + shape)
        bad = ~np.isfinite(got)
        err = np.abs(got[~bad] - want[~bad]).max() if (~bad).any() else float("nan")
        rel = err / (np.abs(want).max() + 1e-30)
        print(
            f"{name:13} non-finite outputs {int(bad.sum()):5d}/{got.size}  max |error| {err:.3g} (relative to the range {rel:.2g});  exact max {np.abs(want).max():.4g}, HTP max {np.nanmax(np.abs(got)):.4g}"
        )


if __name__ == "__main__":
    main()
