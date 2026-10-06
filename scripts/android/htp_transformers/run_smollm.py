#!/usr/bin/env python3
"""Run a prepared SmolLM2 variant on the Xiaomi 12S and report WikiText-2 perplexity and per-window latency.

    python run_smollm.py --work DIR --model smollm_adaptive [--mode htp|htp-fallback|cpu] [--fp16] [--threads N] [--repeat K]

Same transport as run_phone.py (phone lock, per-task directory). Perplexity is over the 128-token windows of prep_smollm.py and is
compared with the PyTorch fp32 perplexity of the same windows.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", default="htp", choices=("cpu", "htp", "htp-fallback"))
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--perf", default="burst")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    rp = load(HERE / "run_phone.py", "run_phone")

    model = a.work / "models" / f"{a.model}.onnx"
    data = [a.work / "data" / f for f in ("win.bin", "win.manifest")]
    ctx = (
        ""
        if a.mode == "cpu"
        else f"{a.model}{'_fp16' if a.fp16 else ''}{a.tag}.ctx.onnx"
    )
    extra = "enable_htp_fp16_precision=1" if a.fp16 else ""
    env = f"QNN_PERF={a.perf} ORT_THREADS={a.threads} REPEAT={a.repeat} WARMUP=1 " + (
        f"QNN_EXTRA={extra} " if extra else ""
    )
    out_file = a.work / "data" / f"out_{a.model}{a.tag}.bin"
    run = (
        f"{rp.push_script(Path('/mnt/data/cache/claude-work/htp/qnn_eval'), model, data)}\n"
        f'adb -s {rp.SERIAL} shell "cd {rp.REMOTE} && {env}LD_LIBRARY_PATH={rp.REMOTE} '
        f"ADSP_LIBRARY_PATH='{rp.REMOTE};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
        f'./qnn_eval {model.name} win.manifest {a.mode} out {ctx} 2>&1"\n'
        f"adb -s {rp.SERIAL} pull -q {rp.REMOTE}/out_o0.bin {out_file} 2>/dev/null || true\n"
        f"adb -s {rp.SERIAL} shell rm -f {rp.REMOTE}/out_o0.bin\n"
    )
    info = rp.parse(rp.sh(run))
    info.update(model=a.model, mode=a.mode, fp16=a.fp16, threads=a.threads, perf=a.perf)
    if info["pass"]:
        win = np.load(a.work / "data" / "win_ids.npy")
        logits = np.fromfile(out_file, np.float32).reshape(
            len(win), 1, win.shape[1], -1
        )
        finite = bool(np.isfinite(logits).all())
        ref = np.load(a.work / "data" / "ref_nll.npy")
        if finite:
            nll = np.concatenate(
                [_nll(logits[i], win[i : i + 1]) for i in range(len(win))]
            )
            info.update(
                ppl=float(np.exp(nll.mean())), ref_ppl=float(np.exp(ref.mean()))
            )
        info["finite"] = finite
    print(json.dumps(info))
    if a.json:
        with a.json.open("a") as f:
            f.write(json.dumps(info) + "\n")
    return 0 if info["pass"] else 1


def _nll(logits, ids):
    lg = logits[:, :-1].astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -np.take_along_axis(logp, ids[:, 1:, None], -1)[..., 0]


if __name__ == "__main__":
    sys.exit(main())
