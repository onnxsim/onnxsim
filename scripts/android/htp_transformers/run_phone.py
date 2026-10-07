#!/usr/bin/env python3
"""Run a prepared model on the Xiaomi 12S (Snapdragon 8+ Gen 1, Hexagon V69) over all SST-2 validation sentences and score it.

    python run_phone.py --work DIR --model distilbert_adaptive [--mode htp|htp-fallback|cpu] [--fp16] [--threads N] [--repeat K]

Uses qnn_eval (scripts/android/htp_exploration/qnn_shell/qnn_eval.cpp) from a plain adb shell, with the QNN runtime libraries from
qnn_shell/fetch_libs.sh. Everything that touches the phone runs under ~/.cache/android-phone/phone-run (the shared exclusive lock) in
a per-task directory, as the repo's phone etiquette requires. Prints accuracy, agreement with the host fp32 predictions, logit error
and per-sentence latency.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
QNN = REPO / "scripts" / "android" / "htp_exploration" / "qnn_shell"
PHONE_RUN = Path.home() / ".cache" / "android-phone" / "phone-run"
SERIAL = "239dbd8f"
REMOTE = "/data/local/tmp/htp-transformers"
LIBS = (
    "libonnxruntime.so",
    "libonnxruntime_providers_qnn.so",
    "libQnnHtp.so",
    "libQnnSystem.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpV69Stub.so",
    "libQnnHtpV69Skel.so",
)


def sh(script, timeout=3600):
    """Run a bash script under the phone lock; returns stdout (stderr is shown whenever the run did not pass)."""
    env = {**os.environ, "PHONE_LOCK_OWNER": "htp-transformers"}
    r = subprocess.run(
        [str(PHONE_RUN), "bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if "PASS" not in r.stdout and r.stderr.strip():
        sys.stderr.write(r.stderr[-2000:])
    return r.stdout


def push_script(binary, model, data):
    a = f"adb -s {SERIAL}"
    lines = [
        "set -u",
        f"{a} shell mkdir -p {REMOTE}",
        # push only when the md5 differs: a re-quantized model keeps its size
        'push() { l=$(md5sum "$1" | cut -d" " -f1); r=$('
        + a
        + ' shell "md5sum $2 2>/dev/null" </dev/null | cut -d" " -f1 | tr -d "\\r" || true); [ "$l" = "$r" ] && return 1; '
        + a
        + ' push -q "$1" "$2"; return 0; }',
        f"push {binary} {REMOTE}/qnn_eval || true",
    ]
    lines += [f"push {QNN / 'libs' / lib} {REMOTE}/{lib} || true" for lib in LIBS]
    for f in data:
        lines.append(f"push {f} {REMOTE}/{f.name} || true")
    lines.append(
        f"if push {model} {REMOTE}/{model.name}; then {a} shell 'rm -f {REMOTE}/*.ctx.onnx'; fi"
    )
    return "\n".join(lines)


def parse(out):
    m = re.search(
        r"latency_ms median ([\d.]+) p10 ([\d.]+) p90 ([\d.]+) mean ([\d.]+) min ([\d.]+) max ([\d.]+) \(n=(\d+)\)",
        out,
    )
    info = {"pass": "PASS" in out}
    if m:
        info.update(
            dict(
                zip(
                    ("median", "p10", "p90", "mean", "min", "max", "n"),
                    map(float, m.groups()),
                )
            )
        )
    for key, rx in (
        ("session_create_ms", r"session_create_ms ([\d.]+)"),
        ("compile_ms", r"compile_ms ([\d.]+)"),
    ):
        mm = re.search(rx, out)
        if mm:
            info[key] = float(mm.group(1))
    fail = re.search(r"FAIL (.*)", out)
    if fail:
        info["fail"] = fail.group(1)[:300]
    return info


def reference_logits(work):
    """fp32 logits of the simplified (unrewritten) model on the host, cached; the yardstick for agreement and logit error."""
    cache = work / "data" / "ref_logits.npy"
    if cache.exists():
        return np.load(cache)
    import onnxruntime as ort

    s = ort.InferenceSession(
        str(work / "models" / "distilbert_orig.onnx"),
        providers=["CPUExecutionProvider"],
    )
    ids, mask = (
        np.fromfile(work / "data" / f"val_{k}.bin", np.int64).reshape(-1, 1, 64)
        for k in ("ids", "mask")
    )
    ref = np.concatenate(
        [
            s.run(None, {"input_ids": ids[i], "attention_mask": mask[i]})[0]
            for i in range(len(ids))
        ]
    )
    np.save(cache, ref)
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--work", required=True, type=Path)
    p.add_argument(
        "--model",
        required=True,
        help="model name in <work>/models without .onnx, e.g. distilbert_adaptive",
    )
    p.add_argument(
        "--mode", default="htp-fallback", choices=("cpu", "htp", "htp-fallback")
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="QNN EP option enable_htp_fp16_precision=1 (float32 model computed in fp16 on the HTP)",
    )
    p.add_argument(
        "--perf", default="burst", help="htp_performance_mode (default burst)"
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ORT intra-op threads (CPU mode / CPU fallback)",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="passes over the set for latency statistics",
    )
    p.add_argument(
        "--tag",
        default="",
        help="suffix for the EP-context file when the options change",
    )
    p.add_argument("--json", type=Path, help="append the result as a JSON line here")
    p.add_argument(
        "--no-ctx",
        action="store_true",
        help="skip the EP-context cache (compile every run)",
    )
    a = p.parse_args()

    model = a.work / "models" / f"{a.model}.onnx"
    data = [
        a.work / "data" / f for f in ("val_ids.bin", "val_mask.bin", "val.manifest")
    ]
    binary = Path("/mnt/data/cache/claude-work/htp/qnn_eval")
    labels = np.load(a.work / "data" / "val_labels.npy")
    ref = reference_logits(a.work)

    ctx = (
        ""
        if a.mode == "cpu" or a.no_ctx
        else f"{a.model}{'_fp16' if a.fp16 else ''}{a.tag}.ctx.onnx"
    )
    extra = "enable_htp_fp16_precision=1" if a.fp16 else ""
    env = f"QNN_PERF={a.perf} ORT_THREADS={a.threads} REPEAT={a.repeat} " + (
        f"QNN_EXTRA={extra} " if extra else ""
    )
    run = (
        f"{push_script(binary, model, data)}\n"
        f'adb -s {SERIAL} shell "cd {REMOTE} && {env}LD_LIBRARY_PATH={REMOTE} '
        f"ADSP_LIBRARY_PATH='{REMOTE};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
        f'./qnn_eval {model.name} val.manifest {a.mode} out {ctx} 2>&1"\n'
        f"adb -s {SERIAL} pull -q {REMOTE}/out_o0.bin {a.work / 'data' / ('out_' + a.model + a.tag + '.bin')} 2>/dev/null || true\n"
    )
    out = sh(run)
    info = parse(out)
    info.update(
        {
            "model": a.model,
            "mode": a.mode,
            "fp16": a.fp16,
            "threads": a.threads,
            "perf": a.perf,
        }
    )
    if info["pass"]:
        logits = np.fromfile(
            a.work / "data" / ("out_" + a.model + a.tag + ".bin"), np.float32
        ).reshape(-1, 2)
        finite = bool(np.isfinite(logits).all())
        info.update(
            acc=float((logits.argmax(1) == labels).mean()) if finite else float("nan"),
            agree=float((logits.argmax(1) == ref.argmax(1)).mean())
            if finite
            else float("nan"),
            rel_rmse=float(
                np.sqrt(np.mean((logits - ref) ** 2)) / (ref.max() - ref.min())
            )
            if finite
            else float("nan"),
            finite=finite,
        )
    print(json.dumps(info))
    if a.json:
        with a.json.open("a") as f:
            f.write(json.dumps(info) + "\n")
    if not info["pass"]:
        print(out[-1200:])
    return 0 if info["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
