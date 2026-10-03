"""max relative difference vs the CPU EP for the m1c512+s2 setting (and for the stock path)."""
import json, subprocess, os, sys, glob, shutil
import numpy as np
from tune import D, MODELS
from validate import configs

for model in sys.argv[1:]:
    cfg = configs(model)["m1c512+s2"]
    path, extra = MODELS[model]
    cmd = (f"cd {D} && export LD_LIBRARY_PATH=. && rm -rf dcpu dtd dbase && mkdir dcpu dtd dbase && "
           f"./bench {path} cpu 1 1 threads=4 dump=dcpu {extra} >/dev/null 2>&1; "
           f"ORT_WEBGPU_CONV_TEXDIRECT=0 ./bench {path} webgpu 1 1 dump=dbase {extra} >/dev/null 2>&1; "
           f"ORT_WEBGPU_CONV_TEXDIRECT={cfg['mode']} ORT_WEBGPU_TEXDIRECT_MAXC={cfg['maxc']} ORT_WEBGPU_TEXDIRECT_TABLE='{cfg['table']}' ./bench {path} webgpu 1 1 dump=dtd {extra} >/dev/null 2>&1")
    tmp = f"acc_{model}"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    open(f"{tmp}/job.sh", "w").write(cmd + "\n")
    subprocess.run(["bash", "-c", f"~/.cache/android-phone/phone-run bash -c 'adb push {tmp}/job.sh {D}/acc.sh >/dev/null; adb shell \"sh {D}/acc.sh\"; for d in dcpu dtd dbase; do adb pull {D}/$d {tmp}/$d >/dev/null; done'"], capture_output=True, text=True)
    for tag in ("dbase", "dtd"):
        w = 0.0
        for f in sorted(glob.glob(f"{tmp}/dcpu/*.bin")):
            a = np.fromfile(f, np.float32); b = np.fromfile(f.replace("dcpu", tag), np.float32)
            if a.size and a.size == b.size: w = max(w, float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))
        print(model, "stock ORT WebGPU" if tag == "dbase" else "texdirect m1c512+s2", "max rel diff vs CPU: %.2e" % w, flush=True)
