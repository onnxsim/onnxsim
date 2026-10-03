"""Whole-model screen of (mode, MAXC) with the default tile, ABAB-interleaved rounds. python screen.py MODEL"""
import re, statistics, sys
from tune_1x1 import D, MODELS, phone

model = sys.argv[1]
path, extra, warm, iters = MODELS[model]
confs = {"base": (0, 512), "m1c512": (1, 512), "m1c1024": (1, 1024), "m2c512": (2, 512), "m2c1024": (2, 1024), "m2c2048": (2, 2048)}
lines = [f"cd {D}", f"LD_LIBRARY_PATH=. ./bench {path} webgpu 20 3 {extra} >/dev/null 2>&1"]
for r in range(4):
    for n, (m, c) in confs.items():
        lines.append(f"echo -n 'R {n} '; LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT={m} ORT_WEBGPU_TEXDIRECT_MAXC={c} ./bench {path} webgpu {warm} {iters} {extra} 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2")
res = {}
for ln in phone(lines).splitlines():
    m = re.match(r"R (\S+) ([0-9.]+)", ln)
    if m:
        res.setdefault(m.group(1), []).append(float(m.group(2)))
base = statistics.median(res["base"])
print(model)
for n, v in res.items():
    print(f"  {n:9s} {statistics.median(v):7.2f} ({base - statistics.median(v):+.2f}) {v}")
