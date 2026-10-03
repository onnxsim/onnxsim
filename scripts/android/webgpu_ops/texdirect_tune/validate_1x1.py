"""End-to-end check of the 1x1 tile tables (ABAB-interleaved rounds). python validate_1x1.py MODEL"""
import json, re, statistics, sys
from tune_1x1 import D, MODELS, phone

model = sys.argv[1]
path, extra, warm, iters = MODELS[model]
tab = ";".join(f"{s}={v['cfg']}" for v in json.load(open(f"classes1x1_{model}_m2_c1024.json")).values() for s in v["shapes"] if v["cfg"] != "2,2,32,2,1")
confs = {
    "base": dict(mode=0, maxc=512, table=""),
    "m1c1024": dict(mode=1, maxc=1024, table=""),
    "m2c1024": dict(mode=2, maxc=1024, table=""),
    "m2c1024+tab": dict(mode=2, maxc=1024, table=tab),
}
lines = [f"cd {D}", f"LD_LIBRARY_PATH=. ./bench {path} webgpu 20 3 {extra} >/dev/null 2>&1"]
for r in range(5):
    for n, c in confs.items():
        lines.append(f"echo -n 'R {n} '; LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT={c['mode']} ORT_WEBGPU_TEXDIRECT_MAXC={c['maxc']} ORT_WEBGPU_TEXDIRECT_TABLE='{c['table']}' ./bench {path} webgpu {warm} {iters} {extra} 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2")
res = {}
for ln in phone(lines).splitlines():
    m = re.match(r"R (\S+) ([0-9.]+)", ln)
    if m:
        res.setdefault(m.group(1), []).append(float(m.group(2)))
base = statistics.median(res["base"])
out = [model]
for n, v in res.items():
    out.append(f"  {n:12s} {statistics.median(v):7.2f} ({base - statistics.median(v):+.2f}) {v}")
print("\n".join(out))
json.dump({"configs": confs, "results": res}, open(f"validate1x1_{model}.json", "w"), indent=1)
