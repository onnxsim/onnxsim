"""End-to-end check of whole tables/settings: configurations are interleaved over several rounds on the phone.

    python validate.py MODEL     (see CONFIGS below)
"""
import json, re, subprocess, sys, tempfile, os, statistics
from tune import D, MODELS, WARM0
from tune_class import shapes_of

S2 = "2,2,16,8,1"

def stride2_table(shapes, tile=S2):
    return ";".join(f"{s}={tile}" for s in shapes if s.split(":")[1] == "2")

def configs(model):
    sh64 = shapes_of(model, 1, 64); sh256 = shapes_of(model, 1, 256); sh512 = shapes_of(model, 1, 512)
    sh2 = shapes_of(model, 2, 128)
    one = ";".join(f"{s}=2,1,16,4,1" for s in sh2 if s.startswith("1:1") and s.split(":")[4] in ("40",)) + ";" + ";".join(f"{s}=2,1,32,2,1" for s in sh2 if s.startswith("1:1") and s.split(":")[4] == "160")
    return {
        "base": dict(mode=0, maxc=64, table=""),
        "m1c64": dict(mode=1, maxc=64, table=""),
        "m1c64+s2": dict(mode=1, maxc=64, table=stride2_table(sh64)),
        "m1c256": dict(mode=1, maxc=256, table=""),
        "m1c256+s2": dict(mode=1, maxc=256, table=stride2_table(sh256)),
        "m1c512+s2": dict(mode=1, maxc=512, table=stride2_table(sh512)),
        "m2c128+s2+1x1": dict(mode=2, maxc=128, table=stride2_table(sh2) + ";" + one),
    }

def run(model, cfgs, rounds=4):
    path, extra = MODELS[model]
    lines = [f"cd {D}", f"LD_LIBRARY_PATH=. ./bench {path} webgpu {WARM0[model]} 5 {extra} >/dev/null 2>&1"]
    for r in range(rounds):
        for name, c in cfgs.items():
            lines.append(f"echo -n 'R {name} '; LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT={c['mode']} ORT_WEBGPU_TEXDIRECT_MAXC={c['maxc']} ORT_WEBGPU_TEXDIRECT_TABLE='{c['table']}' ./bench {path} webgpu 25 14 {extra} 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2")
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write("\n".join(lines) + "\n"); local = f.name
    out = subprocess.run(["bash", "-c", f"~/.cache/android-phone/phone-run bash -c 'adb push {local} {D}/job.sh >/dev/null; adb shell \"sh {D}/job.sh\"'"], capture_output=True, text=True).stdout
    os.unlink(local)
    res = {}
    for ln in out.splitlines():
        m = re.match(r"R (\S+) ([0-9.]+)", ln)
        if m: res.setdefault(m.group(1), []).append(float(m.group(2)))
    return res

if __name__ == "__main__":
    model = sys.argv[1]
    cfgs = configs(model)
    if len(sys.argv) > 2: cfgs = {k: v for k, v in cfgs.items() if k in sys.argv[2].split(",")}
    res = run(model, cfgs)
    base = statistics.median(res["base"])
    print(model, "median over rounds (ms), gain vs base:")
    for k, v in res.items():
        print(f"  {k:16s} {statistics.median(v):7.2f}  ({base - statistics.median(v):+.2f})  runs={v}")
    json.dump({"configs": cfgs, "results": res}, open(f"validate_{model}.json", "w"), indent=1)
