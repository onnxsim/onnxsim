"""Class-wise tile search for the fused Concat->1x1 conv (NhwcConcatConv1x1), end to end on the phone, private library copy.

    python tune_cc.py MODEL TEXDIRECT [--out FILE]

Shapes `rows:cout:ow` (from ORT_WEBGPU_CONCATCONV_LOG) are grouped by output width; a whole class gets the same candidate
(tm,nv,wx,wy) at once. A candidate is timed ABAB against the default tile and accepted if it wins by > 0.3 ms in both pairs.
"""
import json, os, re, subprocess, sys, tempfile

D = "/data/local/tmp/wg-cc-tune"
MARGIN = 0.3
WARM0 = {"yolo11n": 80, "yolo26n": 90}
ITERS = 14


def phone(lines):
    script = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        local = f.name
    out = subprocess.run(["bash", "-c", f"~/.cache/android-phone/phone-run bash -c 'adb push {local} {D}/job.sh >/dev/null; adb shell \"sh {D}/job.sh\"'"],
                         capture_output=True, text=True).stdout
    os.unlink(local)
    return out


def env(td, extra=""):
    return f"LD_LIBRARY_PATH=. ORT_WEBGPU_CONCAT_CONV=1 ORT_WEBGPU_CONV_TEXDIRECT={td} {extra}"


def shapes_of(model, td):
    out = phone([f"cd {D}", f"{env(td)} ORT_WEBGPU_CONCATCONV_LOG=1 ./bench m/{model}.onnx webgpu 0 1 enableInt64=1 2>&1 | grep CONCATCONV | sort -u"])
    return [l.split()[1] for l in out.splitlines() if l.startswith("CONCATCONV")]


def run_batch(model, td, items):
    lines = [f"cd {D}", f"{env(td)} ./bench m/{model}.onnx webgpu {WARM0[model]} 5 enableInt64=1 >/dev/null 2>&1"]
    for label, table in items:
        for which in "ABAB":
            t = "" if which == "A" else table
            lines.append(f"echo -n 'R {label} {which} '; {env(td, f'ORT_WEBGPU_CONCATCONV_TABLE={chr(39)}{t}{chr(39)}')} ./bench m/{model}.onnx webgpu 25 {ITERS} enableInt64=1 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2")
    out = phone(lines)
    res = {}
    for ln in out.splitlines():
        m = re.match(r"R (\S+) ([AB]) ([0-9.]+)", ln)
        if m:
            res.setdefault(m.group(1), []).append(float(m.group(3)))
    return res


def accept(r):
    if len(r) != 4:
        return None
    a1, b1, a2, b2 = r
    return (a1 - b1 > MARGIN and a2 - b2 > MARGIN), ((a1 - b1) + (a2 - b2)) / 2


def cfg(c):
    return ",".join(map(str, c))


if __name__ == "__main__":
    model, td = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"cc_{model}_td{td}.json"
    shapes = shapes_of(model, td)
    cl = {}
    for s in shapes:
        cl.setdefault("w" + s.split(":")[2], []).append(s)
    print(model, td, len(shapes), "shapes", {k: len(v) for k, v in cl.items()}, flush=True)
    default = (2, 2, 32, 2)
    result = json.load(open(out)) if os.path.exists(out) else {}
    for name, ss in cl.items():
        if name in result:
            continue
        st = {"best": default, "gain": 0.0}

        def ev(cands):
            items = [(f"c{i}", ";".join(f"{s}={cfg(c)}" for s in ss)) for i, c in enumerate(cands)]
            res = run_batch(model, td, items)
            for i, c in enumerate(cands):
                r = res.get(f"c{i}")
                a = accept(r) if r else None
                print(f"  {name} {cfg(c)} -> {r} {a}", flush=True)
                if a and a[0] and a[1] > st["gain"]:
                    st["best"], st["gain"] = c, a[1]

        tm, nv = default[0], default[1]
        ev([(tm, nv, 16, 4), (tm, nv, 64, 1), (tm, nv, 16, 8), (tm, nv, 8, 8), (tm, nv, 8, 16), (tm, nv, 128, 1), (tm, nv, 32, 4)])
        b = st["best"]
        ev([(t, b[1], b[2], b[3]) for t in (1, 4, 8) if t != b[0]])
        b = st["best"]
        ev([(b[0], n, b[2], b[3]) for n in (1, 4, 8) if n != b[1]])
        print(f"== {name}: best {cfg(st['best'])} gain {st['gain']:.2f} ms", flush=True)
        result[name] = {"cfg": cfg(st["best"]), "gain_ms": round(st["gain"], 2), "shapes": ss}
        json.dump(result, open(out, "w"), indent=1)
    final = ";".join(f"{s}={v['cfg']}" for v in result.values() for s in v["shapes"] if v["cfg"] != cfg(default))
    print("TABLE", final)
