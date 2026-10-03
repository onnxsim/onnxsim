"""Class-wise tile search for the stride-1 classes (1x1, and dense 3x3 that Winograd does not take) of sam_l0_enc / rtdetr_pre.

    python tune_1x1.py MODEL MODE MAXC [--out FILE]

Own phone dir (/data/local/tmp/wg-1x1-tune, library copy), shorter timed runs than tune.py (SAM is 360 ms per inference), same rule:
accept a candidate for a class only if it beats the default tile by > 0.3 ms in BOTH ABAB pairs.
"""
import json, os, re, subprocess, sys, tempfile

D = "/data/local/tmp/wg-1x1-tune"
MARGIN = 0.3
MODELS = {"sam_l0_enc": ("m/sam_l0_enc.onnx", "enableInt64=1", 6, 8), "rtdetr_pre": ("m/rtdetr_pre.onnx", "enableInt64=1", 8, 8)}
DEFAULT = (2, 2, 32, 2, 1)


def cfg(tm, nv, wx, wy, o):
    return f"{tm},{nv},{wx},{wy},{o}"


def phone(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        local = f.name
    out = subprocess.run(["bash", "-c", f"~/.cache/android-phone/phone-run bash -c 'adb push {local} {D}/job.sh >/dev/null; adb shell \"sh {D}/job.sh\"'"],
                         capture_output=True, text=True).stdout
    os.unlink(local)
    return out


def run_batch(model, mode, maxc, items):
    path, extra, warm, iters = MODELS[model]
    env = f"LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT={mode} ORT_WEBGPU_TEXDIRECT_MAXC={maxc}"
    lines = [f"cd {D}", f"{env} ./bench {path} webgpu 20 3 {extra} >/dev/null 2>&1"]  # ramp the clock
    for label, table in items:
        for which in "ABAB":
            t = "" if which == "A" else table
            lines.append(f"echo -n 'R {label} {which} '; {env} ORT_WEBGPU_TEXDIRECT_TABLE='{t}' ./bench {path} webgpu {warm} {iters} {extra} 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2")
    res = {}
    for ln in phone(lines).splitlines():
        m = re.match(r"R (\S+) ([AB]) ([0-9.]+)", ln)
        if m:
            res.setdefault(m.group(1), []).append(float(m.group(3)))
    return res


def accept(r):
    if len(r) != 4:
        return None
    a1, b1, a2, b2 = r
    return (a1 - b1 > MARGIN and a2 - b2 > MARGIN), ((a1 - b1) + (a2 - b2)) / 2


def shapes_of(model, mode, maxc):
    path, extra, _, _ = MODELS[model]
    out = phone([f"cd {D}", f"LD_LIBRARY_PATH=. ORT_WEBGPU_TEXDIRECT_LOG=1 ORT_WEBGPU_CONV_TEXDIRECT={mode} ORT_WEBGPU_TEXDIRECT_MAXC={maxc} ./bench {path} webgpu 0 1 {extra} 2>&1 | grep TEXDIRECT | sort -u"])
    return [l.split()[1] for l in out.splitlines() if l.startswith("TEXDIRECT")]


def classes(shapes):
    cl = {}
    for s in shapes:
        kh, st, cin, cout, ow = s.split(":")
        if st == "1":
            cl.setdefault(f"k{kh}s{st}w{ow}", []).append(s)
    return cl


def main():
    model, mode, maxc = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"classes1x1_{model}_m{mode}_c{maxc}.json"
    cl = classes(shapes_of(model, mode, maxc))
    print(model, mode, maxc, {k: len(v) for k, v in cl.items()}, flush=True)
    result = json.load(open(out)) if os.path.exists(out) else {}
    for name, ss in cl.items():
        if name in result:
            continue
        state = {"best": DEFAULT, "gain": 0.0}

        def ev(cands):
            cands = [c for c in cands if c != DEFAULT and c[2] * c[3] <= 256]
            if not cands:
                return
            items = [(f"c{i}", ";".join(f"{s}={cfg(*c)}" for s in ss)) for i, c in enumerate(cands)]
            res = run_batch(model, mode, maxc, items)
            for i, c in enumerate(cands):
                r = res.get(f"c{i}")
                a = accept(r) if r else None
                print(f"  {name} {cfg(*c)} -> {r} {a}", flush=True)
                if a and a[0] and a[1] > state["gain"]:
                    state["best"], state["gain"] = c, a[1]

        tm0, nv0 = DEFAULT[0], DEFAULT[1]
        ev([(tm0, nv0, wx, wy, o) for (wx, wy) in [(16, 4), (64, 1), (16, 8), (8, 8), (8, 16), (32, 4), (4, 16)] for o in (1, 0)])
        b = state["best"]
        ev([(t, b[1], b[2], b[3], b[4]) for t in (1, 4, 8) if t != b[0]])
        b = state["best"]
        ev([(b[0], n, b[2], b[3], b[4]) for n in (1, 4, 8) if n != b[1]])
        print(f"== {name}: best {cfg(*state['best'])} gain {state['gain']:.2f} ms", flush=True)
        result[name] = {"cfg": cfg(*state["best"]), "gain_ms": round(state["gain"], 2), "shapes": ss}
        json.dump(result, open(out, "w"), indent=1)
    print("TABLE", ";".join(f"{s}={v['cfg']}" for v in result.values() for s in v["shapes"] if v["cfg"] != cfg(*DEFAULT)))


if __name__ == "__main__":
    main()
