"""Per-shape tile search for the texture-weight direct conv, end-to-end on the phone (no ORT rebuild).

    python tune.py MODEL MODE MAXC shape1,shape2,... [--out table.json]

Each candidate is timed ABAB against the default (no table entry), batched into one phone-run call.
A candidate is accepted if it beats the default by > MARGIN ms in BOTH pairs.
"""
import json, os, subprocess, sys, tempfile, re

D = "/data/local/tmp/wg-td-tune"
HOST = "/mnt/data/cache/claude-work/wgtd"
MARGIN = 0.3
MODELS = {
    "yolo11n": ("m/yolo11n.onnx", "enableInt64=1"),
    "yolo26n": ("m/yolo26n.onnx", "enableInt64=1"),
    "resnet50": ("m/resnet50.onnx", "shape=pixel_values:1,3,224,224"),
}
WARM0 = {"yolo11n": 80, "yolo26n": 90, "resnet50": 90}
ITERS = 14


def run_batch(model, mode, maxc, items, warm=None):
    """items: list of (label, table_str). returns {label: [A1,B1,A2,B2]}"""
    path, extra = MODELS[model]
    env = f"LD_LIBRARY_PATH=. ORT_WEBGPU_CONV_TEXDIRECT={mode} ORT_WEBGPU_TEXDIRECT_MAXC={maxc}"
    lines = ["cd %s" % D]
    w0 = WARM0[model]
    lines.append(f"{env} ./bench {path} webgpu {w0} 5 {extra} >/dev/null 2>&1")  # ramp the clock
    for label, table in items:
        for rnd, which in enumerate("ABAB"):
            t = "" if which == "A" else table
            lines.append(
                f"echo -n 'R {label} {which} '; {env} ORT_WEBGPU_TEXDIRECT_TABLE='{t}' ./bench {path} webgpu 25 {ITERS} {extra} 2>&1 | tail -1 | grep -o 'median=[0-9.]*' | cut -d= -f2"
            )
    script = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        local = f.name
    out = subprocess.run(
        ["bash", "-c", f"~/.cache/android-phone/phone-run bash -c 'adb push {local} {D}/job.sh >/dev/null; adb shell \"sh {D}/job.sh\"'"],
        capture_output=True, text=True).stdout
    os.unlink(local)
    res = {}
    for ln in out.splitlines():
        m = re.match(r"R (\S+) ([AB]) ([0-9.]+)", ln)
        if m:
            res.setdefault(m.group(1), []).append((m.group(2), float(m.group(3))))
    return {k: [v for _, v in vs] for k, vs in res.items()}


def accept(r):
    if len(r) != 4:
        return None
    a1, b1, a2, b2 = r
    return (a1 - b1 > MARGIN and a2 - b2 > MARGIN), ((a1 - b1) + (a2 - b2)) / 2


def cfg(tm, nv, wx, wy, o):
    return f"{tm},{nv},{wx},{wy},{o}"


def search(model, mode, maxc, shape, default=(2, 2, 32, 2, 1), log=print):
    tm0, nv0, wx0, wy0, o0 = default
    best = default
    best_gain = 0.0
    def evaluate(cands):
        nonlocal best, best_gain
        items = [(f"c{i}", f"{shape}={cfg(*c)}") for i, c in enumerate(cands)]
        res = run_batch(model, mode, maxc, items)
        for i, c in enumerate(cands):
            r = res.get(f"c{i}")
            a = accept(r) if r else None
            log(f"  {shape} {cfg(*c)} -> {r} {a}")
            if a and a[0] and a[1] > best_gain:
                best, best_gain = c, a[1]
    # stage A: order / workgroup shape
    A = [(tm0, nv0, 32, 2, 1), (tm0, nv0, 16, 4, 1), (tm0, nv0, 64, 1, 1), (tm0, nv0, 16, 8, 1), (tm0, nv0, 8, 8, 1),
         (tm0, nv0, 16, 4, 0), (tm0, nv0, 8, 8, 0), (tm0, nv0, 32, 2, 0)]
    A = [c for c in A if c != default]
    evaluate(A)
    tmb, nvb, wxb, wyb, ob = best
    B = [(t, nvb, wxb, wyb, ob) for t in (1, 4, 8) if t != tmb]
    evaluate(B)
    tmb, nvb, wxb, wyb, ob = best
    C = [(tmb, n, wxb, wyb, ob) for n in (1, 4) if n != nvb]
    evaluate(C)
    return best, best_gain


if __name__ == "__main__":
    model, mode, maxc = sys.argv[1], sys.argv[2], sys.argv[3]
    shapes = sys.argv[4].split(",")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"table_{model}_m{mode}_c{maxc}.json"
    table = json.load(open(out)) if os.path.exists(out) else {}
    for s in shapes:
        if s in table:
            continue
        best, gain = search(model, mode, maxc, s)
        print(f"== {model} mode{mode} maxc{maxc} {s}: best {cfg(*best)} gain {gain:.2f} ms", flush=True)
        table[s] = {"cfg": cfg(*best), "gain_ms": round(gain, 2)}
        json.dump(table, open(out, "w"), indent=1)
