"""Class-wise tile search: shapes of a model that take the texture direct conv are grouped by (kernel, stride, output width)
and every shape of a class gets the same candidate tile at once, so the end-to-end signal is larger than for one layer.

    python tune_class.py MODEL MODE MAXC [--out FILE]
"""
import json
import os
import subprocess
import sys

from tune import D, MODELS, accept, cfg, run_batch


def shapes_of(model, mode, maxc):
    path, extra = MODELS[model]
    cmd = (
        f"~/.cache/android-phone/phone-run adb shell \"cd {D} && LD_LIBRARY_PATH=. ORT_WEBGPU_TEXDIRECT_LOG=1 "
        f"ORT_WEBGPU_CONV_TEXDIRECT={mode} ORT_WEBGPU_TEXDIRECT_MAXC={maxc} ./bench {path} webgpu 0 1 {extra} 2>&1 | grep TEXDIRECT | sort -u\""
    )
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    return [l.split()[1] for l in out.splitlines() if l.startswith("TEXDIRECT")]


def classes(shapes):
    cl = {}
    for s in shapes:
        kh, st, cin, cout, ow = s.split(":")
        cl.setdefault(f"k{kh}s{st}w{ow}", []).append(s)
    return cl


def table_str(shapes, c):
    return ";".join(f"{s}={cfg(*c)}" for s in shapes)


if __name__ == "__main__":
    model, mode, maxc = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"classes_{model}_m{mode}_c{maxc}.json"
    shapes = shapes_of(model, mode, maxc)
    cl = classes(shapes)
    print(model, mode, maxc, len(shapes), "shapes,", len(cl), "classes:", {k: len(v) for k, v in cl.items()}, flush=True)
    default = (2, 2, 32, 2, 1)
    result = json.load(open(out)) if os.path.exists(out) else {}
    for name, ss in cl.items():
        if name in result:
            continue
        state = {"best": default, "gain": 0.0}

        def ev(cands):
            items = [(f"c{i}", table_str(ss, c)) for i, c in enumerate(cands)]
            res = run_batch(model, mode, maxc, items)
            for i, c in enumerate(cands):
                r = res.get(f"c{i}")
                a = accept(r) if r else None
                print(f"  {name} {cfg(*c)} -> {r} {a}", flush=True)
                if a and a[0] and a[1] > state["gain"]:
                    state["best"], state["gain"] = c, a[1]

        tm0, nv0 = default[0], default[1]
        ev([(tm0, nv0, 16, 4, 1), (tm0, nv0, 64, 1, 1), (tm0, nv0, 16, 8, 1), (tm0, nv0, 8, 8, 1), (tm0, nv0, 16, 4, 0), (tm0, nv0, 8, 8, 0)])
        b = state["best"]
        ev([(t, b[1], b[2], b[3], b[4]) for t in (1, 4, 8) if t != b[0]])
        b = state["best"]
        ev([(b[0], n, b[2], b[3], b[4]) for n in (1, 4) if n != b[1]])
        print(f"== {name}: best {cfg(*state['best'])} gain {state['gain']:.2f} ms", flush=True)
        result[name] = {"cfg": cfg(*state["best"]), "gain_ms": round(state["gain"], 2), "shapes": ss}
        json.dump(result, open(out, "w"), indent=1)
    final = ";".join(f"{s}={v['cfg']}" for v in result.values() for s in v["shapes"] if v["cfg"] != cfg(*default))
    print("TABLE", final)
