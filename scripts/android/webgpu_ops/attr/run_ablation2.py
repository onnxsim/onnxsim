"""Run ABAB ablations on the phone.
   run_ablation.py WORKDIR BASE_NAME TAG SPEC_FILE [rounds] [warm] [iters]
SPEC_FILE: lines 'label|selector[|--identity]'. For each: ablate.py -> push -> ab.sh under phone-run; appends results to WORKDIR/TAG_results.jsonl"""
import sys, os, subprocess, json, re, statistics
work, base, tag, spec = sys.argv[1:5]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 4
warm = int(sys.argv[6]) if len(sys.argv) > 6 else 16
iters = int(sys.argv[7]) if len(sys.argv) > 7 else 8
bench_extra = os.environ.get("BENCH_EXTRA", "")
here = os.path.dirname(os.path.abspath(__file__))
py = "/mnt/data/cache/claude-work/ortenv/bin/python"
shapes = os.path.join(work, f"{tag}_shapes.json")
res_path = os.path.join(work, f"{tag}_results.jsonl")
phone_run = os.path.expanduser("~/.cache/android-phone/phone-run")
subprocess.run(["bash", "-c", f"{phone_run} bash -c 'adb push {here}/ab2.sh /data/local/tmp/wg-attr2/ >/dev/null; adb push {work}/{base} /data/local/tmp/wg-attr2/m/ >/dev/null'"], check=True, capture_output=True)
for line in open(spec):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split("|")
    label, sel = parts[0], parts[1]
    extra = parts[2:] 
    var = f"{tag}_{re.sub('[^A-Za-z0-9]+', '_', label)}.onnx"
    if sel == "SELF":
        subprocess.run(["cp", os.path.join(work, base), os.path.join(work, var)], check=True)
    else:
        r = subprocess.run([py, os.path.join(here, "ablate2.py"), os.path.join(work, base), shapes, os.path.join(work, var), sel] + extra, capture_output=True, text=True)
        if r.returncode != 0:
            print("ablate failed", label, r.stderr[-300:]); continue
        nrep = r.stdout.split()[0]
    cmd = f"{phone_run} bash -c 'adb push {work}/{var} /data/local/tmp/wg-attr2/m/ >/dev/null; adb shell \"sh /data/local/tmp/wg-attr2/ab2.sh {base} {var} {warm} {iters} {rounds} {bench_extra}\"; adb shell rm /data/local/tmp/wg-attr2/m/{var}'"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    A = [float(x) for x in re.findall(r"A median=([0-9.]+)", out)]
    B = [float(x) for x in re.findall(r"B median=([0-9.]+)", out)]
    if not A or len(A) != len(B):
        print("measure failed", label, out[-300:]); continue
    d = [a - b for a, b in zip(A, B)]
    rec = dict(label=label, selector=sel, replaced=nrep if sel != "SELF" else "0", A=A, B=B, saved_ms=statistics.mean(d),
               sd=(statistics.stdev(d) if len(d) > 1 else 0.0), base_ms=statistics.mean(A))
    print(f"{label:34s} nodes={rec['replaced']:>3s} base={rec['base_ms']:.1f} saved={rec['saved_ms']:+.1f} ms (sd {rec['sd']:.1f}, n={len(d)})", flush=True)
    open(res_path, "a").write(json.dumps(rec) + "\n")
    os.remove(os.path.join(work, var))
