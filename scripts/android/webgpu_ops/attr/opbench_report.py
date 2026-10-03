"""opbench_report.py outdir tag ob_results.txt -> per-op ms table (x occurrences)"""
import sys, json, re
outdir, tag, res = sys.argv[1:4]
man = json.load(open(f"{outdir}/manifest_{tag}.json"))
vals = {}
for l in open(res):
    mm = re.match(r"(ob_\w+?_\d+) K2 median=([0-9.]+) K32 median=([0-9.]+)", l.strip())
    if mm and mm.group(1).startswith(f"ob_{tag}_"):
        vals[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
rows = []
floor = 0.0
for k, v in man.items():
    if v["op"] == "Relu" and v["ins"] == [[1, 4]] and k in vals:
        floor = (vals[k][1] - vals[k][0]) / 30
print(f"harness floor per copy (tiny Relu + keeper + output copy): {floor:.3f} ms")
for k, v in man.items():
    if k in vals and not (v["op"] == "Relu" and v["ins"] == [[1, 4]]):
        a, b = vals[k]
        per = max((b - a) / 30 - floor, 0.0)
        rows.append((per * v["n"], per, v["n"], v["op"], v["ins"], v["outs"], k))
rows.sort(reverse=True)
tot = 0
for tot_ms, per, n, op, ins, outs, k in rows:
    tot += tot_ms
    if tot_ms >= 0.15 or "-v" in sys.argv:
        print(f"{tot_ms:7.2f} ms total  {per:6.3f} ms/op x{n:<3d} {op:22s} {ins} -> {outs}")
print(f"sum of isolated op costs: {tot:.1f} ms")

# category sums
cat = {"Gemm/MatMul": ("Gemm", "MatMul"), "Transpose": ("Transpose",), "Gather*/TopK/Reduce": ("Gather", "GatherElements", "TopK", "ReduceMax"),
       "Concat/Resize/Pad/Slice/Tile": ("Concat", "Resize", "Pad", "Slice", "Tile"), "LayerNorm/Softmax": ("LayerNormalization", "SkipLayerNormalization", "Softmax"),
       "elementwise": ("Add", "Mul", "Div", "Sub", "Relu", "Sigmoid", "Erf"), "Reshape (free alias?)": ("Reshape",)}
print("\nper category (isolated, floor-corrected):")
for c, ops in cat.items():
    t = sum(r[0] for r in rows if r[3] in ops)
    print(f"  {c:32s} {t:6.1f} ms")
