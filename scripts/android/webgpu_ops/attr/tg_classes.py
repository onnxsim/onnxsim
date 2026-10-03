"""tg_classes.py PROFILE.txt : aggregate a tg_cl_bench TG_PROFILE_ALL per-call profile by kernel class.
Kernel name classes (tinygrad): E_* elementwise; r_* reductions: names ending in `_3_3_<n>` = 3x3 conv, `_7_7_<n>` = 7x7 stem conv (dims like 7_7 elsewhere in a name are spatial),
other r_ = 1x1 convs / matmul / pooling / softmax."""
import sys, re, collections
agg = collections.OrderedDict()
tot = 0.0
for line in open(sys.argv[1]):
    m = re.match(r"\s+#(\d+)\s+([0-9.]+) ms\s+(\S+)", line)
    if not m:
        continue
    ms, name = float(m.group(2)), m.group(3)
    if name.startswith("E_"):
        c = "elementwise (E_)"
    elif re.search(r"_7_7_\d+(_v\d+)?$", name):
        c = "r_ 7x7 conv (stem)"
    elif re.search(r"_3_3_\d+(_v\d+)?$", name):
        c = "r_ 3x3 conv"
    elif name.startswith("r_"):
        c = "r_ other (1x1 conv / matmul / pool)"
    else:
        c = "other"
    a = agg.setdefault(c, [0, 0.0])
    a[0] += 1
    a[1] += ms
    tot += ms
for k, (n, ms) in agg.items():
    print(f"{k:42s} {n:4d} calls {ms:8.2f} ms")
print(f"{'total GPU kernel time':42s} {sum(v[0] for v in agg.values()):4d} calls {tot:8.2f} ms")
