"""coarse_classes.py shapes.json : rewrite <shapes>_classes.json with coarse labels 'KIND kxk owN' (conv) / op type (others),
keeping the fine classes in <shapes>_rows.json (written by classify2.py)."""
import sys, json, re
sh = sys.argv[1]
rows = json.load(open(sh.replace(".json", "_rows.json")))
out = {}
for r in rows:
    c = r["cls"]
    mm = re.match(r"Conv (\w+) (\d+x\d+) s(\d+) \d+>\d+ ow(\d+)", c)
    out[r["name"]] = f"Conv {mm.group(1)} {mm.group(2)} s{mm.group(3)} ow{mm.group(4)}" if mm else r["op"]
json.dump(out, open(sh.replace(".json", "_classes.json"), "w"))
print(len(set(out.values())), "coarse classes")
