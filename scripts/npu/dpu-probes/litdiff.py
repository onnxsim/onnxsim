"""Diff the literal (non-relocated) 32-bit words of the DPU and DMA sections between the single 1x1 conv blob and the chained
1x1 blob (8->16->16). Relocated words are skipped and reported separately. Run on H8 after reloc.py."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R

D = os.path.expanduser("~/npu-dpu/roadmap/reloc")
single = os.path.join(D, "single.blob")
chain = os.path.join(D, "chain.blob")


def sec_words(path):
    data = open(path, "rb").read()
    S = R.elf_sections(data)
    out = {}
    for s in S:
        if s["name"].startswith("task.dpu.") or s["name"].startswith("task.dma."):
            out[s["name"]] = np.frombuffer(data[s["off"]:s["off"] + s["size"] - (s["size"] % 4)], dtype=np.uint32).copy()
    return out


def reloc_words(path):
    rel = {}
    for target, off, _, _ in R.relocs(path):
        rel.setdefault(target, set()).update({off // 4, off // 4 + 1})
    return rel


ws, wc = sec_words(single), sec_words(chain)
rs, rc = reloc_words(single), reloc_words(chain)
for name in sorted(set(ws) | set(wc)):
    a, b = ws.get(name), wc.get(name)
    print("%-26s single %s words | chain %s words" % (name, None if a is None else len(a), None if b is None else len(b)), flush=True)
    if a is None or b is None:
        continue
    skip_a = rs.get(name, set()); skip_b = rc.get(name, set())
    n = min(len(a), len(b))
    diff = [j for j in range(n) if a[j] != b[j] and j not in skip_a and j not in skip_b]
    for j in diff[:30]:
        print("   word %4d  single 0x%08x  chain 0x%08x" % (j, a[j], b[j]), flush=True)
    print("   literal words differing: %d (first 30 shown)" % len(diff), flush=True)
