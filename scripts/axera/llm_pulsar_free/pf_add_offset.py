"""A copy of a generated program set whose Adds carry an exact offset word
(``u16_retarget.exact_add_offset``) instead of Pulsar2's.

The card's U16 Add computes ``(qa * A + qb * B + off) >> n`` with the scale
ratios rounded to u16 ``Q(n)``. Pulsar2's offset word takes the unrounded
ratios, so every element of the sum is off by ``zp_a * (A / 2^n - r_a) + zp_b *
(B / 2^n - r_b)`` codes, the same for every channel: up to ~2 codes where the
zero points are large (the residual stream, the RoPE sums) and a ratio is
just above 1 (``n = 14``). The native program, and so every retargeted one,
has this bias (``pf_fake_quant.card_add`` models it; ``pf_check.py`` compares
with that model). Here only the offset records change: each Add's four
``0x1ef0..0x1f20`` writes, found by Pulsar2's value. Linears and other
records are untouched; the manifests mark the patched programs
(``"add_offset": "exact"``) for ``pf_check.py`` / ``pf_estimate.py``.

``pf_generate.py`` applies the same patch to each program as it writes it
(``--add-offset exact``, the default); this script patches a set generated
with ``--add-offset pulsar2`` (or before the default changed) into a copy.

usage: MODEL=... pf_add_offset.py <set dir> <out dir>
       (<set dir>: a pf_generate output with manifest_{first,rest[,nl]}.json)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import u16_retarget as ur  # noqa: E402

#: the program sets a pf_generate output can hold (nl: llama-160m's newline steps)
SETS = ("first", "rest", "nl")
#: an Add program (residual Add, a split linear's Add): its tensor names
ADD_NODES = [("Add", ("a", "b"), "y", {})]


def _link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def patch_entry(src: str, out: str, st: str, wh: str, k: str, e: dict) -> int:
    """Patch one manifest entry ``e`` (program ``k`` of set ``st`` at ``wh``):
    read ``<src>/<st>/<wh>/<k>*.axmodel``, write ``<out>/...`` (``out`` may be
    ``src``: in place, as ``pf_generate.py`` does by default), linking the
    parts without an Add. Marks ``e`` ``"add_offset": "exact"`` when an
    offset changed; returns the number of Adds changed."""
    if "shared" in e:  # the rest set's program serves this one
        return 0
    d = os.path.join(out, st, wh)
    os.makedirs(d, exist_ok=True)
    parts = {k: None}
    if e["kind"] == "fused":
        parts[k] = (ur.graph_nodes(os.path.join(src, e["model"])), e["calibration"])
    elif e["kind"] == "add":
        parts[k] = (ADD_NODES, e["calibration"])
    elif e["kind"] == "split":
        parts = {f"{k}_{p}": None for p in ("hi", "lo")}
        parts[f"{k}_add"] = (ADD_NODES, e["split"]["add"]["calibration"])
    n_all = 0
    for pk, spec in parts.items():
        path = os.path.join(src, st, wh, f"{pk}.axmodel")
        dst = os.path.join(d, f"{pk}.axmodel")
        if spec is None:
            _link(path, dst)
            continue
        nodes, cal = spec
        q = {t: tuple(v) for t, v in cal.items()}
        m, n = ur.exact_add_offsets(path, nodes, q)
        open(dst, "wb").write(m.SerializeToString())
        n_all += n
    if n_all:
        e["add_offset"] = "exact"
    return n_all


def patch_set(src: str, out: str) -> dict:
    """Write ``out`` (manifests, patched programs; the rest symlinked to
    ``src``). Returns ``{"<set> <where> <program>": number of Adds changed}``."""
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):  # cal/, acts/, gptq/: read only; no result files
        if f not in SETS and os.path.isdir(os.path.join(src, f)):
            _link(os.path.join(src, f), os.path.join(out, f))
    done = {}
    for st in SETS:
        if not os.path.exists(os.path.join(src, f"manifest_{st}.json")):
            continue  # nl: llama-160m only
        man = json.load(open(os.path.join(src, f"manifest_{st}.json")))
        for wh, ents in man.items():
            for k, e in ents.items():
                n = patch_entry(src, out, st, wh, k, e)
                if n:
                    done[f"{st} {wh} {k}"] = n
        json.dump(man, open(os.path.join(out, f"manifest_{st}.json"), "w"), indent=1)
    return done


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    done = patch_set(sys.argv[1], sys.argv[2])
    print(f"{len(done)} programs, {sum(done.values())} Adds with a new offset")


if __name__ == "__main__":
    main()
