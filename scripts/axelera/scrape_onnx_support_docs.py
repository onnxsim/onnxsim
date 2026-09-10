#!/usr/bin/env python3
"""Regenerate `voyager_op_support_data.py` from Axelera's Voyager SDK docs.

Voyager SDK (https://github.com/axelera-ai-hub/voyager-sdk) publishes a
per-opset, per-operator ONNX support reference for its Metis AIPU compiler at
`docs/reference/compiler/onnx-support.md` (a hand-written summary table) and
`docs/reference/compiler/onnx-opset{14,15,16,17}-support.md` (auto-generated,
one section per operator, each with a `**AIPU Acceleration Constraints**`
block of `- rule: \`<expr>\`` (all must hold) and/or
`- allow_config: \`<expr>\`` (at least one must hold) lines, plus free-text
"Axelera's notes for developers").

This script re-scrapes that reference and regenerates
`voyager_op_support_data.py`, so the data here tracks a specific Voyager SDK
checkout rather than being hand-maintained. Verified (as of the checkout this
was last run against) that support level, rules, and allow_config are
byte-identical across opsets 14-16 vs. opset 17 for every operator that
appears in all of them -- so only opset 17 (the compiler's own recommended
default, see onnx-support.md's "Recommended opset" section) is scraped in
detail; the summary table is still cross-checked against all four opsets to
catch any future divergence.

Usage: clone axelera-ai-hub/voyager-sdk next to this checkout (or anywhere),
then:

    python3 scrape_onnx_support_docs.py /path/to/voyager-sdk > voyager_op_support_data.py

The constraint expression strings are transcribed verbatim -- including any
oddities in Axelera's own DSL (e.g. `B.shape==(0)`, where `(0)` is the bare
Python int `0`, not a 1-tuple -- almost certainly meant as a "no such input"
sentinel, but not "fixed" here since that would mean guessing at Axelera's
intent instead of reporting what the docs actually say). See
`voyager_simulator.py` for how (and how conservatively) these get evaluated.
"""

import re
import sys

OPSETS = (14, 15, 16, 17)
DETAIL_OPSET = 17


def parse_summary_table(docs_dir: str) -> dict:
    with open(f"{docs_dir}/onnx-support.md") as f:
        text = f.read()
    m = re.search(r"\| Operator \| Opset 14.*?\n((?:\|.*\n)+)", text)
    levels = {}
    levels_per_opset = {}
    for row in m.group(1).strip().splitlines():
        cells = [c.strip().strip("*") for c in row.strip("|").split("|")]
        if not cells[0] or set(cells[0]) == {"-"}:
            continue
        name = cells[0]
        per_opset = dict(zip(OPSETS, cells[1:5]))
        levels[name] = per_opset[DETAIL_OPSET]
        levels_per_opset[name] = per_opset
    return levels, levels_per_opset


def parse_opset_detail(docs_dir: str, opset: int) -> dict:
    with open(f"{docs_dir}/onnx-opset{opset}-support.md") as f:
        text = f.read()
    sections = re.split(r"\n## (\S+)\n", text)
    pairs = list(zip(sections[1::2], sections[2::2]))
    out = {}
    for name, body in pairs:
        m = re.search(r"\*\*AIPU Acceleration Constraints\*\*\n((?:- .*\n?)*)", body)
        block = m.group(1) if m else ""
        rules = re.findall(r"- rule: `(.*)`", block)
        allow_config = re.findall(r"- allow_config: `(.*)`", block)
        notes = None
        nm = re.search(r"Axelera's notes for developers\*\*\n\n(.*?)\n\n", body, re.S)
        if nm:
            notes = " ".join(nm.group(1).split())
        out[name] = {"rules": rules, "allow_config": allow_config, "notes": notes}
    return out


def main(voyager_sdk_root: str) -> None:
    docs_dir = f"{voyager_sdk_root}/docs/reference/compiler"
    levels, levels_per_opset = parse_summary_table(docs_dir)
    detail = parse_opset_detail(docs_dir, DETAIL_OPSET)

    for opset in OPSETS:
        if opset == DETAIL_OPSET:
            continue
        d = parse_opset_detail(docs_dir, opset)
        for name, entry in detail.items():
            if d.get(name) != entry:
                print(
                    f"WARNING: opset {opset} constraints for {name!r} differ from "
                    f"opset {DETAIL_OPSET} -- data below is opset-{DETAIL_OPSET}-only "
                    "and no longer accurate for other opsets; extend the data model "
                    "before trusting it across opsets.",
                    file=sys.stderr,
                )
        for name, per_opset in levels_per_opset.items():
            if len(set(per_opset.values())) != 1:
                print(
                    f"WARNING: support level for {name!r} differs across opsets: "
                    f"{per_opset} -- VOYAGER_OP_SUPPORT below only records the "
                    f"opset-{DETAIL_OPSET} value.",
                    file=sys.stderr,
                )

    print(
        '"""Scraped from Voyager SDK\'s docs/reference/compiler/onnx-support.md and\n'
        f"onnx-opset{DETAIL_OPSET}-support.md (opset {DETAIL_OPSET}, the compiler's own\n"
        "recommended default -- see scrape_onnx_support_docs.py's docstring for why\n"
        "only one opset is captured). Auto-generated -- do not hand-edit; re-run\n"
        'scrape_onnx_support_docs.py against a voyager-sdk checkout instead."""'
    )
    print()
    print("VOYAGER_OP_SUPPORT = {")
    for name in sorted(levels):
        level = levels[name]
        d = detail.get(name, {"rules": [], "allow_config": [], "notes": None})
        print(f"    {name!r}: {{")
        print(f"        'level': {level!r},")
        print(f"        'rules': {d['rules']!r},")
        print(f"        'allow_config': {d['allow_config']!r},")
        print(f"        'notes': {d['notes']!r},")
        print("    },")
    print("}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            f"usage: {sys.argv[0]} /path/to/voyager-sdk > voyager_op_support_data.py"
        )
    main(sys.argv[1])
