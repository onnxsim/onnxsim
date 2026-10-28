"""Template registry of the Pulsar-free path: a canonical signature for every
template, the template keys a model needs at a layout, where existing
templates are, and when one may stand in for another.

Names (``pf_templates.py``) carry the widths only (``nrm1024``,
``alyrqkr2048``); the dims that change a template's records (eps, heads,
head size, context, the q / k norm, the graph variant) and the Pulsar2
config are not in the name. A signature holds all of them:

* ``kind`` and ``widths`` (the numbers in the name: ``n``, ``n2``; a linear's
  ``K`` and ``N``), ``alt`` (the ``t`` suffix) as ``variant.alt``;
* ``dims``: the ``pf_templates.Dims`` fields the graph reads, for the kind
  (``DEPS``; a conservative superset where a kind reads several);
* ``variant``: ``qk_norm`` (for the kinds that exist only with a q / k norm,
  and ``rope`` without one), ``rot`` (rotate_half made in the program),
  ``split`` (the split first-set linears);
* ``pulsar2``: image, NPU mode, dtype, calibration method, the layer ops the
  build quantizes, the number of variants, and the calibration draws (count
  and seed).

``sig_id`` is the SHA-256 of the canonical JSON of the signature (16 hex
digits). Two templates with equal ``sig_id`` are the same build up to Pulsar2's
noise (``u16_retarget.compare_jobs``).

A model's keys (``keys_for``) are ``pf_templates.needed`` for its ``ModelSpec``
and layout; a key is then:

* ``exact``: a template directory with the same ``sig_id`` exists in a root;
* ``verified``: a stored reuse record says a directory of another signature
  stands in for it (``compare_jobs`` equal on every variant);
* ``check``: no exact match, but a directory of the same name and another
  signature exists: a fresh build at this signature is compared with it, and
  the verdict is recorded (``resolve``);
* ``missing``: nothing of that name: a build.

Roots: a template root is either a directory with a ``signature.json`` per
template (written by ``resolve`` for the directories it builds; shared
directories are not edited) or a root with an owner (the ``ModelSpec`` whose
dims built it, for the shipped ``templates`` / ``templates_llama``).

The repo manifest ``template_registry.json`` (next to this file) holds the
model specs, the signatures of every key of the shipped models and of the
third model at every layout, and the reuse records. Regenerate its
signatures with ``pf_registry.py manifest``; reuse records are kept.

usage: pf_registry.py keys --model M --layout N
       pf_registry.py report --model M --layout N [--root DIR[=MODEL]]...
       pf_registry.py build --model M --layout N --out DIR [--root ...] [--jobs J]
       pf_registry.py manifest
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pf_layout as lay  # noqa: E402
import pf_templates as tpl  # noqa: E402

SCHEMA = 1
MANIFEST = os.path.join(HERE, "template_registry.json")
SIG_FILE = "signature.json"
GROUPS = ("fused", "linear", "single")
LAYOUTS = tuple(range(12))

NORM = ("H", "eps")
ATTN = ("nh", "nkv", "hd", "S")
QK = ("nh", "nkv", "hd")
LAYER = ("H", "I", "eps", "nh", "nkv", "hd", "S")
#: the Dims fields each kind's graph reads (beyond the widths in its name)
DEPS = {
    "add": (),
    "rope": (),
    "silu": (),
    "crms": ("eps",),
    "rms": ("eps",),
    "nrm": ("eps",),
    "anorm": ("eps",),
    "hcrms": ("hd", "eps"),
    "qkrope": ("hd", "eps"),
    "attn": ATTN,
    "qkf": QK + ("eps",),
    "lanorm": NORM,
    "slanorm": NORM,
    "lanorml": NORM,
    "slanorml": NORM,
    "nrml": NORM,
    "nrmqk": NORM + QK,
    "nrmqkr": NORM + QK,
    "slanormqk": NORM + QK,
    "slanormqkr": NORM + QK,
    "alanorm": NORM + ATTN,
    "alanorml": NORM + ATTN,
    "lyrqk": LAYER,
    "lyrqkr": LAYER,
    "alyrqk": LAYER,
    "alyrqkr": LAYER,
    "alyrqkrs": LAYER,
    "alyrf": LAYER,
}
#: kinds that exist only with a per-head q / k norm (``rope``: without one)
QK_KINDS = {
    "qkrope", "hcrms", "qkf", "nrmqk", "nrmqkr", "slanormqk", "slanormqkr",
    "lyrqk", "lyrqkr", "alyrqk", "alyrqkr", "alyrqkrs", "alyrf",
}  # fmt: skip
#: rotate_half made in the program (Split / Concat), not by two linears
ROT_KINDS = {"nrmqkr", "slanormqkr", "lyrqkr", "alyrqkr", "alyrqkrs"}
SPLIT_KINDS = {"alyrqkrs"}

_NAME = re.compile(r"([a-z]+?)(\d+)(?:x(\d+))?(t?)")
_LINEAR = re.compile(r"k(\d+)n(\d+)")


@dataclass(frozen=True)
class Name:
    """A template name parsed: ``kind`` ``n`` (``x n2``) and the ``t`` suffix."""

    kind: str
    n: int
    n2: int | None = None
    alt: bool = False

    @property
    def text(self) -> str:
        if self.kind == "linear":
            return f"k{self.n}n{self.n2}"
        return (
            f"{self.kind}{self.n}"
            + (f"x{self.n2}" if self.n2 is not None else "")
            + ("t" if self.alt else "")
        )


def parse(name: str) -> Name:
    """``nrm1024`` / ``lanorml2048x6144`` / ``qkrope2048t`` / ``k1024n6144`` /
    ``add1024``; ``ValueError`` for anything else."""
    m = _LINEAR.fullmatch(name)
    if m:
        return Name("linear", int(m[1]), int(m[2]))
    m = _NAME.fullmatch(name)
    if not m or (m[1] not in DEPS and m[1] != "add"):
        raise ValueError(f"{name}: not a template name")
    return Name(m[1], int(m[2]), int(m[3]) if m[3] else None, bool(m[4]))


@dataclass(frozen=True)
class ModelSpec:
    """The dims a model's templates depend on, and its layer types: the q / k
    norm and the split first-set layers (``split_layers``: the linears
    ``split_keys`` as two s8 linears and an Add)."""

    name: str
    H: int
    I: int  # noqa: E741
    nh: int
    nkv: int
    hd: int
    eps: float
    vocab: int
    layers: int
    qk_norm: bool = True
    S: int = lay.S
    split_layers: tuple = ()
    split_keys: tuple = ()
    default_layout: int = 0

    @property
    def kvd(self) -> int:
        return self.nkv * self.hd

    def dims(self) -> tpl.Dims:
        return tpl.Dims(
            self.H, self.I, self.nh, self.nkv, self.hd, self.eps, self.vocab, self.S
        )

    @classmethod
    def from_config(
        cls, name: str, config: dict, *, qk_norm=None, split=None, layout=0
    ):
        """A Hugging Face ``config.json`` (a dict) as a spec. ``split``: the
        ``(layers, keys)`` of the split first set (``None``: none)."""
        nh = config["num_attention_heads"]
        hd = config.get("head_dim") or config["hidden_size"] // nh
        if qk_norm is None:
            qk_norm = str(config.get("model_type", "")).startswith("qwen3")
        layers, keys = split or ((), ())
        return cls(
            name=name,
            H=config["hidden_size"],
            I=config["intermediate_size"],
            nh=nh,
            nkv=config.get("num_key_value_heads", nh),
            hd=hd,
            eps=float(config["rms_norm_eps"]),
            vocab=config["vocab_size"],
            layers=config["num_hidden_layers"],
            qk_norm=bool(qk_norm),
            split_layers=tuple(layers),
            split_keys=tuple(keys),
            default_layout=layout,
        )


#: the shipped models (dims from their Hugging Face configs; ``pf_model.MODELS``)
#: and the third model: Qwen3-4B's dims (the config of the AWQ repository in the
#: HF cache; the template builds need no checkpoint). Its split first set is
#: Qwen3-0.6B's, assumed (not measured).
QWEN_SPLIT = ((0, 1, 2), ("gu", "down_proj"))
SPECS = {
    "qwen3_0.6b": ModelSpec(
        "qwen3_0.6b", H=1024, I=3072, nh=16, nkv=8, hd=128, eps=1e-06,
        vocab=151936, layers=28, qk_norm=True, split_layers=(0, 1, 2),
        split_keys=("gu", "down_proj"), default_layout=8,
    ),
    "llama160m": ModelSpec(
        "llama160m", H=768, I=3072, nh=12, nkv=12, hd=64, eps=1e-06,
        vocab=32000, layers=12, qk_norm=False, default_layout=3,
    ),
    "qwen3_4b": ModelSpec(
        "qwen3_4b", H=2560, I=9728, nh=32, nkv=8, hd=128, eps=1e-06,
        vocab=151936, layers=36, qk_norm=True, split_layers=(0, 1, 2),
        split_keys=("gu", "down_proj"), default_layout=8,
    ),
}  # fmt: skip


class _Cfg:
    """The attributes ``pf_templates.needed`` reads, from a spec and a layout."""

    def __init__(self, spec: ModelSpec, layout: int):
        self.H, self.I, self.nh, self.nkv, self.hd = (
            spec.H,
            spec.I,
            spec.nh,
            spec.nkv,
            spec.hd,
        )
        self.kvd, self.vocab, self.qk_norm = spec.kvd, spec.vocab, spec.qk_norm
        self.QKV = lay.qkv_parts(spec.nh, spec.hd, spec.kvd, spec.qk_norm)
        self.split_first = {
            "layers": list(spec.split_layers),
            "keys": list(spec.split_keys),
        }
        self.fuse = int(layout)  # fmt: skip


def keys_for(spec: ModelSpec, layout: int) -> list[str]:
    """The template names (``fused`` / ``add<n>`` / ``k<K>n<N>``) a model needs
    at ``layout``: ``pf_templates.needed`` for the spec (the same list, so the
    shipped sets' names are unchanged)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout}: one of {LAYOUTS}")
    return tpl.needed(_Cfg(spec, layout))


# ---- signatures -----------------------------------------------------------------------
def _pulsar2(p: Name, d: tpl.Dims) -> dict:
    base = {
        "image": tpl.IMAGE,
        "npu_mode": "NPU1",
        "dtype": "U16",
        "calibration_method": "MinMax",
    }
    if p.kind == "linear":
        return base | {"layer_ops": ["FullyConnected"], "variants": 1,
                        "draws": {"count": 16, "seed": 0}}  # fmt: skip
    if p.kind == "add":
        return base | {"layer_ops": ["Add"], "variants": len(tpl.ADD_RANGES),
                        "draws": {"count": 16, "seed": 100, "ends": 2}}  # fmt: skip
    ops, _, _, _, _, gens = _graph(p.text, d)
    return base | {"layer_ops": list(ops), "variants": len(gens),
                    "draws": {"count": 18, "seed": 300}}  # fmt: skip


@functools.lru_cache(maxsize=None)
def _graph(name: str, d: tpl.Dims):
    """``pf_templates.graph`` (cached per name and dims: a fused graph's
    weights are large, e.g. ``alyrf<n>x<vocab>``'s)."""
    return tpl.graph(name, d)


def signature(name: str, d: tpl.Dims) -> dict:
    """The canonical signature of template ``name`` at dims ``d``."""
    p = parse(name)
    if p.kind == "linear":
        return {
            "schema": SCHEMA,
            "kind": "linear",
            "widths": {"K": p.n, "N": p.n2},
            "dims": {},
            "variant": {},
            "pulsar2": _pulsar2(p, d),
        }
    dims = {f: (float(getattr(d, f)) if f == "eps" else int(getattr(d, f)))
            for f in DEPS.get(p.kind, ())}  # fmt: skip
    variant = {}
    if p.kind == "rope":
        variant["qk_norm"] = False
    elif p.kind in QK_KINDS:
        variant["qk_norm"] = True
    if p.alt:
        variant["alt"] = True
    if p.kind in ROT_KINDS:
        variant["rot"] = True
    if p.kind in SPLIT_KINDS:
        variant["split"] = True
    widths = {"n": p.n} | ({"n2": p.n2} if p.n2 is not None else {})
    return {
        "schema": SCHEMA,
        "kind": p.kind,
        "widths": widths,
        "dims": dims,
        "variant": variant,
        "pulsar2": _pulsar2(p, d),
    }


def sig_id(sig: dict) -> str:
    text = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def name_sig(name: str, spec: ModelSpec) -> tuple[str, dict]:
    sig = signature(name, spec.dims())
    return sig_id(sig), sig


# ---- roots and status -----------------------------------------------------------------
@dataclass
class Root:
    """A template root; ``owner``: the spec whose dims built its templates
    (``None``: only the ``signature.json`` files count)."""

    path: str
    owner: ModelSpec | None = None


def scan(roots: list[Root]) -> dict[str, list[dict]]:
    """``sig_id`` -> the directories of that signature, over the roots."""
    found: dict[str, list[dict]] = {}
    for root in roots:
        for grp in GROUPS:
            base = os.path.join(root.path, grp)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                d = os.path.join(base, name)
                if not os.path.isdir(d):
                    continue
                try:
                    parse(name)
                except ValueError:
                    continue
                sf = os.path.join(d, SIG_FILE)
                if os.path.exists(sf):
                    sid = json.load(open(sf))["sig_id"]
                    how = "signature.json"
                elif root.owner is not None:
                    sid, _ = name_sig(name, root.owner)
                    how = f"owner {root.owner.name}"
                else:
                    continue
                found.setdefault(sid, []).append(
                    {"name": name, "dir": d, "root": root.path, "how": how}
                )
    return found


def load_manifest(path: str = MANIFEST) -> dict:
    if not os.path.exists(path):
        return {"schema": SCHEMA, "models": {}, "signatures": {}, "reuse": {}}
    return json.load(open(path))


def dump_manifest(man: dict, path: str = MANIFEST) -> None:
    with open(path, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")


@dataclass
class Need:
    name: str
    sig: str
    state: str  # exact | verified | check | missing
    dirs: list = field(default_factory=list)  # where the template is (exact / verified)
    candidates: list = field(default_factory=list)  # same name, other signature


def status(spec: ModelSpec, layout: int, found: dict, reuse: dict) -> list[Need]:
    """Each template key of ``spec`` at ``layout`` and its state (see above)."""
    by_name: dict[str, list[dict]] = {}
    for sid, dirs in found.items():
        for d in dirs:
            by_name.setdefault(d["name"], []).append(dict(d, sig=sid))
    out = []
    for name in keys_for(spec, layout):
        sid, _ = name_sig(name, spec)
        if sid in found:
            out.append(Need(name, sid, "exact", found[sid]))
            continue
        rec = reuse.get(sid)
        if rec and rec.get("verdict") == "equal" and rec.get("from") in found:
            out.append(Need(name, sid, "verified", found[rec["from"]]))
            continue
        cands = [c for c in by_name.get(name, []) if c["sig"] != sid]
        out.append(Need(name, sid, "check" if cands else "missing", [], cands))
    return out


def summary(needs: list[Need]) -> dict:
    out = {"keys": len(needs)}
    for s in ("exact", "verified", "check", "missing"):
        out[s] = sum(1 for n in needs if n.state == s)
    return out


# ---- build and check ------------------------------------------------------------------
def _write_sig(d: str, name: str, sid: str, sig: dict, spec: ModelSpec) -> None:
    with open(os.path.join(d, SIG_FILE), "w") as f:
        json.dump(
            {"schema": SCHEMA, "name": name, "sig_id": sid, "signature": sig,
             "built_for": spec.name},
            f, indent=1, sort_keys=True,
        )  # fmt: skip
        f.write("\n")


def compare_variants(fresh_root: str, name: str, cand_dir: str) -> dict:
    """``u16_retarget.compare_jobs`` of each variant ``v<i>`` of a fresh build
    against the candidate's: ``{"verdict": "equal" | "differ", "variants": [..]}``."""
    import graph_stitch as gs
    import u16_retarget as ur

    res = []
    for i in range(16):
        a = os.path.join(
            tpl.template_dir(fresh_root, name), f"v{i}", "out", "compiled.axmodel"
        )
        b = os.path.join(cand_dir, f"v{i}", "out", "compiled.axmodel")
        if not os.path.exists(b):
            break
        if not os.path.exists(a):
            res.append({"v": i, "diffs": ["fresh build missing"]})
            continue
        diffs = ur.compare_jobs(gs.load_model(a), gs.load_model(b))
        res.append({"v": i, "diffs": diffs})
    if not res:
        return {
            "verdict": "differ",
            "variants": [{"v": None, "diffs": ["no candidate build"]}],
        }
    ok = all(not r["diffs"] for r in res)
    return {"verdict": "equal" if ok else "differ", "variants": res}


def resolve(
    spec: ModelSpec,
    layout: int,
    out_root: str,
    roots: list[Root],
    jobs: int = 8,
    mem_gb: int = 20,
    manifest: dict | None = None,
    names: list[str] | None = None,
) -> dict:
    """Serve ``spec`` at ``layout``: build what is missing into ``out_root``,
    and for each ``check`` key build it fresh and compare it with its candidate
    (``compare_variants``); the verdict is written to ``manifest["reuse"]``.
    ``names``: only these keys. Returns the summary and the needs."""
    man = manifest if manifest is not None else load_manifest()
    found = scan(roots + [Root(out_root)])
    needs = status(spec, layout, found, man.get("reuse", {}))
    todo = [n for n in needs if n.state in ("missing", "check")]
    if names:
        todo = [n for n in todo if n.name in names]
    for n in todo:
        same = [d for d in found.get(n.sig, []) if d["root"] == out_root]
        if same:
            raise RuntimeError(
                f"{n.name}: {same[0]['dir']} already holds this signature"
            )
        if os.path.exists(tpl.template_dir(out_root, n.name)) and not os.path.exists(
            os.path.join(tpl.template_dir(out_root, n.name), SIG_FILE)
        ):
            raise RuntimeError(f"{n.name}: {tpl.template_dir(out_root, n.name)} exists")
    names_todo = [n.name for n in todo]
    d = spec.dims()
    if names_todo:
        bad = tpl.build_parallel(out_root, names_todo, d, jobs, mem_gb)
        if bad:
            raise RuntimeError(f"builds failed: {[(u.name, u.variant) for u in bad]}")
    report = []
    for n in todo:
        cdir = tpl.template_dir(out_root, n.name)
        _write_sig(cdir, n.name, n.sig, name_sig(n.name, spec)[1], spec)
        if n.state == "check":
            best = None
            for c in n.candidates:
                v = compare_variants(out_root, n.name, c["dir"])
                best = {"from": c["sig"], "from_dir": c["dir"], **v}
                if v["verdict"] == "equal":
                    break
            man.setdefault("reuse", {})[n.sig] = {
                "name": n.name,
                "model": spec.name,
                "from": best["from"],
                "from_dir": best["from_dir"],
                "verdict": best["verdict"],
                "variants": best["variants"],
            }
            report.append((n.name, best["verdict"]))
    return {
        "summary": summary(needs),
        "built": names_todo,
        "checked": report,
        "needs": needs,
    }


def manifest_for(specs=None, layouts=LAYOUTS) -> dict:
    """The repo manifest: the model specs, the signatures of every key of each
    model at ``layouts``, and the stored reuse records."""
    specs = specs or SPECS
    old = load_manifest()
    sigs = {}
    for spec in specs.values():
        for layout in layouts:
            for name in keys_for(spec, layout):
                sid, sig = name_sig(name, spec)
                sigs.setdefault(sid, {"name": name, "signature": sig})
    man = {
        "schema": SCHEMA,
        "models": {k: asdict(v) for k, v in specs.items()},
        "signatures": dict(sorted(sigs.items())),
        "reuse": old.get("reuse", {}),
    }
    return json.loads(json.dumps(man))  # tuples as lists: what a reload gives


def _roots_from(args) -> list[Root]:
    roots = []
    for r in args.root or []:
        path, _, model = r.partition("=")
        roots.append(Root(os.path.abspath(path), SPECS[model] if model else None))
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("keys", "report", "build"):
        p = sub.add_parser(c)
        p.add_argument("--model", required=True, choices=sorted(SPECS))
        p.add_argument("--layout", type=int, required=True)
        p.add_argument("--root", action="append", help="DIR[=MODEL] (repeatable)")
    sub.choices["build"].add_argument("--out", required=True)
    sub.choices["build"].add_argument("--jobs", type=int, default=8)
    sub.choices["build"].add_argument("--mem-gb", type=int, default=20)
    sub.choices["build"].add_argument("--name", action="append")
    sub.add_parser("manifest")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        man = manifest_for()
        dump_manifest(man)
        print(f"{MANIFEST}: {len(man['signatures'])} signatures")
        return
    spec = SPECS[a.model]
    if a.cmd == "keys":
        for n in keys_for(spec, a.layout):
            print(n, name_sig(n, spec)[0])
        return
    roots = _roots_from(a)
    if a.cmd == "report":
        needs = status(spec, a.layout, scan(roots), load_manifest().get("reuse", {}))
        print(json.dumps(summary(needs)))
        for n in needs:
            print(
                f"{n.state:9s} {n.name:24s} {n.sig} {n.dirs[0]['dir'] if n.dirs else ''}"
            )
        return
    out = os.path.abspath(a.out)
    man = load_manifest()
    res = resolve(spec, a.layout, out, roots, a.jobs, a.mem_gb, man, a.name)
    dump_manifest(man)
    print(
        json.dumps(
            {
                "summary": res["summary"],
                "built": res["built"],
                "checked": res["checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
