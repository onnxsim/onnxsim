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
    # the K-split kinds (``pf_templates.KSPLIT_BASE``): the down linear as two halves
    "lanormk": NORM,
    "slanormk": NORM,
    "slanormqkk": NORM + QK,
    "slanormqkrk": NORM + QK,
    "alyrqkrk": LAYER,
    "alyrfk": LAYER,
}
#: kinds that exist only with a per-head q / k norm (``rope``: without one)
QK_KINDS = {
    "qkrope", "hcrms", "qkf", "nrmqk", "nrmqkr", "slanormqk", "slanormqkr",
    "lyrqk", "lyrqkr", "alyrqk", "alyrqkr", "alyrqkrs", "alyrf",
    "slanormqkk", "slanormqkrk", "alyrqkrk", "alyrfk",
}  # fmt: skip
#: rotate_half made in the program (Split / Concat), not by two linears
ROT_KINDS = {"nrmqkr", "slanormqkr", "lyrqkr", "alyrqkr", "alyrqkrs",
             "slanormqkrk", "alyrqkrk"}  # fmt: skip
SPLIT_KINDS = {"alyrqkrs"}
#: the down linear split along K in two (``pf_templates.KSPLIT_BASE``; variant ``ksplit``)
KSPLIT_KINDS = {
    "lanormk",
    "slanormk",
    "slanormqkk",
    "slanormqkrk",
    "alyrqkrk",
    "alyrfk",
}

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
    #: the down linear (K = I) is absent at its N: the fused programs that hold it
    #: split it along K (``pf_templates.ksplit_name``); the shipped models: False
    ksplit_down: bool = False

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
        split_keys=("gu", "down_proj"), default_layout=8, ksplit_down=True,
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
        self.ksplit = bool(spec.ksplit_down)


def keys_for(spec: ModelSpec, layout: int, man: dict | None = None) -> list[str]:
    """The template names (``fused`` / ``add<n>`` / ``k<K>n<N>``) a model needs
    at ``layout``: ``pf_templates.needed`` for the spec (the same list, so the
    shipped sets' names are unchanged). With ``man`` (a manifest), a linear whose
    width is absent is replaced by its split parts and Add (``split_keys``)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout}: one of {LAYOUTS}")
    names = tpl.needed(_Cfg(spec, layout))
    return names if man is None else split_keys(names, man)[0]


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
    if p.kind in KSPLIT_KINDS:
        variant["ksplit"] = True
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


#: the shipped template roots, by directory name, and the model each was built
#: for. An unsigned directory whose real root has one of these names is that
#: model's (by name: a symlink into the shared root counts); any other unsigned
#: directory has no record and is not used.
SHIPPED_OWNERS = {"templates": "qwen3_0.6b", "templates_llama": "llama160m"}
#: the dims that decide whether a config.json is one of the manifest's models
DIMS_FIELDS = ("H", "I", "nh", "nkv", "hd", "eps", "vocab", "qk_norm")


def model_specs(man: dict | None = None) -> dict[str, ModelSpec]:
    """The model specs of ``template_registry.json`` (its ``models`` table)."""
    man = man if man is not None else load_manifest()
    out = {}
    for k, v in man.get("models", {}).items():
        v = dict(v, split_layers=tuple(v.get("split_layers", ())))
        v["split_keys"] = tuple(v.get("split_keys", ()))
        out[k] = ModelSpec(**v)
    return out


def model_spec(arg: str, man: dict | None = None) -> ModelSpec:
    """A model's spec from its name in ``template_registry.json``, or from a
    Hugging Face ``config.json`` path: the manifest's model with the same dims
    (``ValueError`` when no manifest model has them)."""
    specs = model_specs(man)
    if arg in specs:
        return specs[arg]
    if os.path.isfile(arg):
        c = ModelSpec.from_config("config", json.load(open(arg)))
        for s in specs.values():
            if all(getattr(s, f) == getattr(c, f) for f in DIMS_FIELDS):
                return s
        raise ValueError(
            f"{arg}: its dims (H={c.H} I={c.I} heads={c.nh}/{c.nkv} hd={c.hd} "
            f"layers={c.layers}) match no model of template_registry.json "
            f"({', '.join(sorted(specs))}); add it to pf_registry.SPECS and run "
            "pf_registry.py manifest"
        )
    raise ValueError(
        f"{arg}: neither a model of template_registry.json "
        f"({', '.join(sorted(specs))}) nor a config.json path"
    )


def shipped_owner(d: str, specs: dict | None = None) -> ModelSpec | None:
    """The model a shipped root built ``d`` for, when ``d`` (``<root>/<group>/
    <name>``) really lives in one (its symlink target, if it is one)."""
    real_root = os.path.dirname(os.path.dirname(os.path.realpath(d)))
    name = SHIPPED_OWNERS.get(os.path.basename(real_root))
    if name is None:
        return None
    return (specs or model_specs())[name]


def owner_of(root: "Root", d: str) -> ModelSpec | None:
    """The model whose dims a directory without a signature file was built
    with: the root's declared owner, else the shipped owner of its real root."""
    if root.owner is not None:
        return root.owner
    return shipped_owner(d)


@functools.lru_cache(maxsize=1)
def _default_manifest() -> dict:
    return load_manifest()


def measured_absent(name: str, man: dict | None = None) -> bool:
    """A linear ``k<K>n<N>`` whose width is measured absent at its N (the manifest's
    ``linear_widths``). Its build is no template, whatever its ``result.json`` says: a
    build made before the table check ran can say ok with the table missing."""
    m = _LINEAR.fullmatch(name)
    if not m:
        return False
    e = widths_entry(man if man is not None else _default_manifest(), int(m[2]))
    rec = e.get("measured", {}).get(m[1])
    return rec is not None and not rec.get("present")


def failed(d: str) -> bool:
    """A template whose build failed (``result.json`` ok false: Pulsar2, or the
    builder-side weight-table check of a linear), or a linear whose width is measured
    absent (``measured_absent``). Such a directory is no template."""
    rj = os.path.join(d, "result.json")
    if measured_absent(os.path.basename(os.path.normpath(d))):
        return True
    return os.path.exists(rj) and json.load(open(rj)).get("ok") is False


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
                if failed(d):
                    continue
                sf = os.path.join(d, SIG_FILE)
                owner = owner_of(root, d)
                if os.path.exists(sf):
                    sid = json.load(open(sf))["sig_id"]
                    how = "signature.json"
                elif owner is not None:
                    sid, _ = name_sig(name, owner)
                    how = f"owner {owner.name}"
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


def status(
    spec: ModelSpec, layout: int, found: dict, reuse: dict, man: dict | None = None
) -> list[Need]:
    """Each template key of ``spec`` at ``layout`` and its state (see above)."""
    by_name: dict[str, list[dict]] = {}
    for sid, dirs in found.items():
        for d in dirs:
            by_name.setdefault(d["name"], []).append(dict(d, sig=sid))
    out = []
    for name in keys_for(spec, layout, man):
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


# ---- linear widths (Pulsar2's weight-table limit) -----------------------------------
#: a linear's U16 weight table is absent from Pulsar2's npu_params at some input widths
#: (measured per output width N; ``linear_emit.linear_table_present``). Absence is not
#: monotone in K (K = 5456 absent, 5888 present at N = 2560), so a width is used only
#: if it was measured present at its N. The manifest's ``linear_widths[N].measured``
#: holds each K's verdict and its source. A K that is not measured present is split into
#: the smallest equal count of measured-present parts, the parts sharing one template and
#: one residual Add (``add<N>``, both inputs at the Add's scale).
_LINEAR = re.compile(r"k(\d+)n(\d+)")


def widths_entry(man: dict, n: int) -> dict:
    return man.get("linear_widths", {}).get(str(n), {})


def safe_widths(man: dict, n: int) -> list[int]:
    """The widths measured present at output width ``n``, ascending."""
    meas = widths_entry(man, n).get("measured", {})
    return sorted(int(k) for k, v in meas.items() if v.get("present"))


def linear_split(k: int, n: int, man: dict) -> dict | None:
    """``None`` when output width ``n`` has no measurements (nothing is planned) or
    ``k`` was measured present at ``n``. Otherwise the plan: ``parts`` (the smallest
    equal split of ``k`` whose parts are each measured present), ``k_part``, ``names``
    (one template name per part) and ``add``. A larger measured-present width never
    stands in for ``k``. ``ValueError`` when no equal split into measured widths exists."""
    meas = widths_entry(man, n).get("measured", {})
    safe = set(safe_widths(man, n))
    if not safe:
        return None
    if str(k) in meas and meas[str(k)].get("present"):
        return None
    for parts in range(2, k + 1):
        if k % parts == 0 and k // parts in safe:
            kp = k // parts
            return dict(parts=parts, k_part=kp, names=[f"k{kp}n{n}"] * parts,
                        add=f"add{n}")  # fmt: skip
    raise ValueError(
        f"k{k}n{n}: no equal split into measured-present widths {sorted(safe)}"
    )


#: a single-op Add's name (``add<n>``)
_ADD = re.compile(r"add(\d+)")


def widths_add(man: dict) -> dict:
    """The manifest's ``add_widths``: each Add width's ``measured`` verdict (card
    against the host model of the Add, ``pf_registry`` docs), like ``linear_widths``."""
    return man.get("add_widths", {}).get("measured", {})


def add_split(n: int, man: dict) -> dict | None:
    """``None`` when Add width ``n`` is not measured absent, else the plan: the smallest
    equal split of ``n`` into measured-present widths, ``parts``, ``width`` and the
    ``names`` (one Add per part, over its column slice). ``ValueError`` when none exists."""
    meas = widths_add(man)
    if str(n) not in meas or meas[str(n)].get("present"):
        return None
    safe = sorted(int(k) for k, v in meas.items() if v.get("present"))
    for parts in range(2, n + 1):
        if n % parts == 0 and n // parts in safe:
            w = n // parts
            return dict(parts=parts, width=w, names=[f"add{w}"] * parts)
    raise ValueError(f"add{n}: no equal split into measured-present widths {safe}")


def split_keys(names: list[str], man: dict) -> tuple[list[str], dict]:
    """``(names, plans)``: each linear name whose width is absent (``linear_split``
    is a plan) replaced, in place, by its parts' names and its Add; each Add name
    whose width is absent (``add_split``) by its column parts' names. ``plans`` maps
    each split name to its plan. Other names, and their order, are unchanged. The
    names are deduplicated in order (a part may already be a key of the list)."""
    out, plans = [], {}
    for name in names:
        m = _LINEAR.fullmatch(name)
        if m:
            p = linear_split(int(m[1]), int(m[2]), man)
            if p is None:
                out.append(name)
            else:
                plans[name] = p
                out += p["names"] + [p["add"]]
            continue
        a = _ADD.fullmatch(name)
        p = add_split(int(a[1]), man) if a else None
        if p is None:
            out.append(name)
        else:
            plans[name] = p
            out += p["names"]
    return list(dict.fromkeys(out)), plans


def split_report(spec: ModelSpec, layout: int, man: dict) -> list[str]:
    """One line per linear key of ``spec`` at ``layout``: unsplit, or its split plan."""
    out = []
    for name in keys_for(spec, layout):
        m = _LINEAR.fullmatch(name)
        if not m:
            continue
        p = linear_split(int(m[1]), int(m[2]), man)
        if p is None:
            out.append(f"{name}: unsplit")
        else:
            out.append(f"{name}: split {p['parts']} x {p['names'][0]} + {p['add']}")
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
    needs = status(spec, layout, found, man.get("reuse", {}), man)
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
            try:
                names = keys_for(spec, layout)
            except NotImplementedError:  # a layout the spec's K-split is not wired for
                continue
            for name in names:
                sid, sig = name_sig(name, spec)
                sigs.setdefault(sid, {"name": name, "signature": sig})
    man = {
        "schema": SCHEMA,
        "models": {k: asdict(v) for k, v in specs.items()},
        "signatures": dict(sorted(sigs.items())),
        "reuse": old.get("reuse", {}),
        "linear_widths": old.get("linear_widths", {}),
        "add_widths": old.get("add_widths", {}),
    }
    return json.loads(json.dumps(man))  # tuples as lists: what a reload gives


# ---- resolution for pf_generate ----------------------------------------------------------
@dataclass
class Choice:
    """How one template key is served: ``how`` is ``exact`` (a directory with a
    signature file of the same signature), ``by name`` (a shared root's
    directory of that name, built for an owner whose signature is the same),
    ``reuse`` (a stored equal verdict against a directory that exists) or
    ``missing`` (``reason`` says what is there instead)."""

    name: str
    sig: str
    how: str
    dir: str | None = None
    reason: str = ""


def plan(
    spec: ModelSpec,
    layout: int,
    roots: list[Root],
    reuse: dict | None = None,
    verify=None,
    names: list[str] | None = None,
) -> list[Choice]:
    """Each template key of ``spec`` at ``layout``, resolved in this order:

    1. exact: a signature file with the key's signature;
    2. by name: a directory of the key's name in a shared root, built for an
       owner whose dims give the key's signature;
    3. reuse: a stored ``equal`` record of the key's signature, whose source
       directory (the record's ``from``) exists in the roots.

    Anything else is ``missing``: a directory of the name with another
    signature, or without a record, is never used in its place. ``verify``
    (optional, ``verify(dir, name) -> error or None``): an exact directory whose
    check fails is ``missing`` too, with the error as the reason."""
    reuse = reuse or {}
    owner_sids: dict = {}
    names = keys_for(spec, layout) if names is None else names

    def entries(name: str):
        for root in roots:
            for grp in GROUPS:
                d = os.path.join(root.path, grp, name)
                if not os.path.isdir(d) or failed(d):
                    continue
                sf = os.path.join(d, SIG_FILE)
                if os.path.exists(sf):
                    sid = json.load(open(sf))["sig_id"]
                    yield {"dir": d, "sid": sid, "how": "exact", "owner": None}
                    continue
                owner = owner_of(root, d)
                if owner is None:
                    yield {"dir": d, "sid": None, "how": "unsigned", "owner": None}
                    continue
                key = (name, owner.name)
                if key not in owner_sids:
                    owner_sids[key] = name_sig(name, owner)[0]
                yield {"dir": d, "sid": owner_sids[key], "how": "by name",
                       "owner": owner.name}  # fmt: skip

    out = []
    for name in names:
        sid, _ = name_sig(name, spec)
        ents = list(entries(name))
        hit = [e for e in ents if e["how"] == "exact" and e["sid"] == sid]
        if hit:
            err = verify(hit[0]["dir"], name) if verify else None
            if err is None:
                out.append(Choice(name, sid, "exact", hit[0]["dir"]))
            else:
                out.append(Choice(name, sid, "missing", None,
                    f"{hit[0]['dir']} has the signature but fails its weight-table check: {err}"))  # fmt: skip
            continue
        hit = [e for e in ents if e["how"] == "by name" and e["sid"] == sid]
        if hit:
            out.append(Choice(name, sid, "by name", hit[0]["dir"],
                              f"built for {hit[0]['owner']}"))  # fmt: skip
            continue
        reason = ""
        rec = reuse.get(sid)
        if rec and rec.get("verdict") == "equal":
            src = [e for e in entries(rec.get("name", name)) if e["sid"] == rec["from"]]
            if src:
                out.append(Choice(name, sid, "reuse", src[0]["dir"],
                                  f"equal to signature {rec['from']}"))  # fmt: skip
                continue
            reason = (
                f"reuse record {rec['from']}: its source directory is not in the roots"
            )
        elif rec:
            reason = f"reuse record says {rec.get('verdict')}"
        others = [e for e in ents if e["sid"] != sid]
        if others:
            o = others[0]
            what = "no signature" if o["sid"] is None else f"signature {o['sid']}"
            reason = "; ".join(
                x for x in (reason, f"{o['dir']} has {what}, not {sid}") if x
            )
        out.append(
            Choice(name, sid, "missing", None, reason or "no directory of this name")
        )
    return out


def build_command(spec: ModelSpec, layout: int, out: str, owners=()) -> str:
    """The ``pf_registry.py build`` line that makes the missing keys of ``spec``
    at ``layout`` in ``out`` (an overlay: shared roots are not written)."""
    parts = [
        "python3", os.path.abspath(__file__), "build",
        "--model", spec.name, "--layout", str(layout), "--out", out,
    ]  # fmt: skip
    for r in owners:
        parts += ["--root", f"{r.path}={r.owner.name}"]
    return " ".join(parts)


def _roots_from(args) -> list[Root]:
    roots = []
    for r in args.root or []:
        path, _, model = r.partition("=")
        roots.append(Root(os.path.abspath(path), model_spec(model) if model else None))
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("keys", "report", "build"):
        p = sub.add_parser(c)
        p.add_argument(
            "--model",
            required=True,
            help="a model of template_registry.json, or a config.json path",
        )
        p.add_argument("--layout", type=int, required=True)
        p.add_argument("--root", action="append", help="DIR[=MODEL] (repeatable)")
    sub.choices["build"].add_argument("--out", required=True)
    sub.choices["build"].add_argument("--jobs", type=int, default=8)
    sub.choices["build"].add_argument("--mem-gb", type=int, default=20)
    sub.choices["build"].add_argument("--name", action="append")
    sub.add_parser("manifest")
    p = sub.add_parser("split")
    p.add_argument("--model", required=True)
    p.add_argument("--layout", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        man = manifest_for()
        dump_manifest(man)
        print(f"{MANIFEST}: {len(man['signatures'])} signatures")
        return
    spec = model_spec(a.model)
    if a.cmd == "split":
        print("\n".join(split_report(spec, a.layout, load_manifest())))
        return
    if a.cmd == "keys":
        for n in keys_for(spec, a.layout, load_manifest()):
            print(n, name_sig(n, spec)[0])
        return
    roots = _roots_from(a)
    if a.cmd == "report":
        man = load_manifest()
        ch = plan(
            spec, a.layout, roots, man.get("reuse", {}),
            names=keys_for(spec, a.layout, man),
        )  # fmt: skip
        counts = {
            h: sum(c.how == h for c in ch)
            for h in ("exact", "by name", "reuse", "missing")
        }
        print(json.dumps({"keys": len(ch), **counts}))
        for c in ch:
            print(f"{c.how:9s} {c.name:24s} {c.sig} {c.dir or ''} {c.reason}".rstrip())
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
