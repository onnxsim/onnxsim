"""Manifest trees of the split linears: a linear emitted as several programs.

Two kinds of entry hold sub-programs (every other entry is one program):

* ``split`` (the first set's ~16-bit weights, ``pf_generate.Generator.split``):
  ``split = {hi, lo, add}``; the programs ``<k>_hi``, ``<k>_lo``, ``<k>_add``.
  ``hi`` and ``lo`` are linears on the same input (``hi`` on W, ``lo`` on W - Q(W)).
* ``ksplit`` (a width absent at its N, ``pf_registry.split_keys``): ``parts``
  (linears on K-slices of the input, each on its own output grid, ``k_part`` rows
  each) and ``add`` (an Add requantizing the sum to the consumer's grid); the
  programs ``<k>_p<i>`` and ``<k>_add``.
* ``colsplit`` (an Add whose width is absent, the first set's gate | up sum): ``parts``
  (Adds of ``col[i]``-offset column slices, each of its own width) writing one output;
  the programs ``<k>_<i>``. Only as the ``add`` of a ``split``.

A part, ``hi`` or ``lo`` may itself be a split or a ksplit (the first set's down
projection at an absent width: each half is a ``ksplit``). Program names nest:
``<k>_hi_p0``, ``<k>_hi_add``. ``leaves`` lists the single programs in the order the
card runs them (the parts first, then the Add).
"""

from __future__ import annotations


def children(e: dict, k: str) -> list[tuple[str, dict]]:
    """The sub-programs of entry ``e`` (name ``k``), in run order; ``[]`` for a
    single program."""
    kind = e.get("kind")
    if kind == "split":
        if (
            "split" not in e
        ):  # a stub entry (the fake float set): the names by kind only
            return [
                (f"{k}_hi", {"kind": "linear"}),
                (f"{k}_lo", {"kind": "linear"}),
                (f"{k}_add", {"kind": "add"}),
            ]
        return [
            (f"{k}_hi", e["split"]["hi"]),
            (f"{k}_lo", e["split"]["lo"]),
            (f"{k}_add", e["split"]["add"]),
        ]
    if kind == "ksplit":
        return [(f"{k}_p{i}", p) for i, p in enumerate(e["parts"])] + [
            (f"{k}_add", e["add"])
        ]
    if kind == "colsplit":
        return [(f"{k}_{i}", p) for i, p in enumerate(e["parts"])]
    return []


def leaves(e: dict, k: str) -> list[tuple[str, dict]]:
    """Every single program under entry ``e`` (name ``k``): ``[(name, entry)]`` in
    run order. A single program is its own only leaf."""
    kids = children(e, k)
    if not kids:
        return [(k, e)]
    out = []
    for name, c in kids:
        out += leaves(c, name)
    return out


def _below(c: dict, name: str, pk: str) -> bool:
    return any(n == pk for n, _ in leaves(c, name))


def find(e: dict, k: str, pk: str) -> tuple[dict, list]:
    """The single-program entry ``pk`` under ``e`` (name ``k``), and the path of
    transforms from ``e``'s own program to it, outermost first. A step is
    ``("hi",)`` / ``("lo",)`` (a split's half), ``("part", i, k_part)`` (a K-slice
    of the input's rows) or ``("add",)``. ``KeyError`` when ``pk`` is not a leaf."""
    if not children(e, k):
        if pk == k:
            return e, []
        raise KeyError(f"{pk}: not a program of {k}")
    for name, c in children(e, k):
        if pk == name or _below(c, name, pk):
            sub, path = find(c, name, pk)
            return sub, [_step(e, name)] + path
    raise KeyError(f"{pk}: not a program of {k}")


def _step(e: dict, name: str) -> tuple:
    if e.get("kind") == "split":
        if name.endswith("_hi"):
            return ("hi",)
        if name.endswith("_lo"):
            return ("lo",)
        return ("add",)
    if e.get("kind") == "colsplit":
        return ("col", int(name.rsplit("_", 1)[1]))
    if name.endswith("_add"):
        return ("add",)
    return ("part", int(name.rsplit("_p", 1)[1]), e["k_part"])


def in_ksplit(e: dict, k: str, pk: str) -> bool:
    """Whether leaf ``pk`` lies under a ``ksplit`` entry (its parts and their Add)."""
    for name, c in children(e, k):
        if pk == name or _below(c, name, pk):
            if e.get("kind") == "ksplit":
                return True
            return in_ksplit(c, name, pk)
    return False


def leaf_weight(cfg, k: str, li, path: list):
    """The float weights a linear leaf (reached by ``path``, from ``find``) was
    emitted from. A split's ``hi`` is the merged weight W (the card's s8 rounding
    of W is the emitted weight); its ``lo`` is W - Q(W) (``Config.split_weights``).
    A part is the K-slice of its parent's rows. The split's halves use the float
    weights (``gptq=False``); a plain linear the merged (GPTQ) weights."""
    split = any(p[0] in ("hi", "lo") for p in path)
    w = cfg.merged_weight(k, li, gptq=not split)
    for step in path:
        if step[0] == "lo":
            w = cfg.split_weights(w)[1]
        elif step[0] == "part":
            _, i, kp = step
            w = w[i * kp : (i + 1) * kp]
    return w
