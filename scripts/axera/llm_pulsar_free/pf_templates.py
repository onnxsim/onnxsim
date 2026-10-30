"""Native U16 template builds (Pulsar2) for every program shape of a model,
each at four calibration variants, for ``pf_generate.py`` to retarget. This
is the only step that runs Pulsar2 (``pulsar2_docker``, image
``pulsar2:7.0-lite``), once per shape; the programs themselves are emitted
without it.

Template directory (``--root``, default ``$LLM_PF_TEMPLATES`` or
``<LLM_PF_OUT>/../templates``):

* ``fused/<name>/v<i>``: fused multi-op programs
  (``layer_configs: [{op_types: [...], data_type: U16}]``);
* ``single/add<n>/v<i>``: the single-op residual Add;
* ``linear/k<K>n<N>``: a ``[1, K] @ [K, N]`` FullyConnected at U16
  activations (weights ``default_rng(1).uniform(-0.1, 0.1)``).

Fused graphs (``graph(name, dims)``; the generated programs use the same
graph, so ``FusedEval`` / ``float_graph`` evaluate exactly what the card runs):

* ``crms<n>``: coarse norm ``u = x / sqrt(mean(x^2) + eps)`` (first pass of
  the two-pass RMSNorm), ``rms<n>``: RMSNorm with the gain ``g`` as an input;
* ``rope<n>``: ``y = q * c + r * sn`` (``r`` = rotate_half(q), from the merged
  linear's ``q_rot`` / ``k_rot`` columns);
* ``qkrope<n>`` (Qwen3): per-head q / k RMSNorm fused with RoPE,
  ``y = (q*c + r*sn) / sqrt(mean_head(q^2) + eps)`` with the norm gain folded
  into the host's tables; ``qkrope<n>t`` the same graph with calibrations whose
  two Adds shift differently (an elided ``0x1ea0`` write cannot be restored);
* ``hcrms<n>`` (Qwen3): per-head coarse norm of ``[k | k_rot]``;
* ``attn<n>``: the whole layer's attention, ``q [1, nh*hd]`` as
  ``[nkv, G, hd]``, token-major K / V ``[S + 1, nkv*hd]`` (the plan's cache
  layout), mask ``[1, S + 1]`` (Reshape / Transpose inside);
* ``silu<n>``: ``y = gate * sigmoid(gate) * up``;
* ``nrmqk<H>`` / ``slanormqk<I>`` (``fuse=5``): ``nrm`` / ``slanorm``, then
  the five q / k / v linears from the norm's output and ``qkf`` (outputs
  ``yq``, ``yk``, ``v``); a variant generator may come with its own weights
  (``(gen, {initializer: weights})``); ``nrmqkr<H>`` / ``slanormqkr<I>``
  (``fuse=6``) the same with three linears q, k, v, rotate_half of q and k
  made in the program by Split / Concat (``ROT``);
* merged programs (``pf_layout.LAYER_FUSED*``), several outputs where the
  residual stream continues: ``nrm<n>`` (the two-pass RMSNorm), ``anorm<n>``
  (residual Add + two-pass norm, outputs ``x`` and ``y``), ``qkf<nh*hd>``
  (q RoPE + k coarse norm + k RoPE, outputs ``yq`` and ``yk``),
  ``lanorm<k>`` (a ``[1, k] @ [k, H]`` FullyConnected + ``anorm``),
  ``slanorm<k>`` (fused SiLU + ``lanorm``); ``nrml<n>``, ``lanorml<k>x<n>``,
  ``slanorml<k>x<n>``: the same with a second linear after the norm
  (``y = yn @ w2``, ``[1, H] @ [H, n]``; ``fuse=4``);
  ``alanorm<k>`` / ``alanorml<k>x<n>`` (``fuse=7``): ``attn<k>`` whose output
  is the linear's input ``z`` of ``lanorm<k>`` / ``lanorml<k>x<n>``.
* ``lyrqk<nh*hd>`` / ``lyrqkr<nh*hd>``: ``lanorm<nh*hd>`` (its tensors
  suffixed ``_1``, the attention output ``att``), the gate / up linears from
  its output, then ``slanormqk<I>`` / ``slanormqkr<I>`` on them, with the SiLU
  written ``Mul(sigmoid(gate), gate)``: Pulsar2 fuses ``Mul(gate,
  sigmoid(gate))`` after a FullyConnected into one onnx ``Silu`` op
  (``u16_retarget.fuse_silu``) whose table is ~5x less accurate than the
  Sigmoid table and the Mul; ``alyrqk<n>`` / ``alyrqkr<n>`` (``fuse=8``:
  ``alyrqkr``, the whole layer) with ``alanorm<n>`` (the attention's tensors
  suffixed ``_a``) in place of ``lanorm<n>``; ``alyrf<n>`` (``fuse=9``, the
  last layer) with ``slanorm<I>`` (the final norm) in place of
  ``slanormqkr<I>``: outputs ``x`` and ``y``, no RoPE tables.

usage: MODEL=llama160m pf_templates.py [--root DIR] [--list] [--jobs N]
       [--mem-gb G] [NAME...]
(no NAME: every template the model needs that is not built yet.) The builds
(one per variant, one per linear) run ``--jobs`` at once, the longest (the
lm_head linear) first, each Docker container capped at ``Unit.gb`` with the
caps summing to at most ``--mem-gb``; ``--jobs 1``: one at a time, uncapped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx.parser import parse_model

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

F = np.float32
OPSET = 'ir_version: 9, opset_import: ["" : 17]'
IMAGE = "pulsar2:7.0-lite"


@dataclass(frozen=True)
class Dims:
    H: int
    I: int  # noqa: E741
    nh: int
    nkv: int
    hd: int
    eps: float
    vocab: int
    S: int = 128

    @classmethod
    def of(cls, cfg) -> "Dims":
        return cls(cfg.H, cfg.I, cfg.nh, cfg.nkv, cfg.hd, cfg.eps, cfg.vocab, cfg.S)


CRMS = """
  sq = Mul(x, x)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  y = Div(x, r)
"""
RMS = """
  sq = Mul(x, x)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  d = Div(x, r)
  y = Mul(d, g)
"""
HCRMS = """
  xh = Reshape(x, shh)
  sq = Mul(xh, xh)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  uh = Div(xh, r)
  y = Reshape(uh, shy)
"""
ROPE = """
  qa = Mul(q, c)
  qb = Mul(r, sn)
  y = Add(qa, qb)
"""
QKROPE = """
  qa = Mul(q, c)
  qb = Mul(r, sn)
  t = Add(qa, qb)
  qh = Reshape(q, shh)
  sq = Mul(qh, qh)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  rr = Sqrt(e)
  th = Reshape(t, shh)
  yh = Div(th, rr)
  y = Reshape(yh, shy)
"""
SILU = """
  sg = Sigmoid(gate)
  sl = Mul(gate, sg)
  y = Mul(sl, up)
"""
ATTN = """
  qh = Reshape(q, shq)
  kh = Reshape(k, shk)
  kt = Transpose<perm=[1,2,0]>(kh)
  s = MatMul(qh, kt)
  sm = Add(s, mask)
  p = Softmax<axis=-1>(sm)
  vh = Reshape(v, shk)
  vt = Transpose<perm=[1,0,2]>(vh)
  o = MatMul(p, vt)
  y = Reshape(o, sho)
"""
NORM_OPS = ["Mul", "ReduceMean", "Add", "Sqrt", "Div"]

# ---- merged programs (one execute each instead of two or three) ----------------------
#: (one eps initializer per Add: Pulsar2 renames a shared one per use)
#: the two-pass RMSNorm as one program: coarse pass to ``u``, then the RMSNorm of ``u``
NORM2_C = """
  sq0 = Mul(x, x)
  m0 = ReduceMean<axes=[-1], keepdims=1>(sq0)
  e0 = Add(m0, eps0)
  r0 = Sqrt(e0)
  u = Div(x, r0)
"""
NORM2_R = """
  sq = Mul(u, u)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  d = Div(u, r)
  y = Mul(d, g)
"""
NORM2 = NORM2_C + NORM2_R
#: the row-scaled coarse pass (``pf_rsnorm_probe``): ``xs = x / max|x|`` before the
#: squares, so one row's outlier cannot set the range of ``x * x``; ``u`` is
#: ``x / rms(x)`` up to eps (the scale cancels). Replaces ``NORM2_C`` in the ``rs``
#: families (``rsslanorm``, ``rsslanormqkr``, ``alyrrs*``); the second pass is unchanged
NORM2_CRS = """
  ax = Abs(x)
  mx = ReduceMax<axes=[-1], keepdims=1>(ax)
  xs = Div(x, mx)
  sq0 = Mul(xs, xs)
  m0 = ReduceMean<axes=[-1], keepdims=1>(sq0)
  e0 = Add(m0, eps0)
  r0 = Sqrt(e0)
  u = Div(xs, r0)
"""
#: the ops the row-scaled coarse pass adds to a graph's layer ops
RS_OPS = ("Abs", "ReduceMax")


def _rs_body(body: str) -> str:
    """``body`` with its coarse norm row-scaled (``NORM2_C`` -> ``NORM2_CRS``)."""
    if body.count(NORM2_C) != 1:
        raise ValueError(f"rs: {body.count(NORM2_C)} coarse norms, one expected")
    return body.replace(NORM2_C, NORM2_CRS)


#: residual Add, then the two-pass RMSNorm of its sum; outputs the sum ``x`` (the
#: residual stream) and the normalized ``y``
ADDNORM = "  x = Add(a, b)\n" + NORM2
#: a linear (``z @ w``, a FullyConnected) whose output is the residual Add's
#: second operand, then the two-pass norm: ``o_proj`` + ``res1`` + the second
#: norm, or ``down_proj`` + ``res2`` + the next layer's first norm
LINEAR_ADDNORM = "  b = MatMul(z, w)\n" + ADDNORM
#: ``fuse=3``: the fused SiLU before ``down_proj`` too
SILU_LINEAR_ADDNORM = (
    "  sg = Sigmoid(gate)\n  sl = Mul(gate, sg)\n  z = Mul(sl, up)\n" + LINEAR_ADDNORM
)
#: Qwen3's q and k RoPE programs and the coarse k norm as one program: q's
#: ``qkrope``; k's per-head coarse norm ``uk = k / r0``, ``ur = k_rot / r0``
#: (``rot`` permutes within a head, so k_rot's head RMS is k's); k's ``qkrope``
#: of ``uk``. Outputs ``yq`` and ``yk``.
QKF_Q = """
  qa = Mul(q, cq)
  qb = Mul(qr, sq)
  qt = Add(qa, qb)
  qh = Reshape(q, shq)
  qsq = Mul(qh, qh)
  qm = ReduceMean<axes=[-1], keepdims=1>(qsq)
  qe = Add(qm, eps_q)
  qrr = Sqrt(qe)
  qth = Reshape(qt, shq)
  yqh = Div(qth, qrr)
  yq = Reshape(yqh, shyq)
"""
QKF_KC = """
  kh = Reshape(k, shk)
  ksq = Mul(kh, kh)
  km = ReduceMean<axes=[-1], keepdims=1>(ksq)
  ke = Add(km, eps_k)
  kr0 = Sqrt(ke)
  ukh = Div(kh, kr0)
  krh = Reshape(kr, shk)
  urh = Div(krh, kr0)
  uk = Reshape(ukh, shyk)
  ur = Reshape(urh, shyk)
"""
QKF_K = """
  ka = Mul(uk, ck)
  kb = Mul(ur, sk)
  kt = Add(ka, kb)
  ush = Reshape(uk, shk)
  usq = Mul(ush, ush)
  um = ReduceMean<axes=[-1], keepdims=1>(usq)
  ue = Add(um, eps_u)
  urr = Sqrt(ue)
  kth = Reshape(kt, shk)
  ykh = Div(kth, urr)
  yk = Reshape(ykh, shyk)
"""
QKF = QKF_Q + QKF_KC + QKF_K
#: rotate_half's permutation of ``{t}`` per head, in the program (``fuse=6``):
#: ``[nh, 2, hd / 2]``, the halves swapped (the sign is in the sin tables). Pulsar2
#: runs Split / Concat as data movement, without a job, calibrated as ``{t}``
ROT = """
  {t}3 = Reshape({t}, sh{t}3)
  {t}1, {t}2 = Split<axis=1>({t}3)
  {t}r3 = Concat<axis=1>({t}2, {t}1)
  {t}r = Reshape({t}r3, {sh})
"""


# ---- calibration generators of the four variants (Pulsar2 MinMax over 18 draws) ------
def _norm_gen(n, xs, off, g=None):
    def gen(r):
        d = {"x": (r.standard_normal((1, n)) * xs + off).astype(F)}
        if g is not None:
            d["g"] = r.uniform(g[0], g[1], (1, n)).astype(F)
        return d

    return gen


def _rope_gen(n, qs, qoff, cs):
    def gen(r):
        return {
            "q": (r.standard_normal((1, n)) * qs + qoff).astype(F),
            "r": (r.standard_normal((1, n)) * qs - qoff).astype(F),
            "c": r.uniform(-0.6 * cs, cs, (1, n)).astype(F),
            "sn": r.uniform(-cs, 0.8 * cs, (1, n)).astype(F),
        }

    return gen


def _rope_shift_gen(n, qs, cs, k):
    """r ~ -q and sn ~ c: q*c + r*sn mostly cancels, so its Add's scale ratios
    reach past 1 (shift < 15) while the eps Add stays at 15."""

    def gen(r):
        q = (r.standard_normal((1, n)) * qs).astype(F)
        c = r.uniform(-0.6 * cs, cs, (1, n)).astype(
            F
        )  # straddles 0: nonzero zero points
        return {
            "q": q,
            "r": (-q * k + r.standard_normal((1, n)) * qs * 0.05).astype(F),
            "c": c,
            "sn": (c * r.uniform(0.9, 1.1, (1, n))).astype(F),
        }

    return gen


def _silu_gen(n, gs, goff, us, uoff):
    def gen(r):
        return {
            "gate": (r.standard_normal((1, n)) * gs + goff).astype(F),
            "up": (r.standard_normal((1, n)) * us + uoff).astype(F),
        }

    return gen


def _attn_gen(d: Dims, qs, ks, vs, voff):
    def gen(r):
        valid = int(r.integers(1, d.S))
        m = np.zeros((1, d.S + 1), F)
        m[0, valid:] = -32.0
        return {
            "q": (r.standard_normal((1, d.nh * d.hd)) * qs).astype(F),
            "k": (r.standard_normal((d.S + 1, d.nkv * d.hd)) * ks).astype(F),
            "v": (r.standard_normal((d.S + 1, d.nkv * d.hd)) * vs + voff).astype(F),
            "mask": m,
        }

    return gen


def _addnorm_gen(n, xs, off, bk, bs, g):
    """``b = -bk * a + noise``: with ``bk`` near 1 the sum is smaller than ``a``,
    so the residual Add's scale ratio passes 1 (shift 14)."""

    def gen(r):
        a = (r.standard_normal((1, n)) * xs + off).astype(F)
        return {
            "a": a,
            "b": (-bk * a + r.standard_normal((1, n)) * bs).astype(F),
            "g": r.uniform(g[0], g[1], (1, n)).astype(F),
        }

    return gen


def _lanorm_gen(k, n, w, zs, aoff, bk, g):
    """``a = -bk * (z @ w) + noise``: with ``bk`` near 1 the residual Add cancels
    (its shift drops below 15)."""

    def gen(r):
        z = (r.standard_normal((1, k)) * zs + 0.1 * zs).astype(F)
        b = z @ w
        sb = float(np.abs(b).std())
        return {
            "z": z,
            "a": (
                -bk * b + r.standard_normal((1, n)) * sb * (1.5 - bk) + aoff * sb
            ).astype(F),
            "g": r.uniform(g[0], g[1], (1, n)).astype(F),
        }

    return gen


def attn_ref(d: Dims, q, k, v, mask) -> np.ndarray:
    """``ATTN`` in numpy: ``[1, nh*hd]`` (GQA, token-major K / V)."""
    G = d.nh // d.nkv
    qh = q.reshape(d.nkv, G, d.hd)
    kt = k.reshape(d.S + 1, d.nkv, d.hd).transpose(1, 2, 0)
    s = qh @ kt + mask
    e = np.exp(s - s.max(-1, keepdims=True))
    vt = v.reshape(d.S + 1, d.nkv, d.hd).transpose(1, 0, 2)
    return ((e / e.sum(-1, keepdims=True)) @ vt).reshape(1, -1).astype(F)


def _alanorm_gen(d: Dims, w, ap, bk, aoff, g):
    """The attention's draws (``_attn_gen(*ap)``), its output ``z`` the
    linear's input; ``a`` cancels ``z @ w`` by ``bk`` as in ``_lanorm_gen``."""
    ag = _attn_gen(d, *ap)

    def gen(r):
        f = ag(r)
        b = attn_ref(d, f["q"], f["k"], f["v"], f["mask"]) @ w
        sb = float(np.abs(b).std())
        n = b.shape[1]
        f["a"] = (
            -bk * b + r.standard_normal((1, n)) * sb * (1.5 - bk) + aoff * sb
        ).astype(F)
        f["g"] = r.uniform(g[0], g[1], (1, n)).astype(F)
        return f

    return gen


def _slanorm_gen(k, n, w, gs_, us, aoff, bk, g):
    def gen(r):
        gate = (r.standard_normal((1, k)) * gs_ - 0.2).astype(F)
        up = (r.standard_normal((1, k)) * us + 0.05).astype(F)
        z = gate / (1 + np.exp(-gate)) * up
        b = z @ w
        sb = float(np.abs(b).std())
        return {
            "gate": gate,
            "up": up,
            "a": (
                -bk * b + r.standard_normal((1, n)) * sb * (1.5 - bk) + aoff * sb
            ).astype(F),
            "g": r.uniform(g[0], g[1], (1, n)).astype(F),
        }

    return gen


def template_weight(k: int, n: int, seed: int = 1) -> np.ndarray:
    """A linear template's weights (``default_rng(seed).uniform(-0.1, 0.1)``)."""
    return np.random.default_rng(seed).uniform(-0.1, 0.1, (k, n)).astype(F)


def _qkf_gen(d: Dims, qs, ks, cs, kq, kk):
    """``kq`` / ``kk`` > 0: q's / k's RoPE Add cancels (rotated part ~ -k x the
    plain part, sin ~ cos), so that Add shifts 14 while the eps Adds stay at 15."""
    nq, nk = d.nh * d.hd, d.nkv * d.hd

    def side(r, n, s, k):
        x = (r.standard_normal((1, n)) * s + 0.1 * s).astype(F)
        c = r.uniform(-0.6 * cs, cs, (1, n)).astype(F)
        if k:
            xr = -x * k + r.standard_normal((1, n)).astype(F) * s * 0.05
            sn = c * r.uniform(0.9, 1.1, (1, n))
        else:
            xr = r.standard_normal((1, n)) * s - 0.1 * s
            sn = r.uniform(-cs, 0.8 * cs, (1, n))
        return x, xr.astype(F), c, sn.astype(F)

    def gen(r):
        q, qr, cq, sq = side(r, nq, qs, kq)
        k, kr, ck, sk = side(r, nk, ks, kk)
        return dict(q=q, qr=qr, k=k, kr=kr, cq=cq, sq=sq, ck=ck, sk=sk)

    return gen


NORM_X = ((1.0, 0.1), (3.0, -0.4), (0.5, 0.2), (8.0, 1.0))
NORM_G = ((-0.6, 0.6), (-0.3, 1.2), (0.1, 2.0), (-2.0, 0.5))
#: the single-op Add's input ranges per variant
ADD_RANGES = [
    {"a": (-2.0, 3.0), "b": (-1.0, 1.0)},
    {"a": (-1.0, 1.0), "b": (-0.5, 4.0)},
    {"a": (-3.0, 0.5), "b": (-2.0, 3.0)},
    {"a": (-8.0, 6.0), "b": (-1.0, 0.5)},
]


#: a fused program whose down linear (K = I) is absent from Pulsar2's npu_params
#: (``linear_emit.table_in_params``; K = 9728 at N = 2560) splits it along K into two
#: linears of K / 2 and an Add, both on one output grid (``pf_generate.ksplit_grids``):
#: ``zk0, zk1`` are the input's halves (``Split``, data movement, calibrated as ``z``),
#: ``w`` and ``wl`` the parts' weights. The Add is the exact integer sum (both ratios 1).
KSPLIT_DOWN = """
  zk0, zk1 = Split<axis=1>(z)
  bh = MatMul(zk0, w)
  bl = MatMul(zk1, wl)
  b = Add(bh, bl)
"""
#: the K-split name (``ksplit_name``) of each kind whose down linear is K = I, and its
#: unsplit kind (``graph``: ``<kind>k<n>`` is ``<kind><n>`` with the down linear split)
KSPLIT_BASE = {
    "alyrqkrk": "alyrqkr",
    "alyrfk": "alyrf",
    "lanormk": "lanorm",
    "slanormk": "slanorm",
    "slanormqkk": "slanormqk",
    "slanormqkrk": "slanormqkr",
    "alyrrsqkrk": "alyrrsqkr",
    "alyrrsfk": "alyrrsf",
}
#: the kinds of ``KSPLIT_BASE`` (``pf_registry``: the K-split kinds)
KSPLIT_KINDS = frozenset(KSPLIT_BASE)
_KSPLIT_OF = {v: k for k, v in KSPLIT_BASE.items()}


def ksplit_name(name: str, inter: int) -> str:
    """The K-split name of template ``name`` when the down linear's K is ``inter`` (I):
    ``alyrqkr<n>`` and ``alyrf<n>`` always (their down linear is the layer's), the
    ``lanorm`` / ``slanorm`` / ``slanormqk(r)`` kinds of width ``inter`` only (their
    width is the linear's K). Any other name is returned unchanged."""
    m = re.fullmatch(
        r"(alyrqkr|alyrf|alyrrsqkr|alyrrsf|lanorm|slanormqkr|slanormqk|slanorm)(\d+)",
        name,
    )
    if not m:
        return name
    kind, n = m.group(1), int(m.group(2))
    if not kind.startswith("alyr") and n != inter:
        return name
    return f"{_KSPLIT_OF[kind]}{n}"


def ksplit_graph(base, d: Dims):
    """``graph`` of a K-split kind: the unsplit graph ``base`` (``(ops, body, ins, outs,
    inits, gens)``) with its down linear ``b = MatMul(z, w)`` as ``KSPLIT_DOWN``. The
    parts' weights are the halves of the unsplit ``w`` (the generators keep the whole
    ``w``: their ``b`` is the sum of the halves)."""
    ops, body, ins, outs, inits, gens = base
    old = "  b = MatMul(z, w)\n"
    if body.count(old) != 1:
        raise ValueError(f"ksplit: {body.count(old)} down linears, one expected")
    w = inits["w"]
    if w.shape[0] != d.I or w.shape[0] % 2:
        raise ValueError(f"ksplit: the down linear's K {w.shape[0]} is not I = {d.I}")
    kp = w.shape[0] // 2
    inits = {**inits, "w": w[:kp].copy(), "wl": w[kp:].copy()}
    ops = list(dict.fromkeys([*ops, "Split"]))
    return ops, body.replace(old, KSPLIT_DOWN), ins, outs, inits, gens


def graph(name: str, d: Dims):
    """``(op types, body, inputs {name: shape}, output shape, initializers,
    variant generators)`` of a fused template ``name``."""
    mr = re.fullmatch(r"rs(slanorm|slanormqkr)(\d+)", name)
    if mr:  # a tail (``slanorm`` / ``slanormqkr``) with the row-scaled norm
        ops, body, ins, outs, inits, gens = graph(mr[1] + mr[2], d)
        return [*dict.fromkeys([*ops, *RS_OPS])], _rs_body(body), ins, outs, inits, gens
    mr = re.fullmatch(r"alyrrs(qkr|f)(\d+)", name)
    if mr:  # the whole layer with every norm row-scaled (``rs_name``)
        n = int(mr[2])
        if mr[1] == "qkr":
            return _lyr_graph(n, d, rot=True, attn=True, rs=True)
        return _lyr_graph(n, d, attn=True, rs=True, tail=f"slanorm{d.I}")
    mk = re.fullmatch(
        r"(alyrqkrk|alyrfk|lanormk|slanormk|slanormqkk|slanormqkrk"
        r"|alyrrsqkrk|alyrrsfk)(\d+)",
        name,
    )
    if mk and mk.group(1) in (
        "alyrqkrk",
        "alyrrsqkrk",
    ):  # its variants as ``alyrqkr``'s split ones
        return _lyr_graph(
            int(mk.group(2)), d, rot=True, attn=True, ksplit=True,
            rs=mk.group(1) == "alyrrsqkrk",
        )  # fmt: skip
    if mk:  # the unsplit graph of the same width, its down linear split
        base = KSPLIT_BASE[mk.group(1)] + mk.group(2)
        return ksplit_graph(graph(base, d), d)
    m = re.fullmatch(
        r"(crms|rms|hcrms|rope|qkrope|silu|attn|nrm|anorm|qkf|lanorm|slanorm"
        r"|nrml|lanorml|slanorml|nrmqkr|slanormqkr|nrmqk|slanormqk|alanorml?"
        r"|alyrqkrs|a?lyrqkr?|alyrf)"
        r"(\d+)(?:x(\d+))?(t?)",
        name,
    )
    if not m:
        raise ValueError(f"{name}: not a fused template name")
    kind, n, alt = m.group(1), int(m.group(2)), m.group(4)
    n2 = int(m.group(3)) if m.group(3) else None
    if kind in ("alanorm", "alanorml"):
        # ``fuse=7``: the attention (``attn<n>``), its output ``z`` the input
        # of ``lanorm<n>`` (``alanorml<n>x<n2>``: of ``lanorml<n>x<n2>``)
        aops, abody, ains, _, ainits, _ = graph(f"attn{n}", d)
        ops, body, ins, outs, inits, _ = graph(
            kind[1:] + str(n) + (f"x{n2}" if n2 else ""), d
        )
        abody = abody.replace("  y = Reshape(o, sho)", "  z = Reshape(o, sho)")
        ins = {**ains, **{k: v for k, v in ins.items() if k != "z"}}
        ap = (
            (0.3, 2.0, 0.5, 0.05),
            (0.15, 3.0, 1.0, -0.1),
            (0.5, 1.0, 0.3, 0.02),
            (0.2, 4.0, 0.8, 0.1),
        )
        p = ((0.0, 0.0, NORM_G[0]), (0.9, 0.1, NORM_G[1]))
        p += ((0.3, 1.0, NORM_G[3]), (0.95, -0.2, NORM_G[1]))
        gens = [_alanorm_gen(d, inits["w"], a_, *b_) for a_, b_ in zip(ap, p)]
        ops = list(dict.fromkeys([*aops, *ops]))
        return ops, abody + body, ins, outs, {**ainits, **inits}, gens
    if kind in ("nrml", "lanorml", "slanorml"):
        # a second linear after the norm: ``y = yn @ w2`` (``[1, H] @ [H, n2]``)
        base = kind[:-1]
        if base == "nrm":
            n, n2 = d.H, n  # nrml<n2>: the norm of [1, H], then H -> n2
        ops, body, ins, outs, inits, gens = graph(f"{base}{n}", d)
        body = body.replace("  y = Mul(d, g)", "  yn = Mul(d, g)")
        body += "  y = MatMul(yn, w2)\n"
        inits = {**inits, "w2": template_weight(d.H, n2, 2)}
        if isinstance(outs, dict):
            outs = {**outs, "y": [1, n2]}
        else:
            outs = [1, n2]
        ops = list(dict.fromkeys(["FullyConnected", "MatMul", *ops]))
        return ops, body, ins, outs, inits, gens
    if kind in ("lyrqk", "lyrqkr", "alyrqk", "alyrqkr"):
        return _lyr_graph(n, d, rot=kind.endswith("r"), attn=kind[0] == "a")
    if kind == "alyrqkrs":
        # ``alyrqkr<n>`` with the gate, up and down linears split in two
        # (``W ~ Q8 + R8``: ~16-bit weights for the first set's layers 0-2)
        return _lyr_graph(n, d, rot=True, attn=True, split=True)
    if kind == "alyrf":
        # the last layer as one program: ``alanorm<n>``, gate / up, then
        # ``slanorm<I>`` (the final norm), or ``slanorml<I>x<n2>`` (the final
        # norm and ``lm_head``)
        tail = f"slanorml{d.I}x{n2}" if n2 else f"slanorm{d.I}"
        return _lyr_graph(n, d, attn=True, tail=tail)
    eps = {"eps": np.array([d.eps], F)}
    heads = {
        "shh": np.array([n // d.hd, d.hd], np.int64),
        "shy": np.array([1, n], np.int64),
    }
    four = {"q": [1, n], "r": [1, n], "c": [1, n], "sn": [1, n]}
    if kind == "crms":
        gens = [_norm_gen(n, s, o) for s, o in NORM_X]
        return NORM_OPS, CRMS, {"x": [1, n]}, [1, n], eps, gens
    if kind == "rms":
        gens = [_norm_gen(n, s, o, g) for (s, o), g in zip(NORM_X, NORM_G)]
        return NORM_OPS, RMS, {"x": [1, n], "g": [1, n]}, [1, n], eps, gens
    if kind == "nrm":
        gens = [_norm_gen(n, s, o, g) for (s, o), g in zip(NORM_X, NORM_G)]
        eps2 = {**eps, "eps0": eps["eps"]}
        return NORM_OPS, NORM2, {"x": [1, n], "g": [1, n]}, [1, n], eps2, gens
    if kind == "anorm":
        p = ((3.0, 0.2, 0.0, 1.0), (2.0, 0.1, 0.9, 0.3))
        p += ((8.0, 1.0, 0.2, 0.5), (1.0, -0.2, 0.95, 0.1))
        gens = [_addnorm_gen(n, *a, g) for a, g in zip(p, NORM_G)]
        ins = {"a": [1, n], "b": [1, n], "g": [1, n]}
        eps2 = {**eps, "eps0": eps["eps"]}
        return NORM_OPS, ADDNORM, ins, {"x": [1, n], "y": [1, n]}, eps2, gens
    if kind == "lanorm":
        # n: the linear's input width; output width d.H. The weights are the
        # template's own (a generated program's are written by linear_emit)
        H = d.H
        w = template_weight(n, H)
        p = ((1.0, 0.2, 0.0, NORM_G[0]), (2.0, 0.1, 0.9, NORM_G[1]))
        p += ((0.5, 1.0, 0.3, NORM_G[3]), (3.0, -0.2, 0.95, NORM_G[1]))
        gens = [_lanorm_gen(n, H, w, *a) for a in p]
        ins = {"z": [1, n], "a": [1, H], "g": [1, H]}
        inits = {"w": w, "eps": eps["eps"], "eps0": eps["eps"]}
        ops = ["FullyConnected", "MatMul"] + NORM_OPS
        return ops, LINEAR_ADDNORM, ins, {"x": [1, H], "y": [1, H]}, inits, gens
    if kind == "slanorm":
        H = d.H
        w = template_weight(n, H)
        p = ((1.0, 1.0, 0.2, 0.0, NORM_G[0]), (2.0, 0.7, 0.1, 0.9, NORM_G[1]))
        p += ((0.8, 1.5, 1.0, 0.3, NORM_G[3]), (1.5, 2.0, -0.2, 0.95, NORM_G[1]))
        gens = [_slanorm_gen(n, H, w, *a) for a in p]
        ins = {"gate": [1, n], "up": [1, n], "a": [1, H], "g": [1, H]}
        inits = {"w": w, "eps": eps["eps"], "eps0": eps["eps"]}
        ops = ["FullyConnected", "MatMul", "Sigmoid"] + NORM_OPS
        return ops, SILU_LINEAR_ADDNORM, ins, {"x": [1, H], "y": [1, H]}, inits, gens
    if kind in ("nrmqk", "slanormqk", "nrmqkr", "slanormqkr"):
        # the norm (nrm<H>, or slanorm<I>), then five linears q, q_rot, k, k_rot,
        # v from its output and the qkf RoPE: outputs (x,) yq, yk, v.
        # ``nrmqkr`` / ``slanormqkr`` (``fuse=6``): three linears q, k, v, and
        # q_rot / k_rot made in the program (``ROT``, data movement)
        rot = kind.endswith("r")
        base = "nrm" if kind.startswith("nrm") else "slanorm"
        ops, body, ins, outs, inits, gens = graph(f"{base}{n}", d)
        _, qbody, qins, qouts, qinits, _ = graph(f"qkf{d.nh * d.hd}", d)
        nq, nk = d.nh * d.hd, d.nkv * d.hd
        body = body.replace("  y = Mul(d, g)", "  yn = Mul(d, g)")
        for a, b in NORM_QK_RENAME.items():  # qkf's sin table is ``sq``
            body = body.replace(f"  {a} = ", f"  {b} = ").replace(f"({a})", f"({b})")
        widths = {"q": nq, "qr": nq, "k": nk, "kr": nk, "v": nk}
        if rot:
            widths = {t: widths[t] for t in ("q", "k", "v")}
        body += "".join(f"  {t} = MatMul(yn, w{t})\n" for t in widths)
        if rot:
            body += ROT.format(t="q", sh="shyq") + ROT.format(t="k", sh="shyk")
            half = d.hd // 2
            qinits = {
                **qinits,
                "shq3": np.array([d.nh, 2, half], np.int64),
                "shk3": np.array([d.nkv, 2, half], np.int64),
            }
        body += qbody
        ins = {**ins, **{t: qins[t] for t in ("cq", "sq", "ck", "sk")}}
        outs = {**({"x": outs["x"]} if isinstance(outs, dict) else {}), **qouts}
        outs["v"] = [1, nk]
        variants = ((0.0, 0.0), (0.95, 0.98), (1.0, 0.0), (0.0, 0.9))
        out_gens = []
        for i, (kq, kk) in enumerate(variants):
            ws = {
                f"w{t}": template_weight(d.H, m, 3 + j)
                for j, (t, m) in enumerate(widths.items())
            }
            noise = np.random.default_rng(20 + i)
            for t, kx in (("q", kq), ("k", kk)):
                if not kx:
                    continue
                # the rotated part cancels the plain part: that RoPE Add shifts < 15
                w = ws[f"w{t}"]
                if rot:  # each head's second half ~ -kx x its first half
                    wh = w.reshape(d.H, -1, 2, d.hd // 2).copy()
                    lo = wh[:, :, 0]
                    wh[:, :, 1] = -kx * lo + 0.05 * noise.uniform(-0.1, 0.1, lo.shape)
                    ws[f"w{t}"] = wh.reshape(w.shape).astype(F)
                else:
                    ws[f"w{t}r"] = (
                        -kx * w + 0.05 * noise.uniform(-0.1, 0.1, w.shape)
                    ).astype(F)

            def gen(r, g0=gens[i % len(gens)], kq=kq, kk=kk):
                feeds = g0(r)
                for t, m, kx in (("q", nq, kq), ("k", nk, kk)):
                    c = r.uniform(-0.6, 1.0, (1, m)).astype(F)
                    sn = (
                        c * r.uniform(0.9, 1.1, (1, m))
                        if kx
                        else r.uniform(-1.0, 0.8, (1, m))
                    )
                    feeds[f"c{t}"], feeds[f"s{t}"] = c, sn.astype(F)
                return feeds

            out_gens.append((gen, ws))
        inits = {**inits, **qinits, **out_gens[0][1]}
        ops = list(dict.fromkeys(["FullyConnected", "MatMul", *ops, *qops_of(d)]))
        if rot:
            ops += ["Split", "Concat"]
        return ops, body, ins, outs, inits, out_gens
    if kind == "qkf":
        if n != d.nh * d.hd:
            raise ValueError(f"{name}: the model's q width is {d.nh * d.hd}")
        nk = d.nkv * d.hd
        shapes = {
            "shq": np.array([d.nh, d.hd], np.int64),
            "shk": np.array([d.nkv, d.hd], np.int64),
            "shyq": np.array([1, n], np.int64),
            "shyk": np.array([1, nk], np.int64),
        }
        ins = {"q": [1, n], "qr": [1, n], "k": [1, nk], "kr": [1, nk]}
        ins |= {"cq": [1, n], "sq": [1, n], "ck": [1, nk], "sk": [1, nk]}
        p = ((3.0, 20.0, 0.15, 0.0, 0.0), (1.0, 60.0, 1.0, 0.95, 0.98))
        p += ((6.0, 8.0, 0.3, 1.0, 0.0), (2.0, 30.0, 0.6, 0.0, 0.9))
        gens = [_qkf_gen(d, *a) for a in p]
        ops = NORM_OPS + ["Reshape"]
        outs = {"yq": [1, n], "yk": [1, nk]}
        eps3 = {f"eps_{t}": eps["eps"] for t in "qku"}
        return ops, QKF, ins, outs, {**eps3, **shapes}, gens
    if kind == "hcrms":
        gens = [_norm_gen(n, s, o) for s, o in NORM_X]
        ops = NORM_OPS + ["Reshape"]
        return ops, HCRMS, {"x": [1, n]}, [1, n], {**eps, **heads}, gens
    if kind == "rope":
        p = ((3.0, 0.5, 0.125), (1.0, -0.3, 1.0), (6.0, 1.0, 0.2), (2.0, 0.2, 0.5))
        return ["Mul", "Add"], ROPE, four, [1, n], {}, [_rope_gen(n, *a) for a in p]
    if kind == "qkrope":
        if alt:
            p = ((3.0, 1.0, 0.95), (1.0, 0.5, 1.0), (6.0, 0.3, 0.9), (2.0, 0.8, 0.98))
            gens = [_rope_shift_gen(n, *a) for a in p]
        else:
            p = ((3.0, 0.5, 0.15), (1.0, -0.3, 1.0), (6.0, 1.0, 0.3), (2.0, 0.2, 0.6))
            gens = [_rope_gen(n, *a) for a in p]
        ops = NORM_OPS + ["Reshape"]
        return ops, QKROPE, four, [1, n], {**eps, **heads}, gens
    if kind == "silu":
        p = (
            (1.0, 0.0, 1.0, 0.1),
            (2.0, -0.5, 0.7, -0.2),
            (0.8, 0.3, 1.5, 0.0),
            (1.5, -0.2, 2.0, 0.4),
        )
        gens = [_silu_gen(n, *a) for a in p]
        return (
            ["Sigmoid", "Mul"],
            SILU,
            {"gate": [1, n], "up": [1, n]},
            [1, n],
            {},
            gens,
        )
    # attention
    if n != d.nh * d.hd:
        raise ValueError(f"{name}: the model's attention is attn{d.nh * d.hd}")
    kv = d.nkv * d.hd
    shapes = {
        "shq": np.array([d.nkv, d.nh // d.nkv, d.hd], np.int64),
        "shk": np.array([d.S + 1, d.nkv, d.hd], np.int64),
        "sho": np.array([1, n], np.int64),
    }
    ins = {"q": [1, n], "k": [d.S + 1, kv], "v": [d.S + 1, kv], "mask": [1, d.S + 1]}
    p = (
        (0.3, 2.0, 0.5, 0.05),
        (0.15, 3.0, 1.0, -0.1),
        (0.5, 1.0, 0.3, 0.02),
        (0.2, 4.0, 0.8, 0.1),
    )
    gens = [_attn_gen(d, *a) for a in p]
    ops = ["MatMul", "Add", "Softmax", "Transpose", "Reshape"]
    return ops, ATTN, ins, [1, n], shapes, gens


#: the norm's tensors renamed in ``nrmqk`` / ``slanormqk`` (clashing with qkf's)
NORM_QK_RENAME = {"sq": "sqn"}
#: ``lyrqk``'s first half (``lanorm``): tensor -> name (the rest get ``_1``)
LYR_RENAME = {"z": "att", "g": "g_1", "a": "a"}
#: ``alyrqk``'s attention (``ATTN``): its tensors get ``_a``
LYR_ATTN = ("q", "k", "v", "mask", "qh", "kh", "kt", "s", "sm", "p", "vh", "vt", "o")
LYR_ATTN += ("shq", "shk", "sho")
#: ``lyrqk``'s variants: (``lanorm`` variant, ``slanormqk`` variant, scale of
#: k's RoPE tables; v7 / v8: v4 with the sin tables of q and k calibrated apart)
LYR_VARIANTS = ((0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (0, 1, 1), (0, 2, 1))
LYR_VARIANTS += ((0, 3, 1), (0, 1, 0.8), (0, 1, 1.25))
#: ``alyrqk``'s variants: ``lanorm``'s v1 (the residual Add cancels, so the
#: norm's tensors tie) mis-assigned k's RoPE jobs on the card (yk 1 / sqrt(2)
#: off); every variant keeps ``lanorm``'s v0 but the last two
ALYR_VARIANTS = ((0, 0, 1), (0, 1, 1), (0, 1, 0.8), (0, 1, 1.25), (0, 2, 1))
ALYR_VARIANTS += ((0, 3, 1), (2, 2, 1), (3, 3, 1))
#: ``alyrqkrs``: the split gate / up linears (``W ~ Q8 + R8``) and down linear
SPLIT_MID = """
  gh = MatMul(y_1, wg)
  gl = MatMul(y_1, wgl)
  gate = Add(gh, gl)
  uh = MatMul(y_1, wu)
  ul = MatMul(y_1, wul)
  up = Add(uh, ul)
"""
SPLIT_DOWN = """
  bh = MatMul(z, w)
  bl = MatMul(z, wl)
  b = Add(bh, bl)
"""
#: the template's second parts' weights, relative to the first parts'
SPLIT_LO = 1 / 256
#: ``alyrqkrk``'s variants after the six of ``ALYR_VARIANTS`` (v6, v7, ...; a variant's
#: draws are seeded by its index), ``(i, j, ks)`` as ``ALYR_VARIANTS``: the first norm's
#: variant 3 (its residual Add cancels the attention's ``z @ w`` in the calibration) with
#: each RoPE variant ``j`` and ``ks``. A template's tie and Add-shift pattern depends on the
#: draws, so each is a separate build; the retarget takes the first variant that accepts a
#: layer's calibration (``pf_generate._retarget_fused``). Found for layer 5 of Qwen3-4B
#: (``docs/axera-llm-pulsar-free-decode.md``, the K-split section).
KSPLIT_ALYR_EXTRA = (
    (3, 0, 1),
    (3, 2, 1),
    (3, 0, 1.25),
    (3, 1, 1),
    (3, 3, 1),
    (3, 0, 0.8),
    (3, 2, 0.8),
)
#: ``alyrf``'s variants (no RoPE): ``alanorm``'s v0 with each ``slanorm`` variant
ALYRF_VARIANTS = ((0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1))


def rs_name(name: str) -> str:
    """The row-scaled family's name of a whole-layer template (``alyrqkrk4096`` ->
    ``alyrrsqkrk4096``, ``alyrfk9728`` -> ``alyrrsfk9728``); other names unchanged."""
    m = re.fullmatch(r"alyr(qkr|f)(k?)(\d+)", name)
    return f"alyrrs{m[1]}{m[2]}{m[3]}" if m else name


def lyr_name(t: str) -> str:
    """The name in ``lyrqk`` / ``alyrqk`` of tensor ``t`` of its first half
    (``lanorm`` / ``alanorm``)."""
    if t in LYR_RENAME:
        return LYR_RENAME[t]
    return f"{t}_a" if t in LYR_ATTN else f"{t}_1"


def _lyr_graph(
    n: int,
    d: Dims,
    rot: bool = False,
    attn: bool = False,
    tail: str | None = None,
    split: bool = False,
    ksplit: bool = False,
    rs: bool = False,
):
    """``lyrqk<n>``: ``lanorm<n>`` + gate / up linears + ``slanormqk<I>``
    (``lyrqkr<n>``: ``slanormqkr<I>``, q / k rotated in the program;
    ``alyrqk<n>`` / ``alyrqkr<n>``: ``alanorm<n>``, the attention first).
    ``tail``: another second half (``slanorm<I>`` / ``slanorml<I>x<n2>``:
    ``alyrf``, no q / k / v, no RoPE tables). ``split``: the gate, up and down
    linears each as two linears and an Add (``alyrqkrs``: ``gate = gh + gl``,
    ``gh = y_1 @ wg``, ``gl = y_1 @ wgl``; ``up`` and ``b`` alike)."""
    if n != d.nh * d.hd:
        raise ValueError(f"lyrqk{n}: the model's q width is {d.nh * d.hd}")
    ops1, b1, ins1, _, in1, g1s = graph(f"{'a' if attn else ''}lanorm{n}", d)
    if rs:  # every norm row-scaled: the residual's own norm (the first half) too
        b1, ops1 = _rs_body(b1), [*dict.fromkeys([*ops1, *RS_OPS])]
    tail = tail or f"slanormqk{'r' if rot else ''}{d.I}"
    if rs:
        tail = "rs" + tail
    qk = "slanormqk" in tail
    ops2, b2, ins2, outs2, in2, g2s = graph(tail, d)
    b1 = re.sub(r"\b[A-Za-z_]\w*\b", lambda t: _lyr_word(t.group(0)), b1)
    b2 = b2.replace("  x = Add(a, b)\n", "  x = Add(x_1, b)\n")
    # Mul(sg, gate), not Mul(gate, sg): Pulsar2 then keeps the Sigmoid table and
    # the Mul (a fused Silu's table interpolates x * sigmoid(x) ~5x coarser)
    b2 = b2.replace("  sl = Mul(gate, sg)\n", "  sl = Mul(sg, gate)\n")
    mid = "  gate = MatMul(y_1, wg)\n  up = MatMul(y_1, wu)\n"
    if split:
        mid = SPLIT_MID
        b2 = b2.replace("  b = MatMul(z, w)\n", SPLIT_DOWN)
    if ksplit:  # the down linear as two halves of K (``KSPLIT_DOWN``)
        if b2.count("  b = MatMul(z, w)\n") != 1:
            raise ValueError(f"ksplit: the tail {tail} has no down linear")
        b2 = b2.replace("  b = MatMul(z, w)\n", KSPLIT_DOWN)
    body = b1 + mid + b2
    ins = {lyr_name(k): v for k, v in ins1.items()} | {"g": [1, d.H]}
    if qk:
        ins |= {t: ins2[t] for t in ("cq", "sq", "ck", "sk")}
    inits = {lyr_name(k): v for k, v in in1.items()}
    inits |= {"wg": template_weight(d.H, d.I, 11), "wu": template_weight(d.H, d.I, 12)}
    inits |= in2
    if ksplit:  # the parts' weights: the halves of the down linear's ``w``
        inits["w"], inits["wl"] = (
            inits["w"][: d.I // 2].copy(),
            inits["w"][d.I // 2 :].copy(),
        )
    if split:  # the second parts: about a weight rounding step (~1 / 256)
        for t, (k, m), seed in (("wgl", (d.H, d.I), 13), ("wul", (d.H, d.I), 14)):
            inits[t] = (template_weight(k, m, seed) * F(SPLIT_LO)).astype(F)
        inits["wl"] = (template_weight(d.I, d.H, 15) * F(SPLIT_LO)).astype(F)
    gens = []
    # variant i: the halves' variants i; then lanorm's v0 (the residual Add at
    # shift 15, as every real o_proj target) with slanormqk's v1..v3 (q / k
    # RoPE Adds at lower shifts)
    variants = ALYR_VARIANTS if attn else LYR_VARIANTS
    if (
        split or ksplit
    ):  # lanorm's v2 / v3 halves (v6, v7) gave wrong x / yq / v on the card
        variants = [v for v in variants if v[0] == 0]
    if ksplit and attn:  # the K-split variants after the six (``KSPLIT_ALYR_EXTRA``)
        variants = [*variants, *KSPLIT_ALYR_EXTRA]
    for i, j, ks in variants if qk else ALYRF_VARIANTS:
        ga, gb = g1s[i], g2s[j]
        gb, ws = gb if isinstance(gb, tuple) else (gb, {})

        def gen(r, ga=ga, gb=gb, ks=ks):
            fa, fb = ga(r), gb(r)
            out = {lyr_name(k): v for k, v in fa.items()} | {"g": fb["g"]}
            if not qk:
                return out
            out |= {t: fb[t] for t in ("cq", "sq")}
            return out | {t: (fb[t] * F(ks)).astype(F) for t in ("ck", "sk")}

        gens.append((gen, ws))
    ops = [*ops1, "FullyConnected", "MatMul", "Sigmoid", "Mul", "Silu", *ops2]
    if not attn:  # the order the lyrqk templates were built with
        ops = ["FullyConnected", "MatMul", "Sigmoid", "Mul", "Silu", *ops2]
    if ksplit:
        ops = [*ops, "Split"]
    ops = list(dict.fromkeys(ops))
    return ops, body, ins, outs2, inits, gens


def _lyr_word(w: str) -> str:
    """A word of the first half's body: a tensor gets ``lyr_name``, an op type,
    an attribute name or a keyword stays."""
    tensors = set(LYR_ATTN) | set(LYR_RENAME)
    tensors |= {"b", "x", "sq0", "m0", "e0", "r0", "u", "sq", "m", "e", "r", "d", "y"}
    tensors |= {"ax", "mx", "xs"}  # the row-scaled coarse pass (``NORM2_CRS``)
    tensors |= {"eps", "eps0", "w"}
    return lyr_name(w) if w in tensors else w


def qops_of(d: Dims) -> list[str]:
    return graph(f"qkf{d.nh * d.hd}", d)[0]


def model_of(body: str, ins: dict, out, inits: dict) -> onnx.ModelProto:
    """``out``: the output shape of a one-output graph (output ``y``), or
    ``{name: shape}`` for several outputs."""

    def sig(d):
        return ", ".join(f"float[{', '.join(map(str, s))}] {n}" for n, s in d.items())

    outs = out if isinstance(out, dict) else {"y": out}
    m = parse_model(f"<{OPSET}> g ({sig(ins)}) => ({sig(outs)})\n{{{body}}}\n")
    for n, a in inits.items():
        m.graph.initializer.append(onnx.numpy_helper.from_array(a, n))
    onnx.checker.check_model(m)
    return m


def float_graph(name: str, d: Dims) -> onnx.ModelProto:
    """The float graph of fused template ``name`` (what the program computes)."""
    _, body, ins, out, inits, _ = graph(name, d)
    return model_of(body, ins, out, inits)


def add_model(n: int) -> onnx.ModelProto:
    return parse_model(
        f"<{OPSET}> g (float[1,{n}] a, float[1,{n}] b) => (float[1,{n}] y) "
        "{ y = Add(a, b) }"
    )


def linear_model(w: np.ndarray) -> onnx.ModelProto:
    k, n = w.shape
    m = parse_model(
        f"<{OPSET}> lin (float[1,{k}] x) => (float[1,{n}] y) {{ y = MatMul(x, w) }}"
    )
    m.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))
    return m


def needed(cfg) -> list[str]:
    """Template names (``fused`` / ``add<n>`` / ``k<K>n<N>``) of a model's programs."""
    H, inter, qd, kvd = cfg.H, cfg.I, cfg.nh * cfg.hd, cfg.kvd
    if getattr(cfg, "fuse", 0):
        names = [f"nrm{H}", f"anorm{H}", f"silu{inter}", f"attn{qd}"]
        names += [f"qkf{qd}"] if cfg.qk_norm else [f"rope{qd}", f"rope{kvd}"]
        if cfg.fuse >= 2:
            names += [f"lanorm{qd}", f"lanorm{inter}"]
        if cfg.fuse >= 3:
            names += [f"slanorm{inter}"]
        qkv = sum(n for _, n in cfg.QKV)
        if cfg.fuse == 4:
            names += [
                f"nrml{qkv}",
                f"lanorml{qd}x{2 * inter}",
                f"slanorml{inter}x{qkv}",
            ]
        if cfg.fuse >= 5 and cfg.qk_norm:
            r = "r" if cfg.fuse >= 6 else ""
            names += [f"lanorml{qd}x{2 * inter}", f"nrmqk{r}{H}"]
            names += [f"slanormqk{r}{inter}"]
        if cfg.fuse >= 7:
            names += [f"alanorml{qd}x{2 * inter}" if cfg.qk_norm else f"alanorm{qd}"]
        if cfg.fuse >= 8 and cfg.qk_norm:
            names += [f"alyrqk{r}{qd}"]
        if cfg.fuse >= 9 and cfg.qk_norm:
            names += [f"alyrf{qd}"]
        if cfg.fuse >= 11 and cfg.qk_norm:
            names += [f"alyrf{qd}x{cfg.vocab}"]
        if cfg.fuse >= 10 and cfg.qk_norm and cfg.split_first["layers"]:
            names += [f"alyrqkrs{qd}"]
    elif cfg.qk_norm:
        names = [f"crms{H}", f"rms{H}", f"silu{inter}", f"attn{qd}", f"add{H}"]
        names += [f"qkrope{qd}", f"qkrope{kvd}", f"qkrope{qd}t", f"qkrope{kvd}t"]
        names += [f"hcrms{2 * kvd}"]
    else:
        names = [f"crms{H}", f"rms{H}", f"silu{inter}", f"attn{qd}", f"add{H}"]
        names += [f"rope{qd}", f"rope{kvd}"]
    widths = {"gu": 2 * inter, "down_proj": H}
    names += [
        f"add{widths[k]}" for k in cfg.split_first["keys"] if cfg.split_first["layers"]
    ]
    qkv = sum(n for _, n in cfg.QKV)
    names += [f"k{H}n{qkv}", f"k{qd}n{H}", f"k{H}n{2 * inter}", f"k{inter}n{H}"]
    names += [f"k{H}n{cfg.vocab}"]
    if getattr(cfg, "ksplit", False) and getattr(cfg, "fuse", 0):
        if cfg.fuse == 4 or cfg.fuse >= 10:  # layout 4: a linear after the norm
            raise NotImplementedError(f"ksplit: layout {cfg.fuse} is not wired")
        names = [ksplit_name(n, inter) for n in names]
    if getattr(cfg, "rs_norm", False) and getattr(cfg, "fuse", 0) >= 8:
        names = [rs_name(n) for n in names]
    return list(dict.fromkeys(names))


def default_root(cfg=None) -> str:
    if os.environ.get("LLM_PF_TEMPLATES"):
        return os.environ["LLM_PF_TEMPLATES"]
    out = cfg.out if cfg is not None else os.environ.get("LLM_PF_OUT", ".")
    return os.path.join(os.path.dirname(os.path.abspath(out)), "templates")


def template_dir(root: str, name: str) -> str:
    if re.fullmatch(r"k\d+n\d+", name):
        return os.path.join(root, "linear", name)
    if name.startswith("add"):
        return os.path.join(root, "single", name)
    return os.path.join(root, "fused", name)


# ---- Pulsar2 builds ---------------------------------------------------------------------
def _pulsar2_build(W, ins, samples, ops, memory=None):
    import pulsar2_docker as P

    os.makedirs(os.path.join(W, "dataset"), exist_ok=True)
    cfgs = []
    for n in ins:
        P.make_numpy_calibration_tar(os.path.join(W, "dataset", f"{n}.tar"), samples[n])
        cfgs.append(
            {
                "tensor_name": n,
                "calibration_dataset": f"./dataset/{n}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(samples[n]),
            }
        )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "compiler": {"check": 0},
        "quant": {
            "input_configs": cfgs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "layer_configs": [{"op_types": ops, "data_type": "U16"}],
        },
    }
    json.dump(cfg, open(os.path.join(W, "config.json"), "w"), indent=1)
    t0 = time.time()
    res = P.build(
        W,
        "model.onnx",
        "out",
        config_path="config.json",
        image=IMAGE,
        timeout=1800,
        memory=memory,
    )
    ok = bool(res.success) and os.path.exists(
        os.path.join(W, "out", "compiled.axmodel")
    )
    err = (res.error or "")[-500:]
    json.dump(
        dict(ok=ok, error=err, seconds=time.time() - t0),
        open(os.path.join(W, "result.json"), "w"),
        indent=1,
    )
    return ok, err


def _fused_variant(root, name, g, i, memory=None) -> None:
    """Variant ``i`` of fused template ``name`` (``g``: ``graph(name, d)``)."""
    ops, body, ins, out, inits, gens = g
    gen = gens[i]
    W = os.path.join(template_dir(root, name), f"v{i}")
    os.makedirs(W, exist_ok=True)
    vinits = inits
    if isinstance(gen, tuple):  # a variant with weights of its own
        gen, extra = gen
        vinits = {**inits, **extra}
    onnx.save(model_of(body, ins, out, vinits), os.path.join(W, "model.onnx"))
    r = np.random.default_rng(300 + i)
    draws = [gen(r) for _ in range(18)]
    samples = {n: [x[n] for x in draws] for n in ins}
    ok, err = _pulsar2_build(W, ins, samples, ops, memory=memory)
    print(name, i, ok, "" if ok else err[-200:], flush=True)


def build_fused(root: str, name: str, d: Dims) -> None:
    """Four variants; samples: 18 draws of the variant's generator (seed 300 + i)."""
    g = graph(name, d)
    for i in range(len(g[5])):
        W = os.path.join(template_dir(root, name), f"v{i}")
        if not os.path.exists(os.path.join(W, "result.json")):
            _fused_variant(root, name, g, i)


def _add_variant(root, name, i, memory=None) -> None:
    n, rng = int(name[3:]), ADD_RANGES[i]
    W = os.path.join(template_dir(root, name), f"v{i}")
    os.makedirs(W, exist_ok=True)
    onnx.save(add_model(n), os.path.join(W, "model.onnx"))
    r = np.random.default_rng(100 + i)
    samples = {}
    for t in ("a", "b"):
        lo, hi = rng[t]
        s = [r.uniform(lo, hi, (1, n)).astype(F) for _ in range(16)]
        samples[t] = s + [np.full((1, n), lo, F), np.full((1, n), hi, F)]
    ok, err = _pulsar2_build(W, ("a", "b"), samples, ["Add"], memory=memory)
    print(name, i, ok, "" if ok else err[-200:], flush=True)


def build_add(root: str, name: str) -> None:
    """Four variants; per input 16 uniform samples over the variant's range plus
    the two range ends as constant samples (seed 100 + i)."""
    for i in range(len(ADD_RANGES)):
        W = os.path.join(template_dir(root, name), f"v{i}")
        if not os.path.exists(os.path.join(W, "result.json")):
            _add_variant(root, name, i)


def build_linear(root: str, name: str, memory=None) -> None:
    """One build; calibration: 16 samples of default_rng(0).uniform(-1, 1)."""
    import pulsar2_docker as P

    k, n = map(int, re.fullmatch(r"k(\d+)n(\d+)", name).groups())
    W = template_dir(root, name)
    if os.path.exists(os.path.join(W, "result.json")):
        return
    os.makedirs(os.path.join(W, "dataset"), exist_ok=True)
    w = np.random.default_rng(1).uniform(-0.1, 0.1, (k, n)).astype(F)
    onnx.save(linear_model(w), os.path.join(W, "model.onnx"))
    r = np.random.default_rng(0)
    xs = [r.uniform(-1, 1, (1, k)).astype(F) for _ in range(16)]
    P.make_numpy_calibration_tar(os.path.join(W, "dataset", "x.tar"), xs)
    cfg = P._llm_quant_config({"x": "./dataset/x.tar"}, 16)
    cfg["quant"]["layer_configs"] = [
        {"op_types": ["FullyConnected"], "data_type": "U16"}
    ]
    json.dump(cfg, open(os.path.join(W, "config.json"), "w"), indent=2)
    t0 = time.time()
    res = P.build(
        W,
        "model.onnx",
        "out",
        config_path="config.json",
        image=IMAGE,
        timeout=1800,
        memory=memory,
    )
    ok = bool(res.success) and os.path.exists(
        os.path.join(W, "out", "compiled.axmodel")
    )
    err = (res.error or "")[-500:]
    if ok:
        # builder-side table check: a linear whose table is absent from its compiled
        # npu_params is a failed template (pf_registry skips it; see the docs)
        import linear_emit as le
        import u16_retarget as ur

        q = ur.quant_of(W)
        if not le.linear_table_present(
            os.path.join(W, "out", "compiled.axmodel"), w,
            q["x"][0], q["x"][1], q["y"][0], q["y"][1],
        ):  # fmt: skip
            ok = False
            err = "weight table absent from npu_params (linear_table_present)"
    json.dump(
        dict(ok=ok, error=err, seconds=time.time() - t0),
        open(os.path.join(W, "result.json"), "w"),
        indent=1,
    )
    print(name, ok, f"{time.time() - t0:.0f}s", flush=True)


def build(root: str, name: str, d: Dims) -> None:
    if re.fullmatch(r"k\d+n\d+", name):
        build_linear(root, name)
    elif name.startswith("add"):
        build_add(root, name)
    else:
        build_fused(root, name, d)


# ---- parallel builds ---------------------------------------------------------------------
#: memory cap of one Pulsar2 container (GB, ``docker run --memory``) by build
#: kind. Measured peaks (Qwen3-0.6B, 2026-10-28): fused and Add builds and the
#: layer linears at most 1.1 GB, the lm_head linear (k1024n151936) 4.5 GB
UNIT_GB = {"fused": 2, "add": 2, "linear": 2}
#: a linear's cap grows with its weights (GB of float32 K x N): the lm_head's
#: 0.58 GB -> 2 + 5 GB
LINEAR_GB_PER_GB = 8


@dataclass
class Unit:
    """One Pulsar2 run: a fused / Add template variant or a linear."""

    name: str
    variant: int | None
    run: object  # run(memory=...)
    gb: int
    seconds: float  # a rough estimate, for the order (longest first)


def units(root: str, names: list[str], d: Dims) -> list[Unit]:
    """The builds of ``names`` not yet in ``root`` (no ``result.json``)."""
    out = []
    for nm in names:
        if re.fullmatch(r"k\d+n\d+", nm):
            if os.path.exists(os.path.join(template_dir(root, nm), "result.json")):
                continue
            k, n = map(int, re.fullmatch(r"k(\d+)n(\d+)", nm).groups())
            gb = UNIT_GB["linear"] + -(-4 * k * n * LINEAR_GB_PER_GB // (1 << 30))

            def run(memory=None, nm=nm):
                build_linear(root, nm, memory=memory)

            out.append(Unit(nm, None, run, gb, 20 + 1.8e-6 * k * n))
            continue
        if nm.startswith("add"):
            n_var, g = len(ADD_RANGES), None
        else:
            g = graph(nm, d)
            n_var = len(g[5])
        for i in range(n_var):
            W = os.path.join(template_dir(root, nm), f"v{i}")
            if os.path.exists(os.path.join(W, "result.json")):
                continue
            if g is None:

                def run(memory=None, nm=nm, i=i):
                    _add_variant(root, nm, i, memory=memory)

                out.append(Unit(nm, i, run, UNIT_GB["add"], 20))
            else:

                def run(memory=None, nm=nm, i=i, g=g):
                    _fused_variant(root, nm, g, i, memory=memory)

                out.append(Unit(nm, i, run, UNIT_GB["fused"], 30))
    return out


def _result(root, u: Unit) -> dict:
    W = template_dir(root, u.name)
    if u.variant is not None:
        W = os.path.join(W, f"v{u.variant}")
    try:
        return json.load(open(os.path.join(W, "result.json")))
    except (OSError, ValueError):
        return {"ok": False}


def build_parallel(
    root: str, names: list[str], d: Dims, jobs: int, mem_gb: int, stagger=1.0
) -> list[Unit]:
    """Build every missing template of ``names``, up to ``jobs`` Pulsar2
    containers at once, the longest builds first. Each container is capped at
    its ``Unit.gb`` (``docker run --memory``) and the running caps sum to at
    most ``mem_gb``; the starts are ``stagger`` seconds apart. A build that
    fails is rebuilt alone afterwards, capped at ``mem_gb``. Returns the
    builds that failed both times."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    todo = sorted(units(root, names, d), key=lambda u: -u.seconds)
    cond = threading.Condition()
    state = {"gb": 0, "n": 0, "last": 0.0, "next": 0}

    def one(iu):
        i, u = iu
        gb = min(u.gb, mem_gb)
        with cond:  # start in list order, when a slot and the memory are free
            while state["next"] != i or (
                state["n"] and (state["n"] >= jobs or state["gb"] + gb > mem_gb)
            ):
                cond.wait()
            wait = state["last"] + stagger - time.time()
            if wait > 0:
                time.sleep(wait)
            state["last"] = time.time()
            state["gb"] += gb
            state["n"] += 1
            state["next"] += 1
            cond.notify_all()
        try:
            u.run(memory=f"{gb}g")
        finally:
            with cond:
                state["gb"] -= gb
                state["n"] -= 1
                cond.notify_all()

    with ThreadPoolExecutor(max(1, jobs)) as ex:
        list(ex.map(one, enumerate(todo)))
    failed = [u for u in todo if not _result(root, u).get("ok")]
    for u in failed:
        print("rebuilding alone:", u.name, u.variant, flush=True)
        u.run(memory=f"{mem_gb}g")
    return [u for u in failed if not _result(root, u).get("ok")]


def main():
    import pf_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", default=None)
    ap.add_argument("--list", action="store_true", help="print the names and exit")
    ap.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("LLM_PF_TEMPLATE_JOBS", 8)),
        help="Pulsar2 builds at once (default: LLM_PF_TEMPLATE_JOBS, else 8); "
        "1 builds one at a time without memory caps",
    )
    ap.add_argument(
        "--mem-gb",
        type=int,
        default=20,
        help="with --jobs > 1: the sum of the running containers' memory caps",
    )
    a = ap.parse_args()
    cfg = pf_model.load()
    names = a.names or needed(cfg)
    if a.list:
        print("\n".join(names))
        return
    root = a.root or default_root(cfg)
    t0 = time.time()
    if a.jobs > 1:
        bad = build_parallel(root, names, Dims.of(cfg), a.jobs, a.mem_gb)
        print(f"done, {time.time() - t0:.0f}s, {len(bad)} failed", flush=True)
        sys.exit(1 if bad else 0)
    for nm in names:
        build(root, nm, Dims.of(cfg))


if __name__ == "__main__":
    main()
