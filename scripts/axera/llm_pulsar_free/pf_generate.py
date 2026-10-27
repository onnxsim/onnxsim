"""Emit a model's Pulsar-free U16 program set (one calibration set at a time)
by retargeting the native templates of ``pf_templates.py`` -- no Pulsar2 run.

Per layer (``pf_layout.LAYER``; ``kc`` only with a per-head k norm):

  n1_c    fused coarse norm  x -> u = x / sqrt(mean(x^2) + eps)       crms<H>
  n1_rms  fused RMSNorm      u, g -> a  (second pass of the two-pass norm)
  qkv5    merged linear      a -> the ``QKV`` parts                   k<H>n<..>
  rq, rk  fused RoPE apply   q, r, c, sn -> y      rope<n> (qkrope<n> with q / k norm)
  kc      per-head coarse k norm of [k | k_rot] (Qwen3)              hcrms<2 kvd>
  attn    the whole layer's attention over the token-major K / V cache
  o_proj  linear, res1 Add, n2_c, n2_rms, gu merged linear (gate | up),
          silu_f fused SiLU, down_proj linear, res2 Add
  post:   nf_c, nf_rms, lm_head

Calibration (Pulsar-free, from ``pf_record.py``'s activations): fused /
single-op programs from their samples (real + x2 copies for norms and
residual Adds, x sqrt(2) copies of the activations for products and
attention; tables and the mask as they are), every tensor's U16 MinMax
(MatMul operands signed; ``u16_retarget.calibrate_graph``); the second norm
pass sees the card's u (the coarse program in fake-quant); linears at the real
range x2 (merged: the union of the parts). Each program is retargeted from
the first template variant that accepts the target
(``u16_retarget.retarget_graph`` / ``retarget``, ``linear_emit.emit_linear_u16``).

Merged layouts (``cfg.fuse`` >= 1, ``pf_layout.LAYER_FUSED*``): adjacent
programs run as one (every execute costs ~0.27 ms on the card whatever it
computes). ``n1`` / ``n2`` / ``nf`` (residual Add + two-pass norm,
``anorm<H>``; layer 0's ``n1`` the norm alone, ``nrm<H>``), ``qk`` (``rq`` +
``kc`` + ``rk``, ``qkf<nh*hd>``); ``fuse=2``: ``on2`` / ``dn``, the linear
before the Add joins it (``lanorm<k>``: the weight table is
``linear_emit``'s, written into the fused build's ``npu_params``);
``fuse=3``: ``dn`` also holds the fused SiLU (``slanorm<I>``). Each part of a
merged program is calibrated exactly as its own program was (``_stage``: the
ranges of that part's tensors over that part's samples), the program gets the
union; tensors a template calibrates alike are tied (``_tied``: Pulsar2
elides a write of the value a register already holds, so the template's
records cannot carry two values there).

With ``share_linears`` (Qwen3) the ``rest`` set's linear serves both sets
when the first set's ranges are within ``SHARE_MAX`` (calibrated on the
union); run ``rest`` before ``first``. With ``first_pos == 0`` the first set
also shares the rest's ``rq`` / ``rk`` / ``kc`` / ``qk`` (sin = 0 at position 0
degenerates their own calibration), which are then calibrated on both sets.

Writes ``<out>/<set>/L<l>/<program>.axmodel``, ``<out>/cal/<set>/...`` (the
programs' float graphs and calibration samples) and
``<out>/manifest_<set>.json``.

usage: MODEL=llama160m pf_generate.py SET [first_layer last_layer] [--jobs N]
       (SET: first | rest; the layers run in N worker processes, the files
       are the same for any N)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import linear_emit as le  # noqa: E402
import pf_fake_quant as fqm  # noqa: E402
import pf_model  # noqa: E402
import pf_templates as tpl  # noqa: E402
import u16_retarget as ur  # noqa: E402
from pf_fake_quant import FusedEval  # noqa: E402

F32 = np.float32
K2 = F32(np.sqrt(2.0))


def signed_inputs(model) -> set:
    """Tensors Pulsar2 quantizes signed: both operands of every MatMul of two
    activations (a MatMul by a constant is a FullyConnected, its input U16)."""
    m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    consts = {i.name for i in m.graph.initializer}
    return {
        t
        for n in m.graph.node
        if n.op_type == "MatMul" and n.input[1] not in consts
        for t in n.input
    }


def write_npy_tar(path, arrs):
    """The tar ``pulsar2_docker.make_numpy_calibration_tar`` writes (one
    ``<i>.npy`` per sample, the same headers but the time stamp), without the
    temporary ``.npy`` files."""
    import io
    import tarfile

    mode = 0o666 & ~_umask()
    now = time.time()
    with tarfile.open(path, "w") as tf:
        for i, a in enumerate(arrs):
            b = io.BytesIO()
            np.save(b, a)
            ti = tarfile.TarInfo(f"{i}.npy")
            ti.size, ti.mode, ti.mtime = b.tell(), mode, now
            ti.uid, ti.gid = os.getuid(), os.getgid()
            ti.uname, ti.gname = _owner()
            b.seek(0)
            tf.addfile(ti, b)


def _umask():
    m = os.umask(0)
    os.umask(m)
    return m


def _owner():
    import grp
    import pwd

    try:
        u = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        u = ""
    try:
        g = grp.getgrgid(os.getgid()).gr_name
    except KeyError:
        g = ""
    return u, g


def rng2(*arrs):
    """U16 MinMax of the arrays' range x2 (folded with 0)."""
    lo = min(float(np.min(a)) for a in arrs)
    hi = max(float(np.max(a)) for a in arrs)
    return ur.unsigned_minmax(2 * min(lo, 0), 2 * max(hi, 0))


class Generator:
    def __init__(self, cfg: pf_model.Config, root: str):
        self.cfg, self.root, self.out = cfg, root, cfg.out

    def _rel(self, p, base=None):
        return os.path.relpath(p, base or self.out)

    def _acts(self, st, where):
        return np.load(os.path.join(self.out, "acts", st, f"{where}.npz"))

    def _manifest(self, st):
        return json.load(open(os.path.join(self.out, f"manifest_{st}.json")))

    @staticmethod
    def _tar(d, name, samples):
        """``dataset/<name>.tar`` of the samples (as
        ``pulsar2_docker.make_numpy_calibration_tar`` writes it, built in
        memory); returns the float32 samples for the calibration."""
        os.makedirs(os.path.join(d, "dataset"), exist_ok=True)
        arrs = [np.asarray(a, F32) for a in samples]
        write_npy_tar(os.path.join(d, "dataset", f"{name}.tar"), arrs)
        return arrs

    # ---- one program ------------------------------------------------------------------
    def fused(self, d, dst, name, data):
        os.makedirs(d, exist_ok=True)
        tdir0 = tpl.template_dir(self.root, name)
        shutil.copy(
            os.path.join(tdir0, "v0", "model.onnx"), os.path.join(d, "model.onnx")
        )
        arrs = {n: self._tar(d, n, v) for n, v in data.items()}
        model = os.path.join(d, "model.onnx")
        q = ur.propagate(model, ur.calibrate_graph(d, samples=arrs))
        return self._retarget_fused(dst, name, model, q)

    def _retarget_fused(self, dst, name, model, q, weights=None):
        """The program of fused graph ``model`` at calibration ``q`` (or a
        function of the template directory giving it), from the first template
        variant (``<name>/v*``, then ``<name>t/v*``) that accepts it."""
        refused = []
        tdirs = []
        for nm in (name, name + "t"):
            base = tpl.template_dir(self.root, nm)
            if os.path.isdir(base):
                vs = sorted(
                    v for v in os.listdir(base) if v[:1] == "v" and v[1:].isdigit()
                )
                tdirs += [
                    os.path.join(base, v) for v in sorted(vs, key=lambda v: int(v[1:]))
                ]
        for tdir in tdirs:
            axm = os.path.join(tdir, "out", "compiled.axmodel")
            if not os.path.exists(axm):
                continue
            try:
                qt = q(tdir) if callable(q) else q
                m = ur.retarget_graph(
                    axm, model, ur.quant_of(tdir), qt, weights=weights
                )
            except ValueError as e:
                refused.append(f"{self._rel(tdir, self.root)}: {str(e)[:90]}")
                continue
            open(dst, "wb").write(m.SerializeToString())
            return dict(
                kind="fused",
                op=name,
                template=self._rel(tdir, self.root),
                model=self._rel(model),
                calibration=qt,
                signed=sorted(signed_inputs(model)),
                bytes=os.path.getsize(dst),
                refused=refused,
            )
        raise RuntimeError(f"{dst}: no usable template: {refused}")

    def add(self, d, dst, a, b, n=None):
        """A residual Add (single-op) from the ``add<n>`` templates."""
        n = n or self.cfg.H
        os.makedirs(d, exist_ok=True)
        onnx.save(tpl.add_model(n), os.path.join(d, "model.onnx"))
        arrs = {
            "a": self._tar(d, "a", list(a) + list(a * F32(2))),
            "b": self._tar(d, "b", list(b) + list(b * F32(2))),
        }
        q = ur.calibrate(d, "Add", samples=arrs)
        refused = []
        for i in range(4):
            tdir = os.path.join(tpl.template_dir(self.root, f"add{n}"), f"v{i}")
            try:
                m = ur.retarget(
                    os.path.join(tdir, "out", "compiled.axmodel"),
                    "Add",
                    ur.quant_of(tdir),
                    q,
                    ("a", "b"),
                )
            except (ValueError, OSError) as e:
                refused.append(f"v{i}: {str(e)[:90]}")
                continue
            open(dst, "wb").write(m.SerializeToString())
            return dict(
                kind="add",
                op="Add",
                template=self._rel(tdir, self.root),
                calibration=q,
                bytes=os.path.getsize(dst),
                refused=refused,
            )
        raise RuntimeError(f"{dst}: no usable template: {refused}")

    def linear(self, dst, w, x, ys):
        k, n = w.shape
        tdir = tpl.template_dir(self.root, f"k{k}n{n}")
        sx, sy = rng2(x), rng2(*ys)
        m = le.emit_linear_u16(
            os.path.join(tdir, "out", "compiled.axmodel"),
            w,
            {"x": sx[0], "y": sy[0]},
            {"x": sx[1], "y": sy[1]},
        )
        open(dst, "wb").write(m.SerializeToString())
        return dict(
            kind="linear",
            op="Linear",
            template=self._rel(tdir, self.root),
            calibration={"x": list(sx), "y": list(sy)},
            shape=[k, n],
            bytes=os.path.getsize(dst),
        )

    def split(self, st, li, key, dst, xk, yks):
        """A first-set linear with ~16-bit weights: hi = s8(w), lo = s8(w - Q(w)),
        y = Add(hi, lo). The split carries the precision itself: original weights."""
        where = f"L{li}"
        A = self._acts(st, where)
        x = A[xk].reshape(len(A[xk]), -1).astype(F32)
        w = self.cfg.merged_weight(key, li, gptq=False)
        q, r = self.cfg.split_weights(w)
        C = os.path.join(self.out, "cal", st, where)
        ys = [A[k].reshape(len(A[k]), -1) for k in yks]
        hi = self.linear(dst.replace(".axmodel", "_hi.axmodel"), w, x, ys)
        lo = self.linear(dst.replace(".axmodel", "_lo.axmodel"), r, x, [x @ r])
        ad = self.add(
            os.path.join(C, f"{key}_add"),
            dst.replace(".axmodel", "_add.axmodel"),
            x @ q,
            x @ r,
            w.shape[1],
        )
        return dict(
            kind="split",
            op="SplitLinear",
            split=dict(hi=hi, lo=lo, add=ad),
            bytes=hi["bytes"] + lo["bytes"] + ad["bytes"],
        )

    def lin(self, st, li, key, dst, w, xk, yks):
        """A linear program; with ``share_linears`` the rest set's program serves
        both sets (calibrated on the union) and the first set's manifest points at it."""
        cfg = self.cfg
        where = "post" if li is None else f"L{li}"
        if cfg.is_split(st, li, key):
            return self.split(st, li, key, dst, xk, yks)
        if cfg.share_linears and st == "first":
            e = self._manifest("rest")[where].get(key, {})
            if e.get("union_of"):
                return dict(e, shared="rest")
        sets = (st,)
        if cfg.share_linears and st == "rest":
            R, Fs = self._acts("rest", where), self._acts("first", where)
            ratio = max(
                float(np.abs(Fs[k]).max() / max(np.abs(R[k]).max(), 1e-30))
                for k in [xk] + yks
            )
            if ratio <= pf_model.SHARE_MAX:
                sets = ("rest", "first")
        acts = [self._acts(s_, where) for s_ in sets]
        x = np.concatenate([a[xk].reshape(len(a[xk]), -1) for a in acts])
        ys = [np.concatenate([a[k].reshape(len(a[k]), -1) for a in acts]) for k in yks]
        e = self.linear(dst, w, x, ys)
        if len(sets) > 1:
            e["union_of"] = list(sets)
        return e

    def two_pass(self, C, D, p, x, g, ml):
        """n_c then n_rms; the second pass is calibrated on the card's u (n_c in fake-quant)."""
        H = self.cfg.H
        ml[f"{p}_c"] = e = self.fused(
            f"{C}/{p}_c",
            f"{D}/{p}_c.axmodel",
            f"crms{H}",
            {"x": list(x) + list(x * F32(2))},
        )
        ev = FusedEval(
            os.path.join(self.out, e["model"]), e["calibration"], e["signed"]
        )
        u = np.stack([ev(x=xi) for xi in x])
        ml[f"{p}_rms"] = self.fused(
            f"{C}/{p}_rms",
            f"{D}/{p}_rms.axmodel",
            f"rms{H}",
            {"x": list(u) + list(u * F32(2)), "g": [g] * (2 * len(u))},
        )
        ml[f"{p}_rms"]["u_absmax"] = float(np.abs(u).max())

    # ---- merged programs (``fuse=1``) ------------------------------------------------------
    def _stage(self, body, ins, outs, feeds, inits):
        """Ranges ``{tensor: [lo, hi]}`` of a part of a merged graph (its text
        ``body``) over ``feeds`` (input -> samples), and the part's model."""
        m = tpl.model_of(body, ins, outs, inits)
        ev = FusedEval(m, {})
        n = len(next(iter(feeds.values())))
        rg: dict = {}
        for j in range(n):
            for k, a in ev.trace(
                **{t: np.asarray(v[j], F32) for t, v in feeds.items()}
            ).items():
                if k in ev.consts or np.asarray(a).dtype.kind != "f":
                    continue
                lo, hi = float(np.min(a)), float(np.max(a))
                r = rg.setdefault(k, [lo, hi])
                r[0], r[1] = min(r[0], lo), max(r[1], hi)
        return rg, m

    @staticmethod
    def _union(*rgs):
        out: dict = {}
        for rg in rgs:
            for k, (lo, hi) in rg.items():
                r = out.setdefault(k, [lo, hi])
                r[0], r[1] = min(r[0], lo), max(r[1], hi)
        return out

    @staticmethod
    def _calib(rg, model, signed=()):
        """U16 MinMax of every range, and of the graph's float constants;
        ``signed``: tensors calibrated as a dynamic MatMul operand
        (``ur.signed_absmax``, as ``ur.calibrate_graph`` does)."""
        q = {
            k: (
                ur.signed_absmax(max(abs(lo), abs(hi)))
                if k in signed
                else ur.unsigned_minmax(lo, hi)
            )
            for k, (lo, hi) in rg.items()
        }
        for i in model.graph.initializer:
            a = onnx.numpy_helper.to_array(i)
            if a.dtype.kind == "f":
                q[i.name] = ur.unsigned_minmax(a.min(), a.max())
        return ur.propagate(model, q)

    #: a tie may widen a tensor's range at most this much (else the template is refused)
    TIE_MAX = 4.0

    def _tied(self, rg, model, tq):
        """Ranges with every group of tensors that share one calibration in
        template calibration ``tq`` given one range (the union): Pulsar2 leaves
        out a write of the value a register already holds, so where a template
        has two tensors at one scale or zero point (``d = u / r`` with r ~ 1:
        ``u`` and ``d``; ``mean(u^2)``, ``+ eps`` and its ``Sqrt`` of
        normalized ``u``), its records cannot carry two different values."""
        consts = {i.name for i in model.graph.initializer}
        groups: dict = {}
        for t, (sc, zp) in ur.propagate(model, tq).items():
            if t in rg and t not in consts:
                groups.setdefault(("s", sc), set()).add(t)
                if zp > 255:
                    groups.setdefault(("zp", zp), set()).add(t)
        rg = {k: list(v) for k, v in rg.items()}
        for g in groups.values():
            if len(g) < 2:
                continue
            u = self._union(*({t: rg[t]} for t in g))
            lo = min(r[0] for r in u.values())
            hi = max(r[1] for r in u.values())
            for t in sorted(g):  # sorted: the refusal names one tensor every run
                span = max(rg[t][1], 0) - min(rg[t][0], 0)
                if (max(hi, 0) - min(lo, 0)) > self.TIE_MAX * max(span, 1e-30):
                    raise ValueError(
                        f"template ties {sorted(g)}: {t}'s range would widen past "
                        f"{self.TIE_MAX}x"
                    )
            for t in g:
                rg[t] = [lo, hi]
        return rg

    def merged(self, C, D, key, name, rgs, ml, extra=None, w=None, signed=()):
        """Retarget merged program ``name`` at the union of its parts' ranges,
        with the tensors the template calibrates alike tied (``_tied``). ``w``:
        the weights of its linear, or ``{initializer: weights}`` of its linears
        (the float graph keeps the card's s8 weights, for the host evaluation)."""
        d = os.path.join(C, key)
        os.makedirs(d, exist_ok=True)
        model = os.path.join(d, "model.onnx")
        fm = tpl.float_graph(name, tpl.Dims.of(self.cfg))
        if w is not None and not isinstance(w, dict):
            w = {"w": w}
        for wn, wa in (w or {}).items():
            wi = next(i for i in fm.graph.initializer if i.name == wn)
            wi.CopyFrom(
                onnx.numpy_helper.from_array(fqm.s8_weights(wa).astype(F32), wn)
            )
        onnx.save(fm, model)
        rg = self._union(*rgs)

        def q(tdir):
            return self._calib(self._tied(rg, fm, ur.quant_of(tdir)), fm, signed)

        e = self._retarget_fused(
            os.path.join(D, f"{key}.axmodel"), name, model, q, weights=w
        )
        e.update(extra or {})
        ml[key] = e
        return e

    def norm2(
        self,
        C,
        D,
        key,
        ml,
        g,
        x=None,
        a=None,
        b=None,
        z=None,
        w=None,
        silu=None,
        post=None,
        qkv=None,
        attn=None,
    ):
        """The two-pass RMSNorm as one program (``nrm<H>``), or with the
        residual Add before it (``anorm<H>``: ``x = a + b``, outputs ``x`` and
        ``y``). Each part is calibrated as its own program was: the Add on
        ``a`` / ``b`` + x2 copies, the coarse pass on ``x`` + x2, the second
        pass on the coarse pass's ``u`` in fake-quant + x2. With a linear
        before it (``lanorm<k>``: ``b = z @ w``), its input and output at the
        real range x2, as a linear program's; with ``silu = (gate, up)`` the
        fused SiLU computes ``z`` (``slanorm<k>``), calibrated as ``silu_f``.
        ``post = (w2, x2, ys)``: a second linear on the norm's output (``yn``,
        the graph's ``y`` is then ``yn @ w2``; ``nrml`` / ``lanorml`` /
        ``slanorml``), calibrated as that linear's own program: ``x2`` (the
        recorded norm output) and the union of ``ys`` at the real range x2.
        ``qkv = layer``: the five q / k / v linears of that layer and its q / k
        RoPE after the norm (``nrmqk`` / ``slanormqk``), the RoPE parts
        calibrated as ``qk`` (``qk_ranges``, both sets); with ``fuse=6`` three
        linears, q_rot / k_rot rotated in the program (``nrmqkr`` /
        ``slanormqkr``; their ranges are q's / k's). ``attn = (q, K, V,
        masks)``: the attention before the linear computes ``z``
        (``alanorm`` / ``alanorml``, ``fuse=7``), calibrated as ``attn`` was
        (its samples and x sqrt(2) copies, the MatMul operands signed); the
        attention's output and the linear's input are one tensor, at the union
        of both ranges."""
        H = self.cfg.H
        inits = tpl.graph(f"nrm{H}", tpl.Dims.of(self.cfg))[4]
        rgs = []
        if z is not None:
            rgs.append(
                {
                    t: [2 * min(float(v.min()), 0.0), 2 * max(float(v.max()), 0.0)]
                    for t, v in (("z", z), ("b", b))
                }
            )
        if a is not None:
            rg, _ = self._stage(
                "  x = Add(a, b)\n",
                {"a": [1, H], "b": [1, H]},
                {"x": [1, H]},
                {"a": list(a) + list(a * F32(2)), "b": list(b) + list(b * F32(2))},
                {},
            )
            rgs.append(rg)
            x = (a + b).astype(F32)
        rg, mc = self._stage(
            tpl.NORM2_C,
            {"x": [1, H]},
            {"u": [1, H]},
            {"x": list(x) + list(x * F32(2))},
            inits,
        )
        rgs.append(rg)
        ev = FusedEval(mc, self._calib(rg, mc))
        u = np.stack([ev(x=xi) for xi in x])
        rg, _ = self._stage(
            tpl.NORM2_R,
            {"u": [1, H], "g": [1, H]},
            {"y": [1, H]},
            {"u": list(u) + list(u * F32(2)), "g": [g] * (2 * len(u))},
            inits,
        )
        rgs.append(rg)
        name = f"anorm{H}" if a is not None else f"nrm{H}"
        if z is not None:
            name = f"lanorm{z.shape[-1]}"
        weights = w
        if post is not None:
            w2, x2, ys = post
            for r in rgs:  # the norm's own output is ``yn`` now
                if "y" in r:
                    r["yn"] = r.pop("y")

            def r2(*arrs):
                lo = min(float(np.min(v)) for v in arrs)
                hi = max(float(np.max(v)) for v in arrs)
                return [2 * min(lo, 0.0), 2 * max(hi, 0.0)]

            rgs.append({"yn": r2(x2), "y": r2(*ys)})
            weights = {"w2": w2, **({"w": w} if w is not None else {})}
        if qkv is not None:
            lj = qkv
            cfg = self.cfg
            B = self._acts("rest", f"L{lj}")
            oq = pf_model.lay.offsets(cfg.QKV)
            wm = cfg.merged_weight("qkv5", lj)
            src = {
                "q_proj": "q",
                "q_rot": "qr",
                "k_proj": "k",
                "k_rot": "kr",
                "v_proj": "v",
            }
            weights = {} if w is None else {"w": w}
            for part, n in cfg.QKV:
                if cfg.fuse >= 6 and part in ("q_rot", "k_rot"):
                    continue  # rotated in the program (``nrmqkr`` / ``slanormqkr``)
                weights[f"w{src[part]}"] = wm[:, oq[part] : oq[part] + n]
            for r in rgs:
                if "y" in r:
                    r["yn"] = r.pop("y")
                for a_, b_ in tpl.NORM_QK_RENAME.items():
                    if a_ in r:
                        r[b_] = r.pop(a_)
            # the rest set's program on the rest set's samples; the first set's
            # (position 0, where sin = 0 degenerates the RoPE parts) on both
            both = self._st == "first" and cfg.first_pos == 0
            Bs = [B] + ([self._acts("first", f"L{lj}")] if both else [])

            def r2(k):
                v = np.concatenate([b[k].reshape(len(b[k]), -1) for b in Bs])
                lo, hi = float(v.min()), float(v.max())
                return [2 * min(lo, 0.0), 2 * max(hi, 0.0)]

            here = self._acts(self._st, f"L{lj}")
            xa = here["a"].reshape(len(here["a"]), -1)
            rgs.append(
                {
                    "yn": [
                        2 * min(float(xa.min()), 0.0),
                        2 * max(float(xa.max()), 0.0),
                    ],
                    "q": r2("qp"),
                    "qr": r2("qp"),
                    "k": r2("kp"),
                    "kr": r2("kp"),
                    "v": r2("vp"),
                }
            )
            qrgs, _ = self.qk_ranges("rest", lj, B, both=both)
            rgs += qrgs
        if silu is not None:
            gate, up = silu
            n = gate.shape[-1]
            rg, _ = self._stage(
                "  sg = Sigmoid(gate)\n  sl = Mul(gate, sg)\n  z = Mul(sl, up)\n",
                {"gate": [1, n], "up": [1, n]},
                {"z": [1, n]},
                {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
                {},
            )
            rgs.append(rg)
            name = f"slanorm{n}"
        if qkv is not None:
            r = "r" if self.cfg.fuse >= 6 else ""
            if name.startswith("nrm"):
                name = f"nrmqk{r}{H}"
            elif name.startswith("slanorm"):
                name = name.replace("slanorm", f"slanormqk{r}")
            else:
                raise ValueError(f"{name}: no template with q / k / v after the norm")
        if post is not None:
            n2 = post[0].shape[1]
            if name.startswith("nrm"):
                name = f"nrml{n2}"
            elif name.startswith(("lanorm", "slanorm")):
                name = name.replace("lanorm", "lanorml") + f"x{n2}"
            else:
                raise ValueError(f"{name}: no template with a linear after the norm")
        signed = ()
        if attn is not None:
            if not name.startswith("lanorm"):
                raise ValueError(f"{name}: no template with the attention before it")
            name = "a" + name
            q_, K_, V_, masks = attn
            cfg = self.cfg
            qd, kv, S1 = cfg.nh * cfg.hd, cfg.kvd, cfg.S + 1
            body = tpl.ATTN.replace("  y = Reshape(o, sho)", "  z = Reshape(o, sho)")
            ainits = tpl.graph(f"attn{qd}", tpl.Dims.of(cfg))[4]
            rg, am = self._stage(
                body,
                {"q": [1, qd], "k": [S1, kv], "v": [S1, kv], "mask": [1, S1]},
                {"z": [1, qd]},
                {
                    "q": list(q_) + list(q_ * K2),
                    "k": list(K_) + list(K_ * K2),
                    "v": list(V_) + list(V_ * K2),
                    "mask": list(masks) * 2,
                },
                ainits,
            )
            signed = signed_inputs(am)
            u_ = self._union(*rgs, rg)
            # Reshape passes the attention's output through: one calibration,
            # the union of the attention's output and the linear's input range
            u_["o"] = u_["z"] = self._union({"t": u_["o"]}, {"t": u_["z"]})["t"]
            rgs = [u_]
        return self.merged(
            C,
            D,
            key,
            name,
            rgs,
            ml,
            dict(u_absmax=float(np.abs(u).max())),
            w=weights,
            signed=signed,
        )

    def qk(self, st, li, C, D, A, ml):
        """``rq`` + ``kc`` + ``rk`` as one program (``qkf<nh*hd>``), each part
        calibrated as its own program was (``layer``)."""
        cfg = self.cfg
        nh, hd = cfg.nh, cfg.hd
        w = f"L{li}"
        if st == "first" and cfg.first_pos == 0:
            ml["qk"] = dict(self._manifest("rest")[w]["qk"], shared="rest")
            return
        name = f"qkf{nh * hd}"
        rgs, uk = self.qk_ranges(st, li, A)
        self.merged(C, D, "qk", name, rgs, ml, dict(u_absmax=float(np.abs(uk).max())))

    def qk_ranges(self, st, li, A, both=None):
        """Ranges of the q / k RoPE parts (``qkf``) of layer ``li``, each part as
        its own program was calibrated (rest set: with the first set's samples
        too, unless ``both`` says otherwise); and the coarse k norm's ``u`` in
        fake-quant."""
        cfg = self.cfg
        nh, nkv, hd, kvd = cfg.nh, cfg.nkv, cfg.hd, cfg.kvd
        w = f"L{li}"
        inits = tpl.graph(f"qkf{nh * hd}", tpl.Dims.of(cfg))[4]
        if both is None:
            both = st == "rest" and cfg.first_pos == 0
        F_ = self._acts("first", w) if both else None
        tiles = [cfg.rope_tiles(int(p), li) for p in A["p"]]
        t0 = cfg.rope_tiles(0, li)

        def tab(c, n):
            return [t[c] for t in tiles] + ([t0[c]] * n if both else [])

        nq, nk = nh * hd, kvd
        q = A["qp"].astype(F32)
        if both:
            q = np.concatenate([q, F_["qp"].astype(F32)])
        r = cfg.rot(q, nh)
        nf = len(F_["qp"]) if both else 0
        rq, _ = self._stage(
            tpl.QKF_Q,
            {"q": [1, nq], "qr": [1, nq], "cq": [1, nq], "sq": [1, nq]},
            {"yq": [1, nq]},
            {
                "q": list(q) + list(q * K2),
                "qr": list(r) + list(r * K2),
                "cq": tab("cos_q", nf) * 2,
                "sq": tab("sin_q", nf) * 2,
            },
            inits,
        )
        k = A["kp"].astype(F32)
        if both:
            k = np.concatenate([k, F_["kp"].astype(F32)])
        kr = cfg.rot(k, nkv)
        rkc, mkc = self._stage(
            tpl.QKF_KC,
            {"k": [1, nk], "kr": [1, nk]},
            {"uk": [1, nk], "ur": [1, nk]},
            {"k": list(k) + list(k * F32(2)), "kr": list(kr) + list(kr * F32(2))},
            inits,
        )
        ev = FusedEval(mkc, self._calib(rkc, mkc))
        uu = [ev(k=a, kr=b) for a, b in zip(k, kr)]
        uk = np.stack([x[0] for x in uu])
        urr = np.stack([x[1] for x in uu])
        rk, _ = self._stage(
            tpl.QKF_K,
            {"uk": [1, nk], "ur": [1, nk], "ck": [1, nk], "sk": [1, nk]},
            {"yk": [1, nk]},
            {
                "uk": list(uk) + list(uk * K2),
                "ur": list(urr) + list(urr * K2),
                "ck": tab("cos_k", nf) * 2,
                "sk": tab("sin_k", nf) * 2,
            },
            inits,
        )
        return [rq, rkc, rk], uk

    def layer_fused(self, st, li, man):
        """One layer of the merged layout (``pf_layout.LAYER_FUSED``)."""
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        g1, g2 = cfg.gains(li)
        ml = {}
        x = A["x"].astype(F32)
        if li == 0:
            self.norm2(C, D, "n1", ml, g1, x=x)
        else:
            P = self._acts(st, f"L{li - 1}")
            self.norm2(
                C, D, "n1", ml, g1, a=P["x1"].astype(F32), b=P["down"].astype(F32)
            )
        wq = cfg.merged_weight("qkv5", li)
        ml["qkv5"] = self.lin(
            st, li, "qkv5", f"{D}/qkv5.axmodel", wq, "a", ["qp", "kp", "vp"]
        )
        offs = pf_model.lay.offsets(cfg.QKV)
        ml["qkv5"]["parts"] = {k: dict(col=offs[k], n=n) for k, n in cfg.QKV}
        self._rope(st, li, A, C, D, ml)
        self._attn_mlp(st, li, A, C, D, ml, fused=True)
        man[w] = ml

    def layer_fused2(self, st, li, man):
        """One layer of the ``fuse=2`` layout (``pf_layout.LAYER_FUSED2``)."""
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        g1, g2 = cfg.gains(li)
        ml = {}
        x = A["x"].astype(F32)
        if li == 0:
            self.norm2(C, D, "n1", ml, g1, x=x)
        wq = cfg.merged_weight("qkv5", li)
        ml["qkv5"] = self.lin(
            st, li, "qkv5", f"{D}/qkv5.axmodel", wq, "a", ["qp", "kp", "vp"]
        )
        offs = pf_model.lay.offsets(cfg.QKV)
        ml["qkv5"]["parts"] = {k: dict(col=offs[k], n=n) for k, n in cfg.QKV}
        self._rope(st, li, A, C, D, ml)
        aon2 = "aon2" in cfg.programs(st, w)
        if not aon2:
            self._attn(li, A, C, D, ml)
        r2 = lambda k: A[k].reshape(len(A[k]), -1).astype(F32)  # noqa: E731
        self.norm2(
            C,
            D,
            "aon2" if aon2 else "on2",
            ml,
            g2,
            a=x.reshape(len(x), -1),
            b=r2("op"),
            z=r2("att"),
            w=cfg.merged_weight("o_proj", li),
            attn=self._attn_samples(A) if aon2 else None,
        )
        wg = cfg.merged_weight("gu", li)
        ml["gu"] = self.lin(st, li, "gu", f"{D}/gu.axmodel", wg, "m", ["gate", "up"])
        og = pf_model.lay.offsets(cfg.GU)
        ml["gu"]["parts"] = {k: dict(col=og[k], n=n) for k, n in cfg.GU}
        gate, up = A["gate"].astype(F32), A["up"].astype(F32)
        split = cfg.is_split(st, li, "down_proj")
        if cfg.fuse < 3 or split:
            ml["silu_f"] = self.fused(
                f"{C}/silu_f",
                f"{D}/silu_f.axmodel",
                f"silu{cfg.I}",
                {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
            )
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        x1, dn = r2("x1"), r2("down")
        if not split and cfg.fuse >= 3:
            self.norm2(
                C,
                D,
                "dn",
                ml,
                gn,
                a=x1,
                b=dn,
                z=r2("act"),
                w=cfg.merged_weight("down_proj", li),
                silu=(r2("gate"), r2("up")),
            )
        elif split:
            ml["down_proj"] = self.lin(
                st,
                li,
                "down_proj",
                f"{D}/down_proj.axmodel",
                cfg.merged_weight("down_proj", li),
                "act",
                ["down"],
            )
            self.norm2(C, D, "dn", ml, gn, a=x1, b=dn)
        else:
            self.norm2(
                C,
                D,
                "dn",
                ml,
                gn,
                a=x1,
                b=dn,
                z=r2("act"),
                w=cfg.merged_weight("down_proj", li),
            )
        man[w] = ml

    def layer_fused4(self, st, li, man):
        """One layer of the ``fuse=4`` layout (``pf_model.Config.programs``)."""
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        progs = cfg.programs(st, w)
        g1, g2 = cfg.gains(li)
        ml = {}

        def r2(B, k):
            return B[k].reshape(len(B[k]), -1).astype(F32)

        def qkv_post(B, lj):
            return (
                cfg.merged_weight("qkv5", lj),
                r2(B, "a"),
                [r2(B, k) for k in ("qp", "kp", "vp")],
            )

        if "n1q" in progs:
            self.norm2(C, D, "n1q", ml, g1, x=r2(A, "x"), post=qkv_post(A, li))
        if "qkv5" in progs:
            ml["qkv5"] = self.lin(
                st,
                li,
                "qkv5",
                f"{D}/qkv5.axmodel",
                cfg.merged_weight("qkv5", li),
                "a",
                ["qp", "kp", "vp"],
            )
        self._rope(st, li, A, C, D, ml)
        self._attn(li, A, C, D, ml)
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        x, x1 = r2(A, "x"), r2(A, "x1")
        if "on2g" in progs:
            self.norm2(
                C,
                D,
                "on2g",
                ml,
                g2,
                a=x,
                b=r2(A, "op"),
                z=r2(A, "att"),
                w=cfg.merged_weight("o_proj", li),
                post=(
                    cfg.merged_weight("gu", li),
                    r2(A, "m"),
                    [r2(A, "gate"), r2(A, "up")],
                ),
            )
        else:  # a split first-set layer
            self.norm2(
                C,
                D,
                "on2",
                ml,
                g2,
                a=x,
                b=r2(A, "op"),
                z=r2(A, "att"),
                w=cfg.merged_weight("o_proj", li),
            )
            ml["gu"] = self.lin(
                st,
                li,
                "gu",
                f"{D}/gu.axmodel",
                cfg.merged_weight("gu", li),
                "m",
                ["gate", "up"],
            )
            gate, up = r2(A, "gate"), r2(A, "up")
            ml["silu_f"] = self.fused(
                f"{C}/silu_f",
                f"{D}/silu_f.axmodel",
                f"silu{cfg.I}",
                {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
            )
            ml["down_proj"] = self.lin(
                st,
                li,
                "down_proj",
                f"{D}/down_proj.axmodel",
                cfg.merged_weight("down_proj", li),
                "act",
                ["down"],
            )
            self.norm2(C, D, "dn", ml, gn, a=x1, b=r2(A, "down"))
        if "dnq" in progs or ("dn" in progs and "down_proj" not in progs):
            nxt = None
            if "dnq" in progs:
                nxt = qkv_post(self._acts(st, f"L{li + 1}"), li + 1)
            self.norm2(
                C,
                D,
                "dnq" if nxt else "dn",
                ml,
                gn,
                a=x1,
                b=r2(A, "down"),
                z=r2(A, "act"),
                w=cfg.merged_weight("down_proj", li),
                silu=(r2(A, "gate"), r2(A, "up")),
                post=nxt,
            )
        for k in ("gu", "qkv5"):
            if k in ml:
                parts = cfg.GU if k == "gu" else cfg.QKV
                o = pf_model.lay.offsets(parts)
                ml[k]["parts"] = {p: dict(col=o[p], n=n) for p, n in parts}
        man[w] = ml

    def layer_fused5(self, st, li, man):
        """One layer of the ``fuse=5`` / ``6`` layout (``pf_model.Config.programs``)."""
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        progs = cfg.programs(st, w)
        g1, g2 = cfg.gains(li)
        ml = {}
        self._st = st

        def r2(B, k):
            return B[k].reshape(len(B[k]), -1).astype(F32)

        if "n1qk" in progs:
            self.norm2(C, D, "n1qk", ml, g1, x=r2(A, "x"), qkv=li)
        if "qkv5" in progs:  # after a split first-set layer
            ml["qkv5"] = self.lin(
                st,
                li,
                "qkv5",
                f"{D}/qkv5.axmodel",
                cfg.merged_weight("qkv5", li),
                "a",
                ["qp", "kp", "vp"],
            )
            rgs, uk = self.qk_ranges("rest", li, self._acts("rest", w))
            self.merged(
                C,
                D,
                "qk",
                f"qkf{cfg.nh * cfg.hd}",
                rgs,
                ml,
                dict(u_absmax=float(np.abs(uk).max())),
            )
        if "attn" in progs:
            self._attn(li, A, C, D, ml)
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        x, x1 = r2(A, "x"), r2(A, "x1")
        if "on2g" in progs or "aon2g" in progs:
            self.norm2(
                C,
                D,
                "aon2g" if "aon2g" in progs else "on2g",
                ml,
                g2,
                a=x,
                b=r2(A, "op"),
                z=r2(A, "att"),
                w=cfg.merged_weight("o_proj", li),
                post=(
                    cfg.merged_weight("gu", li),
                    r2(A, "m"),
                    [r2(A, "gate"), r2(A, "up")],
                ),
                attn=self._attn_samples(A) if "aon2g" in progs else None,
            )
        else:  # a split first-set layer
            self.norm2(
                C,
                D,
                "on2",
                ml,
                g2,
                a=x,
                b=r2(A, "op"),
                z=r2(A, "att"),
                w=cfg.merged_weight("o_proj", li),
            )
            ml["gu"] = self.lin(
                st,
                li,
                "gu",
                f"{D}/gu.axmodel",
                cfg.merged_weight("gu", li),
                "m",
                ["gate", "up"],
            )
            gate, up = r2(A, "gate"), r2(A, "up")
            ml["silu_f"] = self.fused(
                f"{C}/silu_f",
                f"{D}/silu_f.axmodel",
                f"silu{cfg.I}",
                {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
            )
            ml["down_proj"] = self.lin(
                st,
                li,
                "down_proj",
                f"{D}/down_proj.axmodel",
                cfg.merged_weight("down_proj", li),
                "act",
                ["down"],
            )
            self.norm2(C, D, "dn", ml, gn, a=x1, b=r2(A, "down"))
        if "dnqk" in progs or ("dn" in progs and "down_proj" not in progs):
            self.norm2(
                C,
                D,
                "dnqk" if "dnqk" in progs else "dn",
                ml,
                gn,
                a=x1,
                b=r2(A, "down"),
                z=r2(A, "act"),
                w=cfg.merged_weight("down_proj", li),
                silu=(r2(A, "gate"), r2(A, "up")),
                qkv=li + 1 if "dnqk" in progs else None,
            )
        for k in ("gu", "qkv5"):
            if k in ml:
                parts = cfg.GU if k == "gu" else cfg.QKV
                o = pf_model.lay.offsets(parts)
                ml[k]["parts"] = {p: dict(col=o[p], n=n) for p, n in parts}
        man[w] = ml

    def post_fused2(self, st, man):
        cfg = self.cfg
        D = os.path.join(self.out, st, "post")
        os.makedirs(D, exist_ok=True)
        ml = {}
        ml["lm_head"] = self.lin(
            st,
            None,
            "lm_head",
            f"{D}/lm_head.axmodel",
            cfg.merged_weight("lm_head"),
            "xn",
            ["logits"],
        )
        man["post"] = ml

    def post_fused(self, st, man):
        cfg = self.cfg
        P = self._acts(st, f"L{cfg.L - 1}")
        D = os.path.join(self.out, st, "post")
        C = os.path.join(self.out, "cal", st, "post")
        os.makedirs(D, exist_ok=True)
        ml = {}
        self.norm2(
            C, D, "nf", ml, cfg.ln_f, a=P["x1"].astype(F32), b=P["down"].astype(F32)
        )
        ml["lm_head"] = self.lin(
            st,
            None,
            "lm_head",
            f"{D}/lm_head.axmodel",
            cfg.merged_weight("lm_head"),
            "xn",
            ["logits"],
        )
        man["post"] = ml

    # ---- one layer / the post programs --------------------------------------------------
    def _rope_programs(self, st, li, A, C, D, ml):
        """``kc`` (Qwen3), ``rq`` and ``rk``: the layout-0 RoPE programs (also the
        merged layouts' without a per-head q / k norm)."""
        cfg = self.cfg
        nh, nkv, hd, kvd = cfg.nh, cfg.nkv, cfg.hd, cfg.kvd
        w = f"L{li}"
        shared_first = st == "first" and cfg.first_pos == 0
        # with qk_norm the layer's q / k gains are folded into the tables
        tiles = [cfg.rope_tiles(int(p), li) for p in A["p"]]
        rope_t = "qkrope" if cfg.qk_norm else "rope"
        if (
            cfg.qk_norm
        ):  # k's per-head norm needs the two-pass form: kc gives u = k / r0_head
            kk = np.concatenate(
                [A["kp"].astype(F32), cfg.rot(A["kp"].astype(F32), nkv)], axis=-1
            )
            if shared_first:
                # shared with the rest set (calibrated on both)
                e = ml["kc"] = dict(self._manifest("rest")[w]["kc"], shared="rest")
            else:
                cal = kk
                if cfg.first_pos == 0:
                    Fk = self._acts("first", w)["kp"].astype(F32)
                    cal = np.concatenate(
                        [kk, np.concatenate([Fk, cfg.rot(Fk, nkv)], axis=-1)]
                    )
                ml["kc"] = e = self.fused(
                    f"{C}/kc",
                    f"{D}/kc.axmodel",
                    f"hcrms{2 * kvd}",
                    {"x": list(cal) + list(cal * F32(2))},
                )
            ev = FusedEval(
                os.path.join(self.out, e["model"]), e["calibration"], e["signed"]
            )
            uk = np.stack([ev(x=xi) for xi in kk])
            ml["kc"]["u_absmax"] = float(np.abs(uk).max())
        for key, src, heads, c, sn in (
            ("rq", "qp", nh, "cos_q", "sin_q"),
            ("rk", "kp", nkv, "cos_k", "sin_k"),
        ):
            q = A[src].astype(F32)
            r = cfg.rot(q, heads)
            if cfg.qk_norm and key == "rk":
                q, r = uk[..., :kvd], uk[..., kvd:]
            if shared_first:
                # sin = 0 at position 0: r * sn has a zero range, the calibration
                # degenerates; the rest set's program (calibrated on both) covers it
                ml[key] = dict(self._manifest("rest")[w][key], shared="rest")
                continue
            tl = [t[c] for t in tiles], [t[sn] for t in tiles]
            if st == "rest" and cfg.first_pos == 0:
                # position 0 has cos = 1 on every dimension: its output reaches past the
                # rest positions' range (Qwen3 rk: 30.7 vs 21.6 at L18 -> K row 0, the
                # attention sink, clipped). Calibrate on both sets.
                qf = self._acts("first", w)[src].astype(F32)
                rf = cfg.rot(qf, heads)
                if cfg.qk_norm and key == "rk":
                    ukf = np.stack(
                        [ev(x=xi) for xi in np.concatenate([qf, rf], axis=-1)]
                    )
                    qf, rf = ukf[..., :kvd], ukf[..., kvd:]
                t0 = cfg.rope_tiles(0, li)
                q, r = np.concatenate([q, qf]), np.concatenate([r, rf])
                tl = (tl[0] + [t0[c]] * len(qf), tl[1] + [t0[sn]] * len(qf))
            ml[key] = self.fused(
                f"{C}/{key}",
                f"{D}/{key}.axmodel",
                f"{rope_t}{heads * hd}",
                {
                    "q": list(q) + list(q * K2),
                    "r": list(r) + list(r * K2),
                    "c": tl[0] * 2,
                    "sn": tl[1] * 2,
                },
            )

    def _rope(self, st, li, A, C, D, ml):
        """The merged layouts' RoPE: ``qk`` with a per-head q / k norm, else
        ``rq`` and ``rk``."""
        if self.cfg.qk_norm:
            self.qk(st, li, C, D, A, ml)
        else:
            self._rope_programs(st, li, A, C, D, ml)

    def layer(self, st, li, man):
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        g1, g2 = cfg.gains(li)
        ml = {}
        x = A["x"].astype(F32)
        self.two_pass(C, D, "n1", x, g1, ml)
        wq = cfg.merged_weight("qkv5", li)
        ml["qkv5"] = self.lin(
            st, li, "qkv5", f"{D}/qkv5.axmodel", wq, "a", ["qp", "kp", "vp"]
        )
        offs = pf_model.lay.offsets(cfg.QKV)
        ml["qkv5"]["parts"] = {k: dict(col=offs[k], n=n) for k, n in cfg.QKV}
        self._rope_programs(st, li, A, C, D, ml)
        self._attn_mlp(st, li, A, C, D, ml)
        man[w] = ml

    def _attn_samples(self, A):
        """The attention's recorded inputs ``(q, K, V, masks)``."""
        masks = [self.cfg.mask_plan(int(p)) for p in A["p_attn"]]
        return A["qr"].astype(F32), A["K"].astype(F32), A["V"].astype(F32), masks

    def _attn(self, li, A, C, D, ml):
        cfg = self.cfg
        q, K, V = A["qr"].astype(F32), A["K"].astype(F32), A["V"].astype(F32)
        mask = [cfg.mask_plan(int(p)) for p in A["p_attn"]]
        ml["attn"] = self.fused(
            f"{C}/attn",
            f"{D}/attn.axmodel",
            f"attn{cfg.nh * cfg.hd}",
            {
                "q": list(q) + list(q * K2),
                "k": list(K) + list(K * K2),
                "v": list(V) + list(V * K2),
                "mask": mask * 2,
            },
        )

    def _attn_mlp(self, st, li, A, C, D, ml, fused=False):
        """attn, o_proj, then (``fused``) n2 = res1 + norm, or res1, n2_c,
        n2_rms; gu, silu_f, down_proj, and res2 unless ``fused`` (the next
        layer's ``n1`` holds it)."""
        cfg = self.cfg
        x = A["x"].astype(F32)
        _, g2 = cfg.gains(li)
        self._attn(li, A, C, D, ml)
        ml["o_proj"] = self.lin(
            st,
            li,
            "o_proj",
            f"{D}/o_proj.axmodel",
            cfg.merged_weight("o_proj", li),
            "att",
            ["op"],
        )
        x1 = A["x1"].astype(F32)
        if fused:
            self.norm2(C, D, "n2", ml, g2, a=x, b=A["op"].astype(F32))
        else:
            ml["res1"] = self.add(
                f"{C}/res1", f"{D}/res1.axmodel", x, A["op"].astype(F32)
            )
            self.two_pass(C, D, "n2", x1, g2, ml)
        wg = cfg.merged_weight("gu", li)
        ml["gu"] = self.lin(st, li, "gu", f"{D}/gu.axmodel", wg, "m", ["gate", "up"])
        og = pf_model.lay.offsets(cfg.GU)
        ml["gu"]["parts"] = {k: dict(col=og[k], n=n) for k, n in cfg.GU}
        gate, up = A["gate"].astype(F32), A["up"].astype(F32)
        ml["silu_f"] = self.fused(
            f"{C}/silu_f",
            f"{D}/silu_f.axmodel",
            f"silu{cfg.I}",
            {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
        )
        ml["down_proj"] = self.lin(
            st,
            li,
            "down_proj",
            f"{D}/down_proj.axmodel",
            cfg.merged_weight("down_proj", li),
            "act",
            ["down"],
        )
        if not fused:
            ml["res2"] = self.add(
                f"{C}/res2", f"{D}/res2.axmodel", x1, A["down"].astype(F32)
            )

    def post(self, st, man):
        cfg = self.cfg
        A = self._acts(st, "post")
        D = os.path.join(self.out, st, "post")
        C = os.path.join(self.out, "cal", st, "post")
        os.makedirs(D, exist_ok=True)
        ml = {}
        self.two_pass(C, D, "nf", A["x"].astype(F32), cfg.ln_f, ml)
        ml["lm_head"] = self.lin(
            st,
            None,
            "lm_head",
            f"{D}/lm_head.axmodel",
            cfg.merged_weight("lm_head"),
            "xn",
            ["logits"],
        )
        man["post"] = ml


#: host memory one worker of ``--jobs`` may take (Qwen3-0.6B layout 5: a layer's
#: recorded activations, calibration samples and weight tables)
JOB_BYTES = 3 << 30
#: the generator the worker processes inherit (``fork``)
_GEN = None


def _mem_free() -> int:
    """Bytes this process may still allocate: MemAvailable, and at most its
    cgroup's ``memory.max`` headroom (``systemd-run -p MemoryMax=...``)."""
    free = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            free = int(line.split()[1]) * 1024
    try:
        cg = open("/proc/self/cgroup").read().split("::", 1)[1].strip()
        base = os.path.join("/sys/fs/cgroup", cg.lstrip("/"))
        mx = open(os.path.join(base, "memory.max")).read().strip()
        if mx != "max":
            cur = int(open(os.path.join(base, "memory.current")).read())
            free = min(free, int(mx) - cur)
    except (OSError, IndexError, ValueError):
        pass
    return max(free, 0)


def default_jobs() -> int:
    """Worker processes: ``LLM_PF_JOBS``, else bounded by the cores (4 threads
    each, at most 8 workers) and by ``JOB_BYTES`` of free memory per worker."""
    if os.environ.get("LLM_PF_JOBS"):
        return max(1, int(os.environ["LLM_PF_JOBS"]))
    cores = len(os.sched_getaffinity(0))
    return max(1, min(8, cores // 4, _mem_free() // JOB_BYTES))


def _run_one(job):
    """One layer (``li``) or the post programs (``li`` None) of set ``st``;
    returns its manifest entry."""
    st, li = job
    gen, cfg = _GEN, _GEN.cfg
    man: dict = {}
    if li is None:
        post = [gen.post, gen.post_fused, gen.post_fused2, gen.post_fused2]
        (post[cfg.fuse] if cfg.fuse < 4 else gen.post_fused2)(st, man)  # noqa
        return man["post"]
    layer = [gen.layer, gen.layer_fused, gen.layer_fused2, gen.layer_fused2]
    layer += [gen.layer_fused4, gen.layer_fused5, gen.layer_fused5]
    layer += [gen.layer_fused5 if cfg.qk_norm else gen.layer_fused2]
    layer[cfg.fuse](st, li, man)
    return man[f"L{li}"]


def main():
    global _GEN
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", choices=("first", "rest"))
    ap.add_argument("layers", nargs="*", type=int, help="first_layer last_layer")
    ap.add_argument("--templates", default=None, help="template root (pf_templates.py)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="worker processes, one layer at a time each (default: LLM_PF_JOBS, "
        "else by cores and free memory); every count writes the same files",
    )
    a = ap.parse_args()
    cfg = pf_model.load()
    gen = Generator(cfg, a.templates or tpl.default_root(cfg))
    lo_l, hi_l = (a.layers[0], a.layers[1]) if len(a.layers) == 2 else (0, cfg.L - 1)
    st = a.set
    mpath = os.path.join(cfg.out, f"manifest_{st}.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    t0 = time.time()
    layers = [(st, li) for li in range(lo_l, hi_l + 1)]
    post = [(st, None)] if hi_l == cfg.L - 1 else []
    n_jobs = min(a.jobs or default_jobs(), len(layers) + len(post))
    _GEN = gen
    if n_jobs > 1:
        import multiprocessing as mp

        # the workers inherit the loaded checkpoint (fork); each writes its own
        # layer's files and returns the layer's manifest entry, which is merged
        # here in layer order. The post programs (lm_head) start first.
        pool = mp.get_context("fork").Pool(n_jobs)
        post_res = [pool.apply_async(_run_one, (j,)) for j in post]
        results = pool.imap(_run_one, layers)
        print(f"{st}: {n_jobs} workers", flush=True)
    else:
        pool, post_res = None, []
        results = map(_run_one, layers)
    for (_, li), entry in zip(layers, results):
        man[f"L{li}"] = entry
        json.dump(man, open(mpath, "w"), indent=1)
        ml = man[f"L{li}"]
        tnames = sorted(
            {
                os.path.basename(os.path.dirname(e.get("template", "split/x")))
                for e in ml.values()
            }
        )
        n1, n2 = [("n1_rms", "n2_rms"), ("n1", "n2")][min(cfg.fuse, 1)]
        if cfg.fuse >= 2:
            n1 = next(k for k in ("on2", "on2g", "aon2g", "aon2") if k in ml)
            n2 = next(k for k in ("dnqk", "dnq", "dn") if k in ml)
        print(
            f"{st} L{li}: {len(ml)} programs, templates {tnames}, u max "
            f"{ml[n1]['u_absmax']:.1f}/{ml[n2]['u_absmax']:.1f}, "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    if post:
        man["post"] = post_res[0].get() if post_res else _run_one(post[0])
        json.dump(man, open(mpath, "w"), indent=1)
        print(f"{st} post done, {time.time() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
