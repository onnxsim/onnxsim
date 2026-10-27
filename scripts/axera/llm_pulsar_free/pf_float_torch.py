"""The float decode step of ``pf_model.Config.float_step`` on torch (CUDA), for
``pf_record.py --device cuda``.

The same ONNX graphs (``llm_layer_decomposition.layer_graph`` per layer, the
final norm) run node by node on torch tensors instead of ``ReferenceEvaluator``:
every intermediate the recorder reads keeps its ONNX name, and the K / V
caches stay on the device. float32 throughout (TF32 off); the values differ
from the numpy path in the last bits (summation order), so a recording made
here is not byte-equal to one made on the CPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_layer_decomposition as ld  # noqa: E402

F32 = np.float32


def _ops(torch):
    def reshape(x, s, a):
        return x.reshape(tuple(int(v) for v in s))

    def reduce_mean(x, a):
        return x.mean(dim=tuple(a["axes"]), keepdim=bool(a.get("keepdims", 1)))

    return {
        "Add": lambda x, y, a: x + y,
        "Mul": lambda x, y, a: x * y,
        "Div": lambda x, y, a: x / y,
        "Sqrt": lambda x, a: torch.sqrt(x),
        "Sigmoid": lambda x, a: torch.sigmoid(x),
        "MatMul": lambda x, y, a: torch.matmul(x, y),
        "ReduceMean": reduce_mean,
        "Reshape": reshape,
        "Concat": lambda *xs: torch.cat(xs[:-1], dim=xs[-1]["axis"]),
        "Transpose": lambda x, a: x.permute(*a["perm"]),
        "Softmax": lambda x, a: torch.softmax(x, dim=a.get("axis", -1)),
    }


class TorchGraph:
    """An ONNX graph of the ops above, evaluated on torch tensors; ``run``
    returns every value by name (inputs, initializers, intermediates)."""

    def __init__(self, model, device):
        import torch

        self.torch, self.device, self.ops = torch, device, _ops(torch)
        self.inits = {}
        for t in model.graph.initializer:
            a = numpy_helper.to_array(t)
            # int64 tensors are shapes: kept on the host (no device sync)
            self.inits[t.name] = (
                a if a.dtype == np.int64 else torch.from_numpy(a.copy()).to(device)
            )
        self.nodes = [
            (
                n.op_type,
                list(n.input),
                n.output[0],
                {x.name: helper.get_attribute_value(x) for x in n.attribute},
            )
            for n in model.graph.node
        ]
        unknown = {op for op, *_ in self.nodes} - set(self.ops)
        if unknown:
            raise NotImplementedError(f"ops without a torch kernel: {sorted(unknown)}")

    def run(self, feeds: dict) -> dict:
        env = dict(self.inits)
        for k, v in feeds.items():
            env[k] = self.tensor(v)
        for op, ins, out, attrs in self.nodes:
            env[out] = self.ops[op](*[env[i] for i in ins], attrs)
        return env

    def tensor(self, v):
        if isinstance(v, self.torch.Tensor):
            return v
        return self.torch.from_numpy(np.ascontiguousarray(v, F32)).to(self.device)


class _Host:
    """A read-only view of a ``TorchGraph.run`` result that copies a value to
    the host (numpy float32) when it is read."""

    def __init__(self, env):
        self.env = env

    def __getitem__(self, k):
        v = self.env[k]
        return v if isinstance(v, np.ndarray) else v.detach().cpu().numpy()

    def __contains__(self, k):
        return k in self.env


class TorchFloat:
    """``new_cache`` / ``float_step`` of a ``pf_model.Config`` on torch."""

    def __init__(self, cfg, device: str = "cuda"):
        import torch

        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except AttributeError:  # older torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.cfg, self.torch, self.device = cfg, torch, device
        m = cfg.model
        self.layers = [
            TorchGraph(ld.layer_graph(m, li, cfg.S), device) for li in range(cfg.L)
        ]
        eps = np.array([cfg.eps], F32)
        self.post = TorchGraph(
            ld.make_graph(
                "post_norm",
                {"input": (1, cfg.H)},
                {"eps": eps, "ln_f": cfg.ln_f.reshape(-1)},
                ld.rmsnorm_lines("nf", "input", "ln_f", "xn"),
                ["xn"],
            ),
            device,
        )
        # [vocab, H]: logits = xn @ head.T
        self.head = torch.from_numpy(np.array(m.weight(m._head), F32)).to(device)

    def new_cache(self):
        z = self.torch.zeros
        mk = [z((self.cfg.S, self.cfg.kvd), device=self.device) for _ in self.layers]
        return mk, [k.clone() for k in mk]

    def float_step(self, Ks, Vs, token, p, record=None):
        """As ``pf_model.Config.float_step``; ``Ks`` / ``Vs`` from ``new_cache``.
        ``record(li, t)`` reads ``t[name]`` as numpy."""
        cfg, S = self.cfg, self.cfg.S
        x = self.layers[0].tensor(
            cfg.model.embed([int(token)]).reshape(1, cfg.H).astype(F32)
        )
        shared = {"mask": cfg.mask_ref(p), **ld.rope_inputs(cfg.model, p)}
        shared = {k: self.layers[0].tensor(v) for k, v in shared.items()}
        for li, g in enumerate(self.layers):
            t = g.run({"input": x, "K_cache": Ks[li], "V_cache": Vs[li], **shared})
            if record:
                record(li, _Host(t))
            Ks[li][p] = t["K_cache_out"][S]
            Vs[li][p] = t["V_cache_out"][S]
            x = t["output"].reshape(1, cfg.H)
        pn = self.post.run({"input": x})
        logits = (pn["xn"].reshape(1, cfg.H) @ self.head.T).reshape(-1)
        return logits.cpu().numpy(), _Host(pn)
