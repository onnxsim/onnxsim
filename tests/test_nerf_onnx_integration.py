"""End-to-end integration: real NeRF-shaped torch modules -> ONNX -> onnxsim.

The unit tests in ``test_exporter_nerf_ops.py`` check each rewrite against
hand-built graphs. This module closes the loop the other way round: it defines
actual ``torch.nn.Module``\\ s that mirror the structure of a NeRF /
novel-view-synthesis renderer, exports them to ONNX the way a user would
(``torch.onnx.export``), runs them through :func:`onnxsim.simplify`, and checks
that the simplified model still computes the same numbers.

That matters because the shape of an exported graph is what the rewrites have to
survive, and it is not what a hand-written graph looks like: torch's exporter
fuses and reorders, emits its own ``Constant``/``Identity`` chains, and picks
whatever opset it defaults to. The pieces exercised here are the ones a NeRF
graph is actually made of:

* a sine/cosine positional encoding head (the ``Sin``/``Cos`` coverage),
* a ray-direction contraction with ``torch.einsum`` (the ``Einsum``
  decomposition),
* an MLP density head over the encoded samples,
* a fixed-trip-count sample loop, which torch exports as ``Loop`` and which
  onnxsim unrolls.

Each module is checked against onnxruntime on the pre- and post-simplification
model, so a rewrite that silently changed a value would fail here even though
every individual op test passed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402


class PositionalEncoding(torch.nn.Module):
    """Sine/cosine encoding of 3-D points -- a NeRF's ``embed`` function."""

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> (B, N, 3 * 2 * num_freqs)
        scaled = x[..., None] * self.freqs  # (B, N, 3, F)
        emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return emb.reshape(*x.shape[:-1], -1)


class RaySampler(torch.nn.Module):
    """Fixed-count ray marching, as a NeRF sampler does before querying the MLP.

    The trip count is a compile-time constant and there is no ``break``, which is
    exactly the form ``eliminate_loop_with_const_trip_count`` unrolls.

    Takes rays already shaped ``(B, 1, 3)`` so no ``unsqueeze`` is needed in the
    graph: torch's legacy ONNX exporter turns an ``unsqueeze`` into a
    ``Reshape`` whose shape it passes in as an ``onnx::Reshape_*`` graph input,
    leaving the exported model with an input no caller can supply.
    """

    def __init__(self, num_samples: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.register_buffer(
            "steps", torch.linspace(0.0, 1.0, num_samples).reshape(1, num_samples, 1)
        )

    def forward(self, origins: torch.Tensor, directions: torch.Tensor):
        # origins/directions: (B, 1, 3) -> points (B, S, 3)
        return origins + directions * self.steps


class TinyNeRF(torch.nn.Module):
    """A miniature NeRF: sample -> encode -> MLP density head -> weights -> RGB.

    Deliberately built from the ops a real renderer uses (positional encoding,
    ``einsum`` contraction, a small MLP, a softmax over sample density) so the
    exported graph contains what the new export coverage targets.
    """

    def __init__(self, hidden: int = 16, num_freqs: int = 4, num_samples: int = 4):
        super().__init__()
        self.encoder = PositionalEncoding(num_freqs)
        feat = 3 * 2 * num_freqs
        self.density = torch.nn.Sequential(
            torch.nn.Linear(feat, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.rgb = torch.nn.Linear(feat, 3)
        self.sampler = RaySampler(num_samples)

    def forward(self, origins: torch.Tensor, directions: torch.Tensor):
        points = self.sampler(origins, directions)  # (B, S, 3)
        emb = self.encoder(points)  # (B, S, F)
        sigma = self.density(emb).squeeze(-1)  # (B, S)
        weights = torch.softmax(sigma, dim=-1)  # (B, S)
        # `bsf,fc->bsc`: a contraction over the feature axis, the shape a NeRF's
        # view-direction projection takes. `rgb.weight` is (3, F) for a
        # Linear(feat, 3), so it is transposed into the `fc` order.
        colors = torch.einsum("bsf,fc->bsc", emb, self.rgb.weight.transpose(0, 1))
        rgb = (weights[..., None] * colors).sum(dim=1)  # (B, 3)
        return rgb, points


class EinsumAttentionLike(torch.nn.Module):
    """A head whose whole body is an ``einsum``, to exercise the decomposition."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # (B, N, C) x (B, N, C) -> (B, N, C), batched matmul via einsum.
        attn = torch.einsum("bnc,bmc->bnm", q, k)
        attn = torch.softmax(attn, dim=-1)
        return torch.einsum("bnm,bmc->bnc", attn, v)


def _assert_equivalent(module, inputs, simplified):
    """The simplified model must still compute the original numbers."""
    import onnxruntime as ort

    feeds = {name: t.numpy() for name, t in inputs.items()}
    got = ort.InferenceSession(simplified.SerializeToString()).run(None, feeds)
    with torch.no_grad():
        expected = module(**inputs)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b.numpy(), atol=1e-4, rtol=1e-3)


def _export_and_simplify(module, inputs):
    """Export ``module`` from a name -> tensor mapping and simplify the result.

    ``input_names`` is passed explicitly, as the repo's other exporter tests do:
    torch's legacy ONNX exporter passes the operand of any broadcast elementwise
    op (``Add``/``Mul``/``Reshape``/...) in as a graph input named
    ``onnx::<Op>_<n>``, so an exported graph otherwise carries inputs no caller
    can supply. Naming the real inputs keeps the exported model's interface to
    exactly the module's own arguments.
    """
    names = list(inputs)
    example = tuple(inputs[n] for n in names)
    return export_simplify_and_check_by_python_api(
        module,
        example,
        export_kwargs={"input_names": names, "opset_version": 17},
        simplify_kwargs={
            "input_data": {n: t.numpy() for n, t in inputs.items()}
        },
    )


def test_tiny_nerf_exports_and_simplifies():
    """A NeRF-shaped module survives export + simplify with its PE head intact."""
    torch.manual_seed(0)
    module = TinyNeRF().eval()
    # (B, 1, 3) so the sampler needs no unsqueeze (see RaySampler).
    inputs = {
        "origins": torch.randn(2, 1, 3),
        "directions": torch.randn(2, 1, 3),
    }
    simplified = _export_and_simplify(module, inputs)
    ops = [n.op_type for n in simplified.graph.node]
    assert "Sin" in ops and "Cos" in ops, f"positional encoding lost: {ops}"
    # The rgb projection's einsum is `bsf,fc->bsc` -- operand 2 carries no batch
    # axis, which the shared decomposer deliberately refuses (it would broadcast
    # a weight matrix against a per-sample tensor, and the exporters' matmul
    # paths are best tested on equal-rank operands). It stays as Einsum, which an
    # exporter then reports as unsupported -- never a silently wrong graph.
    assert "Einsum" in ops, f"expected the refused einsum to survive: {ops}"
    _assert_equivalent(module, inputs, simplified)


def test_einsum_attention_like_exports_and_is_decomposed_for_export():
    """`bnc,bmc->bnm` / `bnm,bmc->bnc` -- what torch emits for attention.

    Note where the rewrite happens: ``simplify()`` deliberately leaves ``Einsum``
    alone, and the decomposition runs on the way into each exporter (Core ML,
    TFLite, WebNN), which is what keeps a non-rewritable equation a hard
    "unsupported op" error instead of a silently wrong graph. So this checks the
    rewrite the exporters actually perform, on a genuinely exported graph.
    """
    from onnxsim.einsum_decompose import decompose_einsum

    torch.manual_seed(0)
    module = EinsumAttentionLike().eval()
    inputs = {
        "q": torch.randn(2, 5, 4),
        "k": torch.randn(2, 5, 4),
        "v": torch.randn(2, 5, 4),
    }
    simplified = _export_and_simplify(module, inputs)
    # Simplification keeps the einsum (both are a form the exporters decompose).
    assert "Einsum" in [n.op_type for n in simplified.graph.node]

    for_export = decompose_einsum(simplified)
    ops = [n.op_type for n in for_export.graph.node]
    assert "Einsum" not in ops, f"einsum survived decomposition: {ops}"
    assert "MatMul" in ops, f"einsum was not lowered to a matmul: {ops}"
    _assert_equivalent(module, inputs, for_export)


def test_ray_sampler_loop_is_unrolled():
    """A fixed-trip-count sampling loop exports as ``Loop`` and is unrolled."""
    torch.manual_seed(0)
    module = RaySampler(num_samples=4).eval()
    inputs = {
        "origins": torch.randn(2, 1, 3),
        "directions": torch.randn(2, 1, 3),
    }
    simplified = _export_and_simplify(module, inputs)
    ops = [n.op_type for n in simplified.graph.node]
    assert "Loop" not in ops, f"the sampling Loop was not unrolled: {ops}"
    _assert_equivalent(module, inputs, simplified)