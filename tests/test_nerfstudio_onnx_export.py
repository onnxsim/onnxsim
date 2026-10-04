"""Export a real nerfstudio ``NerfactoField`` to ONNX and run it under ORT.

This is the automated version of ``docs/nerfstudio-onnx-export.md``: it builds a
genuine nerfstudio field, exports its whole feed-forward core, simplifies it with
:func:`onnxsim.simplify`, and checks onnxruntime still reproduces the torch
numbers. Without it, the measurements in that document are a point-in-time claim
nothing re-verifies.

Everything here is optional. ``nerfstudio`` is not a test requirement of this
package -- it is not even importable on Python 3.11+ without the shim below --
so the module is skipped unless it is installed. The CI job that exercises it
(``nerfstudio-onnx-export`` in ``.github/workflows/``) installs it explicitly.

Two findings this pins down, both easy to regress into:

* ``HashEncoding.hash_fn`` uses ``torch.bitwise_xor``, which the **legacy**
  TorchScript exporter cannot emit at any opset, but the dynamo exporter maps to
  ONNX ``BitwiseXor`` (opset 18+) without complaint.
* ``onnxsim`` must leave that hash graph alone: there is nothing redundant in it,
  and rewriting it would break the indices.
"""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")
onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

pytest.importorskip(
    "onnxscript", reason="onnxscript (needed for dynamo=True) is not installed"
)


# ---------------------------------------------------------------------------
# nerfstudio 1.1.5 cannot be imported on Python 3.11+
# ---------------------------------------------------------------------------
#
# Upstream nerfstudio-project/nerfstudio#3106. `configs/base_config.py` declares
#
#     local_writer: LocalWriterConfig = LocalWriterConfig(enable=True)
#
# as a dataclass field default. Python 3.11 widened the mutable-default check
# from "is the value a list/set/dict" to "is its *type* unhashable"
# (https://discuss.python.org/t/better-communicate-dataclass-mutable-default-check-change-in-python-3-11/19028).
# `LocalWriterConfig` defines `__hash__`, so from 3.11 on the module raises
# `ValueError` at import -- which blocks every `Field`/`Model` import, not just
# the writer config.
#
# Rewriting that one declaration to `dataclasses.field(default_factory=...)` --
# the fix Python's own error message names -- is enough, and leaves the config
# behaving identically (each instance still gets its own `LocalWriterConfig`,
# which is what the check protects). Nothing inside nerfstudio is modified; the
# corrected module is loaded from source under its real name.
#
# If a future nerfstudio fixes this upstream the pattern simply will not be
# found and this becomes a no-op, so the shim is safe to keep.


_OFFENDING = "local_writer: LocalWriterConfig = LocalWriterConfig(enable=True)"
_REPLACEMENT = (
    "local_writer: LocalWriterConfig = dataclasses.field("
    "default_factory=lambda: LocalWriterConfig(enable=True))"
)


def _patch_nerfstudio_mutable_default() -> None:
    """Load a corrected ``nerfstudio.configs.base_config`` under its real name.

    Uses ``importlib.resources``/``__file__`` rather than string-splitting so it
    does not assume a POSIX layout, and registers the module in ``sys.modules``
    *before* executing it so the ``from ... import InstantiateConfig`` that
    nerfstudio's own modules do resolves against the patched copy.
    """
    import importlib.util
    import os
    import sys

    if sys.version_info < (3, 11):
        return
    import nerfstudio

    package_dir = os.path.dirname(os.path.abspath(nerfstudio.__file__))
    path = os.path.join(package_dir, "configs", "base_config.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    if _OFFENDING not in source:
        return  # already fixed upstream
    patched = source.replace(_OFFENDING, _REPLACEMENT)
    if "import dataclasses" not in patched:
        patched = patched.replace(
            "from dataclasses import", "import dataclasses\nfrom dataclasses import", 1
        )
    name = "nerfstudio.configs.base_config"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = path
    module.__package__ = "nerfstudio.configs"
    sys.modules[name] = module
    exec(compile(patched, path, "exec"), module.__dict__)


_patch_nerfstudio_mutable_default()

NerfactoField = pytest.importorskip(
    "nerfstudio.fields.nerfacto_field",
    reason="nerfstudio is not installed (or cannot be imported)",
).NerfactoField


# nerfstudio's own modules emit torch.cuda.amp deprecation warnings on import
# and at every custom-autograd call; they are not this test's concern.
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class NerfactoCore(torch.nn.Module):
    """nerfacto's density + colour path, assembled for export.

    Mirrors ``NerfactoField.get_density`` / ``get_outputs`` on the eval path
    with one image and no transients, predicted normals or semantics:

        h = mlp_base(positions)              # hash grid -> geo-feature MLP
        density, geo = split(h, [1, geo_feat_dim])
        density = clamp_min(expm1(density), 0)     # nerfstudio's trunc_exp
        rgb   = mlp_head(cat([direction_encoding(directions), geo, appearance]))

    Those two methods are not exported directly because they take a
    ``RaySamples`` dataclass, which has no ONNX representation; this is the same
    computation with the dataclass reduced to the two tensors it consumes.
    """

    def __init__(self, field):
        super().__init__()
        self.mlp_base = field.mlp_base
        self.direction_encoding = field.direction_encoding
        self.mlp_head = field.mlp_head
        self.geo_feat_dim = field.geo_feat_dim
        self.register_buffer(
            "appearance", torch.zeros(1, field.appearance_embedding_dim)
        )

    def forward(self, positions, directions):
        h = self.mlp_base(positions.reshape(-1, 3))
        density_before_activation, geo_feat = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        density = torch.expm1(density_before_activation).clamp_min(0.0)
        d = self.direction_encoding(directions.reshape(-1, 3))
        appearance = self.appearance.expand(geo_feat.shape[0], -1)
        rgb = self.mlp_head(torch.cat([d, geo_feat, appearance], dim=-1))
        return density, rgb


def _field():
    """A small but genuine nerfacto: real hash grid, real MLPs, tiny dims."""
    return NerfactoField(
        aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        num_images=1,
        num_layers=2,
        hidden_dim=16,
        geo_feat_dim=8,
        num_levels=4,
        base_res=16,
        max_res=128,
        log2_hashmap_size=14,
        features_per_level=2,
        hidden_dim_color=16,
        num_layers_color=2,
        # "torch" is the CPU path; "tcnn" is a CUDA-only fused kernel.
        implementation="torch",
    ).eval()


def _export(module, positions, directions, dynamic=False):
    kwargs = {}
    if dynamic:
        kwargs["dynamic_shapes"] = {
            "positions": {0: "B"},
            "directions": {0: "B"},
        }
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        (positions, directions),
        buf,
        opset_version=18,
        dynamo=True,  # required: the legacy exporter cannot emit bitwise_xor
        input_names=["positions", "directions"],
        output_names=["density", "rgb"],
        **kwargs,
    )
    import onnx

    return onnx.load_from_string(buf.getvalue())


def _op_hist(model):
    hist = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return hist


def _ort_run(model_or_bytes, positions, directions):
    import onnxruntime as ort

    session = ort.InferenceSession(model_or_bytes, providers=["CPUExecutionProvider"])
    return session.run(
        None, {"positions": positions.numpy(), "directions": directions.numpy()}
    )


def test_nerfacto_core_exports_and_matches_torch():
    """The full density + colour core exports and ORT reproduces torch."""
    torch.manual_seed(0)
    module = NerfactoCore(_field()).eval()
    positions = torch.rand(512, 3)
    directions = torch.nn.functional.normalize(torch.randn(512, 3), dim=-1)
    with torch.no_grad():
        ref_density, ref_rgb = module(positions, directions)

    exported = _export(module, positions, directions)
    hist = _op_hist(exported)
    # The Instant-NGP space hash lowers to BitwiseXor -- this is the whole
    # reason the dynamo exporter is required.
    assert hist.get("BitwiseXor", 0) > 0, f"no BitwiseXor in {hist}"
    assert "Gather" in hist

    got_density, got_rgb = _ort_run(exported.SerializeToString(), positions, directions)
    np.testing.assert_allclose(got_density, ref_density.numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(got_rgb, ref_rgb.numpy(), atol=1e-5, rtol=1e-4)
    assert np.isfinite(got_density).all()
    assert got_rgb.min() >= 0.0 and got_rgb.max() <= 1.0


def _export_single_input(module, example):
    """Export a one-input submodule (the hash grid on its own)."""
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        (example,),
        buf,
        opset_version=18,
        dynamo=True,
        input_names=["positions"],
        output_names=["features"],
    )
    import onnx

    return onnx.load_from_string(buf.getvalue())


class _Wrap(torch.nn.Module):
    """Expose a submodule as a plain single-input module."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def test_hash_grid_is_bit_exact_under_ort():
    """The hash encoding alone reproduces torch exactly.

    It is all integer indexing plus table lookups, so there is no floating-point
    reassociation to hide a wrong index behind -- an exact match here means the
    ``BitwiseXor`` lowering is genuinely correct.
    """
    field = _field()
    torch.manual_seed(0)
    positions = torch.rand(256, 3)
    with torch.no_grad():
        reference = field.mlp_base_grid(positions).numpy()

    model = _export_single_input(_Wrap(field.mlp_base_grid).eval(), positions)
    import onnxruntime as ort

    got = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"positions": positions.numpy()})[0]
    np.testing.assert_array_equal(got, reference)


def test_simplify_preserves_the_hash_graph_and_the_numbers():
    """`onnxsim.simplify` must not disturb the hash lookup, and ORT must still
    agree with torch afterwards."""
    import onnxsim

    torch.manual_seed(0)
    module = NerfactoCore(_field()).eval()
    positions = torch.rand(512, 3)
    directions = torch.nn.functional.normalize(torch.randn(512, 3), dim=-1)
    with torch.no_grad():
        ref_density, ref_rgb = module(positions, directions)

    exported = _export(module, positions, directions)
    import onnx

    path = "/tmp/onnxsim_nerfacto_test.onnx"
    onnx.save(exported, path)
    simplified, ok = onnxsim.simplify(
        path,
        check_n=3,
        input_data={"positions": positions.numpy(), "directions": directions.numpy()},
    )
    assert ok, "onnxsim.simplify reported a correctness failure"

    after = _op_hist(simplified)
    # The hash indices must survive untouched: there is nothing redundant in a
    # gather-from-hash-table, and rewriting one would change which cell is read.
    assert after.get("BitwiseXor", 0) == _op_hist(exported).get("BitwiseXor", 0)

    got_density, got_rgb = _ort_run(
        simplified.SerializeToString(), positions, directions
    )
    np.testing.assert_allclose(got_density, ref_density.numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(got_rgb, ref_rgb.numpy(), atol=1e-5, rtol=1e-4)


def test_dynamic_batch_axis_stays_correct():
    """A dynamic batch axis must not change the numbers -- a NeRF renderer feeds
    a different number of samples per ray every call."""
    torch.manual_seed(1)
    module = NerfactoCore(_field()).eval()
    example_positions = torch.rand(64, 3)
    example_directions = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    exported = _export(module, example_positions, example_directions, dynamic=True)

    for rows in (1, 17, 128):
        positions = torch.rand(rows, 3)
        directions = torch.nn.functional.normalize(torch.randn(rows, 3), dim=-1)
        with torch.no_grad():
            ref_density, ref_rgb = module(positions, directions)
        got_density, got_rgb = _ort_run(
            exported.SerializeToString(), positions, directions
        )
        np.testing.assert_allclose(
            got_density, ref_density.numpy(), atol=1e-5, rtol=1e-4
        )
        np.testing.assert_allclose(got_rgb, ref_rgb.numpy(), atol=1e-5, rtol=1e-4)


def test_legacy_exporter_cannot_emit_the_hash_grid():
    """Pins *why* ``dynamo=True`` is required here.

    If a future torch adds an ``aten::bitwise_xor`` symbolic to the legacy
    exporter, this test fails and the comment explaining the dynamo dependency
    above can be revisited.
    """
    field = _field()
    positions = torch.rand(32, 3)

    with pytest.raises(Exception, match="bitwise_xor"):
        torch.onnx.export(
            _Wrap(field.mlp_base_grid).eval(),
            (positions,),
            io.BytesIO(),
            opset_version=18,
            dynamo=False,
        )
