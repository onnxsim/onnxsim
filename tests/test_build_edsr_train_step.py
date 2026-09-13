"""Tests for ``scripts/axera/build_edsr_train_step.py``'s pure-ONNX helpers
-- ``add_mse_loss_nchw`` and ``trainable_scope``. The real EDSR export
itself needs ``super-image``/``torch``, exercised manually following this
project's convention for every other torch-dependent axera build script:
see ``docs/axera-super-resolution-op-coverage.md`` for that real result.
"""

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_edsr_train_step as m  # noqa: E402


def _sr_model():
    """A minimal stand-in for EDSR's own shape: a single `Conv` producing a
    rank-4 `[N, C, H, W]` super-resolved image, the shape `add_mse_loss_nchw`
    is built for."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,3,4,4] lowres) => (float[1,3,4,4] sr)
        {
          sr = Identity(lowres)
        }
        """
    )
    return model


def test_mse_loss_matches_a_direct_numpy_computation():
    ort = __import__("onnxruntime")
    model = m.add_mse_loss_nchw(_sr_model(), "sr")
    onnx.checker.check_model(model)
    assert model.graph.input[-1].name == "hr"
    assert model.graph.output[-1].name == "loss"

    rng = np.random.RandomState(0)
    lowres = rng.randn(1, 3, 4, 4).astype(np.float32)
    hr = rng.randn(1, 3, 4, 4).astype(np.float32)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (loss,) = session.run(["loss"], {"lowres": lowres, "hr": hr})
    assert np.allclose(loss, np.mean((lowres - hr) ** 2), atol=1e-6)


def test_mse_loss_reduces_over_every_axis_explicitly():
    """The AX650's own bare `ReduceMean` silently reduces only the last
    axis (`docs/axera-on-device-training-handoff.md`'s "Two vendor bugs"
    section) -- this loss must never rely on the default `axes`."""
    model = m.add_mse_loss_nchw(_sr_model(), "sr")
    reduce_node = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    axes_attr = next(a for a in reduce_node.attribute if a.name == "axes")
    assert list(axes_attr.ints) == [0, 1, 2, 3]


def _fwd_with_params(names, shapes=None):
    """A model with one `Conv` per name in `names`, so `trainable_scope`
    has real initializers to filter by prefix. `shapes` (default: every
    name gets a rank-1 `(1,)` tensor) maps a name to its own shape, for
    `weights_only`'s rank filter."""
    shapes = shapes or {}
    model = onnx.ModelProto()
    model.CopyFrom(_sr_model())
    for i, name in enumerate(names):
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros(shapes.get(name, (1,)), np.float32), name)
        )
        model.graph.node.append(
            onnx.helper.make_node("Identity", [name], [f"unused_{i}"])
        )
    return model


def test_trainable_scope_filters_by_prefix():
    fwd = _fwd_with_params(["head.0.weight", "body.0.body.0.weight", "tail.1.weight"])
    assert m.trainable_scope(fwd, "head") == ["head.0.weight"]
    assert m.trainable_scope(fwd, "tail") == ["tail.1.weight"]
    assert m.trainable_scope(fwd, "all") == [
        "head.0.weight",
        "body.0.body.0.weight",
        "tail.1.weight",
    ]


def test_trainable_scope_rejects_an_unknown_name():
    fwd = _fwd_with_params(["head.0.weight"])
    try:
        m.trainable_scope(fwd, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown scope")


def test_trainable_scope_weights_only_drops_rank1_bias_tensors():
    """Real hardware finding, not a style preference: a rank-1 tensor's
    in-graph SGD `Sub` crashes Pulsar2's own NPU backend tiler
    (`docs/axera-super-resolution-op-coverage.md`'s real-hardware
    section) -- confirmed on two different bias shapes there, so this
    filters by rank alone, not by name or size."""
    fwd = _fwd_with_params(
        ["tail.0.0.weight", "tail.0.0.bias", "tail.1.weight", "tail.1.bias"],
        shapes={
            "tail.0.0.weight": (32, 8, 3, 3),
            "tail.0.0.bias": (32,),
            "tail.1.weight": (3, 8, 3, 3),
            "tail.1.bias": (3,),
        },
    )
    assert m.trainable_scope(fwd, "tail", weights_only=True) == [
        "tail.0.0.weight",
        "tail.1.weight",
    ]
    assert m.trainable_scope(fwd, "tail", weights_only=False) == [
        "tail.0.0.weight",
        "tail.0.0.bias",
        "tail.1.weight",
        "tail.1.bias",
    ]
