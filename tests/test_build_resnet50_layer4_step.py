"""Regression test for `scripts/axera/build_resnet50_layer4_step.py` --
`docs/axera-on-device-training-handoff.md`'s "resnet50, first compile"
section's own named next step, extending `layer4.2`-alone's trainable scope
(`build_resnet50_batch_step.py`) to the remaining `layer4` bottleneck
blocks. See that doc's "All three `layer4` bottleneck blocks" section for
the real AX650N hardware result this module's `layer4_all` scope produced
(compiles in 184.6s, trains with a real monotonically non-increasing loss).

Needs `torch`/`timm` (export only, CPU, random init) -- same heavy/optional
dependency this repo's other `timm`-based resnet50 tests already require, so
this skips rather than fails when they aren't installed. Neither Docker nor
a device.
"""

import os
import sys

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ort = pytest.importorskip("onnxruntime")

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_resnet50_layer4_step as m  # noqa: E402


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("resnet50_layer4"))


@pytest.mark.parametrize("scope", ["layer4_1_2", "layer4_all"])
def test_build_step_produces_a_checked_step_graph_with_the_right_scope(
    out_dir, scope
):
    step_path, state, params = m.build_step(out_dir, scope, batch=1)
    assert params == m.SCOPES[scope]
    assert set(state) == set(params)

    model = onnx.load(step_path)
    onnx.checker.check_model(model)

    in_names = {i.name for i in model.graph.input}
    for p in params:
        assert p in in_names, f"{p} missing from step graph inputs (scope={scope})"
    assert "lr" in in_names
    assert "grad_seed" in in_names


def test_layer4_all_has_more_trainable_tensors_and_nodes_than_layer4_1_2(out_dir):
    """A coarse but real check that the two scopes are actually different
    sizes, not accidentally identical (e.g. a copy/paste `SCOPES` bug) --
    node count should grow with `layer4_all`'s extra `layer4.0` block."""
    path_1_2, _, params_1_2 = m.build_step(out_dir, "layer4_1_2", batch=1)
    path_all, _, params_all = m.build_step(out_dir, "layer4_all", batch=1)
    assert len(params_all) > len(params_1_2)
    assert len(onnx.load(path_all).graph.node) > len(onnx.load(path_1_2).graph.node)


def test_layer4_all_gradient_matches_finite_differences(out_dir):
    """Central-difference check against the in-graph analytic gradient for
    two of `layer4_all`'s ten trainable tensors (`layer4.0.conv1/conv2`,
    the two furthest from the loss and so the most exposed to a gradient-
    accumulation mistake at this wider scope) -- the same methodology
    `docs/axera-on-device-training-handoff.md`'s resnet50 sections use
    throughout, confirming `_linearize_trainable_convs` still generalizes
    correctly once `layer4`'s three bottleneck blocks are all trainable at
    once, not just `layer4.2` alone."""
    step_path, state, params = m.build_step(out_dir, "layer4_all", batch=1)
    model = onnx.load(step_path)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    in_shapes = {
        i.name: [d if isinstance(d, int) else 1 for d in i.shape]
        for i in sess.get_inputs()
    }
    out_names = [o.name for o in sess.get_outputs()]
    loss_idx = out_names.index("loss")

    rng = np.random.default_rng(0)
    feeds = {}
    for name, shape in in_shapes.items():
        if name == "x":
            feeds[name] = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        elif name == "y":
            arr = np.zeros(shape, dtype=np.float32)
            arr[0, rng.integers(0, shape[1])] = 1.0
            feeds[name] = arr
        elif name == "lr":
            feeds[name] = np.array([1e-4], dtype=np.float32)
        elif name == "grad_seed":
            feeds[name] = np.array([1.0], dtype=np.float32)
        else:
            feeds[name] = (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def loss_at(overrides):
        f = dict(feeds)
        f.update(overrides)
        out = sess.run(out_names, f)
        return float(np.asarray(out[loss_idx]).reshape(-1)[0])

    base_out = sess.run(out_names, feeds)
    lr = float(feeds["lr"][0])

    eps = 1e-3
    for check_name in ("layer4.0.conv1.weight", "layer4.0.conv2.weight"):
        w = feeds[check_name]
        flat = w.reshape(-1)
        idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)

        num_grad = []
        for i in idx:
            wp, wm = w.copy(), w.copy()
            wp.reshape(-1)[i] += eps
            wm.reshape(-1)[i] -= eps
            lp = loss_at({check_name: wp})
            lm = loss_at({check_name: wm})
            num_grad.append((lp - lm) / (2 * eps))
        num_grad = np.array(num_grad)

        w_next = base_out[out_names.index(state[check_name])]
        an_grad = ((w - w_next) / lr).reshape(-1)[idx]

        cos = float(
            np.dot(an_grad, num_grad)
            / (np.linalg.norm(an_grad) * np.linalg.norm(num_grad) + 1e-30)
        )
        assert cos > 0.99, f"{check_name}: cosine similarity {cos:.5f} too low"
