"""`scripts/axera/make_training_calib.py`'s one-hot placement, checked
offline.

A `y` one-hot label scattered into the *flattened* batch tensor via a single
random index is indistinguishable from a correct per-row one-hot at batch
size 1, but at batch>1 leaves most rows an all-zero "label" in every
calibration sample -- degenerate enough to miscalibrate a training step
graph's loss output on real AX650N hardware and clip a real, non-zero loss
down to exactly 0 (`docs/axera-on-device-training-handoff.md`'s "The loss=0
finding" section has the full story, including the real-hardware evidence
that pinned this down). This is the regression test for the fix: every
calibration sample's label input must carry exactly one nonzero entry *per
batch row*, not one total.

Needs neither Docker nor a card.
"""

import io
import os
import sys
import tarfile

import numpy as np
import onnx.parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import make_training_calib as mtc  # noqa: E402


def _step_model(batch, classes):
    return onnx.parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 18]>
        agraph (float[{batch},3,1,1] x, float[{batch},{classes}] y, float[1] lr)
            => (float[{batch},3,1,1] loss)
        {{
            loss = Identity(x)
        }}
        """
    )


def _read_tar_arrays(path):
    with tarfile.open(path) as tf:
        return [np.load(io.BytesIO(tf.extractfile(n).read())) for n in tf.getnames()]


def test_label_one_hot_is_placed_once_per_batch_row(tmp_path):
    batch, classes, n = 4, 10, 6
    step_path = tmp_path / "step.onnx"
    onnx.save(_step_model(batch, classes), str(step_path))

    work_dir = mtc.make_work_dir(str(step_path), str(tmp_path / "work"), seed=0, n=n)

    arrays = _read_tar_arrays(os.path.join(work_dir, "dataset", "y.tar"))
    assert len(arrays) == n
    for arr in arrays:
        assert arr.shape == (batch, classes)
        assert arr.sum() == batch, "one 1.0 total per calibration sample expected"
        per_row_nonzero = [int(np.count_nonzero(row)) for row in arr]
        assert per_row_nonzero == [1] * batch, (
            "every row must carry its own one-hot label, not a single label "
            f"scattered across the flattened tensor: got {per_row_nonzero}"
        )


def test_label_one_hot_still_correct_at_batch_one():
    # the bug this guards against is invisible at batch=1 (a single label
    # scattered into a one-row tensor is trivially a per-row one-hot too) --
    # check the fix doesn't change that trivial case's behaviour.
    batch, classes, n = 1, 10, 3
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        step_path = os.path.join(td, "step.onnx")
        onnx.save(_step_model(batch, classes), step_path)
        work_dir = mtc.make_work_dir(step_path, os.path.join(td, "work"), seed=0, n=n)
        arrays = _read_tar_arrays(os.path.join(work_dir, "dataset", "y.tar"))
        for arr in arrays:
            assert arr.shape == (batch, classes)
            assert int(np.count_nonzero(arr)) == 1


def test_non_label_inputs_are_dense_random():
    batch, classes, n = 2, 5, 3
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        step_path = os.path.join(td, "step.onnx")
        onnx.save(_step_model(batch, classes), step_path)
        work_dir = mtc.make_work_dir(step_path, os.path.join(td, "work"), seed=0, n=n)
        for name in ("x.tar", "lr.tar"):
            arrays = _read_tar_arrays(os.path.join(work_dir, "dataset", name))
            assert len(arrays) == n
        x_arrays = _read_tar_arrays(os.path.join(work_dir, "dataset", "x.tar"))
        assert all(np.count_nonzero(a) == a.size for a in x_arrays), (
            "x should be dense random, not sparse like the label input"
        )
