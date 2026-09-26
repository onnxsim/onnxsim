import os
import sys

import numpy as np
import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import axcl_session  # noqa: E402


def _model():
    spec = axcl_session.IOSpec("x", 16, np.float32, (1, 4), 1)
    output = axcl_session.IOSpec("y", 16, np.float32, (1, 4), 1)
    return axcl_session.Model(7, "/tmp/test.axmodel", [spec], [output])


def _schedule():
    return {
        "inputs": [{"name": "x", "shape": [1, 4], "elem_type": 1, "nbytes": 16}],
        "outputs": [{"name": "y", "shape": [1, 4], "elem_type": 1, "nbytes": 16}],
        "kernels": [{"name": "kernel_0", "inputs": ["x"], "output": "y"}],
        "allocations": [
            {
                "name": "x",
                "offset": 0,
                "nbytes": 16,
                "first_kernel": 0,
                "last_kernel": 0,
            },
            {
                "name": "y",
                "offset": 64,
                "nbytes": 16,
                "first_kernel": 0,
                "last_kernel": 0,
            },
        ],
        "memory_size": 128,
        "schema_version": 1,
    }


def test_schedule_validation_accepts_matching_model():
    axcl_session._validate_schedule(_model(), _schedule())


def test_schedule_validation_rejects_io_mismatch():
    schedule = _schedule()
    schedule["outputs"][0]["nbytes"] = 32
    with pytest.raises(axcl_session.DeviceError, match="does not match"):
        axcl_session._validate_schedule(_model(), schedule)


def test_schedule_validation_rejects_out_of_bounds_allocation():
    schedule = _schedule()
    schedule["allocations"][1]["offset"] = 128
    with pytest.raises(axcl_session.DeviceError, match="outside"):
        axcl_session._validate_schedule(_model(), schedule)


def test_schedule_validation_rejects_live_allocation_overlap():
    schedule = _schedule()
    schedule["allocations"][1]["offset"] = 0
    with pytest.raises(axcl_session.DeviceError, match="overlap"):
        axcl_session._validate_schedule(_model(), schedule)


def test_schedule_validation_rejects_missing_kernel_buffer_allocation():
    schedule = _schedule()
    schedule["kernels"][0]["inputs"] = ["missing"]
    with pytest.raises(axcl_session.DeviceError, match="kernel buffer"):
        axcl_session._validate_schedule(_model(), schedule)


def test_schedule_validation_rejects_duplicate_kernel_names():
    schedule = _schedule()
    schedule["kernels"].append({"name": "kernel_0", "inputs": ["x"], "output": "y"})
    with pytest.raises(axcl_session.DeviceError, match="duplicate kernel"):
        axcl_session._validate_schedule(_model(), schedule)
