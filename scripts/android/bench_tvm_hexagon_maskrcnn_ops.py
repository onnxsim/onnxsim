#!/usr/bin/env python3
"""Benchmark non-convolution Mask R-CNN operators on a Hexagon DSP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tvm
from test_tvm_hexagon_maskrcnn import (
    _conv_module,
    _conv_transpose_module,
    _cpu_conv,
    _hexagon_target,
    _model_workloads,
    _pool_module,
    _qdq_module,
    _resize_module,
)
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import register_linker
from tvm.rpc.tracker import Tracker
from tvm.topi.testing import roi_align_nchw_python


def _configure_linker():
    toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    if not toolchain:
        return
    clang_link = Path(toolchain) / "bin" / "hexagon-clang++"
    wrapper = Path("/tmp/tvm-hexagon-link-wrapper")
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        f"clang = {str(clang_link)!r}\n"
        "args = ['-Wl,--export-dynamic' if x == '-export-dynamic' else x for x in sys.argv[1:]]\n"
        # Kernels only use C symbols; a dynamic libc++ dependency crashes on the DSP (see
        # relink_hexagon_skel_static_libcxx.sh).
        "raise SystemExit(subprocess.call([clang, '-nostdlib++', *args]))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    register_linker(lambda: str(wrapper))


def _cpu_result(module, arrays, output_specs):
    cpu = tvm.cpu(0)
    inputs = [tvm.nd.array(array, cpu) for array in arrays]
    outputs = [tvm.nd.empty(shape, dtype, cpu) for shape, dtype in output_specs]
    module["main"](*inputs, *outputs)
    return [output.numpy() for output in outputs]


def _workload(op, workloads, rng, dsp_target, cpu_target, roi_batch):
    if op.startswith("conv") and op != "conv_transpose":
        # "conv0".."conv3": ResNet 1x1/3x3, FPN 3x3 and RoI mask-head 3x3 convolution + bias + ReLU.
        name, shape, weight_shape, stride, pad_before, pad_after = workloads["convs"][
            int(op[4:] or 0)
        ]
        dsp_module, output_shape = _conv_module(
            name, shape, weight_shape, stride, pad_before, pad_after, dsp_target
        )
        arrays = [
            rng.normal(0, 0.1, shape).astype("float32"),
            rng.normal(0, 0.05, weight_shape).astype("float32"),
            np.zeros((weight_shape[0],), dtype="float32"),
        ]
        specs = [(output_shape, "float32")]
        expected = [_cpu_conv(*arrays, stride, pad_before, pad_after)]
        compare = lambda got: np.testing.assert_allclose(got[0], expected[0], rtol=2e-3, atol=2e-3)
    elif op == "pool":
        name, shape, kernel, stride, pads = next(
            case for case in workloads["pools"] if case[0] == "backbone_stem_maxpool"
        )
        dsp_module, output_shape = _pool_module(shape, kernel, stride, pads, dsp_target)
        cpu_module, _ = _pool_module(shape, kernel, stride, pads, cpu_target)
        arrays = [rng.normal(size=shape).astype("float32")]
        specs = [(output_shape, "float32")]
        expected = _cpu_result(cpu_module, arrays, specs)
        compare = lambda got: np.testing.assert_array_equal(got[0], expected[0])
    elif op == "resize":
        case = next(case for case in workloads["resizes"] if case[0] == "fpn_resize_14_to_28")
        name, shape, size, coordinate_mode, rounding_mode = case
        dsp_module, output_shape = _resize_module(
            shape, size, coordinate_mode, rounding_mode, dsp_target
        )
        cpu_module, _ = _resize_module(shape, size, coordinate_mode, rounding_mode, cpu_target)
        arrays = [rng.normal(size=shape).astype("float32")]
        specs = [(output_shape, "float32")]
        expected = _cpu_result(cpu_module, arrays, specs)
        compare = lambda got: np.testing.assert_array_equal(got[0], expected[0])
    elif op == "roi":
        case = next(case for case in workloads["rois"] if case[0] == "roi_align_56x56_to_7x7")
        name, shape, pooled_size, scale, sample_ratio, mode = case
        roi_shape = (roi_batch, 5)
        dsp_module, output_shape = _roi_module(
            shape, roi_shape, pooled_size, scale, sample_ratio, mode, dsp_target
        )
        arrays = [rng.normal(0, 0.1, shape).astype("float32"), _make_rois(roi_batch)]
        expected = [
            roi_align_nchw_python(
                arrays[0], arrays[1], pooled_size, scale, sample_ratio, mode=mode.encode()
            )
        ]
        specs = [(output_shape, "float32")]
        compare = lambda got: np.testing.assert_allclose(
            got[0], expected[0], rtol=2e-5, atol=2e-5
        )
    elif op == "deconv":
        name, shape, weight_shape, stride, pads, output_padding = workloads["deconv"]
        dsp_module, output_shape = _conv_transpose_module(
            shape, weight_shape, stride, pads, output_padding, dsp_target
        )
        cpu_module, _ = _conv_transpose_module(
            shape, weight_shape, stride, pads, output_padding, cpu_target
        )
        arrays = [
            rng.normal(0, 0.1, shape).astype("float32"),
            rng.normal(0, 0.05, weight_shape).astype("float32"),
            np.zeros((weight_shape[1],), dtype="float32"),
        ]
        specs = [(output_shape, "float32")]
        expected = _cpu_result(cpu_module, arrays, specs)
        compare = lambda got: np.testing.assert_allclose(
            got[0], expected[0], rtol=2e-3, atol=2e-3
        )
    elif op == "qdq":
        name, shape, scale, zero_point = workloads["qdq"]
        dsp_module = _qdq_module(shape, scale, zero_point, dsp_target)
        arrays = [rng.normal(0, 1.0, shape).astype("float32")]
        specs = [(shape, "uint8"), (shape, "float32")]
        quantized = np.clip(np.rint(arrays[0] / scale) + zero_point, 0, 255).astype("uint8")
        expected = [quantized, (quantized.astype("float32") - zero_point) * scale]

        def compare(got):
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_allclose(got[1], expected[1], rtol=1e-6, atol=1e-5)

    else:
        raise ValueError(f"Unknown operator: {op}")
    return name, dsp_module, arrays, specs, compare


def _make_rois(batch):
    rois = np.zeros((batch, 5), dtype="float32")
    rois[:, 1] = np.arange(batch, dtype="float32") * 3.0
    rois[:, 2] = np.arange(batch, dtype="float32") * 2.0
    rois[:, 3] = np.minimum(rois[:, 1] + 112.0, 223.0)
    rois[:, 4] = np.minimum(rois[:, 2] + 96.0, 223.0)
    return rois


def _roi_module(data_shape, roi_shape, pooled_size, scale, sample_ratio, mode, target):
    data = tvm.te.placeholder(data_shape, name="data", dtype="float32")
    rois = tvm.te.placeholder(roi_shape, name="rois", dtype="float32")
    output = tvm.topi.vision.roi_align_nchw(
        data, rois, pooled_size, scale, mode.encode(), sample_ratio
    )
    schedule = tvm.topi.hexagon.schedule_injective(output)
    module = tvm.build(schedule, [data, rois, output], target=target, name="main")
    return module, tuple(int(dimension) for dimension in output.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--ops", default="pool,resize,roi,deconv,qdq", help="also conv0..conv3")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()
    convs, pools, resizes, rois, deconv, qdq = _model_workloads(args.model, args.roi_batch)
    workloads = {
        "convs": convs,
        "pools": pools,
        "resizes": resizes,
        "rois": rois,
        "deconv": deconv,
        "qdq": qdq,
    }
    target, cpu_target = _hexagon_target(), tvm.target.Target("llvm -mcpu=native")
    tracker = Tracker(host="127.0.0.1", port=9196)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9196,
            "rpc_server_port": 7076,
            "workspace_base": "/data/local/tmp/tvm_hexagon_other_ops",
            "adb_server_socket": None,
        },
    )
    rng = np.random.default_rng(47)
    try:
        launcher.start_server()
        for op in args.ops.split(","):
            name, module, arrays, output_specs, compare = _workload(
                op, workloads, rng, target, cpu_target, args.roi_batch
            )
            path = Path("/tmp") / f"tvm_hexagon_maskrcnn_{name}.so"
            module.save(str(path))
            with launcher.create_session() as session:
                dsp = session.device
                remote = session.load_module(session.upload(str(path), path.name))
                inputs = [tvm.nd.array(array, dsp) for array in arrays]
                outputs = [tvm.nd.empty(shape, dtype, dsp) for shape, dtype in output_specs]
                remote["main"](*inputs, *outputs)
                got = [output.numpy() for output in outputs]
                compare(got)
                samples = remote.time_evaluator(
                    "main", dsp, number=1, repeat=args.repeat
                )(*inputs, *outputs).results
                print(
                    f"{name}: median={np.median(samples) * 1e3:.3f} ms, "
                    f"outputs={[tuple(output.shape) for output in got]}, "
                    "correctness=PASS",
                    flush=True,
                )
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
