"""The ResNet18 training-step runner (``scripts/axera/step_runner.py``), the
persistent AXCL session (``axcl_session.py``) and the tinygrad ``AX`` device.

Offline tests need only the committed fixtures. The step graph itself
(``step.onnx``) and the device are local to the Axera box: those tests skip
elsewhere. Device tests run through ``AXCL_LXD_VM`` like every other device
test here, and hold ``/tmp/axcl-device.lock`` while they do.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import misc_op_record_emit as misc  # noqa: E402
import step_runner as sr  # noqa: E402

_HAVE_STEP = os.path.exists(sr.STEP_ONNX) and os.path.exists(sr.STEP_REF)
needs_step = pytest.mark.skipif(
    not _HAVE_STEP, reason="step.onnx is local to the Axera box"
)


def _device_ok() -> bool:
    # pulsar2_docker.axcl_available() does the same, but importing it needs the
    # built onnxsim extension
    import subprocess

    vm = os.environ.get("AXCL_LXD_VM")
    if not vm:
        return False
    try:
        return (
            subprocess.run(
                ["lxc", "exec", vm, "--", "test", "-x", "/usr/bin/axcl/axcl_run_model"],
                capture_output=True,
                timeout=30,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_device = pytest.mark.skipif(
    not _device_ok(), reason="needs the AX650 (AXCL_LXD_VM)"
)


def test_fake_quant_rounds_and_clips():
    x = np.array([-1.0, 0.0, 0.004, 0.006, 10.0], np.float32)
    got = sr.fake_quant(x, 0.01, 100, False)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.55], atol=1e-6)
    got = sr.fake_quant(x, 0.01, 0, True)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.27], atol=1e-6)


def _adam_graph() -> onnx.ModelProto:
    # w' = w - lr * m'; m' = 0.9 m + 0.1 g; g = x * w (a stand-in backward pass)
    return parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[4] x, float[4] w, float[4] w__m) => (float[4] w_new, float[4] m_new)
          <float b1 = {0.9}, float ob1 = {0.1}, float lr = {0.01}> {
            grad = Mul(x, w)
            a = Mul(b1, w__m)
            b = Mul(ob1, grad)
            m_new = Add(a, b)
            step = Mul(lr, m_new)
            w_new = Sub(w, step)
        }
        """
    )


def test_gradient_tensors_follow_the_first_moment_update():
    m = _adam_graph()
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}"
    state_map = {"w": "w_new", "w__m": "m_new"}
    assert sr.gradient_tensors(m, state_map) == {"w": "grad"}
    # the Adam math (not the gradient's own producer) is the optimizer update
    opt = sr.optimizer_nodes(m, state_map)
    by_out = {n.output[0]: n.name for n in m.graph.node}
    assert by_out["grad"] not in opt
    assert {by_out[t] for t in ("a", "b", "m_new", "step", "w_new")} <= opt


def test_misc_emit_keeps_the_mcode_dims_in_step():
    """A zero-point move re-encodes the MCode to a different length. The
    runtime reads the length from the initializer's dims: a stale one fails
    to load (0x80300709) or, inside a long session, wedged the card. This is
    the ReduceSum_62 of the first whole-step run."""
    key = "ReduceSum:16x1x512x4608:axes0:k0"
    tmpl, meta = misc.load_template(key)
    out = misc.emit_model(
        key,
        {"x": 2.1986086721881293e-05, "y": 0.0001273591333301738},
        {"x": 163, "y": 68},
    )
    init = misc.mcode_initializer(out)
    assert len(init.raw_data) != len(misc.mcode_initializer(tmpl).raw_data)
    assert list(init.dims) == [len(init.raw_data)]


@needs_step
def test_plan_covers_the_validated_nodes_and_no_reshape_is_unsafe():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    segs, host = sr.build_plan(model, records, calib)
    everything, _ = sr.build_plan(model, records, calib, include_unsafe=True)
    # a node inside two chains is recomputed by both: count it once
    covered = len({n for s in everything for n in s.nodes})
    assert (
        covered
        == sr.axb.coverage_report(records, calibration=calib)["totals"]["covered"]
    )
    unsafe = [s for s in everything if s.unsafe]
    # signed Reshapes take the Reshape -> Identity templates, so none is unsafe
    assert not any(s.kind == "reshape" for s in unsafe)
    assert len({n for s in segs for n in s.nodes}) == covered - len(
        {n for s in unsafe for n in s.nodes}
    )


@needs_step
def test_plan_materializes_live_broadcast_binary_operands():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segs, _ = sr.build_plan(model, sr.load_records(), calib)
    broadcast = [s for s in segs if s.output_shape]
    assert len(broadcast) == 42
    assert all(s.input_shapes[-1] == (1,) for s in broadcast)
    assert all(s.output_shape == s.input_shapes[0] for s in broadcast)


def test_emission_cache_reuses_a_validated_segment(tmp_path):
    calls = 0

    def emit():
        nonlocal calls
        calls += 1
        return onnx.helper.make_model(onnx.helper.make_graph([], "cached", [], []))

    segment = sr.Segment(
        "cached_segment",
        "test",
        ["cached_node"],
        [],
        [],
        "test",
        emit,
    )
    first, _ = sr.drop_unemittable([segment], {}, str(tmp_path))
    second, _ = sr.drop_unemittable([segment], {}, str(tmp_path))
    assert first and second
    assert calls == 1


@needs_step
def test_float_mode_reproduces_the_reference_step():
    model = sr.load_step()
    ref = sr.load_reference()
    outs, _ = sr.StepRunner(model, []).run(ref["feeds"], "float")
    loss = float(np.ravel(outs["distill__add_27"])[0])
    assert loss == pytest.approx(
        float(np.ravel(ref["ref"]["distill__add_27"])[0]), rel=1e-5
    )


@needs_device
def test_session_health_check_on_device():
    import axcl_session

    with axcl_session.AXSession() as s:
        assert axcl_session.health_check(s) <= 1.01


@needs_device
@needs_step
def test_one_segment_of_each_kind_matches_its_simulation_on_device():
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segs, _ = sr.build_plan(model, sr.load_records(), calib)
    first: dict[str, sr.Segment] = {}
    for s in segs:
        first.setdefault(s.kind, s)
    ref = sr.load_reference()
    with axcl_session.AXSession() as sess:
        runner = sr.StepRunner(model, list(first.values()), sess, health_every=1)
        _, stats = runner.run(ref["feeds"], "npu")
    assert {st.kind for st in stats} == set(first)
    for st in stats:
        assert sr.segment_passed(vars(st)), st


@needs_device
@needs_step
def test_resnet18_training_graph_runs_pulsar_free_on_axcl_vm():
    """Run the complete calibrated ResNet18 training graph on the AX8850."""
    import axcl_session

    model = sr.load_step()
    calibration = sr.axb.load_calibration(sr.STEP_CALIB)
    segments, _ = sr.build_plan(model, sr.load_records(), calibration)
    reference = sr.load_reference()
    with axcl_session.AXSession() as session:
        runner = sr.StepRunner(
            model,
            segments,
            session,
            health_every=1,
        )
        outputs, stats = runner.run(reference["feeds"], "npu")

    assert outputs
    assert stats
    assert {stat.kind for stat in stats} >= {
        "matmul_chain",
        "elementwise",
        "misc",
    }
    # This is a whole-graph transport smoke test.  The focused segment test
    # above keeps the strict <=2-LSB contract; a full training step also
    # contains known calibration/emitter outliers, which must not turn a
    # successful AXCL execution into a false device failure.
    for stat in stats:
        assert not stat.error, stat


@needs_device
def test_tinygrad_ax_device_runs_a_relu_and_a_matmul_chain():
    tinygrad = pytest.importorskip("tinygrad")
    import tinygrad_ax_backend as axb
    from tinygrad.device import Buffer, Device

    axb.register_ax_device()
    dev = Device["AX"]
    classes = axb.tinygrad_classes()
    try:
        # Relu: an AXCompiler request (template + ElementwiseScaleEdit)
        s = 0.01
        key = axb.TemplateKey("Relu", ((16, 512, 7, 7),), calibration_class="x0,y0")
        src = axb.build_request(key, [axb.ElementwiseScaleEdit({"x": s, "y": s})])
        prog = classes["AXProgram"](dev, classes["AXCompiler"]().compile(src))
        x = np.random.default_rng(0).uniform(0, 2, (16, 512, 7, 7)).astype(np.float32)
        xb = Buffer("AX", x.size, tinygrad.dtypes.float32, initial_value=x.tobytes())
        yb = Buffer("AX", x.size, tinygrad.dtypes.float32).allocate()
        prog(yb._buf, xb._buf, wait=True)
        want = np.clip(np.rint(x / np.float32(s)), 0, 255) * np.float32(s)
        assert np.abs(yb.numpy() - want.ravel()).max() <= s * 1.01

        # a live-operand MatMul chain: the step's fc dX (matmul_record_emit)
        if _HAVE_STEP:
            model = sr.load_step()
            calib = sr.axb.load_calibration(sr.STEP_CALIB)
            segs, _ = sr.build_plan(
                model, sr.load_records(), calib, kinds={"matmul_chain"}
            )
            seg = next(g for g in segs if g.name == "MatMul_36")
            prog = classes["AXProgram"](dev, seg.emit().SerializeToString())
            rng = np.random.default_rng(1)
            ins = [
                rng.uniform(-0.02, 0.02, sp.shape).astype(np.float32)
                for sp in prog.model.inputs
            ]
            bufs = [
                Buffer("AX", a.size, tinygrad.dtypes.float32, initial_value=a.tobytes())
                for a in ins
            ]
            out = prog.model.outputs[0]
            ob = Buffer(
                "AX", int(np.prod(out.shape)), tinygrad.dtypes.float32
            ).allocate()
            prog(ob._buf, *[b._buf for b in bufs], wait=True)
            env = dict(zip(seg.inputs, ins))
            sim = sr.StepRunner(model, [seg])._sim(seg, env)[0]
            lsb = np.abs(ob.numpy() - sim.ravel()).max() / seg.out_q[0][0]
            assert lsb <= 2.01
    finally:
        axb.close_ax_session()


@needs_device
def test_onnx_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Exercise the replacement path on the AX8850, including its schedule."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    shape = numpy_helper.from_array(np.asarray([1, 1, 8, 16], dtype=np.int64), "shape")
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Reshape", ["x", "shape"], ["r"]),
                onnx.helper.make_node("Relu", ["r"], ["y"]),
            ],
            "onnx_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, [1, 8, 4, 4]
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, [1, 1, 8, 16]
                )
            ],
            [shape],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / "onnx_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule))
    rng = np.random.default_rng(1965)
    x = rng.uniform(-1.0, 1.0, (1, 8, 4, 4)).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, np.maximum(x.reshape(1, 1, 8, 16), 0.0), atol=0.02, rtol=0
    )


@needs_device
@pytest.mark.parametrize("zero_point", [0, 128])
def test_standalone_relu_uop_to_mcode_runs_on_axcl_vm(tmp_path, zero_point):
    """Run an unfused standalone ReLU UOp through the AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import elementwise_scale_emit as ew
    import tinygrad_ax_backend as axb

    shape = (16, 64, 56, 56)
    _, meta = ew.load_template("Relu", shape, {"x": zero_point, "y": zero_point})
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Relu", ["x"], ["y"])],
            "onnx_standalone_relu_to_uop_vm",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / f"standalone_relu_uop_z{zero_point}.schedule.json"
    axmodel = axb.compile_onnx(
        model,
        str(schedule),
        {"scales": meta["scales"], "zero_points": meta["zero_points"]},
    )
    rng = np.random.default_rng(1965)
    x = rng.uniform(-1.0, 1.0, shape).astype(np.float32)

    with axcl_session.AXSession(subdir=f"uop_relu_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, np.maximum(x, 0.0), atol=meta["scales"]["y"] * 2, rtol=0
    )


@needs_device
def test_onnx_transpose_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a verified real-shape Transpose through the replacement path."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    input_shape, perm = (16, 1, 256, 49), (0, 1, 3, 2)
    output_shape = tuple(input_shape[axis] for axis in perm)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Transpose", ["x"], ["y"], perm=list(perm))],
            "onnx_transpose_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, input_shape
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    schedule = tmp_path / "onnx_transpose_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule))
    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

    with axcl_session.AXSession(subdir=f"transpose_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    np.testing.assert_array_equal(got, np.transpose(x, perm))


@needs_device
def test_onnx_matmul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a calibrated standalone MatMul emitted from an imported ONNX UOp."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import matmul_record_emit as mre
    import tinygrad_ax_backend as axb

    a_shape, b_shape = (16, 1000), (1000, 512)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["x", "z"], ["y"])],
            "onnx_matmul_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, a_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, b_shape
                ),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, (16, 512)
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, quant = mre.STANDALONE_MATMUL_TEMPLATES[(a_shape, b_shape)]
    old = mre.load_scales(os.path.join(mre.STEP_TEMPLATE_DIR, quant))
    names = list(old)
    scales = {
        "x": old[names[0]][0],
        "z": old[names[1]][0],
        "y": old[names[2]][0],
    }
    zero_points = {
        "x": old[names[0]][1],
        "z": old[names[1]][1],
        "y": old[names[2]][1],
    }
    calibration = {"scales": scales, "zero_points": zero_points}
    schedule = tmp_path / "onnx_matmul_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(-0.02, 0.02, a_shape).astype(np.float32)
    z = rng.uniform(-0.02, 0.02, b_shape).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    want = x @ z
    np.testing.assert_allclose(got, want, atol=scales["y"] * 1.5, rtol=0)


@needs_device
@needs_step
def test_training_step_matmul_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run one live-operand MatMul shape taken from the training step."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import tinygrad_ax_backend as axb

    step = sr.load_step()
    calibration = axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    segments, _ = sr.build_plan(step, records, calibration, kinds={"matmul_chain"})
    segment = next(seg for seg in segments if seg.name == "MatMul_36")
    node = next(node for node in step.graph.node if node.name == "MatMul_36")
    if len(node.input) != 2 or len(node.output) != 1:
        pytest.skip("MatMul_36 is not a two-input training MatMul in this step")

    values = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in (*step.graph.input, *step.graph.value_info, *step.graph.output)
    }
    a_shape, b_shape = (values[name] for name in node.input)
    output_shape = values[node.output[0]]
    if not a_shape or not b_shape or not output_shape:
        pytest.skip("MatMul_36 has no static shapes")

    # graph_generator's standalone MatMul contract deliberately uses x/z/y;
    # retain the training node's shapes and calibration, but normalize names.
    imported = onnx.NodeProto()
    imported.CopyFrom(node)
    imported.input[:] = ["x", "z"]
    imported.output[:] = ["y"]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [imported],
            "training_matmul_36_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, a_shape
                ),
                onnx.helper.make_tensor_value_info(
                    "z", onnx.TensorProto.FLOAT, b_shape
                ),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    if len(segment.in_q) != 2 or len(segment.out_q) != 1:
        pytest.skip("MatMul_36 calibration is not a two-input/single-output form")
    matmul_calibration = {
        "scales": {
            "x": segment.in_q[0][0],
            "z": segment.in_q[1][0],
            "y": segment.out_q[0][0],
        },
        "zero_points": {
            "x": segment.in_q[0][1],
            "z": segment.in_q[1][1],
            "y": segment.out_q[0][1],
        },
    }
    schedule = tmp_path / "training_matmul_36.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), matmul_calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(-0.02, 0.02, a_shape).astype(np.float32)
    z = rng.uniform(-0.02, 0.02, b_shape).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got,
        x @ z,
        atol=matmul_calibration["scales"]["y"] * 2.0,
        rtol=0,
    )


@needs_device
def test_onnx_add_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path):
    """Run a calibrated two-input Add emitted from an imported ONNX UOp."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["x", "z"], ["y"])],
            "onnx_add_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template("Add", shape, {"x": 0, "y": 0, "z": 0})
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / "onnx_add_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(0.0, 1.0, shape).astype(np.float32)
    z = rng.uniform(0.0, 1.0, shape).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    np.testing.assert_allclose(
        got, x + z, atol=float(meta["scales"]["y"]) * 1.5, rtol=0
    )


@needs_device
@pytest.mark.parametrize("op", ["Sub", "Mul", "Div"])
def test_onnx_binary_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(tmp_path, op):
    """Run the remaining same-shape binary UOps through AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import binary_op_scale_emit as bse
    import tinygrad_ax_backend as axb

    shape = (1, 64)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["x", "z"], ["y"])],
            f"onnx_{op.lower()}_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape),
                onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, shape),
            ],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = bse.load_template(op, shape, {"x": 0, "y": 0, "z": 0})
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / f"onnx_{op.lower()}_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    x = rng.uniform(0.1, 1.0, shape).astype(np.float32)
    z = rng.uniform(0.2, 1.0, shape).astype(np.float32)

    with axcl_session.AXSession() as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x, z])
        finally:
            session.unload(loaded)

    want = {"Sub": x - z, "Mul": x * z, "Div": x / z}[op]
    np.testing.assert_allclose(got, want, atol=float(meta["scales"]["y"]) * 1.5, rtol=0)


@needs_device
@pytest.mark.parametrize(
    "op, input_shape, output_shape, attrs, template_key",
    [
        (
            "ReduceMean",
            (16, 512, 7, 7),
            (16, 512, 1, 1),
            {"axes": [2, 3], "keepdims": 1},
            "ReduceMean:16x512x7x7:axes2,3:k1",
        ),
        (
            "Softmax",
            (16, 1000),
            (16, 1000),
            {"axis": 1},
            "Softmax:16x1000:axis1",
        ),
        (
            "MaxPool",
            (16, 64, 112, 112),
            (16, 64, 56, 56),
            {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
            "MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1",
        ),
        (
            "ReduceSum",
            (16, 64, 112, 112),
            (64,),
            {"axes": [0, 2, 3], "keepdims": 0},
            "ReduceSum:16x64x112x112:axes0,2,3:k0",
        ),
        (
            "Sqrt",
            (512, 512, 3, 3),
            (512, 512, 3, 3),
            {},
            "Sqrt:512x512x3x3",
        ),
        (
            "Log",
            (16, 1000),
            (16, 1000),
            {},
            "Log:16x1000",
        ),
        (
            "Neg",
            (1, 1),
            (1, 1),
            {},
            "Neg:1x1",
        ),
    ],
)
def test_onnx_misc_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, input_shape, output_shape, attrs, template_key
):
    """Run reduction/normalization UOps used by the training step on AXCL."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import misc_op_record_emit as misc
    import tinygrad_ax_backend as axb

    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node(op, ["x"], ["y"], **attrs)],
            f"onnx_{op.lower()}_to_uop_vm",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, input_shape
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, output_shape
                )
            ],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    _, meta = misc.load_template(template_key)
    calibration = {
        "scales": meta["scales"],
        "zero_points": meta["zero_points"],
    }
    schedule = tmp_path / f"onnx_{op.lower()}_to_uop.schedule.json"
    axmodel = axb.compile_onnx(model, str(schedule), calibration)
    rng = np.random.default_rng(1965)
    bounds = {
        "ReduceSum": (-0.02, 0.02),
        "ReduceMean": (0.0, 1.0),
        "MaxPool": (0.0, 1.0),
        "Sqrt": (0.01, 1.0),
        "Log": (0.1, 1.0),
        "Neg": (-0.02, 0.02),
    }.get(op, (-1.0, 1.0))
    x = rng.uniform(*bounds, input_shape).astype(np.float32)

    # AXCL's virtiofs layer can retain the previous m0.axmodel by pathname;
    # isolate each operator family in its own session directory.
    with axcl_session.AXSession(subdir=f"uop_{op.lower()}_{tmp_path.name}") as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    if op == "ReduceMean":
        want = x.mean(axis=(2, 3), keepdims=True)
    elif op == "MaxPool":
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        want = windows[:, :, ::2, ::2].max(axis=(-1, -2))
    elif op == "ReduceSum":
        want = x.sum(axis=(0, 2, 3))
    elif op == "Sqrt":
        want = np.sqrt(x)
    elif op == "Log":
        want = np.log(x)
    elif op == "Neg":
        want = -x
    else:
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        want = exp / exp.sum(axis=1, keepdims=True)
    tolerance = 4.0 if op == "Sqrt" else 2.0
    np.testing.assert_allclose(
        got, want, atol=float(meta["scales"]["y"]) * tolerance, rtol=0
    )


@needs_device
@pytest.mark.parametrize(
    "op, input_shape, template_key",
    [
        ("Greater", (16, 64, 112, 112), "GreaterCast:16x64x112x112"),
    ],
)
def test_onnx_comparison_to_tinygrad_uop_to_mcode_runs_on_axcl_vm(
    tmp_path, op, input_shape, template_key
):
    """Run calibration-free comparison/cast UOps through AXCL VM."""
    pytest.importorskip("tinygrad")
    import axcl_session
    import misc_op_record_emit as misc
    import tinygrad_ax_backend as axb
    from tinygrad import Tensor

    _, meta = misc.load_template(template_key)
    schedule = tmp_path / f"onnx_{op.lower()}cast_to_uop.schedule.json"
    root = (
        (Tensor.empty(*input_shape) > 0).cast("float32")
        if op == "Greater"
        else (Tensor.empty(*input_shape) < 0).cast("float32")
    ).uop
    axmodel = axb.compile_uop(root, str(schedule))
    rng = np.random.default_rng(1965)
    x = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    with axcl_session.AXSession(
        subdir=f"uop_{op.lower()}cast_{tmp_path.name}"
    ) as session:
        loaded = session.load(axmodel, str(schedule))
        try:
            (got,) = session.run(loaded, [x])
        finally:
            session.unload(loaded)

    want = (x > 0.0 if op == "Greater" else x < 0.0).astype(np.float32)
    np.testing.assert_array_equal(got, want)


class _EchoSession:
    """Stands in for AXSession: records each run's input shapes and returns
    the first input times two."""

    def __init__(self):
        self.calls = []

    def load(self, blob):
        return object()

    def unload(self, m):
        pass

    def run(self, m, ins):
        self.calls.append([x.shape for x in ins])
        return [ins[0] * 2]


def test_batch_split_segment_runs_on_batch_slices_and_concatenates():
    model = parser.parse_model(
        """<ir_version: 8, opset_import: ["" : 13]>
        g (float[4, 3] x, float[5, 3] w) => (float[4, 3] y) {
            y = Identity(x)
        }"""
    )
    seg = sr.Segment(
        "y", "matmul_chain", [model.graph.node[0].name or "n0"], ["x", "w"], ["y"],
        "test", lambda: model, batch_split=2, split=[True, False],
    )  # fmt: skip
    model.graph.node[0].name = seg.nodes[0]
    session = _EchoSession()
    runner = sr.StepRunner(model, [seg], session=session)
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    w = np.ones((5, 3), np.float32)
    (y,) = runner._device(seg, {"x": x, "w": w})
    assert session.calls == [[(2, 3), (5, 3)], [(2, 3), (5, 3)]]
    np.testing.assert_array_equal(y, x * 2)
