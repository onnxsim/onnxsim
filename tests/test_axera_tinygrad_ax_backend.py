"""No-device tests for the template+patch tinygrad backend skeleton.

Every comparison is against committed Pulsar2-built fixtures: the backend's
output must equal what the underlying emitter already reproduces byte for byte
(or within the tolerance that emitter's own tests document).
"""

import dataclasses
import gzip
import json
import os
import struct
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

# The UOp-lowering tests import tinygrad inside the test body (the backend pulls in the Axera emitters
# from sys.path first, so a module-level import would run before that). Skip the module when tinygrad is
# absent instead of letting every one of them fail on the import: the coverage job does not install it.
_tinygrad = pytest.importorskip("tinygrad", reason="tinygrad is not installed")

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import binary_op_scale_emit as bse  # noqa: E402
import binary_op_scale_validate as bsv  # noqa: E402
import elementwise_scale_emit as ew  # noqa: E402
import emitter  # noqa: E402
import llm_build_dtype_analysis as lbd  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures")
_STEP_OPS = os.path.join(_FIX, "tinygrad_ax_backend", "resnet18_step_ops.json.gz")


def _load_gz(path):
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


def _init(model, name):
    return bytes(next(i for i in model.graph.initializer if i.name == name).raw_data)


def _mcode(model):
    return bytes(
        next(i for i in model.graph.initializer if i.name.endswith("_neu")).raw_data
    )


def _outside_noise(mc):
    data = bytearray(mc)
    data[301:326] = bytes(25)
    return bytes(data)


def _gather_key():
    return axb.TemplateKey("Gather", ((1, 1, 4, 16),), (("axis", 3), ("indices", 8)))


def test_template_key_json_round_trip():
    key = axb.TemplateKey(
        "Transpose", ((16, 512),), (("perm", (1, 0)),), calibration_class=""
    )
    assert axb.TemplateKey.from_json(json.loads(json.dumps(key.to_json()))) == key


@pytest.mark.parametrize(
    "key",
    [
        axb.TemplateKey("Mul", ((1, 64, 56, 56),)),
        axb.TemplateKey("Gather", ((1, 1, 4, 16),), (("axis", 3), ("indices", 7))),
        axb.TemplateKey("Transpose", ((16, 513),), (("perm", (1, 0)),)),
        axb.TemplateKey("Relu", ((16, 64, 56, 56),), calibration_class="x1,y1"),
        axb.TemplateKey("Relu", ((16, 64, 56, 56),), dtypes=("int8",)),
        axb.TemplateKey("Relu", ((16, 64, 56, 56),), weight_dtype="s8"),
    ],
)
def test_cache_refuses_unmeasured_keys(key):
    with pytest.raises(ValueError):
        axb.TemplateCache().lookup(key)


def test_cache_miss_is_not_built():
    with pytest.raises(NotImplementedError, match="Pulsar2 build"):
        axb.TemplateCache().get_or_build(axb.TemplateKey("Mul", ((1, 64, 56, 56),)))


def test_cache_generates_same_topology_without_pulsar2(tmp_path):
    template_path = os.path.join(
        _FIX, "compose_gather_reshape_matmul_transpose_add.axmodel.gz"
    )
    template = _load_gz(template_path)
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    axmodel = tmp_path / "template.axmodel"
    output = tmp_path / "generated.axmodel"
    for path in (source, template_source, axmodel):
        onnx.save(template, str(path))

    got = axb.TemplateCache().generate_graph_template(
        str(source), str(template_source), str(axmodel), str(output)
    )

    assert got == str(output)
    assert onnx.load(str(output), load_external_data=False).SerializeToString() == (
        template.SerializeToString()
    )


def test_cache_graph_template_refuses_topology_change(tmp_path):
    template_path = os.path.join(
        _FIX, "compose_gather_reshape_matmul_transpose_add.axmodel.gz"
    )
    template = _load_gz(template_path)
    source = onnx.ModelProto()
    source.CopyFrom(template)
    source.graph.node[0].attribute[0].s = b"different"
    source_path = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    axmodel = tmp_path / "template.axmodel"
    onnx.save(source, str(source_path))
    onnx.save(template, str(template_source))
    onnx.save(template, str(axmodel))

    with pytest.raises(ValueError, match="topology"):
        axb.TemplateCache().generate_graph_template(
            str(source_path),
            str(template_source),
            str(axmodel),
            str(tmp_path / "out.axmodel"),
        )


def test_compiler_request_generates_graph_template_without_pulsar2(tmp_path):
    template_path = os.path.join(
        _FIX, "compose_gather_reshape_matmul_transpose_add.axmodel.gz"
    )
    template = _load_gz(template_path)
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    axmodel = tmp_path / "template.axmodel"
    for path in (source, template_source, axmodel):
        onnx.save(template, str(path))

    request = axb.build_graph_template_request(
        str(source), str(template_source), str(axmodel)
    )
    assert axb.compile_request(request) == template.SerializeToString()


def test_compiler_request_runs_measured_generator_without_pulsar2(tmp_path):
    shape = numpy_helper.from_array(np.array([1, 1, 8, 16], dtype=np.int64), "shape")
    source_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node("Reshape", ["x", "shape"], ["r"]),
                onnx.helper.make_node("Relu", ["r"], ["y"]),
            ],
            "source",
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
    source = tmp_path / "source.onnx"
    output = tmp_path / "generated.axmodel"
    schedule = tmp_path / "generated.schedule.json"
    onnx.save(source_model, source)
    request = axb.build_generated_graph_request(str(source), str(output), str(schedule))
    generated = onnx.load_from_string(axb.compile_request(request))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reshape_relu"


def test_lower_and_compile_tinygrad_reshape_relu_uop_without_pulsar2(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = Tensor.empty(1, 8, 4, 4).reshape(1, 1, 8, 16).relu().uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Reshape", "Relu"]
    schedule = tmp_path / "uop.schedule.json"
    generated = onnx.load_from_string(axb.compile_uop(root, str(schedule)))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reshape_relu"


def test_lower_and_compile_tinygrad_relu_reshape_uop_without_pulsar2(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = Tensor.empty(1, 8, 4, 4).relu().reshape(1, 1, 8, 16).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Relu", "Reshape"]
    schedule = tmp_path / "uop_after.schedule.json"
    generated = onnx.load_from_string(axb.compile_uop(root, str(schedule)))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reshape_relu"


def test_lower_and_compile_tinygrad_add_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = (Tensor.empty(1, 64) + Tensor.empty(1, 64)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Add"]
    _, meta = bse.load_template("Add", (1, 64), {"x": 0, "y": 0, "z": 0})
    schedule = tmp_path / "add.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "add"


def test_lower_and_compile_tinygrad_mul_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = (Tensor.empty(1, 64) * Tensor.empty(1, 64)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Mul"]
    _, meta = bse.load_template("Mul", (1, 64), {"x": 0, "y": 0, "z": 0})
    schedule = tmp_path / "mul.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "mul"


@pytest.mark.parametrize("op", ["sub", "div"])
def test_lower_and_compile_tinygrad_compound_binary_uop_with_explicit_calibration(
    tmp_path, op
):
    Tensor = pytest.importorskip("tinygrad").Tensor

    left, right = Tensor.empty(1, 64), Tensor.empty(1, 64)
    root = (left - right if op == "sub" else left / right).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == [op.title()]
    _, meta = bse.load_template(op.title(), (1, 64), {"x": 0, "y": 0, "z": 0})
    schedule = tmp_path / f"{op}.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == op


def test_lower_and_compile_tinygrad_neg_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = (-Tensor.empty(1, 1)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Neg"]
    _, meta = misc.load_template("Neg:1x1")
    schedule = tmp_path / "neg.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "neg"


@pytest.mark.parametrize(
    "op,shape",
    [("sqrt", (512, 512, 3, 3)), ("log", (16, 1000))],
)
def test_lower_and_compile_tinygrad_misc_uop_with_explicit_calibration(
    tmp_path, op, shape
):
    Tensor = pytest.importorskip("tinygrad").Tensor

    tensor = Tensor.empty(*shape)
    root = (tensor.sqrt() if op == "sqrt" else tensor.log()).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == [op.title()]
    _, meta = misc.load_template(f"{op.title()}:{'x'.join(map(str, shape))}")
    schedule = tmp_path / f"{op}.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == op


def test_lower_and_compile_tinygrad_softmax_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = Tensor.empty(16, 1000).softmax().uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Softmax"]
    _, meta = misc.load_template("Softmax:16x1000:axis1")
    schedule = tmp_path / "softmax.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "softmax"


def test_lower_and_compile_tinygrad_reducemean_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = Tensor.empty(16, 512, 7, 7).mean(axis=(2, 3), keepdim=True).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceMean"]
    _, meta = misc.load_template("ReduceMean:16x512x7x7:axes2,3:k1")
    schedule = tmp_path / "reducemean.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducemean"


def test_lower_and_compile_tinygrad_reducesum_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = Tensor.empty(16, 64, 112, 112).sum(axis=(0, 2, 3)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceSum"]
    _, meta = misc.load_template("ReduceSum:16x64x112x112:axes0,2,3:k0")
    schedule = tmp_path / "reducesum.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducesum"


def test_lower_and_compile_tinygrad_maxpool_uop_with_explicit_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = (
        Tensor.empty(16, 64, 112, 112)
        .max_pool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        .uop
    )
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["MaxPool"]
    _, meta = misc.load_template("MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1")
    schedule = tmp_path / "maxpool.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "maxpool"


def test_lower_and_compile_tinygrad_greatercast_uop_without_calibration(tmp_path):
    Tensor = pytest.importorskip("tinygrad").Tensor

    root = (Tensor.empty(16, 64, 112, 112) > 0).cast("float32").uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Greater", "Cast"]
    schedule = tmp_path / "greatercast.schedule.json"
    generated = onnx.load_from_string(axb.compile_uop(root, str(schedule)))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "greatercast"


def test_lower_and_compile_tinygrad_lesscast_uop_without_calibration(tmp_path):
    from tinygrad import Tensor

    root = (Tensor.empty(1024, 9, 3136) < 0).cast("float32").uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Less", "Cast"]
    schedule = tmp_path / "lesscast.schedule.json"
    generated = onnx.load_from_string(axb.compile_uop(root, str(schedule)))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "lesscast"


def test_lower_and_compile_tinygrad_reducesum_reshaped_uop_with_explicit_calibration(
    tmp_path,
):
    from tinygrad import Tensor

    root = Tensor.empty(16, 1, 64, 3136).sum(axis=(0, 3)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceSum"]
    _, meta = misc.load_template("ReduceSum:16x1x64x3136:axes0,3:k0")
    schedule = tmp_path / "reducesum_reshaped.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducesum"


@pytest.mark.parametrize("axis, output_shape", [(0, (1, 1000)), (1, (16, 1))])
def test_lower_and_compile_tinygrad_classifier_reducesum_uop(
    tmp_path, axis, output_shape
):
    from tinygrad import Tensor

    root = Tensor.empty(16, 1000).sum(axis=axis, keepdim=True).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceSum"]
    key = "ReduceSum:16x1000:axes0:k1" if axis == 0 else "ReduceSum:16x1000:axes1:k1"
    _, meta = misc.load_template(key)
    schedule = tmp_path / f"reducesum_axis{axis}.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducesum"
    assert (
        tuple(
            generated.graph.output[0].type.tensor_type.shape.dim[i].dim_value
            for i in range(2)
        )
        == output_shape
    )


def test_lower_and_compile_tinygrad_transpose_uop_without_pulsar2(tmp_path):
    from tinygrad import Tensor

    root = Tensor.empty(16, 512).permute(1, 0).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["Transpose"]
    schedule = tmp_path / "transpose.schedule.json"
    generated = onnx.load_from_string(axb.compile_uop(root, str(schedule)))
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "transpose"


def test_lower_and_compile_tinygrad_wide_reducesum_uop(tmp_path):
    from tinygrad import Tensor

    root = Tensor.empty(16, 1, 512, 49).sum(axis=(0, 3)).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceSum"]
    _, meta = misc.load_template("ReduceSum:16x1x512x49:axes0,3:k0")
    schedule = tmp_path / "reducesum_wide.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducesum"


def test_generic_reducesum_lowering_selects_indexed_template(tmp_path):
    from tinygrad import Tensor

    root = Tensor.empty(16, 1, 64, 576).sum(axis=0).uop
    lowered = axb.lower_uop_to_onnx(root)
    assert [node.op_type for node in lowered.graph.node] == ["ReduceSum"]
    _, meta = misc.load_template("ReduceSum:16x1x64x576:axes0:k0")
    schedule = tmp_path / "generic_reducesum.schedule.json"
    generated = onnx.load_from_string(
        axb.compile_uop(
            root,
            str(schedule),
            {"scales": meta["scales"], "zero_points": meta["zero_points"]},
        )
    )
    assert [node.op_type for node in generated.graph.node] == ["neu mode"]
    assert json.loads(schedule.read_text())["kernels"][0]["chain"] == "reducesum"


def test_emit_spec_selects_reducesum_mcode_by_shape_and_axes():
    _, meta = misc.load_template("ReduceSum:16x1x64x576:axes0:k0")
    model = misc.emit_spec(
        "ReduceSum",
        (16, 1, 64, 576),
        axes=(0,),
        keepdims=0,
        scales=meta["scales"],
        zero_points=meta["zero_points"],
    )
    assert [node.op_type for node in model.graph.node] == ["neu mode"]


def test_lower_uop_rejects_unvalidated_pattern():
    Tensor = pytest.importorskip("tinygrad").Tensor

    with pytest.raises(ValueError, match="reshape-backed"):
        axb.lower_uop_to_onnx((Tensor.empty(4) + Tensor.empty(4)).uop)


def test_cache_graph_template_bytes_avoids_output_file(tmp_path):
    template_path = os.path.join(
        _FIX, "compose_gather_reshape_matmul_transpose_add.axmodel.gz"
    )
    template = _load_gz(template_path)
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    axmodel = tmp_path / "template.axmodel"
    for path in (source, template_source, axmodel):
        onnx.save(template, str(path))

    got = axb.TemplateCache().generate_graph_template_bytes(
        str(source), str(template_source), str(axmodel)
    )
    assert got == template.SerializeToString()


def test_gather_edit_writes_indices_and_keeps_mcode():
    key = _gather_key()
    template = axb.TemplateCache().load(key)
    indices = [15, 0, 7, 7, 3, 12, 1, 14]
    model = axb.EditSet([axb.GatherIndexEdit(indices)]).build(key)
    table = _init(model, "npu_params")
    assert list(struct.unpack("<8I", table[:32])) == indices
    assert table[32:] == _init(template, "npu_params")[32:]
    assert _mcode(model) == _mcode(template)


@pytest.mark.parametrize("indices", [[0] * 7, [16] + [0] * 7, [-1] + [0] * 7])
def test_gather_edit_refuses_bad_indices(indices):
    with pytest.raises(ValueError):
        axb.EditSet([axb.GatherIndexEdit(indices)]).build(_gather_key())


def test_transpose_template_only_is_the_committed_template():
    key = axb.TemplateKey("Transpose", ((16, 512),), (("perm", (1, 0)),))
    model = axb.EditSet([axb.TemplateOnly()]).build(key)
    template = _load_gz(
        os.path.join(_FIX, "transpose_real", "transpose_real_16x512_perm10.axmodel.gz")
    )
    assert model.SerializeToString() == template.SerializeToString()


def test_template_only_refused_where_an_edit_is_required():
    with pytest.raises(ValueError):
        axb.EditSet([axb.TemplateOnly()]).build(_gather_key())


_ORACLES = os.path.join(ew.TEMPLATE_DIR, "oracles")
with open(os.path.join(_ORACLES, "index.json")) as _f:
    _ORACLE_INDEX = json.load(_f)


@pytest.mark.parametrize(
    "oracle",
    [n for n in sorted(_ORACLE_INDEX) if n.startswith("relu_16x512")]
    + ["sqrt_64x64x3x3_zp0_h2.axmodel.gz"],
)
def test_elementwise_scale_edit_reproduces_held_out_native_build(oracle):
    meta = _ORACLE_INDEX[oracle]
    zp = meta["zero_points"]
    key = axb.TemplateKey(
        meta["op"],
        (tuple(meta["shape"]),),
        calibration_class=f"x{zp['x']},y{zp['y']}",
    )
    model = axb.EditSet([axb.ElementwiseScaleEdit(meta["scales"])]).build(key)
    native = _load_gz(os.path.join(_ORACLES, oracle))
    assert _outside_noise(_mcode(model)) == _outside_noise(_mcode(native))
    assert _init(model, "npu_params") == _init(native, "npu_params")


_BINARY_ORACLES = os.path.join(bse.TEMPLATE_DIR, "oracles")
with open(os.path.join(_BINARY_ORACLES, "index.json")) as _f:
    _BINARY_ORACLE_INDEX = json.load(_f)


@pytest.mark.parametrize(
    "oracle",
    [
        n
        for n in sorted(_BINARY_ORACLE_INDEX)
        if _BINARY_ORACLE_INDEX[n]["shape"] in ([16, 64, 56, 56], [16, 128, 28, 28])
    ],
)
def test_elementwise_scale_edit_reproduces_held_out_binary_build(oracle):
    meta = _BINARY_ORACLE_INDEX[oracle]
    zp = meta["zero_points"]
    key = axb.TemplateKey(
        meta["op"],
        (tuple(meta["shape"]),),
        calibration_class=",".join(f"{k}{v}" for k, v in sorted(zp.items())),
    )
    model = axb.EditSet([axb.ElementwiseScaleEdit(meta["scales"])]).build(key)
    native = _load_gz(os.path.join(_BINARY_ORACLES, oracle))
    r = bsv.compare(model, native)
    assert r["params"] and r["segments"], r


def test_relu_tile_prediction_matches_template():
    key = axb.TemplateKey("Relu", ((16, 128, 28, 28),), calibration_class="x0,y0")
    template = axb.TemplateCache().load(key)
    assert axb.predicted_npu_params("Relu", (16, 128, 28, 28)) == _init(
        template, "npu_params"
    )


def _npy_gz(name):
    with gzip.open(os.path.join(_FIX, "conv_weight_learn", name), "rb") as f:
        return np.load(f)


def test_conv_weight_edit_stage1_matches_native_code_region():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (1, 1, 1, 1)), ("strides", (1, 1)), ("w", (64, 64, 3, 3))),
    )
    w, b = _npy_gz("holdout_w.npy.gz"), _npy_gz("holdout_b.npy.gz")
    # holdout calibration recorded in docs/axera-conv-weight-learn-stem.md
    edit = axb.ConvWeightEdit(
        w, b, 0.007058821618556976, 127.0, 0.03239550068974495, 125.0
    )
    model = axb.EditSet([edit]).build(key)
    native = _load_gz(
        os.path.join(_FIX, "conv_weight_learn", "holdout_native.axmodel.gz")
    )
    got = np.frombuffer(_init(model, "npu_params"), np.uint8)
    want = np.frombuffer(_init(native, "npu_params"), np.uint8)
    assert np.array_equal(got[: w.size], want[: w.size])
    block = slice(36864, 36864 + 512)
    assert (
        np.abs(got[block].view(np.float32) - want[block].view(np.float32)).max() < 2e-4
    )


def _quant(name, tensor):
    with gzip.open(os.path.join(_FIX, "conv_learn_downsample", name), "rt") as f:
        doc = json.load(f)
    tc = list(doc["tensor_configs"].values())[0][tensor]
    v = doc["values"][str(tc["hash"])]
    return float(v["scale"][0]), float(v["zero_point"][0])


def test_conv_weight_edit_1x1_downsample_matches_native_outside_block():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (0, 0, 0, 0)), ("strides", (2, 2)), ("w", (128, 64, 1, 1))),
    )
    wts = np.load(os.path.join(_FIX, "conv_learn_downsample", "s1_holdout_weights.npz"))
    xs, xz = _quant("s1_reference_quant.json.gz", "x")
    ys, yz = _quant("s1_holdout_quant.json.gz", "y")
    model = axb.EditSet([axb.ConvWeightEdit(wts["w"], wts["b"], xs, xz, ys, yz)]).build(
        key
    )
    native = _load_gz(
        os.path.join(_FIX, "conv_learn_downsample", "s1_holdout_native.axmodel.gz")
    )
    got = np.frombuffer(_init(model, "npu_params"), np.uint8)
    want = np.frombuffer(_init(native, "npu_params"), np.uint8)
    outside = np.ones(len(got), bool)
    outside[9216 : 9216 + 1024] = False
    assert np.array_equal(got[outside], want[outside])


def test_conv_weight_edit_stem_routes_its_shape_specific_mcode_emitter():
    key = axb.TemplateKey(
        "Conv",
        ((16, 3, 224, 224),),
        (("pads", (3, 3, 3, 3)), ("strides", (2, 2)), ("w", (64, 3, 7, 7))),
    )
    stem = os.path.join(_FIX, "conv_learn_stem")
    weights = np.load(os.path.join(stem, "holdout_weights.npz"))

    def q(path, tensor):
        with gzip.open(os.path.join(stem, path), "rt") as f:
            doc = json.load(f)
        for cfg in doc["tensor_configs"].values():
            if tensor in cfg:
                value = doc["values"][str(cfg[tensor]["hash"])]
                return float(value["scale"][0]), float(value["zero_point"][0])
        raise KeyError(tensor)

    xs, xz = q("reference_quant.json.gz", "x")
    ys, yz = q("holdout_quant.json.gz", "y")
    model = axb.EditSet(
        [axb.ConvWeightEdit(weights["w"], weights["b"], xs, xz, ys, yz)]
    ).build(key)
    native = _load_gz(os.path.join(stem, "holdout_native.axmodel.gz"))
    got = np.frombuffer(_init(model, "npu_params"), np.uint8)
    want = np.frombuffer(_init(native, "npu_params"), np.uint8)
    assert np.array_equal(got[: weights["w"].size], want[: weights["w"].size])
    block = slice(18432, 18432 + 512)
    assert (
        np.max(np.abs(got[block].view(np.float32) - want[block].view(np.float32)))
        < 1e-3
    )
    assert len(_mcode(model)) == len(_mcode(native))


def test_conv_weight_edit_refuses_wrong_weight_shape():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (1, 1, 1, 1)), ("strides", (1, 1)), ("w", (64, 64, 3, 3))),
    )
    w = np.zeros((64, 64, 1, 1), np.float32)
    with pytest.raises(ValueError):
        axb.EditSet(
            [axb.ConvWeightEdit(w, np.zeros(64, np.float32), 1, 0, 1, 0)]
        ).build(key)


def test_compile_request_round_trip():
    key = _gather_key()
    src = axb.build_request(key, [axb.GatherIndexEdit(list(range(8)))])
    out = axb.compile_request(src)
    assert axb.gather_indices_of(out, 8) == list(range(8))


def _step_records():
    with gzip.open(_STEP_OPS, "rt") as f:
        return json.load(f)


# misc_op_record_emit: Greater 18 + Less 1 + Cast 19 covered; ReduceSum 44
# (one through a same-bytes equivalent), Sqrt [512,512,3,3] 3, Softmax 3,
# Log 2, MaxPool 1, ReduceMean 1 and Neg 2 conditional
_MISC_COVERED = 38
_MISC_CONDITIONAL = 56
# the Squeeze takes the Reshape step template of the same shapes
_SQUEEZE_CONDITIONAL = 1


def _step_reshape_templated():
    """Non-fused step Reshapes with a validated step template
    (fixtures/reshape_step_templates/manifest.json)."""
    n = 0
    for r in _step_records():
        if r["op"] != "Reshape":
            continue
        try:
            axb.rre.step_template(r["shapes"][0], r["attrs"]["out"])
        except ValueError:
            continue
        n += 1
    return n


def test_coverage_report_on_the_resnet18_step():
    report = axb.coverage_report(_step_records())
    assert report["nodes"] == 1104
    assert report["per_op"]["Gather"] == {"covered": 41}
    assert report["per_op"]["Transpose"] == {"covered": 41}
    assert report["per_op"]["Relu"] == {"conditional": 17}
    # the 3 Sqrt [512,512,3,3] and the step-shape ReduceSum / Greater / Less ->
    # Cast nodes come from misc_op_record_emit.py; Greater/Less -> Cast is not
    # quantized, so its template is the whole program
    assert report["per_op"]["Sqrt"] == {"conditional": 42}
    assert report["per_op"]["ReduceSum"] == {"conditional": 44}
    assert report["per_op"]["Greater"] == {"covered": 18}
    assert report["per_op"]["Less"] == {"covered": 1}
    assert report["per_op"]["Cast"] == {"covered": 19}
    for op, n in (("Softmax", 3), ("Log", 2), ("MaxPool", 1), ("ReduceMean", 1)):
        assert report["per_op"][op] == {"conditional": n}
    assert report["per_op"]["Neg"] == {"conditional": 2}
    assert report["per_op"]["Squeeze"] == {"conditional": 1}
    man = axb.mre.step_manifest()
    live_conv = sum(
        man["templates"][e["template"]]["kind"] == "conv" for e in man["nodes"].values()
    )
    want = {"refused": 20 - live_conv, "conditional": live_conv}
    assert report["per_op"]["Conv"] == {k: v for k, v in want.items() if v}
    # same-shape binary ops: ElementwiseScaleEdit (binary_op_scale_emit.py);
    # constant and broadcast operands stay refused
    assert report["per_op"]["Add"] == {"conditional": 101, "refused": 43}
    assert report["per_op"]["Sub"] == {"conditional": 42, "refused": 4}
    assert report["per_op"]["Mul"] == {"conditional": 63, "refused": 334}
    assert report["per_op"]["Div"] == {"conditional": 44, "refused": 8}
    # Live-operand MatMul/Gemm/Conv nodes with a step template move from
    # refused to conditional (fixtures/matmul_step_templates/manifest.json).
    live = len(axb.mre.step_manifest()["nodes"])
    rs = _step_reshape_templated()
    want_rs = {"conditional": 18 + rs, "refused": 152 - rs}
    assert report["per_op"]["Reshape"] == {k: v for k, v in want_rs.items() if v}
    assert report["totals"] == {
        "covered": 82 + _MISC_COVERED,
        "conditional": 324 + live + _MISC_CONDITIONAL + rs + _SQUEEZE_CONDITIONAL,
        "refused": 698
        - live
        - _MISC_COVERED
        - _MISC_CONDITIONAL
        - rs
        - _SQUEEZE_CONDITIONAL,
    }


def test_trainable_conv_is_refused_even_with_a_template():
    # Every step Conv now has a live-weight chain template (planned through
    # matmul_record_emit); one without such a template is still refused, the
    # frozen-weight Conv template notwithstanding.
    rec = next(
        r
        for r in _step_records()
        if r["op"] == "Conv" and r["attrs"]["w"] == [64, 64, 3, 3]
    )
    assert axb.plan_node(rec)[0] == "conditional"
    status, detail = axb.plan_node({**rec, "name": "Conv_without_live_template"})
    assert status == "refused" and "graph input" in detail


def test_tinygrad_compiler_seam():
    pytest.importorskip("tinygrad")
    classes = axb.tinygrad_classes()
    src = axb.build_request(_gather_key(), [axb.GatherIndexEdit(list(range(8)))])
    out = classes["AXCompiler"]().compile_cached(src)
    assert out == axb.compile_request(src)
    pytest.importorskip("tinygrad")
    from tinygrad.device import Allocator, Program

    assert issubclass(classes["AXProgram"], Program)
    assert issubclass(classes["AXAllocator"], Allocator)


def test_ax_allocator_is_host_visible():
    """AX buffers are host-staged: tinygrad reads and writes them without a
    copy program, so a covered op's inputs/outputs need no device for this."""
    pytest.importorskip("tinygrad")
    pytest.importorskip("tinygrad")
    from tinygrad.device import Buffer
    from tinygrad.dtype import dtypes

    axb.register_ax_device()
    x = np.arange(12, dtype=np.float32)
    buf = Buffer("AX", 12, dtypes.float32, initial_value=x.tobytes())
    assert np.array_equal(buf.numpy(), x)


# --------------------------------------------------------------------------
# Weight dtype selection
# --------------------------------------------------------------------------

_LLM_FIX = os.path.join(_FIX, "llm_build_dtype_analysis")
_Q_PROJ = lbd.tiny_llama_weights(512)["model.layers.0.self_attn.q_proj.weight"]
_STAGE1 = axb.TemplateKey(
    "Conv",
    ((16, 64, 56, 56),),
    (("pads", (1, 1, 1, 1)), ("strides", (1, 1)), ("w", (64, 64, 3, 3))),
)


class _FakeDType:
    """Stands in for a tinygrad ``DType`` (only ``name``/``count`` are read)."""

    def __init__(self, name, count=1):
        self.name, self.count = name, count


@pytest.mark.parametrize(
    "dtype,want",
    [
        (_FakeDType("signed char"), "s8"),
        (_FakeDType("__bf16"), "bf16"),
        (_FakeDType("half"), "fp16"),
        (_FakeDType("float"), "fp32"),
        (_FakeDType("float8_e4m3"), "fp8_e4m3"),
        (_FakeDType("float8_e5m2"), "fp8_e5m2"),
        ("int4", "s4"),
        ("int8", "s8"),
        ("bfloat16", "bf16"),
        ("fp8e5m2", "fp8_e5m2"),
    ],
)
def test_axera_weight_dtype_maps_tinygrad_names(dtype, want):
    assert axb.axera_weight_dtype(dtype) == want


@pytest.mark.parametrize(
    "dtype",
    [
        _FakeDType("float8_e4m3fnuz"),  # different encoding from llm_build's e4m3
        _FakeDType("unsigned char"),  # both paths store signed symmetric codes
        _FakeDType("signed char", count=4),  # vector dtype
        "float64",
        "uint4",
    ],
)
def test_axera_weight_dtype_refuses_the_rest(dtype):
    with pytest.raises(ValueError):
        axb.axera_weight_dtype(dtype)


def test_axera_weight_dtype_on_the_real_tinygrad_fork():
    tinygrad = pytest.importorskip("tinygrad")
    dtypes = tinygrad.dtypes
    got = [
        axb.axera_weight_dtype(d)
        for d in (dtypes.int8, dtypes.float16, dtypes.bfloat16, dtypes.float32)
    ]
    assert got == ["s8", "fp16", "bf16", "fp32"]
    assert axb.axera_weight_dtype(dtypes.fp8e4m3) == "fp8_e4m3"


@pytest.mark.parametrize(
    "path,op,dtype,shape,match",
    [
        ("build", "Conv", "s4", None, "does not offer"),
        ("build", "Conv", "bf16", None, "does not offer"),
        ("build", "Conv", "fp32", None, "needs a Pulsar2 build"),
        ("build", "MatMul", "s8", None, "not decoded"),
        ("build", "Conv", "s8", (64, 64, 5, 5), "no validated s8 Conv template"),
        ("llm_build", "Linear", "fp32", None, "not decoded"),
        ("llm_build", "Linear", "s8", (48, 256), "multiple of 32"),
        ("llm_build", "Linear", "s4", (64, 300), "in_features"),
        ("llm_build", "Linear", "fp16", (64, 2048), "in_features"),
        ("npu", "Linear", "s8", None, "unknown Pulsar2 path"),
    ],
)
def test_validate_weight_choice_refuses_undecoded(path, op, dtype, shape, match):
    with pytest.raises(ValueError, match=match):
        axb.validate_weight_choice(path, op, dtype, shape)


@pytest.mark.parametrize("dtype", ["s8", "s4", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"])
def test_encode_weight_matches_llm_build_bytes(dtype):
    got = np.load(os.path.join(_LLM_FIX, "blocks.npz"))[f"q_proj_block0_{dtype}"]
    blocks = axb.encode_weight(_Q_PROJ, dtype, path="llm_build")
    assert len(blocks) == len(_Q_PROJ) // 32
    assert blocks[0][: len(got)] == got.tobytes()


def test_encode_weight_build_conv_is_the_s8_codes():
    w = np.random.RandomState(4).randn(64, 64, 3, 3).astype(np.float32)
    (codes,) = axb.encode_weight(w, "int8", path="build")
    assert codes == emitter.codes_of(w).tobytes()
    with pytest.raises(ValueError, match="needs a Pulsar2 build"):
        axb.encode_weight(w, "float32", path="build")


def test_weight_dtype_costs_rank_the_llm_build_types():
    costs = {c["dtype"]: c for c in axb.weight_dtype_costs(_Q_PROJ)}
    assert list(costs) == ["s4", "s8", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]
    assert costs["s4"]["bytes_per_param"] < costs["s8"]["bytes_per_param"] < 4.0
    assert {costs[d]["bytes_per_param"] for d in lbd.FLOAT_TYPES} == {4.0}
    err = [costs[d]["rel_rms_error"] for d in ("fp16", "bf16", "s8", "s4")]
    assert err == sorted(err)
    # the byte count is the real stored block size
    assert costs["s8"]["bytes_per_param"] == pytest.approx(
        len(lbd.encode_block(_Q_PROJ[:32], "s8")) / (32 * 256)
    )


def test_choose_weight_dtype_is_smallest_within_budget():
    assert axb.choose_weight_dtype(_Q_PROJ, 0.5) == "s4"
    assert axb.choose_weight_dtype(_Q_PROJ, 0.05) == "s8"
    # every float type stores 4 B/param, so the most precise one wins
    assert axb.choose_weight_dtype(_Q_PROJ, 1e-3) == "fp16"
    with pytest.raises(ValueError, match="no validated dtype"):
        axb.choose_weight_dtype(_Q_PROJ, 1e-9)
    # fp8 stores the same 4 B/param as bf16 on llm_build, so it never wins
    assert axb.choose_weight_dtype(_Q_PROJ, 0.5, dtypes=["fp8e4m3", "bf16"]) == "bf16"


def test_quant_policy_overrides_and_auto():
    policy = axb.QuantPolicy(
        default="auto",
        path="llm_build",
        error_budget=0.05,
        overrides={"MatMul": "s4", "lm_head": _FakeDType("__bf16")},
    )
    assert policy.choice_for("MatMul") == "s4"
    assert policy.choice_for("MatMul", "lm_head") == "bf16"
    assert policy.choice_for("Gemm") == "auto"
    assert policy.resolve("Gemm", _Q_PROJ) == "s8"
    assert policy.resolve("MatMul", _Q_PROJ, node="lm_head") == "bf16"
    assert axb.QuantPolicy.from_json(policy.to_json()) == policy
    with pytest.raises(ValueError, match="error_budget"):
        axb.QuantPolicy(default="auto")
    with pytest.raises(ValueError, match="does not offer"):
        axb.QuantPolicy(default="s4", path="build")


def test_conv_template_key_carries_the_weight_dtype():
    cache = axb.TemplateCache()
    assert (
        cache.lookup(_STAGE1).path
        == cache.lookup(dataclasses.replace(_STAGE1, weight_dtype="s8")).path
    )
    with pytest.raises(ValueError, match="needs a Pulsar2 build"):
        cache.lookup(dataclasses.replace(_STAGE1, weight_dtype="fp32"))
    key = axb.apply_policy(_STAGE1, axb.QuantPolicy(default="int8"))
    assert key.weight_dtype == "s8"
    assert axb.TemplateKey.from_json(json.loads(json.dumps(key.to_json()))) == key
    with pytest.raises(ValueError, match="needs a Pulsar2 build"):
        axb.apply_policy(_STAGE1, axb.QuantPolicy(default="float32"))
    with pytest.raises(ValueError, match="policy says"):
        axb.apply_policy(
            dataclasses.replace(_STAGE1, weight_dtype="fp32"),
            axb.QuantPolicy(default="s8"),
        )
    with pytest.raises(ValueError, match="no llm_build engine templates"):
        axb.apply_policy(_STAGE1, axb.QuantPolicy(default="s8", path="llm_build"))
    # ops without a weight pass through untouched
    assert axb.apply_policy(_gather_key(), axb.QuantPolicy()) == _gather_key()


def test_compile_request_applies_the_policy():
    src = axb.build_request(_gather_key(), [axb.GatherIndexEdit(list(range(8)))])
    policy = axb.QuantPolicy(default="s8")
    assert axb.compile_request(src, policy=policy) == axb.compile_request(src)


def test_coverage_report_weight_dtypes_on_the_resnet18_step():
    policy = axb.QuantPolicy(default="s8")
    report = axb.coverage_report(_step_records(), policy)
    # Live-operand MatMul/Gemm/Conv nodes with a step template move from
    # refused to conditional (fixtures/matmul_step_templates/manifest.json).
    live = len(axb.mre.step_manifest()["nodes"])
    rs = _step_reshape_templated()
    want_rs = {"conditional": 18 + rs, "refused": 152 - rs}
    assert report["per_op"]["Reshape"] == {k: v for k, v in want_rs.items() if v}
    assert report["totals"] == {
        "covered": 82 + _MISC_COVERED,
        "conditional": 324 + live + _MISC_CONDITIONAL + rs + _SQUEEZE_CONDITIONAL,
        "refused": 698
        - live
        - _MISC_COVERED
        - _MISC_CONDITIONAL
        - rs
        - _SQUEEZE_CONDITIONAL,
    }
    assert len(report["per_node"]) == 1104
    # no weight in the training step is a constant: a weight dtype choice
    # changes none of its nodes
    assert all(n["weight_dtype"] is None for n in report["per_node"])
    runtime = (
        "None: weight is {}: a runtime tensor, quantized with the activation "
        "calibration; no weight dtype applies"
    )
    assert report["weight_dtypes"] == {
        "Conv": {runtime.format("graph input"): 20},
        "Gemm": {runtime.format("graph input"): 1},
        "MatMul": {runtime.format("computed"): 40, runtime.format("graph input"): 1},
    }


def _frozen_model(tmp_path):
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[16, 64, 56, 56] x, float[64, 256] a) => (float[16, 64, 56, 56] y,
                                                         float[64, 64] m) {
            y = Conv <pads = [1, 1, 1, 1], strides = [1, 1]> (x, cw, cb)
            m = MatMul (a, lw)
        }
        """
    )
    rng = np.random.RandomState(5)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(rng.randn(64, 64, 3, 3).astype(np.float32), "cw"),
            numpy_helper.from_array(rng.randn(64).astype(np.float32), "cb"),
            numpy_helper.from_array(rng.randn(256, 64).astype(np.float32), "lw"),
        ]
    )
    path = str(tmp_path / "frozen.onnx")
    onnx.save(model, path)
    return axb.extract_step_ops(path)


def test_weight_dtypes_on_frozen_weights(tmp_path):
    conv, matmul = _frozen_model(tmp_path)
    assert conv["attrs"]["weight_source"] == "initializer"
    assert matmul["attrs"]["weight_source"] == "initializer"
    assert matmul["attrs"]["w"] == [256, 64]
    assert axb.plan_node(conv) == ("covered", "ConvWeightEdit")
    build = axb.QuantPolicy(default="s8")
    assert axb.weight_dtype_for_record(conv, build)["weight_dtype"] == "s8"
    assert "not decoded" in axb.weight_dtype_for_record(matmul, build)["note"]
    llm = axb.QuantPolicy(default="s4", path="llm_build", overrides={"Conv": "s8"})
    assert axb.weight_dtype_for_record(matmul, llm) == {
        "weight_dtype": "s4",
        "note": "validated",
    }
    assert "llm_build has no Conv" in axb.weight_dtype_for_record(conv, llm)["note"]
    bf16 = axb.QuantPolicy(default="bfloat16", path="llm_build")
    assert axb.weight_dtype_for_record(matmul, bf16)["weight_dtype"] == "bf16"


def test_tinygrad_compiler_takes_a_weight_dtype():
    tinygrad = pytest.importorskip("tinygrad")
    classes = axb.tinygrad_classes()
    comp = classes["AXCompiler"](weight_dtype=tinygrad.dtypes.int8)
    assert comp.policy.default == "s8"
    src = axb.build_request(_gather_key(), [axb.GatherIndexEdit(list(range(8)))])
    assert comp.compile(src) == axb.compile_request(src)
    bad = classes["AXCompiler"](weight_dtype=tinygrad.dtypes.float32)
    with pytest.raises(ValueError, match="needs a Pulsar2 build"):
        bad.compile(axb.build_request(_STAGE1, [axb.TemplateOnly()], node="conv0"))
