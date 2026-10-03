"""Integer activations with half/block constant formats against Quark."""

import contextlib
import io
import warnings

import numpy as np
import onnx
import pytest
import test_quark_amp_parity as A
import test_quark_block_preproc_parity as P
import test_quark_xint8_parity as I
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc

pytestmark = P.pytestmark
_run_in_tmp_dir = P._run_in_tmp_dir


@pytest.mark.parametrize("act", ["Int8Spec", "UInt8Spec", "Int16Spec", "UInt16Spec"])
@pytest.mark.parametrize(
    "weight", ["Float16Spec", "BFloat16Spec", "BFP16Spec", "MXInt8Spec"]
)
@pytest.mark.parametrize("conv", [False, True, "gather"])
@pytest.mark.parametrize("int32_bias", [False, True])
def test_integer_activations_fake_weights(act, weight, conv, int32_bias, tmp_path):
    if conv == "gather":
        text = '<ir_version: 9, opset_import: ["": 21]> g (int64[2] x) => (float[4,2] y) {g=Gather(w,x)\ny=Transpose<perm=[1,0]>(g)}'
        shape = (2,)
        wshape = (8, 4)
    elif conv:
        text = '<ir_version: 9, opset_import: ["": 21]> g (float[1,2,4,4] x) => (float[1,3,4,4] y) { y=Conv(x,w,b) }'
        shape = (1, 2, 4, 4)
        wshape = (3, 2, 1, 1)
    else:
        text = '<ir_version: 9, opset_import: ["": 21]> g (float[1,8] x) => (float[1,4] y) { y=MatMul(x,w) }'
        shape = (1, 8)
        wshape = (8, 4)
    model = parser.parse_model(text)
    model.graph.node[0].name = "layer"
    rng = np.random.default_rng(3)
    model.graph.initializer.append(
        numpy_helper.from_array(rng.normal(size=wshape).astype(np.float32), "w")
    )
    if conv is True:
        model.graph.initializer.append(
            numpy_helper.from_array(rng.normal(size=(3,)).astype(np.float32), "b")
        )
    data = [{"x": rng.normal(size=shape).astype(np.float32)} for _ in range(3)]
    if conv == "gather":
        data = [{"x": np.array([i, 7 - i], np.int64)} for i in range(3)]
    extra = {
        "SkipPreprocess": True,
        "ForceQuantizeNoInputCheck": False,
        "Int32Bias": int32_bias,
    }
    import quark.onnx as real

    cfg = real.QConfig(
        global_config=A._target((act, weight), real, A.quark_spec), extra_options=extra
    )
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        real.ModelQuantizer(cfg).quantize_model(src, dst, P._reader(data))
    q = onnx.load(dst)
    cfg = qc.QConfig(
        global_config=A._target((act, weight), qc, qc), extra_options=extra
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=P._reader(data)
        )
    I._assert_same_graph(q, m, f"{act}/{weight}")
    P._assert_same_outputs(q, m, data, f"{act}/{weight}")
