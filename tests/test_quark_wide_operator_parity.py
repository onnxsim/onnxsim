"""Non-XINT8 coverage for resizing, strided convolution, constants and Expand."""

import numpy as np
import onnx
import pytest
import test_quark_block_preproc_parity as B
import test_quark_int_presets_parity as I
from onnx import numpy_helper, parser

pytestmark = B.pytestmark
_run_in_tmp_dir = B._run_in_tmp_dir
PRESETS = (*I.PRESETS, "FP16", "BF16", "BFP16", "MXINT8", "MX9")


def _model(case, seed):
    rng = np.random.default_rng(seed)
    shape = (1, 2, 4, 4)
    arrays = {}
    if case == "resize_pool":
        body = 'c=Conv(x,w)\nr=Resize<mode="nearest">(c,"","",size)\ny=MaxPool<kernel_shape=[2,2],strides=[2,2]>(r)'
        arrays = {
            "w": rng.normal(size=(3, 2, 1, 1)).astype(np.float32),
            "size": np.array([1, 3, 8, 8], np.int64),
        }
    elif case == "strided_convtranspose":
        body = "c=Conv<strides=[2,2],pads=[1,1,1,1]>(x,w,b)\ny=ConvTranspose<strides=[2,2],pads=[1,1,1,1],output_padding=[1,1]>(c,t)"
        arrays = {
            "w": rng.normal(size=(3, 2, 3, 3)).astype(np.float32),
            "b": rng.normal(size=(3,)).astype(np.float32),
            "t": rng.normal(size=(3, 2, 3, 3)).astype(np.float32),
        }
    elif case == "constant_weight":
        body = "w=Constant()\ny=Conv(x,w)"
    else:
        shape = (1, 2, 1, 4)
        body = "c=Conv(x,w)\ne=Expand(c,size)\ny=Conv(e,v)"
        arrays = {
            "w": rng.normal(size=(3, 2, 1, 1)).astype(np.float32),
            "size": np.array([1, 3, 4, 4], np.int64),
            "v": rng.normal(size=(2, 3, 1, 1)).astype(np.float32),
        }
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float['
        + ",".join(map(str, shape))
        + "] x) => (float y) {"
        + body
        + "}"
    )
    for n in model.graph.node:
        n.name = n.output[0]
        if n.op_type == "Constant":
            n.attribute.append(
                onnx.helper.make_attribute(
                    "value",
                    numpy_helper.from_array(
                        rng.normal(size=(3, 2, 1, 1)).astype(np.float32)
                    ),
                )
            )
    model.graph.initializer.extend(
        numpy_helper.from_array(a, n) for n, a in arrays.items()
    )
    return model, shape


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize(
    "case", ["resize_pool", "strided_convtranspose", "constant_weight", "expand"]
)
@pytest.mark.parametrize("seed", range(3))
def test_wide_operator_graphs(preset, case, seed, tmp_path):
    model, shape = _model(case, seed)
    data = I.P._data(shape, seed=seed + 41)
    extra = {"SkipPreprocess": True}
    q = B._quark(model, data, tmp_path, preset, extra)
    m = B._mine(model, data, preset, extra)
    if preset in I.PRESETS:
        I.P._assert_same_graph(q, m, f"{preset}/{case}/{seed}")
    else:
        B._assert_same(q, m, f"{preset}/{case}/{seed}")
    B._assert_same_outputs(q, m, data, f"{preset}/{case}/{seed}")
