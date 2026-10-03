"""Float16 input quantization and UseFP32Scale graph conversion parity."""

import numpy as np
import pytest
import test_quark_block_preproc_parity as P
import test_quark_xint8_parity as I
from onnx import numpy_helper, parser

pytestmark = P.pytestmark
_run_in_tmp_dir = P._run_in_tmp_dir


@pytest.mark.parametrize(
    "preset",
    [
        "FP16",
        "BF16",
        "BFP16",
        "MXINT8",
        "BF16_BFP16",
        "A8W8",
        "A16W8",
        "U8S8_AAWS",
        "VINT8",
    ],
)
@pytest.mark.parametrize("fp32_scale", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_fp16_model(preset, fp32_scale, skip, tmp_path):
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 21]> g (float16[1,8] x) => (float16[1,4] y) {y=MatMul(x,w)}'
    )
    model.graph.node[0].name = "mm"
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.random.default_rng(3).normal(size=(8, 4)).astype(np.float16), "w"
        )
    )
    data = [
        {"x": np.random.default_rng(i).normal(size=(1, 8)).astype(np.float16)}
        for i in range(3)
    ]
    extra = {"SkipPreprocess": skip, "UseFP32Scale": fp32_scale}
    q = P._quark(model, data, tmp_path, preset, extra)
    m = P._mine(model, data, preset, extra)
    if preset in ("A8W8", "A16W8", "U8S8_AAWS", "VINT8"):
        I._assert_same_graph(q, m, preset)
    else:
        P._assert_same(q, m, preset)
    if fp32_scale:
        P._assert_same_outputs(q, m, data, preset)


@pytest.mark.parametrize(
    "preset", ["FP16", "BF16", "BFP16", "A8W8", "A16W8", "U8S8_AAWS", "VINT8"]
)
@pytest.mark.parametrize("per_channel", [False, True])
def test_fp16_conv_bias(preset, per_channel, tmp_path):
    if per_channel and preset in ("FP16", "BF16", "BFP16"):
        pytest.skip("Quark rejects per-channel half/block weights")
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 21]> g (float16[1,2,4,4] x) => (float16[1,3,4,4] y) {c=Conv(x,w,b)\ny=Relu(c)}'
    )
    rng = np.random.default_rng(12)
    for name, shape in [("w", (3, 2, 1, 1)), ("b", (3,))]:
        model.graph.initializer.append(
            numpy_helper.from_array(rng.normal(size=shape).astype(np.float16), name)
        )
    for n in model.graph.node:
        n.name = n.output[0]
    data = [{"x": rng.normal(size=(1, 2, 4, 4)).astype(np.float16)} for _ in range(3)]
    extra = {"SkipPreprocess": True, "PerChannel": per_channel}
    # PerChannel is a legacy config field, not an extra option for Quark.
    import contextlib
    import copy
    import io

    import onnx
    from quark.onnx import ModelQuantizer, QConfig

    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    cfg.global_quant_config.include_cle = False
    cfg.global_quant_config.per_channel = per_channel
    cfg.global_quant_config.extra_options.update(extra)
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(cfg).quantize_model(src, dst, P._reader(data))
    q = onnx.load(dst)
    m = P._mine(model, data, preset, extra)
    if preset in ("A8W8", "A16W8", "U8S8_AAWS", "VINT8"):
        I._assert_same_graph(q, m, preset)
    else:
        P._assert_same(q, m, preset)
    P._assert_same_outputs(q, m, data, preset)
