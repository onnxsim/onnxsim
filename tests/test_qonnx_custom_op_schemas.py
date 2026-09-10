"""Tests for onnxsim's built-in schemas for QONNX/FINN's fake-quantization
custom ops -- Brevitas's default ONNX export format
(onnxsim/qonnx_schemas.cpp).

Same proof technique as test_bev_custom_op_schemas.py, and for the same
reason: none of these tests call ``onnx.defs.register_schema`` themselves, so
a folded-to-a-literal ``Shape``/``Gather`` output is proof that
``RegisterQonnxCustomOpSchemas()`` -- run internally, with no opt-in -- is
what let shape inference see through the custom op, not anything the test
itself registered.

``Quant``/``BipolarQuant``/``Trunc``/``FloatQuant`` are all "fake-quantize,
then immediately dequantize back to the input's own dtype" ops, so their
shape/type inference is simpler than the BEV ops': the output is always
shaped exactly like the first input. The Shape/Gather chain here is just
`axis=0` of that same shape, which folds to the input's own leading dimension
-- unremarkable on its own, except that it can only fold if shape inference
actually ran the custom op's ``TypeAndShapeInferenceFunction`` rather than
stopping dead at an unresolved shape.
"""

from onnx import numpy_helper, parser

import onnxsim


def _model(body, domain, opset=17, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "{domain}": 1]
        >
        {body}
        """
    )


def _folded_leading_dim(model):
    """Simplify and return the sole remaining output's folded scalar value,
    asserting the graph collapsed to a pure initializer (proof the
    Shape/Gather chain was fully constant-folded, not left as live nodes) --
    while the custom-op node itself survives, proving onnxsim treats it as
    opaque rather than guessing at (and folding away) its semantics."""
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    custom_ops = [
        n.op_type for n in sim_model.graph.node if n.domain not in ("", "ai.onnx")
    ]
    assert len(custom_ops) == 1, custom_ops
    assert len(sim_model.graph.initializer) >= 1
    out_name = sim_model.graph.output[0].name
    out_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    return int(numpy_helper.to_array(out_init)[0])


def test_quant_output_shape_inferred_from_input():
    model = _model(
        """
        agraph (float[2,3,4] X) => (int64[1] out_dim)
        <float scale = {0.1}, float zeropoint = {0.0}, float bitwidth = {8.0}>
        {
          Xq = qonnx.custom_op.general.Quant<signed=1, narrow=0, rounding_mode="ROUND">(X, scale, zeropoint, bitwidth)
          shp = Shape(Xq)
          idx = Constant<value = int64[1] {0}>()
          out_dim = Gather<axis = 0>(shp, idx)
        }
        """,
        domain="qonnx.custom_op.general",
    )
    assert _folded_leading_dim(model) == 2


def test_bipolar_quant_output_shape_inferred_from_input():
    model = _model(
        """
        agraph (float[5,6] X) => (int64[1] out_dim)
        <float scale = {0.5}>
        {
          Xq = qonnx.custom_op.general.BipolarQuant(X, scale)
          shp = Shape(Xq)
          idx = Constant<value = int64[1] {1}>()
          out_dim = Gather<axis = 0>(shp, idx)
        }
        """,
        domain="qonnx.custom_op.general",
    )
    assert _folded_leading_dim(model) == 6


def test_trunc_output_shape_inferred_from_input():
    model = _model(
        """
        agraph (float[7,2,2] X) => (int64[1] out_dim)
        <float scale = {0.1}, float zeropoint = {0.0}, float in_bw = {32.0}, float out_bw = {8.0}>
        {
          Xq = qonnx.custom_op.general.Trunc<rounding_mode="ROUND">(X, scale, zeropoint, in_bw, out_bw)
          shp = Shape(Xq)
          idx = Constant<value = int64[1] {0}>()
          out_dim = Gather<axis = 0>(shp, idx)
        }
        """,
        domain="qonnx.custom_op.general",
    )
    assert _folded_leading_dim(model) == 7


def test_float_quant_output_shape_inferred_from_input():
    model = _model(
        """
        agraph (float[3,9] X) => (int64[1] out_dim)
        <float scale = {1.0}, float ebw = {4.0}, float mbw = {3.0}, float ebias = {7.0}, float maxv = {448.0}>
        {
          Xq = qonnx.custom_op.general.FloatQuant<signed=1, narrow=1>(X, scale, ebw, mbw, ebias, maxv)
          shp = Shape(Xq)
          idx = Constant<value = int64[1] {1}>()
          out_dim = Gather<axis = 0>(shp, idx)
        }
        """,
        domain="qonnx.custom_op.general",
    )
    assert _folded_leading_dim(model) == 9


def test_finn_legacy_domain_also_registered():
    # Older Brevitas/FINN exports use the pre-QONNX-split domain name; it is
    # registered identically (see qonnx_schemas.h's own header comment).
    model = _model(
        """
        agraph (float[4,4] X) => (int64[1] out_dim)
        <float scale = {0.1}, float zeropoint = {0.0}, float bitwidth = {8.0}>
        {
          Xq = finn.custom_op.general.Quant(X, scale, zeropoint, bitwidth)
          shp = Shape(Xq)
          idx = Constant<value = int64[1] {0}>()
          out_dim = Gather<axis = 0>(shp, idx)
        }
        """,
        domain="finn.custom_op.general",
    )
    assert _folded_leading_dim(model) == 4
