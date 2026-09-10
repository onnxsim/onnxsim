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
    """Simplify with no extra optimizers/rewrite passes and return the sole
    remaining output's folded scalar value, asserting the graph collapsed to
    a pure initializer -- same technique and same assertion as
    test_bev_custom_op_schemas.py's own ``_folded_dim``. The custom-op node
    is not expected to survive here: once the Shape/Gather chain folds to a
    literal, its own output has no remaining consumer, so onnxsim's ordinary
    dead-node elimination removes it along with the rest of the chain -- that
    it *can* be removed is itself downstream of shape inference having
    resolved it in the first place, since an op whose output shape onnxsim
    could not determine would leave the Shape/Gather chain live."""
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert len(sim_model.graph.node) == 0, [n.op_type for n in sim_model.graph.node]
    assert len(sim_model.graph.initializer) == 1
    return int(numpy_helper.to_array(sim_model.graph.initializer[0])[0])


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


def test_quant_node_itself_survives_when_its_value_is_actually_used():
    # Unlike the folding tests above (where Xq's only consumer, Shape, itself
    # folds away and takes the now-unused Quant node with it), a Quant node
    # whose fake-quantized *value* is actually read is never a folding
    # candidate: onnxsim's constant folder only ever considers the default
    # ONNX domain (IsOfficialOp in constant_folding.cpp), so a custom-domain
    # node survives simplification unchanged regardless of whether its
    # inputs are constant -- proof onnxsim treats it as opaque rather than
    # guessing at its semantics.
    model = _model(
        """
        agraph (float[2,3] X) => (float[2,3] Y)
        <float scale = {0.1}, float zeropoint = {0.0}, float bitwidth = {8.0}>
        {
          Xq = qonnx.custom_op.general.Quant<signed=1, narrow=0>(X, scale, zeropoint, bitwidth)
          Y = Identity(Xq)
        }
        """,
        domain="qonnx.custom_op.general",
    )
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    op_types = [(n.op_type, n.domain) for n in sim_model.graph.node]
    assert ("Quant", "qonnx.custom_op.general") in op_types, op_types


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
