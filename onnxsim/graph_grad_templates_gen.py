# SPDX-License-Identifier: Apache-2.0
#
# GENERATED FILE -- do not edit by hand. Produced by
#   python3 scripts/codegen/generate_grad_templates.py
# from the onnxscript function definitions in that script; see its
# module docstring for what this is and why it takes no ONNX-level
# attributes.
"""ONNX function text for onnxsim.graph_grad's templated
gradient rules -- parsed back via onnx.parser.parse_function,
never onnxscript itself, which this module does not import."""

from __future__ import annotations

GRAD_ADD = """<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradAdd (g) => (da, db)
{
   [n0] da = Identity (g)
   [n1] db = Identity (g)
}"""

GRAD_BATCH_NORMALIZATION = """<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradBatchNormalization (g, x, mean_b, var_b, scale_b, eps, channel_axes) => (dx, dscale, dbias, dmean, dvar)
{
   [n0] xc = Sub (x, mean_b)
   [n1] const = Constant <value: tensor = float const {1}> ()
   [n2] tmp = CastLike (const, x)
   [n3] tmp_0 = Add (var_b, eps)
   [n4] tmp_1 = Sqrt (tmp_0)
   [n5] inv = Div (tmp, tmp_1)
   [n6] xhat = Mul (xc, inv)
   [n7] gs = Mul (g, scale_b)
   [n8] dx = Mul (gs, inv)
   [n9] tmp_2 = Mul (g, xhat)
   [n10] dscale = ReduceSum <keepdims: int = 0> (tmp_2, channel_axes)
   [n11] dbias = ReduceSum <keepdims: int = 0> (g, channel_axes)
   [n12] tmp_3 = ReduceSum <keepdims: int = 0> (dx, channel_axes)
   [n13] dmean = Neg (tmp_3)
   [n14] tmp_4 = Mul (dx, xhat)
   [n15] tmp_5 = Mul (tmp_4, inv)
   [n16] tmp_6 = ReduceSum <keepdims: int = 0> (tmp_5, channel_axes)
   [n17] const_7 = Constant <value: tensor = float const_7 {-0.5}> ()
   [n18] tmp_8 = CastLike (const_7, x)
   [n19] dvar = Mul (tmp_6, tmp_8)
}"""
