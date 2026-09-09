// SPDX-License-Identifier: Apache-2.0
//
// GENERATED FILE -- do not edit by hand. Produced by
//   python3 scripts/codegen/generate_grad_templates.py
// from the onnxscript function definitions in that script; see its
// module docstring for what this is and why it takes no ONNX-level
// attributes. graph_grad_templates_gen.py is the same text for the
// Python side -- both are produced from the same entries so they
// cannot drift from each other.
#ifndef ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_
#define ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_

// No enclosing namespace -- graph_grad.cpp, this header's only consumer, has
// none either (it mirrors graph_grad.py's flat module directly).
constexpr const char* kGradAddTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradAdd (g) => (da, db)
{
   [n0] da = Identity (g)
   [n1] db = Identity (g)
})GRAD_TPL";

constexpr const char* kGradBatchNormalizationTemplate = R"GRAD_TPL(<
  domain: "onnxsim.grad",
  opset_import: ["" : 17]
>
GradBatchNormalization (g, x, mean_b, var_b, scale_b, eps, channel_axes, one, neg_half) => (dx, dscale, dbias, dmean, dvar)
{
   [n0] xc = Sub (x, mean_b)
   [n1] tmp = Add (var_b, eps)
   [n2] tmp_0 = Sqrt (tmp)
   [n3] inv = Div (one, tmp_0)
   [n4] xhat = Mul (xc, inv)
   [n5] gs = Mul (g, scale_b)
   [n6] dx = Mul (gs, inv)
   [n7] tmp_1 = Mul (g, xhat)
   [n8] dscale = ReduceSum <keepdims: int = 0> (tmp_1, channel_axes)
   [n9] dbias = ReduceSum <keepdims: int = 0> (g, channel_axes)
   [n10] tmp_2 = ReduceSum <keepdims: int = 0> (dx, channel_axes)
   [n11] dmean = Neg (tmp_2)
   [n12] tmp_3 = Mul (dx, xhat)
   [n13] tmp_4 = Mul (tmp_3, inv)
   [n14] tmp_5 = ReduceSum <keepdims: int = 0> (tmp_4, channel_axes)
   [n15] dvar = Mul (tmp_5, neg_half)
})GRAD_TPL";

#endif  // ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_
