"""Formal check for FuseAddBiasIntoConv (fuse_add_bias_into_conv.h).

onnxsim registers its own copy of this pass
(``onnxsim/passes/fuse_add_bias_into_conv.h``, which additionally handles
ConvTranspose) via ``RegisterOrReplace``, overwriting the upstream
onnx-optimizer entry of the same name
(``third_party/onnx-optimizer/onnxoptimizer/passes/fuse_add_bias_into_conv.h``)
in the pass registry that ``getPassName()`` keys into -- so onnxsim's version
is the one that actually runs; both are byte-for-byte identical apart from
the ConvTranspose handling.

The rewrite folds ``Add``'s constant operand into ``Conv``'s (optional) third
("B") input::

    Z = Conv(X, W)        # exactly 2 inputs -- no existing bias
    Y = Add(Z, add_bias)
    =>
    Y = Conv(X, W, add_bias)   # add_bias squeezed to 1-D, length Cout

``patternMatchPredicate`` requires, precisely:

* ``CheckKind(node, kAdd, 0, kConv)`` -- Conv (or, in onnxsim's fork,
  ConvTranspose) must be operand **0** of Add specifically; operand 1 is
  never checked, so ``Add(add_bias, Conv(...))`` -- bias first -- never
  matches, exactly like ``fuse_matmul_add_bias_into_gemm``'s own
  order-sensitivity (this file's sibling
  ``test_formal_verify_fuse_matmul_add_bias_into_gemm.py``), not like the
  order-agnostic ``fuse_matmul_add_bias_into_gemm_batched``.
* ``GetInputsOfPreNode(node, 0).size() == 2`` -- the Conv node itself must
  have **exactly 2 inputs (X, W)**. This is the single most consequential
  fact this file's tests turn on: **unlike** ``fuse_bn_into_conv`` (which
  folds into an existing Conv bias via ``new_b = (B - mean) * s + bias``,
  using ``B = 0`` when there is none), this predicate simply declines
  outright whenever Conv already carries a bias -- there is no "sum both
  biases" code path here at all. This was verified empirically (see
  ``test_fuse_add_bias_into_conv_declines_when_conv_already_has_bias``
  below) before writing this file, per this repo's own experience of a
  previous author wrongly assuming a branch existed without checking the
  compiled pass.
* ``orig_bias->node()->kind()`` must be ``kConstant`` or ``kParam`` (a
  graph initializer) -- ``add_bias`` must be a compile-time constant.
* ``orig_conv->uses().size() <= 1`` -- Conv's output must be consumed only
  by this Add (the same single-use precondition seen throughout this
  fusion-pass family, avoiding a fusion that would silently drop a second
  consumer of the pre-Add Conv output).
* The broadcast precondition on ``add_bias``'s shape, checked in
  ``runTransform`` (``M`` = Conv's output-channel count, ``rank`` = Conv
  output rank): either (a) ``add_bias`` has exactly 1 element -- it is
  squeezed/unsqueezed to a scalar and then ``Tile``d out to length ``M``
  (the header comment's "case 2"), or (b) ``add_bias`` right-aligns against
  the Conv output the way NCHW-style per-channel broadcasting requires --
  concretely, ``rank <= add_bias.dims.size() + 1`` and the dim of
  ``add_bias`` that lines up with Conv's channel axis equals ``M`` (the
  header comment's "case 1", which its comment glosses as "A is 1D tensor
  and A.dim[0] == Z.dim[1]" but the actual code is more general: for a
  standard rank-4 NCHW Conv output this is satisfied by ``add_bias`` shaped
  ``[Cout, 1, 1]``, not by a literal 1-D ``[Cout]`` -- a literal 1-D
  ``[Cout]`` bias against a rank-4 Conv output is not even valid ONNX
  broadcasting (numpy-style alignment would pair ``Cout`` against the
  *last*, spatial axis) and fails shape inference outright, which was
  likewise confirmed empirically before writing the tests below rather than
  assumed from the header comment's simplified example.
* When it matches, the exact code is
  ``orig_conv->node()->addInput(conv_3rd_input)`` where ``conv_3rd_input``
  is ``add_bias`` reshaped (via ``Squeeze``/``Unsqueeze``, and ``Tile`` for
  the 1-element case) to a 1-D tensor of length ``M`` -- i.e. the new bias
  is *exactly* ``add_bias`` (reshaped only, no arithmetic), since ``B_new =
  B_existing + add_bias`` degenerates to ``add_bias`` because
  ``B_existing`` is always 0 (Conv having no existing bias is a
  precondition of the match, not a fallback computed at fusion time).

Soundness is simpler than ``fuse_bn_into_conv``'s (this file's closest
template, see its own docstring): folding a plain ``Add`` into Conv's bias
input is pure real *addition*, with no ``sqrt``/scale and hence no side
condition analogous to BN's ``sq * sq == var + eps`` -- Conv's linearity
means ``Conv(x, W) + B_existing + add_bias`` and
``Conv(x, W) + (B_existing + add_bias)`` differ only by real-number
associativity, which Z3 discharges unconditionally. The proof below still
states the fully general two-term formula (``B_existing`` need not be zero)
for clarity and so it would remain valid if a future version of this pass
grew a "fold into existing bias" branch; the differential tests demonstrate
that today's compiled pass only ever instantiates it at ``B_existing = 0``.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_fuse_add_bias_into_conv_is_sound():
    w0, w1, x0, x1 = z3.Reals("w0 w1 x0 x1")
    existing_bias, add_bias = z3.Reals("existing_bias add_bias")

    conv_out = w0 * x0 + w1 * x1  # Conv(x, W), no bias -- a 2-tap dot product

    # Before: Conv(x, W) [+ existing_bias] , then Add(add_bias).
    original = (conv_out + existing_bias) + add_bias
    # After: Conv(x, W, new_bias) with new_bias = existing_bias + add_bias.
    new_bias = existing_bias + add_bias
    fused = conv_out + new_bias

    prove(original == fused)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_add_bias_into_conv_pass_matches_no_existing_bias():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 3, 3, 3))
    # [Cout, 1, 1] -- the shape that actually right-aligns against a rank-4
    # NCHW Conv output per the predicate's broadcast rule (see module
    # docstring); a literal 1-D [8] is not valid ONNX here.
    add_bias = rng.standard_normal((8, 1, 1))
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          Y = Add(c, add_bias)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(add_bias, "add_bias")])
    sim_model, ops = simplify_isolated(model, "fuse_add_bias_into_conv")
    assert ops["Add"] == 0
    assert ops["Conv"] == 1

    conv_node = next(n for n in sim_model.graph.node if n.op_type == "Conv")
    assert len(conv_node.input) == 3, "fused Conv should now carry a bias input"
    by_name = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in sim_model.graph.initializer
    }
    fused_bias = by_name[conv_node.input[2]]

    # The formula just proven sound above at existing_bias = 0: the fused
    # bias is exactly add_bias, squeezed to 1-D.
    np.testing.assert_allclose(fused_bias, add_bias.squeeze(), rtol=1e-6, atol=1e-6)


def test_fuse_add_bias_into_conv_declines_when_conv_already_has_bias():
    # Conv already carries its own bias (3 inputs) -- patternMatchPredicate's
    # `GetInputsOfPreNode(node, 0).size() == 2` check fails, so the pass
    # never fires, even though the Add's bias shape is otherwise perfectly
    # fusable. There is no "sum the two biases" fallback in this pass (see
    # module docstring) -- confirmed empirically before writing this test.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 3, 3, 3))
    conv_bias = rng.standard_normal(8)
    add_bias = rng.standard_normal((8, 1, 1))
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W, conv_bias)
          Y = Add(c, add_bias)
        }
        """
    )
    model.graph.initializer.extend(
        [_f32(W, "W"), _f32(conv_bias, "conv_bias"), _f32(add_bias, "add_bias")]
    )
    _, ops = simplify_isolated(model, "fuse_add_bias_into_conv")
    assert ops["Add"] == 1
    assert ops["Conv"] == 1


def test_fuse_add_bias_into_conv_declines_swapped_operand_order():
    # Add(add_bias, c) -- bias as the *first* operand -- is never matched:
    # the predicate only checks Add's operand 0 for a Conv (or
    # ConvTranspose), exactly as fuse_matmul_add_bias_into_gemm only checks
    # operand 0 for a MatMul (see this file's sibling
    # test_formal_verify_fuse_matmul_add_bias_into_gemm.py).
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 3, 3, 3))
    add_bias = rng.standard_normal((8, 1, 1))
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          Y = Add(add_bias, c)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(add_bias, "add_bias")])
    _, ops = simplify_isolated(model, "fuse_add_bias_into_conv")
    assert ops["Add"] == 1
    assert ops["Conv"] == 1
