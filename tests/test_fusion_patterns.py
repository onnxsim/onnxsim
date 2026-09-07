"""Pass-isolated fusion / elimination tests.

OnnxSlim ships a suite of small, single-pattern tests (``test_fusion_patterns``,
``test_dead_node_elimination``, ``test_elimination_patterns``,
``test_subexpression_elimination``) that each check one optimization in
isolation. onnxsim performs the equivalent optimizations through onnxoptimizer
and its constant folding, but only exercised them indirectly through full
torchvision / timm models. This module adds the missing isolated coverage.

Every model is built directly with the ONNX text format parser (``onnx.parser``,
no torch dependency) and run through ``onnxsim.simplify`` with ``check_n=3`` so
onnxsim's own random-input equivalence check guards correctness of each
rewrite. Weight-shaped initializers still need real (usually random) data, so
those are built with numpy and attached to the parsed graph programmatically
rather than spelled out as text literals.

ConvTranspose+BN, ConvTranspose+Add-bias and no-op ``Dropout`` fusions -- once
gaps versus OnnxSlim -- are now covered by onnxsim's optimizer (issue #543) and
are regular tests below. GELU and LayerNorm subgraph fusion -- also once gaps
versus OnnxSlim -- are covered the same way now that onnxsim has fuse_gelu and
fuse_layer_norm.
"""

import collections

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim


def _simplify(model):
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _simplify_extra(model, extra_optimizers, skipped_optimizers=None):
    # Like _simplify, but for a PassType::Other pass that is not part of the
    # default fuse/elimination set (see extra_optimizers' own doc comment in
    # onnxsim.h) -- e.g. fuse_matmul_into_conv, which must be opted into by
    # name. skipped_optimizers additionally excludes named default passes
    # that would otherwise race the opted-in one for the same node (see
    # fuse_matmul_into_conv.h's own file comment).
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        extra_optimizers=extra_optimizers,
        skipped_optimizers=skipped_optimizers,
    )
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pin a low IR version by default so the model loads under the older
    # onnxruntime bundled with some CI wheels (which cap at IR version 11);
    # onnxsim's check_n runs the model through onnxruntime. `body` is the
    # ONNX text form graph declaration (and, optionally, its own inline
    # literal initializers); `initializer` holds extra TensorProtos -- e.g.
    # random weights -- attached after parsing rather than spelled out as
    # text literals.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f64(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float64), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


# --------------------------------------------------------------------------- #
# Fusion patterns
# --------------------------------------------------------------------------- #
def test_fuse_conv_bn_into_conv():
    # Conv followed by BatchNormalization folds the BN affine transform into the
    # Conv weights/bias (fuse_bn_into_conv), leaving a single Conv.
    W = np.random.randn(8, 3, 3, 3)
    scale = np.random.rand(8) + 0.5
    bias = np.random.randn(8)
    mean = np.random.randn(8)
    var = np.random.rand(8) + 0.5
    model = _model(
        """
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          Y = BatchNormalization(c, scale, bias, mean, var)
        }
        """,
        initializer=[
            _f32(W, "W"),
            _f32(scale, "scale"),
            _f32(bias, "bias"),
            _f32(mean, "mean"),
            _f32(var, "var"),
        ],
    )
    _, ops = _simplify(model)
    assert ops["BatchNormalization"] == 0
    assert ops["Conv"] == 1


def test_fuse_conv_with_bias_bn_into_conv():
    # Same fusion, but the Conv already has its own bias input -- a distinct
    # code path in fuse_bn_into_conv (the BN mean is subtracted from the
    # existing bias, not folded from zero).
    W = np.random.randn(8, 3, 3, 3)
    B = np.random.randn(8)
    scale = np.random.rand(8) + 0.5
    bias = np.random.randn(8)
    mean = np.random.randn(8)
    var = np.random.rand(8) + 0.5
    model = _model(
        """
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W, B)
          Y = BatchNormalization(c, scale, bias, mean, var)
        }
        """,
        initializer=[
            _f32(W, "W"),
            _f32(B, "B"),
            _f32(scale, "scale"),
            _f32(bias, "bias"),
            _f32(mean, "mean"),
            _f32(var, "var"),
        ],
    )
    _, ops = _simplify(model)
    assert ops["BatchNormalization"] == 0
    assert ops["Conv"] == 1


def test_fuse_conv_bn_into_conv_double():
    # Same fusion in float64: fuse_bn_into_conv's numeric fast path is
    # templated on the tensor element type (float or double), so this
    # exercises the double instantiation independently of the float32 tests
    # above. onnxruntime has no CPU kernel for a double-typed Conv (a
    # pre-existing runtime gap, unrelated to this fusion), so check_n's
    # normal onnxruntime-backed equivalence check can't run here; instead
    # this validates the fused Conv's weight/bias directly against the
    # pass's own documented formula, computed independently with numpy.
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 3, 3, 3))
    scale = rng.random(8) + 0.5
    bias = rng.standard_normal(8)
    mean = rng.standard_normal(8)
    var = rng.random(8) + 0.5
    model = _model(
        """
        g (double[1,3,16,16] X) => (double[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          Y = BatchNormalization(c, scale, bias, mean, var)
        }
        """,
        initializer=[
            _f64(w, "W"),
            _f64(scale, "scale"),
            _f64(bias, "bias"),
            _f64(mean, "mean"),
            _f64(var, "var"),
        ],
    )
    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["BatchNormalization"] == 0
    assert ops["Conv"] == 1

    conv_node = next(n for n in sim_model.graph.node if n.op_type == "Conv")
    by_name = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in sim_model.graph.initializer
    }
    fused_w = by_name[conv_node.input[1]]
    fused_b = by_name[conv_node.input[2]]

    s = scale / np.sqrt(var + 1e-5)
    expected_w = w * s.reshape(-1, 1, 1, 1)
    expected_b = (0.0 - mean) * s + bias
    np.testing.assert_allclose(fused_w, expected_w, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fused_b, expected_b, rtol=1e-10, atol=1e-12)


def test_fuse_matmul_add_into_gemm():
    # MatMul followed by a bias Add on 2-D inputs fuses into a single Gemm.
    W = np.random.randn(16, 8)
    B = np.random.randn(8)
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          mm = MatMul(X, W)
          Y = Add(mm, B)
        }
        """,
        initializer=[_f32(W, "W"), _f32(B, "B")],
    )
    _, ops = _simplify(model)
    assert ops["Gemm"] == 1
    assert ops["MatMul"] == 0 and ops["Add"] == 0


def test_batched_matmul_add_into_gemm():
    # A transformer linear layer applies a 2-D weight to a rank-3 activation and
    # adds a bias (exported bias-first, as HuggingFace does). onnxsim converts
    # the batched MatMul+Add into a Gemm (fuse_matmul_add_bias_into_gemm_batched,
    # opted in by onnxsim) so runtimes can dispatch tuned GEMM kernels.
    W = np.random.randn(8, 16)
    b = np.random.randn(16)
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,16] Y)
        {
          mm = MatMul(X, W)
          Y = Add(b, mm)
        }
        """,
        initializer=[_f32(W, "W"), _f32(b, "b")],
    )
    _, ops = _simplify(model)
    assert ops["Gemm"] == 1
    assert ops["MatMul"] == 0


def test_fuse_matmul_into_conv_2d():
    # A bare 2-D MatMul against a constant weight (e.g. a CNN's classifier
    # head after global pooling + Flatten) is a Linear layer in disguise, and
    # rewrites to a 1x1 Conv -- fuse_matmul_into_conv, opted in via
    # extra_optimizers for accelerators whose Conv datapath is far better
    # optimized than their generic MatMul one.
    W = np.random.randn(16, 8)
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    _, ops = _simplify_extra(model, extra_optimizers=["fuse_matmul_into_conv"])
    assert ops["Conv"] == 1
    assert ops["MatMul"] == 0


def test_fuse_matmul_into_conv_2d_with_bias():
    # The default fuse set would otherwise fuse this MatMul+Add into a Gemm
    # first (fuse_matmul_add_bias_into_gemm), so it's skipped here to isolate
    # fuse_matmul_into_conv's own MatMul+Add(bias) -> Conv(bias) fusion.
    W = np.random.randn(16, 8)
    B = np.random.randn(8)
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          mm = MatMul(X, W)
          Y = Add(mm, B)
        }
        """,
        initializer=[_f32(W, "W"), _f32(B, "B")],
    )
    _, ops = _simplify_extra(
        model,
        extra_optimizers=["fuse_matmul_into_conv"],
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
    )
    assert ops["Conv"] == 1
    assert ops["MatMul"] == 0 and ops["Add"] == 0 and ops["Gemm"] == 0


def test_fuse_batched_matmul_into_conv_with_bias():
    # Same rank-3, bias-first (HuggingFace-style) Linear layer as
    # test_batched_matmul_add_into_gemm, but targeting fuse_matmul_into_conv
    # instead -- fuse_matmul_add_bias_into_gemm_batched is skipped since it is
    # otherwise unconditionally in the default set and would race this one
    # for the same Add(MatMul) node.
    W = np.random.randn(8, 16)
    b = np.random.randn(16)
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,16] Y)
        {
          mm = MatMul(X, W)
          Y = Add(b, mm)
        }
        """,
        initializer=[_f32(W, "W"), _f32(b, "b")],
    )
    _, ops = _simplify_extra(
        model,
        extra_optimizers=["fuse_matmul_into_conv"],
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm_batched"],
    )
    assert ops["Conv"] == 1
    assert ops["MatMul"] == 0 and ops["Add"] == 0 and ops["Gemm"] == 0


def test_fuse_gemm_into_conv_transb():
    # A plain nn.Linear typically exports as Gemm(X, W, B, transB=1) (W kept
    # in its natural [out, in] layout). fuse_matmul_into_conv also rewrites
    # this shape, not just MatMul -- no default pass turns Gemm back into
    # MatMul, so no skipped_optimizers is needed here. fuse_matmul_into_conv
    # itself never special-cases transB=1 to skip inserting a Transpose for
    # W (it always builds the Conv weight the same way regardless of
    # layout); the default fuse set's own fuse_consecutive_transposes +
    # eliminate_nop_transpose cancel the resulting Transpose<perm=[1,0]>
    # pair, so no Transpose survives either.
    W = np.random.randn(8, 16)  # [N, K], transB=1
    B = np.random.randn(8)
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          Y = Gemm<transB = 1>(X, W, B)
        }
        """,
        initializer=[_f32(W, "W"), _f32(B, "B")],
    )
    _, ops = _simplify_extra(model, extra_optimizers=["fuse_matmul_into_conv"])
    assert ops["Conv"] == 1
    assert ops["Gemm"] == 0
    assert ops["Transpose"] == 0


def test_fuse_gemm_into_conv_transa():
    # transA=1 has no counterpart anywhere else in the rewrite to cancel
    # against, so -- unlike transB above -- the Transpose(X) this pass
    # inserts for it is genuine work and survives in the final graph.
    W = np.random.randn(16, 8)
    model = _model(
        """
        g (float[16,4] X) => (float[4,8] Y)
        {
          Y = Gemm<transA = 1>(X, W)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    _, ops = _simplify_extra(model, extra_optimizers=["fuse_matmul_into_conv"])
    assert ops["Conv"] == 1
    assert ops["Gemm"] == 0
    assert ops["Transpose"] == 1


def test_fuse_matmul_into_conv_declines_non_default_alpha():
    # alpha != 1 would need the folded weight rescaled; fuse_matmul_into_conv
    # doesn't do that and leaves this Gemm untouched.
    W = np.random.randn(16, 8)
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    _, ops = _simplify_extra(model, extra_optimizers=["fuse_matmul_into_conv"])
    assert ops["Gemm"] == 1
    assert ops["Conv"] == 0


def test_fuse_pad_into_conv():
    # A constant zero-value Pad on the spatial dims is folded into the Conv pads
    # attribute (fuse_pad_into_conv), removing the Pad node.
    W = np.random.randn(8, 3, 3, 3)
    model = _model(
        """
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        <int64[8] pads = {0, 0, 1, 1, 0, 0, 1, 1}>
        {
          p = Pad<mode = "constant">(X, pads)
          Y = Conv<kernel_shape = [3, 3]>(p, W)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    _, ops = _simplify(model)
    assert ops["Pad"] == 0
    assert ops["Conv"] == 1


def test_fuse_consecutive_reduce_unsqueeze():
    # ReduceSum(keepdims=0) immediately followed by an Unsqueeze on the reduced
    # axis collapses into a single keepdims reduction.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3,1] Y)
        <int64[1] raxes = {2}, int64[1] uaxes = {2}>
        {
          r = ReduceSum<keepdims = 0>(X, raxes)
          Y = Unsqueeze(r, uaxes)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Unsqueeze"] == 0
    assert ops["ReduceSum"] == 1


def test_fuse_consecutive_reduce():
    # Two consecutive ReduceSum nodes collapse into one over the union of
    # their axes. The second reduction's axis (1) is expressed relative to
    # the already-reduced (rank-3) tensor, where it refers to original axis 2
    # -- exercising the index remapping through the dropped axis 1, not just
    # a trailing/no-shift axis.
    model = _model(
        """
        g (float[2,3,4,5] X) => (float[2,5] Z)
        <int64[1] ax0 = {1}, int64[1] ax1 = {1}>
        {
          y = ReduceSum<keepdims = 0>(X, ax0)
          Z = ReduceSum<keepdims = 0>(y, ax1)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["ReduceSum"] == 1


def test_fuse_consecutive_reduce_through_reshape():
    # ReduceSum(keepdims=0) -> Reshape (re-adding the reduced axis as size 1,
    # equivalent to keepdims=1) -> ReduceSum(keepdims=0) collapses into a
    # single ReduceSum, eliminating the Reshape as well.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3] Z)
        <int64[1] ax = {2}, int64[3] shp = {2,3,1}>
        {
          y = ReduceSum<keepdims = 0>(X, ax)
          r = Reshape(y, shp)
          Z = ReduceSum<keepdims = 0>(r, ax)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Reshape"] == 0
    assert ops["ReduceSum"] == 1


def test_fuse_consecutive_reduce_through_unsqueeze():
    # Same idiom as test_fuse_consecutive_reduce_through_reshape, but the
    # keepdims-toggling reshape is spelled as an Unsqueeze on the dropped
    # axis instead of a generic Reshape.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3] Z)
        <int64[1] ax = {2}>
        {
          y = ReduceSum<keepdims = 0>(X, ax)
          u = Unsqueeze(y, ax)
          Z = ReduceSum<keepdims = 0>(u, ax)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Unsqueeze"] == 0
    assert ops["ReduceSum"] == 1


def test_fuse_consecutive_reduce_through_squeeze():
    # Mirror image of the Unsqueeze case: the first reduction keeps its
    # reduced axis (keepdims=1) and a Squeeze drops it back down to the
    # keepdims=False shape before the second reduction.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2] Z)
        <int64[1] ax = {2}, int64[1] ax1 = {1}, int64[1] sq = {2}>
        {
          y = ReduceSum<keepdims = 1>(X, ax)
          s = Squeeze(y, sq)
          Z = ReduceSum<keepdims = 0>(s, ax1)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Squeeze"] == 0
    assert ops["ReduceSum"] == 1


def test_fuse_consecutive_reduce_declines_different_kind():
    # ReduceSum followed by ReduceMax must not fuse -- the two ops don't
    # combine associatively across a shared axis group.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3] Z)
        <int64[1] ax = {2}>
        {
          y = ReduceSum<keepdims = 1>(X, ax)
          Z = ReduceMax<keepdims = 0>(y, ax)
        }
        """,
        opset=18,
    )
    _, ops = _simplify(model)
    assert ops["ReduceSum"] == 1
    assert ops["ReduceMax"] == 1


def test_fuse_consecutive_reduce_declines_sum_square():
    # ReduceSumSquare does not have the fold-into-one-application property
    # ((sum(x^2))^2 != sum(x^4)), so consecutive ReduceSumSquare nodes must
    # be left alone.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3] Z)
        <int64[1] ax = {2}>
        {
          y = ReduceSumSquare<keepdims = 1>(X, ax)
          Z = ReduceSumSquare<keepdims = 0>(y, ax)
        }
        """,
        opset=18,
    )
    _, ops = _simplify(model)
    assert ops["ReduceSumSquare"] == 2


def test_fuse_rms_norm():
    # x.pow(2).mean(-1, keepdim=True) -> x * rsqrt(variance + eps) -> weight *
    # (...): the textbook RMSNorm forward used by LLaMA/Mistral/Qwen-family
    # exports (see test_mnn_llm_export.py's ``_RMSNorm``). At opset >= 23
    # (RMSNormalization's introducing version) the whole chain collapses to a
    # single node (fuse_rms_norm). RMSNormalization needs ir_version 11
    # (opset 23 shipped in onnx 1.18).
    weight = np.random.randn(8) * 0.02 + 1.0
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,8] Y)
        <float two = {2.0}, int64[1] axes = {-1}, float eps = {1e-06}>
        {
          sq = Pow(X, two)
          var = ReduceMean<keepdims = 1>(sq, axes)
          var_eps = Add(var, eps)
          rms = Sqrt(var_eps)
          inv_rms = Reciprocal(rms)
          normed = Mul(X, inv_rms)
          Y = Mul(weight, normed)
        }
        """,
        initializer=[_f32(weight, "weight")],
        opset=23,
        ir_version=11,
    )
    _, ops = _simplify(model)
    assert ops["RMSNormalization"] == 1
    assert ops["Mul"] == 0
    assert ops["Pow"] == 0
    assert ops["ReduceMean"] == 0
    assert ops["Add"] == 0
    assert ops["Sqrt"] == 0
    assert ops["Reciprocal"] == 0


def test_fuse_rms_norm_below_opset_23_untouched():
    # Below opset 23, RMSNormalization does not exist yet: the pass must not
    # fire, leaving the decomposition (and its behavior on older runtimes)
    # intact. ReduceMean's axes is an attribute (not a second input) below
    # opset 18, so this also exercises that spelling of the pattern.
    weight = np.random.randn(8) * 0.02 + 1.0
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,8] Y)
        <float two = {2.0}, float eps = {1e-06}>
        {
          sq = Pow(X, two)
          var = ReduceMean<axes = [-1], keepdims = 1>(sq)
          var_eps = Add(var, eps)
          rms = Sqrt(var_eps)
          inv_rms = Reciprocal(rms)
          normed = Mul(X, inv_rms)
          Y = Mul(weight, normed)
        }
        """,
        initializer=[_f32(weight, "weight")],
        opset=17,
    )
    _, ops = _simplify(model)
    assert ops["RMSNormalization"] == 0
    assert ops["Mul"] == 2


# --------------------------------------------------------------------------- #
# fuse_rope -- HuggingFace-style "rotate_half" rotary position embedding
# application (see fuse_rope.h's own top comment and test_mnn_llm_export.py's
# ``_Attention`` for a real traced example) collapsed into a single standard
# ONNX ``RotaryEmbedding`` node (opset 23). Q's and K's applications share one
# ``cos``/``sin``/``Concat`` computation (``emb = concat([angle, angle])``);
# both must be recognized as reading the *same* ``angle`` Value for the
# fusion to be numerically exact -- see MatchRopeCosSin's own doc comment.
# --------------------------------------------------------------------------- #
def _rope_apply_body(x_name, cos_bcast_name, sin_bcast_name, prefix, half):
    # x_embed = x * cos_bcast + rotate_half(x) * sin_bcast
    return f"""
          {prefix}_a = Mul({x_name}, {cos_bcast_name})
          {prefix}_x1 = Slice({x_name}, slice_start0, slice_end{half}, slice_axism1)
          {prefix}_x2 = Slice({x_name}, slice_start{half}, slice_end_max, slice_axism1)
          {prefix}_neg_x2 = Neg({prefix}_x2)
          {prefix}_rotated = Concat<axis = -1>({prefix}_neg_x2, {prefix}_x1)
          {prefix}_b = Mul({prefix}_rotated, {sin_bcast_name})
          {prefix}_embed = Add({prefix}_a, {prefix}_b)
    """


def _rope_slice_inits(half):
    int_max = np.iinfo(np.int64).max
    return f"""
          int64[1] slice_start0 = {{0}},
          int64[1] slice_end{half} = {{{half}}},
          int64[1] slice_start{half} = {{{half}}},
          int64[1] slice_end_max = {{{int_max}}},
          int64[1] slice_axism1 = {{-1}},
          int64[1] unsq_axis1 = {{1}}
    """


def _rope_model(B=2, NH=4, S=6, Dh=8, share_angle=True, opset=23):
    half = Dh // 2
    angle2 = "angle" if share_angle else "angle2"
    inputs = f"float[{B},{NH},{S},{Dh}] q, float[{B},{NH},{S},{Dh}] k, float[{B},{S},{half}] angle"
    if not share_angle:
        # A second, independent (shape-identical but not the same Value, and
        # not foldable back to `angle`) input: the duplicating Concat's two
        # inputs are then genuinely different, so fuse_rope must decline
        # rather than assume this is duplication.
        inputs += f", float[{B},{S},{half}] angle2"
    body = f"""
        g ({inputs}) => (float[{B},{NH},{S},{Dh}] q_embed, float[{B},{NH},{S},{Dh}] k_embed)
        <{_rope_slice_inits(half)}>
        {{
          emb = Concat<axis = -1>(angle, {angle2})
          cos_full = Cos(emb)
          sin_full = Sin(emb)
          cos_bcast = Unsqueeze(cos_full, unsq_axis1)
          sin_bcast = Unsqueeze(sin_full, unsq_axis1)
          {_rope_apply_body("q", "cos_bcast", "sin_bcast", "q", half)}
          {_rope_apply_body("k", "cos_bcast", "sin_bcast", "k", half)}
        }}
        """
    return _model(body, opset=opset, ir_version=11)


def test_fuse_rope():
    model = _rope_model()
    _, ops = _simplify(model)
    assert ops["RotaryEmbedding"] == 2
    # The whole shared cos/sin/Concat/Unsqueeze chain, and each side's own
    # Slice/Neg/Concat/Mul/Add chain, must be fully torn down -- not just
    # left as dead code (see fuse_rope.h's MaybeAppendSharedChain).
    assert ops["Concat"] == 0
    assert ops["Slice"] == 0
    assert ops["Neg"] == 0
    assert ops["Unsqueeze"] == 0
    assert ops["Mul"] == 0
    assert ops["Add"] == 0


def test_fuse_rope_below_opset_23_untouched():
    # Below opset 23, RotaryEmbedding does not exist yet: the pass must not
    # fire, leaving the decomposition intact.
    model = _rope_model(opset=17)
    model.ir_version = 10
    _, ops = _simplify(model)
    assert ops["RotaryEmbedding"] == 0
    assert ops["Concat"] == 3  # shared emb + each side's rotate_half Concat


def test_fuse_rope_declines_non_shared_angle():
    # cos/sin are computed from concat([angle, angle2]) where angle2 is a
    # *different* Value from angle (even though shape-identical) rather than
    # literally the same one -- MatchRopeCosSin's reference-identity check on
    # the Concat's two inputs must catch this and decline, since wiring a
    # non-duplicated cos/sin into RotaryEmbedding would compute something
    # else entirely.
    model = _rope_model(share_angle=False)
    _, ops = _simplify(model)
    assert ops["RotaryEmbedding"] == 0


def test_fuse_rope_partial_match_preserves_shared_chain():
    # Only Q's side matches the rotate_half pattern; K instead just multiplies
    # by cos_bcast directly (a stand-in for "some other, unmatched use").
    # fuse_rope must still fuse Q, and must NOT tear down the shared
    # cos/sin/Concat chain (K still reads it) -- MaybeAppendSharedChain's own
    # live-use-count check should correctly see cos_bcast/sin_bcast still
    # have a remaining consumer and leave them alone.
    B, NH, S, Dh = 2, 4, 6, 8
    half = Dh // 2
    model = _model(
        f"""
        g (float[{B},{NH},{S},{Dh}] q, float[{B},{NH},{S},{Dh}] k, float[{B},{S},{half}] angle) => (float[{B},{NH},{S},{Dh}] q_embed, float[{B},{NH},{S},{Dh}] k_embed)
        <{_rope_slice_inits(half)}>
        {{
          emb = Concat<axis = -1>(angle, angle)
          cos_full = Cos(emb)
          sin_full = Sin(emb)
          cos_bcast = Unsqueeze(cos_full, unsq_axis1)
          sin_bcast = Unsqueeze(sin_full, unsq_axis1)
          {_rope_apply_body("q", "cos_bcast", "sin_bcast", "q", half)}
          k_embed = Mul(k, cos_bcast)
        }}
        """,
        opset=23,
        ir_version=11,
    )
    simplified, ops = _simplify(model)
    assert ops["RotaryEmbedding"] == 1
    # K's own Mul(k, cos_bcast) must still resolve correctly -- proving the
    # shared cos_bcast (and everything upstream of it) was left intact.
    k_embed = next(o for o in simplified.graph.output if o.name == "k_embed")
    assert k_embed is not None
    remaining_muls = [n for n in simplified.graph.node if n.op_type == "Mul"]
    assert any("k" in n.input for n in remaining_muls)


def test_fuse_concat_into_reshape():
    # A Concat of constant shape pieces feeding a Reshape is folded into a single
    # Reshape with a constant target shape (fuse_concat_into_reshape).
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,12] Y)
        <int64[1] c0 = {2}, int64[1] c1 = {-1}>
        {
          shape = Concat<axis = 0>(c0, c1)
          Y = Reshape(X, shape)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Concat"] == 0
    assert ops["Reshape"] == 1


def test_fuse_consecutive_reshapes():
    # Only the final target shape matters -- Reshape(Reshape(X, s1), s2)
    # collapses to Reshape(X, s2) (fuse_consecutive_reshapes).
    model = _model(
        """
        g (float[2,3,4] X) => (float[3,8] Y)
        <int64[2] s1 = {6,4}, int64[2] s2 = {3,8}>
        {
          y = Reshape(X, s1)
          Y = Reshape(y, s2)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Reshape"] == 1


def test_fuse_consecutive_reshapes_chain():
    # A longer chain collapses all the way down to the final reshape.
    model = _model(
        """
        g (float[2,3,4] X) => (float[3,8] Y)
        <int64[2] s1 = {4,6}, int64[2] s2 = {2,12}, int64[2] s3 = {3,8}>
        {
          y1 = Reshape(X, s1)
          y2 = Reshape(y1, s2)
          Y = Reshape(y2, s3)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Reshape"] == 1


def test_fuse_consecutive_reshapes_declines_ambiguous_zero_copy():
    # The outer reshape's `0` (under the default allowzero=0) means "copy
    # this dim from the node's own input", i.e. from the inner reshape's
    # output shape [2,3,4] (dim0=2) -- not from X's shape [6,4] (dim0=6).
    # Fusing the two would silently change which shape the `0` copies from,
    # so the pass must leave this chain alone.
    model = _model(
        """
        g (float[6,4] X) => (float[2,12] Y)
        <int64[3] s1 = {2,3,4}, int64[2] s2 = {0,-1}>
        {
          y = Reshape(X, s1)
          Y = Reshape(y, s2)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Reshape"] == 2


def test_fuse_reshape_family_flatten_then_reshape():
    # Flatten and Reshape both just rearrange a tensor's shape, so a Flatten
    # immediately followed by a Reshape collapses into one Reshape
    # (fuse_reshape_family), same as two consecutive Reshapes would.
    model = _model(
        """
        g (float[2,3,4] X) => (float[3,8] Y)
        <int64[2] s = {3,8}>
        {
          y = Flatten<axis = 1>(X)
          Y = Reshape(y, s)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Flatten"] == 0
    assert ops["Reshape"] == 1


def test_fuse_reshape_family_squeeze_then_reshape():
    # Squeeze followed by Reshape collapses the same way.
    model = _model(
        """
        g (float[1,2,3,4] X) => (float[6,4] Y)
        <int64[1] axes = {0}, int64[2] s = {6,4}>
        {
          y = Squeeze(X, axes)
          Y = Reshape(y, s)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Squeeze"] == 0
    assert ops["Reshape"] == 1


def test_fuse_reshape_family_reshape_then_squeeze():
    # The *outer* op need not be Reshape either -- Reshape followed by
    # Squeeze collapses to a single Reshape (the outer Squeeze is replaced,
    # not just bypassed).
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,12] Y)
        <int64[3] s = {2,1,12}, int64[1] axes = {1}>
        {
          y = Reshape(X, s)
          Y = Squeeze(y, axes)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Squeeze"] == 0
    assert ops["Reshape"] == 1


def test_fuse_reshape_family_declines_unresolved_output_shape():
    # y's shape ([N, unk__0]) has two independent unresolved dims -- N and M
    # are both symbolic -- and Unsqueeze's shape inference can't propagate a
    # rank for Y from that, so the pass has no target shape to fuse to and
    # must leave both nodes (rather than, say, guess a scalar target from an
    # empty shape). N and M are dynamic, so this calls onnxsim.simplify
    # directly with check_n=0 -- onnxsim's random-input equivalence check
    # needs concrete dims -- and only asserts on graph structure.
    model = _model(
        """
        g (float[N,M,4,5] X) => (float Y)
        <int64[1] axes = {0}>
        {
          y = Flatten<axis = 1>(X)
          Y = Unsqueeze(y, axes)
        }
        """
    )
    sim_model, _ = onnxsim.simplify(model, check_n=0)
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Flatten"] == 1
    assert ops["Unsqueeze"] == 1
    assert ops["Reshape"] == 0


# --------------------------------------------------------------------------- #
# Dead-node / no-op elimination
# --------------------------------------------------------------------------- #
def test_eliminate_identity():
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Identity(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Identity"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_transpose():
    # A Transpose whose permutation is the identity ordering is a no-op.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Relu(X)
          Y = Transpose<perm = [0, 1]>(a)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Transpose"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_expand():
    # Expand to the already-existing shape does nothing and is removed.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] eshape = {4, 8}>
        {
          a = Relu(X)
          Y = Expand(a, eshape)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Expand"] == 0
    assert ops["Relu"] == 1


def test_eliminate_mul_by_one():
    # Multiplying by a unit constant is a no-op and is eliminated.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <float[1] one = {1.0}>
        {
          a = Relu(X)
          Y = Mul(a, one)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Mul"] == 0
    assert ops["Relu"] == 1


def test_eliminate_consecutive_cancelling_transposes():
    # Two transposes that invert each other collapse away entirely.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          t = Transpose<perm = [1, 0]>(X)
          Y = Transpose<perm = [1, 0]>(t)
        }
        """
    )
    sim_model, ops = _simplify(model)
    # The pair either cancels to a bare passthrough (<=1 node) with no residual
    # transpose logic changing the data.
    assert ops["Transpose"] <= 1


# --------------------------------------------------------------------------- #
# Common subexpression elimination
# --------------------------------------------------------------------------- #
def test_eliminate_common_subexpression():
    # Two structurally identical Sqrt nodes over the same input are deduplicated
    # into one shared node (eliminate_common_subexpression).
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          s1 = Sqrt(X)
          s2 = Sqrt(X)
          Y = Add(s1, s2)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Sqrt"] == 1
    assert ops["Add"] == 1


# --------------------------------------------------------------------------- #
# Deferred folding of large-tensor ops (ConstantOfShape / Constant -> Expand)
#
# With --no-large-tensor these ops are not folded into a (large) initializer,
# but they are still treated as constant so a downstream node whose output is
# small keeps folding: the producing op is inlined into the sub-model the
# operator executor runs, so the large intermediate is computed transiently and
# never stored. The result is that the whole constant chain collapses to the
# small final initializer.
# --------------------------------------------------------------------------- #
def _simplify_no_large_tensor(model):
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=3, tensor_size_threshold="1KB"
    )
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def test_defer_constantofshape_folds_small_consumer():
    # ConstantOfShape produces a large (256*256 f32 = 256KB > 1KB) tensor, so it
    # is not materialized under --no-large-tensor. ReduceSum over it yields a
    # scalar, which the executor still folds by running ConstantOfShape+ReduceSum
    # together; the scalar feeds a runtime Add. The large tensor is never stored
    # and the ConstantOfShape/ReduceSum chain disappears.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] shape = {256, 256}>
        {
          big = ConstantOfShape<value = float[1] {2.0}>(shape)
          s = ReduceSum<keepdims = 0>(big)
          Y = Add(X, s)
        }
        """,
        opset=11,
    )
    sim_model, ops = _simplify_no_large_tensor(model)
    assert ops["ConstantOfShape"] == 0
    assert ops["ReduceSum"] == 0
    assert ops["Add"] == 1
    # No large intermediate tensor was materialized as an initializer.
    assert all(
        onnx.numpy_helper.to_array(init).size < 256 * 256
        for init in sim_model.graph.initializer
    )


def test_defer_constant_expand_folds_small_consumer():
    # Constant -> Expand blows a scalar up to a large (512*512) tensor. Under
    # --no-large-tensor the Expand is not materialized, but ReduceSum over it is
    # a small (scalar) foldable output whose value depends on the data (so it is
    # not handled by shape propagation). The executor still folds it by running
    # Expand+ReduceSum together, and the scalar feeds a runtime Add, so the
    # constant chain collapses without ever storing the expanded tensor. The
    # folded scalar is materialized as a Constant node rather than an
    # initializer -- a deferred (large-tensor) output is never treated as
    # purely initializer-derived, so its consumer isn't either.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] eshape = {512, 512}>
        {
          small = Constant<value = float[1,1] {3.0}>()
          big = Expand(small, eshape)
          s = ReduceSum<keepdims = 0>(big)
          Y = Add(X, s)
        }
        """,
        opset=11,
    )
    sim_model, ops = _simplify_no_large_tensor(model)
    assert ops["Expand"] == 0
    assert ops["ReduceSum"] == 0
    assert ops["Constant"] == 1
    assert ops["Add"] == 1
    assert all(
        onnx.numpy_helper.to_array(init).size < 512 * 512
        for init in sim_model.graph.initializer
    )
    (s_node,) = [n for n in sim_model.graph.node if n.op_type == "Constant"]
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(s_node.attribute[0].t), 3.0 * 512 * 512
    )


# --------------------------------------------------------------------------- #
# ConvTranspose / no-op Dropout fusions (issue #543). Formerly xfail gaps versus
# OnnxSlim, now handled by onnxsim's optimizer: no-op Dropout removal and the
# fuse_bn_into_conv / fuse_add_bias_into_conv passes extended to ConvTranspose.
# --------------------------------------------------------------------------- #
def test_eliminate_nop_dropout():
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <float ratio = {0.0}>
        {
          a = Relu(X)
          Y = Dropout(a, ratio)
        }
        """
    )
    _, ops = _simplify(model)
    assert ops["Dropout"] == 0
    assert ops["Relu"] == 1


def test_fuse_convtranspose_bn():
    W = np.random.randn(3, 8, 3, 3)  # ConvTranspose: [Cin, Cout, kH, kW]
    scale = np.random.rand(8) + 0.5
    bias = np.random.randn(8)
    mean = np.random.randn(8)
    var = np.random.rand(8) + 0.5
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,8,10,10] Y)
        {
          c = ConvTranspose<kernel_shape = [3, 3]>(X, W)
          Y = BatchNormalization(c, scale, bias, mean, var)
        }
        """,
        initializer=[
            _f32(W, "W"),
            _f32(scale, "scale"),
            _f32(bias, "bias"),
            _f32(mean, "mean"),
            _f32(var, "var"),
        ],
    )
    _, ops = _simplify(model)
    assert ops["BatchNormalization"] == 0
    assert ops["ConvTranspose"] == 1


def test_fuse_convtranspose_add():
    W = np.random.randn(3, 8, 3, 3)
    bias = np.random.randn(1, 8, 1, 1)
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,8,10,10] Y)
        {
          c = ConvTranspose<kernel_shape = [3, 3]>(X, W)
          Y = Add(c, bias)
        }
        """,
        initializer=[_f32(W, "W"), _f32(bias, "bias")],
    )
    _, ops = _simplify(model)
    assert ops["Add"] == 0
    assert ops["ConvTranspose"] == 1


# --------------------------------------------------------------------------- #
# Reshape cancellation around a batched-Gemm element-wise chain. onnxsim's
# batched MatMul+bias -> Gemm rewrite brackets each rank-3 linear layer in a
# Reshape(-> 2-D) / Gemm / Reshape(-> N-D) sandwich so runtimes can dispatch a
# tuned GEMM kernel. When two such linears surround an element-wise chain, the
# inverse reshapes between them cancel (eliminate_reshape_around_elementwise):
# the chain runs on the 2-D Gemm output and both Gemms are kept. This is the
# node-count half of the batched-Gemm rewrite (issue: model-regression node
# reduction).
# --------------------------------------------------------------------------- #
def test_eliminate_reshape_around_elementwise():
    # The post-batched-fusion shape, built explicitly: Gemm -> Reshape(N-D) ->
    # Relu -> Reshape(2-D) -> Gemm. The two middle reshapes are exact inverses
    # (they only split / merge the leading dims), so they collapse and the Relu
    # ends up directly between the two Gemms.
    W1 = np.random.randn(8, 16)
    b1 = np.random.randn(16)
    W2 = np.random.randn(16, 8)
    b2 = np.random.randn(8)
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,8] Y)
        <
          int64[2] s1 = {-1, 8},
          int64[3] s2 = {2, 4, 16},
          int64[2] s3 = {-1, 16},
          int64[3] s4 = {2, 4, 8}
        >
        {
          rx1 = Reshape(X, s1)
          g1 = Gemm(rx1, W1, b1)
          u1 = Reshape(g1, s2)
          r = Relu(u1)
          f2 = Reshape(r, s3)
          g2 = Gemm(f2, W2, b2)
          Y = Reshape(g2, s4)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(b1, "b1"), _f32(W2, "W2"), _f32(b2, "b2")],
    )
    _, ops = _simplify(model)
    # Both Gemms are kept (tuned-kernel dispatch preserved)...
    assert ops["Gemm"] == 2
    assert ops["Relu"] == 1
    # ...and the two inverse reshapes bracketing the Relu are gone: only the
    # entry (X -> 2-D) and the final (2-D -> N-D output) reshapes remain.
    assert ops["Reshape"] == 2


# --------------------------------------------------------------------------- #
# GELU subgraph fusion: onnxsim recognizes the exact-erf decomposition and
# fuses it to a single ``Gelu`` node (fuse_gelu, opset >= 20). This used to be
# a documented gap versus OnnxSlim; promoted from xfail once onnxsim gained
# the pass.
# --------------------------------------------------------------------------- #
def test_fuse_gelu():
    # 0.5 * x * (1 + erf(x / sqrt(2))) is the exact-erf GELU formulation.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <float half = {0.5}, float one = {1.0}, float sqrt2 = {1.4142135623730951}>
        {
          t0 = Div(X, sqrt2)
          t1 = Erf(t0)
          t2 = Add(t1, one)
          t3 = Mul(X, t2)
          Y = Mul(t3, half)
        }
        """,
        # Gelu is only in the default domain from opset 20.
        opset=20,
    )
    _, ops = _simplify(model)
    assert ops["Gelu"] == 1
    assert ops["Erf"] == 0


def test_fuse_gelu_commuted_operands():
    # Same pattern, but with every commutative Mul/Add's operands swapped.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <float half = {0.5}, float one = {1.0}, float sqrt2 = {1.4142135623730951}>
        {
          t0 = Div(X, sqrt2)
          t1 = Erf(t0)
          t2 = Add(one, t1)
          t3 = Mul(t2, X)
          Y = Mul(half, t3)
        }
        """,
        opset=20,
    )
    _, ops = _simplify(model)
    assert ops["Gelu"] == 1
    assert ops["Erf"] == 0


def test_fuse_gelu_skips_old_opset():
    # Gelu is only in the default domain from opset 20; an older-opset graph
    # keeps the decomposition rather than emitting an invalid node.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        <float half = {0.5}, float one = {1.0}, float sqrt2 = {1.4142135623730951}>
        {
          t0 = Div(X, sqrt2)
          t1 = Erf(t0)
          t2 = Add(t1, one)
          t3 = Mul(X, t2)
          Y = Mul(t3, half)
        }
        """,
        opset=13,
    )
    _, ops = _simplify(model)
    assert ops["Gelu"] == 0
    assert ops["Erf"] == 1


# --------------------------------------------------------------------------- #
# LayerNorm subgraph fusion: onnxsim recognizes the textbook last-axis
# decomposition (mean/var via ReduceMean, either Mul(diff,diff) or
# Pow(diff,2) for the square) and fuses it to a single
# ``LayerNormalization`` node (fuse_layer_norm, opset >= 17).
# --------------------------------------------------------------------------- #
def _layer_norm_body(square_op):
    square = "sq = Mul(diff, diff)" if square_op == "Mul" else "sq = Pow(diff, two)"
    return f"""
          mean = ReduceMean<axes = [-1], keepdims = 1>(X)
          diff = Sub(X, mean)
          {square}
          var = ReduceMean<axes = [-1], keepdims = 1>(sq)
          var_eps = Add(var, eps)
          std = Sqrt(var_eps)
          norm = Div(diff, std)
          scaled = Mul(norm, scale)
          Y = Add(scaled, bias)
    """


def _layer_norm_inits():
    return [
        _f32(np.array([1e-5]), "eps"),
        _f32(np.array([2.0]), "two"),
        _f32(np.random.RandomState(0).randn(8), "scale"),
        _f32(np.random.RandomState(1).randn(8), "bias"),
    ]


@pytest.mark.parametrize("square_op", ["Mul", "Pow"])
def test_fuse_layer_norm(square_op):
    model = _model(
        f"""
        g (float[2,4,8] X) => (float[2,4,8] Y)
        {{
          {_layer_norm_body(square_op)}
        }}
        """,
        initializer=_layer_norm_inits(),
        opset=17,
    )
    _, ops = _simplify(model)
    assert ops["LayerNormalization"] == 1
    assert ops["ReduceMean"] == 0


def test_fuse_layer_norm_skips_old_opset():
    # LayerNormalization is only in the default domain from opset 17.
    model = _model(
        f"""
        g (float[2,4,8] X) => (float[2,4,8] Y)
        {{
          {_layer_norm_body("Mul")}
        }}
        """,
        initializer=_layer_norm_inits(),
        opset=13,
    )
    _, ops = _simplify(model)
    assert ops["LayerNormalization"] == 0
    assert ops["ReduceMean"] == 2


def test_fuse_layer_norm_skips_mismatched_scale_shape():
    # scale/bias must exactly match X's last dimension for
    # LayerNormalization's Scale/B inputs; a scalar scale (which still
    # broadcasts fine in the plain decomposition, so the model stays valid)
    # does not match and is left unfused.
    inits = _layer_norm_inits()
    inits[2] = _f32(np.array(1.5), "scale")
    model = _model(
        f"""
        g (float[2,4,8] X) => (float[2,4,8] Y)
        {{
          {_layer_norm_body("Mul")}
        }}
        """,
        initializer=inits,
        opset=17,
    )
    _, ops = _simplify(model)
    assert ops["LayerNormalization"] == 0


# --------------------------------------------------------------------------- #
# Reshape family applied to a constant: Reshape / Squeeze / Unsqueeze / Flatten
# only rearrange a tensor's shape, so applying one to an initializer (or a
# Constant node) is a pure metadata change and the node itself is redundant
# (eliminate_reshape_family_on_constant). Constant folding removes these too,
# so every test below runs with ``skip_constant_folding=True`` -- that is what
# makes them test the pass rather than the folder.
# --------------------------------------------------------------------------- #
def _simplify_no_folding(model):
    # Constant folding would remove a reshape-family node on a constant on its
    # own (by executing it); disabling it isolates the pure-graph pass.
    sim_model, check_ok = onnxsim.simplify(model, check_n=3, skip_constant_folding=True)
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _initializer_dims(model, exclude=()):
    return sorted(
        tuple(init.dims) for init in model.graph.initializer if init.name not in exclude
    )


def test_eliminate_reshape_on_initializer():
    # Reshape(W, {6,4}) on a [2,3,4] weight is just W with a different dims
    # list: the node goes away and the weight is stored pre-reshaped.
    W = np.random.randn(2, 3, 4)
    model = _model(
        """
        g (float[6,4] X) => (float[6,4] Y)
        <int64[2] s = {6,4}>
        {
          w = Reshape(W, s)
          Y = Add(X, w)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Reshape"] == 0
    # The old [2,3,4] initializer is unused after the rewrite and dropped by
    # eliminate_unused_initializer, so only the reshaped weight is left (`s`
    # goes with the Reshape node that used it).
    assert _initializer_dims(sim_model) == [(6, 4)]


def test_eliminate_reshape_on_initializer_resolves_minus_one_and_zero():
    # `-1` (infer from the element count) and, under the default allowzero=0,
    # `0` (copy this dim from the data input) both resolve against the
    # constant's own dims.
    W = np.random.randn(2, 3, 4)
    model = _model(
        """
        g (float[2,12] X) => (float[2,12] Y)
        <int64[2] s = {0,-1}>
        {
          w = Reshape(W, s)
          Y = Add(X, w)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Reshape"] == 0
    assert _initializer_dims(sim_model) == [(2, 12)]


def test_eliminate_squeeze_unsqueeze_flatten_on_initializer():
    # The rest of the family is handled the same way, all in one graph.
    A = np.random.randn(1, 3, 4)
    B = np.random.randn(3, 4)
    C = np.random.randn(2, 3, 4)
    model = _model(
        """
        g (float[3,4] X) => (float[3,4] Ya, float[1,3,4] Yb, float[2,12] Yc)
        <int64[1] axes = {0}>
        {
          a = Squeeze(A, axes)
          Ya = Add(X, a)
          b = Unsqueeze(B, axes)
          Yb = Add(a, b)
          c = Flatten<axis = 1>(C)
          Yc = Add(c, c)
        }
        """,
        initializer=[_f32(A, "A"), _f32(B, "B"), _f32(C, "C")],
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Squeeze"] == 0
    assert ops["Unsqueeze"] == 0
    assert ops["Flatten"] == 0
    assert _initializer_dims(sim_model) == [(1, 3, 4), (2, 12), (3, 4)]


def test_eliminate_reshape_on_constant_node_stays_a_constant():
    # A Constant node's value is not graph weight data, so the rewrite keeps it
    # a Constant node rather than promoting it to an initializer (the same
    # provenance distinction onnxsim's constant folding maintains).
    model = _model(
        """
        g (float[3,2] X) => (float[3,2] Y)
        <int64[2] s = {3,2}>
        {
          c = Constant<value = float[2,3] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}>()
          w = Reshape(c, s)
          Y = Add(X, w)
        }
        """
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Reshape"] == 0
    assert ops["Constant"] == 1
    assert not sim_model.graph.initializer
    (constant,) = [n for n in sim_model.graph.node if n.op_type == "Constant"]
    (value,) = [a.t for a in constant.attribute if a.name == "value"]
    assert list(value.dims) == [3, 2]


def test_eliminate_reshape_on_constant_declines_shared_weight():
    # W feeds the Reshape *and* an Add, so rewriting it would leave a second
    # copy of the weight in the graph next to the original. Not worth one node.
    W = np.random.randn(2, 3)
    model = _model(
        """
        g (float[2,3] X, float[3,2] Z) => (float[2,3] Ya, float[3,2] Yb)
        <int64[2] s = {3,2}>
        {
          Ya = Add(X, W)
          w = Reshape(W, s)
          Yb = Add(Z, w)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Reshape"] == 1
    assert _initializer_dims(sim_model, exclude={"s"}) == [(2, 3)]


def test_eliminate_reshape_on_constant_declines_non_constant_shape():
    # The target shape is only known at runtime, so there is no static dims
    # list to bake into the weight. It has to come from a *dynamically* shaped
    # tensor: onnxsim resolves Shape() on a statically shaped one into a
    # constant, and then the rewrite is both possible and correct. D's dims are
    # symbolic, so this asserts on graph structure only (check_n needs concrete
    # dims to run the model).
    W = np.random.randn(2, 3)
    model = _model(
        """
        g (float[3,2] X, float[N,M] D) => (float[3,2] Y)
        {
          s = Shape(D)
          w = Reshape(W, s)
          Y = Add(X, w)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    sim_model, check_ok = onnxsim.simplify(model, check_n=0, skip_constant_folding=True)
    assert check_ok
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Reshape"] == 1


def test_eliminate_reshape_on_constant_declines_graph_output():
    # Replacing the Reshape's output with the initializer would rename a value
    # the graph itself hands back, changing the model's interface.
    W = np.random.randn(2, 3)
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y, float[3,2] W_out)
        <int64[2] s = {3,2}>
        {
          Y = Add(X, X)
          W_out = Reshape(W, s)
        }
        """,
        initializer=[_f32(W, "W")],
    )
    sim_model, ops = _simplify_no_folding(model)
    assert ops["Reshape"] == 1
    assert [o.name for o in sim_model.graph.output] == ["Y", "W_out"]
