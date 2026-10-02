"""End-to-end coverage for the optimizer passes that onnxsim owns.

These four passes -- ``fuse_mul_into_conv``, ``fuse_consecutive_mul``,
``fuse_matmul_add_bias_into_gemm_batched`` and
``eliminate_reshape_around_elementwise`` -- are onnxsim-specific graph rewrites
that live under ``onnxsim/passes/`` and are injected into onnxoptimizer's
registry at runtime (see ``onnxsim/custom_optimizer_passes.*``). They used to be
tested in the onnxoptimizer fork in isolation; here they are exercised through
the full ``onnxsim.simplify`` pipeline, which also guards each rewrite with
onnxsim's own random-input equivalence check (``check_n``).
"""

import collections

import numpy as np
import onnx
from onnx import parser

import onnxsim
from onnxsim import model_checking


def _simplify(model, **kwargs):
    kwargs.setdefault("check_n", 3)
    sim_model, check_ok = onnxsim.simplify(model, **kwargs)
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _model(body, initializer=(), opset=13):
    # Pin a low IR version so the model loads under the older onnxruntime
    # bundled with some CI wheels; onnxsim's check_n runs the model through it.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _conv_out_feeds_mul(model):
    conv_outs = {o for n in model.graph.node if n.op_type == "Conv" for o in n.output}
    return any(
        n.op_type == "Mul" and any(i in conv_outs for i in n.input)
        for n in model.graph.node
    )


# --------------------------------------------------------------------------- #
# fuse_mul_into_conv
# --------------------------------------------------------------------------- #
def test_fuse_mul_into_conv_per_channel():
    # Conv -> Mul(per-output-channel [1, C, 1, 1] scale): the scale folds into
    # the Conv weights, so nothing multiplies the Conv output afterwards.
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    s = _f32(np.random.rand(1, 4, 1, 1), "S")
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,4,8,8] Y)
        {
          Z = Conv<pads = [1, 1, 1, 1]>(X, W)
          Y = Mul(Z, S)
        }
        """,
        initializer=[w, s],
    )
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1
    assert not _conv_out_feeds_mul(sim)


def test_fuse_mul_into_conv_scalar():
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,4,8,8] Y)
        <float S = {2.0}>
        {
          Z = Conv<pads = [1, 1, 1, 1]>(X, W)
          Y = Mul(Z, S)
        }
        """,
        initializer=[w],
    )
    sim, _ = _simplify(model)
    assert not _conv_out_feeds_mul(sim)


def test_fuse_mul_into_conv_singleton_rank_scale_keeps_bias_rank_one():
    w = _f32(np.arange(2, dtype=np.float32).reshape(2, 1, 1, 1) + 1, "W")
    b = _f32(np.array([0.5, -0.5]), "B")
    s = _f32(np.array([[[[2.0]]]]), "S")
    model = _model(
        """
        g (float[1,1,2,2] X) => (float[1,2,2,2] Y)
        {
          Z = Conv(X, W, B)
          Y = Mul(Z, S)
        }
        """,
        initializer=[w, b, s],
    )
    sim, ops = _simplify(model)
    onnx.checker.check_model(sim)
    assert ops["Conv"] == 1
    conv = next(n for n in sim.graph.node if n.op_type == "Conv")
    bias = next(t for t in sim.graph.initializer if t.name == conv.input[2])
    assert list(bias.dims) == [2]
    x = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    assert model_checking.compare(
        sim, model, n_times=1, input_data={"X": x}, verbose=False
    )


# --------------------------------------------------------------------------- #
# fuse_preceding_mul_into_conv
# --------------------------------------------------------------------------- #
def _mul_feeds_conv_in(model):
    conv_first_inputs = {n.input[0] for n in model.graph.node if n.op_type == "Conv"}
    return any(
        n.op_type == "Mul" and n.output[0] in conv_first_inputs
        for n in model.graph.node
    )


def test_fuse_preceding_mul_into_conv_per_channel():
    # Mul(X, per-input-channel [1, C, 1, 1] scale) -> Conv: the scale folds
    # into the Conv weights, so nothing multiplies the Conv input beforehand.
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    b = _f32(np.random.rand(4), "B")
    s = _f32(np.random.rand(1, 3, 1, 1), "S")
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,4,8,8] Y)
        {
          X2 = Mul(X, S)
          Y = Conv<pads = [1, 1, 1, 1]>(X2, W, B)
        }
        """,
        initializer=[w, b, s],
    )
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1
    assert not _mul_feeds_conv_in(sim)


def test_fuse_preceding_mul_into_conv_scalar():
    w = _f32(np.random.rand(4, 3, 3, 3), "W")
    model = _model(
        """
        g (float[1,3,8,8] X) => (float[1,4,8,8] Y)
        <float S = {2.0}>
        {
          X2 = Mul(X, S)
          Y = Conv<pads = [1, 1, 1, 1]>(X2, W)
        }
        """,
        initializer=[w],
    )
    sim, _ = _simplify(model)
    assert not _mul_feeds_conv_in(sim)


def test_fuse_preceding_mul_into_conv_grouped_per_channel_not_fused():
    # A per-channel scale on a grouped Conv is left alone: it would need
    # re-slicing per group to line up with the weight layout, which this pass
    # does not attempt. The model must still simplify correctly.
    w = _f32(np.random.rand(4, 1, 3, 3), "W")
    s = _f32(np.random.rand(1, 4, 1, 1), "S")
    model = _model(
        """
        g (float[1,4,8,8] X) => (float[1,4,8,8] Y)
        {
          X2 = Mul(X, S)
          Y = Conv<pads = [1, 1, 1, 1], group = 4>(X2, W)
        }
        """,
        initializer=[w, s],
    )
    sim, ops = _simplify(model)
    assert ops["Conv"] == 1


# --------------------------------------------------------------------------- #
# fuse_consecutive_mul
# --------------------------------------------------------------------------- #
def test_fuse_consecutive_mul_scalar():
    # Mul(X, C1) -> Mul(., C2) with constant C1/C2 collapses to a single Mul by
    # a fused C1*C2 constant (X is a runtime input, so it cannot be folded away).
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Z)
        <float C1 = {2.0}, float C2 = {3.0}>
        {
          Y = Mul(X, C1)
          Z = Mul(Y, C2)
        }
        """
    )
    sim, ops = _simplify(model)
    assert ops["Mul"] == 1


def test_fuse_consecutive_mul_per_channel():
    # Per-channel (C, 1, 1) scale composed with a scalar factor (LayerScale).
    model = _model(
        """
        g (float[1,4,2,2] X) => (float[1,4,2,2] Z)
        <float[4,1,1] C1 = {1.0, 2.0, 3.0, 4.0}, float C2 = {0.5}>
        {
          Y = Mul(X, C1)
          Z = Mul(Y, C2)
        }
        """
    )
    sim, ops = _simplify(model)
    assert ops["Mul"] == 1


# --------------------------------------------------------------------------- #
# fuse_matmul_add_bias_into_gemm_batched
# --------------------------------------------------------------------------- #
def test_fuse_matmul_add_bias_into_gemm_batched():
    # A rank-3 linear layer MatMul(X[2,3,4], W[4,5]) + b[5] that the 2-D-only
    # fuse_matmul_add_bias_into_gemm cannot fuse becomes a Gemm (with reshape
    # scaffolding), so no batched MatMul remains.
    w = _f32(np.random.randn(4, 5), "W")
    b = _f32(np.random.randn(5), "B")
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3,5] A)
        {
          Z = MatMul(X, W)
          A = Add(Z, B)
        }
        """,
        initializer=[w, b],
    )
    sim, ops = _simplify(model)
    assert ops["Gemm"] >= 1
    assert "MatMul" not in ops


def test_fuse_matmul_add_bias_into_gemm_batched_dynamic_many_matches():
    # X has dynamic leading dims (symbolic "batch"/"seq"), so runTransform
    # takes the Shape/Slice/Concat path (see the pass's own comment) instead
    # of a plain Reshape-to-a-constant-shape -- that path mints up to 5 fresh
    # initializer names per match instead of 2. Many independent linear
    # layers over the same input force many such matches within one
    # runPass() call, exercising FuseMatMulAddBiasIntoGemmBatched's batched
    # name-reservation (nextReservedName/reserveUniqueNames): a bug in how it
    # hands out/consumes reserved names would surface as a duplicate
    # initializer name here, silently corrupting one layer's weights/bias
    # with another's.
    n_layers = 8
    initializers = []
    body = ""
    outputs = []
    for i in range(n_layers):
        w = _f32(np.random.randn(4, 5), f"W{i}")
        b = _f32(np.random.randn(5), f"B{i}")
        initializers += [w, b]
        body += f"Z{i} = MatMul(X, W{i})\nA{i} = Add(Z{i}, B{i})\n"
        outputs.append(f"float[batch,seq,5] A{i}")
    model = _model(
        f"""
        g (float[batch,seq,4] X) => ({", ".join(outputs)})
        {{
          {body}
        }}
        """,
        initializer=initializers,
        opset=13,
    )
    sim, ops = _simplify(model, test_input_shapes={"X": (2, 3, 4)})
    assert ops["Gemm"] >= n_layers
    assert "MatMul" not in ops

    names = [init.name for init in sim.graph.initializer]
    assert len(names) == len(set(names)), (
        "duplicate initializer name: reserved-name collision"
    )


# --------------------------------------------------------------------------- #
# eliminate_reshape_around_elementwise (cancels the batched-Gemm reshape
# scaffolding between two linear layers separated by an element-wise op)
# --------------------------------------------------------------------------- #
def test_eliminate_reshape_around_elementwise():
    # Two rank-3 linear layers with a Relu in between. Each linear becomes a
    # Gemm wrapped in Reshape(2-D)/Reshape(N-D); the inverse reshapes around the
    # Relu cancel, leaving the Gemms without the intermediate reshape pair.
    w0 = _f32(np.random.randn(4, 6), "W0")
    b0 = _f32(np.random.randn(6), "B0")
    w1 = _f32(np.random.randn(6, 5), "W1")
    b1 = _f32(np.random.randn(5), "B1")
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3,5] A1)
        {
          Z0 = MatMul(X, W0)
          A0 = Add(Z0, B0)
          R = Relu(A0)
          Z1 = MatMul(R, W1)
          A1 = Add(Z1, B1)
        }
        """,
        initializer=[w0, b0, w1, b1],
    )
    sim, ops = _simplify(model)
    # Both linear layers dispatch as Gemms and the batched MatMuls are gone.
    assert ops["Gemm"] >= 2
    assert "MatMul" not in ops
    # The reshape scaffolding around the Relu is cancelled: at most the two
    # outer reshapes survive (never the 4 a naive per-layer rewrite would add).
    assert ops["Reshape"] <= 2
