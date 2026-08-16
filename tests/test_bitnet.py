# Integration/regression test: converting a BitNet b1.58 ONNX export into
# onnxruntime's native low-bit kernels, with onnxsim in the middle of the
# pipeline.
#
# BitNet b1.58 (https://github.com/microsoft/BitNet) is Microsoft's ternary-
# weight LLM family: every projection weight is one of {-s, 0, +s}. An ONNX
# export still stores those as dense float32 initializers, so the converter in
# ``scripts/bitnet`` rewrites them into either 2-bit ``com.microsoft::
# MatMulNBits`` or BitNet's own W1.58A8 ``MatMulInteger`` arithmetic.
#
# onnxsim sits on both sides of that rewrite, and this test pins both halves of
# its role:
#
#   * **Before** the rewrite, it folds the export's rotary and shape arithmetic
#     away. It also folds ``Transpose(weight) -> MatMul`` into the plain
#     ``[K, N]`` initializers ternary detection needs: when the exporter has not
#     already done that itself, onnxsim is the difference between zero
#     convertible layers and all of them (``test_simplify_exposes_..``).
#   * **After** the rewrite, onnxsim must still simplify a graph containing
#     contrib-domain ``MatMulNBits`` without breaking it.
#
# The model is a small, randomly-initialised BitNet decoder built offline by
# ``scripts/bitnet/bitnet_model.py`` (no checkpoint download): RMSNorm,
# subln, grouped-query attention with rotary embeddings, and a squared-ReLU
# gated FFN, with BitLinear on all seven projections. Only the sizes differ
# from the released 2B-4T model; the op set and graph topology are the same.
#
# ``torch`` is only needed for the export leg; the converter itself is
# onnx+numpy only, so the packing/detection tests always run. To run the whole
# file locally::
#
#     pip install torch onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_bitnet.py -v

import os
import sys

import numpy as np
import onnx
import pytest

import onnxsim

_BITNET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "bitnet"
)
if _BITNET_DIR not in sys.path:
    sys.path.insert(0, _BITNET_DIR)

import bitnet_ort  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")

# Seven BitLinear projections per layer: q/k/v/o and gate/up/down.
BITLINEARS_PER_LAYER = 7
LAYERS = 2

# 2-bit MatMulNBits is only in recent onnxruntime builds; older ones accept just
# 4 and 8 bits and raise when the session is created ("additional bits support
# is planned"). Conversion still works there -- the model is simply not loadable
# locally -- so only the cases that *execute* a 2-bit graph are gated.
requires_2bit = pytest.mark.skipif(
    not bitnet_ort.matmul_nbits_supported(2),
    reason="onnxruntime build has no 2-bit MatMulNBits kernel",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A tiny BitNet decoder exported to ONNX, plus reference feeds/outputs."""
    torch = pytest.importorskip("torch")
    del torch
    import bitnet_model

    config = bitnet_model.BitNetConfig(num_hidden_layers=LAYERS)
    path = str(tmp_path_factory.mktemp("bitnet") / "bitnet.onnx")
    bitnet_model.export_onnx(bitnet_model.build(config), path, seq=16)
    model = onnx.load(path)
    feeds = bitnet_ort.random_feeds(model, batch=2, seq=24)
    return model, feeds, bitnet_ort.run(model, feeds)


@pytest.fixture(scope="module")
def simplified(exported):
    model, feeds, _ = exported
    sim, check_ok = onnxsim.simplify(model, check_n=1, input_data=feeds)
    assert check_ok, "onnxsim numerical check failed on the BitNet export"
    return sim


def test_ternary_detection_round_trips():
    """as_ternary + pack_matmul_nbits reproduce the weight exactly."""
    rng = np.random.RandomState(0)
    scale = np.float32(0.037)
    weight = (rng.choice([-1, 0, 1], size=(96, 8)) * scale).astype(np.float32)

    ternary = bitnet_ort.as_ternary(weight)
    assert ternary is not None
    assert ternary.per_tensor
    np.testing.assert_allclose(ternary.scales, scale)
    np.testing.assert_array_equal(
        ternary.codes.astype(np.float32) * ternary.scales, weight
    )

    packed, scales, zero_points = bitnet_ort.pack_matmul_nbits(ternary, block_size=32)
    assert packed.shape == (8, 3, 8)  # [N, K/block, block * 2bits / 8]
    assert scales.shape == (8 * 3,)
    # Zero-points pack along the block dimension, one 2-bit field per block,
    # and every field must be 1 so that code 0/1/2 dequantizes to -1/0/+1.
    assert zero_points.shape == (8, 1)
    for block in range(3):
        np.testing.assert_array_equal((zero_points >> (2 * block)) & 0b11, 1)


def test_non_ternary_weights_are_rejected():
    rng = np.random.RandomState(0)
    assert (
        bitnet_ort.as_ternary(rng.standard_normal((32, 8)).astype(np.float32)) is None
    )
    # Four levels is one too many.
    four_level = (rng.choice([-2, -1, 0, 1], size=(32, 8)) * 0.1).astype(np.float32)
    assert bitnet_ort.as_ternary(four_level) is None
    # An all-zero matrix carries no BitNet signal.
    assert bitnet_ort.as_ternary(np.zeros((32, 8), np.float32)) is None


def test_simplify_shrinks_the_export(exported, simplified):
    raw, _, _ = exported
    _, stats = bitnet_ort.convert(simplified)

    assert stats.converted == BITLINEARS_PER_LAYER * LAYERS
    assert len(simplified.graph.node) < len(raw.graph.node)
    # The LM head is full precision in BitNet and must stay a float MatMul.
    assert [name for name, _ in stats.skipped] == ["/lm_head/MatMul"]


def test_simplify_exposes_transposed_weights(tmp_path):
    """onnxsim's constant folding is what makes an unfolded export convertible.

    ``F.linear`` is ``x @ weight.T``, so a BitLinear exports as
    ``Transpose(weight) -> MatMul``. Ternary detection needs a plain
    initializer: when the exporter does not fold that transpose itself, onnxsim
    is what turns an unconvertible graph into a convertible one.
    """
    pytest.importorskip("torch")
    import bitnet_model

    config = bitnet_model.BitNetConfig(num_hidden_layers=LAYERS)
    path = str(tmp_path / "unfolded.onnx")
    bitnet_model.export_onnx(
        bitnet_model.build(config), path, seq=16, constant_folding=False
    )
    unfolded = onnx.load(path)

    _, before = bitnet_ort.convert(unfolded)
    assert before.converted == 0

    simplified, _ = onnxsim.simplify(unfolded, check_n=0)
    _, after = bitnet_ort.convert(simplified)
    assert after.converted == BITLINEARS_PER_LAYER * LAYERS


@pytest.mark.parametrize("mode", [pytest.param("nbits", marks=requires_2bit), "int8"])
def test_convert_runs_in_onnxruntime(exported, simplified, mode):
    _, feeds, reference = exported
    converted, stats = bitnet_ort.convert(simplified, mode=mode)
    onnx.checker.check_model(converted)

    assert stats.converted == BITLINEARS_PER_LAYER * LAYERS
    assert stats.weight_bytes_after < stats.weight_bytes_before

    op_types = {node.op_type for node in converted.graph.node}
    assert ("MatMulNBits" if mode == "nbits" else "MatMulInteger") in op_types

    actual = bitnet_ort.run(converted, feeds)
    assert len(actual) == len(reference)
    # Both modes quantize activations to int8, so logits move by ~1%; the
    # prediction itself must not.
    assert bitnet_ort.max_relative_error(reference, actual) < 0.05
    predicted = np.argmax(actual[0], axis=-1)
    expected = np.argmax(reference[0], axis=-1)
    assert np.mean(predicted == expected) >= 0.9


def test_nbits_shrinks_the_weights(simplified):
    """Ternary float32 -> 2 bits is a 16x saving before per-block scales.

    Packing needs no onnxruntime kernel, so this holds even where the 2-bit
    kernel is missing.
    """
    _, stats = bitnet_ort.convert(simplified)
    assert stats.compression > 8.0


@requires_2bit
def test_nbits_weights_are_losslessly_packed(exported, simplified):
    """2-bit packing is exact: only the compute type costs accuracy."""
    _, feeds, reference = exported
    converted, _ = bitnet_ort.convert(
        simplified, accuracy_level=bitnet_ort.ACCURACY_LEVEL_FP32
    )
    actual = bitnet_ort.run(converted, feeds)
    assert bitnet_ort.max_relative_error(reference, actual) < 1e-5


def test_block_size_avoids_the_unvectorized_kernels(exported, simplified):
    """Auto-selected block sizes must have an int8 MatMulNBits kernel.

    onnxruntime accepts block sizes 16 and 256 but has no int8 kernel for them
    at 2 bits, so it silently falls back to a path ~40x slower than plain
    float. Picking one of those would quietly undo the conversion's point.
    """
    converted, _ = bitnet_ort.convert(simplified)
    blocks = {
        attr.i
        for node in converted.graph.node
        if node.op_type == "MatMulNBits"
        for attr in node.attribute
        if attr.name == "block_size"
    }
    assert blocks
    assert blocks <= {32, 64, 128}


@pytest.mark.parametrize("mode", [pytest.param("nbits", marks=requires_2bit), "int8"])
def test_onnxsim_simplifies_the_converted_model(exported, simplified, mode):
    _, feeds, reference = exported
    converted, _ = bitnet_ort.convert(simplified, mode=mode)

    resimplified, check_ok = onnxsim.simplify(converted, check_n=1, input_data=feeds)
    assert check_ok, f"onnxsim numerical check failed on the {mode} model"
    assert len(resimplified.graph.node) <= len(converted.graph.node)

    actual = bitnet_ort.run(resimplified, feeds)
    assert bitnet_ort.max_relative_error(reference, actual) < 0.05
