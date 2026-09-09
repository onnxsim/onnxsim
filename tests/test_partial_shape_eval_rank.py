"""Partial shape evaluation must not fold a rank>=2 value from flat shape data.

Regression tests for onnxsim issue #1284.

``_EvalPartialShape`` / ``_EvalPartialShapeOnGraph``
(onnxsim/partial_shape_eval.cpp) fold a node whose value ONNX's *data
propagation* -- or onnxsim's own ``SymTensor`` evaluator -- managed to resolve.
Both of those represent a value as a **flat sequence with no rank of its own**:
data propagation uses a ``TensorShapeProto`` (a shape vector), and every ONNX
``PartialDataPropagationFunction`` is written for that rank<=1 form (``Slice``'s
own says "Only supports axis = 0 since the data comes from Shape", and
``DataPropagationContextImpl::getInputData`` only converts a rank-0/rank-1
initializer); ``SymTensor`` is likewise a scalar or a vector and nothing else
(``EvalSlice``: "rank-1 data, axis 0 only"; ``Transpose`` is evaluated as the
identity, which only holds at rank <= 1).

The folder used to accept such a sequence for an output of *any* rank as long as
the element count matched, which silently reinterprets a rank-1 element order as
a rank-N one. The graph below -- the pads construction PyTorch emits for
``F.pad``, which NNSmith reproduces -- is exactly that case: a rank-2 ``[3, 2]``
``Slice`` with ``steps=-1`` reverses the 3 *rows*, but the propagator reverses
all 6 flat entries, so the ``Pad`` downstream pads the wrong sides and every
value derived from it changes.

These tests use only the ``onnx`` Python API, so they run without torch.
"""

import numpy as np
import onnx
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

import onnxsim

# The sentinel PyTorch's ``F.pad`` export writes as the reversing Slice's `ends`.
# Any value below ``-numel`` triggered the bug; this is the one real exports use.
_INT64_END_SENTINEL = -9223372036854775807


def _model(body, initializer=(), opset=17, ir_version=8):
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


def _pads_chain_initializers(torch_pad):
    """The constants of the ``F.pad`` pads construction.

    ``torch_pad`` is torch's own padding list -- ``(before, after)`` per axis,
    last axis first -- which the chain rearranges into ONNX ``Pad``'s
    ``[before..., after...]`` layout. It must cover every axis of the tensor
    being padded, which is when torch emits the empty ``ConstantOfShape``
    below (its zero-fill is what pads a shorter list out to full rank).
    """
    return [
        numpy_helper.from_array(np.array([0], np.int64), "cos_in"),
        numpy_helper.from_array(np.array(torch_pad, np.int64), "torch_pad"),
        numpy_helper.from_array(np.array([-1, 2], np.int64), "sh_m1_2"),
        numpy_helper.from_array(np.array([-1], np.int64), "sl_starts"),
        numpy_helper.from_array(np.array([_INT64_END_SENTINEL], np.int64), "sl_ends"),
        numpy_helper.from_array(np.array([0], np.int64), "sl_axes"),
        numpy_helper.from_array(np.array([-1], np.int64), "sl_steps"),
        numpy_helper.from_array(np.array([-1], np.int64), "sh_m1"),
    ]


# Reused as a text fragment so callers can interpolate it like the rest of the
# body (see CLAUDE.md's note on reusable node sequences).
_PADS_CHAIN = """
  cos = ConstantOfShape <value = int64[1] {0}> (cos_in)
  cc0 = Concat <axis = 0> (torch_pad, cos)
  rows = Reshape (cc0, sh_m1_2)
  rev = Slice (rows, sl_starts, sl_ends, sl_axes, sl_steps)
  cols = Transpose <perm = [1, 0]> (rev)
  flat = Reshape (cols, sh_m1)
  pads = Cast <to = 7> (flat)
"""


def _run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def test_reversing_slice_on_rank2_is_folded_correctly():
    # The pads chain on its own: `pads` is entirely constant, so simplify()
    # folds it away -- but it must fold to the value the graph actually
    # computes, not to a flat reversal of it.
    model = _model(
        f"""
        g (float[1] dummy) => (int64[6] pads, float[1] keep)
        {{
        {_PADS_CHAIN}
          keep = Identity (dummy)
        }}
        """,
        initializer=_pads_chain_initializers((1, 2, 0, 0, 0, 0)),
    )
    onnx.checker.check_model(model)
    feeds = {"dummy": np.zeros(1, np.float32)}

    expected = _run(model, feeds)[0]
    # ONNX Pad's [before_axis0.., after_axis0..] for a rank-3 tensor padded on
    # its last axis only -- what torch's `F.pad(x, (1, 2))` means. Reversing
    # the flat sequence instead of the 3 rows (the bug) yields the *reversed*
    # (2, 1) padding, [0, 0, 2, 0, 0, 1], which is why the mismatch is a shift
    # rather than a crash: the element count is right, the values are not.
    assert expected.tolist() == [0, 0, 1, 0, 0, 2]

    sim, ok = onnxsim.simplify(model, check_n=3)

    assert ok
    assert "Slice" not in [n.op_type for n in sim.graph.node]
    np.testing.assert_array_equal(_run(sim, feeds)[0], expected)


def test_pad_trilu_chain_survives_simplification():
    # onnxsim issue #1284: the wrongly folded pads reach a `Pad`, so `Trilu`
    # sees a differently-shaped tensor and every value downstream of the
    # Sigmoid/Mul/Cos chain moves -- by up to 0.36 on the reported graph, far
    # outside `Cos`'s own [-1, 1] range being compared at rtol 1e-4.
    model = _model(
        f"""
        g (float[1,2,3] data, float[3] side) => (float[1,2,6] out)
        {{
        {_PADS_CHAIN}
          padded = Pad <mode = "constant"> (data, pads, pad_value)
          upper = Trilu <upper = 1> (padded, trilu_k)
          gate = Sigmoid (upper)
          scale = Concat <axis = 0> (side, weight)
          scaled = Mul (gate, scale)
          out = Cos (scaled)
        }}
        """,
        initializer=_pads_chain_initializers((1, 2, 0, 0, 0, 0))
        + [
            numpy_helper.from_array(np.array(0.0, np.float32), "pad_value"),
            numpy_helper.from_array(np.array(0, np.int64), "trilu_k"),
            numpy_helper.from_array(np.array([0.5, 1.5, 2.5], np.float32), "weight"),
        ],
    )
    onnx.checker.check_model(model)

    sim, ok = onnxsim.simplify(model, check_n=5)
    assert ok

    rng = np.random.RandomState(0)
    for _ in range(5):
        feeds = {
            "data": rng.randn(1, 2, 3).astype(np.float32),
            "side": rng.randn(3).astype(np.float32),
        }
        np.testing.assert_allclose(
            _run(sim, feeds)[0], _run(model, feeds)[0], rtol=1e-4, atol=1e-5
        )


def test_rank1_shape_scaffolding_is_still_folded():
    # The rank<=1 case the folder exists for must keep working: a reversing
    # Slice straight off a rank-1 Concat is exactly what data propagation
    # models, and it still folds (and folds correctly).
    model = _model(
        """
        g (float[1] dummy) => (int64[6] rev, float[1] keep)
        {
          cos = ConstantOfShape <value = int64[1] {0}> (cos_in)
          cc0 = Concat <axis = 0> (flat_values, cos)
          rev = Slice (cc0, sl_starts, sl_ends, sl_axes, sl_steps)
          keep = Identity (dummy)
        }
        """,
        initializer=[
            numpy_helper.from_array(np.array([0], np.int64), "cos_in"),
            numpy_helper.from_array(
                np.array([12, 0, -65487, 0, 45, 0], np.int64), "flat_values"
            ),
            numpy_helper.from_array(np.array([-1], np.int64), "sl_starts"),
            numpy_helper.from_array(
                np.array([_INT64_END_SENTINEL], np.int64), "sl_ends"
            ),
            numpy_helper.from_array(np.array([0], np.int64), "sl_axes"),
            numpy_helper.from_array(np.array([-1], np.int64), "sl_steps"),
        ],
    )
    onnx.checker.check_model(model)
    feeds = {"dummy": np.zeros(1, np.float32)}

    expected = _run(model, feeds)[0]
    assert expected.tolist() == [0, 45, 0, -65487, 0, 12]

    sim, ok = onnxsim.simplify(model, check_n=3)

    assert ok
    assert "Slice" not in [n.op_type for n in sim.graph.node]
    np.testing.assert_array_equal(_run(sim, feeds)[0], expected)
