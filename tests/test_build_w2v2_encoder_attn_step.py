"""Tests for ``scripts/axera/build_w2v2_encoder_attn_step.py``'s pure-ONNX
graph transforms -- ``_strip_isnan_guard`` and ``_find_q_proj_weight``.

These are the two model-specific pieces of that script's build pipeline:
neither needs ``torch``/``transformers`` (the heavy, optional deps the real
`Wav2Vec2Model` export requires), so this exercises them directly against
small hand-built graphs that reproduce the exact node shapes a real wav2vec2
export produces, per ``docs/axera-audio-speech-op-coverage.md``'s trace --
``attn_weights = Where(IsNaN(attn_weights), 0, attn_weights)`` sitting
inline between a layer's own ``Softmax`` and its ``MatMul`` with ``V``. The
real end-to-end build (export -> strip -> ``build_resident_step`` ->
Pulsar2 compile -> real AX650N run) is exercised manually, following this
project's convention for every other torch/transformers-dependent axera
build script (``build_whisper_train_step.py``,
``build_w2v2_feature_extractor_step.py``): see
``docs/axera-audio-speech-op-coverage.md``'s wav2vec2 section for that real
hardware result.
"""

import os
import sys

import onnx
from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_w2v2_encoder_attn_step as m  # noqa: E402


def _attn_tail_model():
    """A minimal stand-in for one encoder layer's attention tail: `scores ->
    Softmax -> (the numerical-stability guard) -> MatMul(V) -> out`, with
    the guard's shape matching the real export exactly --
    `Where(IsNaN(attn), zero, attn)`, condition/X/Y in that order.
    """
    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,4,4] scores, float[1,4,4] v) => (float[1,4,4] out)
        <float zero = {0.0}>
        {
          probs = Softmax<axis=-1>(scores)
          is_nan = IsNaN(probs)
          probs_safe = Where(is_nan, zero, probs)
          out = MatMul(probs_safe, v)
        }
        """
    )


def test_strip_isnan_guard_removes_both_nodes_and_rewires_consumers():
    model = _attn_tail_model()
    n = m._strip_isnan_guard(model)
    assert n == 1
    op_types = [node.op_type for node in model.graph.node]
    assert "IsNaN" not in op_types
    assert "Where" not in op_types
    onnx.checker.check_model(model)

    matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    assert matmul.input[0] == softmax.output[0]


def test_strip_isnan_guard_is_numerically_exact_when_no_row_is_masked():
    """The whole point of stripping: this rewrite only changes what compiles,
    never what the graph computes, for inputs the guard was never protecting
    (no fully-masked row -> `IsNaN` never true)."""
    ort = __import__("onnxruntime")
    import numpy as np

    before = _attn_tail_model()
    after = _attn_tail_model()
    m._strip_isnan_guard(after)
    onnx.checker.check_model(after)

    rng = np.random.RandomState(0)
    scores = rng.randn(1, 4, 4).astype(np.float32)
    v = rng.randn(1, 4, 4).astype(np.float32)
    outs = []
    for model in (before, after):
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outs.append(sess.run(None, {"scores": scores, "v": v})[0])
    assert np.allclose(outs[0], outs[1], atol=1e-6), np.abs(outs[0] - outs[1]).max()


def test_strip_isnan_guard_leaves_unrelated_where_alone():
    """Not every `Where` is the numerical-stability guard -- the real export
    also has a mask-bias `Where` whose condition comes from `Expand`, not
    `IsNaN` (`docs/axera-audio-speech-op-coverage.md`'s trace). Only a
    `Where` whose condition is produced by `IsNaN` should be touched."""
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,4] scores, bool[1,4] mask) => (float[1,4] out)
        <float neg_inf = {-1e9}>
        {
          out = Where(mask, scores, neg_inf)
        }
        """
    )
    n = m._strip_isnan_guard(model)
    assert n == 0
    assert [node.op_type for node in model.graph.node] == ["Where"]


def test_find_q_proj_weight_traces_producer_edges_not_names():
    """Real wav2vec2 exports name every encoder-layer weight generically
    (`onnx::MatMul_NNN`, per this module's own docstring) -- the function
    under test must find `q_proj`'s weight by tracing node names/edges, not
    by pattern-matching an initializer's own name.

    `_find_q_proj_weight` searches by walking node *outputs* (an exact
    tensor name match, e.g. `/m/encoder/layers.{layer}/attention/
    Softmax_output_0`) down to a `MatMul` node whose own *name* (the
    `NodeProto.name` field torch.onnx's exporter derives from the traced
    module path, e.g. `.../q_proj/MatMul`) ends in `q_proj/MatMul`. The
    ONNX text format's `<...>` node syntax sets node *attributes*, not the
    `NodeProto.name` field itself (confirmed directly: a `<name = "...">`
    entry parses without error but leaves `NodeProto.name` empty), so this
    model is built with `onnx.helper.make_node` instead -- the CLAUDE.md-
    documented exception for a case the text form cannot express.
    """
    import numpy as np
    from onnx import helper

    nodes = [
        helper.make_node(
            "MatMul",
            ["hidden", "qw"],
            ["/m/encoder/layers.0/attention/q_proj/MatMul_output_0"],
            name="/m/encoder/layers.0/attention/q_proj/MatMul",
        ),
        helper.make_node(
            "MatMul",
            ["hidden", "kw"],
            ["/m/encoder/layers.0/attention/k_proj/MatMul_output_0"],
            name="/m/encoder/layers.0/attention/k_proj/MatMul",
        ),
        helper.make_node(
            "MatMul",
            [
                "/m/encoder/layers.0/attention/q_proj/MatMul_output_0",
                "/m/encoder/layers.0/attention/k_proj/MatMul_output_0",
            ],
            ["/m/encoder/layers.0/attention/scores/MatMul_output_0"],
            name="/m/encoder/layers.0/attention/scores/MatMul",
        ),
        helper.make_node(
            "Softmax",
            ["/m/encoder/layers.0/attention/scores/MatMul_output_0"],
            ["/m/encoder/layers.0/attention/Softmax_output_0"],
            name="/m/encoder/layers.0/attention/Softmax",
            axis=-1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info(
                "/m/encoder/layers.0/attention/Softmax_output_0",
                onnx.TensorProto.FLOAT,
                [1, 4],
            )
        ],
        initializer=[
            onnx.numpy_helper.from_array(np.eye(4, dtype=np.float32), "qw"),
            onnx.numpy_helper.from_array(np.eye(4, dtype=np.float32), "kw"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    assert m._find_q_proj_weight(model, layer=0) == "qw"
