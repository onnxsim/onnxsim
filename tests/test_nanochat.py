# Integration/regression test: a nanochat-style GPT
# (https://github.com/karpathy/nanochat) simplified by onnxsim.
#
# nanochat has no ONNX export path of its own to reproduce (see
# scripts/nanochat/model.py's docstring for why -- its attention backend is
# Flash Attention 3 plus a sliding-window pattern, neither of which traces to
# ONNX). This test instead guards against onnxsim regressions on the
# standalone, ONNX-exportable reimplementation of nanochat's core
# architecture in scripts/nanochat/model.py: rotary embeddings, QK norm,
# untied embeddings, relu^2 MLP, Group-Query Attention, and RMSNorm with no
# learnable params.
#
# The model is built with random weights at a tiny size (no checkpoint or
# dataset needed) -- the graph structure onnxsim simplifies is identical
# across model sizes, so this stays fast and fully offline. The standalone
# harness in scripts/nanochat/simplify_nanochat.py exercises nanochat's own
# depth-based sizing (``--depth``); see scripts/nanochat/RESULTS.md for a
# captured run.
#
# ``torch`` is not a normal test dependency, so this test ``importorskip``s
# it and skips unless it is already installed. To run it locally::
#
#     pip install torch onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_nanochat.py -v

import os
import sys

import onnxruntime
import pytest

torch = pytest.importorskip("torch")

_NANOCHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "nanochat")
if _NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, _NANOCHAT_DIR)

from model import GPT, GPTConfig  # noqa: E402

from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402


@pytest.mark.parametrize(
    "n_kv_head",
    [
        pytest.param(4, id="mha"),  # n_kv_head == n_head, matches base_train.py
        pytest.param(2, id="gqa"),  # n_kv_head < n_head, exercises the GQA path
    ],
)
def test_nanochat_export_simplify(n_kv_head):
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=n_kv_head,
        n_embd=32,
    )
    model = GPT(config, pad_vocab_size_to=8).eval()
    dummy_input = torch.randint(
        0, config.vocab_size, (1, config.sequence_len), dtype=torch.long
    )

    opt = export_simplify_and_check_by_python_api(
        model,
        (dummy_input,),
        export_kwargs={
            "opset_version": 17,
            "input_names": ["input_ids"],
            "output_names": ["logits"],
        },
    )

    op_types = {node.op_type for node in opt.graph.node}
    # Core architectural ops must survive simplification.
    assert "Softmax" in op_types  # attention
    assert "Tanh" in op_types  # logit softcap

    session = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = session.run(None, {"input_ids": dummy_input.numpy()})
    assert outputs[0].shape == (1, config.sequence_len, config.vocab_size)
