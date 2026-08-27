#!/usr/bin/env python3
"""Builds a tiny, hand-crafted decoder-only export (decoder_model.onnx +
decoder_with_past_model.onnx, no encoder_model.onnx -- the causal-LM shape,
per README.md's "What optimum-onnx actually exports") whose KV cache is
INT8 from the start, WITHOUT needing torch/transformers/optimum -- just the
`onnx` package, mirroring make_toy_seq2seq.py's role for the float case.

This exists specifically to regression-test detail::BorrowView's INT8 case
(kv_cache_pipeline.h) end to end through the real KvCachePipeline/CLI: an
onnxsim.quantize_kv_cache-quantized model has an INT8 past_key_values.*/
present.* stream, and nothing previously exercised that path through this
tool's own C++ (the Python-side numerics are already covered by
tests/test_kv_cache_quantization.py in the main onnxsim package -- this
script is deliberately *not* built by running that quantizer over a real
export, so it can isolate the C++ pipeline's own INT8 tensor threading from
the Python quantizer's calibration/scale logic).

The model (single stream, no real attention -- just enough structure to
exercise Concat-growing an INT8 cache across many steps):

  decoder_model.onnx (step 0, no past):
    present.0.key = Cast(input_ids, INT8)               -- seeds the cache
    logits = OneHot(Mod(ReduceSum(Cast(present.0.key, INT64)), vocab_size),
                     vocab_size)

  decoder_with_past_model.onnx (every step after):
    new_key = Cast(input_ids, INT8)
    present.0.key = Concat(past_key_values.0.key, new_key, axis=1)  -- INT8,
      genuinely grows every step (unlike make_toy_seq2seq.py's single-slot
      Identity cache) -- exactly the shape onnxsim.quantize_kv_cache
      produces and detail::BorrowView needs to thread across Run() calls.
    logits = OneHot(Mod(ReduceSum(Cast(present.0.key, INT64)), vocab_size),
                     vocab_size)

Since `logits` at every step is a deterministic function of the *entire*
cache's own token history (not just the latest token, unlike
make_toy_seq2seq.py's "always +1" trick), a pipeline that fails to thread
the INT8 cache correctly across steps -- drops it, re-feeds a stale or
zeroed buffer, or corrupts it during the borrow/move dance -- changes the
generated sequence, not just its performance. See compute_expected_ids()
for the independent reference.

Usage:
    python3 make_toy_int8_kv_decoder.py -o toy_int8_kv --vocab-size 7 \\
        --decoder-start-token-id 2
"""

import argparse
import os

import onnx
from onnx import TensorProto, checker, helper


def _cache_sum_logits_subgraph(prefix, cache_name, vocab_size):
    """Nodes computing
    logits = OneHot(Mod(ReduceSum(Cast(cache_name, INT64)), vocab_size), vocab_size),
    reshaped to [1, 1, vocab_size] -- cache_name is an INT8 tensor of any
    rank/length (the whole KV cache so far). Returns (nodes, initializers,
    logits_name).
    """
    cache_i64 = f"{prefix}_cache_i64"
    total = f"{prefix}_total"
    idx0 = f"{prefix}_idx0"
    idx2d = f"{prefix}_idx2d"
    logits = "logits"
    nodes = [
        helper.make_node("Cast", [cache_name], [cache_i64], to=TensorProto.INT64),
        helper.make_node("ReduceSum", [cache_i64], [total], keepdims=0),
        helper.make_node("Mod", [total, f"{prefix}_vocab_size"], [idx0]),
        helper.make_node("Reshape", [idx0, f"{prefix}_shape_1x1"], [idx2d]),
        helper.make_node(
            "OneHot",
            [idx2d, f"{prefix}_depth", f"{prefix}_onehot_values"],
            [logits],
            axis=-1,
        ),
    ]
    initializers = [
        helper.make_tensor(f"{prefix}_vocab_size", TensorProto.INT64, [], [vocab_size]),
        helper.make_tensor(f"{prefix}_shape_1x1", TensorProto.INT64, [2], [1, 1]),
        helper.make_tensor(f"{prefix}_depth", TensorProto.INT64, [], [vocab_size]),
        helper.make_tensor(
            f"{prefix}_onehot_values", TensorProto.FLOAT, [2], [0.0, 1.0]
        ),
    ]
    return nodes, initializers, logits


def make_decoder_model(vocab_size):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["present.0.key"], to=TensorProto.INT8),
    ]
    onehot_nodes, onehot_inits, logits_name = _cache_sum_logits_subgraph(
        "d0", "present.0.key", vocab_size
    )
    nodes += onehot_nodes
    assert logits_name == "logits"

    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab_size]),
        helper.make_tensor_value_info("present.0.key", TensorProto.INT8, [1, 1]),
    ]
    graph = helper.make_graph(
        nodes, "toy_decoder", [input_ids], outputs, initializer=onehot_inits
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )


def make_decoder_with_past_model(vocab_size):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])
    past_key = helper.make_tensor_value_info(
        "past_key_values.0.key", TensorProto.INT8, [1, "seq_past"]
    )
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["new_key"], to=TensorProto.INT8),
        helper.make_node(
            "Concat", ["past_key_values.0.key", "new_key"], ["present.0.key"], axis=1
        ),
    ]
    onehot_nodes, onehot_inits, logits_name = _cache_sum_logits_subgraph(
        "dp0", "present.0.key", vocab_size
    )
    nodes += onehot_nodes
    assert logits_name == "logits"

    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab_size]),
        helper.make_tensor_value_info(
            "present.0.key", TensorProto.INT8, [1, "seq_present"]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_decoder_with_past",
        [input_ids, past_key],
        outputs,
        initializer=onehot_inits,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )


def compute_expected_ids(
    decoder_start_token_id, vocab_size, max_new_tokens, eos_token_id=None
):
    """Reference implementation of the toy model's math, independent of ONNX
    Runtime: at every step, the generated token is the running sum of every
    token in the cache so far (including itself once generated), mod
    vocab_size -- see this module's own docstring for the derivation.
    """
    cache = [decoder_start_token_id]
    generated = []
    for _ in range(max_new_tokens):
        token = sum(cache) % vocab_size
        generated.append(token)
        cache.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=7)
    parser.add_argument("--decoder-start-token-id", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = {
        "decoder_model.onnx": make_decoder_model(args.vocab_size),
        "decoder_with_past_model.onnx": make_decoder_with_past_model(args.vocab_size),
    }
    for name, model in models.items():
        checker.check_model(model)
        onnx.save(model, os.path.join(args.output_dir, name))
        print(f"wrote {name}")

    print(
        f"decoder_start_token_id={args.decoder_start_token_id} vocab_size={args.vocab_size}"
    )


if __name__ == "__main__":
    main()
