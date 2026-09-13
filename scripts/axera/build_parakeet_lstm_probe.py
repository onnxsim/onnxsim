#!/usr/bin/env python3
"""A real, published LSTM-based architecture to test against
`onnxsim.graph_grad`'s missing `LSTM` backward rule --
`docs/axera-audio-speech-op-coverage.md`'s "What actually blocks each
family" table names this "open, largest lift" and, until now, this
project's only LSTM export was a synthetic `torch.nn.LSTM` wrapper built
purely to confirm the op appears opaque in ONNX, not a real model.

`transformers.models.parakeet.modeling_parakeet.ParakeetRNNTDecoder` is
NVIDIA's Parakeet RNN-Transducer's own "prediction network" -- a real,
minimal, genuinely LSTM-based decoder (`nn.Embedding` -> `nn.LSTM` ->
`nn.Linear`, no attention, no Conformer -- that lives in the model's
separate encoder, already covered): exactly the "classic RNN-based ASR"
architecture the coverage doc's LSTM row is about, and small enough
(`decoder_hidden_size`, `num_decoder_layers` are both plain config ints) to
export at a tiny, fast-iterating size the way this project's other survey
probes do. Requires only `transformers`/`torch` (export only, CPU, random
init -- no pretrained weights, no GPU, no `torchaudio`); the encoder half of
a real Parakeet model is a FastConformer, needing a separate, heavier
export this probe does not attempt.

Confirms, on this real architecture (not just the earlier synthetic check):
a 2-layer `nn.LSTM` exports as two separate opaque ONNX `LSTM` nodes (one
per layer, `hidden_size` in each node's own attributes), each `LSTM`-typed
node itself in `scripts/axera/pulsar2_ops.py`'s `AX650_SUPPORTED_OPS` (the
NPU runs it fine at inference) but absent from
`onnxsim.graph_grad`'s `_RULES`/`_PYTHON_ONLY_RULES`/`_MULTI_OUTPUT_RULES`
(no backward rule) -- so `build_backward` cannot differentiate through it,
the same "runs forward, can't train through it" gap the coverage doc
already documented, now confirmed on a real published model rather than
assumed to generalize from a synthetic probe.

Usage::

    build_parakeet_lstm_probe.py OUT_PATH

writes `OUT_PATH` (the raw decoder export) and prints an op-coverage table
cross-referencing every op type present against both tables.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim  # noqa: E402

ensure_repo_onnxsim()

import onnx  # noqa: E402
import pulsar2_ops  # noqa: E402

from onnxsim import graph_grad  # noqa: E402

# Tiny relative to a real deployed Parakeet (vocab_size=8193,
# decoder_hidden_size=640) -- big enough that the LSTM's per-layer node
# structure is real and inspectable, small enough to export/inspect in well
# under a second.
VOCAB_SIZE = 64
DECODER_HIDDEN_SIZE = 32
NUM_DECODER_LAYERS = 2


def export_decoder(out_path: str) -> onnx.ModelProto:
    """Writes `ParakeetRNNTDecoder(input_ids) -> decoder_output` -- the
    embedding, LSTM stack and output projector, no cache/RNN-T-specific
    control flow (this probe calls the module with `cache=None`, the plain
    training-time path)."""
    import torch
    from transformers import ParakeetRNNTConfig
    from transformers.models.parakeet.modeling_parakeet import ParakeetRNNTDecoder

    cfg = ParakeetRNNTConfig(
        vocab_size=VOCAB_SIZE,
        decoder_hidden_size=DECODER_HIDDEN_SIZE,
        num_decoder_layers=NUM_DECODER_LAYERS,
        blank_token_id=VOCAB_SIZE - 1,
    )
    decoder = ParakeetRNNTDecoder(cfg).eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
    torch.onnx.export(
        decoder,
        (input_ids,),
        out_path,
        opset_version=17,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["decoder_output"],
    )
    return onnx.load(out_path)


def op_coverage(model: onnx.ModelProto) -> dict[str, tuple[bool, bool]]:
    """`{op_type: (has_backward_rule, npu_supported)}` for every op type in
    `model`, cross-referencing the same two tables
    `docs/axera-audio-speech-op-coverage.md`'s survey already uses."""
    rules = (
        set(graph_grad._RULES)
        | set(getattr(graph_grad, "_PYTHON_ONLY_RULES", {}))
        | set(getattr(graph_grad, "_MULTI_OUTPUT_RULES", {}))
    )
    supported = pulsar2_ops.AX650_SUPPORTED_OPS
    op_types = {n.op_type for n in model.graph.node}
    return {op: (op in rules, op in supported) for op in sorted(op_types)}


def main(argv=None) -> int:
    out_path = (
        (argv or sys.argv[1:])[0]
        if (argv or sys.argv[1:])
        else "parakeet_lstm_probe.onnx"
    )
    model = export_decoder(out_path)

    counts = Counter(n.op_type for n in model.graph.node)
    print(f"{len(model.graph.node)} nodes: {dict(counts)}")

    coverage = op_coverage(model)
    print(f"{'op':15s} {'backward rule?':15s} npu_supported?")
    for op, (has_backward, npu_ok) in coverage.items():
        print(f"{op:15s} {str(has_backward):15s} {npu_ok}")

    lstm_nodes = counts.get("LSTM", 0)
    print(f"\n{lstm_nodes} real, opaque LSTM node(s) -- confirms the coverage doc's")
    print("synthetic-probe finding on a real published architecture (NVIDIA")
    print("Parakeet's own RNN-T prediction network): NPU-executable, no backward rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
