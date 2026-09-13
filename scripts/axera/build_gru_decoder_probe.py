#!/usr/bin/env python3
"""A real-shaped GRU decoder to take `unroll_gru` to real AX650N hardware --
the GRU counterpart of `build_parakeet_lstm_probe.py`/`build_parakeet_lstm_
train_step.py`'s real LSTM hardware test.

`torchaudio.models.WaveRNN` (`build_wavernn_gru_probe.py`, the real
published GRU architecture the coverage doc's host-only GRU check used) has
a separate, unrelated `torch.onnx` exporter bug -- an `Unsqueeze` axis
miscomputation inside its own `UpsampleNetwork`/`Stretch2d` export path,
reproduced across every opset (13-18) and config size tried on this torch/
torchaudio version pairing, so the raw export never even executes via
onnxruntime, let alone reaches Pulsar2. Not a quick fix (verified directly,
not assumed): it is in `Stretch2d`'s own `repeat_interleave` decomposition,
not anything `unroll_gru`/this project's own code touches.

This module is the fallback the coverage doc's own text names for exactly
this situation: `nn.Embedding -> nn.GRU -> nn.Linear`, the same
"prediction network" shape `ParakeetRNNTDecoder` uses for its real LSTM,
with `nn.LSTM` swapped for `nn.GRU` -- a standard, real RNN-T/seq2seq
decoder shape in the ASR/TTS literature (not a contrived synthetic probe),
just not tied to one specific already-broken published export. Requires
only `torch` (export only, CPU, random init -- no pretrained weights, no
GPU, no `torchaudio`).

Usage::

    build_gru_decoder_probe.py OUT_PATH

writes `OUT_PATH` (the raw decoder export) and prints an op-coverage table
cross-referencing every op type present against both tables (mirroring
`build_parakeet_lstm_probe.py`'s own).
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

# Same scale as build_parakeet_lstm_probe.py's own LSTM decoder, for a
# direct, apples-to-apples comparison between the two.
VOCAB_SIZE = 64
HIDDEN_SIZE = 32
NUM_LAYERS = 2


class GruDecoder:
    """Built lazily inside `export_decoder` (needs `torch` at import time,
    the same deferred-import convention every other torch-dependent axera
    build script here follows)."""


def export_decoder(out_path: str) -> onnx.ModelProto:
    """Writes `decoder(input_ids) -> decoder_output`: `nn.Embedding` ->
    `num_layers`-deep `nn.GRU` -> `nn.Linear` projector, `input_ids` a real
    rank-2 `[batch, seq]` token-id tensor -- the same shape and calling
    convention `build_parakeet_lstm_probe.export_decoder` uses."""
    import torch
    from torch import nn

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
            self.gru = nn.GRU(
                input_size=HIDDEN_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                batch_first=True,
            )
            self.decoder_projector = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        def forward(self, input_ids):
            embeddings = self.embedding(input_ids)
            gru_output, _hidden = self.gru(embeddings)
            return self.decoder_projector(gru_output)

    decoder = Decoder().eval()
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
    """`{op_type: (has_backward_rule, npu_supported)}` -- identical to
    `build_parakeet_lstm_probe.op_coverage`, duplicated rather than
    imported since each probe is meant to stand alone (this project's
    established convention for these small, single-purpose scripts)."""
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
        else "gru_decoder_probe.onnx"
    )
    model = export_decoder(out_path)

    counts = Counter(n.op_type for n in model.graph.node)
    print(f"{len(model.graph.node)} nodes: {dict(counts)}")

    coverage = op_coverage(model)
    print(f"{'op':15s} {'backward rule?':15s} npu_supported?")
    for op, (has_backward, npu_ok) in coverage.items():
        print(f"{op:15s} {str(has_backward):15s} {npu_ok}")

    gru_nodes = counts.get("GRU", 0)
    print(f"\n{gru_nodes} real, opaque GRU node(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
