#!/usr/bin/env python3
"""A real, published GRU-based architecture, the companion probe to
`build_parakeet_lstm_probe.py` -- `docs/axera-audio-speech-op-coverage.md`'s
coverage table has always listed `GRU` alongside `LSTM` as "open, largest
lift" but, until now, nothing in this project ever exported a real
GRU-containing model to check it.

`torchaudio.models.WaveRNN` is DeepMind's real, published WaveRNN vocoder
("Efficient Neural Audio Synthesis") -- two `nn.GRU` layers (`self.rnn1`,
`self.rnn2` in `torchaudio/models/wavernn.py`) driving its autoregressive
sample generation, with an ordinary, non-pretrained, config-driven
constructor (`upsample_scales`, `n_rnn`, `n_freq`, etc. are all plain ints),
so it fits this project's tiny-random-init-export convention exactly the
way `build_parakeet_lstm_probe.py`'s decoder does. Requires
`torchaudio`/`torch` (export only, CPU, random init -- no pretrained
weights, no GPU).

Confirms a real, previously-only-assumed finding: unlike `LSTM` (NPU-
executable, just missing a backward rule -- see the LSTM probe), **`GRU` is
not in `scripts/axera/pulsar2_ops.py`'s `AX650_SUPPORTED_OPS` at all** --
a strictly bigger gap than LSTM's, since a GRU-containing model cannot even
*run* on this hardware at inference, before training enters the picture.

Usage::

    build_wavernn_gru_probe.py OUT_PATH

writes `OUT_PATH` (the raw WaveRNN export) and prints an op-coverage table
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

# Tiny relative to a real deployed WaveRNN (n_rnn/n_fc default to 512) --
# big enough that both GRU layers' node structure is real and inspectable,
# small enough to export/inspect in well under a second.
UPSAMPLE_SCALES = [2, 2]
HOP_LENGTH = 4  # must equal the product of UPSAMPLE_SCALES
N_CLASSES = 32
N_RES_BLOCK = 2
N_RNN = 16
N_FC = 16
KERNEL_SIZE = 5
N_FREQ = 8
N_HIDDEN = 8
N_OUTPUT = 8  # must be a multiple of 4 (WaveRNN.__init__'s n_aux = n_output // 4)
N_TIME = 10  # spectrogram frames; must be >= KERNEL_SIZE


def export_wavernn(out_path: str) -> onnx.ModelProto:
    """Writes `WaveRNN(waveform, specgram) -> class_logits` -- the full
    model (upsample network, both GRU layers, output FC stack), no
    autoregressive sample-by-sample generation loop (this probe exercises
    the teacher-forced training-time forward pass, the same shape every
    other model in this project's survey exports)."""
    import torch
    import torchaudio.models as tm

    model = tm.WaveRNN(
        upsample_scales=UPSAMPLE_SCALES,
        n_classes=N_CLASSES,
        hop_length=HOP_LENGTH,
        n_res_block=N_RES_BLOCK,
        n_rnn=N_RNN,
        n_fc=N_FC,
        kernel_size=KERNEL_SIZE,
        n_freq=N_FREQ,
        n_hidden=N_HIDDEN,
        n_output=N_OUTPUT,
    ).eval()

    n_samples = (N_TIME - KERNEL_SIZE + 1) * HOP_LENGTH
    waveform = torch.randn(1, 1, n_samples)
    specgram = torch.randn(1, 1, N_FREQ, N_TIME)
    torch.onnx.export(
        model,
        (waveform, specgram),
        out_path,
        opset_version=17,
        dynamo=False,
        input_names=["waveform", "specgram"],
        output_names=["class_logits"],
    )
    return onnx.load(out_path)


def op_coverage(model: onnx.ModelProto) -> dict[str, tuple[bool, bool]]:
    """`{op_type: (has_backward_rule, npu_supported)}` for every op type in
    `model`, cross-referencing the same two tables
    `docs/axera-audio-speech-op-coverage.md`'s survey (and
    `build_parakeet_lstm_probe.py`) already use."""
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
        else "wavernn_gru_probe.onnx"
    )
    model = export_wavernn(out_path)

    counts = Counter(n.op_type for n in model.graph.node)
    print(f"{len(model.graph.node)} nodes: {dict(counts)}")

    coverage = op_coverage(model)
    print(f"{'op':15s} {'backward rule?':15s} npu_supported?")
    for op, (has_backward, npu_ok) in coverage.items():
        print(f"{op:15s} {str(has_backward):15s} {npu_ok}")

    gru_nodes = counts.get("GRU", 0)
    gru_npu_ok = coverage.get("GRU", (False, False))[1]
    print(f"\n{gru_nodes} real, opaque GRU node(s) -- unlike LSTM, GRU is")
    print(f"NPU-supported={gru_npu_ok}: not just missing a backward rule, GRU")
    print("cannot run on this hardware at inference at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
