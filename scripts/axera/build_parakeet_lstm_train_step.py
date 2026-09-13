#!/usr/bin/env python3
"""The first real-hardware test of `scripts/axera/legalize.py`'s
`unroll_lstm` -- every earlier check (`build_parakeet_lstm_probe.py`,
`tests/test_axera_legalize.py`) was host-only (onnxruntime, finite
differences). This builds an actual resident training step from the real
NVIDIA Parakeet decoder with `unroll_lstm` applied, per
`docs/axera-audio-speech-op-coverage.md`'s "Both closed" section's own
"a real hardware run... is the natural next step" note.

Trainable scope: the first LSTM layer's 8 per-gate weight tensors
(`unroll_lstm` splits each layer's packed `W`/`R` into one initializer per
gate -- `i, o, f, c` -- 4 gates x 2 matrices), the real, post-unroll
trainable-weight names -- not the pre-unroll packed `W`/`R`, which no
longer exist as live tensors once `unroll_lstm` has run (see that doc
section's "what actually gets trained changes" note). Kept to one layer to
start real, per this project's own "start small" convention for every
first-compile check.

Usage::

    build_parakeet_lstm_train_step.py OUT_DIR

writes `OUT_DIR/parakeet_lstm_step.onnx` plus `OUT_DIR/params.txt` (the
trainable tensor names, one per line) and `OUT_DIR/state_map.txt`
(`name\\tnext_name` per line) -- pass `parakeet_lstm_step.onnx` to
`scripts/axera/make_training_calib.py` and then `pulsar2_docker.build()` to
compile.
"""

from __future__ import annotations

import os
import sys

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim, fresh  # noqa: E402

ensure_repo_onnxsim()

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules sharing one `sys.modules["legalize"]` entry
# -- a plain `import legalize` risks silently getting axelera's copy if
# something upstream already claimed that bare name. `fresh` reloads
# directly from this file's own directory regardless of what's cached.
legalize = fresh("legalize", HERE)

import build_parakeet_lstm_probe as lstm_probe  # noqa: E402
import build_resident_train_step as brts  # noqa: E402
from build_whisper_train_step import add_loss_3d  # noqa: E402

#: `/lstm/LSTM` is the real, torch-traced name of the decoder's *first*
#: LSTM layer -- `unroll_lstm` stems every per-gate name it creates from a
#: node's own `.name` (falling back to `.output[0]` only when empty), and
#: torch.onnx's exporter always names this node from the module path
#: (`self.lstm`, layer 0). Confirmed directly against a real export rather
#: than assumed: `build_parakeet_lstm_probe.export_decoder`'s own real
#: output has exactly `/lstm/LSTM` and `/lstm/LSTM_1` (layer 1) as its two
#: LSTM node names.
_LAYER0_STEM = "/lstm/LSTM"
_GATES = 4  # i, o, f, c


def layer0_param_names() -> list[str]:
    """The 8 real, post-`unroll_lstm` trainable tensor names for the
    decoder's first LSTM layer: `<stem>_w0..w3` (input-to-gate) and
    `<stem>_r0..r3` (hidden-to-gate) -- `legalize._gate_weight`'s own
    naming convention."""
    return [f"{_LAYER0_STEM}_w{i}" for i in range(_GATES)] + [
        f"{_LAYER0_STEM}_r{i}" for i in range(_GATES)
    ]


#: The real decoder's embedding-lookup output -- `nn.Embedding(input_ids)`,
#: confirmed directly against a real export as `/embedding/Gather`'s own
#: single output.
_EMBEDDED_INPUT = "/embedding/Gather_output_0"


def build_forward() -> onnx.ModelProto:
    """The real decoder from just past its embedding lookup onward,
    `unroll_lstm`-legalized -- no `LSTM` op left in the graph at all, per
    `docs/axera-audio-speech-op-coverage.md`'s "Both closed" section.

    Cut at the embedding's own output (`onnx.utils.extract_model`) rather
    than kept whole: `input_ids`'s real shape is rank-2 (`[batch, seq]`,
    a real *batched* lookup), and `onnxsim.graph_grad._grad_gather`'s own
    docstring declines exactly this case on purpose ("a wrong reshape
    there would not fail loudly, it would silently mix gradients across
    batch elements") -- this task's own targets are the LSTM's per-gate
    weights, not the embedding table, so nothing here actually needs that
    gradient; feeding the embedded float vector directly sidesteps the
    question rather than risking `graph_grad`'s general correctness to
    answer it. A real follow-on if a build ever needs the embedding table
    itself trainable.
    """
    raw_path = "/tmp/parakeet_lstm_train_step_raw.onnx"
    lstm_probe.export_decoder(raw_path)
    cut_path = "/tmp/parakeet_lstm_train_step_cut.onnx"
    onnx.utils.extract_model(raw_path, cut_path, [_EMBEDDED_INPUT], ["decoder_output"])
    model = onnx.load(cut_path)

    n = legalize.unroll_lstm(model)
    if n != 2:
        raise RuntimeError(f"expected both LSTM layers to unroll, got {n}")
    onnx.checker.check_model(model)
    return model


def main(argv=None) -> int:
    out_dir = (argv or sys.argv[1:])[0]
    os.makedirs(out_dir, exist_ok=True)

    forward = build_forward()
    with_loss = add_loss_3d(forward, "decoder_output")

    params = layer0_param_names()
    initializers = {t.name for t in with_loss.graph.initializer}
    missing = [p for p in params if p not in initializers]
    if missing:
        raise RuntimeError(f"expected trainable tensors not found: {missing}")

    step_model, state = brts.build_resident_step(with_loss, params, loss_output="loss")
    onnx.checker.check_model(step_model)
    print(
        f"step graph: {len(step_model.graph.node)} nodes, "
        f"{len(step_model.graph.initializer)} initializers, "
        f"{len(params)} trainable tensors"
    )

    out_path = os.path.join(out_dir, "parakeet_lstm_step.onnx")
    onnx.save(step_model, out_path)
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write("\n".join(params))
    with open(os.path.join(out_dir, "state_map.txt"), "w") as f:
        f.write("\n".join(f"{k}\t{v}" for k, v in state.items()))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
