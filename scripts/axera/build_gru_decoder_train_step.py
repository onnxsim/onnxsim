#!/usr/bin/env python3
"""The first real-hardware test of `scripts/axera/legalize.py`'s
`unroll_gru` -- the GRU counterpart of `build_parakeet_lstm_train_step.py`,
built from `build_gru_decoder_probe.py`'s real-shaped GRU decoder (see that
module's own docstring for why it stands in for `torchaudio.models.
WaveRNN`, whose own export hits a separate, unrelated `torch.onnx`
exporter bug).

`GRU`'s stakes are higher than `LSTM`'s: `GRU` is not in
`AX650_SUPPORTED_OPS` at all, so `unroll_gru` is not just adding a
backward rule -- it is the only way a `GRU`-containing model runs on this
hardware at all, training or not.

Trainable scope: the first GRU layer's 6 per-gate weight tensors
(`unroll_gru` splits each layer's packed `W`/`R` into one initializer per
gate -- `z, r, h` -- 3 gates x 2 matrices), the real, post-unroll
trainable-weight names -- not the pre-unroll packed `W`/`R`, which no
longer exist as live tensors once `unroll_gru` has run. Kept to one layer
to start real, mirroring `build_parakeet_lstm_train_step.py`'s own
"start small" convention.

Usage::

    build_gru_decoder_train_step.py OUT_DIR

writes `OUT_DIR/gru_decoder_step.onnx` plus `OUT_DIR/params.txt` (the
trainable tensor names, one per line) and `OUT_DIR/state_map.txt`
(`name\\tnext_name` per line) -- pass `gru_decoder_step.onnx` to
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

import build_gru_decoder_probe as gru_probe  # noqa: E402
import build_resident_train_step as brts  # noqa: E402
from build_whisper_train_step import add_loss_3d  # noqa: E402

#: `/gru/GRU` is the real, torch-traced name of the decoder's *first* GRU
#: layer -- `unroll_gru` stems every per-gate name it creates from a
#: node's own `.name` (falling back to `.output[0]` only when empty), and
#: torch.onnx's exporter always names this node from the module path
#: (`self.gru`, layer 0). Confirmed directly against a real export:
#: `build_gru_decoder_probe.export_decoder`'s own output has exactly
#: `/gru/GRU` and `/gru/GRU_1` (layer 1) as its two GRU node names.
_LAYER0_STEM = "/gru/GRU"
_GATES = 3  # z, r, h


def layer0_param_names() -> list[str]:
    """The 6 real, post-`unroll_gru` trainable tensor names for the
    decoder's first GRU layer: `<stem>_w0..w2` (input-to-gate) and
    `<stem>_r0..r2` (hidden-to-gate) -- `legalize._gate_weight`'s own
    naming convention."""
    return [f"{_LAYER0_STEM}_w{i}" for i in range(_GATES)] + [
        f"{_LAYER0_STEM}_r{i}" for i in range(_GATES)
    ]


#: The real decoder's embedding-lookup output -- `nn.Embedding(input_ids)`,
#: confirmed directly against a real export as `/embedding/Gather`'s own
#: single output, the same name `ParakeetRNNTDecoder`'s own export uses.
_EMBEDDED_INPUT = "/embedding/Gather_output_0"


def build_forward() -> onnx.ModelProto:
    """The real decoder from just past its embedding lookup onward,
    `unroll_gru`-legalized -- no `GRU` op left in the graph at all.

    Cut at the embedding's own output (`onnx.utils.extract_model`), the
    same reason `build_parakeet_lstm_train_step.build_forward` does:
    `input_ids`'s real `[batch, seq]` shape is a genuinely *batched*
    `Gather`, a case `graph_grad._grad_gather`'s own docstring declines on
    purpose, and nothing here needs the embedding table's own gradient.
    """
    raw_path = "/tmp/gru_decoder_train_step_raw.onnx"
    gru_probe.export_decoder(raw_path)
    cut_path = "/tmp/gru_decoder_train_step_cut.onnx"
    onnx.utils.extract_model(raw_path, cut_path, [_EMBEDDED_INPUT], ["decoder_output"])
    model = onnx.load(cut_path)

    n = legalize.unroll_gru(model)
    if n != 2:
        raise RuntimeError(f"expected both GRU layers to unroll, got {n}")
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

    out_path = os.path.join(out_dir, "gru_decoder_step.onnx")
    onnx.save(step_model, out_path)
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write("\n".join(params))
    with open(os.path.join(out_dir, "state_map.txt"), "w") as f:
        f.write("\n".join(f"{k}\t{v}" for k, v in state.items()))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
