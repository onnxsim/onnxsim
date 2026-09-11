#!/usr/bin/env python3
"""On-device fine-tuning for the AX650N: loss scaling, and the loop around it.

A compiled training step returns its gradient as a quantised tensor, at a
range fixed when the model was built. That is fine at the start of training
and fatal at the end: as the loss falls the true gradient shrinks, drops below
half a quantisation step, and rounds to **zero**. Measured on real hardware,
with the gradient's cosine against onnxruntime alongside:

===========================  ===============  ==================
gradient tensor              gradient dies    best SNR reached
===========================  ===============  ==================
U8                           step ~1,000      30.88 dB
U16                          step ~5,000      34.02 dB
===========================  ===============  ==================

Widening to U16 buys a factor of five in steps and 3 dB, and does not remove
the mechanism. **Loss scaling does.** Multiply the upstream gradient seed by
`s`, divide the returned gradient by `s`, and the tensor crossing the
quantiser stays in the range the build calibrated for however small the true
gradient becomes.

**It does not work yet, and the reason is structural.** The seed has to reach
the graph as a *tensor*, and a tensor on this hardware is quantised to a
fixed range with linear levels. Calibrated over `[1, 2**21]`, a U8 seed has
255 evenly spaced levels, so a seed of 1.0 rounds to **zero** and multiplies
the whole backward pass by nothing; calibrated narrowly around 1.0 it pins to
a constant and scaling does nothing at all. Both were measured: a probe
sweeping the seed from 1 to 2**24 returned bit-identical gradients at every
value. A multiplicative scale meant to span decades cannot live in a linear
fixed-point tensor.

The way out is for the seed and its consumer to stay in float --
`layer_configs` accepts `data_type: "FP32"` for elementwise ops, and the seed
feeds a `Mul` -- which is untested here. Until then `LossScaler` **detects
that it is having no effect and gets out of the way** (`ineffective`), which
is what makes it safe to leave on by default.

**The half that is easy to forget, for when the seed does work.** A first
attempt only ever grew the scale. On fixed-point hardware, scaling up does not
trade underflow for a wider exponent the way fp16 does; it trades underflow
for **clipping**, and a grow-only controller reads a stalled run as "scale
further". So `LossScaler` also halves and **discards the step** on saturation,
as `torch.amp.GradScaler` does. Note the detector is a proxy: fixed-point
clipping happens to intermediates that never reach an output, so a graph that
wants reliable back-off must expose `ReduceMax(|t|)` on those tensors as
extra outputs.
"""

from __future__ import annotations

import numpy as np

#: Grow the scale when at least this fraction of the gradient has rounded to
#: zero. Some zeros are ordinary -- a ReLU's gradient is half zeros by
#: construction -- so this is deliberately not near-zero.
DEFAULT_UNDERFLOW = 0.10

#: Back off when at least this fraction of entries sit on the same extreme.
#: One entry is always the maximum; a whole percent of them being *equal* to
#: it is quantiser clipping, not a coincidence.
DEFAULT_SATURATION = 0.01


def zero_fraction(grad):
    grad = np.asarray(grad)
    return float((grad == 0).mean()) if grad.size else 0.0


def saturated_fraction(grad):
    """How much of `grad` is pinned to its own extreme magnitude."""
    grad = np.asarray(grad)
    if grad.size == 0:
        return 0.0
    peak = float(np.abs(grad).max())
    if peak == 0.0:
        return 0.0
    return float((np.abs(grad) >= peak * (1.0 - 1e-6)).mean())


class LossScaler:
    """Keeps a quantised gradient inside the range its build calibrated for.

    Feed it each step's returned gradient; it hands back the gradient to
    apply, or `None` when the step must be discarded because the tensor
    saturated. `scale` is what to multiply the graph's gradient seed by.

    On by default in `train`, because without it a run silently stops
    learning while continuing to report a loss.
    """

    def __init__(
        self,
        init_scale=1.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=20,
        max_scale=2.0**24,
        min_scale=1.0,
        underflow=DEFAULT_UNDERFLOW,
        saturation=DEFAULT_SATURATION,
        stand_down_after=32,
    ):
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.underflow = underflow
        self.saturation = saturation
        self.good_steps = 0
        self.grew = 0
        self.backed_off = 0
        self.skipped = 0
        self.ineffective = False
        self.stand_down_after = stand_down_after
        self._seen = 0
        self._first_zeros = None
        self._zero_at_scale = None
        self._grew_since_check = 0

    def unscale(self, grad):
        """The true gradient, or `None` if this step saturated and must go.

        `grad` is what the card returned, still multiplied by `self.scale`.
        Once the scale has grown several times without the gradient's zero
        fraction moving at all, the seed is not reaching the graph -- a
        quantised seed pins to a constant -- and the scaler stands down rather
        than dividing by a factor nothing applied.
        """
        grad = np.asarray(grad, dtype=np.float32)
        if self.ineffective:
            return grad
        self._seen += 1
        if self._first_zeros is None:
            self._first_zeros = zero_fraction(grad)
        elif (
            self._seen >= self.stand_down_after
            and self.scale != self.min_scale
            and abs(zero_fraction(grad) - self._first_zeros) < 1e-6
        ):
            # the scale has moved and the gradient has not: whatever the seed
            # is doing, it is not reaching the graph. Dividing by a factor
            # nothing applied would corrupt every update, so stop.
            self.ineffective = True
            self.scale = self.min_scale
            return grad
        if saturated_fraction(grad) >= self.saturation and self.scale > self.min_scale:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self.good_steps = 0
            self.backed_off += 1
            self.skipped += 1
            return None
        out = grad / self.scale
        self.good_steps += 1
        if (
            zero_fraction(grad) >= self.underflow
            and self.good_steps >= self.growth_interval
            and self.scale < self.max_scale
        ):
            self.scale = min(self.scale * self.growth_factor, self.max_scale)
            self.good_steps = 0
            self.grew += 1
        return out

    def stats(self):
        return {
            "scale": self.scale,
            "grew": self.grew,
            "backed_off": self.backed_off,
            "skipped": self.skipped,
            "ineffective": self.ineffective,
        }


def train(
    runner,
    feeds,
    weights,
    *,
    steps,
    lr,
    grad_index=0,
    seed_name="gseed",
    scaler=True,
    on_step=None,
):
    """A fine-tuning loop over a compiled training step.

    `runner` runs one step; `feeds` builds the input frame from the current
    weights and the scaler's scale. Loss scaling is on unless `scaler=False`,
    which exists to reproduce the failure rather than to be used.

    Returns the trained weights and the scaler.
    """
    scaler = LossScaler() if scaler is True else scaler
    w = np.array(weights, dtype=np.float32, copy=True)
    for step in range(steps):
        scale = scaler.scale if scaler else 1.0
        outs = runner.run(feeds(w, scale))
        grad = np.frombuffer(outs[grad_index], np.float32).reshape(w.shape)
        grad = scaler.unscale(grad) if scaler else grad
        if grad is None:  # saturated: this step is discarded
            if on_step:
                on_step(step, w, None, scaler)
            continue
        w = (w - lr * grad).astype(np.float32)
        if on_step:
            on_step(step, w, grad, scaler)
    return w, scaler
