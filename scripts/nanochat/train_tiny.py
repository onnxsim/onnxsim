#!/usr/bin/env python3
"""Train the tiny nanochat reimplementation on a character-level corpus and
plot the loss curve.

This is the training half of the nanochat x onnxsim demo: nanochat itself
(https://github.com/karpathy/nanochat) trains a BPE tokenizer and a GPT on
web-scale data across multiple stages (base pretraining, midtraining, SFT,
RL). This script reproduces just the *shape* of the base-pretraining loop --
next-token cross-entropy on a sliding window of tokenized text -- at a scale
that trains on a laptop CPU in well under a minute, using `model.GPT`
(see model.py's docstring for the exact architectural diff against upstream)
and a plain character-level tokenizer instead of nanochat's BPE.

The corpus is a 64KB slice of the "tinyshakespeare" dataset
(data/tinyshakespeare_sample.txt, from
https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt,
public domain) -- the same char-level demo dataset used across Karpathy's own
char-rnn/minGPT/nanoGPT tutorials, bundled here so the demo trains offline.

Usage::

    python train_tiny.py                        # default tiny model, 300 steps
    python train_tiny.py --steps 800 --n-embd 128 --n-layer 4
    python train_tiny.py --checkpoint ckpt.pt    # also save weights for export

See simplify_nanochat.py to export the (optionally trained) model to ONNX and
run it through onnxsim.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

_NANOCHAT_DIR = os.path.dirname(os.path.abspath(__file__))
if _NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, _NANOCHAT_DIR)

from model import GPT, GPTConfig  # noqa: E402

_DEFAULT_CORPUS = os.path.join(_NANOCHAT_DIR, "data", "tinyshakespeare_sample.txt")


def build_dataset(text: str) -> tuple[torch.Tensor, dict[str, int]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=_DEFAULT_CORPUS)
    parser.add_argument("--block-size", type=int, default=64, help="context length")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2, help="< n-head exercises GQA")
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-path", default=os.path.join(_NANOCHAT_DIR, "loss_curve.png")
    )
    parser.add_argument(
        "--checkpoint", default=None, help="optional path to save the trained weights"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    data, stoi = build_dataset(text)
    print(f"corpus: {len(text)} chars, vocab: {len(stoi)} unique chars")

    config = GPTConfig(
        sequence_len=args.block_size,
        vocab_size=len(stoi),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )
    model = GPT(config, pad_vocab_size_to=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {config}, {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    for step in range(args.steps):
        x, y = get_batch(data, args.block_size, args.batch_size, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step:4d}/{args.steps} loss {loss.item():.4f}")

    plot_loss_curve(losses, config, args.plot_path)

    if args.checkpoint:
        torch.save(
            {"model": model.state_dict(), "config": vars(config), "stoi": stoi},
            args.checkpoint,
        )
        print(f"checkpoint saved to {args.checkpoint}")


def plot_loss_curve(losses: list[float], config: GPTConfig, plot_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display backend needed
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, linewidth=0.8, alpha=0.4, color="tab:blue", label="loss (per step)")

    window = max(1, len(losses) // 40)
    if len(losses) >= window * 2:
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        ax.plot(
            range(window - 1, len(losses)),
            smoothed,
            linewidth=2,
            color="tab:orange",
            label=f"loss ({window}-step moving avg)",
        )

    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(
        f"nanochat-tiny training loss\n"
        f"n_layer={config.n_layer} n_head={config.n_head} n_kv_head={config.n_kv_head} "
        f"n_embd={config.n_embd}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
