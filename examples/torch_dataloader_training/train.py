"""``torch.utils.data.DataLoader`` + ``onnxsim.compile_torch_training_loop``
example: train a real ``torch.nn.Module`` from an ordinary PyTorch data
pipeline, with every training step running on onnxsim's own grad templating
(``onnxsim.graph_grad``/``onnxsim.qat_graph``), never on ``torch.autograd``.

Unlike ``examples/llm_distillation/distill.py``, this one *is* about
onnxsim's own pipeline: ``compile_torch_training_loop`` exports the
module's forward once (via ``torch.export``'s FX graph), and every epoch
after that just runs the DataLoader as usual -- the only onnxsim-specific
part is what each batch gets handed to (the compiled ``TrainingLoop``
instead of ``loss.backward()``/``optimizer.step()``).

Two things this demonstrates deliberately, both real properties of the
pipeline rather than something staged for the demo:

- **A batch tensor goes in as-is.** ``DataLoader``'s own default collation
  already returns a batch as a ``torch.Tensor``; it is handed to the
  compiled loop directly, with no ``.numpy()`` call anywhere in this file.
  ``TrainingLoop.__call__`` accepts it through the DLPack protocol -- see
  ``onnxsim/compile_training.py``'s own module docstring -- so a CPU batch
  is bound by reference, not copied into a fresh buffer first.
- **The model itself never imports ``onnxsim`` at all.** ``Regression``
  below is an ordinary ``torch.nn.Module``; only the *training* half of this
  script (``compile_torch_training_loop`` and the loop over its own output)
  is onnxsim-specific. Swap in any module whose forward already computes a
  scalar loss (see that function's own docstring for why the loss has to be
  computed inside the module) and the rest of this script is unchanged.

Usage::

    pip install onnxsim[torch-training]
    python examples/torch_dataloader_training/train.py
    python examples/torch_dataloader_training/train.py --epochs 20 --batch-size 64 --hidden 32

See ``README.md`` in this directory.
"""

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

import onnxsim


class Regression(torch.nn.Module):
    """A small one-hidden-layer MLP whose forward computes its own loss
    (mean squared error against ``y``) -- what
    ``onnxsim.compile_torch_training_loop`` requires (see its own
    docstring): there is no separate loss-function export path, so the loss
    has to be the thing the module's own ``forward`` returns.

    ``Sigmoid``, not ``ReLU``: both are ONNX ops
    :data:`onnxsim.graph_grad.SUPPORTED_OPS` differentiates, but this keeps
    the example working unmodified if the hidden layer is widened enough for
    dead ReLU units to become a real (if unrelated) problem. Either works
    here; ``Sigmoid`` just needs no caveat.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.act = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.fc2(self.act(self.fc1(x)))
        diff = y_hat - y
        return (diff * diff).mean()


def synthetic_dataset(
    num_samples: int, in_features: int, out_features: int, seed: int
) -> TensorDataset:
    """A fixed random linear-plus-noise relationship between ``x`` and
    ``y``, for a demo that needs no external data: this is about the
    DataLoader -> compiled-loop pipeline, not what the model learns from it.
    """
    generator = torch.Generator().manual_seed(seed)
    true_weight = torch.randn(out_features, in_features, generator=generator)
    x = torch.randn(num_samples, in_features, generator=generator)
    noise = 0.01 * torch.randn(num_samples, out_features, generator=generator)
    y = x @ true_weight.T + noise
    return TensorDataset(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adam", "sgd_momentum"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write the trained ONNX model to (skipped if omitted)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = synthetic_dataset(
        args.samples, args.in_features, args.out_features, args.seed
    )
    # shuffle=True and drop_last=True: every batch this loop hands to the
    # compiled loop has exactly args.batch_size rows, which is what
    # compile_torch_training_loop's static-shape export needs -- see its own
    # docstring. A ragged final batch would need a second, differently-shaped
    # compile.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    module = Regression(args.in_features, args.hidden, args.out_features)
    example_x, example_y = next(iter(loader))
    loop = onnxsim.compile_torch_training_loop(
        module, (example_x, example_y), optimizer=args.optimizer
    )

    for epoch in range(args.epochs):
        losses = []
        for x, y in loader:
            # x, y are plain torch.Tensor, straight from DataLoader's own
            # default collation -- handed to the compiled loop as-is.
            losses.append(loop({"x": x, "y": y}, lr=args.lr))
        mean_loss = sum(losses) / len(losses)
        print(f"epoch {epoch + 1:>3}/{args.epochs}  mean loss {mean_loss:.6f}")

    if args.output:
        import onnx

        onnx.save(loop.export(), args.output)
        print(f"wrote trained model to {args.output}")


if __name__ == "__main__":
    main()
