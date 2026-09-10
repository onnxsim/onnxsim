# DataLoader + `compile_torch_training_loop` demo

A standalone example of onnxsim's own training pipeline: a real
`torch.nn.Module`, trained from an ordinary `torch.utils.data.DataLoader`,
with every step running on onnxsim's own gradient/optimizer machinery
(`onnxsim.graph_grad` + `onnxsim.qat_graph`) instead of `torch.autograd`. See
`onnxsim/torch_training.py` and `onnxsim/compile_training.py` for how that
actually works; this directory is only the "plug an ordinary PyTorch data
pipeline into it" half.

## What it actually does

- `Regression` is an ordinary two-layer MLP whose `forward` computes its own
  loss (mean squared error) -- `compile_torch_training_loop` needs the loss
  computed inside the module, since there is no separate loss-function
  export path (see its own docstring).
- `onnxsim.compile_torch_training_loop` exports that module once, via
  `torch.export`'s FX graph, into a compiled `TrainingLoop`.
- The training loop itself is exactly what it would be with any other
  PyTorch optimizer: iterate the `DataLoader`, hand each batch to the
  compiled loop. The batch tensors go in as plain `torch.Tensor`, with no
  `.numpy()` call anywhere in `train.py` -- `TrainingLoop.__call__` accepts
  them directly via the DLPack protocol (a CPU batch is bound by reference,
  not copied into a fresh buffer first; see
  `onnxsim/compile_training.py`'s own module docstring).
- The data is synthetic (a fixed random linear relationship plus noise) so
  the demo needs no external dataset; swap `synthetic_dataset()` for a real
  `Dataset` and the rest of the script is unchanged.

## Install

    pip install onnxsim[torch-training]

Pulls in `torch>=2.5` and `onnxscript` (the dynamo exporter's own
dependency).

## Usage

    python examples/torch_dataloader_training/train.py

With different hyperparameters, and writing the trained model to ONNX:

    python examples/torch_dataloader_training/train.py \
        --epochs 30 --batch-size 64 --hidden 32 --lr 5e-3 \
        --optimizer sgd_momentum --output ./trained.onnx

`--output` writes exactly what `TrainingLoop.export()` produces: the
original module's own ONNX graph, with each trained parameter's initializer
replaced by its final value -- an ordinary forward-only model, loadable by
any ONNX runtime, with no onnxsim-specific ops or domains in it.
