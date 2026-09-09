#!/usr/bin/env python3
"""Internal helper for nnsmith_simplify_fuzz.py: generate a model via
NNSmith's ``torch`` model type, then export it to ONNX ourselves.

NNSmith 0.1.0 (the latest PyPI release as of writing) has its own built-in
``model.type=onnx`` path, but two independent problems make it a poor fit:

1. Its ONNX export calls ``torch.onnx.export(...)`` without passing
   ``dynamo=``, so it gets whatever torch's current default is. On torch
   versions where that default is the newer ``torch.export``-based dynamo
   exporter, NNSmith's own internal ``debug_numeric`` sanity check (a plain
   Python ``any(torch.isinf(t).any() for t in tensors)`` over the traced
   module's tensors, decorated ``@torch.jit.ignore`` -- a TorchScript-only
   annotation dynamo does not honor) gets symbolically traced too, and
   torch.export's data-dependent-value guard rejects it
   (``GuardOnDataDependentSymNode``). Concretely, on this environment's
   torch 2.14.0, every single candidate op failed NNSmith's own opset
   self-test under ``model.type=onnx`` (0 exportable ops) before this was
   worked around.
2. Determining which ops are exportable this way is expensive: NNSmith
   trial-exports every candidate op/dtype combination once per host and
   caches the result (``~/.cache/nnsmith-<ver>/*.yaml``) -- measured at
   ~3.5 minutes for ``model.type=onnx`` (each trial is a full ONNX export).

``model.type=torch`` sidesteps both: its own self-test only calls the
module's ``forward()`` eagerly (no export, so no exporter-default and no
debug_numeric/dynamo interaction at all), and it measured at ~2.7 seconds
here -- about 80x faster -- for a broader resulting opset (nothing is
excluded merely because *export* of it happens to be unsupported by
whichever torch default is active). We then export the generated model to
ONNX ourselves via the legacy TorchScript-trace exporter
(``torch.onnx.export(..., dynamo=False)``), explicitly, which is the exact
thing ``model.type=onnx`` would have done internally anyway (minus the
opset-self-test cost and the dynamo pitfall) -- see ``_export_to_onnx``
below.

Known flakiness, independent of any of the above: NNSmith's own topset
self-test (``nnsmith.narrow_spec.infer_topset_from_scratch``, shared by
every model type) occasionally raises its own internal
``nnsmith.error.InternalError`` while constructing a single-op test
program for a reshape-like op (a z3-model-dependent assertion,
non-deterministic across runs since the solver isn't seeded) -- confirmed
by hand: two consecutive fresh-cache builds, one crashed this way and the
next didn't. Not caught anywhere in NNSmith's own code, so it aborts
whichever process hit it; since this only matters on a topset *cache miss*
(normally a once-per-environment cost -- see the nightly workflow's
``actions/cache`` step) and each retry is now cheap regardless, this
script does not special-case it: a hit shows up as this one model's
`gen_error` and the next model's fresh subprocess just tries the topset
build again.
"""

import pickle
import sys
from pathlib import Path

import torch
from nnsmith.cli.model_gen import main as _nnsmith_model_gen
from nnsmith.materialize.torch import TorchModelCPU

# A stable choice for the legacy TorchScript-trace exporter we force via
# dynamo=False below; confirmed working by hand on this environment's torch.
_ONNX_OPSET = 17

# model.type=torch's broader topset (see module docstring) includes complex64/
# complex128, which torch eager execution supports but the legacy ONNX
# exporter's shape/type inference does not ("RuntimeError: ScalarType
# ComplexFloat is an unexpected tensor scalar type", confirmed by hand) --
# restricting generation to these real dtypes avoids wasting a chunk of
# generated models on complex-dtype graphs guaranteed to fail at export.
_DTYPE_CHOICES = "[float16,float32,float64,int8,int16,int32,int64,uint8]"


def _export_to_onnx(save_dir: Path) -> None:
    """Load the ``model.pth``/``gir.pkl``/``oracle.pkl`` NNSmith just saved
    into `save_dir` and export it to `save_dir/model.onnx` ourselves."""
    pth_path = save_dir / (TorchModelCPU.name_prefix() + TorchModelCPU.name_suffix())
    model = TorchModelCPU.load(str(pth_path))
    net = model.torch_model
    net.eval()
    input_names = list(model.input_like.keys())
    output_names = list(model.output_like.keys())

    with open(save_dir / "oracle.pkl", "rb") as f:
        oracle = pickle.load(f)
    dummy_inputs = tuple(
        torch.from_numpy(oracle["input"][name]) for name in input_names
    )

    torch.onnx.export(
        net,
        dummy_inputs,
        str(save_dir / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        opset_version=_ONNX_OPSET,
        dynamo=False,
    )


def main() -> int:
    argv = sys.argv[1:]
    save_dir = None
    for arg in argv:
        if arg.startswith("mgen.save="):
            save_dir = Path(arg.split("=", 1)[1])
    if save_dir is None:
        raise SystemExit(
            "usage: _nnsmith_gen.py mgen.save=<dir> [more hydra overrides...]"
        )

    # model.type is always torch here -- see module docstring for why; a
    # caller-supplied model.type or mgen.dtype_choices override would
    # conflict with these.
    sys.argv = [
        sys.argv[0],
        *argv,
        "model.type=torch",
        f"mgen.dtype_choices={_DTYPE_CHOICES}",
    ]
    _nnsmith_model_gen()
    _export_to_onnx(save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
