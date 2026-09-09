#!/usr/bin/env python3
"""Internal helper for nnsmith_simplify_fuzz.py: run NNSmith's model_gen CLI
with torch's legacy (TorchScript-trace) ONNX exporter forced on.

NNSmith 0.1.0 (the latest PyPI release as of writing) calls
``torch.onnx.export(...)`` without passing ``dynamo=``, so it gets whatever
torch's current default is. On torch versions where that default is the
newer ``torch.export``-based dynamo exporter, NNSmith's own internal
``debug_numeric`` sanity check (a plain Python ``any(torch.isinf(t).any()
for t in tensors)`` over the traced module's tensors, decorated
``@torch.jit.ignore`` -- a TorchScript-only annotation dynamo does not
honor) gets symbolically traced too, and torch.export's data-dependent-value
guard rejects it (``GuardOnDataDependentSymNode``). Concretely, on this
environment's torch 2.14.0, every single candidate op failed NNSmith's own
opset self-test (0 exportable ops) until this was patched; forcing
``dynamo=False`` (confirmed directly: a bare ``torch.onnx.export(dynamo=False)``
on a trivial module succeeds on the same torch install) restores the
TorchScript-trace exporter NNSmith was actually written against, and the
self-test then passes for the large majority of ops. This is a compatibility
shim for NNSmith's current release, not a documented/supported flag of
NNSmith's own -- revisit if a newer NNSmith release fixes this upstream.
"""

import functools
import sys

import torch

_orig_export = torch.onnx.export
torch.onnx.export = functools.partial(_orig_export, dynamo=False)

from nnsmith.cli.model_gen import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
