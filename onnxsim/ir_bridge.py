# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Boundary between ONNX protobuf models and :mod:`onnx_ir` models.

Modules that edit graphs in Python convert once at entry with :func:`as_ir` and
once at exit with :func:`as_proto`, so the editing code works on ``onnx_ir``
objects (values, nodes, graphs and functions) instead of protobuf messages.
The input protobuf is not modified.
"""

from __future__ import annotations

from typing import Union

import onnx
import onnx_ir as ir

__all__ = ["as_ir", "as_proto"]


def as_ir(model: Union[str, onnx.ModelProto, ir.Model]) -> ir.Model:
    """An ``onnx_ir`` model for a file path, a protobuf model or an IR model.

    An IR model is returned as is, so the caller owns any edits made to it.
    """
    if isinstance(model, ir.Model):
        return model
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return ir.from_proto(model)


def as_proto(model: ir.Model) -> onnx.ModelProto:
    """The protobuf form of an ``onnx_ir`` model."""
    return ir.to_proto(model)
