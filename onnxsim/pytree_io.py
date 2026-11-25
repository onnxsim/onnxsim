"""Record the pytree structure of an exported model's inputs and outputs.

``torch.export`` flattens a model's inputs and outputs (``args``/``kwargs`` and
the returned containers) into a flat list of tensors, and the dynamo ONNX
exporter then writes that flat list as graph inputs and outputs. The nesting --
which input is ``kwargs["cfg"]["scale"][0]``, which output is ``out[1][0]`` --
is kept only in ``ExportedProgram.call_spec`` and is lost in the saved file.

:func:`annotate_pytree_io` writes each graph input and output's path into
``metadata_props`` under :data:`PYTREE_PATH_KEY`, so downstream tools can map
flat ONNX I/O back to the Python structure. Paths are ``/``-separated: inputs
start with ``args/<i>`` or ``kwargs/<name>``, outputs start at the returned
value (``""`` when the model returns a single tensor).

The annotation survives :func:`onnxsim.simplify`, because the IR round trip
keeps ``ValueInfo.metadata_props``.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Sequence, Tuple, Union

import onnx

#: ``metadata_props`` key written on each graph input and output.
PYTREE_PATH_KEY = "onnxsim.pytree.path"

_ARGS_INDEX = "0"
_KWARGS_INDEX = "1"


def _key_text(key: Any) -> str:
    # torch.utils._pytree key entries: MappingKey(key), SequenceKey(idx),
    # GetAttrKey(name), FlattenedIndexKey(idx).
    for attr in ("key", "idx", "name"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def _spec_leaf_paths(spec: Any) -> List[Tuple[str, ...]]:
    """Return the key path of every leaf of ``spec``, in flattened order."""
    from torch.utils import _pytree as pytree

    num_leaves = spec.num_leaves
    # Unflatten with the leaf index as each leaf's value, then flatten with
    # paths: leaf ``i`` sits at the path returned alongside value ``i``.
    probe = pytree.tree_unflatten(list(range(num_leaves)), spec)
    paths: List[Tuple[str, ...]] = [()] * num_leaves
    for path, index in pytree.tree_flatten_with_path(probe)[0]:
        paths[index] = tuple(_key_text(key) for key in path)
    return paths


def pytree_io_paths(exported_program: Any) -> Tuple[List[str], List[str]]:
    """Return ``(input_paths, output_paths)`` for ``exported_program``.

    One path per flattened leaf, in the same order torch.export and the ONNX
    exporter use for graph inputs and outputs.
    """
    call_spec = exported_program.call_spec
    input_paths: List[str] = []
    for keys in _spec_leaf_paths(call_spec.in_spec):
        # The input tree is (args_tuple, kwargs_dict).
        head = {_ARGS_INDEX: "args", _KWARGS_INDEX: "kwargs"}.get(keys[0], keys[0])
        input_paths.append("/".join((head,) + keys[1:]))
    output_paths = ["/".join(keys) for keys in _spec_leaf_paths(call_spec.out_spec)]
    return input_paths, output_paths


def _graph_io(model: Any) -> Tuple[Sequence[Any], Sequence[Any]]:
    if isinstance(model, onnx.ModelProto):
        return list(model.graph.input), list(model.graph.output)
    # onnx_ir.Model (ONNXProgram.model).
    return list(model.graph.inputs), list(model.graph.outputs)


def _set_path(value: Any, path: str) -> None:
    if isinstance(value, onnx.ValueInfoProto):
        for entry in value.metadata_props:
            if entry.key == PYTREE_PATH_KEY:
                entry.value = path
                return
        entry = value.metadata_props.add()
        entry.key = PYTREE_PATH_KEY
        entry.value = path
    else:
        value.metadata_props[PYTREE_PATH_KEY] = path


def annotate_pytree_io(
    model: Union[onnx.ModelProto, Any], exported_program: Any
) -> None:
    """Write pytree paths onto ``model``'s graph inputs and outputs, in place.

    ``model`` is either an :class:`onnx.ModelProto` or an ``onnx_ir.Model``
    (such as ``ONNXProgram.model``). Entries are matched by position against
    the exporter's own user input and output lists, so a model whose graph I/O
    count differs from the export's (for example because a non-tensor leaf has
    no graph input) raises :class:`ValueError` instead of annotating the wrong
    values.
    """
    signature = exported_program.graph_signature
    input_paths, output_paths = pytree_io_paths(exported_program)
    if len(input_paths) != len(signature.user_inputs) or len(output_paths) != len(
        signature.user_outputs
    ):
        raise ValueError(
            "pytree annotation needs one graph input per flattened input leaf "
            "and one output per flattened output leaf; the export has "
            f"{len(input_paths)} input leaves / {len(signature.user_inputs)} "
            f"user inputs and {len(output_paths)} output leaves / "
            f"{len(signature.user_outputs)} user outputs"
        )
    inputs, outputs = _graph_io(model)
    if len(inputs) != len(input_paths) or len(outputs) != len(output_paths):
        raise ValueError(
            "model graph I/O does not match the export: "
            f"{len(inputs)} inputs vs {len(input_paths)} expected, "
            f"{len(outputs)} outputs vs {len(output_paths)} expected"
        )
    for value, path in zip(inputs, input_paths):
        _set_path(value, path)
    for value, path in zip(outputs, output_paths):
        _set_path(value, path)


def export_exported_program_to_onnx(
    exported_program: Any,
    path: str,
    *,
    annotate_pytree: bool = True,
    **export_kwargs: Any,
) -> str:
    """Export an ``ExportedProgram`` to ``path`` with the dynamo ONNX exporter.

    Equivalent to ``torch.onnx.export(exported_program, f=path, dynamo=True,
    **export_kwargs)``, plus the pytree I/O annotation (see
    :func:`annotate_pytree_io`) unless ``annotate_pytree`` is false. The
    exporter runs in memory, so the annotation is written before the file is
    saved; external tensor data is written next to ``path`` as before.
    """
    import torch

    program = torch.onnx.export(exported_program, f=None, dynamo=True, **export_kwargs)
    if annotate_pytree:
        try:
            annotate_pytree_io(program.model, exported_program)
        except ValueError as err:
            # Export itself succeeded; an unmappable I/O layout only costs the
            # annotation, so warn and keep the model rather than failing it.
            warnings.warn(f"pytree I/O annotation skipped: {err}", stacklevel=2)
    program.save(path, external_data=True)
    return path
