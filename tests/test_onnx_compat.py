"""Tests for ``onnxsim._onnx_compat`` -- importing against an older ``onnx``.

Downstream integrations pin old onnx versions and install onnxsim beside them
(X2Paddle 1.6.0 needs ``onnx.mapping``, removed in onnx 1.16, so its regression
harness installs ``onnx<1.16``). ``onnx.TensorProto.UINT4``/``INT4`` arrived in
onnx 1.16, so touching them while a module is being imported breaks
``import onnxsim`` outright on such an install -- long before any 4-bit code
could run. ``_onnx_compat`` holds the guarded lookups; these tests check the
fallback values are right and that no module goes back to the direct attribute.
"""

import ast
import pathlib

import onnx
import pytest

from onnxsim import _onnx_compat, model_info

# Element types onnxsim references that older supported onnx releases lack:
# UINT4/INT4 came in onnx 1.16, FLOAT4E2M1 and FLOAT8E8M0 later still.
NEW_TENSOR_DTYPES = {"UINT4", "INT4", "FLOAT4E2M1", "FLOAT8E8M0"}


def test_fallbacks_match_the_installed_onnx():
    assert _onnx_compat.UINT4 == onnx.TensorProto.UINT4
    assert _onnx_compat.INT4 == onnx.TensorProto.INT4


def test_fallbacks_are_the_spec_wire_values():
    # The fallbacks are hard-coded numbers, valid only because the ONNX spec
    # fixes them; if onnx ever renumbered, these would silently mismatch.
    assert (onnx.TensorProto.UINT4, onnx.TensorProto.INT4) == (21, 22)


def _module_level_nodes(tree):
    """Yield every node evaluated at import time: module-level statements and
    class bodies, plus the parts of a function definition (decorators,
    annotations, argument defaults) that run when the ``def`` is executed -- but
    not function bodies, which run only when called."""

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for part in [
                    *child.decorator_list,
                    *(child.args.defaults or []),
                    *[d for d in (child.args.kw_defaults or []) if d is not None],
                    *([child.returns] if child.returns else []),
                ]:
                    yield from ast.walk(part)
            else:
                yield child
                yield from walk(child)

    yield from walk(tree)


@pytest.mark.parametrize(
    "path",
    sorted(pathlib.Path(model_info.__file__).parent.glob("*.py")),
    ids=lambda p: p.name,
)
def test_new_tensor_dtypes_are_not_touched_at_import_time(path):
    if path.name == "_onnx_compat.py":
        return  # the one module allowed to look them up (it guards the lookup)
    offenders = [
        f"{path.name}:{node.lineno} {ast.unparse(node)}"
        for node in _module_level_nodes(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute) and node.attr in NEW_TENSOR_DTYPES
    ]
    assert offenders == [], (
        "these run at import and break `import onnxsim` on an older onnx; "
        "take the value from onnxsim._onnx_compat instead: " + ", ".join(offenders)
    )
