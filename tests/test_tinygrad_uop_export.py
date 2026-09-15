"""Tests for ``onnxsim.tinygrad_uop_export`` -- checks that exporting a real
tinygrad ``UOp`` graph (the same per-kernel ``Ops.SINK``-rooted AST
``onnxsim.webgpu_tinygrad_codegen`` itself renders to WGSL) to ONNX and
back reconstructs a graph that behaves identically, not just "looks similar
structurally": the real bar is re-rendering the *reconstructed* graph
through the exact same ``to_program``/``WGSLRenderer`` pipeline and getting
byte-identical WGSL out, for both export shapes (the flat ``ModelProto``
and the local-function ``FunctionProto``).
"""

import pytest

pytest.importorskip("tinygrad")

import onnx  # noqa: E402

from onnxsim.tinygrad_uop_export import (  # noqa: E402
    DOMAIN,
    onnx_function_to_uop,
    onnx_model_to_uop,
    uop_to_onnx_function,
    uop_to_onnx_model,
)


def _conv_kernel_ast():
    """A real per-kernel AST -- the same one
    ``onnxsim.webgpu_tinygrad_codegen._lower_tensor_program`` itself passes
    to ``to_program``/``WGSLRenderer`` -- from a small Conv2D+Relu (fused by
    tinygrad's own scheduler into one kernel), tagged for the ``WEBGPU``
    device (never actually opened; see that module's own docstring for why
    no real GPU is needed just to schedule/render).
    """
    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops

    x = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="WEBGPU")
    w = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device="WEBGPU")
    y = x.conv2d(w).relu()
    linear = y.schedule_linear()
    kernel_calls = [
        u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK
    ]
    assert kernel_calls, "expected at least one real compute kernel to be scheduled"
    return kernel_calls[0].src[0]


def _render(ast) -> str:
    from tinygrad.codegen import to_program
    from tinygrad.helpers import Target
    from tinygrad.renderer.wgsl import WGSLRenderer
    from tinygrad.uop.ops import Ops

    prg = to_program(ast, WGSLRenderer(Target()))
    return next(s for s in prg.src if s.op is Ops.SOURCE).arg


def test_exports_one_node_per_uop_in_toposort_order():
    ast = _conv_kernel_ast()
    order = list(ast.toposort())

    model = uop_to_onnx_model(ast)
    onnx.checker.check_model(model)

    assert len(model.graph.node) == len(order)

    index = {u: i for i, u in enumerate(order)}
    by_output_name = {node.output[0]: node for node in model.graph.node}
    for u in order:
        node = by_output_name[f"u{index[u]}"]
        assert node.domain == DOMAIN
        assert node.op_type == u.op.name
        assert list(node.input) == [f"u{index[s]}" for s in u.src]
        attrs = {a.name: a for a in node.attribute}
        assert attrs["dtype"].s.decode() == str(u.dtype)


def test_declares_the_custom_domain_opset():
    model = uop_to_onnx_model(_conv_kernel_ast())
    domains = {opset.domain: opset.version for opset in model.opset_import}
    assert domains[DOMAIN] == 1


def test_model_round_trip_renders_byte_identical_wgsl():
    """The real bar for "faithful": not just structurally similar after
    import, but everything ``to_program``/``WGSLRenderer`` actually reads
    off the graph survives the round trip.
    """
    ast = _conv_kernel_ast()
    original_wgsl = _render(ast)

    model = uop_to_onnx_model(ast)
    onnx.checker.check_model(model)
    reconstructed = onnx_model_to_uop(model)

    assert _render(reconstructed) == original_wgsl


def test_function_round_trip_renders_byte_identical_wgsl():
    """Same bar as the flat-model round trip, but through the local-function
    export shape: every leaf (``PARAM``) UOp becomes one of the function's
    own formal parameters (named by its own argument slot -- a kernel's
    ``PARAM`` nodes don't carry any real-world tensor identity of their own,
    only an ``OnnxRunner``-external caller with the surrounding schedule
    context would know that; see the module's own docstring), so the
    function body holds every *non-leaf* UOp.
    """
    from tinygrad.uop.ops import Ops

    ast = _conv_kernel_ast()
    original_wgsl = _render(ast)

    order = list(ast.toposort())
    params = [u for u in order if u.op is Ops.PARAM]
    assert params, "expected at least one PARAM leaf in a real Conv2D+Relu kernel"
    leaf_names = {p: f"param{p.arg.slot}" for p in params}

    function = uop_to_onnx_function(ast, "kernel_fn", leaf_names)
    assert function.domain == DOMAIN
    assert function.name == "kernel_fn"
    assert sorted(function.input) == sorted(leaf_names.values())
    # every leaf must be a pure formal parameter -- no node in the body may
    # also define one of those names (see _uop_nodes's own docstring for why).
    body_outputs = {name for node in function.node for name in node.output}
    assert body_outputs.isdisjoint(leaf_names.values())

    # a function is only meaningful embedded in a model that calls it.
    graph_node = onnx.helper.make_node(
        op_type="kernel_fn", inputs=list(function.input), outputs=["y"], domain=DOMAIN
    )
    graph = onnx.helper.make_graph(
        [graph_node],
        "g",
        inputs=[
            onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [])
            for n in function.input
        ],
        outputs=[onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [])],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid(DOMAIN, 1),
        ],
        functions=[function],
    )
    model.ir_version = onnx.IR_VERSION
    onnx.checker.check_model(model)

    inputs = {name: p for p, name in leaf_names.items()}
    reconstructed = onnx_function_to_uop(function, inputs)
    assert _render(reconstructed) == original_wgsl


def test_onnx_function_to_uop_requires_exactly_the_formal_parameters():
    ast = _conv_kernel_ast()
    from tinygrad.uop.ops import Ops

    params = [u for u in ast.toposort() if u.op is Ops.PARAM]
    leaf_names = {p: f"param{p.arg.slot}" for p in params}
    function = uop_to_onnx_function(ast, "kernel_fn", leaf_names)

    with pytest.raises(ValueError):
        onnx_function_to_uop(function, {})  # missing every formal parameter


def test_encoding_an_unsupported_arg_type_raises_clearly():
    """The encoder is a whitelist, not a general object serializer -- an
    out-of-scope ``UOp.arg`` (a movement op's ``ShapeTracker``, say) must
    raise, never silently guess or produce something undecodable.
    """
    from tinygrad import dtypes
    from tinygrad.uop.ops import Ops, UOp

    bogus = UOp(Ops.CONST, dtypes.float, src=(), arg=object())
    with pytest.raises(TypeError):
        uop_to_onnx_model(bogus)
