import onnx
from onnx import parser

import onnxsim


def _model_with_const_cond_function() -> onnx.ModelProto:
    # A local function whose body branches on a compile-time-constant `cond`
    # (a literal `Constant` node): after inlining, onnx-optimizer's default
    # `eliminate_if_with_const_cond` pass (run as part of simplify()'s fixed
    # point) should collapse the `If` into just the then-branch (`Relu`).
    #
    # The main graph's own opset_import deliberately omits the function's
    # "custom" domain -- inline_local_functions() has to add it before the
    # inliner will look at the function at all (see its own docstring).
    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        agraph (float[4] X) => (float[4] Y) {
          Y = custom.CondIdentity(X)
        }
        <
          domain: "custom",
          opset_import: ["": 18]
        >
        CondIdentity (x) => (y) {
          cond = Constant<value = bool[1] {1}>()
          y = If<
            then_branch = then_g () => (float[4] t) { t = Relu(x) },
            else_branch = else_g () => (float[4] e) { e = Neg(x) }
          >(cond)
        }
        """
    )


def _model_with_dynamic_cond_function() -> onnx.ModelProto:
    # Same shape, but `cond` is derived from the function's own input --
    # genuine data-dependent control flow that no pass can legally eliminate.
    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18, "custom": 1]
        >
        agraph (float[4] X) => (float[4] Y) {
          Y = custom.DynIdentity(X)
        }
        <
          domain: "custom",
          opset_import: ["": 18]
        >
        DynIdentity (x) => (y) {
          summed = ReduceSum<keepdims=0>(x)
          cond = Greater(summed, summed)
          y = If<
            then_branch = then_g () => (float[4] t) { t = Relu(x) },
            else_branch = else_g () => (float[4] e) { e = Neg(x) }
          >(cond)
        }
        """
    )


def test_no_functions_is_a_no_op():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 18]>
        agraph (float[4] X) => (float[4] Y) {
          Y = Relu(X)
        }
        """
    )
    out = onnxsim.inline_local_functions(model)
    # Same object back -- nothing to inline, so no copy is made either.
    assert out is model


def test_eliminates_an_if_whose_condition_is_a_compile_time_constant():
    model = _model_with_const_cond_function()
    assert "custom" not in {e.domain for e in model.opset_import}

    out = onnxsim.inline_local_functions(model)

    op_types = {n.op_type for n in out.graph.node}
    assert "If" not in op_types
    assert "CondIdentity" not in op_types
    assert not out.functions
    # The taken (then) branch survives, inlined directly into the graph.
    assert "Relu" in op_types
    assert "Neg" not in op_types


def test_raises_when_an_if_condition_is_not_a_compile_time_constant():
    model = _model_with_dynamic_cond_function()
    try:
        onnxsim.inline_local_functions(model)
    except ValueError as error:
        assert "If" in str(error)
    else:
        raise AssertionError("expected a ValueError naming the surviving If node")


def test_does_not_mutate_the_caller_supplied_model():
    model = _model_with_const_cond_function()
    original_bytes = model.SerializeToString()
    onnxsim.inline_local_functions(model)
    assert model.SerializeToString() == original_bytes
