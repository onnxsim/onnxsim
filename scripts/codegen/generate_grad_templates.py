#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generates onnxsim/graph_grad_templates_gen.py from onnxscript.

Proof of concept for the design discussed alongside
onnxsim.graph_grad: instead of hand-transcribing a VJP rule's graph
construction once in graph_grad.py and a second time in graph_grad.cpp (the
duplication that let a real bug -- a wrong `dvar` factor in
_grad_batch_normalization/GradBatchNormalization -- through review until a
finite-difference test caught it), author the rule *once*, in onnxscript, and
have both languages instantiate the same checked-in FunctionProto text via
ONNX's own function-inlining machinery (onnx.inliner in Python,
onnx::inliner::InlineLocalFunctions in C++ -- both already vendored,
see third_party/onnx/onnx/inliner/). This script only produces the Python
side; a C++ mirror is a followup once this pattern is proven, exactly like
generate_moe_function_templates.py above it in this directory produces a
checked-in header for contrib_schemas.cpp's own FunctionBuilder-based
instantiation -- the two scripts share the same shape: onnxscript is a
dev-only tool, never a build or runtime dependency, and its output is
checked in rather than regenerated on every build.

**Why these functions take no ONNX-level attributes.** graph_grad.py's
existing hand-written rules already resolve every rank/shape-dependent
choice (which axes broadcast, how many spatial dims a per-channel parameter
needs reshaping across) into concrete tensors -- constant axes lists via
`_Backward.int64_const`, reshaped broadcast operands via `Reshape` -- *before*
appending the op-specific arithmetic. Keeping that split (host code resolves
shape/rank into tensor-shaped data, the template consumes only tensors) means
every function below is a plain, rank-generic, attribute-free ONNX dataflow
graph: no `ref_attr_name` forwarding, no runtime `If`/`Loop`, none of the
sharp edges generate_moe_function_templates.py's own docstring documents for
onnxscript's attribute-parameter authoring. Broadcast-undoing on Add's
gradient is deliberately left to the *caller* (`_Backward.reduce_to`) for the
same reason: that logic already exists once, is already tested, and does not
need to be inside the template just because the template's caller was
rewritten.

**Numeric validation.** Each function's compiled FunctionProto is checked
(`onnx.checker.check_function`) and executed here against a finite-difference
reference computed from the exact formulas graph_grad.py's own hand-written
rule docstrings already document (`_grad_add`, `_grad_batch_normalization`) --
not merely compiled and trusted. See tests/test_graph_grad_templates.py for
the further check against torch.autograd on a real BatchNorm.

Run this script whenever the generated templates need to change:
    python3 scripts/codegen/generate_grad_templates.py \
        onnxsim/graph_grad_templates_gen.py \
        onnxsim/graph_grad_templates_gen.h
(the C++ header is optional -- pass just the first argument to regenerate
only the Python side; with no arguments at all, prints the Python module to
stdout instead of writing either file).
"""

import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnxscript import FLOAT, INT64, script
from onnxscript import opset17 as op
from onnxscript.values import Opset

# A private domain this repo fully controls -- never registered with ONNX,
# never seen by a runtime (onnx.inliner/InlineLocalFunctions expands every
# call site before a step graph is returned; see qat_graph.GraphBuilder.call
# and make_step_graph). Version 1 forever: these functions are checked-in,
# reviewed source, not a public/evolving op set that needs versioning.
GRAD_DOMAIN = Opset("onnxsim.grad", 1)


@script(opset=GRAD_DOMAIN)
def GradAdd(g: FLOAT["..."]):
    """``Add``'s VJP before broadcast-undoing: da = db = g. Trivial by
    design -- this function exists to validate the instantiation mechanism
    itself (call node -> model-local function -> onnx.inliner expansion)
    against the simplest possible math, isolating plumbing bugs from math
    bugs. See _grad_add in graph_grad.py for the broadcast-undoing
    (`_Backward.reduce_to`) this deliberately leaves to the caller."""
    da = op.Identity(g)
    db = op.Identity(g)
    return da, db


@script(opset=GRAD_DOMAIN)
def GradBatchNormalization(
    g: FLOAT["..."],
    x: FLOAT["..."],
    mean_b: FLOAT["..."],
    var_b: FLOAT["..."],
    scale_b: FLOAT["..."],
    eps: FLOAT,
    channel_axes: INT64["..."],
    one: FLOAT,
    neg_half: FLOAT,
):
    """``BatchNormalization``'s five gradients (inference mode), transcribed
    from graph_grad.py's _grad_batch_normalization docstring:

        xc = x - mean         inv = 1 / sqrt(var + eps)
        xhat = xc * inv       dx = g * scale * inv
        dscale = sum(g * xhat)         db    = sum(g)
        dmean  = -sum(dx)              dvar  = -0.5 * sum(dx * xhat * inv)

    `mean_b`/`var_b`/`scale_b` arrive already reshaped to broadcast against
    `x` (``[1, C, 1, ..., 1]``, rank matching x) and `channel_axes` already
    holds "every axis but 1" as an int64 tensor -- both rank-dependent
    choices the caller (_Backward, via int64_const/Reshape) resolves before
    the call, exactly as today's hand-written rule does inline. That is what
    keeps this function itself rank-generic: no attribute here depends on
    x's rank, so the same compiled FunctionProto instantiates for a rank-2
    (just batch and channel) or rank-5 input alike.

    `one`/`neg_half` are the literals `1.0`/`-0.5` as ordinary float32
    tensor inputs, the same way the hand-written GradBatchNormalization
    passes them (`ctx.b().Const(1.0f)`/`ctx.b().Const(-0.5f)`) rather than
    as in-body `Constant`/`CastLike` nodes: this function is checked against
    graph_grad.py's BACKWARD_OPS/graph_grad.cpp's BackwardOps() allowlist
    once inlined, and neither `Constant` nor `CastLike` is a member of it
    (see qat_graph.py's EP_FRIENDLY_OPS note for why the allowlist is
    deliberately narrow) -- so this function takes them as data instead of
    manufacturing them itself.
    """
    xc = op.Sub(x, mean_b)
    inv = op.Div(one, op.Sqrt(op.Add(var_b, eps)))
    xhat = op.Mul(xc, inv)
    gs = op.Mul(g, scale_b)
    dx = op.Mul(gs, inv)
    dscale = op.ReduceSum(op.Mul(g, xhat), channel_axes, keepdims=0)
    dbias = op.ReduceSum(g, channel_axes, keepdims=0)
    dmean = op.Neg(op.ReduceSum(dx, channel_axes, keepdims=0))
    dvar = op.Mul(
        op.ReduceSum(op.Mul(op.Mul(dx, xhat), inv), channel_axes, keepdims=0),
        neg_half,
    )
    return dx, dscale, dbias, dmean, dvar


_NP_TO_ONNX = {
    np.dtype("float32"): onnx.TensorProto.FLOAT,
    np.dtype("int64"): onnx.TensorProto.INT64,
}


def _run_function(fn: onnx.FunctionProto, feeds: dict, output_shapes: dict) -> list:
    """Evaluates a bare FunctionProto by wrapping it in a minimal model with
    a single call node -- the same call-node shape the real step-graph
    builder will use (onnx.reference.ReferenceEvaluator has no direct "run
    this FunctionProto" entry point). ``output_shapes`` must give every
    output's concrete shape: onnx.checker.check_model rejects a graph output
    with no shape at all, so this validator -- which always knows the shape
    it expects, being the one that chose the inputs -- states it explicitly
    rather than reaching for an "unknown rank" placeholder."""
    call = onnx.helper.make_node(
        fn.name, list(fn.input), list(fn.output), domain=fn.domain
    )
    value_info = [
        onnx.helper.make_tensor_value_info(
            n, _NP_TO_ONNX[feeds[n].dtype], list(feeds[n].shape)
        )
        for n in fn.input
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            n, onnx.TensorProto.FLOAT, list(output_shapes[n])
        )
        for n in fn.output
    ]
    graph = onnx.helper.make_graph([call], "poc", value_info, outputs)
    model = onnx.helper.make_model(
        graph,
        functions=[fn],
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid(fn.domain, 1),
        ],
    )
    onnx.checker.check_model(model)
    return ReferenceEvaluator(model).run(None, feeds)


def _validate_grad_add() -> None:
    fn = GradAdd.to_function_proto()
    onnx.checker.check_function(fn)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 3)).astype(np.float32)
    da, db = _run_function(fn, {"g": g}, {"da": g.shape, "db": g.shape})
    np.testing.assert_array_equal(da, g)
    np.testing.assert_array_equal(db, g)


def _validate_grad_batch_normalization() -> None:
    fn = GradBatchNormalization.to_function_proto()
    onnx.checker.check_function(fn)

    rng = np.random.default_rng(0)
    n, c, h, w = 2, 3, 4, 4
    x = rng.standard_normal((n, c, h, w)).astype(np.float32)
    mean = rng.standard_normal((c,)).astype(np.float32)
    var = np.abs(rng.standard_normal((c,))).astype(np.float32) + 0.1
    scale = rng.standard_normal((c,)).astype(np.float32)
    bias = rng.standard_normal((c,)).astype(np.float32)
    eps = 1e-5
    g = rng.standard_normal((n, c, h, w)).astype(np.float32)

    def bcast(v: np.ndarray) -> np.ndarray:
        return v.reshape(1, c, 1, 1)

    analytic = _run_function(
        fn,
        {
            "g": g,
            "x": x,
            "mean_b": bcast(mean),
            "var_b": bcast(var),
            "scale_b": bcast(scale),
            "eps": np.array(eps, dtype=np.float32),
            "channel_axes": np.array([0, 2, 3], dtype=np.int64),
            "one": np.array(1.0, dtype=np.float32),
            "neg_half": np.array(-0.5, dtype=np.float32),
        },
        {
            "dx": x.shape,
            "dscale": (c,),
            "dbias": (c,),
            "dmean": (c,),
            "dvar": (c,),
        },
    )

    def forward(x, mean, var, scale, bias):
        xc = x - bcast(mean)
        inv = 1.0 / np.sqrt(bcast(var) + eps)
        return xc * inv * bcast(scale) + bcast(bias)

    def loss(**kw):
        return float(np.sum(forward(**kw) * g))

    base = {"x": x, "mean": mean, "var": var, "scale": scale, "bias": bias}
    step = 1e-3

    def fd_grad(param: str) -> np.ndarray:
        grad = np.zeros_like(base[param], dtype=np.float64)
        flat = grad.reshape(-1)
        for i in range(flat.size):
            plus = {k: v.copy() for k, v in base.items()}
            minus = {k: v.copy() for k, v in base.items()}
            plus[param].reshape(-1)[i] += step
            minus[param].reshape(-1)[i] -= step
            flat[i] = (loss(**plus) - loss(**minus)) / (2 * step)
        return grad

    names = ["x", "scale", "bias", "mean", "var"]
    for name, value in zip(names, analytic):
        fd = fd_grad(name)
        err = np.max(np.abs(value.astype(np.float64) - fd))
        rel = err / (np.max(np.abs(fd)) + 1e-6)
        if rel >= 1e-2:
            raise AssertionError(f"GradBatchNormalization.{name}: relative error {rel}")


# (python identifier, C++ identifier, onnxscript function). Both languages'
# generated files are produced from this single list, so they cannot drift
# from each other -- the whole point of checking in ONNX function *text*
# rather than hand-porting the graph construction twice.
_ENTRIES = [
    ("GRAD_ADD", "kGradAddTemplate", GradAdd),
    (
        "GRAD_BATCH_NORMALIZATION",
        "kGradBatchNormalizationTemplate",
        GradBatchNormalization,
    ),
]


def _python_module(entries) -> str:
    out = []
    out.append("# SPDX-License-Identifier: Apache-2.0")
    out.append("#")
    out.append(
        "# GENERATED FILE -- do not edit by hand. Produced by\n"
        "#   python3 scripts/codegen/generate_grad_templates.py\n"
        "# from the onnxscript function definitions in that script; see its\n"
        "# module docstring for what this is and why it takes no ONNX-level\n"
        "# attributes."
    )
    out.append('"""ONNX function text for onnxsim.graph_grad\'s templated')
    out.append("gradient rules -- parsed back via onnx.parser.parse_function,")
    out.append('never onnxscript itself, which this module does not import."""')
    out.append("")
    out.append("from __future__ import annotations")
    out.append("")
    for py_ident, _cpp_ident, text in entries:
        out.append(f'{py_ident} = """{text}"""')
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _cpp_header(entries) -> str:
    """The same checked-in ONNX text, as C++ string-literal constants --
    onnxsim/graph_grad.cpp reads these with onnx::OnnxParser::Parse (the
    full onnx.defs.parser text format, not the per-statement-line format
    generate_moe_function_templates.py's own header needs for
    onnx::FunctionBuilder): the checked-in text is identical between the two
    languages, only the wrapper differs."""
    out = []
    out.append("// SPDX-License-Identifier: Apache-2.0")
    out.append("//")
    out.append(
        "// GENERATED FILE -- do not edit by hand. Produced by\n"
        "//   python3 scripts/codegen/generate_grad_templates.py\n"
        "// from the onnxscript function definitions in that script; see its\n"
        "// module docstring for what this is and why it takes no ONNX-level\n"
        "// attributes. graph_grad_templates_gen.py is the same text for the\n"
        "// Python side -- both are produced from the same entries so they\n"
        "// cannot drift from each other."
    )
    out.append("#ifndef ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    out.append("#define ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    out.append("")
    out.append(
        "// No enclosing namespace -- graph_grad.cpp, this header's only"
        " consumer, has\n// none either (it mirrors graph_grad.py's flat"
        " module directly)."
    )
    for _py_ident, cpp_ident, text in entries:
        out.append(f'constexpr const char* {cpp_ident} = R"GRAD_TPL({text})GRAD_TPL";')
        out.append("")
    out.append("#endif  // ONNXSIM_GRAPH_GRAD_TEMPLATES_GEN_H_")
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> None:
    _validate_grad_add()
    _validate_grad_batch_normalization()

    entries = [
        (py_ident, cpp_ident, onnx.printer.to_text(fn.to_function_proto()))
        for py_ident, cpp_ident, fn in _ENTRIES
    ]

    argv = sys.argv[1:]
    if not argv:
        sys.stdout.write(_python_module(entries))
        return
    with open(argv[0], "w") as f:
        f.write(_python_module(entries))
    if len(argv) > 1:
        with open(argv[1], "w") as f:
            f.write(_cpp_header(entries))


if __name__ == "__main__":
    main()
