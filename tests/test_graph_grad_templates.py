"""Proof of concept for onnxsim.graph_grad's *templated* gradient rules --
see graph_grad.py's own "Templated rules (proof of concept)" section and
scripts/codegen/generate_grad_templates.py's module docstring for what these
are: a VJP rule authored once in onnxscript, compiled offline to a checked-in
ONNX FunctionProto (onnxsim.graph_grad_templates_gen), and instantiated via a
call node + ``onnx.inliner.inline_local_functions`` instead of hand-writing
the same graph construction directly in Python (and a second time in C++).

Not exercising the production ``_RULES`` table: :func:`graph_grad.build_backward`
takes an explicit ``rules=`` override for exactly this purpose, so this file
substitutes ``_grad_add_templated``/``_grad_batch_normalization_templated``
without touching what every other caller of the module gets.

Two independent checks:

* ``test_grad_add_templated_matches_closed_form`` -- Add's gradient before
  broadcast-undoing is exactly ``da = db = g``, so this checks the templated
  path (call node -> model-local function -> inliner expansion) against a
  closed-form expectation rather than a numeric approximation. This isolates
  *plumbing* correctness (did the mechanism wire the right tensors through)
  from math correctness, which BatchNormalization's case below is for.

* ``test_grad_batch_normalization_templated_matches_torch_autograd`` -- the
  substantive case, and the rule this repo's own dvar-derivation bug was in
  (see graph_grad.py's history: a wrong factor in the hand-written
  ``_grad_batch_normalization`` survived review until a finite-difference
  test caught it). Checks the templated rule's output against
  ``torch.autograd.grad`` on the *identical* formula, written out in plain
  torch ops rather than ``torch.nn.functional.batch_norm`` -- so this is
  genuinely two independent implementations (onnxscript-compiled ONNX
  dataflow vs. PyTorch's own autodiff) agreeing on the same math, not one
  checking its own arithmetic. Also cross-checked against this repo's
  existing hand-written ``_grad_batch_normalization`` rule on the same
  graph, which should agree with the templated rule to near bit-identity
  since both compute the same formula.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnx.inliner
import onnx.parser
import onnx.shape_inference
import pytest

from onnxsim import graph_grad, qat_graph

ort = pytest.importorskip("onnxruntime")

# Same opset/IR pairing qat_graph builds its step graphs with, and
# tests/test_graph_grad.py's own convention.
_HEADER = '<ir_version: 8, opset_import: ["": 17]>'


def _model(body: str) -> onnx.ModelProto:
    return onnx.parser.parse_model(f"{_HEADER}\n{body}")


def _static_shapes(model: onnx.ModelProto) -> dict:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _backward_model(
    model: onnx.ModelProto, targets, rules, seed_name="dY"
) -> onnx.ModelProto:
    """``model``'s forward nodes followed by the emitted backward (built with
    ``rules``), as one runnable graph whose outputs are the requested
    gradients -- attaching any templated rule's model-local functions and
    inlining them before returning, exactly as qat_graph.make_step_graph
    does for a real step graph."""
    shapes = _static_shapes(model)
    output = model.graph.output[0].name
    b = qat_graph.GraphBuilder("bw_")
    grads = graph_grad.build_backward(
        b, list(model.graph.node), shapes, {output: seed_name}, targets, rules=rules
    )

    nodes = list(model.graph.node) + list(b.nodes)
    outputs = []
    for target in targets:
        name = f"grad_{target}"
        nodes.append(onnx.helper.make_node("Identity", [grads[target]], [name]))
        outputs.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, shapes[target]
            )
        )
    inputs = list(model.graph.input) + [
        onnx.helper.make_tensor_value_info(
            seed_name, onnx.TensorProto.FLOAT, shapes[output]
        )
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "backward",
        inputs,
        outputs,
        initializer=list(model.graph.initializer) + list(b.initializer),
    )
    opset_imports = [onnx.helper.make_opsetid("", 17)]
    opset_imports += [onnx.helper.make_opsetid(fn.domain, 1) for fn in b.functions]
    built = onnx.helper.make_model(
        graph, functions=list(b.functions), opset_imports=opset_imports
    )
    built.ir_version = 8
    onnx.checker.check_model(built)
    if b.functions:
        built = onnx.inliner.inline_local_functions(built)
    return built


def _run(model: onnx.ModelProto, output_names, feeds) -> list:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(output_names, feeds)


def _templated_rules() -> dict:
    rules = dict(graph_grad._RULES)
    rules["Add"] = graph_grad._grad_add_templated
    rules["BatchNormalization"] = graph_grad._grad_batch_normalization_templated
    return rules


def test_grad_add_templated_matches_closed_form():
    """``Y = Add(A, B)`` with ``B`` broadcasting against ``A`` -- exercises
    both the template call (``da = db = g``) and the broadcast-undoing
    ``_grad_add_templated`` still does outside it (``_Backward.reduce_to``,
    unchanged from the hand-written rule): dA is the seed itself, dB is the
    seed summed over the axis that broadcast."""
    model = _model(
        """
        g (float[3,4] A, float[4] B) => (float[3,4] Y) {
          Y = Add(A, B)
        }
        """
    )
    backward = _backward_model(model, ["A", "B"], _templated_rules())
    assert {n.name for n in backward.functions} == set()  # fully inlined

    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b_val = rng.standard_normal((4,)).astype(np.float32)
    g = rng.standard_normal((3, 4)).astype(np.float32)

    (da, db) = _run(backward, ["grad_A", "grad_B"], {"A": a, "B": b_val, "dY": g})
    np.testing.assert_array_equal(da, g)
    np.testing.assert_allclose(db, g.sum(axis=0), rtol=1e-6, atol=1e-6)


def test_grad_batch_normalization_templated_matches_torch_autograd():
    torch = pytest.importorskip(
        "torch", reason="this comparison specifically needs torch.autograd"
    )

    model = _model(
        """
        g (float[2,3,4,4] X, float[3] S, float[3] Bs, float[3] M, float[3] V)
            => (float[2,3,4,4] Y) {
          Y = BatchNormalization(X, S, Bs, M, V)
        }
        """
    )
    targets = ["X", "S", "Bs", "M", "V"]

    templated = _backward_model(model, targets, _templated_rules())
    assert {n.name for n in templated.functions} == set()  # fully inlined
    # The inlined template must stay inside the same execution-provider
    # allowlist a hand-written rule does -- graph_grad.py's own BACKWARD_OPS
    # comment explains why (WebGPU/WebNN coverage). Excludes the forward
    # model's own op_types (BatchNormalization itself, not a backward op)
    # and Identity (this test's own copy-out scaffolding, not something a
    # real caller emits -- see _backward_model). The first version of this
    # template failed this exact check: it built `1.0`/`-0.5` in-body via
    # Constant/CastLike, neither of which is in BACKWARD_OPS, instead of
    # taking them as data the way the hand-written rule's ctx.b.const(...)
    # already does for `eps` -- fixed by adding `one`/`neg_half` as ordinary
    # function inputs (see generate_grad_templates.py's GradBatchNormalization
    # docstring).
    forward_ops = {node.op_type for node in model.graph.node}
    emitted = {node.op_type for node in templated.graph.node} - forward_ops
    assert emitted <= graph_grad.BACKWARD_OPS | {"Identity"}, (
        f"templated backward reached outside the allowlist: "
        f"{sorted(emitted - graph_grad.BACKWARD_OPS - {'Identity'})}"
    )
    hand_written = _backward_model(model, targets, dict(graph_grad._RULES))

    rng = np.random.default_rng(0)
    n, c, h, w = 2, 3, 4, 4
    x = rng.standard_normal((n, c, h, w)).astype(np.float32)
    scale = rng.standard_normal((c,)).astype(np.float32)
    bias = rng.standard_normal((c,)).astype(np.float32)
    mean = rng.standard_normal((c,)).astype(np.float32)
    var = (np.abs(rng.standard_normal((c,))) + 0.1).astype(np.float32)
    eps = 1e-5
    seed = rng.standard_normal((n, c, h, w)).astype(np.float32)
    feeds = {"X": x, "S": scale, "Bs": bias, "M": mean, "V": var, "dY": seed}
    output_names = [f"grad_{t}" for t in targets]

    analytic = _run(templated, output_names, feeds)
    reference = _run(hand_written, output_names, feeds)
    for name, got, expected in zip(targets, analytic, reference):
        np.testing.assert_allclose(
            got, expected, rtol=1e-5, atol=1e-6, err_msg=f"{name}: vs hand-written rule"
        )

    # torch.autograd on the identical formula, written out in plain ops
    # (deliberately not torch.nn.functional.batch_norm, whose fused kernel
    # need not support differentiating w.r.t. running_mean/running_var even
    # with requires_grad_() set) -- this is two independent autodiff
    # implementations agreeing on the same math, the actual point of this
    # test.
    tx = torch.tensor(x, requires_grad=True)
    tscale = torch.tensor(scale, requires_grad=True)
    tbias = torch.tensor(bias, requires_grad=True)
    tmean = torch.tensor(mean, requires_grad=True)
    tvar = torch.tensor(var, requires_grad=True)
    tseed = torch.tensor(seed)

    def bcast(t):
        return t.reshape(1, c, 1, 1)

    xhat = (tx - bcast(tmean)) / torch.sqrt(bcast(tvar) + eps)
    y = xhat * bcast(tscale) + bcast(tbias)
    loss = (y * tseed).sum()
    torch_grads = torch.autograd.grad(loss, [tx, tscale, tbias, tmean, tvar])

    for name, got, expected in zip(targets, analytic, torch_grads):
        np.testing.assert_allclose(
            got,
            expected.detach().numpy(),
            rtol=2e-3,
            atol=2e-4,
            err_msg=f"{name}: onnxsim templated rule vs torch.autograd",
        )
