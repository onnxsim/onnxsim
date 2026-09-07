"""Reverse-mode automatic differentiation over a slice of an ONNX graph, with
the gradient itself emitted as ordinary ONNX nodes.

:mod:`onnxsim.qat_graph` explains why a training step can run on an inference
runtime at all: a hand-derived backward pass is plain dataflow, so it is
expressible as an ordinary ONNX graph and therefore runs wherever an ONNX
model runs -- CUDA, an NPU execution provider, WebGPU in the browser. This
module removes the "hand-derived" part of that sentence.

**Why that matters.** Every gradient in this repo today is written out by
hand, once per pass, for one fixed expression:
:func:`onnxsim.adaround._build_rounding_step_graph` is twenty lines of
carefully transcribed chain rule for *one* layer's reconstruction error. That
scales as long as the forward is a single MatMul, and stops scaling the
moment it is not -- which is exactly the wall :mod:`onnxsim.brecq` documents
in its own docstring: its block discovery recognizes only a **linear chain**
of MatMul/Gemm layers with no normalization or activation node in between,
because every additional op shape would mean another hand-derived backward.
Block-wise QAT (``docs/qat.md``, deliverable B) needs the gradient of a real
transformer block -- MatMuls, a GELU, a residual Add, a Softmax -- against
its own output, and nobody should transcribe that by hand.

So: given the block's forward nodes, walk them in reverse, and for each one
append the nodes computing its vector-Jacobian product. The output is more
ONNX nodes in the same :class:`onnxsim.qat_graph.GraphBuilder`, so the result
composes with :func:`onnxsim.qat_graph.adam_update` and
:func:`onnxsim.qat_graph.make_step_graph` exactly as a hand-derived gradient
does, and reaches the same execution providers.

**What this is not.** It is not an autograd framework: there is no tape, no
``Tensor`` wrapper, no ``Gradient`` operator, and no runtime involvement --
differentiation happens once, at graph build time, and what ships is an
inference graph. It is also deliberately incomplete: a rule exists for the
ops a quantization-reconstruction block is made of
(:data:`SUPPORTED_OPS`), and an op without a rule raises
:class:`UnsupportedOpError` rather than being approximated or skipped. That
conservative boundary is the same one :mod:`onnxsim.pruning` and
:mod:`onnxsim.finetune` draw: refusing a case is recoverable, silently
emitting a wrong gradient is not -- it shows up as a model that trains to a
slightly worse answer, which is nearly impossible to attribute after the
fact.

**The one subtlety worth naming up front: broadcasting.** ``Add``, ``Mul``,
``Div`` and friends broadcast their inputs numpy-style, and the gradient of a
broadcast is a *sum* over the axes that were broadcast. A rule that returns
the incoming gradient unchanged for ``[4, 3] + [3]`` produces a ``[4, 3]``
gradient for a ``[3]`` parameter; ONNX will happily carry that shape
mismatch into the optimizer, where it broadcasts again and updates the
parameter with four times the intended step. Every rule here therefore
routes each contribution through :meth:`_Backward.reduce_to`, and
``tests/test_graph_grad.py`` tests broadcasting shapes on their own.

Everything is float32 at opset 17 / IR version 8 -- the pairing
:mod:`onnxsim.qat_graph` already builds and the accelerator backends already
run.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from onnxsim import qat_graph

# The complete set of operators the rules below can emit. Same standard the
# step graphs in ``tests/test_qat_graph.py`` are held to, and for the same
# reason: this machinery exists so training can run on WebGPU and NPU
# execution providers, and a backward graph that reached for a convenient op
# no such provider implements would be numerically perfect and practically
# useless. Everything here is plain arithmetic, a comparison, a reduction or
# a reshape. Deliberately absent: ``Where`` and boolean logic (a mask is a
# float 0/1 from ``Cast(Greater(...))``, multiplied in -- ``GraphBuilder``'s
# own convention), ``Expand`` (a broadcast is a multiply by a constant
# instead, see :func:`_grad_reduce`), and anything resembling control flow.
BACKWARD_OPS = frozenset(
    {
        "Add",
        "Cast",
        "Div",
        "Exp",
        "Greater",
        "Less",
        "MatMul",
        "Mul",
        "Neg",
        "ReduceSum",
        "Reshape",
        "Sub",
        "Transpose",
    }
)


class UnsupportedOpError(ValueError):
    """Raised for a node whose op type has no VJP rule.

    Also raised for a node whose op type *is* covered but whose particular
    configuration is not (a 1-D ``MatMul`` operand, a ``Reduce*`` whose
    reduced axes cannot be recovered from the shapes alone). The distinction
    does not matter to a caller: either way this module refuses to
    differentiate that node, and the caller must exclude it from the slice --
    which is the point. Guessing would produce a gradient that is quietly
    wrong.
    """


def _attr(node: onnx.NodeProto, name: str, default: Any) -> Any:
    """One of ``node``'s attributes by name, or ``default``.

    Typed loosely on purpose: an ONNX attribute is an int, a float or a list
    of ints depending on which one it is, and every caller below knows which
    it asked for.
    """
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


class _Backward:
    """Build-time state shared by the rules: the builder they append to, and
    the static shape of every tensor in the slice.

    Shapes are needed at build time, not run time, because the two things a
    correct VJP cannot do without -- undoing a broadcast, and undoing a
    reduction -- are both shape arithmetic. Requiring them up front is also
    why this module never emits ``Shape``/``Gather`` plumbing, which is the
    part of a generic autodiff implementation that accelerator backends
    handle worst.
    """

    def __init__(
        self, b: qat_graph.GraphBuilder, shapes: Dict[str, Sequence[int]]
    ) -> None:
        self.b = b
        self.shapes = shapes

    def shape(self, name: str) -> Tuple[int, ...]:
        if name not in self.shapes:
            raise ValueError(
                f"no static shape given for tensor {name!r}; build_backward needs "
                "the shape of every value the slice touches"
            )
        return tuple(int(d) for d in self.shapes[name])

    def int64_const(self, values: Sequence[int], hint: str = "i") -> str:
        """An int64 initializer, for the ``axes``/``shape`` inputs that
        ``ReduceSum`` and ``Reshape`` take as tensors from opset 13 on.

        ``GraphBuilder.const`` is float32-only, which is right for everything
        it was written for; these two are the exceptions.
        """
        array = np.asarray(list(values), dtype=np.int64)
        name = self.b.name(hint)
        self.b.initializer.append(onnx.numpy_helper.from_array(array, name))
        return name

    def reduce_to(
        self, grad: str, grad_shape: Sequence[int], target_shape: Sequence[int]
    ) -> str:
        """Sums ``grad`` back down to ``target_shape``, undoing a numpy-style
        broadcast.

        A binary op broadcasts a ``[3]`` operand against a ``[4, 3]`` one by
        replicating it four times; each replica gets its own gradient, and the
        gradient of the original is their sum. Leading axes the operand did
        not have at all are summed away entirely; axes it had as size 1 are
        summed with ``keepdims`` and then reshaped back, since ONNX's
        ``ReduceSum`` cannot drop *some* axes and keep others as size 1 in one
        node.
        """
        grad_shape = tuple(int(d) for d in grad_shape)
        target_shape = tuple(int(d) for d in target_shape)
        if grad_shape == target_shape:
            return grad

        offset = len(grad_shape) - len(target_shape)
        if offset < 0:
            raise ValueError(
                f"cannot reduce a gradient of shape {grad_shape} to {target_shape}: "
                "the gradient has fewer dimensions than the tensor it belongs to"
            )
        axes = list(range(offset))
        for i, dim in enumerate(target_shape):
            actual = grad_shape[offset + i]
            if dim == actual:
                continue
            if dim == 1:
                axes.append(offset + i)
            else:
                raise ValueError(
                    f"gradient shape {grad_shape} is not a broadcast of {target_shape}"
                )

        out = grad
        if axes:
            out = self.b.op(
                "ReduceSum",
                [out, self.int64_const(axes, "axes")],
                "unbcast",
                keepdims=1,
            )
        summed = tuple(1 if i in set(axes) else d for i, d in enumerate(grad_shape))
        if summed != target_shape:
            out = self.b.op(
                "Reshape", [out, self.int64_const(target_shape, "shape")], "unbcast"
            )
        return out

    def transpose_last_two(self, name: str, shape: Sequence[int]) -> str:
        """``name`` with its last two axes swapped -- what a MatMul's own VJP
        needs, and what a bare ``Transpose`` (which reverses *all* axes) would
        get wrong for a batched operand."""
        rank = len(shape)
        perm = list(range(rank - 2)) + [rank - 1, rank - 2]
        return self.b.transpose(name, perm)

    def mask_greater(self, x: str, bound: str) -> str:
        """``(x > bound)`` as a float32 0/1 tensor, with ``bound`` a tensor
        name rather than ``GraphBuilder.greater_mask``'s python float."""
        gt = self.b.op("Greater", [x, bound])
        return self.b.op("Cast", [gt], to=onnx.TensorProto.FLOAT)

    def mask_less(self, x: str, bound: str) -> str:
        """``(x < bound)`` as a float32 0/1 tensor."""
        lt = self.b.op("Less", [x, bound])
        return self.b.op("Cast", [lt], to=onnx.TensorProto.FLOAT)


# A rule takes the build context, the forward node, and the name of the
# gradient flowing into that node's single output; it appends nodes and
# returns one gradient name per node input (``None`` where an input takes no
# gradient -- a ``Reshape``'s shape operand, a ``Clip``'s bounds).
Rule = Callable[[_Backward, onnx.NodeProto, str], List[Optional[str]]]


def _grad_matmul(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    a, b = node.input[0], node.input[1]
    sa, sb = ctx.shape(a), ctx.shape(b)
    if len(sa) < 2 or len(sb) < 2:
        # ONNX MatMul promotes a 1-D operand to a matrix and then removes the
        # inserted axis from the result. Differentiating that means undoing
        # the removal, which is a different code path from the batched case
        # and is not worth carrying until a block needs it.
        raise UnsupportedOpError(
            f"MatMul with a 1-D operand is not differentiated here (node "
            f"{node.output[0]!r}, operand shapes {sa} and {sb})"
        )
    # dA = G @ B^T, dB = A^T @ G, both then summed back over whatever batch
    # axes broadcasting replicated.
    batch = tuple(np.broadcast_shapes(sa[:-2], sb[:-2]))
    ga = ctx.b.matmul(g, ctx.transpose_last_two(b, sb))
    gb = ctx.b.matmul(ctx.transpose_last_two(a, sa), g)
    return [
        ctx.reduce_to(ga, batch + (sa[-2], sa[-1]), sa),
        ctx.reduce_to(gb, batch + (sb[-2], sb[-1]), sb),
    ]


def _grad_gemm(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    alpha = float(_attr(node, "alpha", 1.0))
    beta = float(_attr(node, "beta", 1.0))
    trans_a = bool(_attr(node, "transA", 0))
    trans_b = bool(_attr(node, "transB", 0))
    a, b = node.input[0], node.input[1]
    sa, sb = ctx.shape(a), ctx.shape(b)
    if len(sa) != 2 or len(sb) != 2:
        raise UnsupportedOpError(
            f"Gemm expects 2-D A and B, got {sa} and {sb} (node {node.output[0]!r})"
        )

    # Y = alpha * A' B' + beta * C, with A' = A^T when transA. Differentiate
    # with respect to A' and B' first -- that is the plain matrix-product VJP
    # -- then transpose back into A's and B's own layouts. alpha scales the
    # incoming gradient once instead of scaling both results.
    gs = ctx.b.mul(g, ctx.b.const(alpha)) if alpha != 1.0 else g
    ga = ctx.b.matmul(gs, b if trans_b else ctx.b.transpose(b, [1, 0]))
    if trans_a:
        ga = ctx.b.transpose(ga, [1, 0])
    gb = ctx.b.matmul(a if trans_a else ctx.b.transpose(a, [1, 0]), gs)
    if trans_b:
        gb = ctx.b.transpose(gb, [1, 0])

    grads: List[Optional[str]] = [ga, gb]
    if len(node.input) > 2:
        if node.input[2]:
            # C broadcasts against [M, N], so its gradient needs the same
            # broadcast-undoing every elementwise rule needs.
            gc = ctx.reduce_to(g, ctx.shape(node.output[0]), ctx.shape(node.input[2]))
            grads.append(ctx.b.mul(gc, ctx.b.const(beta)) if beta != 1.0 else gc)
        else:
            # C spelled as an omitted optional input ("") rather than left off
            # the node entirely -- there is no tensor to give a gradient to.
            grads.append(None)
    return grads


def _grad_add(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    out = ctx.shape(node.output[0])
    return [
        ctx.reduce_to(g, out, ctx.shape(node.input[0])),
        ctx.reduce_to(g, out, ctx.shape(node.input[1])),
    ]


def _grad_sub(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    out = ctx.shape(node.output[0])
    # Negate after reducing, not before: the reduced tensor is the smaller of
    # the two, so this is the same value for fewer elementwise operations.
    gb = ctx.reduce_to(g, out, ctx.shape(node.input[1]))
    return [ctx.reduce_to(g, out, ctx.shape(node.input[0])), ctx.b.op("Neg", [gb])]


def _grad_mul(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    a, b = node.input[0], node.input[1]
    out = ctx.shape(node.output[0])
    return [
        ctx.reduce_to(ctx.b.mul(g, b), out, ctx.shape(a)),
        ctx.reduce_to(ctx.b.mul(g, a), out, ctx.shape(b)),
    ]


def _grad_div(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    a, b = node.input[0], node.input[1]
    y = node.output[0]
    out = ctx.shape(y)
    # d/db (a/b) = -a/b^2 = -y/b, reusing the forward quotient rather than
    # recomputing a square: one fewer node and no risk of overflowing b^2.
    gb = ctx.b.op("Neg", [ctx.b.div(ctx.b.mul(g, y), b)])
    return [
        ctx.reduce_to(ctx.b.div(g, b), out, ctx.shape(a)),
        ctx.reduce_to(gb, out, ctx.shape(b)),
    ]


def _grad_neg(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    return [ctx.b.op("Neg", [g])]


def _grad_identity(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # An alias, not a node: the gradient of the output *is* the gradient of
    # the input, and emitting an Identity to say so would only add a copy.
    return [g]


def _grad_relu(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # The subgradient at exactly 0 is taken as 0 (strict Greater), matching
    # the straight-through masks adaround.py already builds.
    return [ctx.b.mul(g, ctx.b.greater_mask(node.input[0], 0.0))]


def _grad_sigmoid(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # y (1 - y), from the forward output: the forward already computed the
    # sigmoid, so the backward never calls it again.
    y = node.output[0]
    dy = ctx.b.mul(y, ctx.b.sub(ctx.b.const(1.0), y))
    return [ctx.b.mul(g, dy)]


def _grad_tanh(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    y = node.output[0]
    dy = ctx.b.sub(ctx.b.const(1.0), ctx.b.mul(y, y))
    return [ctx.b.mul(g, dy)]


def _grad_erf(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # 2/sqrt(pi) * exp(-x^2). This one is here entirely for GELU, which
    # docs/qat.md's block-wise fine-tuning meets in every transformer FFN.
    x = node.input[0]
    dy = ctx.b.mul(
        ctx.b.const(2.0 / np.sqrt(np.pi)),
        ctx.b.op("Exp", [ctx.b.op("Neg", [ctx.b.mul(x, x)])]),
    )
    return [ctx.b.mul(g, dy)]


def _grad_exp(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    return [ctx.b.mul(g, node.output[0])]


def _grad_sqrt(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # 0.5 / sqrt(x), again reusing the forward result. Singular at x = 0, as
    # the derivative genuinely is -- not something to paper over here.
    return [ctx.b.div(ctx.b.mul(g, ctx.b.const(0.5)), node.output[0])]


def _grad_transpose(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    rank = len(ctx.shape(node.input[0]))
    perm = _attr(node, "perm", None)
    axes = list(reversed(range(rank))) if perm is None else [int(p) for p in perm]
    inverse = [0] * rank
    for position, axis in enumerate(axes):
        inverse[axis] = position
    return [ctx.b.transpose(g, inverse)]


def _grad_reshape(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    shape = ctx.shape(node.input[0])
    return [ctx.b.op("Reshape", [g, ctx.int64_const(shape, "shape")]), None]


def _reduced_axes(
    node: onnx.NodeProto, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]
) -> List[int]:
    """Which axes a ``Reduce*`` node reduced over.

    From opset 13 the axes are a *tensor input*, and ``build_backward`` is
    given nodes and shapes but not the initializers behind them, so the axes
    have to be recovered from the shapes. With ``keepdims=1`` that is exact
    (a reduced axis is one that became 1). With ``keepdims=0`` the axes were
    deleted, and recovering them means asking which deletions produce the
    observed output shape -- usually one answer, but ``[3, 3] -> [3]`` has
    two that disagree about where the gradient goes, and that case is refused
    rather than guessed. An explicit ``axes`` *attribute* (the pre-opset-13
    spelling, still seen on older graphs) short-circuits all of this.
    """
    rank = len(in_shape)
    attribute = _attr(node, "axes", None)
    if attribute is not None:
        return sorted({int(a) % rank for a in attribute})

    keepdims = bool(_attr(node, "keepdims", 1))
    if keepdims:
        if len(out_shape) != rank:
            raise UnsupportedOpError(
                f"{node.op_type} with keepdims=1 changed rank {rank} to "
                f"{len(out_shape)} (node {node.output[0]!r})"
            )
        # An axis that is 1 on both sides may or may not have been reduced,
        # and it makes no difference: summing over a length-1 axis and
        # broadcasting back over it are both the identity.
        return [i for i in range(rank) if out_shape[i] == 1 and in_shape[i] != 1]

    dropped = rank - len(out_shape)
    if dropped < 0 or rank > 16:
        raise UnsupportedOpError(
            f"cannot recover the reduced axes of {node.output[0]!r} from shapes "
            f"{in_shape} -> {out_shape}"
        )
    candidates = [
        combo
        for combo in itertools.combinations(range(rank), dropped)
        if tuple(d for i, d in enumerate(in_shape) if i not in combo) == out_shape
    ]
    if not candidates:
        raise UnsupportedOpError(
            f"no set of reduced axes takes {in_shape} to {out_shape} "
            f"(node {node.output[0]!r})"
        )
    # Two candidates that disagree only about length-1 axes imply the same
    # backward graph, so compare what actually gets built rather than the
    # axis sets themselves.
    expanded = {
        tuple(1 if i in combo else d for i, d in enumerate(in_shape))
        for combo in candidates
    }
    if len(expanded) != 1:
        raise UnsupportedOpError(
            f"the reduced axes of {node.output[0]!r} are ambiguous from shapes "
            f"{in_shape} -> {out_shape}; use keepdims=1 so they can be recovered"
        )
    return list(candidates[0])


def _grad_reduce(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``ReduceSum`` and ``ReduceMean``: broadcast the gradient back over the
    axes that were reduced away, scaled by 1/count for the mean.

    The broadcast is a multiply by a constant rather than an ``Expand``,
    which keeps the emitted graph inside :data:`BACKWARD_OPS`. The constant
    only spans the *reduced* axes (size 1 everywhere else) and broadcasting
    does the rest, so it costs the size of the reduction rather than the size
    of the tensor.
    """
    x = node.input[0]
    in_shape = ctx.shape(x)
    out_shape = ctx.shape(node.output[0])
    rest: List[Optional[str]] = [None] * (len(node.input) - 1)
    axes = set(_reduced_axes(node, in_shape, out_shape))
    if not axes:
        return [g] + rest

    keepdims_shape = tuple(1 if i in axes else d for i, d in enumerate(in_shape))
    grad = g
    if out_shape != keepdims_shape:
        grad = ctx.b.op("Reshape", [grad, ctx.int64_const(keepdims_shape, "shape")])

    fill = 1.0
    if node.op_type == "ReduceMean":
        fill = 1.0 / float(np.prod([in_shape[i] for i in axes]))
    ones_shape = tuple(d if i in axes else 1 for i, d in enumerate(in_shape))
    ones = ctx.b.const(np.full(ones_shape, fill, dtype=np.float32), "bcast")
    return [ctx.b.mul(grad, ones)] + rest


def _grad_softmax(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # dx = y * (g - sum(g * y)) along the softmax axis. Written from the
    # forward output y, so the backward re-runs neither the exponential nor
    # its normalization.
    y = node.output[0]
    rank = len(ctx.shape(node.input[0]))
    axis = int(_attr(node, "axis", -1)) % rank
    total = ctx.b.op(
        "ReduceSum", [ctx.b.mul(g, y), ctx.int64_const([axis], "axes")], keepdims=1
    )
    return [ctx.b.mul(y, ctx.b.sub(g, total))]


def _grad_clip(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """Pass the gradient through where the input was strictly inside the
    bounds, zero it elsewhere.

    The bounds are used as the tensors they are, so their values need not be
    known at build time -- and the comparison is strict, so an input sitting
    exactly on a bound gets no gradient. That matches the straight-through
    masks in :mod:`onnxsim.adaround` (``greater_mask`` * ``less_mask``) and
    keeps a clamped-to-the-limit parameter from drifting further out.
    """
    x = node.input[0]
    rest: List[Optional[str]] = [None] * (len(node.input) - 1)
    mask: Optional[str] = None
    if len(node.input) > 1 and node.input[1]:
        mask = ctx.mask_greater(x, node.input[1])
    if len(node.input) > 2 and node.input[2]:
        upper = ctx.mask_less(x, node.input[2])
        mask = upper if mask is None else ctx.b.mul(mask, upper)
    if mask is None:
        return [g] + rest
    return [ctx.b.mul(g, mask)] + rest


_RULES: Dict[str, Rule] = {
    "Add": _grad_add,
    "Clip": _grad_clip,
    "Div": _grad_div,
    "Erf": _grad_erf,
    "Exp": _grad_exp,
    "Gemm": _grad_gemm,
    "Identity": _grad_identity,
    "MatMul": _grad_matmul,
    "Mul": _grad_mul,
    "Neg": _grad_neg,
    "ReduceMean": _grad_reduce,
    "ReduceSum": _grad_reduce,
    "Relu": _grad_relu,
    "Reshape": _grad_reshape,
    "Sigmoid": _grad_sigmoid,
    "Softmax": _grad_softmax,
    "Sqrt": _grad_sqrt,
    "Sub": _grad_sub,
    "Tanh": _grad_tanh,
    "Transpose": _grad_transpose,
}

# The op types :func:`build_backward` can differentiate. Callers that pick
# the slice themselves -- block discovery for QAT, say -- should test against
# this rather than rediscovering the list by catching
# :class:`UnsupportedOpError`.
SUPPORTED_OPS = frozenset(_RULES)


def build_backward(
    b: qat_graph.GraphBuilder,
    nodes: Sequence[onnx.NodeProto],
    shapes: Dict[str, Sequence[int]],
    grad_outputs: Dict[str, str],
    targets: Sequence[str],
) -> Dict[str, str]:
    """Appends the reverse-mode gradient of ``nodes`` to ``b`` and returns
    where each target's gradient landed.

    The forward slice is differentiated by walking it backwards: each node's
    rule turns the gradient of its output into gradients of its inputs, and a
    tensor read by several nodes collects the sum of their contributions
    (which is the chain rule for a value used more than once -- a residual
    connection's own input being the case that matters here).

    Nothing is added to the forward graph, and no forward node is modified:
    the rules read the forward tensors by name, including node *outputs*
    where reusing them is cheaper than recomputing (``Sigmoid``, ``Tanh``,
    ``Exp``, ``Sqrt``, ``Softmax``). So the caller must place these nodes
    after the forward ones in the same graph, and keep the forward
    intermediates available -- which for a step graph they always are, since
    it is one graph evaluated once.

    :param b: the builder to append gradient nodes to. Passing the same
            builder the forward was built with is the normal case; the point
            is that the result composes with
            :func:`onnxsim.qat_graph.adam_update` and
            :func:`onnxsim.qat_graph.make_step_graph`.
    :param nodes: the forward nodes to differentiate, topologically ordered
            (the order they appear in a valid ONNX graph). Nodes outside the
            slice -- everything upstream of its inputs, everything downstream
            of where ``grad_outputs`` starts -- must not be included.
    :param shapes: the static shape of every tensor the slice touches, its
            inputs and outputs included. ``onnx.shape_inference.infer_shapes``
            on the forward model is the usual source.
    :param grad_outputs: ``{forward tensor name: tensor holding dL/d(that
            tensor)}``, the seed of the backward pass -- typically the single
            output of the block, with the gradient of the reconstruction loss
            against it. Seeding an *intermediate* tensor is allowed and adds
            to whatever the slice itself contributes to it, which is what an
            auxiliary loss on an intermediate activation means.
    :param targets: the tensors to return gradients for -- the parameters
            being trained, and any activation whose gradient the caller wants
            to propagate further.
    :returns: ``{target name: the tensor holding its gradient}``. A returned
            name may be one of ``grad_outputs``' own values when the path is
            a pure alias (a lone ``Identity``), so it is not guaranteed to be
            produced by a node in ``b``.
    :raises UnsupportedOpError: for a node this module will not
            differentiate. It is raised for *every* node in the slice with an
            unknown op type, including one no gradient reaches, so a caller
            learns the slice is out of scope from the shape of the graph
            rather than from whether a particular seed happened to reach it.
    :raises ValueError: if a target is not reachable from ``grad_outputs``
            through ``nodes`` (a disconnected target almost always means the
            slice or the target list is wrong, and a zero gradient would hide
            it), or if a shape is missing from ``shapes``.
    """
    ctx = _Backward(b, shapes)
    grads: Dict[str, str] = dict(grad_outputs)

    for node in reversed(list(nodes)):
        rule = _RULES.get(node.op_type)
        if rule is None:
            raise UnsupportedOpError(
                f"no gradient rule for op type {node.op_type!r} "
                f"(node {node.name or node.output[0]!r}); "
                f"onnxsim.graph_grad differentiates {sorted(SUPPORTED_OPS)}"
            )
        if len(node.output) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} has {len(node.output)} outputs; only "
                "single-output nodes are differentiated here"
            )
        g = grads.get(node.output[0])
        if g is None:
            # Nothing downstream depends on this node, so every gradient it
            # would produce is zero. Emitting those zeros would be correct
            # and pure waste.
            continue
        contributions = rule(ctx, node, g)
        if len(contributions) != len(node.input):
            raise AssertionError(
                f"the {node.op_type} rule returned {len(contributions)} gradients "
                f"for {len(node.input)} inputs"
            )
        for name, contribution in zip(node.input, contributions):
            if not name or contribution is None:
                continue
            # A tensor read by several nodes -- or twice by one node, as in
            # Mul(x, x) -- accumulates. Reverse topological order guarantees
            # every reader is visited before the producer, so by the time a
            # producer asks for its output gradient the sum is complete.
            existing = grads.get(name)
            grads[name] = (
                contribution if existing is None else b.add(existing, contribution)
            )

    result: Dict[str, str] = {}
    for target in targets:
        if target not in grads:
            raise ValueError(
                f"no gradient reaches {target!r}: it is not downstream of any of "
                f"{sorted(grad_outputs)} within the given nodes"
            )
        result[target] = grads[target]
    return result
