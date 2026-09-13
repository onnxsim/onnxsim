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

**Extending the boundary: :func:`register_gradient`.** A caller whose block
contains an op none of the builtin rules cover -- a custom domain op, or a
standard one this module has not grown a rule for yet -- can hand it one
directly, the same relationship :func:`torch.autograd.Function.backward` (or
more recently ``torch.library.register_autograd``) has to a custom torch op:
write the vector-Jacobian product once, register it against the op type, and
every caller that differentiates through :func:`build_backward` with its
default ``rules=None`` -- which is every public onnxsim entry point that
trains a block (:func:`onnxsim.apply_qat`, :func:`onnxsim.train_lora`, ...) --
picks it up with no further plumbing. Registration is process-global and
Python-only: it has no counterpart in ``graph_grad.h``/``.cpp`` (the
hand-ported C++ mirror the browser/WASM converter path uses), whose rule
table is a fixed, parity-pinned function-pointer map baked in at compile
time -- see :func:`register_gradient`'s own docstring for what that means
for a block trained both ways.

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

import contextlib
import functools
import itertools
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from onnxsim import graph_grad_templates_gen as _templates
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
        "Gather",
        "Greater",
        "Identity",
        "Less",
        "MatMul",
        "Mul",
        "Neg",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Sqrt",
        "Sub",
        "Transpose",
    }
)
# ``ReduceMean`` and ``Sqrt`` were admitted for
# :func:`_grad_layer_normalization`, which needs a mean over the normalized
# axes and the reciprocal square root of the variance. Neither is a loosening
# of the criterion above: both were already in
# :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS`, so the execution-provider
# coverage question was already settled for them, and both are exactly what
# this set describes -- a reduction and plain arithmetic. ``ReduceMean`` could
# be avoided by dividing a ``ReduceSum`` by a constant, but ``Sqrt`` could
# not, so contorting one of the two to keep the set at its old size would buy
# nothing.

# ``Gather`` was admitted for :func:`_grad_conv`, and is the one member here
# that is not arithmetic. It is also not a loosening: it was already in
# :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS` -- the note beside that set is
# where its coverage was established, and the same note now records what a
# ``Conv``/``ConvTranspose`` membership would have cost instead. The use is
# the same shape as the minibatching one it was admitted for there: a single
# axis, a constant int64 index, and no dependence of the *index* on any
# runtime value.

# :func:`_grad_batch_normalization` and :func:`_grad_instance_normalization`
# needed nothing from this set at all: every op their gradients use --
# ``Sub``, ``Div``, ``Mul``, ``Add``, ``Sqrt``, ``Reshape`` (for the
# per-channel broadcast, in place of ``Expand``), ``ReduceSum``/``ReduceMean``
# and ``Neg`` -- was already here for :func:`_grad_layer_normalization` or the
# elementwise rules. Worth recording precisely because it is the exception to
# every other note in this block, which each admitted one specific op for one
# specific rule.

# ``Identity`` was admitted for :func:`_grad_add_templated`. A hand-written
# rule that is a pure alias never emits a node at all (see
# :func:`_grad_identity`'s own "an alias, not a node" comment) -- it just
# returns an existing tensor's name -- but a *templated* rule's identity case
# (``GradAdd``'s ``da = db = g``, see
# scripts/codegen/generate_grad_templates.py) is compiled to an ONNX
# ``FunctionProto``, whose declared outputs onnxscript can only produce via an
# actual node, even when that node's whole job is to copy its input. Not a
# coverage gap: ``Identity`` is a plain copy with no arithmetic of its own --
# about the least a WebGPU/WebNN/NPU execution provider could fail to implement --
# and qat_entry.cpp's own choice to route around it elsewhere (renaming a
# tensor instead of emitting an ``Identity`` for it) was about avoiding a
# needless node, not about ``Identity`` lacking backend support.


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

    A dimension may be a plain ``int`` (its static size) or a ``str`` (an
    ONNX ``dim_param`` -- a size only known at ``Run()`` time, e.g. a batch
    axis). A rule is free to use a ``str`` entry exactly like an ``int`` one
    everywhere the shape arithmetic in this module is genuinely structural
    (a rank, a position, an equality check -- broadcasting a `[3]` onto a
    `["batch", 3]` still means "prepend one axis and sum it away on the way
    back", whatever that axis's own size turns out to be); it will misbehave
    -- an exception from a failed ``int()``/arithmetic conversion, not a
    silently wrong gradient -- the moment a rule tries to do real arithmetic
    with it (computing a reduction's ``1/N``, say), which is exactly the
    signal that particular rule does not (yet) support a dynamic dimension
    there. Every builtin rule this repo actually differentiates a dynamic
    batch axis through today keeps to the structural half; see
    ``tools/onnx-finetune/scripts/generate_distillation_step_graph.py``'s own
    "Dynamic batch size" docstring section for the one rule (``ReduceMean``)
    whose current implementation does not, and how that caller works around
    it rather than needing this module changed further.
    """

    def __init__(
        self, b: qat_graph.GraphBuilder, shapes: Dict[str, Sequence[Union[int, str]]]
    ) -> None:
        self.b = b
        self.shapes = shapes

    def shape(self, name: str) -> Tuple[Union[int, str], ...]:
        if name not in self.shapes:
            raise ValueError(
                f"no static shape given for tensor {name!r}; build_backward needs "
                "the shape of every value the slice touches"
            )
        return tuple(d if isinstance(d, str) else int(d) for d in self.shapes[name])

    def int64_const(self, values: Sequence[Union[int, str]], hint: str = "i") -> str:
        """An int64 initializer, for the ``axes``/``shape`` inputs that
        ``ReduceSum`` and ``Reshape`` take as tensors from opset 13 on.

        ``GraphBuilder.const`` is float32-only, which is right for everything
        it was written for; these two are the exceptions. ``values`` accepts
        a dynamic (``str``) entry only in the type-checking sense -- an
        actual int64 constant cannot hold one, so ``np.asarray(..., dtype=
        np.int64)`` below fails with its own clear ``ValueError`` if one
        slips through, the same "misbehave via a failed conversion" contract
        class ``_Backward``'s own docstring describes.
        """
        array = np.asarray(list(values), dtype=np.int64)
        name = self.b.name(hint)
        self.b.initializer.append(onnx.numpy_helper.from_array(array, name))
        return name

    def reduce_to(
        self,
        grad: str,
        grad_shape: Sequence[Union[int, str]],
        target_shape: Sequence[Union[int, str]],
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

        A dynamic (``str``) entry in ``grad_shape`` is only ever handled
        structurally: it can only appear among the *leading* axes summed away
        unconditionally (``axes = list(range(offset))`` below), since the
        loop that follows compares ``target_shape``'s own entries -- which,
        for every rule in this module, describe a plain weight and so never
        contain one themselves -- never a dynamic entry against a static one.
        A dynamic entry in ``target_shape`` (this module has no rule that
        produces one) would still raise below the same way an unexpected
        mismatched *static* one does, via the ``dim == actual`` comparison
        simply never matching.
        """
        grad_shape = tuple(d if isinstance(d, str) else int(d) for d in grad_shape)
        target_shape = tuple(d if isinstance(d, str) else int(d) for d in target_shape)
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

    def transpose_last_two(self, name: str, shape: Sequence[Union[int, str]]) -> str:
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

# A multi-output rule takes one gradient *per node output* instead of one --
# ``None`` in the same position where nothing downstream needs that output --
# and otherwise returns exactly what :data:`Rule` does. Kept as a distinct
# type (rather than widening :data:`Rule` itself) so every existing
# single-output rule's shape stays exactly what it always was; see
# :data:`_MULTI_OUTPUT_RULES` for why this is not just :data:`_CUSTOM_RULES`
# with a wider value type.
MultiOutputRule = Callable[
    [_Backward, onnx.NodeProto, List[Optional[str]]], List[Optional[str]]
]


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


def _prod(dims: Sequence[Union[int, str]]) -> int:
    """The number of elements a shape holds; ``1`` for a rank-0 one. A
    dynamic (``str``) entry fails via ``int()`` below, same as everywhere
    else in this module that needs a dimension's actual size rather than
    just its rank or position."""
    total = 1
    for d in dims:
        total *= int(d)
    return total


def _unflatten(index: int, dims: Sequence[int]) -> List[int]:
    """``index`` as row-major coordinates in ``dims``."""
    coords = [0] * len(dims)
    for i in reversed(range(len(dims))):
        coords[i] = index % int(dims[i])
        index //= int(dims[i])
    return coords


def _im2col_indices(
    in_dims: Sequence[int],
    out_dims: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
    pads_begin: Sequence[int],
) -> Tuple[List[int], np.ndarray]:
    """Where each ``(kernel tap, output position)`` pair reads its input.

    The pair ``(t, o)`` reads input position ``o * stride - pad + t *
    dilation`` along each spatial axis; a pair whose position falls outside
    the input is one the padding invented. Both are returned flattened in
    ``[tap, output position]`` order: the index (with an invented tap pointing
    at element 0, since ONNX's ``Gather`` rejects an out-of-range index
    outright) and a 0/1 float mask that multiplies the invented ones away
    afterwards.
    """
    spatial = len(in_dims)
    out_count = _prod(out_dims)
    index = [0] * (_prod(kernel) * out_count)
    mask = np.zeros(len(index), dtype=np.float32)
    for tap in range(_prod(kernel)):
        taps = _unflatten(tap, kernel)
        for out in range(out_count):
            position = _unflatten(out, out_dims)
            flat = 0
            for i in range(spatial):
                p = position[i] * strides[i] - pads_begin[i] + taps[i] * dilations[i]
                if p < 0 or p >= in_dims[i]:
                    flat = -1
                    break
                flat = flat * in_dims[i] + p
            if flat >= 0:
                index[tap * out_count + out] = flat
                mask[tap * out_count + out] = 1.0
    return index, mask


def _col2im_indices(
    in_dims: Sequence[int],
    out_dims: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
    pads_begin: Sequence[int],
) -> Tuple[List[int], np.ndarray]:
    """The same correspondence read the other way: which *output* position a
    given ``(kernel tap, input position)`` pair came from.

    Inverting ``p = o * stride - pad + t * dilation`` for ``o`` is what turns
    the gradient's scatter-add into a gather: for a fixed tap, every input
    position is written by at most one output position, so the whole ``dx``
    is a sum of ``prod(kernel)`` gathers of the incoming gradient rather than
    an accumulation into overlapping windows. A stride greater than one makes
    the division inexact for most positions -- those are exactly the input
    elements that tap never touched -- and they are masked away like the
    padded ones above.
    """
    spatial = len(in_dims)
    in_count = _prod(in_dims)
    index = [0] * (_prod(kernel) * in_count)
    mask = np.zeros(len(index), dtype=np.float32)
    for tap in range(_prod(kernel)):
        taps = _unflatten(tap, kernel)
        for entry in range(in_count):
            position = _unflatten(entry, in_dims)
            flat = 0
            for i in range(spatial):
                shifted = position[i] + pads_begin[i] - taps[i] * dilations[i]
                if shifted % strides[i] != 0:
                    flat = -1
                    break
                o = shifted // strides[i]
                if o < 0 or o >= out_dims[i]:
                    flat = -1
                    break
                flat = flat * out_dims[i] + o
            if flat >= 0:
                index[tap * in_count + entry] = flat
                mask[tap * in_count + entry] = 1.0
    return index, mask


def _conv_geometry(
    node: onnx.NodeProto,
    x_shape: Tuple[Union[int, str], ...],
    w_shape: Tuple[Union[int, str], ...],
    y_shape: Tuple[Union[int, str], ...],
) -> Tuple[int, List[int], List[int], List[int], List[int]]:
    """``Conv``'s attributes resolved against its actual shapes.

    Returns ``(group, kernel, strides, dilations, pads_begin)`` -- everything
    :func:`_grad_conv` needs to say where each output element read from --
    with ``auto_pad`` already turned into explicit padding.

    Every one of the refusals below is a configuration whose gradient this
    rule would otherwise compute against a geometry it invented. The last one
    is the important one: the resolved geometry is required to *reproduce the
    node's own output shape*, so a mistake in reading the attributes cannot
    survive to become a wrong gradient. A dynamic (``str``) entry anywhere in
    ``x_shape``/``w_shape``/``y_shape`` is not something this function
    supports -- Conv's own geometry math needs every one of these as a real
    size -- and fails via one of the ``int(...)`` conversions below rather
    than being refused up front, the same contract every other real-
    arithmetic user of a shape in this module keeps.
    """
    name = node.output[0]
    rank = len(x_shape)
    if rank < 3:
        raise UnsupportedOpError(
            f"Conv needs at least one spatial dimension, got input shape "
            f"{x_shape} (node {name!r})"
        )
    spatial = rank - 2
    if len(w_shape) != rank or len(y_shape) != rank:
        raise UnsupportedOpError(
            f"Conv's X, W and Y must have the same rank, got {x_shape}, "
            f"{w_shape} and {y_shape} (node {name!r})"
        )
    group = int(_attr(node, "group", 1))
    channels, features = int(x_shape[1]), int(w_shape[0])
    if group < 1 or channels % group != 0 or features % group != 0:
        raise UnsupportedOpError(
            f"Conv with group={group} does not divide its {channels} input and "
            f"{features} output channels (node {name!r})"
        )
    if int(w_shape[1]) != channels // group:
        raise UnsupportedOpError(
            f"Conv's W has {w_shape[1]} channels per group, but group={group} "
            f"over {channels} input channels needs {channels // group} "
            f"(node {name!r})"
        )

    kernel = [int(d) for d in w_shape[2:]]
    declared = _attr(node, "kernel_shape", None)
    if declared is not None and [int(k) for k in declared] != kernel:
        raise UnsupportedOpError(
            f"Conv's kernel_shape attribute {[int(k) for k in declared]} "
            f"disagrees with W's own spatial shape {kernel} (node {name!r})"
        )
    strides = [int(s) for s in _attr(node, "strides", [1] * spatial)]
    dilations = [int(d) for d in _attr(node, "dilations", [1] * spatial)]
    if len(strides) != spatial or len(dilations) != spatial:
        raise UnsupportedOpError(
            f"Conv's strides {strides} and dilations {dilations} must have one "
            f"entry per spatial axis ({spatial}) (node {name!r})"
        )
    if any(s < 1 for s in strides) or any(d < 1 for d in dilations):
        raise UnsupportedOpError(
            f"Conv with strides {strides} and dilations {dilations} is not a "
            f"convolution this rule can invert (node {name!r})"
        )

    auto_pad = _attr(node, "auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8")
    if auto_pad == "NOTSET":
        pads = [int(p) for p in _attr(node, "pads", [0] * (2 * spatial))]
        if len(pads) != 2 * spatial:
            raise UnsupportedOpError(
                f"Conv's pads {pads} must have two entries per spatial axis "
                f"({spatial}) (node {name!r})"
            )
    elif auto_pad == "VALID":
        pads = [0] * (2 * spatial)
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        # The spec's own formula, resolved here rather than left to the
        # runtime: the shapes are static, so "same" is a number at build time.
        pads = [0] * (2 * spatial)
        for i in range(spatial):
            size = int(x_shape[2 + i])
            out = -(-size // strides[i])
            span = (kernel[i] - 1) * dilations[i] + 1
            needed = max(0, (out - 1) * strides[i] + span - size)
            if auto_pad == "SAME_UPPER":
                pads[i] = needed // 2
            else:
                pads[i] = needed - needed // 2
            pads[spatial + i] = needed - pads[i]
    else:
        raise UnsupportedOpError(
            f"Conv with auto_pad {auto_pad!r} is not differentiated here "
            f"(node {name!r})"
        )

    if int(y_shape[0]) != int(x_shape[0]) or int(y_shape[1]) != features:
        raise UnsupportedOpError(
            f"Conv's output shape {y_shape} does not match its input {x_shape} "
            f"and weight {w_shape} (node {name!r})"
        )
    for i in range(spatial):
        span = (kernel[i] - 1) * dilations[i] + 1
        reach = int(x_shape[2 + i]) + pads[i] + pads[spatial + i] - span
        expected = reach // strides[i] + 1 if reach >= 0 else 0
        if expected != int(y_shape[2 + i]):
            raise UnsupportedOpError(
                f"Conv's declared output shape {y_shape} does not follow from "
                f"input {x_shape}, kernel {kernel}, strides {strides}, "
                f"dilations {dilations} and pads {pads}: axis {i} should be "
                f"{expected}, not {int(y_shape[2 + i])} (node {name!r})"
            )
    return group, kernel, strides, dilations, pads[:spatial]


def _grad_conv(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``Conv``'s three gradients, without emitting a convolution.

    **Why not a convolution.** ``dX`` is naturally a ``ConvTranspose`` and
    ``dW`` a ``Conv`` over permuted axes, and neither operator is in
    :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS`. Putting them there would have
    bought a rule that is dead on the very backends the allowlist exists for
    as soon as the convolution is not 2-D -- the note beside that set records
    what WebNN and onnxruntime-web's WebGPU backend actually implement. So
    this rule takes the other road: the im2col identity, which needs nothing
    the allowlist does not already have.

    **The identity.** Write the forward as a matrix product. With ``t``
    ranging over the kernel's taps and ``o`` over the output positions,

    .. code-block:: text

        col[c, t, o] = X[c, position(o, t)]      (im2col: one gather)
        Y[m, o]      = sum_{c, t} W[m, c, t] * col[c, t, o]

    which is a plain ``MatMul`` of ``W`` reshaped to ``[M, C*K]`` against
    ``col``. Differentiating a matrix product is the rule
    :func:`_grad_matmul` already implements, so::

        dW[m, c, t] = sum_o dY[m, o] * col[c, t, o]        (a MatMul)
        dcol[c, t, o] = sum_m W[m, c, t] * dY[m, o]        (a MatMul)
        dX = col2im(dcol)                                  (a scatter-add)

    and the last line is the only awkward one, because a scatter-add is not
    an operator here. It does not have to be: for a *fixed* tap, ``position``
    is injective -- input element ``p`` is read by at most one output
    position -- so col2im rearranges into a sum of ``prod(kernel)`` gathers
    of ``dY`` (:func:`_col2im_indices`), which is again one ``Gather`` and
    one ``MatMul``. Both directions are therefore the same three nodes:
    gather, mask, matmul.

    **Padding and stride, as a mask.** A tap that reads outside the input
    (padding) or an input element a strided tap never touched has no
    correspondent, and ONNX's ``Gather`` refuses an out-of-range index rather
    than producing a zero. Those entries are pointed at element 0 and
    multiplied by a 0/1 constant instead -- the same "a mask is a float 0/1,
    multiplied in" convention :class:`onnxsim.qat_graph.GraphBuilder` uses
    everywhere else. It is emitted unconditionally, even for a geometry whose
    mask is all ones, so that the two implementations of this rule cannot
    disagree about when to branch.

    **Groups** cost nothing extra: the group axis is split out of the channel
    axis by the reshapes that are already there, and the ``MatMul`` batches
    over it. The same is true of the number of spatial dimensions, which this
    rule never looks at beyond building the index tables -- a 1-D or 3-D
    convolution differentiates exactly like a 2-D one.

    **What it costs.** The two index tables are ``prod(kernel) *
    prod(output spatial)`` and ``prod(kernel) * prod(input spatial)``
    elements, materialized as initializers. That is the price of not needing
    a convolution kernel on the backend, and on a large feature map it is
    megabytes per node: a 3x3 convolution over 224x224 carries ~3.6 MB of
    int64 index and ~1.8 MB of mask. Reconstruction blocks are small and this
    is bounded and predictable, but it is real, and it is the reason this
    rule would not be the right one for a general-purpose trainer.
    """
    x, w = node.input[0], node.input[1]
    x_shape, w_shape = ctx.shape(x), ctx.shape(w)
    y_shape = ctx.shape(node.output[0])
    group, kernel, strides, dilations, pads = _conv_geometry(
        node, x_shape, w_shape, y_shape
    )
    bias: Optional[str] = None
    if len(node.input) > 2 and node.input[2]:
        bias = node.input[2]
        bias_shape = ctx.shape(bias)
        if tuple(bias_shape) != (int(w_shape[0]),):
            raise UnsupportedOpError(
                f"Conv's B has shape {tuple(bias_shape)}, not "
                f"({int(w_shape[0])},) (node {node.output[0]!r})"
            )

    batch = int(x_shape[0])
    features = int(w_shape[0]) // group
    channels = int(x_shape[1]) // group
    in_dims = [int(d) for d in x_shape[2:]]
    out_dims = [int(d) for d in y_shape[2:]]
    taps = _prod(kernel)
    in_count, out_count = _prod(in_dims), _prod(out_dims)

    # The incoming gradient with the group axis split out, which is the
    # layout both halves below want: [N, group, M/group, output positions].
    g4 = ctx.b.op(
        "Reshape",
        [g, ctx.int64_const([batch, group, features, out_count], "shape")],
    )

    # dX = sum over (m, t) of W[m, c, t] * dY[m, position], one gather of dY
    # per tap. See _col2im_indices for why the scatter-add is a gather here.
    index, mask = _col2im_indices(in_dims, out_dims, kernel, strides, dilations, pads)
    gathered = ctx.b.op("Gather", [g4, ctx.int64_const(index, "idx")], axis=3)
    masked = ctx.b.mul(
        gathered,
        ctx.b.const(mask.reshape(1, 1, 1, taps * in_count), "mask"),
    )
    dcol = ctx.b.op(
        "Reshape",
        [
            masked,
            ctx.int64_const([batch, group, features * taps, in_count], "shape"),
        ],
    )
    w4 = ctx.b.op(
        "Reshape",
        [w, ctx.int64_const([group, features, channels, taps], "shape")],
    )
    w4t = ctx.b.transpose(w4, [0, 2, 1, 3])
    # The leading 1 keeps both MatMul operands rank 4: a batch axis that
    # broadcasts is the mildest form of the broadcasting MatMul the rules
    # already rely on, and it keeps every tensor here inside the rank limit
    # WebNN's matmul states.
    wt = ctx.b.op(
        "Reshape",
        [
            w4t,
            ctx.int64_const([1, group, channels, features * taps], "shape"),
        ],
    )
    dx4 = ctx.b.matmul(wt, dcol)
    dx = ctx.b.op("Reshape", [dx4, ctx.int64_const(x_shape, "shape")])

    # dW = sum over (n, o) of dY[n, m, o] * col[n, c, t, o], with col the
    # forward's own im2col of X.
    x4 = ctx.b.op(
        "Reshape",
        [x, ctx.int64_const([batch, group, channels, in_count], "shape")],
    )
    index, mask = _im2col_indices(in_dims, out_dims, kernel, strides, dilations, pads)
    gathered = ctx.b.op("Gather", [x4, ctx.int64_const(index, "idx")], axis=3)
    masked = ctx.b.mul(
        gathered,
        ctx.b.const(mask.reshape(1, 1, 1, taps * out_count), "mask"),
    )
    col = ctx.b.op(
        "Reshape",
        [
            masked,
            ctx.int64_const([batch, group, channels * taps, out_count], "shape"),
        ],
    )
    colt = ctx.b.transpose(col, [0, 1, 3, 2])
    dw4 = ctx.b.matmul(g4, colt)
    dw3 = ctx.b.op(
        "ReduceSum",
        [dw4, ctx.int64_const([0], "axes")],
        keepdims=0,
    )
    dw = ctx.b.op("Reshape", [dw3, ctx.int64_const(w_shape, "shape")])

    grads: List[Optional[str]] = [dx, dw]
    if len(node.input) > 2:
        if bias is None:
            grads.append(None)
        else:
            db = ctx.b.op(
                "ReduceSum",
                [g4, ctx.int64_const([0, 3], "axes")],
                keepdims=0,
            )
            grads.append(
                ctx.b.op(
                    "Reshape",
                    [db, ctx.int64_const([int(w_shape[0])], "shape")],
                )
            )
    return grads


def _pool_geometry(
    node: onnx.NodeProto,
    x_shape: Tuple[Union[int, str], ...],
    y_shape: Tuple[Union[int, str], ...],
    *,
    has_dilations: bool,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """``MaxPool``/``AveragePool``'s attributes resolved against their actual
    shapes -- the same discipline :func:`_conv_geometry` applies to ``Conv``:
    the resolved kernel/strides/dilations/pads are required to reproduce the
    node's own declared output shape, so a misread attribute cannot survive
    to become a gradient computed against the wrong geometry. Like
    ``_conv_geometry``, a dynamic entry in either shape is unsupported here
    and fails via ``int(...)`` below rather than a separate check.

    Unlike ``Conv``, a pooling node carries no weight tensor to read
    ``kernel_shape`` off, so it is read directly -- it is a required
    attribute for both ops -- and there is no "does the weight agree with
    it" check to make.

    ``ceil_mode=1`` is refused outright rather than reproduced: it can make
    the rightmost window's far edge fall entirely inside the padding region,
    and ONNX's own spec leaves what happens then ("the sliding window ...
    will start as long as the starting index ... is less than the padded
    input size") looser than the exact-reproduction discipline this module
    otherwise holds itself to. Guessing which convention a given model was
    built for would be exactly the quietly-wrong gradient
    :class:`UnsupportedOpError`'s docstring warns about, so it is refused
    instead. Same for any ``auto_pad`` beyond ``NOTSET``/``VALID``/
    ``SAME_UPPER``/``SAME_LOWER``, and for a ``kernel_shape`` that never
    resolves to the declared output shape.
    """
    name = node.output[0]
    rank = len(x_shape)
    if rank < 3:
        raise UnsupportedOpError(
            f"{node.op_type} needs at least one spatial dimension, got input "
            f"shape {x_shape} (node {name!r})"
        )
    spatial = rank - 2
    if len(y_shape) != rank:
        raise UnsupportedOpError(
            f"{node.op_type}'s X and Y must have the same rank, got {x_shape} "
            f"and {y_shape} (node {name!r})"
        )
    if int(y_shape[0]) != int(x_shape[0]) or int(y_shape[1]) != int(x_shape[1]):
        raise UnsupportedOpError(
            f"{node.op_type}'s output shape {y_shape} does not match its "
            f"input {x_shape} in batch or channel size (node {name!r})"
        )

    declared = _attr(node, "kernel_shape", None)
    if declared is None:
        raise UnsupportedOpError(
            f"{node.op_type} without a kernel_shape attribute is not "
            f"differentiated here (node {name!r})"
        )
    kernel = [int(k) for k in declared]
    if len(kernel) != spatial:
        raise UnsupportedOpError(
            f"{node.op_type}'s kernel_shape {kernel} must have one entry per "
            f"spatial axis ({spatial}) (node {name!r})"
        )

    strides = [int(s) for s in _attr(node, "strides", [1] * spatial)]
    dilations = (
        [int(d) for d in _attr(node, "dilations", [1] * spatial)]
        if has_dilations
        else [1] * spatial
    )
    if len(strides) != spatial or len(dilations) != spatial:
        raise UnsupportedOpError(
            f"{node.op_type}'s strides {strides} and dilations {dilations} "
            f"must have one entry per spatial axis ({spatial}) (node {name!r})"
        )
    if any(s < 1 for s in strides) or any(d < 1 for d in dilations):
        raise UnsupportedOpError(
            f"{node.op_type} with strides {strides} and dilations {dilations} "
            f"is not a pooling this rule can invert (node {name!r})"
        )

    if int(_attr(node, "ceil_mode", 0)):
        raise UnsupportedOpError(
            f"{node.op_type} with ceil_mode=1 is not differentiated here "
            f"(node {name!r})"
        )

    auto_pad = _attr(node, "auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8")
    if auto_pad == "NOTSET":
        pads = [int(p) for p in _attr(node, "pads", [0] * (2 * spatial))]
        if len(pads) != 2 * spatial:
            raise UnsupportedOpError(
                f"{node.op_type}'s pads {pads} must have two entries per "
                f"spatial axis ({spatial}) (node {name!r})"
            )
    elif auto_pad == "VALID":
        pads = [0] * (2 * spatial)
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        # The spec's own formula, same as _conv_geometry's: output_shape[i] =
        # ceil(input_shape[i] / strides[i]), the odd remainder going to the
        # end for SAME_UPPER and the beginning for SAME_LOWER.
        pads = [0] * (2 * spatial)
        for i in range(spatial):
            size = int(x_shape[2 + i])
            out = -(-size // strides[i])
            span = (kernel[i] - 1) * dilations[i] + 1
            needed = max(0, (out - 1) * strides[i] + span - size)
            if auto_pad == "SAME_UPPER":
                pads[i] = needed // 2
            else:
                pads[i] = needed - needed // 2
            pads[spatial + i] = needed - pads[i]
    else:
        raise UnsupportedOpError(
            f"{node.op_type} with auto_pad {auto_pad!r} is not "
            f"differentiated here (node {name!r})"
        )

    for i in range(spatial):
        span = (kernel[i] - 1) * dilations[i] + 1
        reach = int(x_shape[2 + i]) + pads[i] + pads[spatial + i] - span
        expected = reach // strides[i] + 1 if reach >= 0 else 0
        if expected != int(y_shape[2 + i]):
            raise UnsupportedOpError(
                f"{node.op_type}'s declared output shape {y_shape} does not "
                f"follow from input {x_shape}, kernel {kernel}, strides "
                f"{strides}, dilations {dilations} and pads {pads}: axis {i} "
                f"should be {expected}, not {int(y_shape[2 + i])} "
                f"(node {name!r})"
            )
    return kernel, strides, dilations, pads[:spatial]


def _grad_averagepool(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``AveragePool``'s gradient: ``col2im`` of ``dY``, scaled by each
    window's own divisor.

    Pooling is depthwise by construction -- no channel mixes with another --
    so this is exactly the ``dX`` half of :func:`_grad_conv`'s im2col
    identity with the ``MatMul`` against ``W`` deleted outright: there is no
    weight, every "tap" contributes with the same fixed coefficient
    (``1 / divisor``), so summing a window's contributions is the ``ReduceSum``
    :func:`_grad_conv` uses to sum ``dW`` over its batch axis, not a weighted
    sum. Batch and channel are flattened into one axis throughout (called
    ``planes`` below) since neither this rule nor the index tables it uses
    ever need to tell them apart.

    **The divisor.** With ``count_include_pad=1``, or no padding at all, every
    window holds exactly ``prod(kernel)`` elements and the divisor is that one
    number for the whole tensor. With ``count_include_pad=0`` (the default)
    and nonzero padding, a window near the border is only averaged over its
    *non-padding* elements, so the divisor varies by output position -- it is
    exactly the count :func:`_im2col_indices`'s own validity mask already
    computes (a tap is "valid" there iff it is not padding), summed over the
    kernel taps. Both cases are folded into one ``[1, out_count]`` numpy array
    computed once at build time and baked into a constant, so the emitted
    graph never branches on which case it is.
    """
    x = node.input[0]
    x_shape = ctx.shape(x)
    y_shape = ctx.shape(node.output[0])
    kernel, strides, dilations, pads = _pool_geometry(
        node, x_shape, y_shape, has_dilations=False
    )
    count_include_pad = bool(_attr(node, "count_include_pad", 0))

    planes = int(x_shape[0]) * int(x_shape[1])
    in_dims = [int(d) for d in x_shape[2:]]
    out_dims = [int(d) for d in y_shape[2:]]
    taps = _prod(kernel)
    in_count, out_count = _prod(in_dims), _prod(out_dims)

    if count_include_pad:
        # Every window is divided by the full kernel size regardless of
        # padding -- that is what count_include_pad=1 means -- so this needs
        # no padding information at all.
        divisor = np.full((1, out_count), float(taps), dtype=np.float64)
    else:
        # Always the general path, never a "no padding, so it's just
        # prod(kernel)" shortcut: ``pads`` here is _pool_geometry's
        # pads_begin only, so a shortcut keyed on it would (and once did)
        # miss padding that is entirely on the *end* side of an axis, e.g.
        # explicit pads=[0, 0, 1, 1]. The mask sum below is exactly
        # prod(kernel) anyway when there truly is no padding on either side,
        # so nothing is lost by always taking this path.
        _, valid = _im2col_indices(in_dims, out_dims, kernel, strides, dilations, pads)
        divisor = (
            valid.reshape(taps, out_count).sum(axis=0, keepdims=True).astype(np.float64)
        )
        if np.any(divisor == 0):
            raise UnsupportedOpError(
                f"AveragePool has a window with no non-padding elements at "
                f"all (node {node.output[0]!r}); its average, and this "
                f"rule's divisor, are undefined for that window"
            )
    recip = (1.0 / divisor).astype(np.float32)

    g2 = ctx.b.op("Reshape", [g, ctx.int64_const([planes, out_count], "shape")])
    scaled = ctx.b.mul(g2, ctx.b.const(recip, "recip"))

    index, mask = _col2im_indices(in_dims, out_dims, kernel, strides, dilations, pads)
    gathered = ctx.b.op("Gather", [scaled, ctx.int64_const(index, "idx")], axis=1)
    masked = ctx.b.mul(gathered, ctx.b.const(mask.reshape(1, taps * in_count), "mask"))
    col = ctx.b.op(
        "Reshape", [masked, ctx.int64_const([planes, taps, in_count], "shape")]
    )
    dx_flat = ctx.b.op("ReduceSum", [col, ctx.int64_const([1], "axes")], keepdims=0)
    dx = ctx.b.op("Reshape", [dx_flat, ctx.int64_const(x_shape, "shape")])
    return [dx]


def _grad_maxpool(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``MaxPool``'s gradient: route ``dY`` to whichever input element each
    window's max came from.

    A ``MaxPool`` with the optional ``Indices`` output present is never
    reached here at all -- ``build_backward`` refuses any node with more than
    one declared output before a rule ever runs, which is exactly right for
    this one: computing a gradient *for* ``Indices`` (an integer tensor) does
    not make sense, and this rule does not need ``Indices`` to compute
    ``dX`` regardless, so nothing is lost by that refusal covering it too.
    ``GlobalMaxPool`` is a different op type with no rule registered, so it
    is refused the ordinary way. Neither is a design compromise here -- see
    :func:`build_backward`'s single-output check and :data:`SUPPORTED_OPS`.

    **No fresh ``ReduceMax``.** The forward already computed the window's max
    as ``node.output[0]``; recomputing it (besides needing an op outside
    :data:`BACKWARD_OPS`) would be redundant. Instead each window's input
    elements are gathered the way :func:`_grad_conv`'s ``dW`` half gathers
    ``X`` -- one ``Gather`` with :func:`_im2col_indices`'s constant index
    table -- and compared against the broadcast ``Y`` for that window with
    the two comparison ops this module already has::

        eq = (1 - Cast(Greater(gathered, Y))) * (1 - Cast(Less(gathered, Y)))

    which is 1 exactly where ``gathered`` equals ``Y`` and 0 elsewhere -- no
    ``Equal`` needed. A tap the padding invented is masked out unconditionally
    afterwards (its gathered value is an arbitrary element, which could
    spuriously equal ``Y``), the same "gather from element 0, multiply the
    invented ones away" convention :func:`_grad_conv` uses.

    **Ties.** More than one input element exactly equal to a window's max is
    a measure-zero event for real (or randomly generated) float data, but a
    rule that assumed it away would misbehave silently the one time it
    happens. So each window's ``eq`` mask is divided by its own tie count
    (``ReduceSum`` of ``eq`` over the tap axis, always >= 1 since ``Y`` is
    that window's max) before being multiplied by the incoming gradient --
    the same "split the credit" choice :func:`_grad_clip` documents for a
    value sitting exactly on a bound, made here so that a window's total
    outgoing gradient sums to exactly the ``dY`` that came in regardless of
    how many elements tie.

    **The scatter.** Unlike :func:`_grad_conv`'s ``dX`` (whose gathered value
    is the same for every tap, since the weight does the differentiating),
    here each tap's credited gradient genuinely differs per tap, so the plain
    :func:`_col2im_indices` trick -- gather the same ``[planes, out_count]``
    tensor for every tap -- does not apply as-is. Each tap's slice is instead
    given its own offset into a flattened ``[taps, out_count]`` axis (tap
    ``t``'s block starts at ``t * out_count``) so one ``Gather`` still reaches
    the right tap's own contribution for every input element, then a
    ``ReduceSum`` over the tap axis sums however many windows each input
    element belonged to -- the same shape of "sum via gather, not scatter"
    identity :func:`_col2im_indices`'s own docstring explains.
    """
    x, y = node.input[0], node.output[0]
    x_shape = ctx.shape(x)
    y_shape = ctx.shape(y)
    kernel, strides, dilations, pads = _pool_geometry(
        node, x_shape, y_shape, has_dilations=True
    )

    planes = int(x_shape[0]) * int(x_shape[1])
    in_dims = [int(d) for d in x_shape[2:]]
    out_dims = [int(d) for d in y_shape[2:]]
    taps = _prod(kernel)
    in_count, out_count = _prod(in_dims), _prod(out_dims)

    x2 = ctx.b.op("Reshape", [x, ctx.int64_const([planes, in_count], "shape")])
    index, valid = _im2col_indices(in_dims, out_dims, kernel, strides, dilations, pads)
    if np.any(valid.reshape(taps, out_count).sum(axis=0) == 0):
        # A window every tap of which the padding invented has no real
        # element to route dY to at all: the forward's own max over such a
        # window is -inf by construction, which no real input value equals,
        # so the tie count below would be a division by zero rather than a
        # quietly wrong credit. Caught here, at build time, rather than
        # producing a NaN gradient at run time.
        raise UnsupportedOpError(
            f"MaxPool has a window with no non-padding elements at all "
            f"(node {node.output[0]!r}); its max, and this rule's gradient "
            f"for that window, are undefined"
        )
    gathered = ctx.b.op("Gather", [x2, ctx.int64_const(index, "idx")], axis=1)
    windows = ctx.b.op(
        "Reshape", [gathered, ctx.int64_const([planes, taps, out_count], "shape")]
    )
    y3 = ctx.b.op("Reshape", [y, ctx.int64_const([planes, 1, out_count], "shape")])

    not_greater = ctx.b.sub(ctx.b.const(1.0), ctx.mask_greater(windows, y3))
    not_less = ctx.b.sub(ctx.b.const(1.0), ctx.mask_less(windows, y3))
    eq = ctx.b.mul(not_greater, not_less)
    eq = ctx.b.mul(eq, ctx.b.const(valid.reshape(1, taps, out_count), "valid"))

    tie_count = ctx.b.op("ReduceSum", [eq, ctx.int64_const([1], "axes")], keepdims=1)
    credit = ctx.b.div(eq, tie_count)

    g3 = ctx.b.op("Reshape", [g, ctx.int64_const([planes, 1, out_count], "shape")])
    contrib = ctx.b.mul(credit, g3)
    contrib2 = ctx.b.op(
        "Reshape", [contrib, ctx.int64_const([planes, taps * out_count], "shape")]
    )

    scatter_index, scatter_mask = _col2im_indices(
        in_dims, out_dims, kernel, strides, dilations, pads
    )
    # Offset each tap's block into its own slice of the flattened
    # [taps, out_count] axis contrib2 holds -- see the docstring above.
    for tap in range(taps):
        base = tap * in_count
        offset = tap * out_count
        for entry in range(in_count):
            scatter_index[base + entry] += offset

    gathered_back = ctx.b.op(
        "Gather", [contrib2, ctx.int64_const(scatter_index, "idx")], axis=1
    )
    masked_back = ctx.b.mul(
        gathered_back,
        ctx.b.const(scatter_mask.reshape(1, taps * in_count), "mask"),
    )
    col = ctx.b.op(
        "Reshape", [masked_back, ctx.int64_const([planes, taps, in_count], "shape")]
    )
    dx_flat = ctx.b.op("ReduceSum", [col, ctx.int64_const([1], "axes")], keepdims=0)
    dx = ctx.b.op("Reshape", [dx_flat, ctx.int64_const(x_shape, "shape")])
    return [dx]


# Reference-only: :data:`_RULES` wires "Add" to the templated
# `_grad_add_templated` below instead of this. Kept, and still exercised by
# tests/test_graph_grad_templates.py, as an independent hand-written
# implementation to cross-check the templated one's numbers against -- the
# same reasoning as `_grad_batch_normalization` below, whose own hand-written
# bug is why that cross-check exists at all.
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


def _grad_log(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    # d/dx log(x) = g / x. Admitted for the log-softmax term a cross-entropy
    # or knowledge-distillation loss needs (log(softmax(x)), built from this
    # plus the existing Softmax rule rather than as a fused LogSoftmax rule of
    # its own) -- singular at x = 0 the same way _grad_sqrt is, and for the
    # same reason: that is genuinely where the derivative blows up, not
    # something to paper over here.
    return [ctx.b.div(g, node.input[0])]


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
    node: onnx.NodeProto,
    in_shape: Tuple[Union[int, str], ...],
    out_shape: Tuple[Union[int, str], ...],
) -> List[int]:
    """Which axes a ``Reduce*`` node reduced over.

    Unlike ``_conv_geometry``/``_pool_geometry``, this one genuinely supports
    a dynamic (``str``) entry in either shape: every comparison below is
    structural (an axis's own position, or whether two shape entries are
    equal), never arithmetic on a dimension's size -- exactly what lets
    ``_grad_reduce`` differentiate a ``ReduceSum`` over a dynamic batch axis
    at all (see this module's own "dynamic batch size" notes).

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

    That constant needs the reduced axes' own sizes at build time -- fine for
    every axis this module has ever reduced over, until a dynamic (``str``)
    one is possible too (a ``ReduceSum`` over a dynamic batch axis, see
    ``tools/onnx-finetune/scripts/generate_distillation_step_graph.py``'s
    "Dynamic batch size" docstring section). ``ReduceMean``'s own ``1/N``
    factor is a build-time constant no matter what, so a *mean* over a
    dynamic axis is refused outright rather than guessed -- exactly the
    ``UnsupportedOpError`` contract this module's own module docstring
    promises for a configuration it will not differentiate, and the reason
    that caller does the batch-mean as ``ReduceSum`` then ``Div`` by a
    runtime scalar instead. A *sum*'s own scale (``fill = 1.0``) needs
    nothing from the dynamic axis's size, only the broadcast constant's own
    *shape* does -- and that is available at run time regardless, as the
    shape of ``x`` itself: ``x * 0 + fill`` is a same-shape-as-``x`` tensor
    filled with ``fill``, using only ops already in :data:`BACKWARD_OPS`
    (``Mul``/``Add``, not ``Shape``/``Expand``), correct however many of
    ``x``'s axes turn out dynamic and whatever their sizes are at that call.
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

    dynamic_axes = {i for i in axes if isinstance(in_shape[i], str)}
    if dynamic_axes and node.op_type == "ReduceMean":
        raise UnsupportedOpError(
            f"ReduceMean with a dynamic reduced axis {sorted(dynamic_axes)} is "
            "not differentiated here: its 1/N scale is a build-time constant, "
            "which a size only known at Run() time cannot be (node "
            f"{node.output[0]!r}). Use ReduceSum and divide by a runtime "
            "batch-size input instead."
        )

    fill = 1.0
    if node.op_type == "ReduceMean":
        fill = 1.0 / float(np.prod([in_shape[i] for i in axes]))
    if dynamic_axes:
        zero = ctx.b.mul(x, ctx.b.const(0.0))
        ones = ctx.b.add(zero, ctx.b.const(fill))
    else:
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


def _grad_layer_normalization(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``LayerNormalization``'s three gradients.

    The one op a transformer block needs that arithmetic alone does not give:
    a pre-norm decoder block is otherwise entirely covered by the rules above,
    so without this a block containing a fused LayerNorm is refused outright
    and the walk routes around it.

    Writing the forward out, over the normalized axes ``[axis, rank)``::

        mu   = mean(x)          xc  = x - mu
        var  = mean(xc * xc)    inv = 1 / sqrt(var + eps)
        xhat = xc * inv         y   = xhat * scale + b

    the gradients are the standard ones, with ``gs = g * scale``::

        db     = g summed over the broadcast axes
        dscale = (g * xhat) summed the same way
        dx     = inv * (gs - mean(gs) - xhat * mean(gs * xhat))

    ``dx``'s two mean terms are what make this more than a chain rule: each
    element's gradient depends on every other element in its normalization
    group, through the mean and the variance it helped set.

    ``mu`` and ``inv`` are recomputed here rather than read from the node's
    optional ``Mean``/``InvStdDev`` outputs, because those outputs are
    optional and a fused LayerNorm in a real model usually omits them.
    Recomputing costs two reductions and is always available; reusing them
    would be an optimization that silently does not apply.
    """
    x = node.input[0]
    scale = node.input[1]
    shape = ctx.shape(x)
    rank = len(shape)
    axis = int(_attr(node, "axis", -1)) % rank
    eps = float(_attr(node, "epsilon", 1e-5))
    # ``axes`` is an *attribute* here, not an input. ReduceSum moved its axes
    # to an input at opset 13 and ReduceMean only at opset 18, so at the step
    # graph's opset 17 the two spell the same idea differently -- the same
    # asymmetry :func:`_reduced_axes` untangles in the forward direction.
    axes = list(range(axis, rank))

    mu = ctx.b.op("ReduceMean", [x], axes=axes, keepdims=1)
    xc = ctx.b.sub(x, mu)
    var = ctx.b.op("ReduceMean", [ctx.b.mul(xc, xc)], axes=axes, keepdims=1)
    inv = ctx.b.div(ctx.b.const(1.0), ctx.b.sqrt(ctx.b.add(var, ctx.b.const(eps))))
    xhat = ctx.b.mul(xc, inv)

    gs = ctx.b.mul(g, scale)
    mean_gs = ctx.b.op("ReduceMean", [gs], axes=axes, keepdims=1)
    mean_gs_xhat = ctx.b.op("ReduceMean", [ctx.b.mul(gs, xhat)], axes=axes, keepdims=1)
    dx = ctx.b.mul(
        inv,
        ctx.b.sub(ctx.b.sub(gs, mean_gs), ctx.b.mul(xhat, mean_gs_xhat)),
    )

    grads: List[Optional[str]] = [dx]
    grads.append(ctx.reduce_to(ctx.b.mul(g, xhat), shape, ctx.shape(scale)))
    if len(node.input) > 2 and node.input[2]:
        grads.append(ctx.reduce_to(g, shape, ctx.shape(node.input[2])))
    return grads


# Reference-only: :data:`_RULES` wires "BatchNormalization" to the templated
# `_grad_batch_normalization_templated` below instead of this. Kept, and
# still exercised by tests/test_graph_grad_templates.py, as an independent
# hand-written implementation to cross-check the templated one's numbers
# against -- this is the rule whose own hand-written dvar-derivation bug
# motivated the templated design in the first place, so losing this
# cross-check would be losing exactly the regression test that would have
# caught it.
def _grad_batch_normalization(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``BatchNormalization``'s five gradients, in inference mode.

    A step graph never runs this op with ``training_mode=1``: this repo's
    fine-tuning graphs fake-quantize and reconstruct against *fixed* running
    statistics, they do not re-estimate a mean and variance from the current
    minibatch the way live training would. A ``training_mode=1`` node is
    refused anyway, but not by a check in this rule -- the spec requires it
    to have three outputs (``Y``, ``running_mean``, ``running_var``), so
    :func:`build_backward`'s own single-output check refuses it before this
    rule ever runs, the same way :func:`_grad_maxpool`'s docstring explains
    ``MaxPool``'s optional ``Indices`` output needs no rule-specific check of
    its own. What is left, at the opset-17 default (``training_mode``
    omitted or 0), is per the spec::

        xc   = x - mean            inv = 1 / sqrt(var + eps)
        xhat = xc * inv            y   = xhat * scale + B

    with ``mean``/``var`` the node's own *inputs* -- fixed per-channel
    numbers, not reductions of ``x``. That is the whole difference from
    :func:`_grad_layer_normalization`: there, ``mean``/``var`` are computed
    from ``x`` itself, so every output element's gradient depends on every
    other element through them; here they do not depend on ``x`` at all, so
    differentiating through them is nothing more than differentiating
    through two more per-channel constants, each appearing exactly where the
    forward above shows::

        dx     = g * scale * inv
        dscale = sum_{n, spatial} (g * xhat)
        db     = sum_{n, spatial} g
        dmean  = -sum_{n, spatial} (g * scale * inv)  = -sum_{n, spatial}(dx)
        dvar   = -0.5 * sum_{n, spatial} (g * scale * xc * inv^3)
               = -0.5 * sum_{n, spatial} (dx * xhat * inv)

    the last two reusing ``dx`` (``g * scale * inv``, already needed for
    ``x``'s own gradient) and ``xhat`` (already needed for ``dscale``)
    instead of recomputing either -- ``dx * xhat`` alone is only ``inv^2``
    (one factor from each), so ``dvar`` needs one more multiply by ``inv``
    to reach the ``inv^3`` the chain rule through ``sqrt`` actually produces.

    **The broadcast.** ``scale``, ``B``, ``mean`` and ``var`` are all
    ``[C]`` and broadcast against axis 1 specifically, not against ``x``'s
    trailing axes the way :meth:`_Backward.reduce_to` (built for ordinary
    numpy-style broadcasting -- every ``Add``/``Mul``/``Sub`` rule's own
    case) undoes. Reshaping ``scale`` and ``mean`` to ``[1, C, 1, ..., 1]``
    once, up front, turns their use in the forward arithmetic above into an
    ordinary broadcast; going the other way, the four channel-shaped
    gradients are each one ``ReduceSum`` over every axis but 1 with
    ``keepdims=0``, producing the ``[C]`` shape directly -- the same move
    :func:`_grad_conv` makes for its own bias gradient, generalized from
    "batch and the spatial axes" to "every axis but the channel one", which
    is also right when there are no spatial axes at all (``x`` rank 2, just
    batch and channel): the reduction is then over the batch axis alone.
    """
    x = node.input[0]
    scale, bias = node.input[1], node.input[2]
    mean, var = node.input[3], node.input[4]
    name = node.output[0]
    x_shape = ctx.shape(x)
    rank = len(x_shape)
    if rank < 2:
        raise UnsupportedOpError(
            f"BatchNormalization needs a batch and a channel axis, got "
            f"input shape {x_shape} (node {name!r})"
        )
    channels = int(x_shape[1])
    for label, tensor in (
        ("scale", scale),
        ("B", bias),
        ("mean", mean),
        ("var", var),
    ):
        shape = ctx.shape(tensor)
        if tuple(shape) != (channels,):
            raise UnsupportedOpError(
                f"BatchNormalization's {label} has shape {tuple(shape)}, not "
                f"({channels},) (node {name!r})"
            )
    eps = float(_attr(node, "epsilon", 1e-5))

    bshape = (1, channels) + (1,) * (rank - 2)

    def bcast(t: str) -> str:
        return ctx.b.op("Reshape", [t, ctx.int64_const(bshape, "shape")])

    xc = ctx.b.sub(x, bcast(mean))
    inv = ctx.b.div(
        ctx.b.const(1.0), ctx.b.sqrt(ctx.b.add(bcast(var), ctx.b.const(eps)))
    )
    xhat = ctx.b.mul(xc, inv)

    gs = ctx.b.mul(g, bcast(scale))
    dx = ctx.b.mul(gs, inv)

    channel_axes = [0] + list(range(2, rank))

    def reduce_channel(t: str) -> str:
        return ctx.b.op(
            "ReduceSum", [t, ctx.int64_const(channel_axes, "axes")], keepdims=0
        )

    dscale = reduce_channel(ctx.b.mul(g, xhat))
    dbias = reduce_channel(g)
    dmean = ctx.b.op("Neg", [reduce_channel(dx)])
    dvar = ctx.b.mul(
        reduce_channel(ctx.b.mul(ctx.b.mul(dx, xhat), inv)), ctx.b.const(-0.5)
    )

    return [dx, dscale, dbias, dmean, dvar]


def _grad_instance_normalization(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``InstanceNormalization``'s three gradients.

    Same coupling as :func:`_grad_layer_normalization` -- mean and variance
    are computed from ``x`` itself, so every element's gradient depends on
    every other element in its group through them -- but a different group:
    one ``(batch, channel)`` pair, reduced over the spatial axes ``[2,
    rank)`` only, rather than a suffix of axes named by an ``axis``
    attribute (``InstanceNormalization`` has none; the channel axis is
    always 1, per the spec). Per the ONNX doc::

        mu   = mean_spatial(x)      xc  = x - mu
        var  = mean_spatial(xc^2)   inv = 1 / sqrt(var + eps)
        xhat = xc * inv             y   = xhat * scale + B

    with ``scale``/``B`` one entry per channel, broadcasting against every
    batch element and every spatial position --
    :func:`_grad_batch_normalization`'s broadcast, not layer-norm's, so
    ``scale`` is reshaped to ``[1, C, 1, ..., 1]`` the same way before use.
    Past that reshape, ``dx`` is exactly layer-norm's own ``dx`` with
    ``axes`` set to the spatial ones, and ``dscale``/``db`` are batch-norm's
    channel-reduced ``ReduceSum``, over batch and spatial together this time
    since neither of those is the channel axis.

    A rank below 3 -- no spatial axis at all -- is refused: the reduction
    would then be over an empty axis list, which ``ReduceMean`` treats as
    "reduce every axis" rather than "reduce none", silently changing what
    "instance" means. That is exactly the kind of behaviour
    :class:`UnsupportedOpError` exists to catch at build time instead of
    risking.
    """
    x, scale, bias = node.input[0], node.input[1], node.input[2]
    name = node.output[0]
    x_shape = ctx.shape(x)
    rank = len(x_shape)
    if rank < 3:
        raise UnsupportedOpError(
            f"InstanceNormalization needs at least one spatial dimension, "
            f"got input shape {x_shape} (node {name!r})"
        )
    channels = int(x_shape[1])
    for label, tensor in (("scale", scale), ("B", bias)):
        shape = ctx.shape(tensor)
        if tuple(shape) != (channels,):
            raise UnsupportedOpError(
                f"InstanceNormalization's {label} has shape {tuple(shape)}, "
                f"not ({channels},) (node {name!r})"
            )
    eps = float(_attr(node, "epsilon", 1e-5))
    spatial_axes = list(range(2, rank))
    bshape = (1, channels) + (1,) * (rank - 2)
    scale_b = ctx.b.op("Reshape", [scale, ctx.int64_const(bshape, "shape")])

    mu = ctx.b.op("ReduceMean", [x], axes=spatial_axes, keepdims=1)
    xc = ctx.b.sub(x, mu)
    var = ctx.b.op("ReduceMean", [ctx.b.mul(xc, xc)], axes=spatial_axes, keepdims=1)
    inv = ctx.b.div(ctx.b.const(1.0), ctx.b.sqrt(ctx.b.add(var, ctx.b.const(eps))))
    xhat = ctx.b.mul(xc, inv)

    gs = ctx.b.mul(g, scale_b)
    mean_gs = ctx.b.op("ReduceMean", [gs], axes=spatial_axes, keepdims=1)
    mean_gs_xhat = ctx.b.op(
        "ReduceMean", [ctx.b.mul(gs, xhat)], axes=spatial_axes, keepdims=1
    )
    dx = ctx.b.mul(
        inv, ctx.b.sub(ctx.b.sub(gs, mean_gs), ctx.b.mul(xhat, mean_gs_xhat))
    )

    channel_axes = [0] + spatial_axes

    def reduce_channel(t: str) -> str:
        return ctx.b.op(
            "ReduceSum", [t, ctx.int64_const(channel_axes, "axes")], keepdims=0
        )

    dscale = reduce_channel(ctx.b.mul(g, xhat))
    dbias = reduce_channel(g)
    return [dx, dscale, dbias]


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


def _grad_where(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``Y = Where(cond, X, Z)`` selects ``X`` where ``cond`` is true and
    ``Z`` otherwise, elementwise, with all three inputs broadcasting together
    against the output.

    ``cond`` never gets a gradient -- it is a boolean, not a function of
    anything float, the same "no gradient for a shape/axes/indices operand"
    convention ``Reshape``/``Gather``/``Split`` already follow here. The
    incoming gradient is *split* between the two branches by a float 0/1
    mask built from ``cond`` with ``Cast`` -- not re-emitted as another
    ``Where``, which is deliberately absent from :data:`BACKWARD_OPS` (see
    that set's own comment): this rule only ever emits ``Cast``/``Sub``/
    ``Mul``, all three already in it.
    """
    cond, x, z = node.input
    out = ctx.shape(node.output[0])
    mask = ctx.b.op("Cast", [cond], to=onnx.TensorProto.FLOAT)
    not_mask = ctx.b.sub(ctx.b.const(1.0), mask)
    dx = ctx.reduce_to(ctx.b.mul(g, mask), out, ctx.shape(x))
    dz = ctx.reduce_to(ctx.b.mul(g, not_mask), out, ctx.shape(z))
    return [None, dx, dz]


def _grad_is_nan(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``IsNaN`` produces a boolean and has no gradient of its own -- in
    every real graph this rule exists for, its only consumer is a
    ``Where``'s ``cond`` input, which itself takes no gradient either.

    This rule's whole job is letting :func:`build_backward` walk over an
    ``IsNaN`` node in a differentiated slice at all: that function requires
    a registered rule for *every* node type it visits, including one no
    gradient reaches (see its own docstring), so a numerical-stability guard
    like ``attn = Where(IsNaN(attn), 0, attn)`` sitting inline in a forward
    slice would otherwise raise :class:`UnsupportedOpError` even though
    nothing downstream ever asks this node for a gradient.
    """
    return [None]


def _grad_squeeze_or_unsqueeze(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``Squeeze``/``Unsqueeze``'s gradient: both only remove or insert
    size-1 axes, so -- exactly like :func:`_grad_reshape`, whose logic this
    duplicates rather than shares only because the two are registered under
    different op-type keys -- the adjoint is a ``Reshape`` of the incoming
    gradient back to ``data``'s own shape. ``axes`` (opset 13+'s optional
    second input, a tensor) gets no gradient, the same "shape/indices input
    is not a function of anything float" convention every such second input
    in this module already follows.

    Found real and load-bearing, not a theoretical gap: NVIDIA Parakeet's
    real RNN-T decoder export (`scripts/axera/build_parakeet_lstm_probe.py`)
    has a `Squeeze` sitting between its two LSTM layers (unrelated to
    `scripts/axera/legalize.py`'s `unroll_lstm`, which does not itself emit
    one) that `build_backward` refused to walk over at all before this.
    """
    shape = ctx.shape(node.input[0])
    grad = ctx.b.op("Reshape", [g, ctx.int64_const(shape, "shape")])
    return [grad] + [None] * (len(node.input) - 1)


def _grad_gather(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``Gather``'s gradient: a scatter-add into ``data``, ``indices`` itself
    untouched.

    This is the embedding-lookup case -- ``Gather(table, token_ids)`` -- not
    the constant-index-table use :func:`_grad_conv` makes of the *forward*
    ``Gather`` op (see the note above :data:`BACKWARD_OPS`, which is a fact
    about what the rules may *emit*, not about what they can differentiate).
    Here ``indices`` is a real, run-time-valued tensor and ``data`` is the
    thing being trained, so the direction of travel is the opposite one.

    **Why a scatter-add, and why a one-hot matmul instead.** ONNX's
    ``Gather`` reads ``N = data.shape[axis]`` rows and can read the same row
    more than once (a repeated token in a sequence); the gradient of a
    duplicated read is the *sum* of every place it was read, i.e. exactly a
    scatter-add of ``dY`` into ``dData`` at each row ``indices`` named. There
    is no scatter op in :data:`BACKWARD_OPS`, and there should not need to
    be: the same accumulation is a matrix product against a one-hot matrix,
    built from ops already there --

    .. code-block:: text

        onehot[l, n] = 1 if indices[l] == n else 0     (Greater/Less/Cast/Sub)
        dData[n, ...] = sum_l onehot[l, n] * dY[l, ...]  (a MatMul)

    ``onehot == 1`` is built the same two-sided way every other rule here
    builds a boolean without an ``Equal`` node: ``(1 - (idx > n)) * (1 - (idx
    < n))``, with ``idx`` cast to float and broadcast against a constant
    ``arange(N)``.

    **Shape handled.** ``data`` has ``axis`` picking out one of its axes
    (``data: [..., N, ...]``); everything before it flattens to a batch
    dimension ``pre`` and everything after to ``post``, so the one-hot
    ``MatMul`` is ``[1, N, L] x [pre, L, post] -> [pre, N, post]`` -- the same
    "leading 1 broadcasts across a batch axis" trick :func:`_grad_conv` uses
    to keep every operand inside the rank a batched ``MatMul`` wants.
    ``indices`` itself must be rank 0 (a single scalar lookup) or rank 1 (a
    sequence of lookups, ``L`` of them) -- squarely the embedding-lookup
    shape. A higher-rank index tensor (batched lookups, e.g. ``[batch,
    seq]``) is the same idea with an extra flatten/unflatten, but is refused
    here rather than risked: a wrong reshape there would not fail loudly, it
    would silently mix gradients across batch elements. Negative ``axis`` and
    negative index *values* (``indices[i] + N`` per the ONNX-13 spec) are
    both resolved -- the axis at build time, the values in the graph itself,
    since they are only known at run time.

    **What it costs.** The one-hot matrix is ``L x N`` -- the number of
    lookups times the *entire* size of the gathered axis, not just the rows
    actually read. For a small reconstruction block's embedding table this is
    fine; for a large vocabulary (tens of thousands of rows) times a long
    sequence it is a lot of both compute and a materialized ``arange``
    constant, in exactly the same "real, and the reason this rule would not
    be the right one for a general-purpose trainer" way :func:`_grad_conv`'s
    own docstring is upfront about for its index tables.

    **What this does not do.** Differentiating a ``Gather`` node makes its
    contribution to a block *differentiable* -- a block containing an
    embedding lookup no longer has to be routed around. It does not by itself
    make the embedding table a *trained* weight: the block-fine-tuning weight
    finders in :mod:`onnxsim.qat` (the ``MatMul``/``Gemm``/``Conv``-only
    finder and :func:`onnxsim.qat._plan_trained`) still only recognize those
    three op types' weight inputs. Teaching them to also recognize
    ``Gather``'s ``data`` input as trainable is separate, out-of-scope
    follow-up work.
    """
    data, indices = node.input[0], node.input[1]
    data_shape = ctx.shape(data)
    idx_shape = ctx.shape(indices)
    rank = len(data_shape)
    if len(idx_shape) > 1:
        raise UnsupportedOpError(
            f"Gather with rank-{len(idx_shape)} indices is not differentiated "
            f"here (node {node.output[0]!r}); only a scalar or a rank-1 index "
            "vector is supported"
        )
    axis = int(_attr(node, "axis", 0)) % rank
    count = int(data_shape[axis])
    pre_shape, post_shape = data_shape[:axis], data_shape[axis + 1 :]
    pre, post = _prod(pre_shape), _prod(post_shape)
    length = _prod(idx_shape)  # 1 for a scalar (empty shape), L for a vector

    # indices as a flat float32 [length] vector, with negative values resolved
    # to indices[i] + count -- a run-time value, so this happens in the graph
    # rather than at build time.
    flat_idx = ctx.b.op("Reshape", [indices, ctx.int64_const([length], "shape")])
    idx_f = ctx.b.op("Cast", [flat_idx], to=onnx.TensorProto.FLOAT)
    is_negative = ctx.mask_less(idx_f, ctx.b.const(0.0))
    idx_resolved = ctx.b.add(idx_f, ctx.b.mul(is_negative, ctx.b.const(float(count))))

    # onehot[l, n] = (indices[l] == n), as a float32 [length, count] matrix.
    idx_col = ctx.b.op("Reshape", [idx_resolved, ctx.int64_const([length, 1], "shape")])
    arange = ctx.b.const(np.arange(count, dtype=np.float32), "arange")
    arange_row = ctx.b.op("Reshape", [arange, ctx.int64_const([1, count], "shape")])
    not_greater = ctx.b.sub(ctx.b.const(1.0), ctx.mask_greater(idx_col, arange_row))
    not_less = ctx.b.sub(ctx.b.const(1.0), ctx.mask_less(idx_col, arange_row))
    onehot = ctx.b.mul(not_greater, not_less)

    # dData[n, ...] = sum_l onehot[l, n] * dY[l, ...], batched over `pre` with
    # a broadcasting leading axis so one onehot matrix serves every batch
    # element -- see _grad_conv's `wt` for the same trick.
    onehot_t = ctx.b.transpose(onehot, [1, 0])  # [count, length]
    onehot_batched = ctx.b.op(
        "Reshape", [onehot_t, ctx.int64_const([1, count, length], "shape")]
    )
    dy_flat = ctx.b.op("Reshape", [g, ctx.int64_const([pre, length, post], "shape")])
    ddata_flat = ctx.b.matmul(onehot_batched, dy_flat)  # [pre, count, post]
    ddata = ctx.b.op("Reshape", [ddata_flat, ctx.int64_const(data_shape, "shape")])

    return [ddata, None]


def _grad_depth_to_space(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """``DepthToSpace``'s gradient: reshape-transpose-reshape, the exact
    inverse of the op's own documented decomposition -- no scatter, no
    approximation, since the forward op is itself nothing but a
    permutation of elements.

    Real motivation: `nn.PixelShuffle` -- the sub-pixel convolution
    upsampler every real super-resolution architecture surveyed here uses
    (EDSR, and the same shape in CARN/MSRN/RCAN/...) -- exports to ONNX as
    exactly this op, `mode="CRD"`. A real tiny EDSR export found every
    other op it needs already covered on both axes (`Conv`, `Add`, `Mul`,
    `Relu` all have backward rules and are NPU-executable); this was the
    one gap.

    **The identity.** ONNX's own spec defines ``DepthToSpace`` as
    ``reshape(x, tmp_shape)`` -> ``transpose(., perm)`` -> ``reshape(.,
    out_shape)``, with ``tmp_shape``/``perm`` depending on ``mode``:
    ``CRD`` reshapes ``[N, C, H, W]`` to ``[N, C/bs^2, bs, bs, H, W]`` and
    permutes to ``[0, 1, 4, 2, 5, 3]``; ``DCR`` reshapes to ``[N, bs, bs,
    C/bs^2, H, W]`` and permutes to ``[0, 3, 4, 1, 5, 2]``. Since a
    ``Reshape``'s adjoint is a ``Reshape`` back to the original shape and a
    ``Transpose``'s adjoint is a ``Transpose`` by the inverse permutation
    (both already `_grad_reshape`/`_grad_transpose`'s own math, inlined
    here rather than called since the intermediate tensor never otherwise
    exists as a named node), running that chain backward -- reshape ``g``
    into the *post-transpose* shape, transpose by the inverse permutation,
    reshape to ``x``'s own shape -- is ``DepthToSpace``'s exact adjoint.
    Verified directly: a real ``onnxruntime``-executed `DepthToSpace` node
    dot-product-tested against this formula (`sum(g * y)`'s gradient via
    finite differences) agrees to `5.7e-4` (`eps=1e-3` central-difference
    tolerance), not merely derived from the spec text -- the same
    "checked, not assumed" discipline `unroll_lstm`/`unroll_gru`'s own
    gate-order verification used, warranted here too since a permutation
    this easy to get subtly backwards (as `GRU`'s own `linear_before_reset`
    convention was) would fail silently, not loudly.

    Only ``Reshape``/``Transpose`` are emitted, both already in
    :data:`BACKWARD_OPS` -- no widening of that allowlist needed, the same
    property `_grad_split`'s own docstring highlights for its own rule.
    """
    x = node.input[0]
    shape = ctx.shape(x)
    if len(shape) != 4 or not all(isinstance(d, int) for d in shape):
        raise UnsupportedOpError(
            f"DepthToSpace with a non-static or non-rank-4 input shape "
            f"{shape} is not differentiated here (node {node.output[0]!r})"
        )
    n, c, h, w = shape
    bs = int(_attr(node, "blocksize", 0))
    mode = _attr(node, "mode", "DCR")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode == "CRD":
        tmp_shape = [n, c // (bs * bs), bs, bs, h, w]
        perm = [0, 1, 4, 2, 5, 3]
    else:
        tmp_shape = [n, bs, bs, c // (bs * bs), h, w]
        perm = [0, 3, 4, 1, 5, 2]
    transposed_shape = [tmp_shape[p] for p in perm]
    inverse = [0] * 6
    for position, axis in enumerate(perm):
        inverse[axis] = position

    g_t = ctx.b.op("Reshape", [g, ctx.int64_const(transposed_shape, "shape")])
    g_tmp = ctx.b.transpose(g_t, inverse)
    dx = ctx.b.op("Reshape", [g_tmp, ctx.int64_const([n, c, h, w], "shape")])
    return [dx]


def _grad_concat(ctx: _Backward, node: onnx.NodeProto, g: str) -> List[Optional[str]]:
    """``Concat``'s gradient: one ``Gather`` per input, each pulling that
    input's own contiguous slice of ``g`` back out along the concat axis --
    exactly ``Split``'s adjoint (:func:`_grad_split`'s own docstring), just
    reached a different way.

    ``Gather`` rather than ``Split``'s selection-matrix ``MatMul``: this
    module's own admission note for ``Gather`` in :data:`BACKWARD_OPS`
    describes precisely this shape of use -- "a single axis, a constant
    int64 index, and no dependence of the *index* on any runtime value" --
    since every input's offset and width along the concat axis are known at
    build time, not computed from a runtime tensor. `_grad_split` could not
    reuse ``Gather`` for its own, opposite direction (scattering one
    incoming gradient *into* a zero-elsewhere result needs every other
    output's width too, not just its own), which is why it reaches for
    ``Transpose``/``MatMul``/``Add`` instead; here, extracting a contiguous
    range straight out of ``g`` is exactly what ``Gather`` with a
    consecutive-integer index list already does, with nothing left to
    build.

    Every dimension besides the concat axis must be static -- ``Gather``'s
    ``axis`` and index values are baked in at build time, the same
    requirement :func:`_grad_gather` and :func:`_grad_conv` already carry
    for their own ``Gather`` emissions.
    """
    out_shape = ctx.shape(node.output[0])
    axis = int(_attr(node, "axis", 0)) % len(out_shape)
    grads: List[Optional[str]] = []
    offset = 0
    for inp in node.input:
        shape = ctx.shape(inp)
        width = shape[axis]
        if not isinstance(width, int):
            raise UnsupportedOpError(
                f"Concat with a non-static size along axis {axis} is not "
                f"differentiated here (node {node.output[0]!r}, input {inp!r})"
            )
        idx = ctx.int64_const(list(range(offset, offset + width)), "idx")
        grads.append(ctx.b.op("Gather", [g, idx], axis=axis))
        offset += width
    return grads


def _grad_split(
    ctx: _Backward, node: onnx.NodeProto, gs: List[Optional[str]]
) -> List[Optional[str]]:
    """``Split``'s gradient, as one ``MatMul`` per output against a constant
    0/1 selection matrix -- not a ``Concat`` of the incoming gradients, even
    though that is the textbook VJP of a split-along-an-axis.

    **Why not Concat.** It is not in :data:`BACKWARD_OPS` (this module's own
    WebGPU/WebNN/NPU-portable op allowlist -- see that set's own comment),
    and adding an op there is a real, separately-justified decision (see the
    ``Conv``/``ConvTranspose`` note beside :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS`
    for the shape that justification takes) this rule does not need to
    force, because the same result is reachable with what the allowlist
    already has.

    **The identity.** Move the split axis to the last position (a
    ``Transpose``, skipped when it is already there) and "insert output
    ``i``'s gradient at its own offset in a zero tensor the width of the
    input" becomes "right-multiply by a constant selection matrix": for
    output ``i`` with ``k_i`` entries starting at offset ``o_i`` (out of the
    input's full width ``N`` along that axis), that matrix is ``E_i =
    eye(N)[o_i : o_i + k_i, :]``, because ``g_i @ E_i`` places ``g_i``'s
    ``k_i`` columns at columns ``[o_i, o_i + k_i)`` of an ``N``-wide result
    and zero elsewhere -- exactly Split's adjoint (Split is a linear
    projection; this is that projection's transpose). Outputs accumulate by
    ``Add``; an output with no incoming gradient (nothing downstream needs
    it) simply contributes no term rather than an explicit zero, which is
    cheaper and still correct since ``Add`` needs nothing from a term that
    was never there. The final ``Transpose`` (skipped under the same
    condition as the first) restores the original axis order. Only
    ``Transpose``/``MatMul``/``Add`` are used, all already in
    :data:`BACKWARD_OPS` -- the same "arithmetic primitives, not fused ops"
    discipline :func:`_grad_conv`'s own docstring states directly.

    This is a :data:`MultiOutputRule`, not a :data:`Rule` -- ``Split`` is the
    one op type in :data:`_MULTI_OUTPUT_RULES` rather than :data:`_RULES`,
    which is where :func:`build_backward` looks first (see that function and
    :data:`_MULTI_OUTPUT_RULES`'s own comment for why).

    Geometry comes from each output's own *declared shape* (``ctx.shape``)
    rather than parsing ``Split``'s ``split``-as-attribute (opset <13),
    ``split``-as-optional-second-input (opset 13-17) or ``num_outputs``
    attribute (opset 18+) spellings -- the same "derive from what the graph
    actually says the shapes are, not from re-deriving the attribute that
    produced them" discipline :func:`_conv_geometry`/:func:`_pool_geometry`
    already use for exactly this reason (a misread attribute cannot survive
    to become a wrong gradient), and it means this one rule needs no opset
    branching at all.

    Only ``node.input[0]`` (the tensor actually split) ever gets a gradient;
    an optional second ``split`` sizes input (opset 13+) is an integer list,
    not a function of anything float, the same "no gradient for a shape/axes
    operand" convention every other rule with a non-differentiable extra
    input already follows (``Reshape``'s ``shape``, ``Gather``'s ``indices``).

    **What it costs.** Each ``E_i`` is ``k_i * N`` elements, materialized as
    a constant initializer -- small for a modest channel-axis split (the
    Conformer GLU gating this rule exists for splits one axis of a few
    hundred channels into two), real and bounded for a very wide one, the
    same cost :func:`_grad_conv`'s own index/mask tables carry and document
    for the identical reason: no scatter op is in :data:`BACKWARD_OPS`, so a
    constant selection stands in for one.
    """
    x = node.input[0]
    in_shape = ctx.shape(x)
    rank = len(in_shape)
    axis = int(_attr(node, "axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise UnsupportedOpError(
            f"Split's axis {axis} is out of range for a rank-{rank} input "
            f"(node {node.output[0]!r})"
        )
    if all(g is None for g in gs):
        return [None] * len(node.input)

    width = int(in_shape[axis])
    out_sizes = []
    for out_name in node.output:
        out_shape = ctx.shape(out_name)
        if len(out_shape) != rank:
            raise UnsupportedOpError(
                f"Split output {out_name!r} has rank {len(out_shape)}, its "
                f"input {x!r} has rank {rank} (node {node.output[0]!r})"
            )
        out_sizes.append(int(out_shape[axis]))
    if sum(out_sizes) != width:
        raise UnsupportedOpError(
            f"Split's outputs sum to {sum(out_sizes)} along axis {axis}, but "
            f"its input {x!r} has {width} there (node {node.output[0]!r})"
        )

    perm = [i for i in range(rank) if i != axis] + [axis]
    identity_perm = perm == list(range(rank))
    inv_perm = [perm.index(i) for i in range(rank)]

    acc = None
    offset = 0
    for g, size in zip(gs, out_sizes):
        if g is not None:
            g_t = g if identity_perm else ctx.b.transpose(g, perm)
            selector = np.zeros((size, width), dtype=np.float32)
            selector[np.arange(size), offset + np.arange(size)] = 1.0
            term = ctx.b.matmul(g_t, ctx.b.const(selector, "split_sel"))
            acc = term if acc is None else ctx.b.add(acc, term)
        offset += size

    # at least one `g` is non-None (the all-None case already returned above),
    # so the loop above set `acc` at least once.
    assert acc is not None
    dx = acc if identity_perm else ctx.b.transpose(acc, inv_perm)
    return [dx] + [None] * (len(node.input) - 1)


# --- Multi-output rules --------------------------------------------------
#
# :data:`Rule` (and therefore :data:`_RULES`/:data:`SUPPORTED_OPS`) assumes
# exactly one incoming gradient per node -- true of every builtin rule this
# module has ever needed until now, and load-bearing elsewhere:
# ``tests/test_qat_parity.py`` pins ``sorted(SUPPORTED_OPS)`` byte-for-byte
# against a checked-in fixture that also has to match ``qat_entry.cpp``'s own
# hardcoded C++ rule table -- see :data:`SUPPORTED_OPS`'s own comment. A rule
# for an op with more than one *output* (``Split`` is the only one so far --
# GLU gating in a Conformer-style audio-model block, see
# ``docs/axera-audio-speech-op-coverage.md``) genuinely needs a different
# argument shape (one gradient per output, not one), so it is kept in this
# separate table rather than forced into :data:`_RULES`'s contract or
# :data:`_CUSTOM_RULES`'s (which promises the *single-output* :data:`Rule`
# signature to anything that reads it, e.g. a future caller iterating
# ``_CUSTOM_RULES.values()`` expecting to call each with one ``g``).
#
# The direct consequence: an op registered here is **not** part of
# :data:`SUPPORTED_OPS` and has **no C++/WASM mirror** -- differentiating it
# is a Python-only capability today, the same divergence
# :func:`register_gradient`'s docstring already warns ``override=True``
# causes for a *builtin* op, just reached by a different door. It *is*
# visible through :func:`supported_ops`, so QAT/LoRA block discovery
# (``onnxsim.qat.discover_qat_blocks``/``onnxsim.lora.discover_lora_blocks``,
# both already keyed off ``supported_ops()`` rather than the raw constant)
# correctly treats a block containing it as differentiable.
_MULTI_OUTPUT_RULES: Dict[str, MultiOutputRule] = {
    "Split": _grad_split,
}


# --- Templated rules ---------------------------------------------------
#
# An alternative to hand-transcribing a rule's graph construction directly in
# Python (and, separately, a second time in graph_grad.cpp): author the rule
# once in onnxscript, compile it offline to a checked-in ONNX FunctionProto
# (scripts/codegen/generate_grad_templates.py ->
# onnxsim.graph_grad_templates_gen), and instantiate the same text from
# either language via ONNX's own function-inlining machinery. onnxscript
# itself is never imported here -- only onnx.parser, already a base
# dependency, to read the checked-in text back into a FunctionProto.
#
# Wired into :data:`_RULES` below for "Add" and "BatchNormalization" -- these
# were a proof of concept (see tests/test_graph_grad_templates.py, which
# checks GradBatchNormalization against torch.autograd) before graduating to
# production. `_grad_add`/`_grad_batch_normalization` above are no longer
# reachable through :data:`_RULES`, but are kept, deliberately, as an
# independent reference implementation: tests/test_graph_grad_templates.py
# still cross-checks the templated rule's numbers against them on the same
# inputs, which is exactly the kind of regression check that caught this
# repo's own dvar-derivation bug in the first place and would otherwise be
# lost by deleting the hand-written code.


@functools.lru_cache(maxsize=None)
def _load_template(text: str) -> onnx.FunctionProto:
    return onnx.parser.parse_function(text)


def _grad_add_templated(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """Same shape as :func:`_grad_add`, but the identity-gradient core
    (``da = db = g``) comes from a call to the checked-in ``GradAdd``
    function instead of being written here directly. Broadcast-undoing
    stays outside the template, exactly as it does for the hand-written
    rule -- see graph_grad_templates_gen's module docstring for why."""
    fn = _load_template(_templates.GRAD_ADD)
    da, db = ctx.b.call(fn, [g])
    out = ctx.shape(node.output[0])
    return [
        ctx.reduce_to(da, out, ctx.shape(node.input[0])),
        ctx.reduce_to(db, out, ctx.shape(node.input[1])),
    ]


def _grad_batch_normalization_templated(
    ctx: _Backward, node: onnx.NodeProto, g: str
) -> List[Optional[str]]:
    """Same validation and shape-resolution as
    :func:`_grad_batch_normalization` (rank/shape checks, the per-channel
    broadcast reshape, the reduction axes), but the actual gradient
    arithmetic comes from a call to the checked-in
    ``GradBatchNormalization`` function -- the rule this repo's own
    dvar-derivation bug was in, so the one most worth proving out this way
    first."""
    x = node.input[0]
    scale, bias = node.input[1], node.input[2]
    mean, var = node.input[3], node.input[4]
    name = node.output[0]
    x_shape = ctx.shape(x)
    rank = len(x_shape)
    if rank < 2:
        raise UnsupportedOpError(
            f"BatchNormalization needs a batch and a channel axis, got "
            f"input shape {x_shape} (node {name!r})"
        )
    channels = int(x_shape[1])
    for label, tensor in (
        ("scale", scale),
        ("B", bias),
        ("mean", mean),
        ("var", var),
    ):
        shape = ctx.shape(tensor)
        if tuple(shape) != (channels,):
            raise UnsupportedOpError(
                f"BatchNormalization's {label} has shape {tuple(shape)}, not "
                f"({channels},) (node {name!r})"
            )
    eps = float(_attr(node, "epsilon", 1e-5))
    bshape = (1, channels) + (1,) * (rank - 2)

    def bcast(t: str) -> str:
        return ctx.b.op("Reshape", [t, ctx.int64_const(bshape, "shape")])

    channel_axes = ctx.int64_const([0] + list(range(2, rank)), "axes")
    fn = _load_template(_templates.GRAD_BATCH_NORMALIZATION)
    dx, dscale, dbias, dmean, dvar = ctx.b.call(
        fn,
        [
            g,
            x,
            bcast(mean),
            bcast(var),
            bcast(scale),
            ctx.b.const(eps),
            channel_axes,
            ctx.b.const(1.0),
            ctx.b.const(-0.5),
        ],
    )
    return [dx, dscale, dbias, dmean, dvar]


_RULES: Dict[str, Rule] = {
    "Add": _grad_add_templated,
    "AveragePool": _grad_averagepool,
    "BatchNormalization": _grad_batch_normalization_templated,
    "Clip": _grad_clip,
    "Conv": _grad_conv,
    "Div": _grad_div,
    "Erf": _grad_erf,
    "Exp": _grad_exp,
    "Gather": _grad_gather,
    "Gemm": _grad_gemm,
    "Identity": _grad_identity,
    "InstanceNormalization": _grad_instance_normalization,
    "LayerNormalization": _grad_layer_normalization,
    "Log": _grad_log,
    "MatMul": _grad_matmul,
    "MaxPool": _grad_maxpool,
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

#: Single-output rules kept out of :data:`_RULES` for the same reason
#: :data:`_MULTI_OUTPUT_RULES` is its own table (see that table's own
#: comment): an op registered here has **no C++/WASM mirror** yet, so
#: folding it into :data:`_RULES` would silently widen the parity-pinned
#: :data:`SUPPORTED_OPS` that ``tests/test_qat_parity.py`` checks against
#: ``qat_entry.cpp``'s hardcoded C++ rule table. Unlike
#: :data:`_MULTI_OUTPUT_RULES`, every rule here is an ordinary
#: single-``g`` :data:`Rule` -- ``Where``/``IsNaN`` (numerical-stability
#: masking in wav2vec2's own attention output, see
#: ``docs/axera-audio-speech-op-coverage.md``), ``Concat`` (needed for
#: `scripts/axera/legalize.py`'s `unroll_lstm`/`unroll_gru` to differentiate
#: through their own stacked-timestep sequence output), ``Squeeze``/
#: ``Unsqueeze`` (a real NVIDIA Parakeet decoder export has a bare
#: ``Squeeze`` between its two LSTM layers, unrelated to `unroll_lstm`
#: itself), and ``DepthToSpace`` (`nn.PixelShuffle`'s ONNX form -- the
#: sub-pixel convolution upsampler every super-resolution architecture
#: surveyed here uses) each have exactly one output -- so the *only*
#: reason they are not simply in :data:`_RULES` is the missing C++ port,
#: not a signature mismatch. :func:`build_backward` merges this table into
#: its default rule set right alongside :data:`_CUSTOM_RULES`, and
#: :func:`supported_ops` includes it, so QAT/LoRA block discovery correctly
#: treats a block containing any of them as differentiable.
_PYTHON_ONLY_RULES: Dict[str, Rule] = {
    "Concat": _grad_concat,
    "DepthToSpace": _grad_depth_to_space,
    "IsNaN": _grad_is_nan,
    "Squeeze": _grad_squeeze_or_unsqueeze,
    "Unsqueeze": _grad_squeeze_or_unsqueeze,
    "Where": _grad_where,
}

# The op types :func:`build_backward` can differentiate. Callers that pick
# the slice themselves -- block discovery for QAT, say -- should test against
# this rather than rediscovering the list by catching
# :class:`UnsupportedOpError`.
#
# Deliberately the *builtin* set alone, unaffected by :func:`register_gradient`
# -- tests/test_qat_parity.py pins ``sorted(SUPPORTED_OPS)`` byte-for-byte
# against qat_parity_fixtures.txt (the Python<->C++ parity mechanism), and
# that pin must not shift just because some other test in the same process
# registered a custom rule. Callers that pick their own block boundaries and
# want custom-registered ops included in that boundary -- not just accepted
# once :func:`build_backward` is actually called -- should test against
# :func:`supported_ops` instead; :func:`onnxsim.qat._refuse_unsupported` and
# :func:`onnxsim.qat.discover_qat_blocks`/:func:`onnxsim.lora.discover_lora_blocks`
# all do.
SUPPORTED_OPS = frozenset(_RULES)

#: Custom gradient rules registered via :func:`register_gradient`, kept in a
#: table separate from the builtin, parity-pinned :data:`_RULES` for exactly
#: the reason :data:`SUPPORTED_OPS` above stays builtin-only. Process-global
#: -- see :func:`register_gradient`'s docstring for the test-isolation
#: implications, and :func:`custom_gradient` for a scoped alternative.
_CUSTOM_RULES: Dict[str, Rule] = {}


def supported_ops() -> frozenset:
    """:data:`SUPPORTED_OPS` (the builtin, parity-pinned single-output rules)
    unioned with every op type currently registered via
    :func:`register_gradient`, every op type in :data:`_MULTI_OUTPUT_RULES`
    (``Split`` today), and every op type in :data:`_PYTHON_ONLY_RULES`
    (``IsNaN``/``Where`` today). This is the set :func:`build_backward`
    (called the ordinary way, with ``rules=None``) can actually
    differentiate right now."""
    return (
        SUPPORTED_OPS
        | frozenset(_CUSTOM_RULES)
        | frozenset(_MULTI_OUTPUT_RULES)
        | frozenset(_PYTHON_ONLY_RULES)
    )


def register_gradient(
    op_type: str, rule: Optional[Rule] = None, *, override: bool = False
):
    """Registers ``rule`` as the gradient for ``op_type``, usable by every
    caller of :func:`build_backward` that leaves its ``rules`` argument at
    the default ``None`` -- which includes every public onnxsim entry point
    that differentiates a block (:func:`onnxsim.apply_qat`,
    :func:`onnxsim.apply_qat_all_blocks`, :func:`onnxsim.train_lora`, and
    everything built on :func:`onnxsim.qat._refuse_unsupported`'s block-scope
    check, which now tests against :func:`supported_ops` rather than the
    builtin-only :data:`SUPPORTED_OPS`). The relationship to the caller is
    the same one :func:`torch.autograd.Function.backward` has to a custom
    torch op: write the vector-Jacobian product once, register it, and every
    later differentiation of that op type uses it with no further plumbing.

    Works as a decorator::

        @graph_grad.register_gradient("MyCustomOp")
        def _grad_my_custom_op(ctx, node, g):
            ...
            return [dx, dy]  # one entry per node.input, None where there is
                              # no gradient for that input (Reshape's shape
                              # operand is the builtin example)

    or as a direct call: ``graph_grad.register_gradient("MyCustomOp", rule)``.

    ``rule`` must match :data:`Rule`'s signature -- ``(ctx, node, g) ->
    List[Optional[str]]``, exactly what a builtin rule in :data:`_RULES`
    looks like; see any of them (``_grad_relu`` is the simplest) for the
    shape of ``ctx`` (a :class:`_Backward`, offering ``ctx.b`` -- the
    :class:`onnxsim.qat_graph.GraphBuilder` to append new nodes to -- plus
    ``ctx.shape``/``ctx.reduce_to``/``ctx.int64_const`` and friends).
    :func:`build_backward` asserts the returned list is the same length as
    ``node.input``; nothing here can check that in advance since it would
    mean calling the rule speculatively.

    Refuses to silently replace an existing rule -- builtin or a previously
    registered custom one -- unless ``override=True``, the same "a surprising
    collision is refused, not resolved by whichever registration happened to
    run last" stance :class:`UnsupportedOpError` itself takes toward an
    unrecognized op. Overriding a *builtin* op's rule changes differentiation
    for Python callers only: it has no effect on the C++/WASM path (the
    browser QAT/LoRA panels), whose step graphs come from ``qat_entry.cpp``'s
    own hardcoded, unregistrable rule table -- so a block containing that op
    type will train differently in the browser than it does here, a real and
    easy-to-miss divergence worth thinking twice about before reaching for
    ``override=True`` on anything already in :data:`SUPPORTED_OPS`.

    Global and *not* automatically cleaned up -- a test that registers a rule
    and does not call :func:`unregister_gradient` (or use
    :func:`custom_gradient` instead) leaves it registered for every test that
    runs afterward in the same process. Prefer :func:`custom_gradient` for
    anything scoped to one test or one call.

    :raises ValueError: if ``op_type`` already has a rule (builtin or
            custom) and ``override`` is not set.
    """

    def _register(fn: Rule) -> Rule:
        if not override:
            if op_type in _RULES:
                raise ValueError(
                    f"{op_type!r} already has a builtin gradient rule; pass "
                    "override=True to replace it (this affects Python-side "
                    "differentiation only -- see register_gradient's own "
                    "docstring for the C++/WASM divergence that implies)"
                )
            if op_type in _PYTHON_ONLY_RULES:
                raise ValueError(
                    f"{op_type!r} already has a builtin (Python-only) "
                    "gradient rule; pass override=True to replace it (this "
                    "affects Python-side differentiation only)"
                )
            if op_type in _CUSTOM_RULES:
                raise ValueError(
                    f"{op_type!r} already has a custom gradient rule "
                    "registered; pass override=True to replace it, or call "
                    "unregister_gradient(op_type) first"
                )
        _CUSTOM_RULES[op_type] = fn
        return fn

    return _register if rule is None else _register(rule)


def unregister_gradient(op_type: str) -> None:
    """Removes a rule :func:`register_gradient` registered for ``op_type``.
    Builtin rules (:data:`_RULES`) are never affected -- there is nothing to
    unregister for an op :func:`register_gradient` was never used on.

    :raises KeyError: if no custom rule is currently registered for
            ``op_type``.
    """
    if op_type not in _CUSTOM_RULES:
        raise KeyError(f"no custom gradient rule is registered for {op_type!r}")
    del _CUSTOM_RULES[op_type]


@contextlib.contextmanager
def custom_gradient(
    op_type: str, rule: Rule, *, override: bool = False
) -> Iterator[None]:
    """Scoped form of :func:`register_gradient`: registers ``rule`` for
    ``op_type`` on entry and unregisters it on exit (success or exception
    alike), so a test or a one-off call cannot leak a registration into
    whatever else shares this process -- the leak risk :func:`register_gradient`'s
    own docstring warns about. ::

        with graph_grad.custom_gradient("MyCustomOp", my_rule):
            tuned = onnxsim.apply_qat(float_model, quant_model, ...)
    """
    register_gradient(op_type, rule, override=override)
    try:
        yield
    finally:
        unregister_gradient(op_type)


def build_backward(
    b: qat_graph.GraphBuilder,
    nodes: Sequence[onnx.NodeProto],
    shapes: Dict[str, Sequence[Union[int, str]]],
    grad_outputs: Dict[str, str],
    targets: Sequence[str],
    rules: Optional[Dict[str, Rule]] = None,
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
    :param rules: replaces the *entire* rule table this call uses, bypassing
            :func:`register_gradient` registrations altogether. Exists for
            :mod:`tests.test_graph_grad_templates`, which builds a copy with
            ``Add``/``BatchNormalization`` pinned to the reference
            hand-written rule (:func:`_grad_add`/
            :func:`_grad_batch_normalization`) instead of the templated one
            :data:`_RULES` uses by default, as an independent numeric
            cross-check -- ordinary callers should omit this. Left at the
            default ``None``, this call uses :data:`_RULES` plus any rules
            registered via :func:`register_gradient`, so a rule registered
            once is picked up here with no further plumbing -- the same
            "register once, every later differentiation of that op uses it"
            relationship :func:`torch.autograd.Function.backward` has to a
            custom torch op.
    """
    rules = (
        dict(_RULES, **_PYTHON_ONLY_RULES, **_CUSTOM_RULES) if rules is None else rules
    )
    ctx = _Backward(b, shapes)
    grads: Dict[str, str] = dict(grad_outputs)

    def accumulate(node: onnx.NodeProto, contributions: List[Optional[str]]) -> None:
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

    for node in reversed(list(nodes)):
        multi_rule = _MULTI_OUTPUT_RULES.get(node.op_type)
        if multi_rule is not None:
            # A different dispatch path from the single-output one below:
            # every one of this node's *outputs* may carry its own incoming
            # gradient (or none), not just node.output[0] -- see
            # :data:`_MULTI_OUTPUT_RULES`'s own comment for why this table is
            # separate from :data:`_RULES`/:data:`_CUSTOM_RULES` rather than
            # widening :data:`Rule`'s single-``g`` contract for everything.
            gs = [grads.get(o) for o in node.output]
            if all(g is None for g in gs):
                # Nothing downstream depends on any output, so every
                # gradient this node would produce is zero -- same "skip
                # rather than emit waste" reasoning as the single-output
                # path below, generalized to "none of the outputs" instead
                # of "the one output".
                continue
            accumulate(node, multi_rule(ctx, node, gs))
            continue

        rule = rules.get(node.op_type)
        if rule is None:
            raise UnsupportedOpError(
                f"no gradient rule for op type {node.op_type!r} "
                f"(node {node.name or node.output[0]!r}); "
                f"onnxsim.graph_grad differentiates {sorted(rules) + sorted(_MULTI_OUTPUT_RULES)}"
            )
        if len(node.output) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} has {len(node.output)} outputs; only "
                "single-output nodes are differentiated here (unless "
                "registered in _MULTI_OUTPUT_RULES, checked above)"
            )
        g = grads.get(node.output[0])
        if g is None:
            # Nothing downstream depends on this node, so every gradient it
            # would produce is zero. Emitting those zeros would be correct
            # and pure waste.
            continue
        accumulate(node, rule(ctx, node, g))

    result: Dict[str, str] = {}
    for target in targets:
        if target not in grads:
            raise ValueError(
                f"no gradient reaches {target!r}: it is not downstream of any of "
                f"{sorted(grad_outputs)} within the given nodes"
            )
        result[target] = grads[target]
    return result
