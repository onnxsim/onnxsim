"""TIR pass that makes TVM emit chained HVX qfloat multiply-accumulate code on Hexagon.

Hexagon's LLVM backend converts qf32/qf16 <-> IEEE around *every* fp32/fp16 operation
(`v.qf32 = vmpy(a.sf, b.sf)`, `t.sf = v.qf32`, `v.qf32 = vadd(acc.sf, t.sf)`, `acc.sf = v.qf32`,
~4 HVX ops per multiply-add) and `-fast-math` does not change that. Hand-written kernels avoid it
by keeping accumulators in qfloat form and calling `vmpy.qf32.sf` + `vadd.qf32` directly.

This pass does that automatically from an ordinary vectorized TE/TIR reduction. After
vectorization and unrolling an accumulator is a vector-typed buffer updated as

    acc[i] = acc[i] + A * B

The pass finds such accumulator buffers (every store is a zero initialiser or a multiply-accumulate
update), rewrites the updates into qfloat intrinsic chains, and wraps every other read of the
buffer in the matching `vconv`. Supported accumulators:

    float32x32  acc += float32x32 * float32x32                -> vmpy.qf32.sf + vadd.qf32
    float32x64  acc += f32(float16x64) * f32(float16x64)      -> vmpy.qf32.hf (widening pair)
                                                                 + 2x vadd.qf32
    float16x64  acc += float16x64 * float16x64                -> vmpy.qf16.hf + vadd.qf16
    float32x64  acc += f32(partial)   (partial: a qf16 accumulator above)
                                                              -> vmpy.qf32.qf16 by qf16 1.0
                                                                 + 2x vadd.qf32

Scalar operands broadcast into a vector (`Broadcast(x, L)`, also through a widening `Cast`) are
recognised. Use `build()` to apply the pass and to declare the 128-byte buffer alignment HVX
loads need (otherwise LLVM emits `vmem` + `valign` pairs).
"""

from __future__ import annotations

import tvm
from tvm import tir

_HVX_INT = "int32x32"
_HVX_PAIR = "int32x64"

# Vector dtype of the accumulator -> update kind.
_KINDS = {"float32x32": "qf32", "float32x64": "qf32w", "float16x64": "qf16"}


def _intrin(dtype, name, *args):
    return tir.call_llvm_intrin(
        dtype, f"llvm.hexagon.V6.{name}.128B", tir.const(len(args), "uint32"), *args
    )


def _bits(expr, dtype=_HVX_INT):
    return tir.reinterpret(dtype, expr)


def _is_zero_init(value):
    return (
        isinstance(value, tir.Broadcast)
        and isinstance(value.value, (tir.FloatImm, tir.IntImm))
        and float(value.value.value) == 0.0
    )


def _same_buffer(a, b):
    return a.data.same_as(b.data)


def _split_update(store):
    """Return (A, B) if `store` is `acc[i] = acc[i] + A * B` (either operand order), else None."""
    value = store.value
    if not isinstance(value, tir.Add):
        return None
    for load, product in ((value.a, value.b), (value.b, value.a)):
        if (
            isinstance(load, tir.BufferLoad)
            and _same_buffer(load.buffer, store.buffer)
            and len(load.indices) == len(store.indices)
            and all(tvm.ir.structural_equal(x, y) for x, y in zip(load.indices, store.indices))
            and isinstance(product, tir.Mul)
        ):
            return product.a, product.b
    return None


def _split_sum(store):
    """Return the partial-sum load X if `store` is `acc[i] = acc[i] + f32(X)`, else None."""
    value = store.value
    if not isinstance(value, tir.Add):
        return None
    for load, other in ((value.a, value.b), (value.b, value.a)):
        if (
            isinstance(load, tir.BufferLoad)
            and _same_buffer(load.buffer, store.buffer)
            and len(load.indices) == len(store.indices)
            and all(tvm.ir.structural_equal(x, y) for x, y in zip(load.indices, store.indices))
            and isinstance(other, tir.Cast)
            and other.dtype == "float32x64"
            and isinstance(other.value, tir.BufferLoad)
            and other.value.dtype == "float16x64"
        ):
            return other.value
    return None


def _as_half_vector(expr, lanes):
    """Normalise a widening operand to a float16xL expression, or None."""
    if isinstance(expr, tir.Cast) and expr.dtype == f"float32x{lanes}":
        inner = expr.value
        if inner.dtype == f"float16x{lanes}":
            return inner
        return None
    if isinstance(expr, tir.Broadcast) and expr.lanes == lanes:
        inner = expr.value
        if isinstance(inner, tir.Cast) and inner.dtype == "float32" and inner.value.dtype == "float16":
            return tir.Broadcast(inner.value, lanes)
    return None


def _update_operands(kind, product_ab):
    """Validate/normalise the multiply operands for `kind`; returns (A, B) or None."""
    a, b = product_ab
    if kind == "qf32":
        if a.dtype == b.dtype == "float32x32":
            return a, b
    elif kind == "qf16":
        if a.dtype == b.dtype == "float16x64":
            return a, b
    else:  # qf32w
        ha, hb = _as_half_vector(a, 64), _as_half_vector(b, 64)
        if ha is not None and hb is not None:
            return ha, hb
    return None


def _rewrite_update(kind, store, a, b):
    buf, indices = store.buffer, store.indices
    load = tir.BufferLoad(buf, indices)
    if kind == "qf32":
        product = _intrin(_HVX_INT, "vmpy.qf32.sf", _bits(a), _bits(b))
        total = _intrin(_HVX_INT, "vadd.qf32", _bits(load), product)
        value = _bits(total, "float32x32")
    elif kind == "qf16":
        product = _intrin(_HVX_INT, "vmpy.qf16.hf", _bits(a), _bits(b))
        total = _intrin(_HVX_INT, "vadd.qf16", _bits(load), product)
        value = _bits(total, "float16x64")
    else:
        pair = tir.Var("pair", _HVX_PAIR)  # bind the widening product once for both halves
        acc = _bits(load, _HVX_PAIR)
        halves = [
            _intrin(_HVX_INT, "vadd.qf32", part(_HVX_INT, acc), part(_HVX_INT, pair))
            for part in (tir.op.vectorlow, tir.op.vectorhigh)
        ]
        combined = _bits(tir.op.vectorcombine(_HVX_PAIR, halves[0], halves[1]), "float32x64")
        value = tir.Let(pair, _intrin(_HVX_PAIR, "vmpy.qf32.hf", _bits(a), _bits(b)), combined)
    return tir.BufferStore(buf, value, indices)


def _rewrite_sum(store, partial_load):
    """acc(float32x64 pair) += widen(qf16 partial): multiply the partial by qf16 1.0."""
    one_hf = _intrin(_HVX_INT, "lvsplath", tir.const(0x3C00, "int32"))
    one_qf16 = _intrin(_HVX_INT, "vmpy.qf16.hf", one_hf, one_hf)
    wide = tir.Var("wide", _HVX_PAIR)
    acc = _bits(tir.BufferLoad(store.buffer, store.indices), _HVX_PAIR)
    halves = [
        _intrin(_HVX_INT, "vadd.qf32", part(_HVX_INT, acc), part(_HVX_INT, wide))
        for part in (tir.op.vectorlow, tir.op.vectorhigh)
    ]
    combined = _bits(tir.op.vectorcombine(_HVX_PAIR, halves[0], halves[1]), "float32x64")
    value = tir.Let(
        wide, _intrin(_HVX_PAIR, "vmpy.qf32.qf16", _bits(partial_load), one_qf16), combined
    )
    return tir.BufferStore(store.buffer, value, store.indices)


def _convert_load(kind, load):
    """Turn a qfloat-domain accumulator read back into an ordinary IEEE vector."""
    if kind == "qf32":
        return _bits(_intrin(_HVX_INT, "vconv.sf.qf32", _bits(load)), "float32x32")
    if kind == "qf16":
        return _bits(_intrin(_HVX_INT, "vconv.hf.qf16", _bits(load)), "float16x64")
    pair = _bits(load, _HVX_PAIR)
    halves = [
        _bits(_intrin(_HVX_INT, "vconv.sf.qf32", part(_HVX_INT, pair)), "float32x32")
        for part in (tir.op.vectorlow, tir.op.vectorhigh)
    ]
    # The widening multiply de-interleaves lanes (evens low, odds high): restore natural order.
    order = [i // 2 + 32 * (i % 2) for i in range(64)]
    return tir.Shuffle(halves, order)


def _eligible_buffers(body):
    """Map buffer data var -> kind for accumulators whose every store is init or update."""
    stores = {}

    def visit(node):
        if isinstance(node, tir.BufferStore):
            stores.setdefault(node.buffer.data, []).append(node)

    tir.stmt_functor.post_order_visit(body, visit)

    def classify(group, kind, partials):
        updates = 0
        for store in group:
            if _is_zero_init(store.value):
                continue
            split = _split_update(store)
            if split is not None and _update_operands(kind, split) is not None:
                updates += 1
                continue
            source = _split_sum(store) if kind == "qf32w" else None
            if source is not None and source.buffer.data in partials:
                updates += 1
                continue
            return False
        return updates > 0

    eligible = {}
    # Phase 1: multiply-accumulate accumulators.
    for var, group in stores.items():
        kind = _KINDS.get(str(group[0].value.dtype))
        if kind is None or any(str(s.value.dtype) != str(group[0].value.dtype) for s in group):
            continue
        if classify(group, kind, ()):
            eligible[var] = kind
    # Phase 2: fp32 totals that accumulate qf16 partial sums (only reads of eligible qf16 buffers).
    partials = {v for v, k in eligible.items() if k == "qf16"}
    for var, group in stores.items():
        if var in eligible or str(group[0].value.dtype) != "float32x64":
            continue
        if any(str(s.value.dtype) != "float32x64" for s in group):
            continue
        if classify(group, "qf32w", partials):
            eligible[var] = "qf32w"
    return eligible


def _transform(func):
    eligible = _eligible_buffers(func.body)
    if not eligible:
        return func

    def preorder(node):
        if isinstance(node, tir.BufferStore) and node.buffer.data in eligible:
            split = _split_update(node)
            kind = eligible[node.buffer.data]
            if split is not None and _update_operands(kind, split) is not None:
                a, b = _update_operands(kind, split)
                return _rewrite_update(kind, node, a, b)
            source = _split_sum(node)
            if source is not None:
                return _rewrite_sum(node, source)
        return None

    def postorder(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data in eligible:
            return _convert_load(eligible[node.buffer.data], node)
        return None

    body = tir.stmt_functor.ir_transform(
        func.body, preorder, postorder, ["tir.BufferStore", "tir.BufferLoad"]
    )
    return func.with_body(body)


def qfloat_accumulate_pass():
    """The pass; register at lowering phase 2 (after vectorization, before codegen)."""
    return tir.transform.prim_func_pass(
        lambda func, _mod, _ctx: _transform(func), opt_level=0, name="HexagonQFloatAccumulate"
    )


def build(schedule, args, target, name="main", alignment=128, enable_pass=True):
    """`tvm.build` with the qfloat pass and HVX-aligned argument buffers."""
    # offset_factor=0 pins elem_offset to 0; a symbolic offset hides the alignment from LLVM.
    binds = {
        t: tir.decl_buffer(t.shape, t.dtype, name=t.op.name, data_alignment=alignment, offset_factor=0)
        for t in args
    }
    config = {"tir.add_lower_pass": [(2, qfloat_accumulate_pass())]} if enable_pass else {}
    with tvm.transform.PassContext(config=config):
        return tvm.build(schedule, list(args), target=target, name=name, binds=binds)
