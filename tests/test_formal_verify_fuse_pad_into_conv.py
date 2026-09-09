"""Formal check for FusePadIntoConv (fuse_pad_into_conv.h).

The rewrite requires the preceding Pad to use ``mode="constant"`` with a
value of exactly 0, no padding on the N/C axes, non-negative pad amounts,
and ``auto_pad`` absent or ``"NOTSET"`` -- all side conditions guarding the
same underlying assumption: the Pad node contributes only *zero-fill* on the
*spatial* axes. Under that assumption it merges the Pad's begin/end amounts
into Conv's own ``pads`` attribute elementwise (additively -- a pre-existing
nonzero Conv ``pads`` is not replaced, it is added to), for every spatial
axis, and rewires Conv's input directly to Pad's input.

Soundness (per spatial axis, since axes are independent): convolving with a
constant-0 Pad in front is the same zero-fill window read as convolving with
combined pads directly, because both scenarios read a sample at the same
absolute input position and return 0 in exactly the same out-of-bounds
cases. Modeling that shared zero-fill rule once (``_read`` below) and
instantiating it for both the two-step (Pad then Conv-with-original-pads)
and single-step (Conv-with-combined-pads) formulas turns the pass's "add the
pad amounts" claim into an arithmetic identity Z3 can check outright. This
proves the per-output-element value formula for one spatial axis; the
matching output *length* is separate (and simpler) integer arithmetic --
both scenarios sum the same two pad amounts into the convolution's total
padding -- and isn't re-derived here.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser

_KERNEL_SIZE = 3


def _read(x, length, idx):
    # Zero-fill boundary condition shared by Pad (mode="constant", value=0)
    # and by a Conv's own implicit zero-fill outside its input, for a
    # symbolic 1-D input of the given length.
    return z3.If(z3.And(idx >= 0, idx < length), x(idx), z3.RealVal(0))


def test_fuse_pad_into_conv_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    length, pad_begin, conv_pad_begin, p = z3.Ints("length pad_begin conv_pad_begin p")
    domain = z3.And(length > 0, pad_begin >= 0, conv_pad_begin >= 0)

    def conv_at(read_at, pad_begin_amount, position):
        return sum(
            read_at(position - pad_begin_amount + k) for k in range(_KERNEL_SIZE)
        )

    def padded_read(j):
        return _read(x, length, j - pad_begin)

    def direct_read(j):
        return _read(x, length, j)

    # Two-step: Pad(X, pad_begin) materializes a zero-padded array; Conv then
    # applies its own (pre-existing) conv_pad_begin against *that* array.
    two_step = conv_at(padded_read, conv_pad_begin, p)
    # Single-step: fuse_pad_into_conv's own formula -- combined pad is the
    # elementwise sum -- applied directly against X.
    direct = conv_at(direct_read, pad_begin + conv_pad_begin, p)

    prove(z3.Implies(domain, two_step == direct))


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_pad_into_conv_pass_matches_with_existing_conv_pads():
    # Conv already declares its own nonzero `pads` here (unlike this repo's
    # existing test_fuse_pad_into_conv in test_fusion_patterns.py, which
    # leaves Conv's pads at the all-zero default) -- this specifically
    # exercises fuse_pad_into_conv.h's additive-merge branch rather than the
    # zero-plus-zero case.
    W = np.random.default_rng(0).standard_normal((8, 3, 3, 3))
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,8,18,18] Y)
        <int64[8] node_pads = {0, 0, 1, 1, 0, 0, 1, 1}>
        {
          p = Pad<mode = "constant">(X, node_pads)
          Y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(p, W)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W")])
    sim_model, ops = simplify_isolated(model, "fuse_pad_into_conv")
    assert ops["Pad"] == 0
    assert ops["Conv"] == 1

    conv_node = next(n for n in sim_model.graph.node if n.op_type == "Conv")
    pads = next(a.ints for a in conv_node.attribute if a.name == "pads")
    # Elementwise sum of the Pad node's spatial pads and Conv's own
    # pre-existing pads -- fuse_pad_into_conv.h's own additive-merge formula.
    assert list(pads) == [2, 2, 2, 2]
