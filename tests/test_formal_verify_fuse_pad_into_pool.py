"""Formal check for FusePadIntoPool (fuse_pad_into_pool.h).

The rewrite requires the preceding Pad to use ``mode="constant"`` (or have no
``mode`` at all, which defaults to it), no padding on the N/C axes, and
non-negative pad amounts -- guarding the same underlying assumption
``fuse_pad_into_conv`` relies on: the Pad node contributes only *fill-value*
padding on the *spatial* axes, at exactly the fill value Pool's own implicit
padding already reads outside its input. Under that assumption it merges the
Pad's begin/end amounts into Pool's own ``pads`` attribute elementwise
(additively -- a pre-existing nonzero Pool ``pads`` is not replaced, it is
added to), for every spatial axis, and rewires Pool's input directly to
Pad's input.

Soundness (per spatial axis, since axes are independent): reading through a
constant-fill Pad in front of Pool's own padding is the same fill-value
window read as reading directly with the combined pads, because both
scenarios read a sample at the same absolute input position and return the
*same* fill constant in exactly the same out-of-bounds cases. Modeling that
shared fill rule once (``_read`` below) and instantiating it for both the
two-step (Pad then Pool-with-original-pads) and single-step
(Pool-with-combined-pads) formulas turns the pass's "add the pad amounts"
claim into an arithmetic identity Z3 can check outright -- exactly
``test_formal_verify_fuse_pad_into_conv.py``'s own technique, generalized
from a hardcoded fill of 0 to an arbitrary shared fill constant ``c`` (see
below for why Pool needs that generalization where Conv did not). This
proves the per-output-element *input* formula for one spatial axis -- i.e.
that the value Pool's window reads at each position is identical either way
-- which is all the soundness argument needs: the pass does not touch
Pool's own windowed reduction (sum-then-divide for AveragePool, max for
MaxPool), so whatever that downstream computation does with the (now
provably identical) padded input, both scenarios do it identically. Output
*length* is separate (and simpler) integer arithmetic -- both scenarios sum
the same two pad amounts into Pool's total padding -- and isn't re-derived
here.

Why a shared constant ``c`` rather than a hardcoded 0 (unlike Conv's proof):
Conv has no notion of "the value it implicitly reads outside its input" --
its own zero-padding is *always* 0, so Conv's fuse only ever needs to check
that Pad's fill is 0 too. Pool is different: this repo's compiled pass is
onnxsim's own patched ``FusePadIntoPool`` (registered over the vanilla
onnxoptimizer submodule's same-named pass via ``RegisterOrReplace`` in
``custom_optimizer_passes.cpp`` -- see ``onnxsim/passes/fuse_pad_into_pool.h``,
which is what actually runs, not ``third_party/onnx-optimizer``'s copy), and
it requires Pad's constant fill to match *the fill value Pool's own padding
uses*, which differs by op:

- AveragePool, once fused, always gets ``count_include_pad=1`` forced on
  (see below) -- which makes its own implicit padding read as plain 0 -- so
  Pad's fill must be 0.
- MaxPool has no ``count_include_pad`` notion; it simply never selects an
  out-of-bounds/padded position as the max, which is equivalent to reading
  -inf there. So Pad's fill must be -inf, *not* 0 -- confirmed empirically
  below (this is onnxsim's fix for
  https://github.com/onnxsim/onnxsim/issues/290: the *vanilla*
  onnxoptimizer pass this repo's differs from would happily fuse a
  zero-filling Pad into MaxPool, which is unsound whenever a window's real
  values are all negative, since 0 > every real value there but -inf is
  not).

Both are instances of the same "Pad's fill equals Pool's own implicit-pad
fill" lemma, which is what ``_read``/``c`` below prove once, generically.

``count_include_pad`` and AveragePool: this attribute is not part of the
index-algebra proof above (the proof is only about which *input* value a
read returns, not how Pool's reduction consumes it), but it is a necessary
side effect for AveragePool's soundness specifically. AveragePool's spec
says padded positions are *excluded* from the averaging divisor by default
(``count_include_pad=0``); after fusing, the positions the pass folds in
were produced by Pad as genuine (if zero-valued) tensor elements, which
*must* count in the divisor like any other element. Setting
``count_include_pad=1`` is exactly what makes AveragePool treat the folded
positions as ordinary 0-valued data rather than divisor-excluded padding --
without it, the fused model would average over a different (smaller) set of
elements than the original two-node graph did. This is confirmed empirically
below by inspecting the real compiled pass's output attribute directly,
rather than merely asserting op counts. (MaxPool has no such attribute or
side effect -- confirmed empirically below too, by checking it is *not*
spuriously added.)

Caveat on the additive-merge branch for AveragePool specifically (not
re-derived here, since none of the differential tests below exercise it):
if Pool *already* declares its own nonzero ``pads`` before fusion, with
``count_include_pad`` left at its default of 0, the pass's unconditional
``count_include_pad=1`` override changes how *that pre-existing* padding is
treated too (previously divisor-excluded, now divisor-included) -- not just
the newly-folded Pad padding, which is the only part this proof's "same
fill constant" assumption covers. Empirically (in a throwaway script, not
committed) this does change the numeric result, caught by onnxsim's own
``--check``. The differential test below for the additive-merge branch
therefore uses MaxPool (which has no such attribute and so no such gap) to
stay strictly within what this proof covers.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser

_KERNEL_SIZE = 3


def _read(x, length, idx, fill):
    # Fill-value boundary condition shared by Pad (mode="constant", value=fill)
    # and by Pool's own implicit padding outside its input, for a symbolic 1-D
    # input of the given length.
    return z3.If(z3.And(idx >= 0, idx < length), x(idx), fill)


def test_fuse_pad_into_pool_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    length, pad_begin, pool_pad_begin, p = z3.Ints("length pad_begin pool_pad_begin p")
    c = z3.Real("c")
    domain = z3.And(length > 0, pad_begin >= 0, pool_pad_begin >= 0)

    def pool_at(read_at, pad_begin_amount, position):
        return sum(
            read_at(position - pad_begin_amount + k) for k in range(_KERNEL_SIZE)
        )

    def padded_read(j):
        return _read(x, length, j - pad_begin, c)

    def direct_read(j):
        return _read(x, length, j, c)

    # Two-step: Pad(X, pad_begin, fill=c) materializes a c-filled array; Pool
    # then applies its own (pre-existing) pool_pad_begin against *that* array
    # -- and Pool's own implicit padding there also reads c (this is exactly
    # the side condition the compiled pass enforces: Pad's constant_value
    # must equal 0 for AveragePool after count_include_pad=1 is forced on, or
    # -inf for MaxPool's own implicit out-of-bounds behavior).
    two_step = pool_at(padded_read, pool_pad_begin, p)
    # Single-step: fuse_pad_into_pool's own formula -- combined pad is the
    # elementwise sum -- applied directly against X, still reading c outside.
    direct = pool_at(direct_read, pad_begin + pool_pad_begin, p)

    prove(z3.Implies(domain, two_step == direct))


def test_fuse_pad_into_pool_pass_matches_averagepool_zero_fill():
    # AveragePool(Pad(X, pads, 0.0)) with zero padding on batch/channel and
    # positive padding on the spatial dims -- the compiled pass should fuse:
    # Pool's data input becomes X directly, Pool gets a `pads` attribute
    # equal to the spatial pad amounts, and count_include_pad == 1 is set
    # (see the module docstring for why that flag is necessary here).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,18,20] Y)
        <int64[8] node_pads = {0, 0, 1, 2, 0, 0, 1, 2}>
        {
          p = Pad<mode = "constant">(X, node_pads)
          Y = AveragePool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 0
    assert ops["AveragePool"] == 1

    pool_node = next(n for n in sim_model.graph.node if n.op_type == "AveragePool")
    assert list(pool_node.input) == ["X"]
    pads = next(a.ints for a in pool_node.attribute if a.name == "pads")
    # [x1_begin, x2_begin, x1_end, x2_end] -- the Pad's spatial pads verbatim,
    # since AveragePool had no pre-existing `pads` attribute of its own.
    assert list(pads) == [1, 2, 1, 2]
    count_include_pad = next(
        a.i for a in pool_node.attribute if a.name == "count_include_pad"
    )
    assert count_include_pad == 1


def test_fuse_pad_into_pool_pass_matches_maxpool_neg_inf_fill():
    # MaxPool(Pad(X, pads, -inf)) -- the compiled pass is onnxsim's own
    # patched FusePadIntoPool (not the vanilla onnxoptimizer submodule's
    # version), which requires -inf here rather than 0.0: MaxPool ignores
    # padded/out-of-bounds positions, equivalent to implicitly reading -inf
    # there, so only a Pad that materializes -inf is safe to fold into it
    # (see the module docstring; this is onnxsim's fix for
    # https://github.com/onnxsim/onnxsim/issues/290). MaxPool has no
    # count_include_pad concept, so confirm the fusion happens correctly
    # without that attribute being spuriously added.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,18,20] Y)
        <int64[8] node_pads = {0, 0, 1, 2, 0, 0, 1, 2}, float node_cv = {-inf}>
        {
          p = Pad<mode = "constant">(X, node_pads, node_cv)
          Y = MaxPool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 0
    assert ops["MaxPool"] == 1

    pool_node = next(n for n in sim_model.graph.node if n.op_type == "MaxPool")
    assert list(pool_node.input) == ["X"]
    pads = next(a.ints for a in pool_node.attribute if a.name == "pads")
    assert list(pads) == [1, 2, 1, 2]
    assert not any(a.name == "count_include_pad" for a in pool_node.attribute)


def test_fuse_pad_into_pool_pass_declines_maxpool_zero_fill():
    # The naive expectation (matching the vanilla onnxoptimizer submodule's
    # fuse_pad_into_pool.h, and this repo's own analogous fuse_pad_into_conv)
    # would be that a zero-filling Pad folds into any pool. onnxsim's own
    # patched pass deliberately refuses this for MaxPool (see the module
    # docstring / issue #290): only -inf is a safe fill to fold in, so this
    # must NOT fuse, and this is exactly the empirical surprise this test
    # pins down against the real compiled pass rather than assuming it away.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,18,20] Y)
        <int64[8] node_pads = {0, 0, 1, 2, 0, 0, 1, 2}>
        {
          p = Pad<mode = "constant">(X, node_pads)
          Y = MaxPool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    _sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1
    assert ops["MaxPool"] == 1


def test_fuse_pad_into_pool_pass_matches_with_existing_pool_pads():
    # MaxPool already declares its own nonzero `pads` here -- exercises
    # fuse_pad_into_pool.h's additive-merge branch rather than the
    # zero-plus-zero case. MaxPool is used (rather than AveragePool) so this
    # stays strictly within what the Z3 proof above covers: MaxPool has no
    # count_include_pad attribute, so there is no risk of the pre-existing
    # padding's *treatment* changing under fusion the way there can be for
    # AveragePool (see the module docstring's caveat paragraph).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,18,18] Y)
        <int64[8] node_pads = {0, 0, 1, 1, 0, 0, 1, 1}, float node_cv = {-inf}>
        {
          p = Pad<mode = "constant">(X, node_pads, node_cv)
          Y = MaxPool<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(p)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 0
    assert ops["MaxPool"] == 1

    pool_node = next(n for n in sim_model.graph.node if n.op_type == "MaxPool")
    pads = next(a.ints for a in pool_node.attribute if a.name == "pads")
    # Elementwise sum of the Pad node's spatial pads and MaxPool's own
    # pre-existing pads -- fuse_pad_into_pool.h's own additive-merge formula.
    assert list(pads) == [2, 2, 2, 2]


def test_fuse_pad_into_pool_pass_declines_pad_on_non_spatial_axis():
    # Padding on the batch axis (index 0 of the 8-long `pads`) must not be
    # folded into Pool's own (spatial-only) `pads` attribute.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[2,3,18,18] Y)
        <int64[8] node_pads = {1, 0, 1, 1, 0, 0, 1, 1}>
        {
          p = Pad<mode = "constant">(X, node_pads)
          Y = AveragePool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    _sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1
    assert ops["AveragePool"] == 1


def test_fuse_pad_into_pool_pass_declines_nonzero_fill_for_averagepool():
    # A nonzero constant fill value would, if folded in, change the sum (and
    # -- since count_include_pad would be forced to 1 -- the divisor too) at
    # every output position touching the padded region: unsound.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,18,18] Y)
        <int64[8] node_pads = {0, 0, 1, 1, 0, 0, 1, 1}, float node_cv = {2.0}>
        {
          p = Pad<mode = "constant">(X, node_pads, node_cv)
          Y = AveragePool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    _sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1
    assert ops["AveragePool"] == 1


def test_fuse_pad_into_pool_pass_declines_negative_pad_amount():
    # A negative pad amount means Pad is cropping, not padding -- folding it
    # into Pool's `pads` (which only ever grows the input) would be unsound.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,3,?,18] Y)
        <int64[8] node_pads = {0, 0, -1, 1, 0, 0, 1, 1}>
        {
          p = Pad<mode = "constant">(X, node_pads)
          Y = AveragePool<kernel_shape = [3, 3]>(p)
        }
        """
    )
    _sim_model, ops = simplify_isolated(model, "fuse_pad_into_pool")
    assert ops["Pad"] == 1
    assert ops["AveragePool"] == 1
