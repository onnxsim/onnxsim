"""Formal check for ReplaceEinsumWithMatmul (opt-in; onnx-optimizer's
``replace_einsum_with_matmul.h``): rewrites a 2-input ``Einsum`` into a
``MatMul`` when the ``equation`` string is one of exactly two recognized
shapes.

``runTransform`` parses ``equation`` into ``term1``/``term2``/``result``
(the two operands' index-label strings and the output's own), requires all
three the same length (at least 2) and every "batch" position -- every
position except the last two -- identical across all three strings
verbatim. Only the *last two* labels of each string can differ, and must
form one of:

  1. ``"...ij,...jd->...id"`` (``term1_m==result_m``, ``term1_k==term2_k``,
     ``term2_n==result_n``): plain ``Z = MatMul(X, Y)``.
  2. ``"...id,...jd->...ij"`` (``term1_m==result_m``, ``term1_k==term2_n``,
     ``term2_k==result_n``): ``X`` and ``Y`` share their own last label as
     the contracted axis, so ``Z = MatMul(X, Transpose(Y, perm=[...,-1,-2]))``
     -- swapping only ``Y``'s trailing two axes.
  Anything else declines, including the batch-label check failing.

Soundness is a genuine linear-algebra identity, not index bookkeeping: with
batch dims held fixed (the predicate requires them identical across term1/
term2/result, so the rewrite only ever touches the trailing two axes -- that
part is structural and not re-derived here), case 1's ``"ij,jd->id"`` is
*exactly* Einsum's own summation-convention meaning (implicit sum over any
label -- here ``j`` -- that appears in the inputs but not the output):
``Z[i,d] = sum_j X[i,j] * Y[j,d]``, which is verbatim ONNX MatMul's own
defining formula, so proving it is close to definitional (see the docstring
on the case-1 test below -- that's expected, not a shortcut). Case 2's
``"id,jd->ij"`` has real content: ``Z[i,j] = sum_d X[i,d] * Y[j,d]``, which
only becomes ``MatMul(X, Y^T)`` once you use the defining property of
``Y^T``, i.e. that swapping ``Y``'s last two axes gives
``Y^T[d,j] == Y[j,d]`` -- the case-2 proof below states that property as an
explicit axiom and has Z3 actually use it to derive the equivalence, rather
than starting from formulas that are already textually identical. A
negative-control test confirms the two branch formulas are not
interchangeable in general, so the pass's branch selection is doing real
work.

Both formulas are additionally composed with an arbitrary uninterpreted
``consumer`` (this suite's standard idiom, e.g.
tests/test_formal_verify_eliminate_identity.py) to confirm that whatever
reads the rewritten output downstream sees the same value the original
Einsum would have produced, not just that the two raw formulas agree.

Dtype gating (``patternMatchPredicate`` restricts to MatMul's supported
dtypes: FLOAT/DOUBLE/FLOAT16/INT32/UINT32/INT64/UINT64) is not exercised
differentially here -- constructing an *unsupported*-dtype Einsum that
onnxruntime's own reference execution (used by ``simplify_isolated_extra``'s
correctness check) can also run is awkward, since onnxruntime declines to
even execute plain ``Einsum`` for most non-float dtypes in the first place
(confirmed empirically: an int8 Einsum fails at the ORT-session-creation
step, independent of this pass); the two firing cases and the declining
"unsupported equation shape" case below cover the pass's actual rewrite
logic.
"""

from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

_K = 3  # concrete contraction dim -- enough to exercise the summation


def test_replace_einsum_with_matmul_case1_is_matmul_definition():
    """``"ij,jd->id"``: Einsum's own defining sum for this equation
    (implicit sum over ``j``, the label shared by both inputs but absent
    from the output) is ``sum_j X(i,j) * Y(j,d)``. ONNX MatMul's own
    defining formula for the same two operands is the *same* sum -- this
    looks close to tautological because it is: the actual content of case 1
    is recognizing that Einsum's summation-convention semantics for this
    exact index pattern collapses to matmul's own formula, not deriving new
    algebra. Composed with an arbitrary uninterpreted ``consumer`` to also
    confirm downstream reads see the same value either way.
    """
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Y = z3.Function("Y", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, d = z3.Ints("i d")

    einsum_ij_jd_id = sum(X(i, k) * Y(k, d) for k in range(_K))
    matmul_definition = sum(
        X(i, k) * Y(k, d) for k in range(_K)
    )  # MatMul's own formula

    prove(einsum_ij_jd_id == matmul_definition)
    prove(consumer(einsum_ij_jd_id) == consumer(matmul_definition))


def test_replace_einsum_with_matmul_case2_is_matmul_of_transposed_operand():
    """``"id,jd->ij"``: Einsum's own defining sum is ``sum_d X(i,d) *
    Y(j,d)`` (implicit sum over ``d``, shared by both inputs, absent from
    the output). The rewrite instead computes ``MatMul(X, Yt)`` where ``Yt``
    is ``Transpose(Y, perm=[...,-1,-2])`` -- ``Y`` with its own last two
    axes swapped, so ``Yt(d, j) == Y(j, d)`` by Transpose's own definition.
    Unlike case 1, that transpose property is stated here as an explicit
    axiom (``Yt`` is a *separate* uninterpreted function, not textually
    ``Y`` with swapped arguments), so the proof genuinely needs Z3 to
    instantiate the axiom to connect ``MatMul(X, Yt)``'s formula back to
    Einsum's -- there is real substitution content here, not just relabeling.
    Also composed with an uninterpreted ``consumer``.
    """
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Y = z3.Function("Y", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Yt = z3.Function("Yt", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")
    d0, j0 = z3.Ints("d0 j0")

    # Transpose(Y, perm=[...,-1,-2])'s own defining property: swapping Y's
    # last two axes.
    transpose_axiom = z3.ForAll([d0, j0], Yt(d0, j0) == Y(j0, d0))

    einsum_id_jd_ij = sum(X(i, d) * Y(j, d) for d in range(_K))
    matmul_of_transpose = sum(X(i, d) * Yt(d, j) for d in range(_K))

    prove(z3.Implies(transpose_axiom, einsum_id_jd_ij == matmul_of_transpose))
    prove(
        z3.Implies(
            transpose_axiom,
            consumer(einsum_id_jd_ij) == consumer(matmul_of_transpose),
        )
    )


def test_replace_einsum_with_matmul_cases_are_not_interchangeable():
    """Negative control: case 1's formula (``sum_k X(i,k) * Y(k,j)``) and
    case 2's formula (``sum_k X(i,k) * Y(j,k)``) are genuinely different
    computations -- a counterexample exists where they disagree -- so the
    pass's choice of which branch to rewrite into (based on which labels
    coincide) is doing real, load-bearing work, not picking between two
    formulas that happen to always match.
    """
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Y = z3.Function("Y", z3.IntSort(), z3.IntSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    case1_value = sum(X(i, k) * Y(k, j) for k in range(_K))  # "ij,jd->id" at (i,j)
    case2_value = sum(X(i, k) * Y(j, k) for k in range(_K))  # "id,jd->ij" at (i,j)

    solver = z3.Solver()
    solver.add(z3.Not(case1_value == case2_value))
    assert solver.check() == z3.sat, (
        "case 1 and case 2 formulas are always equal -- negative control is vacuous"
    )


def _model(body, opset=13, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def test_replace_einsum_with_matmul_pass_matches_plain_matmul():
    # "ij,jd->id" is the plain-matmul pattern -- should become MatMul(X, Y)
    # directly, no Transpose node anywhere.
    model = _model(
        """
        g (float[2,3] X, float[3,4] Y) => (float[2,4] Z)
        {
          Z = Einsum<equation = "ij,jd->id">(X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "replace_einsum_with_matmul")
    assert ops["Einsum"] == 0
    assert ops["Transpose"] == 0

    matmul_node = producer(sim_model, "Z")
    assert matmul_node.op_type == "MatMul"
    assert list(matmul_node.input) == ["X", "Y"]


def test_replace_einsum_with_matmul_pass_matches_transpose_matmul():
    # "id,jd->ij" shares the trailing "d" label between X and Y -- should
    # become Transpose(Y, perm=[1, 0]) followed by MatMul(X, transposed_Y).
    model = _model(
        """
        g (float[2,4] X, float[3,4] Y) => (float[2,3] Z)
        {
          Z = Einsum<equation = "id,jd->ij">(X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "replace_einsum_with_matmul")
    assert ops["Einsum"] == 0

    matmul_node = producer(sim_model, "Z")
    assert matmul_node.op_type == "MatMul"
    assert matmul_node.input[0] == "X"
    transpose_node = producer(sim_model, matmul_node.input[1])
    assert transpose_node.op_type == "Transpose"
    assert list(transpose_node.input) == ["Y"]
    perm = next(a for a in transpose_node.attribute if a.name == "perm")
    assert list(perm.ints) == [1, 0]


def test_replace_einsum_with_matmul_pass_matches_batched():
    # "bij,bjd->bid": leading batch label "b" identical across all three
    # strings, only the trailing two labels vary in the matmul-relevant way
    # -- should fire the plain-matmul branch with the batch axis passed
    # through untouched (still no Transpose).
    model = _model(
        """
        g (float[2,5,3] X, float[2,3,4] Y) => (float[2,5,4] Z)
        {
          Z = Einsum<equation = "bij,bjd->bid">(X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "replace_einsum_with_matmul")
    assert ops["Einsum"] == 0
    assert ops["Transpose"] == 0

    matmul_node = producer(sim_model, "Z")
    assert matmul_node.op_type == "MatMul"
    assert list(matmul_node.input) == ["X", "Y"]


def test_replace_einsum_with_matmul_declines_outer_product():
    # "i,j->ij" is a genuine outer product: no shared contracted label at
    # all (and term1/term2 have length 1, below the pass's shape_size>=2
    # floor), so it's neither recognized pattern -- Einsum must stay.
    model = _model(
        """
        g (float[3] X, float[4] Y) => (float[3,4] Z)
        {
          Z = Einsum<equation = "i,j->ij">(X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "replace_einsum_with_matmul")
    assert ops["Einsum"] == 1
    assert ops["MatMul"] == 0

    z_producer = producer(sim_model, "Z")
    assert z_producer.op_type == "Einsum"
    equation_attr = next(a for a in z_producer.attribute if a.name == "equation")
    assert equation_attr.s == b"i,j->ij"
