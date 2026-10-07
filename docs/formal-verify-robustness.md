# Keeping the Z3 formal-verification suite from hanging

`tests/test_formal_verify_*.py` prove, with Z3, that onnxsim's rewrites are sound. All of them go
through `prove()` in `tests/_formal_verify_common.py`.

## What went wrong

CI's **Formal verification (Z3 translation-validation proofs)** job normally takes 12-14 minutes. On
two commits in October 2026 it ran until GitHub cancelled it at the 6-hour job limit. The proof that
was running,
`test_formal_verify_rewrite_deform_conv_to_gather.py::test_fully_in_range_reduces_to_plain_four_corner_formula`,
is a pure Z3 proof; it never calls `simplify()`. Three facts about it were measured:

* **Its difficulty depends on process state, not on the claim.** Run alone it did not finish in
  250 s. Run right after the three proofs before it in its file it finished in 0.2 s. Run inside the
  full suite it took 116 s. Z3 searches differently depending on the AST ids and other state that
  earlier work left in the global context.
* **`prove()` had no timeout**, so "hard" meant "forever".
* **On `unknown` the failure message crashed.** `assert result == z3.unsat, f"...{solver.model()}"`
  evaluates `solver.model()` for the message, which raises `model is not available` when the result is
  `unknown`. A second proof (`..._tensor_scatter_to_scatter_elements_witness_arithmetic_is_sound`)
  was seen returning `unknown` only inside the full sequence while passing alone in 5 s.

## What `prove()` does now

1. **Fresh context per proof.** The claim is `translate`d into a new `z3.Context`, so its AST
   numbering is the same every time whatever ran before. This alone makes behaviour reproducible but
   not fast: the four-corner proof is then *consistently* slow (`unknown` at 60 s, warm or cold).
2. **A short portfolio of strategies**, each with a small cap, then the plain solver with the full
   timeout (`PROOF_TIMEOUT_S`, 120 s). No single strategy is fast on every proof. Measured, in a fresh
   context:

   | proof | default | `qfnra` | `simplify;propagate-values;ctx-simplify;smt` | default + `arith.solver=2` |
   |---|---|---|---|---|
   | quantized MAC bounds (nonlinear reals) | 0.2-40 s | **0.2-0.8 s** | gave up | 40 s |
   | four-corner formula (uninterpreted function) | gave up (> 60 s) | gave up at once | **0.1 s** | **0.0 s** |
   | flat-BEV-index bijection, scatter witness | **0.0 s** | burns its whole cap | 0.0 s | 0.0 s or burns cap |

   so the order is: default (3 s), `qfnra` (6 s), the smt tactic (3 s), `arith.solver=2` (3 s), default
   (full timeout). The short caps bound the time wasted on a strategy that is wrong for a proof.
3. **Conclusive results only.** `unsat` or `sat` from any strategy is sound, so trying more than one
   cannot weaken what is proved; it only decides how long finding the proof takes. A proof no strategy
   can decide fails with `solver gave up on every strategy ... neither proved nor refuted`; it is never
   passed silently and `model()` is only read after `sat`.

## Measured effect (whole suite, CI order, one CPU)

|  | before | after |
|---|---|---|
| tests | 1002 passed, 3 skipped | 1007 passed, 3 skipped (5 are new tests of the helper) |
| wall-clock | 11 min 33 s | 1 min 53 s |
| summed proof time | 677 s | 100 s |
| slowest proof | 116.6 s (four-corner) | 11.5 s (a negative control that finds a counterexample) |
| proofs > 2x faster / > 2x slower | | 23 / 0 |
| reverse file order | | 1007 passed in 1 min 45 s |

Each previously fragile proof also passes **alone, in a fresh process**, in about 3 s (the short
default round plus the strategy that works): four-corner 3.1 s (was > 250 s alone), MAC bound 3.2 s,
`qoperator_quantize_gemm` layer 1 3.2 s, scatter witness 0.01 s.

## CI

The formal job has `timeout-minutes: 45` (about 3x its former runtime, and far above the new one), so a
hang fails in under an hour instead of holding a runner for six.

## Adding a proof

Call `prove(claim)`; build `claim` with the module-level `z3` functions as before. For a proof that
is legitimately slow, pass `timeout=` rather than removing the cap. If a new proof is slow, measure
the strategies on it in a fresh context (translate the claim, try each solver) before adding another
strategy to `_STRATEGIES`.
