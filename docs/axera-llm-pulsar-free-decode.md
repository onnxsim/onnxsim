# Whole-model LLM decode with Pulsar-free programs (AX650N / AX8850)

This page covers the decode of a whole LLM on the AX650N card where no program is
compiled per model. Every program of every layer is made without Pulsar2, at the model's
own calibration. Fused programs, the residual Add and the linear layers are retargeted from
native template builds (`docs/axera-width-and-linear-emit.md`), and the calibration is
U16 MinMax computed on the host (`docs/axera-pulsar-free-calibration.md`). Pulsar2 runs
only once per program shape, to build the templates. A new checkpoint of the same shapes
needs no Pulsar2 run at all.

Two checkpoints were run on the card: JackFram/llama-160m and Qwen/Qwen3-0.6B. Both are
compared with Pulsar2's own `pulsar2 llm_build` output on the same steps.

- Code: `scripts/axera/llm_pulsar_free/` (one script per step, below).
- Tests (no card, no Pulsar2, no checkpoint): `tests/test_axera_llm_pulsar_free.py`.
- Related: `docs/axera-llm-rpc-decode.md` (the `llm_build` decode loop used for the
  baseline).

## Pipeline

```sh
cd scripts/axera/llm_pulsar_free
export MODEL=qwen3_0.6b LLM_PF_OUT=/data/pf/qwen3_0.6b   # LLM_PF_TEMPLATES: default <out>/../templates
python pf_templates.py            # native U16 template builds (Pulsar2, Docker), 4 variants per shape
python pf_record.py               # float decode over the calibration passages -> <out>/acts
python pf_gptq.py --device cuda   # optional: GPTQ weights on the card's s8 grid -> <out>/gptq
GPTQ=1 python pf_generate.py rest # every program of the `rest` set (rest before first); layers in
                                  # parallel, --jobs / LLM_PF_JOBS (default: by cores and memory)
GPTQ=1 python pf_generate.py first
python pf_generate.py nl          # llama-160m only: the newline steps' set (below)
GPTQ=1 python pf_estimate.py --doc 0      # host estimate (fake-quant), no card
python pf_plan.py --fake --float-set      # wiring check against the float model, no card / Pulsar2
GPTQ=1 python pf_plan.py --doc 0          # on the card -> <out>/plan_doc0.json
GPTQ=1 python pf_plan.py --generate 48    # free-running greedy decode vs float -> <out>/gen.json
# LLM_PF_FUSE=0..11 picks the program layout (Qwen3-0.6B default 8, llama-160m 3);
# LLM_PF_ADD_OFFSET / pf_generate.py --add-offset exact|pulsar2 the Adds' offset words
# (default Qwen3-0.6B exact, llama-160m pulsar2; "Defaults" below)
python pf_add_offset.py <out> <out>_x     # exact Add offsets for a set generated with pulsar2
python pf_baseline_llm_build.py --build-dir <llm_build out>   # Pulsar2 llm_build on the same steps
python pf_profile.py [--set first]        # per-program card time of the rest (first) set
```

| script | what it does |
| --- | --- |
| `pf_model.py` | `Config`: the checkpoint (pinned Hugging Face revision, or `LLM_PF_SNAPSHOT`), the sizes, the float reference of one step (`llm_layer_decomposition.layer_graph`), RoPE tables, merged weights |
| `pf_layout.py` | plan layout (mask, merged-linear column offsets, program order), the evaluation passages, `FakeAX` (numpy stand-in for the runner's plan mode) |
| `pf_templates.py` | every fused graph the programs use, and the Pulsar2 template builds |
| `pf_record.py` | each program's inputs over real-text float decode steps |
| `pf_generate.py` | calibration and emission of every program (`u16_retarget.retarget_graph` / `retarget`, `linear_emit.emit_linear_u16`), `manifest_<set>.json` |
| `pf_fake_quant.py` | fake-quant evaluation of a program the way the card runs it |
| `pf_estimate.py` | the whole decode in fake-quant on the host |
| `pf_plan.py` | the decode on the card: every program resident, one `PLAN` per token; `--generate N`: free-running greedy decode against the float model's |
| `pf_gptq.py` | `linear_emit.gptq_weight` for every linear (CUDA torch) |
| `pf_add_offset.py` | a copy of a program set with every Add's offset word exact (`u16_retarget.exact_add_offset`); `patch_entry` is what `pf_generate.py` applies to each program by default for Qwen3-0.6B |
| `pf_baseline_llm_build.py`, `pf_profile.py` | the `llm_build` reference point; per-program card time |

Calibration text: the prose of `README.md`, `docs/qat.md`, `docs/accuracy-drop.md` and
`docs/attention-quantization.md` (positions 1..127 of each). Evaluation text: the first 48
tokens of `docs/adaquant.md` and `docs/activation-memory-planning.md`, teacher-forced. Both
are read from the checkout. The calibration changes when those files change.

## The program set

Per layer, in order (`pf_layout.LAYER`), with the native template each one is retargeted
from:

| program | computes | template |
| --- | --- | --- |
| `n1_c` | coarse norm `u = x / sqrt(mean(x^2) + eps)` | `crms<H>` |
| `n1_rms` | RMSNorm of `u` with the gain as an input | `rms<H>` |
| `qkv5` | one linear: q, k, v, `q_rot`, `k_rot` (rotate_half folded into the weights) | `k<H>n<..>` |
| `rq`, `rk` | RoPE apply `q*c + r*sn`, or with Qwen3's q / k norm `(q*c + r*sn) / rms_head(q)` | `rope<n>` / `qkrope<n>` |
| `kc` (Qwen3) | per-head coarse norm of `[k \| k_rot]` (two-pass k norm) | `hcrms<2 kvd>` |
| `attn` | the whole layer's attention: q, the token-major K / V cache `[129, nkv*hd]`, mask | `attn<nh*hd>` |
| `o_proj`, `res1` | linear, residual Add | `k<..>n<H>`, `add<H>` |
| `n2_c`, `n2_rms`, `gu`, `silu_f`, `down_proj`, `res2` | second norm, gate \| up linear, fused SiLU, down linear, residual Add | |

The post programs are `nf_c`, `nf_rms` and `lm_head`. All activations are U16 and the
weights s8 per output channel. The two-pass norm keeps the fused RMSNorm's inner `x^2`
inside 16 bits. Its second pass is calibrated on the card's `u`, which is the coarse
program evaluated in fake-quant.

Calibration follows the SmolLM2 pipeline (`docs/axera-width-and-linear-emit.md`). Fused and
single-op programs are calibrated on their recorded samples plus x2 copies (norms,
residual Adds) or x sqrt(2) copies (products, attention). Linears use the real range x2;
a merged linear uses the union of its parts. Each program is retargeted from the first of
four template variants that accepts the target.

Two calibration sets cover the position whose activations are massive:

- `first`: position `first_pos`, with up to 512 different tokens.
- `rest`: every later position.
- `nl` (llama-160m only): every newline token, see "Newline steps" below.

llama-160m's BOS embedding is all zeros, so position 0 is zero end to end and is skipped.
Position 1 carries residuals of about 1.7e3-2e3 at layers 3..10, against at most about 300
later. Qwen3-0.6B has no BOS: position 0 reaches about 6.4e3-7e3 from layer 4 on, against
at most about 400 later.

### Merged programs (`LLM_PF_FUSE`)

Every `axclrtEngineExecute` costs about 0.27 ms on this card whatever the program
computes (a `[1, 1024]` Add takes as long as a two-pass norm), and the runtime has no
asynchronous execute. Decode speed therefore follows the number of programs run per
token. The table above is layout `0`. Layouts `1`..`8` run neighbouring programs as one
(`pf_layout.LAYER_FUSED*`; `4` to `7` need a per-head q / k norm):

| `LLM_PF_FUSE` | merged programs (template) | per layer | per token |
| --- | --- | --- | --- |
| 0 | none (`LAYER`) | 15 | 423 |
| 1 | `n1` / `n2` = residual Add + two-pass norm (`anorm<H>`; layer 0 `nrm<H>`), `qk` = `rq` + `kc` + `rk` (`qkf<nh*hd>`); the final Add + norm `nf` | 9 | 254 |
| 2 | `on2` = `o_proj` + `res1` + second norm, `dn` = `down_proj` + `res2` + the next layer's first norm (or the final norm) (`lanorm<k>`) | 7 | 198 |
| 3 (llama-160m default) | `dn` also takes the fused SiLU (`slanorm<I>`) | 6 | 170 |
| 4 | the linear after each norm joins it: `on2g` = `on2` + `gu` (`lanorml<k>x<n>`), `dnq` = `dn` + the next layer's `qkv5` (`slanorml<k>x<n>`); layer 0 `n1q` (`nrml<n>`) | 4 | 114 |
| 5 | the next layer's q / k / v and its RoPE join `dn`: `dnqk` (`slanormqk<I>`: five linears q, q_rot, k, k_rot, v, then `qkf`; outputs the residual, q, and the K / V rows); layer 0 `n1qk` (`nrmqk<H>`) | 3 | 86 |
| 6 | the programs of `5`, with rotate_half made inside `dnqk` / `n1qk` (`slanormqkr<I>` / `nrmqkr<H>`): three linears q, k, v (4096 columns instead of 7168), and q_rot / k_rot by `Split` / `Concat` of each head's halves | 3 | 86 |
| 7 | the attention joins the program after it: `aon2g` = `attn` + `on2g` (`alanorml<k>x<n>`); without a per-head q / k norm (llama-160m) layout 3's `attn` + `on2` = `aon2` (`alanorm<k>`) | 2 (llama: 6) | 58 (llama: 74) |
| 8 (Qwen3 default, with exact Add offsets) | the whole layer as one program: `alyr` = `aon2g` + SiLU + `dnqk` (`alyrqkr<nh*hd>`: the attention, `o_proj` + `res1` + second norm, gate / up, SiLU, `down_proj` + `res2` + the next layer's norm, q / k / v, rotation and RoPE; six linears, 17 MB); the last layer keeps `aon2g` + `dn` | 1 | 31 |
| 9 (opt-in, Qwen3) | the rest set's last layer and the final norm as one program too: `alyrf` = `aon2g` + SiLU + `dn` (`alyrf<nh*hd>`) | 1 | 30 |
| 10 (opt-in, Qwen3) | the first set's split layers (0-2) as whole-layer programs too: `alyrs` = `alyr` with the gate, up and down linears as two s8 linears and an Add each (`alyrqkrs<nh*hd>`, 37 MB); first token 32 executes instead of 64 | 1 | 30 |
| 11 (opt-in, Qwen3) | `lm_head` in the rest set's last program: `alyrfh` = `alyrf` + `lm_head` (`alyrf<nh*hd>x<vocab>`, 180 MB); the first set keeps `aon2g`, `dn` and an `lm_head` of its own | 1 | 29 |

A merged program has several outputs where the residual stream goes on (`x` and the
normalized `y`; `qk`'s `yq` and `yk`, which is bound to K row `p` as `rk`'s was). Only
layer 0 keeps a separate first norm, and a split first-set `down_proj` (layers 0-2) stays
a linear program followed by an `anorm` `dn`.

How the merged programs are made:

- **Templates.** One Pulsar2 build per merged graph, four calibration variants each, as for
  the other fused programs (`pf_templates.py`). Each Add gets its own `eps` initializer:
  Pulsar2 renames a shared one per use.
- **Calibration.** Each part of a merged program is calibrated as its own program was, on
  its own samples (`pf_generate.Generator._stage`): the Add on `a`, `b` and x2 copies, the
  coarse pass on `x` and x2, the second pass on the coarse pass's `u` in fake-quant and x2,
  the linear's input and output at the real range x2, the SiLU on sqrt(2) copies. A tensor
  that two parts share gets the union of their ranges.
- **Ties.** Where the chosen template calibrates two tensors alike, the target gets one
  range for both (`_tied`). Pulsar2 leaves out a write of the value a register already holds,
  so the template's records cannot carry two values there. This happens in every build:
  `d = u / r` with `r` the RMS of an already normalized `u` has `d`'s zero point equal to
  `u`'s, and the mean of `u^2`, its `+ eps` and the `Sqrt` share one scale. Without the
  ties the first `qk` programs scaled `yk` by exactly 1/sqrt(2) on the card. The program
  alone was right on the card's inputs from the recording; inside the plan it was not.
- **Job assignment.** `u16_retarget._assign_core_jobs` gives each CORE job a node by the
  topological order of the graph that explains the most calibration records (dynamic
  programming over the sets of computed nodes). The old brute force stopped at 8 nodes;
  `qkf` has 22. A score-only (Hungarian) assignment put `anorm`'s Div before its Sqrt
  where the scales tie, and the card's `y` came out several times too large.
- **Add shifts.** `_check_add_shift_writes` now walks the jobs: a PARAM job resets `0x1ea0`
  to 0, and an Add whose shift differs from the register's value needs its own write.
  `qkf`'s template `v1` (both RoPE Adds cancel, shift 11) keeps all five writes. It
  serves the targets whose q or k RoPE Add shifts below 15 (12 of 28 layers).
- **Layout 5's K / V rows.** `dnqk` of layer `l` writes layer `l + 1`'s K and V rows; the
  plan rebinds its `yk` and `v` outputs onto row `p` of that layer's caches every step
  (`Config.kv_producer`), so the V row copy of the older layouts is gone too. A layer
  after a split first-set layer keeps its own `qkv5` and `qk`. The rest set's `dnqk` is
  calibrated on the rest set's samples; the first set's RoPE parts on both sets (at
  position 0 sin = 0 makes `r * sn` a zero range).
- **Layout 7's attention.** The program before the attention (`dnqk`, or `rk` and the V
  copy) writes the step's K / V rows into the caches, so the attention cannot join it
  without moving the new row into the program (the cache input and the row output
  would be one device buffer in one execute). The program after it has no such
  dependency: `aon2g` reads q, both whole caches and the mask, as `attn` did. Each
  part keeps its calibration; the attention's output and the linear's input are one
  tensor (Reshape is data movement), at the union of both ranges (`z`, the linear's
  real range x2). Retargeting `alanorml2048x6144` between its four variants gives the
  native build byte for byte wherever it is not refused. For llama-160m's `alanorm768`,
  `v1` / `v2` retargeted to `v0`'s calibration differ from the native `v0` in 193
  records (unchecked: native `v0` is laid out differently or the retarget is wrong);
  every generated `aon2` came from `v1` and passes `pf_check.py`. Calibrating that tensor at the
  attention's own range instead was worse on the card (96 steps: 3.60% / 2.98% against
  3.56% / 2.97%).
- **Several linears in one program** (`on2g`, `dnq`, `dnqk`: two to six). Each has its
  table in `npu_params` (found from the template's own weights and calibration) and its
  own core `zp_x` records, told apart by the template's zero points (refused when two
  linears' inputs share one).
- **Rotation in the program (layout 6).** rotate_half is a permutation within each head (its
  sign is in the sin tables), so `nrmqkr` / `slanormqkr` make q_rot and k_rot from the q and k
  linears' outputs: Reshape to `[heads, 2, hd / 2]`, `Split` the halves, `Concat` them swapped,
  Reshape back (`pf_templates.ROT`). Pulsar2 runs Split / Concat without a calibration of
  their own (every side as the input) and adds two CORE jobs without records.
  `u16_retarget` treats them as data movement (`MOVES`: every side of a Split / Concat
  calibrated alike, `propagate`), and the strict job assignment allows one blank job per
  Split / Concat. The cancelling RoPE-Add variant (`v1`) gets its cancellation from the
  q / k weights instead (each head's second half about `-k` times its first half). The
  same variants serve the same layers as in layout 5 (rest set: `v0` 14, `v1` 13, `nrmqkr`
  `v1`).
- **The whole MLP in one program (layout 7).** `lyrqkr` is `lanorm<nh*hd>` (its tensors
  renamed `_1`, the attention output `att`), the gate and up linears from its output, and
  `slanormqkr<I>` on them (`pf_templates._lyr_graph`). Each half is calibrated as its own
  program was (`Generator.lyr`: `norm2` returns the ranges of each half), the gate / up
  outputs as the SiLU's inputs (sqrt(2) copies, as `dnqk`'s inputs were), and the residual
  between the halves gets the union. Real targets need RoPE-Add shift combinations the four
  usual variants do not cover together with `lanorm`'s, so `lyrqkr` has nine variants
  (`LYR_VARIANTS`: `lanorm`'s `v0` with each `slanormqkr` variant, and `v1` with the k
  tables scaled so that `sq` and `sk` are not calibrated alike). The rest set takes
  `v0` 14, `v1` 7, `v2` 4, `v3` 1 and `v7` 1 times. The last layer keeps `on2g` + `dn`.
- **Pulsar2's fused `Silu`.** `Mul(gate, sigmoid(gate))` right after a FullyConnected is
  fused into one onnx `Silu` op. Its input is quantized at U8 unless `Silu` is in the U16
  `op_types`. Its job is the U16 Sigmoid table layout (256 entries, one per 256 input
  codes, interpolated) of `x * sigmoid(x)`; the last entry is code 65535's, not 65536's.
  `u16_retarget` retargets it (`fuse_silu`: a Sigmoid without a calibration of its own and
  its Mul become one `Silu` node; checked byte-equal against native builds). It is not used:
  on Qwen3's gate ranges (step 0.12-0.36 per table entry) the interpolated SiLU is 1-3e-3
  off in relative terms, against 1.5-6e-4 for the Sigmoid table and a Mul, and the first
  layout 7 set with it gave 4.15% / 3.39% mean logit error against layout 5's 3.85% /
  3.09%. Written `Mul(sigmoid(gate), gate)`, the pair is not fused and `lyrqkr` keeps the
  Sigmoid table and the Mul.
- **Uneven tables.** For wide ranges Pulsar2 spaces the 256 entries of a Sigmoid or Silu
  table unevenly: 16 segments of 16 entries, each with its own codes per entry
  (`65536 / 0x1860[k]`) and first code (`0x1960[k] >> 16`). The entry `16 k + j` is the
  function at code `start_k + j * 65536 / step_k` (checked exactly on a Silu build). The
  retargets always write the even table; a template with an uneven one is refused
  (`_check_even_tables`). The even table is a valid program, but not byte-equal to a native
  build whose calibration would get an uneven one.
- **The whole layer in one program (layout 8).** `alyrqkr` is `alanorm<nh*hd>` (the
  attention's tensors renamed `_a`, the first norm's `_1`; `pf_templates.lyr_name`), the gate
  and up linears from its output, and `slanormqkr<I>` on them (`pf_templates._lyr_graph`;
  `lyrqk` / `lyrqkr` are the same without the attention). Each half is calibrated as its own
  program was (`Generator.alyr`: `norm2(..., ranges=True)` returns each half's ranges),
  the gate / up outputs as the SiLU's inputs (sqrt(2) copies, as `dnqk`'s inputs were), and
  the residual between the halves gets the union. The SiLU is written
  `Mul(sigmoid(gate), gate)` (see the fused `Silu` below). Its variants (`ALYR_VARIANTS`) keep
  `lanorm`'s `v0` with each `slanormqkr` variant (and `v1`'s k tables scaled 0.8 / 1.25 so
  that `sq` and `sk` are not calibrated alike): the rest set takes `v0` 14 and `v1` 13 times,
  the same split as `dnqk`. With `lanorm`'s `v1` (its residual Add cancels, so `x`, `u` and
  `d` tie) the card's `yk` came out 1/sqrt(2) off on every layer that took it.
- **Job assignment of a whole layer** (`u16_retarget._assign_after_attention`,
  `retarget_graph(..., attention_first=True)`). `alyrqkr` runs 109 CORE jobs for 53 compute
  nodes, about 55 of them data movement whose records collide with calibration values, so
  neither the strict rule nor "any node in any job" works (the latter is ambiguous: many
  tensors share values in a variant). Pulsar2 runs the attention's jobs first. The rest's
  nodes are assigned strictly from the first CORE job holding a record that only the
  residual Add explains; the attention's nodes may sit in any job before it. Starting from
  the first job that merely fits shifted every assignment and gave 0.5-14x errors on the
  card.
- **The last layer, the first set's split layers and `lm_head` (layouts 9-11).**
  `pf_templates._lyr_graph` takes another second half: `alyrf<n>` is `alanorm<n>`, gate /
  up and `slanorm<I>` (the final norm; outputs `x` and `y`, no RoPE tables; four variants,
  `alanorm`'s v0 with each `slanorm` variant), `alyrf<n>x<v>` the same with `slanorml<I>x<v>`
  (`lm_head` after the final norm; one 1.7 GB build, 4.7 min). `alyrqkrs<n>` is `alyrqkr<n>`
  with `gate = y_1 @ wg + y_1 @ wgl` (up and the down linear alike): the split linears of
  the first set's layers 0-2, `W ~ Q8 + R8` from the original weights as their separate
  programs were. Its variants keep `alanorm`'s v0: v6 (`alanorm`'s v2 half) was accepted by
  the retarget for layers 0 and 2 and was wrong on the card (`x` 0.17-0.29 in
  `pf_check.py`), so it is left out. Each part is calibrated as its program was; the first
  part of the down linear (`bh`) at sqrt(2) x its range x2, so that `b = bh + bl` shifts
  14 like the gate / up sums (calibrated as the SiLU's inputs, x sqrt(2)): a template
  elides the third, equal shift write, and with `bh` at x2 the sum shifted 15 or 14 by
  layer. The first set's last layer keeps `aon2g` + `dn`: at position 0 the last MLP
  cancels the massive residual (`x` far below `x_1`), so its residual Add shifts 13, a
  write every `alyrf` variant elides. That keeps the shared `lm_head`'s input buffer: the
  rest set's `alyrf` writes its `y` into the first set's `dn` output buffer.
- **Pulsar2's fused `Silu`.** `Mul(gate, sigmoid(gate))` right after a FullyConnected is
  fused into one onnx `Silu` op. Its input is quantized at U8 unless `Silu` is in the U16
  `op_types`. Its job is the U16 Sigmoid table layout (256 entries, one per 256 input
  codes, interpolated) of `x * sigmoid(x)`; the last entry is code 65535's, not 65536's.
  `u16_retarget` retargets it (`fuse_silu`: a Sigmoid without a calibration of its own and
  its Mul become one `Silu` node; byte-equal to native builds on four calibrations). It is
  not used: on Qwen3's gate ranges (0.12-0.36 per table entry) the interpolated SiLU is
  1-3e-3 off in relative terms, against 1.5-6e-4 for the Sigmoid table and a Mul. A
  merged MLP program with it (`attn` + `on2g` + SiLU + `dnqk`) gave 4.15% / 3.39% mean
  logit error against layout 5's 3.85% / 3.09%; written `Mul(sigmoid(gate), gate)`, the
  same program gave 3.85% / 3.14% at 55 ms against 65-68 ms for layout 6 (59 executes).
- **Uneven tables.** For wide ranges Pulsar2 spaces the 256 entries of a Sigmoid or Silu
  table unevenly: 16 segments of 16 entries, each with its own codes per entry
  (`65536 / 0x1860[k]`) and first code (`0x1960[k] >> 16`). Entry `16 k + j` is the
  function at code `start_k + j * 65536 / step_k` (checked exactly on a Silu build). The
  retargets always write the even table, a valid program though not byte-equal to a native
  build whose calibration would get an uneven one; a template with an uneven table is
  refused (`_check_even_tables`).
- **Merged CORE jobs.** With q / k / v and the RoPE in one program, Pulsar2 runs 36 CORE
  jobs for 33 compute nodes (`nrmqk`), three of them without any calibration record (data
  movement around the linears), and in other builds runs two nodes in one job. The
  assignment DP lets a job without records take no node, and a job take two nodes, up to
  `MAX_MERGED` either way.
- **A linear inside a fused program** (`lanorm`, `slanorm`) is a FullyConnected with no
  main-engine CORE job. Its `npu_params` table is `linear_emit.linear_params_u16`'s
  table byte for byte, two bytes into the fused program's `npu_params`. `retarget_graph`
  finds the template's table, rewrites the calibration records around it, writes the new
  weights' table, and moves the core's copy of `zp_x` (`0x03d0`). The weights are not a
  calibration tensor. Checked against native builds: all 24 variant pairs of `lanorm2048`
  / `lanorm3072`, `slanorm3072` (or refused), and `lanorm2048` with weights the template
  never saw (`tests/test_axera_u16_retarget.py`).
- **Byte identity with native builds.** Pulsar2 is not byte-reproducible. A rebuild of
  a template variant (same model, config and dataset, so the same calibration) is never
  byte-equal to the original. Of 15 rebuilds (every variant of `nrm1024`, `anorm1024`
  and `qkf2048`, v0 twice), none was. Segment 0's slot table comes out permuted, and in
  9 the graph inputs are listed in another order. The A8 slot records (and the A7
  records that follow them) follow that order, including where they sit among a job's
  other records. The order is a per-build coin flip, not a function of the calibration,
  so there is no native order a retarget could pick. `u16_retarget.compare_jobs` compares up to it: each job's other records in
  order, its A7 / A8 records as a multiset (or their count, when the IO orders differ),
  the slot table as a set, and `npu_params` byte for byte. Every rebuild equals its
  original under it.

  The merged groups' variant pairs come in three kinds. Some are refused (zero points
  below 256, ties, elided Add shifts). The rest equal the target's native build under
  `compare_jobs`, except where the native build stores its Add constants differently
  from the template. Pulsar2 keeps each distinct Add constant once in
  `npu_params` and loads each with a DMA job of its own. That adds a job, a sync, and
  shifts later scratch addresses by `0x20`. An eps Add's ratio `s_eps / s_e` rounds to 0
  for most calibrations. Where `e`'s range is small it rounds to a few units
  (`nrm1024` v2, `anorm1024` v1 / v3), and the two eps Adds stop sharing a constant. A
  residual update's ratio can round to 0 and then share the eps Adds' `(0x8000, 0)`.
  These follow the calibration (the rebuilds agree), and the template fixes the program's
  constant layout. So such a pair cannot be byte-equal without adding or removing a job.
  Where the template shares a constant that the target needs twice, `retarget_graph`
  leaves the shared word as the template's. llama-160m's layer-0 coarse norm is an
  example: it has the ratio 1.0 for `m0` against 0.99994, and no eps, which is about
  eps. That pass's output is normalized again by the second pass.
  `add_const_sharing` gives the sharing. For the shipped sets it differs from the
  template's for 17 of 176 Qwen3-0.6B layout-5 programs and 4 of 122 llama-160m layout-3
  programs. Both sets pass `pf_check.py`.

  `nrmqk1024`'s four variants refuse each other, and `slanormqk3072`'s templates in use
  (v0, v1) refuse the other variants. Six held-out native builds, two per template in
  use, cover them. Each is the variant's own draws with the norm input scaled 1.7x and
  the gain 1.3x, or its generator with another seed. Pulsar2's MinMax gives tensors that
  the template ties (`u` / `d`, `e` / `urr`, `kt` / `kth` / `ykh` / `yk`: a normalized
  tensor and its copy divided by an RMS of about 1) scales one float32 ulp apart, where
  the template has one scale. `pf_generate` ties them the same way for a real target. With the template's ties
  on the build's calibration, the retargeted program equals the build except for the
  moved scales' float32 lanes, 2-12 ulps apart (`nrmqk1024` v1: 16 of 2564 records;
  `slanormqk3072` v0 needs no tie and is equal outright). This is stronger than
  `pf_check.py` for `nrmqk1024`: a native program at a calibration the template never saw,
  record for record.

**Per-program card check (`pf_check.py`).** After a few real steps it reads back every
program's input and output buffers as the plan bound them, runs the program's host
model on the card's own inputs (the float graph at the program's calibration, the s8
linear), and compares. The host model computes each U16 Add the way the card does
(`pf_fake_quant.card_add`, see the Add's zero-point bias below). A wrong record (scale,
zero point, Add shift, weight table) shows up as 1e-1 or more. On the shipped sets the
check reads at most 8.7e-3 (Qwen3-0.6B layout 5, both sets; the last layer's `dn` and the
attention are the largest) and 1.1e-2 (llama-160m layout 3, first set's last `dn`; rest
set within 7.5e-4, median 4.5e-5). The tolerance is 2e-2 for both sets. With a float Add
as the host model, the check read up to 1.3e-2 (llama-160m layer 1's `on2`) and 3.3e-2
(Qwen3's first set); that was the Add's bias, not a wrong program. Rerun it after
regenerating a set: `LLM_PF_OUT=... python pf_check.py [--set first]` (exit status 1 when
a program is above tolerance).

### The plan

`pf_plan.Plan` loads both sets and binds every program to shared device buffers once. The
shared buffers are the embedding row, the RoPE tables (one row per layer with Qwen3, whose
q / k norm gains are folded into the tables), the mask, and per layer a token-major
K / V cache. A step is one `PLAN` of the runner (`scripts/axera/vm/axcl_batch_runner.c`):

- three `W` lines for the token's inputs;
- for `rk`, a rebinding of its output onto K row `p` (no copy);
- one device copy of v's slice of the `qkv5` output into V row `p`;
- one `X` per program;
- one `R` for the logits.

The mask leaves rows `0..p` valid. In float this equals `layer_graph`'s layout (earlier
rows plus the new row at the end), and the test checks that.

### What Qwen3-0.6B needed beyond llama-160m

- **q / k RMSNorm before RoPE.** `qkrope` computes `(q*c + rot(q)*sn) / rms_head(q)` with
  the gain folded into the host's tables (`c = g * cos`, `sn = rot(g) * sin`). This is
  exact because `rot` permutes within a head. The k norm needs the two-pass form: the U16
  step of k's `x^2` is up to 0.83x its typical mean (layer 8). So `kc` computes `u = k / r0`
  per head on `[k | k_rot]`, one slice since `qkv5` is laid out `q | q_rot | k | k_rot | v`,
  and `rk` then normalizes `u` exactly.
- **Two Adds at one shift.** qkrope's two Adds (`q*c + r*sn`, and `+ eps`) shift 15 in every
  first template, so Pulsar2 writes `0x1ea0` once. A target whose RoPE Add needs 14 (L11
  `rk`) needs both writes. `retarget_graph` refuses that template, and the `qkrope<n>t`
  variants, whose Adds shift differently, supply it.
- **GQA attention** in one program: q as `[nkv, G, hd]`, K / V token-major.
- **Shared linears.** A linear's program serves both sets, so one copy of its weights sits
  on the card, when position 0's ranges are within 2x of the rest's. It is calibrated on
  the union of both sets. 12 first-set linears keep their own program.
- **Split first-set MLP at layers 0-2.** Position 0 creates the massive activation there.
  s8 weights times that activation put K / V row 0, the attention sink, off. `gu` and
  `down_proj` of the first set are therefore two s8 linears, `W ~ Q8 + R8`, and an Add
  (~16-bit weights).
- **RoPE programs calibrated on both sets.** The first set shares the rest set's `rq` /
  `rk` / `kc`, because sin = 0 at position 0 degenerates their own calibration. They are
  calibrated on both sets: position 0's cos = 1 output reaches past the rest range
  (`rk` 30.7 against 21.6 at L18), and without position 0 in the calibration K row 0 is
  clipped.
- **U16 linears at k = 1024 and 2048** (segments `[240, 272]` and `[208, 272 x 3]`) and the
  1024 x 151936 lm_head (9 DEQUANT jobs, a 180-byte IO tail) are byte-equal to native
  builds. `linear_emit.U16_K` / `U16_TAILS` now include them.

## Results on the card

AX8850 (AX650N) card in the LXD VM, 2026-10-24. 48 teacher-forced steps per passage
(llama-160m: 47, position 0 skipped), compared with the float model (`Config.float_step`).
Top-1 means agreement with float's top-1. The logit error is the RMS error of the logits
over their RMS, as a median (p90) over the steps. The card's health check read 0.0 before
and after every run.

| model | program set | adaquant | activation-memory-planning | s / token |
| --- | --- | --- | --- | --- |
| llama-160m | Pulsar-free, round-to-nearest | 44/47, 1.31% (1.82%) | 46/47, 1.22% (1.48%) | 0.12 (card 85 ms) |
| llama-160m | Pulsar2 `llm_build` | 44/47, 1.89% (2.62%) | 46/47, 1.56% (2.03%) | 0.15 |
| llama-160m | rebuilt, `LLM_PF_FUSE=0` | 44/47, 1.28% (1.84%) | 46/47, 1.25% (1.50%) | 0.07 (card 61 ms) |
| llama-160m | rebuilt, `LLM_PF_FUSE=1` | 45/47, 1.27% (1.82%) | 46/47, 1.22% (1.59%) | 0.057 (card 48 ms) |
| llama-160m | rebuilt, `LLM_PF_FUSE=2` | 45/47, 1.27% (1.82%) | 46/47, 1.22% (1.59%) | 0.05 (card 39-41 ms) |
| llama-160m | rebuilt, `LLM_PF_FUSE=3` (default) | 46/47, 1.32% (1.86%) | 46/47, 1.24% (1.48%) | 0.045 (card 35-38 ms) |
| Qwen3-0.6B | Pulsar-free, round-to-nearest | 44/48, 4.4% | 46/48, 4.0% | 0.21-0.22 |
| Qwen3-0.6B | Pulsar-free, GPTQ | 45/48, 3.36% (5.63%) | 44/48, 2.91% (3.99%) | 0.18 (card 163 ms) |
| Qwen3-0.6B | Pulsar-free, GPTQ from `pf_gptq.py` | 43/48, 3.29% (5.16%) | 44/48, 2.89% (4.27%) | 0.19 (card 162 ms) |
| Qwen3-0.6B | the same, merged programs `LLM_PF_FUSE=1` | 44/48, 3.27% (5.25%) | 44/48, 2.94% (4.30%) | 0.14 (card 119 ms) |
| Qwen3-0.6B | the same, `LLM_PF_FUSE=2` | 44/48, 3.28% (5.21%) | 44/48, 2.94% (4.39%) | 0.12-0.13 (card 103-106 ms) |
| Qwen3-0.6B | the same, `LLM_PF_FUSE=3` | 44/48, 3.25% (5.66%) | 44/48, 2.91% (4.42%) | 0.12 (card 96-97 ms) |
| Qwen3-0.6B | rebuilt (see below), `LLM_PF_FUSE=3` | 44/48, 3.37% (5.89%) | 43/48, 2.77% (3.53%) | card 94-95 ms |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=4` (not taken) | 44/48, 3.46% (6.37%) | 42/48, 2.84% (4.41%) | card 77-78 ms |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=5` (default) | 43/48, 3.47% (6.00%) | 43/48, 2.77% (3.68%) | 0.08-0.09 (card 68-70 ms) |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=7` | 44/48, 3.55% (6.18%) | 43/48, 2.81% (3.80%) | 0.074-0.079 (card 57-58 ms) |
| llama-160m | rebuilt, `LLM_PF_FUSE=7` | 45/47, 1.29% (1.84%) | 46/47, 1.26% (1.58%) | card 34 ms |
| Qwen3-0.6B | Pulsar2 `llm_build` | 45/48, 7.47% (9.73%) | 47/48, 7.27% (9.66%) | 0.29 |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=7` | 43/48, 3.45% (6.12%) | 43/48, 2.81% (3.76%) | card 55 ms |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=8` (whole layer) | 43/48, 3.52% (6.11%) | 42/48, 2.81% (3.76%) | card 47 ms |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=10` (split layers whole) | 43/48, 3.55% (6.12%) | 43/48, 2.71% (3.79%) | card 46.5 ms, first token 49-51 ms (layout 8: 59-61) |
| Qwen3-0.6B | rebuilt, `LLM_PF_FUSE=11` (+ `lm_head`) | 43/48, 3.53% (6.11%) | 43/48, 2.71% (3.78%) | card 46.4 ms |
| Qwen3-0.6B | layout 5 + exact Add offsets (`pf_add_offset.py`) | 45/48, 2.85% (6.04%) | 44/48, 2.24% (3.56%) | card 70 ms |
| llama-160m | layout 3 + exact Add offsets (`pf_add_offset.py`) | 45/47, 1.23% (1.75%) | 46/47, 1.14% (1.40%) | card 37 ms |
| Qwen3-0.6B | layout 7 + exact Add offsets (2026-10-28) | 45/48, 2.88% (6.01%) | 45/48, 2.29% (3.81%) | card 56-57 ms |
| Qwen3-0.6B | layout 8 + exact Add offsets (default since 2026-10-28) | 45/48, 2.91% (5.98%) | 45/48, 2.36% (3.95%) | card 47-49 ms |
| llama-160m | layout 7 as `pf_generate.py` makes it now (2026-10-28) | 45/47, 1.26% (1.89%) | 46/47, 1.25% (1.52%) | card 33-35 ms |
| llama-160m | the same + exact Add offsets | 45/47, 1.22% (1.75%) | 46/47, 1.15% (1.44%) | card 33-34 ms |

Notes on the table:

- **Program counts.** Qwen3-0.6B: 673 programs (827 MB) resident, 423 programs run per
  token (15 per layer + 3). With merged programs: 391 resident / 254 per token
  (`LLM_PF_FUSE=1`), 331 / 198 (`2`), 278 / 170 (`3`, about 920 MB: `on2` / `dn` carry their own
  copy of the linear's weights in each set). llama-160m: 342 programs (336 MB), 171 per
  token.
- **Merged programs, back to back** (2026-10-27, the same 16 steps, host load 2-4): card
  time per token 166 ms (`0`), 122 ms (`1`), 106 ms (`2`), 94 ms (`3`). The `0` row's
  programs give the same accuracy as on 2026-10-25 (43/48 3.29%, 44/48 2.89%). Their card
  time was 172 ms that day against 162 ms then, so compare only runs made back to back. In
  `pf_profile.py` the merged elementwise programs take 0.34-0.46 ms each (`qk` 342 us,
  `on2` 413 us, `dn` 462 us), and `qkv5` / `gu` 0.6-0.66 ms. Every merged row misses the
  same positions as the `0` row (adaquant 0 / 3 / 10 / 13, activation-memory-planning
  0 / 13 / 28 / 34). The median error is unchanged. The p90 of `3` on adaquant (5.66%
  against 5.16%) comes from 2 of 48 steps. Its host estimate is 44/48 and 45/48, 2.34%
  and 2.34% (layout `0`: 2.30% and 2.31%).
- **Rebuilt rows** (2026-10-27, later). The scratch templates, recorded activations and
  GPTQ weights of the rows above were deleted from the scratch disk. They were rebuilt from
  this directory's scripts (`pf_templates.py`, `pf_record.py`, `pf_gptq.py`), so these rows
  are their own baseline: compare layout 5 with the rebuilt layout 3, run back to back,
  not with the older rows. Layout 5 against 3: mean logit error 3.85% / 3.09% against
  3.89% / 3.09%, the median and p90 within 0.1 points, top-1 43 against 44 on adaquant.
  The misses are 0 / 3 / 10 / 13 / 17 against 0 / 3 / 13 / 32, where the float top-1 /
  top-2 gaps are 0.021 / 0.068 / 0.084 / 0.116 / 0.047. Host estimates: 45/48 2.34%
  and 44/48 2.18% for layout 5, against 44/48 2.50% and 44/48 2.30% for layout 3.
  Layout 4 is not taken: it is worse on both passages (42/48, mean 3.25% against 3.09%).
  Layout 5's first version calibrated the rest set's RoPE parts on both sets, as
  layout 3's shared `qk` does: 43/48 3.42%, 43/48 2.78%. In `pf_profile.py`, layout 5
  takes `dnqk` 908 us (the down and q / k / v linears, 10.4 MB), `on2g` 764 us,
  `attn` 452 us and `lm_head` 8.3 ms per token. Resident: 198 programs.
- **Layout 6 against 5** (2026-10-27, back to back, host load 4-7 from other jobs). The
  card computes the same thing bit for bit: the s8 q_rot / k_rot columns were the q / k
  columns permuted, with the same per-channel scales and the same output calibration, so
  the permuted q / k outputs equal them. `pf_check.py` gives every program the same error as
  layout 5's (rest 198 outputs within 9e-3, first 222 within 3.3e-2), and the 48 steps give
  the same logit error on every step (43/48, mean 3.85%; 43/48, mean 3.09%). Card time per
  token, medians: 75.4 / 72.2 / 73.9 / 73.5 ms against 80.4 / 81.3 / 78.8 / 77.4 ms for
  layout 5 in four alternating pairs, so about 4-5 ms (6%) less. In `pf_profile.py`,
  `dnqk` takes 758 us against 898 us and `n1qk` 550 us against 733 us; the other programs
  are unchanged. The rest set is 643 MB against 738 MB (`dnqk` 7.9 MB against 11.3 MB per
  layer). Two of three `pf_profile.py` runs of layout 6 read every program, the unchanged
  ones included, 30-70% slow, so the profile is taken from the third run.
- **Layout 7, back to back** (2026-10-27). Qwen3-0.6B: 58 executes and 145 resident
  programs per token against 86 and 198. Card time, medians of alternating runs (host
  load 6.5-8.5): 57.2-57.6 ms against 66.2-66.4 ms for layout 6; in an earlier session
  at host load 3-5, the attention merge on top of layout 5 gave 61.5-62.3 ms against
  69.3-69.8 ms. Layout 7's programs are layout 6's plus `aon2g`, and layout 6 computes
  what layout 5 does bit for bit, so its 48 steps equal the merge on top of layout 5:
  mean logit error 3.94% / 3.18% against 3.85% / 3.09%, top-1 44/48 and 43/48 against
  43/48 and 43/48 (it no longer misses adaquant position 17). Over 96 steps the
  difference is 3.56% / 2.97% against 3.51% / 2.93%, top-1 90/96 and 87/96 against 89
  and 87, and steps 48-95 alone are equal (mean difference +0.002 / -0.001 points).
  Host estimate 45/48 2.45% and 44/48 2.20% (layout 5: 2.34%, 2.18%). llama-160m (layout
  3 with `aon2`): 74 executes and 148 programs against 86 and 172, card 33.6-34.3 ms
  against 38.5-38.7 ms; mean 1.55% / 1.42% against 1.54% / 1.40%, top-1 45/47 and 46/47
  against 46/47 and 46/47 (the extra miss is adaquant position 30, float gap 0.036,
  which layout 0 also missed). `pf_check.py` (with `card_add`): every `aon2g` / `aon2`
  output within 1.3e-3 on both sets of both models. Every program layout 7 shares with
  layout 6 (llama-160m: layout 3) is byte-identical. Layout 7 was not made the default: on the
  48 steps its mean error is 0.1 points higher on Qwen3, and llama-160m misses one more
  near tie (with exact Add offsets too; Defaults, below).
- **Layout 8, back to back** (2026-10-28, host load 1-2, three alternating rounds
  6 / 7 / 8). 31 executes and 94 resident programs per token (1.1 GB) against 58 / 145
  (layout 7) and 86 / 198 (layout 6). Card time, medians: 46.9 / 47.8 / 47.0 ms against
  56.9 / 55.9 / 55.5 ms (layout 7) and 63.4 / 64.0 / 64.5 ms (layout 6). A native
  `alyrqkr2048` build takes 1.37-1.46 ms per execute against 0.97 + 0.77 ms for
  `alanorml2048x6144` and `slanormqkr3072`. Accuracy at layout 7's level: mean logit error
  3.92% / 3.19% against 3.94% / 3.18% (layout 6: 3.85% / 3.09%), median 3.52% / 2.81%,
  p90 6.11% / 3.76%; top-1 43/48 and 42/48 against 44/48 and 43/48 (layout 6: 43 and 43).
  The extra miss is activation-memory-planning position 34 (float top-1 / top-2 gap 0.202,
  missed by the GPTQ rows above too). Host estimate 44/48 2.38% and 44/48 2.13%.
  `pf_check.py` (with `card_add`): every program within the tolerance, `alyr` within
  1.1e-2 (rest, `v`) and 2.3e-3 (first). With exact Add offsets it is the default
  (Defaults, below).
- **Layouts 9-11 against 8** (2026-10-28, back to back, host load 1-12; plain sets, the
  layout-8 set's programs reused where they are the same). Per token:

  | layout | executes (first token) | resident programs, MB | `pf_profile.py` step | first token | adaquant (mean) | activation-memory-planning (mean) |
  | --- | --- | --- | --- | --- | --- | --- |
  | 8 | 31 (64) | 94, 1159 | 47.6 ms | 59-62 ms | 43/48, 3.92% | 42/48, 3.19% |
  | 9 | 30 (64) | 93, 1159 | 46.3 ms | 60-64 ms | 44/48, 3.92% | 42/48, 3.19% |
  | 10 | 30 (32) | 60, 1149 | 46.0-46.8 ms | 49-51 ms | 43/48, 3.79% | 43/48, 3.06% |
  | 11 | 29 (32) | 60, 1317 | 45.7 ms | 50-52 ms | 43/48, 3.79% | 43/48, 3.05% |
  | 9, first set unsplit (`LLM_PF_SPLIT_FIRST=-`) | 30 (31) | 60, 1119 | | 48-50 ms | 43/48, 4.09% | 43/48, 3.29% |

  The 48-step wall medians of the steady steps are within 0.5 ms of each other at this
  host load (46.3-47.3 ms for 8, 46.3-46.9 ms for 10, 46.4-46.7 ms for 11); the
  `pf_profile.py` step times are the cleaner comparison. In the profile `alyrf` takes 1.09-1.12
  ms against 1.01 + 0.52 ms for `aon2g` + `dn`, and `alyrfh` 9.21-9.36 ms against 1.10 + 8.52-8.65
  ms for `alyrf` + `lm_head` (alternating runs). Layout 9 computes what 8 does (every step's
  logit error equal). The first set of layout 8 spends 15.9 ms of its 58.7 ms (`pf_profile.py
  --set first`) in the 37 programs of layers 0-3 (split linears 0.29-0.63 ms each); layout 10
  runs them as four. Its accuracy is layout 8's level: plain, every step's error 0.1 points
  lower on average from step 16 on (the position-0 K / V rows). With exact Add offsets
  (against the default set, layout 8 with them, run back to back at host load 1-5):
  3.47% / 2.88% and 44/48 / 44/48 against 3.41% / 2.80% and 45/48 / 45/48 (the new misses:
  adaquant 46, activation-memory-planning 7), the extra error in steps 2-15; first token
  48-50 ms against 59-60 ms, steady medians 45.9-47.7 ms against 46.8-47.6 ms. Layout 11 costs
  a second 155 MB `lm_head` (the first set's, which no longer shares one) for 0.3-0.4 ms per
  token. `pf_check.py` (with `card_add`): `alyrf` within 8.6e-3, `alyrfh` 1.4e-2 (its
  logits; 1.7e-2 with exact offsets), `alyrs` within 4.9e-3 (5.3e-3 with exact offsets).
  Without the split (s8 linears at layers 0-2 of the first set, as alyr) the first token
  is as fast, but the error is 0.1-0.17 points higher, all of it at position 0 (15.4% against
  12.7%) and steps 2-15; splitting only layer 2's linears (which make the massive activation:
  `down` 7562 at position 0) brings position 0 back (12.2%) but not steps 2-15 (4.01% / 3.25%).
- **llama-160m, merged layouts** (rebuilt the same way, round-to-nearest weights,
  layouts 0-3 back to back). Executes per token: 171 (`0`), 122 (`1`), 98 (`2`) and 86
  (`3`); resident programs 342 / 244 / 196 / 172. Without a per-head q / k norm, `rq` and
  `rk` stay separate programs in every layout (`4` and `5` need `qkf`). Layout 3 against
  0: top-1 46/47 against 44/47 on adaquant (it no longer misses positions 30 and 46). Mean
  logit error 1.54% / 1.40% against 1.53% / 1.40%. Host estimates 45/47 and 46/47, 1.22%
  and 1.16% (layout 0: 1.26% and 1.14%). `pf_check.py` on layout 3 with a float Add as
  the host model: rest set within 1.3e-2, first set within 1.8e-2. The 1.3e-2 was layer
  1's `on2` normed output, from the Add's bias (below); layout 0's programs read at most
  2.8e-3.
- **Card time.** "card" is the runner's execution time of the step's plan. The rest of
  the wall time is the host side: tables and logits over the shared folder, and the plan
  file.
- **`llm_build` timing.** The `llm_build` rows run through the session path of
  `llm_layer_loop` (K / V caches uploaded per call). That path is slower than its RPC
  path with resident caches (`docs/axera-llm-rpc-decode.md`: Qwen3-0.6B at 224 ms per
  token).
- **Qwen3 `pf_gptq.py` row.** The programs `pf_generate.py` emits from this repo's
  `linear_emit.gptq_weight` weights (2026-10-25). The GPTQ row above used the scratch
  GPTQ loop that `gptq_weight` reimplements. Its adaquant misses are at positions
  0, 3, 10, 13 and 46, and its activation-memory-planning misses at 0, 13, 28 and 34.
  Run back to back, both GPTQ sets take 162 ms of card time per token.
- **Qwen3 round-to-nearest row.** These are the card results reported from the run before
  GPTQ. Its JSON was overwritten by the GPTQ run, so no p90 is given.

Every top-1 miss of the Pulsar-free runs picks the float model's runner-up. The float
logit gap between the two is at most 0.2, against a logit standard deviation of 2-4:

| model, passage, position | float top-1 / top-2 gap | missed by |
| --- | --- | --- |
| llama-160m, adaquant, 24 / 30 / 46 | 0.033 / 0.036 / 0.002 | Pulsar-free (24 also `llm_build`) |
| llama-160m, activation-memory-planning, 46 | 0.072 | both |
| Qwen3-0.6B, both passages, 0 | 0.008 | Pulsar-free; `llm_build` picks a token 1.45 below top-1 |
| Qwen3-0.6B, adaquant, 3 / 10 | 0.021 / 0.068 | Pulsar-free (GPTQ) |
| Qwen3-0.6B, activation-memory-planning, 13 / 28 / 34 | 0.040 / 0.006 / 0.202 | Pulsar-free (GPTQ) |

### Defaults (2026-10-28)

`pf_model.MODELS` picks each model's layout (`fuse`, `LLM_PF_FUSE`) and Add offset words
(`add_offset`, `LLM_PF_ADD_OFFSET`, `pf_generate.py --add-offset`). The rule: the fastest
configuration whose 48-step accuracy is at least the old default's (layout 5 for Qwen3,
layout 3 for llama-160m) on both passages, top-1 and mean logit error. With `exact`,
`pf_generate.py` patches every program as it writes it (`pf_add_offset.patch_entry`, the
same records `pf_add_offset.py` rewrites, and the same `"add_offset": "exact"` manifest
mark). With `pulsar2` it writes what it wrote before, so every older set regenerates byte
for byte.

| model | default | was |
| --- | --- | --- |
| Qwen3-0.6B | layout 8, exact Add offsets: 31 executes per token, 47-49 ms, 45/48 and 45/48 | layout 5, Pulsar2's offsets: 86, 68-74 ms, 43/48 and 43/48 |
| llama-160m | layout 3 with the `nl` set (`pf_generate.py nl`, "Newline steps" below), Pulsar2's offsets: 86 executes per token | layout 3, Pulsar2's offsets, no `nl` set |

Card runs, back to back: each set alternated with the others in the same sessions, two
rounds per passage, the card lock taken per run, health check 0.0 before and after
every run. Host load (1-minute) 3-16 from other jobs. 48 steps (llama-160m 47).
Accuracy is deterministic: it was the same in every round. ms is the range of the runs'
median card time per token; a single slow run is given apart.

| set | adaquant: top-1, median (p90), mean | activation-memory-planning | card ms / token | `pf_check.py` worst, rest / first |
| --- | --- | --- | --- | --- |
| Qwen3 layout 5 (`n5b`) | 43/48, 3.47% (6.00%), 3.85% | 43/48, 2.77% (3.68%), 3.09% | 68-74 | 8.7e-3 / 8.6e-3 |
| Qwen3 layout 5 + offsets | 45/48, 2.85% (6.04%), 3.28% | 44/48, 2.24% (3.56%), 2.63% | 70-74 (one 86) | 8.7e-3 / 8.4e-3 |
| Qwen3 layout 7 | 44/48, 3.55% (6.18%), 3.94% | 43/48, 2.81% (3.80%), 3.18% | 56-58 | 8.6e-3 / 8.2e-3 |
| Qwen3 layout 7 + offsets | 45/48, 2.88% (6.01%), 3.37% | 45/48, 2.29% (3.81%), 2.73% | 56-57 | 8.8e-3 / 8.3e-3 |
| Qwen3 layout 8 | 43/48, 3.52% (6.11%), 3.92% | 42/48, 2.81% (3.76%), 3.19% | 47-49 (one 60) | 1.1e-2 / 8.6e-3 |
| **Qwen3 layout 8 + offsets** | **45/48, 2.91% (5.98%), 3.41%** | **45/48, 2.36% (3.95%), 2.80%** | **47-49** | 1.1e-2 / 8.8e-3 |
| llama layout 3 (`l3`) | 46/47, 1.32% (1.86%), 1.54% | 46/47, 1.24% (1.48%), 1.40% | 36-40 | 7.5e-4 / 1.1e-2 |
| llama layout 3 + offsets | 45/47, 1.23% (1.75%), 1.43% | 46/47, 1.14% (1.40%), 1.28% | 38-40 (one 53) | 1.1e-3 / 1.1e-2 |
| llama layout 7 | 45/47, 1.26% (1.89%), 1.53% | 46/47, 1.25% (1.52%), 1.40% | 33-35 | 1.5e-3 / 1.2e-2 |
| llama layout 7 + offsets | 45/47, 1.22% (1.75%), 1.45% | 46/47, 1.15% (1.44%), 1.30% | 33-34 | 1.7e-3 / 1.0e-2 |

- **Qwen3.** Layout 8 with offsets beats layout 5 without them on every measure but
  activation-memory-planning's p90 (3.95% against 3.68%): two more top-1 hits on each
  passage, mean 3.41% / 2.80% against 3.85% / 3.09%, at 47 ms against about 70 ms.
  In every layout the offsets fix adaquant 0 (and 17 where it was missed) and
  activation-memory-planning 0 and 5 (layout 8 also 34); layout 5 with offsets misses
  activation-memory-planning 7 instead. Layout 5 with offsets stays
  the most accurate (3.28% / 2.63%), 0.1-0.2 points ahead of layout 8 with offsets, at
  1.5x the card time. Free-running generation on the new default (4 prompts, 48 tokens)
  agrees with the float model as long as layout 5's does (with or without offsets): 48,
  1, 48 and 19 tokens, diverging at the same near ties (float gaps 0.18 and 0.19) into
  continuations that stay fluent ("... girl named Lila who lived in a village. She had a
  big garden"). The two complete prompts are token for token layout 5's.
- **llama-160m.** No candidate meets the rule: every one misses adaquant position 30,
  where the float top-1 / top-2 gap is 0.036, and layout 3 does not. The offsets take
  0.1 points off the mean error in both layouts; layout 7 saves 3-5 ms. Layout 3 with
  Pulsar2's offsets stays the default, now with the `nl` set, which changes no prose
  step (the rows here) and fixes the newline steps. On the passages with their line
  breaks, exact offsets with `nl` read 47/47 and 46/47 against 47/47 and 45/47 ("Newline
  steps"), but the prose rule decides. The `nl` set regenerates byte for byte on this
  code from the shared templates, with Pulsar2's offsets and with exact ones
  (`--add-offset exact`, equal to `pf_add_offset.py`'s copy).
- **The layout 7 sets.** Qwen3's layout 7 set regenerates byte for byte from master's
  `pf_generate.py`. llama-160m's round-one layout 7 set (`aon2` from `alanorm768` v1, 47
  steps 45/47 1.29% / 46/47 1.26%, the row above) does not. Master calibrates `aon2`'s
  attention output at the union of the attention's and the linear's ranges (scale
  6.1e-5 against 3.5e-5 at layer 0). The llama rows here are master's.
- **The templates.** The templates layouts 6-8 need (`nrmqkr1024`, `slanormqkr3072`,
  `alanorml2048x6144`, `lyrqkr2048`, `alyrqkr2048`; llama-160m `alanorm768`) are now in the
  shared template root next to the others. From that root alone, layouts 3, 5 and 6
  (Qwen3, layers 0-1 of both sets), 7 and 8 (Qwen3, both whole sets) and llama-160m's 3
  and 7 (whole sets) all generate. Each set is byte-identical to its card-tested set with
  Pulsar2's offsets, and to the `pf_add_offset.py` copy with exact ones: every `.axmodel`
  and manifest entry. The one exception is a tie refusal's message in llama-160m's layout
  3 manifest, which can name another tensor of the same tie.

### Free-running generation (`pf_plan.py --generate`)

The rows above are teacher-forced: every step gets the passage's next token.
`--generate N` instead feeds each argmax back, after a prompt, for `N` tokens. The float
model (`Config.float_step`) does the same on its own tokens. The run reports how many
leading tokens agree, the first divergence, and the float model's top-1 / top-2 gap
there; both models have seen the same prefix at that step. Four prompts
(`pf_plan.GEN_PROMPTS`), 48 tokens each, on the card, 2026-10-27. The shipped sets were
used: llama-160m layout 3, Qwen3-0.6B layout 5 with GPTQ.

| model | prompt | agree | first divergence (float gap) |
| --- | --- | --- | --- |
| llama-160m | "The capital of France is" | 3 / 48 | after " Paris.\n": "T" (of "Today") for "The" (1.87) |
| llama-160m | "Once upon a time, there was a little" | 48 / 48 | |
| llama-160m | "def fibonacci(n):" | 4 / 48 | after " 1.\n": "z" for "1" (0.30) |
| llama-160m | "Quantization makes a neural network smaller by" | 48 / 48 | |
| Qwen3-0.6B | "The capital of France is" | 48 / 48 | |
| Qwen3-0.6B | "Once upon a time, there was a little" | 1 / 48 | " girl named" for " girl who" (0.18) |
| Qwen3-0.6B | "def fibonacci(n):" | 48 / 48 | (a correct recursive function, with newlines) |
| Qwen3-0.6B | "Quantization makes a neural network smaller by" | 19 / 48 | (0.19) |

Qwen3-0.6B diverges only where the float model is near a tie (gaps 0.18 and 0.19).
Both continuations then stay fluent ("... girl named Lila who lived in a village called
Willowbrook"). It is the same for the sets patched by `pf_add_offset.py` (below), and
for the default since 2026-10-28, layout 8 with exact offsets (Defaults, above).

llama-160m breaks at a newline. Both of its divergences come at the first token after a
generated `\n`, and the float gap there is 1.87. The `\n` token makes a massive
activation like the attention-sink position 1: the residual reaches about 650 at layers
3..10. The calibration text has no line breaks (`pf_layout.prose` joins lines with
spaces), and the rest set's residual ranges there are about 90..220. The step clips. The
host estimate (`pf_estimate.Chain` on the same tokens) gives a 58% logit error at the
newline step, and the card gives the same 58%. The next steps are 30%, 16% and 11% off,
because the clipped K / V row stays in the cache. The programs compute what their
calibration says; the calibration does not cover the token. Prompts without a generated
newline agree for all 48 tokens. Qwen3-0.6B has no such token: its fibonacci
continuation has ten newlines and agrees throughout. llama-160m's newline steps now run
on a third program set, below.

### Newline steps (llama-160m's `nl` set)

In llama-160m the first newline of a sequence (and a newline right after it) acts as a
second attention sink, at any position. The residual reaches 400-560 at layers 3..9
(1.2e3 for a second newline at position 3), against at most 38-140 there in the
calibration prose. Layer 3's `down_proj` output reaches 27-31x its recorded maximum. A sweep of all 32000
tokens at position 20 finds no other common token that does this: the only others are
byte tokens and `<unk>`. Later newlines are ordinary. Qwen3-0.6B's newlines stay within
1.1x of its rest set's recorded ranges (the calibration takes 2x). That was checked on the
raw `README.md`, `docs/qat.md`, `docs/adaquant.md` and `pf_record.py`.

Calibrating the `rest` set on text with line breaks covers the newline, but it costs
ordinary text a lot. The residual ranges at layers 3..9 widen about 12x. In the host
estimate, the mean logit error on the two passages goes from 1.56% / 1.42% to 2.23% /
2.08%, and "The capital of France is" then diverges at its first token. So the newline
gets its own set. `nl` holds the `rest` set's programs, calibrated on the calibration
passages both as prose and with their line breaks kept (`pf_layout.lines`;
`pf_record.py` records `acts/nl`, `pf_generate.py nl` emits it). The plan runs it for
every `Config.nl_tokens` token (`"\n"`, id 13), and `rest` and `first` stay as they
were. That adds 86 resident programs (258 against 172, 161 MB more, about 3.5 s more
setup) but no executes: the per-token time is unchanged (44-46 ms of wall time).
`pf_check.py --set nl` runs the first eval passage with its line breaks up to its first
newline. `pf_add_offset.py` patches all three sets. Qwen3 does not need the set and has
no `nl_tokens`.

On the card, 2026-10-28 (layout 3, run back to back; health 0.0 before and after every
run). Prose teacher-forced, 47 steps: the newline set never runs there, so the results
are equal step for step.

| set | adaquant | activation-memory-planning | `pf_check.py` worst (rest / first / nl) |
| --- | --- | --- | --- |
| layout 3 | 46/47, 1.32% (1.86%), mean 1.54% | 46/47, 1.24% (1.48%), mean 1.40% | |
| layout 3 + `nl` | the same | the same | 7.5e-4 / 1.1e-2 / 9.3e-4 |
| exact Add offsets | 45/47, 1.23% (1.75%), mean 1.43% | 46/47, 1.14% (1.40%), mean 1.28% | |
| exact Add offsets + `nl` | the same | the same | nl 9.8e-4 |

The same passages with their line breaks (`pf_plan.py --lines`; newlines at positions
22 / 23 / 28 / 29 and 14 / 15 / 20 / 21 / 44), teacher-forced:

| set | adaquant, lines | activation-memory-planning, lines |
| --- | --- | --- |
| layout 3 | 40/47, 7.43% (15.9%), mean 12.3% | 39/47, 7.82% (16.3%), mean 14.0% |
| layout 3 + `nl` | 47/47, 1.54% (2.53%), mean 1.76% | 45/47, 1.55% (6.15%), mean 2.47% |
| exact Add offsets | 40/47, 7.35% (16.4%), mean 12.4% | 39/47, 8.20% (16.6%), mean 14.2% |
| exact Add offsets + `nl` | 47/47, 1.50% (2.22%), mean 1.65% | 46/47, 1.47% (5.42%), mean 2.32% |

Without the set, the newline step is 120-180% off and the next ones 8-30%. With it,
the newline steps are 2.6-2.8% off. The p90 of activation-memory-planning comes from
four later steps (8-11%: `.`, `plan`, `ation`, `memory` inside a backticked
identifier). They stay with an exact `nl` set, and leaving `dn` exact removes them. No
`dn` tensor is near its range there, so this is `rest`'s own error on that text, not
clipping.

Free-running, 48 tokens, on the card. The host estimate gives the same agreement on
every prompt, and exact Add offsets give the same as without them.

| prompt | layout 3 | layout 3 + `nl` |
| --- | --- | --- |
| "The capital of France is" | 3 (after " Paris.\n", gap 1.87) | 5 (`is` for `of` after "\nThe city", gap 0.094) |
| "Once upon a time, there was a little" | 48 | 48 |
| "def fibonacci(n):" | 4 (after " 1.\n", gap 0.30) | 48 |
| "Quantization makes a neural network smaller by" | 48 | 48 |
| "Shopping list:\n1. Milk\n2." | 2 (gap 0.047) | 2 (gap 0.047) |
| "def add(a, b):\n    return" | 4 (gap 0.084) | 48 |
| "Q: What is the capital of France?\nA:" | 8 (after "Paris.\n", gap 1.57) | 48 |

Every divergence left is at a float top-1 / top-2 gap below 0.1. Both newline
divergences with gaps of 1.57 and 1.87 are gone.

### Where the error comes from

The host estimate (`pf_estimate.py`) evaluates every program in fake-quant. With
round-to-nearest weights it reproduces the card's Qwen3 accuracy closely: 44/48 and 47/48,
3.41% and 3.32%. Keeping every linear's float weights while still rounding every
activation (`--float-weights qkv5,o_proj,gu,down_proj,lm_head`) brings that to 45/48 and
48/48, 0.82% and 0.72%. Most of the remaining error is therefore s8 weight rounding, not
the U16 activations. The scratch run reported the same ablation on the card as 4.15% -> 1.45% median; that run's JSON was not kept.
GPTQ (`linear_emit.gptq_weight`, `docs/axera-width-and-linear-emit.md`) takes part of it
back.

llama-160m's host estimate gives 45/47 and 46/47, 1.26% and 1.14%.

### The Add's zero-point bias (`pf_add_offset.py`)

llama-160m layer 1's `on2` read 1.3e-2 in `pf_check.py`; the other layers read at most
3e-3. A dump of its card inputs and outputs over 47 steps shows why. Its residual sum `x`
is 1.55 codes above the rounded float sum, on every channel and at every step (s.d. 0.02
across steps). Layers 0, 2 and 5 are off by 0.27, -0.28 and -0.39 codes. Layer 1's
normalized `y` makes this visible: the massive channels set the RMS, and the gain is
small there, so the other channels' share of `y` is large.

The card's U16 Add computes `(qa * A + qb * B + off) >> n`, rounded half up. The scale
ratios `A`, `B` are u16 `Q(n)` (`n = 15 - k`, `u16_retarget.add_shift`). Pulsar2's offset
word `off` takes the unrounded ratios, `int((zp_y - zp_a * r_a - zp_b * r_b) * 2^n)`, so
every element is off by `zp_a * (A / 2^n - r_a) + zp_b * (B / 2^n - r_b)` codes. That
formula gives 0.27, 1.55, -0.28 and -0.39 codes for layers 0, 1, 2 and 5, matching the
dump to 0.01. Layer 1 is the worst case: its residual input's range is a little wider
than the sum's (`r_a = 1.011`), so the shift is `k = 1` and the ratios have half the
precision, and the zero points are large (57309). A model of this arithmetic
(`pf_fake_quant.card_add`) gives the card's codes for 99.8-99.99% of the elements; the
rounded float sum gives them for 0% (layer 1) to 73%. Native Pulsar2 builds have the same
offset, so every retargeted program has it too. It is not a calibration or range-sharing
error.

The bias is the same on all channels of the residual stream and of the RoPE sums, and it
costs accuracy. In the host estimate (`pf_estimate.py --card-add`), Qwen3-0.6B layout 5
goes from 45/48 2.34% and 44/48 2.18% (float Adds) to 43/48 2.87% and 43/48 2.68%,
close to the card's 43/48 3.47% and 43/48 2.77%. Only the bias matters: card ratios with
an exact offset give 45/48 2.42% and 45/48 2.22%, and exact ratios with Pulsar2's offset
give 43/48 3.04% and 43/48 2.77%. llama-160m goes from 1.22% / 1.16% to 1.30% / 1.25%
(card: 1.32% / 1.24%).

`u16_retarget.exact_add_offset` is the offset word that cancels the bias:
`zp_y * 2^n - zp_a * A - zp_b * B`. `pf_add_offset.py` copies a generated set and
rewrites each Add's four offset records (`0x1ef0..0x1f20`), found by Pulsar2's value.
Nothing else changes, and the manifests mark the programs for `pf_check.py` and
`pf_estimate.py`. `pf_generate.py` applies the same patch to each program as it writes it
when the model's `add_offset` is `exact` (Qwen3-0.6B's default since 2026-10-28;
`--add-offset` / `LLM_PF_ADD_OFFSET` choose); `pf_add_offset.py` remains for sets
generated with Pulsar2's offsets. Qwen3-0.6B: 179 programs and 286 Adds. Each program differs from
Pulsar2's only in those records; in ten first-set `dnqk` programs a residual Add's exact
offset is 0, so two writes are elided as Pulsar2 would elide them. llama-160m: 120
programs and Adds.

On the card (2026-10-27), with the same speed:

| set | adaquant | activation-memory-planning | `pf_check.py` worst (rest / first) |
| --- | --- | --- | --- |
| Qwen3-0.6B layout 5 | 43/48, 3.47% (6.00%), mean 3.85% | 43/48, 2.77% (3.68%), mean 3.09% | 8.7e-3 / 8.6e-3 |
| the same, exact Add offsets | 45/48, 2.85% (6.04%), mean 3.28% | 44/48, 2.24% (3.56%), mean 2.63% | 8.7e-3 / 8.4e-3 |
| llama-160m layout 3 | 46/47, 1.32% (1.86%), mean 1.54% | 46/47, 1.24% (1.48%), mean 1.40% | 7.5e-4 / 1.1e-2 |
| the same, exact Add offsets | 45/47, 1.23% (1.75%), mean 1.43% | 46/47, 1.14% (1.40%), mean 1.28% | 1.1e-3 / 1.1e-2 |

The Qwen3 misses go from adaquant 0 / 3 / 10 / 13 / 17 to 3 / 10 / 13, and from 0 / 5 /
10 / 13 / 28 to 7 / 10 / 13 / 28. llama-160m's added miss is position 30, where the
float gap is 0.036. A calibration-side fix was tried as well. Each Add's output range was
widened by up to 2% to a scale where the bias cancels. It does not reach the first
set's residual Adds, whose `b` ratio is a few units at Q15 (the massive `x` range against
a small update), and its host estimate was no better (45/48 3.03%, 44/48 2.53%).
In those Adds the update can even round to `B = 0`: Qwen3's first set at layer 11 adds
nothing of `down_proj`'s output to the residual on the card (`b` range 0.07 against
`x` 15000).

## Checks

- **Merged layouts** (`tests/test_axera_llm_pulsar_free.py`): the whole-plan wiring check
  for `LLM_PF_FUSE=1..11` on the qwen3-like synthetic model, `1..3` on the llama ones. It includes the split
  first-set `down_proj`, several-output binding and `qk`'s K row. Also programs per layer,
  template names, FusedEval of each merged graph against `ReferenceEvaluator`, and the
  calibration ties. Regenerating layout `0` from the same templates is byte-identical to
  the 2026-10-25 programs (layers 0..11 checked).

- **Offline** (`tests/test_axera_llm_pulsar_free.py`):
  - The plan mask attends to the same rows as `layer_graph`'s.
  - `FusedEval` without a calibration equals `ReferenceEvaluator` on every fused graph.
  - The split weights sum to the original.
  - `FakeAX` runs the runner's plan lines.
  - **Whole-plan wiring check.** On `llm_layer_decomposition.make_synthetic`'s qwen3-like,
    llama GQA and llama MHA models, the whole plan reproduces the float model step by step
    to 1e-4 relative error, and so does the host chain with every program exact. Wiring
    covered: both program sets, shared and split programs, K row rebinding, V row copies,
    merged-linear slices. Float stand-ins replace the programs (`write_float_set`).
- **Against the runs above.** The scripts in this directory regenerate those runs' programs
  from the same templates and recorded activations byte for byte:
  - Qwen3-0.6B: layer 0 of both sets, with GPTQ weights, including `kc`, the shared RoPE
    programs and the split linears.
  - llama-160m: layer 11 and the post programs of both sets.

  `pf_estimate.py` reproduces the runs' host estimates exactly, and `pf_plan.py --fake` the
  float wiring check (llama-160m: median 1.1e-4).
- **Native identity** (`tests/test_axera_u16_retarget.py`, scratch builds): every
  rebuild equals its original under `compare_jobs`. Every accepted pair of the 11 merged
  groups equals the target's native build under it, or the native build's `npu_params`
  has another length (the Add constants). The held-out builds of `nrmqk1024` v1 and
  `slanormqk3072` v0 / v1 are covered too.
- **The card's Add** (`tests/test_axera_llm_pulsar_free.py`): `card_add` has
  llama-160m layer 1's 1.55-code bias, and none with `exact_add_offset`.
  `tests/test_axera_u16_retarget.py`: `exact_add_offsets` on a native `anorm` build
  changes only the residual Add's four offset records.
- **GPTQ.** `pf_gptq.py` uses master's `linear_emit.gptq_weight`. That function is a
  reimplementation of the GPTQ loop the card runs used, and about 5% of its codes differ
  by a rounding step. Its host estimate is 44/48 and 45/48, 2.30% and 2.31%, against
  44/48 and 43/48, 2.38% and 2.42% for the scratch weights. On the card its programs
  give 43/48 and 44/48, 3.29% and 2.89% (the `pf_gptq.py` row of the results table).

## Build time

How long a checkpoint takes to become card programs (not decode speed), on the same
32-core host, one heavy job at a time (2026-10-27). Wall time, CPU time and peak memory
are systemd's for the job (Docker's container cgroup for `llm_build`).

| stage | Qwen3-0.6B, layout 5 (GPTQ) | llama-160m, layout 3 |
| --- | --- | --- |
| Pulsar2 `llm_build` (s8, bf16, `--prefill_len 128 --kv_cache_len 255 --parallel 8`) | 7.7 min, CPU 19.4 min, 6.8 GB; 29 files, 666 MB | 119 s, CPU 344 s, 5.2 GB; 13 files, 158 MB |
| `pf_templates.py` (one-off per shape, cached), before (one build at a time) | 22.1 min for the 15 shapes layout 5 uses (`k1024n151936` alone 4.8 min); 28.1 min for all 66 builds | 12.4 min, 37 builds |
| `pf_templates.py`, now (`--jobs 8`, 2026-10-28) | 6.0 min for the 57 builds of `--list` (18 names, 24.8 min of serial build time), bounded by `k1024n151936` (5.9 min); Docker containers at most 7.7 GB together | (not measured) |
| `pf_record.py`, before (numpy `ReferenceEvaluator`) | 84 s, CPU 255 s, 16.7 GB (2026-10-28, back to back with the next row: 110 s, CPU 281 s, 13.8 GB) | 20 s, CPU 69 s, 4.6 GB |
| `pf_record.py --device cuda` (torch, 2026-10-28) | 50 s, CPU 34 s, 12.6 GB (the float steps 93 s -> 20 s; the rest is the checkpoint load and the `npz` writes) | (not measured) |
| `pf_gptq.py --device cuda` | 106 s, CPU 214 s, 13.9 GB | (not used) |
| `pf_generate.py rest`, before | 7.6 min, CPU 7.6 min, 17.7 GB | 57 s, CPU 64 s, 3.8 GB |
| `pf_generate.py first`, before | 6.8 min, CPU 6.8 min, 11.3 GB | 44 s, CPU 52 s, 3.6 GB |
| `pf_generate.py rest`, now (7 workers) | 29 s, CPU 200 s, 14.6 GB | 4.1 s, CPU 32 s, 3.3 GB |
| `pf_generate.py first`, now (7 workers) | 49 s, CPU 142 s, 13.2 GB | 5.6 s, CPU 27 s, 2.9 GB |
| `pf_generate.py rest` / `first`, now, `--jobs 1` | 97 s / 95 s | 29 s / 25 s |
| `pf_generate.py rest` + `first` against `both`, back to back (7 workers, 2026-10-28, load 5-10) | 101 s + 43 s = 144 s against 119 s, 20.5 GB | 20 s + 14 s = 34 s against 19 s, 5.2 GB |
| output | 198 programs, 1.34 GB (and 12.3 GB of `cal/`) | 172 programs, 336 MB (3.6 GB of `cal/`) |

The "before" rows of `pf_generate.py` are master's serial code under the phase timer
below, which adds 10-30% (without it: llama-160m's `rest` 43 s, and the shipped Qwen3
set's log 6.2 / 5.3 min). Layout 6 (Qwen3, `nrmqkr` / `slanormqkr` templates), without
the timer: `rest` 6.9 min -> 41 s, `first` 5.4 min -> 44 s.

The templates are built once per program shape, so a new checkpoint of a known shape
costs `pf_record.py` + `pf_gptq.py` + `pf_generate.py`: 4.5 min for Qwen3-0.6B (before:
17.5 min), against 7.7 min for `llm_build`; llama-160m 30 s (before: 2 min), against
2 min. `llm_build` cannot make these programs: it takes only `fp16` / `bf16` / `fp32` hidden states (no U16 activations), quantizes the
weights itself (`s8` / `s4` / `fp8`; no externally rounded GPTQ weights), builds one
program per layer (no merged layouts, no first / rest sets), and runs Pulsar2 for every
layer of every checkpoint.

**Where `pf_generate.py` spent its time** (Qwen3-0.6B layout 5, `rest`, serial;
exclusive time per phase; the `first` set is alike):

| phase | before | now (`--jobs 1`) |
| --- | --- | --- |
| retargeting: rewriting the records and `npu_params` (a Python scan of every 32-bit word of a 10 MB `npu_params`) | 170 s | 17 s |
| retargeting: CORE job assignment (a DP per program, the same for every layer) | 87 s | 4 s |
| weight tables: the U16 table layout (an index per byte), codes, requantization | 92 s | 7.5 s |
| template load: weights, `quant_axmodel.json`, LZ77 decode of the blob | 28 s | 8 s |
| recorded activations (`npz` reads) | 15 s | 12 s |
| calibration math (staged ranges, fake-quant, ReferenceEvaluator) | 20 s | 17.5 s |
| program encode (LZ77), float graphs, shape inference, file and tar writes, other | 38 s | 28 s |
| total | 451 s | 94 s |

llama-160m (many small fused programs) also spent 18 s of its 55 s writing the
calibration tars and reading them back. CPU time about equalled wall time: the code ran
on one core.

**What changed** (all output byte-identical, below):

- `u16_retarget._rewrite`: the `npu_params` scan finds the candidate words with numpy
  and re-reads only the word after a rewrite (a rewrite changes the next, overlapping
  word, which the old scan read rewritten).
- `u16_retarget`: the CORE job assignment and the template's own weight table are
  memoized (both depend on the template only); `short_unit_codec.decode_segments` keeps
  the last 64 decoded blobs.
- `linear_emit._u16_table`: the table is written through reshapes instead of an index
  array per byte.
- `pf_generate.py --jobs N` (`LLM_PF_JOBS`; default: the cores / 4, at most 8, and one
  worker per 3 GB of free memory, the cgroup's `MemoryMax` included): the layers and the
  post programs run in forked workers, each writing its own files; the manifest is
  merged in layer order.
- The calibration tars are written from memory (no temporary `.npy` files), and
  `u16_retarget.calibrate` / `calibrate_graph` take the samples directly
  (`samples=`) instead of reading the tars back.
- `pf_model.Config.head_t` is computed on first use (a 0.6 GB strided copy that
  `pf_generate.py` with GPTQ never needs).
- `Generator._tied`'s refusal message names the first offending tensor in sorted order.
  It iterated a set before, so the manifests of two runs differed in that one string.

**Checked** on full sets (both sets each) against master's serial `pf_generate.py`:
llama-160m layout 3 (`--jobs 7` and `--jobs 1`) and layout 7 (`--jobs 7`), Qwen3-0.6B
layout 5 (`--jobs 7` and `--jobs 1`) and layout 6 (`--jobs 7`); Qwen3 layout 7 on
layers 4-6. Every `.axmodel`, `model.onnx` and manifest is byte-identical (584 / 440 /
624 / 624 files), with two exceptions that hold between two serial runs too. The
calibration tars differ only in their time stamps: names, headers and data are equal,
and `pulsar2_docker.make_numpy_calibration_tar` stores the temporary file's mtime, so no
two runs ever wrote equal tars. And a tie refusal's message in the manifest can name
another tensor of the same tie. Master's serial code reproduces the shipped `n5b` / `l3`
sets the same way. `tests/test_axera_pf_build_time.py` checks each fast path against
the code it replaced. No card run was needed.

**Second round (2026-10-28)**, the first four items of the list that stood here:

- **Template builds in parallel**: `pf_templates.py --jobs N` (default 8,
  `LLM_PF_TEMPLATE_JOBS`) runs one Pulsar2 container per variant (per linear), the
  longest first (by `Unit.seconds`: the lm_head linear), the starts 1 s apart. Each
  container is capped with `docker run --memory` (no swap) at `Unit.gb`, and the running
  caps sum to at most `--mem-gb` (default 20): the containers run in the Docker daemon's
  cgroup, so a `systemd-run -p MemoryMax` around the script does not bound them. The
  measured container peaks: 1.1 GB at most for the fused, Add and layer-linear builds,
  4.7 GB for `k1024n151936`; the caps are 2 GB and 7 GB. A build that fails is rebuilt
  alone at the whole budget. `pulsar2_docker.build` names its container with a random
  suffix, so two builds started in the same millisecond no longer collide. Checked:
  every one of the 57 builds of two parallel runs (`--jobs 6` with larger caps, 7.2 min;
  `--jobs 8`, 6.0 min) against the shared serial templates: 4 byte-equal, the rest
  equal under `u16_retarget.compare_jobs` with the same `quant_of`, none different. A
  layout-5 set generated from the first run's templates passes `pf_check.py` on the
  card (rest 198 outputs, worst 8.7e-3; first 222, worst 8.6e-3; tolerance 2e-2).
- **`pf_record.py --device cuda`** (`pf_float_torch.py`): the same ONNX layer and norm
  graphs, node by node on torch tensors (float32, TF32 off), the K / V caches on the
  device. Not byte-identical to the CPU recording: against it every recorded tensor is
  within 8.7e-5 of its absmax (rest; first 9.3e-6), the MinMax ends within 1.6e-5, the
  positions and attention samples equal. `pf_gptq.py` on those activations rounds 1-4
  in 10^4 weights differently (rerun on the CPU recording, it reproduces the shipped
  weights exactly). A layout-5 set from the CUDA recording and its GPTQ passes
  `pf_check.py` (rest worst 8.8e-3, first 8.4e-3), and its 48 steps on the card, back
  to back with `n5b`: 43/48, mean 3.90% against 43/48, 3.85% (adaquant); 43/48, 3.13%
  against 43/48, 3.09% (activation-memory-planning). The CPU stays the default.
- **`pf_generate.py both`**: the two sets as one schedule over one pool: the post
  programs and the first set's split layers first, and a `first` layer as soon as its
  `rest` entry is in `manifest_rest.json` (written to a new file and renamed, since a
  worker may be reading it). Checked byte-identical with `rest` then `first` (same
  code and templates): Qwen3 layout 5, 624 / 624 files, and llama-160m layout 3,
  584 / 584 (tars equal but for their time stamps); `rest` then `first` also still
  reproduces the shipped `n5b`.
- **`LLM_PF_CAL_TARS=0`**: no calibration tars under `cal/` (llama-160m 3.4 GB -> 0.27 GB);
  no later step reads them (`pf_check.py`, `pf_estimate.py`, `pf_plan.py` read the float
  graph and the manifest's calibration). Every other file is byte-identical.

**What remains, ranked** (estimate, and how to check it):

1. **A program cache by content hash** (template bytes, float graph, calibration
   samples, weights, code version): a rerun after one change would rebuild only what
   changed. Check by perturbing each input in turn.
2. **Templates named by their full dims** (eps, heads, head size, context): a checkpoint
   whose dims match another's could reuse its templates without any Pulsar2 run.
   The registry ("Template registry" above) now has the signatures and the verified-reuse
   check; `pf_generate.py` still resolves by name, so wiring it in is the next step.

Moving the weight tables and the quantization parameters to CUDA is not on the list:
after the changes above they take about 7.5 s of a 94 s serial set.

## Template registry (`pf_registry.py`)

Template names carry the widths only (`nrm1024`, `alyrqkr2048`). The dims that change
a template's records are not in the name: eps, heads and head size, the context, the q / k
norm, the graph variant, and the Pulsar2 config. `pf_registry.py` gives every template a
canonical signature and says which templates a model needs and which are missing.

- **Signature** (`signature(name, dims)`): `kind` and `widths` (the numbers in the name; a
  linear's `K` and `N`); `dims` (the `Dims` fields the kind reads: `eps` for the norms, `hd`
  for the q / k norm, `nh` / `nkv` / `hd` / `S` for attention, `H` / `I` for the layer kinds;
  rope, SiLU and the linears read none of them); `variant` (`qk_norm` for kinds that exist
  only with a q / k norm and for `rope`, `alt` for the `t` suffix, `rot` for rotate_half in the
  program, `split` for the split first-set layers); `pulsar2` (image, NPU1, U16, MinMax, the
  layer ops, the variant count, the calibration draws). `sig_id` is the first 16 hex digits of
  SHA-256 over the canonical JSON.
- **Keys** (`keys_for(spec, layout)`): `pf_templates.needed` for a `ModelSpec` (dims, eps,
  heads, the q / k norm, the split first set). The list is the same one, so shipped names
  do not change.
- **States** (`status`): `exact` (a directory with the same `sig_id`), `verified` (a stored
  reuse record says `compare_jobs` is equal on every variant against a directory that exists),
  `check` (no exact match, but a directory of the same name and another signature: a fresh
  build is compared with it, and the verdict is stored), `missing` (a build).
- **Aliases.** A directory with no `signature.json` (every shipped template) is found by its
  owner's dims, and one written by `resolve` is found by its signature. The names still
  resolve, and `template_dir` is unchanged.
- **Manifest.** `template_registry.json` (next to `pf_registry.py`) holds the model specs
  (from their Hugging Face configs), the signatures of every key of the shipped models and of
  Qwen3-4B at every layout, and the reuse records. Directories `resolve` builds carry a
  `signature.json`; the shared directories are not edited.
- **Resolution in `pf_generate.py`** (`resolve_templates`, which calls `pf_registry.plan`).
  Before anything is generated, each key of the model at its layout is resolved in this order:
  1. **exact**: a directory with a `signature.json` of the key's signature (an overlay built by
     `pf_registry build`, e.g. `R/templates`);
  2. **by name**: a directory of the key's name in a shared root whose owner gives the same
     signature. A shipped root is `templates` (Qwen3-0.6B) or `templates_llama` (llama-160m),
     recognised by the real path, so a symlink into them counts. Another unsigned root needs
     `--owner DIR=MODEL`;
  3. **reuse**: a stored `equal` verdict in `template_registry.json` whose source directory
     (its `from`) is in the roots;
  4. otherwise **missing**: nothing is used in its place (a directory of the name with another
     signature, or with no signature file outside a shared root, is named as the reason).

  The log has one line per key (`exact`, `by name` or `reuse`, and the directory). The model
  spec is `MODEL`'s name, or `--spec` (a manifest name or a `config.json` path; its dims must be
  the checkpoint's). Generation reads a template only through the resolved map (`Generator.tdir`).
  The `<name>t` alternative directory is no longer looked up: no such directory exists in any
  root.
- **Missing keys fail loudly.** With a key that has no record, `pf_generate` prints the keys,
  their reasons and the build line to stderr and exits 2:
  `pf_registry.py build --model M --layout N --out ROOT [--root DIR=OWNER]`. `--out` is an
  overlay: a shared root is not written, and the shared directories a run reuses are symlinked
  into the overlay.
- **Not reused yet.** No reuse record is stored (none has been built with `compare_jobs` against
  another signature), so the reuse step is exercised only by the host tests. `pf_registry build`
  of a `check` key writes one. The manifest of each program set is unchanged: the resolution is
  in the log only. `pf_plan`, `pf_check`, `pf_record` and `pf_gptq` still resolve by name.
  `--spec` checks the dims, not the layer count (no template depends on it).

**Qwen3-4B at the template level (not generated).** `pf_model.MODELS` has `qwen3_4b`
(Qwen3-4B's dims; the split first set as Qwen3-0.6B's, assumed). No float checkpoint is cached
(only `Qwen/Qwen3-4B-AWQ`, packed int4), so the runs use a synthetic 2-layer checkpoint with
Qwen3-4B's dims and random float16 weights (`LLM_PF_SNAPSHOT`); its outputs test the plumbing,
not the model. What is verified:
- **Built:** the 20 layout-8 keys exist in `R/templates` (overlay), each with a `signature.json`
  of the key's signature, so all 20 resolve as `exact`.
- **Weight-table check (`pf_generate --verify-tables`, `template_table_error`):** 5 of the 20
  fail it: `k9728n2560`, `lanorm9728`, `slanorm9728`, `slanormqkr9728`, `alyrqkr4096`. Each one's
  own weights, retargeted onto its own graph (fused variants) or emitted (linears), do not find
  their table in the compiled `npu_params` ("the template's weight table is not found once"; for
  the linear, the table is short of its tail). The shipped K=3072 linear and the
  `slanorm3072` template pass the same checks. The four `slanorm9728` variants of `R/templates`
  fail; a fresh single-key rebuild of `slanorm9728` (`pf_registry build --name`, `W/T2`) fails
  the same way. The likely cause is the table format for K=9728 (the pattern is that
  every failing key has the 9728 width), not a damaged build. This is a hypothesis: not
  checked against Pulsar2's table layout.
- **Generation:** not reached. With the check on, `pf_generate both 0 1` exits 2 at
  resolution, naming the 5 keys. Without the check (default), the run fails at
  `rest/L1/dn` ("no usable template", the same weight-table error), so layers 0-1 do not
  generate. Layers 0 and 1 are not produced for Qwen3-4B.

**Shipped byte-identity under the registry** (this branch vs pristine master or the shipped
sets; `F/cmpset.py`, `pf_generate.py both 0 1`, the shared templates, `LLM_PF_CAL_TARS=0`):
Qwen3-0.6B layouts 3, 5, 6, 7, 8, 9, 10, 11 and llama-160m layouts 3 and 7 are byte-identical
on layers 0-1 of both sets. Layout 8 matches `F/n8o`, llama layout 3 matches `l3`, the others
match pristine master's output (the `n5b` set is gone; layout 5 is compared with pristine).
Every shipped key resolves as `by name` (the shared directories have no signature file).

Third model: Qwen3-4B's dims (`config.json` of its AWQ repository in the HF cache; the
template builds need no checkpoint), layout 8, split first set as Qwen3-0.6B's (assumed). It
needs 20 keys (69 Pulsar2 units: 4 variants per fused key, 8 for `alyrqkr`, 5 linears, 2
Adds). None exists yet: 0 exact, 0 verified, 0 checked, 20 built. All 69 builds succeeded in
24 min 38 s wall (`--jobs 8`, 20 GB of caps; the `k2560n151936` lm_head linear is the long
pole). Four `alyrqkr4096` variants failed in the parallel pass and were rebuilt alone. Summed
per-build time 3848 s. Built into `R/templates` (an overlay, not the shared directories).

**Byte-identity** (this branch vs master's code, `pf_generate.py both 0 1`, the same shared
templates; `F/cmpset.py`): Qwen3-0.6B layouts 5, 6, 7, 8, 9, 10, 11 and llama-160m layouts 3
and 7: every `.axmodel` of layers 0-1 of both sets equal (Qwen 7 / 23 / 3 / 3 / 23 / 23 /
3 / 3 axmodels; llama 15 / 15 / 13 / 13). Layout 8 also equals the shipped `F/n8o` set,
and llama layout 3 equals the shipped `l3` set. The references named in the brief
(`n5b`, `A/n6`, `B/n8`, `C/n7`, `C/l6`) are gone from the work directory, so those layouts
have only the pristine-master comparison.

## Split linears (Qwen3-4B's down projection, K = 9728)

**Status: wired (2026-10-28).** `pf_generate.py` emits the split for every linear whose
width is absent and every Add whose width is absent, from the measured widths in
`template_registry.json`; `pf_plan.py` and `pf_check.py` run and check the split trees.
The shipped layouts do not change: no width of Qwen3-0.6B or llama-160m is measured absent
(`split_keys` returns the key list unchanged at every layout; byte-identity below).
Qwen3-4B runs at **layout 1**, the only layout whose down projection is a standalone
program. At layouts 2-8 the down projection is inside a fused program (`lanorm9728`,
`alyrqkr4096`, ...) whose template still has the K = 9728 table absent: those keys stay
missing (exit 2), and fusing a split into a merged graph is not written.

**Why.** Pulsar2's U16 linear at N = 2560 compiles without its weight table for some input
widths (the table check below). K = 9728 is absent, so the linear is split.

**The measured widths** (`linear_emit.linear_table_present`, Pulsar2 `pulsar2:7.0-lite`):

| K (N = 2560) | table | present | source |
|---|---|---|---|
| 3072, 4096 | 8.4, 11.2 MB | yes | pulsar-confirm control |
| 4864 | 13.4 MB | yes | this branch |
| 5456 | 15.0 MB | **no** | this branch |
| 5888 | 16.2 MB | yes | this branch |
| 6014, 6016 | 16.4 MB | **no** | this branch |
| 6144, 8192, 9728 | 16.8, 22.4, 26.6 MB | **no** | pulsar-confirm control |

The width rule is not the table size: `k1024n151936` (a 166 MB table) is present, and a
two-linear program (two K = 4864 MatMuls and an Add, `npu_params` 26.7 MB) has both tables
present. The absent widths all have a first segment (`linear_emit.u16_layout`) of 280 pairs
or more, the present ones 256 or less. That is a hypothesis: only the widths above are
measured.

**The linear split (`pf_registry.split_keys`, `Generator.ksplit`).** `y = x_0 W_0 + x_1 W_1`
with x split along K at 4864. Both parts read the input on its whole grid `rng2(x)` (a byte
offset of the same buffer on the card: part 1 starts at byte `4864 x 4`). Both parts and
their sum share one output grid `(s, 32768)`, `s = max(|y|, |y_0|, |y_1|) / 32767` over the
calibration samples, so the Add is the exact integer sum `q = q_0 + q_1 - 32768` (both ratios
1). The Add is a single-op `add2560`.

**The Add split (`pf_registry.add_split`, `Generator.add_cols`).** Qwen3-4B's first set
splits gate | up as two s8 linears and an Add (`gu`, N = 19456: the `add19456` Add). That Add
is wrong on the card. Measured with one retargeted Add program per width on random inputs,
against the host Add model of `pf_fake_quant.AddEval` (`card_add`):

| width | card minus host, max | elements above 0.01 |
|---|---|---|
| 9728 (`add9728`) | 4e-4 (one output LSB) | 0 |
| 19456 (`add19456`) | 23.9 | 3068 of 19456 (the first ~460 right) |

The width is measured absent and the Add is split into two `add9728` column halves (gate, up).
Both write one output buffer at their byte offsets (`colsplit`; `pf_plan.chain_add`), so the
consumer reads the whole width as before. In the card check `gu_add_0` and `gu_add_1` are
2e-6 and 0 relative to their host models.

**The first set keeps the gate | up split.** Position 0 is the massive activation, and the
first set's split layers (0-2) exist for it: their weights are ~16-bit (two s8 halves). The
split is not separately measured here as unnecessary for Qwen3-4B; its Add is now a column
split (above) instead of the N = 19456 Add the card gets wrong.

**Card checks** (`pf_check.py`, 2 layers, Qwen3-4B dims, `check_<set>.json` and
`check_<set>_slices.json`; health 0.0 before and after; tolerance 2e-2; all programs below
2e-3 on both sets, the K-split inputs equal to their slices bit for bit):

| program (layers 0 and 1, worst over both) | worst rel |
|---|---|
| `down_proj_p0`, `down_proj_p1` (rest) | 1.8e-3, 5.2e-3 |
| `down_proj_hi_p0`, `_hi_p1`, `_lo_p0`, `_lo_p1` (first) | 1.3e-4 .. 5.7e-4 |
| `down_proj_add` (rest), `_hi_add`, `_lo_add` (first) | 6e-8 .. 0 |
| `gu_hi`, `gu_lo` (first) | 5e-5 |
| `gu_add_0`, `gu_add_1` (first) | 2.6e-6, 0 |
| `silu_f` (rest) | 6e-3 |
| `attn`, `lm_head` | 3e-3, 4e-5 |

**An error found on the way.** The first version took a K-split part's input offset as
2 bytes per element (U16). The card's program I/O is float32 (a 4864-wide linear's input is
19456 bytes), so part 1 read the wrong slice. `pf_check` did not catch it: the host read the
same bytes the card did. `pf_check` now reads the parent input and compares each part's input
with its slice (`check_<set>_slices.json`; all zero now), and `pf_plan.ITEM = 4` is the
element size of every program's I/O.

**Accuracy.** Teacher-forced, 48 steps of `docs/adaquant.md` (`pf_plan.py --doc 0 --prompt 48`,
the card against the float model of the same checkpoint):

2 layers (`q4b5`, the build of this section), 47 / 48 top-1 agreement, median 1.48% and p90
1.80% relative logit error (`plan_doc0.json`). The same two layers with the earlier
requantizing Add (each part on its own grid, `q4b4`): 48 / 48, median 1.49%, p90 1.83%.
Both are the same accuracy within the step noise.

**36 layers: not run.** The float reference of `pf_record` for the whole Qwen3-4B (the
dequantized 36-layer checkpoint, `pf_awq_snapshot.py`, 8 GB) was killed at the 24 GB cap
(`MemoryMax`) at record time. The 2-layer result is the only card result for Qwen3-4B here.
Layer 35 was therefore not checked on the card.

**Execute time** (`pf_profile.py --set rest`, per program, 2 layers, median of 10 runs each):

| program | us |
|---|---|
| `down_proj_p0` / `_p1` (K = 4864 each) | 935 / 939 |
| `down_proj_add` (N = 2560) | 304 |
| split down projection, total | 2178 |
| unsplit (estimate, below) | about 1600 |
| `gu` (K = 2560, N = 19456, rest set) | 2750 |
| `lm_head` (N = 151936) | 19710 |

Per token of the rest set, 2 layers (`s_per_token` 49 ms wall; execute 38.0-39.0 ms, median
of 47 steps, 24 runs per token). Per layer, the split down projection is 2.15 ms of about
9.26 ms (both halves 0.92 + 0.94 ms, the Add 0.29 ms). The unsplit estimate is 1.6 ms (above),
so the split costs about 0.58 ms per layer more. For a 36-layer Qwen3-4B at layout 1, the
split path is about 353 ms per token of execute (36 x 9.26 ms + `lm_head` 19.7 ms), and the
unsplit path about 332 ms (estimate). Both figures are for layout 1, the layout that runs
Qwen3-4B; the fused layouts are not possible for it. These are weight-bandwidth bound (each
linear streams its whole s8 table; the 0.27 ms execute floor is a small part).

Load average during the card runs was 2.4-3.5 (the host shares the machine; the profile
was run back to back with the decode).

**Estimated, not measured:** the unsplit down projection cannot be built (its table is
absent), so the unsplit figure is an estimate: one execute with the body of two K = 4864
halves (0.65 ms each, `935 - floor` with the floor about 0.27 ms, the doc's figure) plus one
floor, about 1.6 ms, against 2.18 ms for the split. About 0.6 ms per layer, about 21 ms per
token over 36 layers.

**Generating Qwen3-4B** (layout 1; the fused layouts are not possible):

```sh
# the float checkpoint of the AWQ int4 one (no float checkpoint is cached)
python pf_awq_snapshot.py <Qwen/Qwen3-4B-AWQ snapshot> <snap>            # LLM_PF_SNAPSHOT
# the absent template: one overlay with the layout's keys (builds k4864n2560 and add9728)
python pf_registry.py build --model qwen3_4b --layout 1 --out <overlay> --root <shared roots>
MODEL=qwen3_4b LLM_PF_SNAPSHOT=<snap> LLM_PF_OUT=<out> LLM_PF_FUSE=1 LLM_PF_TEMPLATES=<overlay> \
  python pf_record.py --n-first 128 --n-attn 16
python pf_generate.py both 0 1 --add-offset exact          # the card run above; the log: "split k9728n2560 ..."
# the whole model (both 0 35) is the same command on a 36-layer snapshot: not run (see Accuracy)
```

`pf_awq_snapshot.py` reads the AWQ `gemm` packing (nibble `i` is output column `ORDER[i]`,
autoawq's interleave). The order is not checked against a float reference (none is cached);
the per-channel scales do not discriminate it, so it is unverified.

**Not updated:** `pf_estimate.py` and `pf_fake_quant` (the host fake-quant of a whole decode)
do not know the split trees; use `pf_plan.py` and `pf_check.py` for Qwen3-4B.

**Byte-identity of the shipped layouts** (new code vs pristine `origin/master`, both sets
of layers 0-1, `--add-offset exact`, `LLM_PF_CAL_TARS=0`, the shared templates):

Qwen3-0.6B layouts 3, 5, 6, 7, 8, 9, 10, 11 and llama-160m layouts 3 and 7: every `.axmodel`
of layers 0-1 of both sets equal, and every manifest equal, to pristine `origin/master`
(`F/cmpset.py`: 0 differing files in each; 2 / 2 manifest entries). Layout 8 of
Qwen3-0.6B also equals the shipped `F/n8o` set (first 23, rest 3 axmodels, 0 differing).
Since no width of these models is measured absent, `pf_registry.split_keys` returns their
key lists unchanged, and the generation path is the pre-existing one.

## Fused K-split (Qwen3-4B layout 8): status 2026-10-29, layer 16 blocks the 36-layer run

**Status: layout 8 does not generate for 36 layers yet. Layers 0-15 (both sets) now generate,
including layer 5, which needed new template variants; the first refusal is at
`rest/L16/alyr.axmodel`, and that one is a calibration problem that no template variant can fix.** No card check was run for Qwen3-4B's fused layers
(the 36-layer set does not exist), and no Qwen3-4B ms per token was measured. Everything below
is measured on the host.

**The K-split program.** The down linear is split along K into two K = 4864 linears
(`KSPLIT_DOWN` in `pf_templates.py`: `Split` of the input, two `MatMul`s, one `Add`), both on one
output grid (`pf_generate.ksplit_grids`). One execute per layer is kept (`alyrqkrk4096`, and
`slanormk9728` for the last layer). Names: `pf_templates.KSPLIT_BASE` / `ksplit_name`;
`ModelSpec.ksplit_down` (only `qwen3_4b`) and `Config.ksplit` switch them on.

**Builds.** `k4864n2560`, `add9728`, `lanormk9728`, `slanormk9728`, `slanormqkrk9728` and the
six `alyrqkrk4096` variants v0-v5 were built earlier (`U`, heavy lock, each variant alone:
about 95 s each). The `KSPLIT_ALYR_EXTRA` variants v6-v12 (see below) were built on
2026-10-29 with a host build script (heavy lock, one at a time, MemoryMax 24G; 7 of 7 `ok`,
92-100 s each, peak 6.2 GB). The signature of `alyrqkrk4096` now has 13 variants
(`template_registry.json`: `2b7b2dda0b2f7e23`, replacing `3f81906688f5aae4`); only that key's
signature changed (64 keys of the three models at layouts 3-11 compared with `origin/master`).

**Layer 5 (fixed by the extra variants).** The six old variants refuse layer 5's `alyr` for
four reasons, all from the template's own calibration (`u16_retarget` ties and Add shifts;
`pf_generate._tied`):

* v0: `d_1`, `u_1`, `x_1` share a template zero point (`d_1` and `u_1` also share a scale).
  At this layer `x_1` is [-93, 173], `u_1` [-39, 83] (x2 copies included) and `d_1` [-19.6, 41],
  so `d_1`'s range would widen 4.39x in the union. v0 also ties `sk` / `sq` (`sq` widens 9.8x).
* v3: `d`, `u`, `x` tie; `d` widens 4.06x.
* v4: `ck` and `cq` tie (both 2.441e-5 / 24576); `cq` widens 21.9x (`cq` is [-0.21, 0.26], `ck`
  is [-2.0, 8.19] at this layer).
* v5: `cq` and `sk` tie; `cq` widens 8.6x.
* v1, v2: the residual Add `x = x_1 + b` (`n32`) needs Add shift 14 (`0x1ea0`, ratio in (1, 2])
  but all six templates have shift 15 there, so Pulsar2 left out that write (`_check_add_shift_writes`).

`KSPLIT_ALYR_EXTRA` (appended to the six, v6 on; a variant's draws are seeded by its index)
has seven variants `(lanorm variant, RoPE variant, ks)`: `(3,0,1)`, `(3,2,1)`, `(3,0,1.25)`,
`(3,1,1)`, `(3,3,1)`, `(3,0,0.8)`, `(3,2,0.8)`. Lanorm variant 3 (residual cancelled, `bk` 0.95)
was chosen because it is the one whose n32 Add can shift 14 in a template. A host prediction of
each template's calibration (the same 18 draws, `u16_retarget.calibrate_graph`; host probe, not
committed) matched the real zero points only to a few codes and did not reproduce every tie, so
the real builds and the guard decided. Real refusals for layer 5 `rest`: v6 and v7 ambiguous record match
(`0x1ea0 = 0xf` matches `n14` and `n19`), v8 `d_1/u_1/x_1` tie, v9 the same ambiguous match,
v0-v5 as above. **v10** (`(3,3,1)`) is accepted: layer 5 `rest` and `first` use
`alyrqkrk4096/v10` (checked with a host wrapper that prints each tie group's widen for every
variant tried; not committed).

**The 36-layer run (`pf_generate.py both 0 35 --add-offset exact`, `--jobs 1`, heavy lock,
MemoryMax 24G).** It generated `rest` and `first` for layers 0-15. The `alyr` variant chosen
per layer, `rest` / `first`: 0-4 `v1` (first from layer 3); 5 `v10` / `v10`; 6 `v3` / `v12`;
7 `v3` / `v10`; 8 `v1` / `v12`; 9 `v3` / `v1`; 10 `v1` / `v1`; 11 `v1` / `v10`; 12 `v1` / `v12`;
13 `v10` / `v1`; 14, 15 `v1` / `v1`. It stopped at `rest/L16/alyr.axmodel` with
`no usable template` for all 13 variants:

* v0 `sk`/`sq` tie (`sq` widens past 4x); v2 and v9 `k` / `uk` tie (`uk` past 4x); v3 `d`/`u`/`x`
  tie; v4, v5 `ck`/`cq`, `cq`/`sk` ties (as at layer 5);
* v1, v6, v7, v8, v10, v11, v12: `cannot convert float NaN to integer`. This is the real
  failure, and it is not a tie.

**Why layer 16's norm is NaN.** The second norm of `alyr` (after the MLP residual) takes
`x = x1 + down`. At layer 16 the recorded `rest` set has one row (row 43) whose `down` is
about 6300 at essentially one channel: that row's `x` reaches 6309 (12617 with the x2 copy), its
mean(x^2) is 16010 (the median row: 0.90). The coarse pass's `sq0 = Mul(x, x)` has the range [0, 1.59e8] (x2 copies
included) and so U16 scale 2429. Every ordinary row's `sq0` (0.5 to 500) quantizes to 0, then
`m0`, `e0` (eps) and `r0` are 0, and `u = x / r0` is NaN: the fake-quant `u` of 507 of 508 rows
is NaN (host probes, not committed). With row 43 left out of the calibration the scale is
0.0726 and the fake-quant `u` has a median relative error of 1.0e-3 on 200 rows, so the cause is
that one row. The card would compute the same zeros: the template cannot represent a per-tensor
U16 `sq0` whose range is 1e8 wide. In the `rest` set, layers 16-32 each have one row above 300
(about 6300 in magnitude; row 43 at 16-30, checked), layer 33 has 8 such rows, and layers 34 and 35
have 414 and 385. So the per-tensor MinMax calibration of the norms after the MLP residual fails
for these layers, whatever the template variant (checked at 16 only).
Fixing it needs a calibration policy or a template structure change (for example, a norm on
x scaled by a power of two per tensor, or clipping the `sq0` range); neither was attempted here,
and either needs a card check of the decode against float before it counts.

**Peak memory.** The 36-layer run used one worker and hit the 24 GB `MemoryMax` (systemd reports
a peak of 24G; no kill; the run ended with the RuntimeError at L16). The recording of the 36
layers (`pf_record.py`, layer-major, 4 layers resident) peaked at 14.7 GB.

**Byte identity of the shipped code.** Qwen3-0.6B layout 5, layers 0-1, `both`: `origin/master`
and this branch produce the same files (30 of 30 `axmodel`s byte-equal; rest 7, first 23;
(host runs, not committed). Qwen3-0.6B layout 8, layers 0-1 vs the shipped `F/n8o`: 26 of 26 byte-equal.
llama-160m was not regenerated; its shipped template signatures are unchanged.

**Not measured:** the host `FusedEval` checks of the K-split programs; the card checks (`pf_check`
on layers 0, a middle fused layer and 35; the 48-step `pf_plan` decode against float); any
Qwen3-4B ms per token. No card event occurred (`card-events.log` is untouched).

**Row-scaled norm probe (2026-10-29, `pf_rsnorm_probe.py`): Pulsar2 accepts it, the retarget does not.**
The fix candidate divides `x` by its per-row max `|x|` before the squares:
`s = ReduceMax(Abs(x))`, `xs = x / s`, `u = xs / sqrt(mean(xs^2) + eps)`. The scale cancels in
`u`, and every U16 tensor after the division lies in [-1, 1], so one row's outlier cannot set
the range of the squares. One program, U16 `layer_configs` on all seven op types, 18 calibration
rows (two massive-channel rows of opposite sign, +6309 and -4200), 2560 wide:

* **Pulsar2 (7.0-lite, AX650 target):** compiles it in about 20 s. Every op is accepted:
  `Abs`, `ReduceMax`, `Div`, `Mul`, `ReduceMean`, `Add`, `Sqrt`. The frontend keeps the eight
  nodes, the quantized model has `AxAbs`, `AxReduceMax` and `AxQuantizedDiv` / `Mul` / `Add` /
  `Sqrt` / `ReduceMean`, and the `.axmodel` is one `neu mode` node with one NPU subgraph (no CPU
  fallback). Nothing was refused. A native build with the same calibration is record-identical
  (`compare_jobs` empty); a different calibration changes segment 2, as expected.
* **Card (AX8850, `axcl-card.lock`, `health_check` 0.0 before and after; `card_final_a.json`):**
  ordinary rows read 1.98e-2 and 2.53e-2 rms relative error, and the host fake-quant of the
  input alone (the input's per-tensor U16 rounding, step 0.160 over [-4200, 6309]) reads
  1.99e-2 and 2.56e-2. So the card matches the float norm up to the input's own rounding; the
  norm adds nothing measurable. Outlier rows (+6309, -4200 at channel 4) read 1.3e-3 and 1.1e-3
  rms (their ordinary elements 4e-2 median); the input floor there is 4e-4 and 6e-4, so about
  1e-3 comes from the `xs` path. No NaN in any row. A row scaled to 1e-2 is below the input's
  step and reads 1.0 (informational; the calibration range sets it, not the norm).
* **Not supported yet: the retarget.** `u16_retarget.OPS` has no `Abs` or `ReduceMax`
  (`candidates` raises `not a retargetable U16 op`), so a calibrated copy of this norm cannot be
  placed in a layer program. Their record formulas (the ratio lanes for the reduce, the
  sign-free abs) would have to be learned from native builds at two calibrations and
  validated as retarget-equals-native on the card, as the other ops were. Until then layout 8
  stays at layer 16. Not built: the new template family, the 36-layer run, the host and card
  checks of the K-split programs, and any Qwen3-4B ms per token.

## Limits

- **Short context, no prefill.** The accuracy runs cover 48 teacher-forced steps of two
  passages, and free-running generation was run for four prompts of 48 tokens. The
  context is 128 positions (the cache has 129 rows), and there is no prefill: a prompt
  goes through one token at a time.
- **llama-160m's newline set covers one token.** `nl_tokens` is `"\n"` alone, from a
  sweep of the vocabulary at one position (position 20, layers 0..3). Other positions
  and contexts were checked only through the calibration passages and the prompts above.
- **Calibration covers positions 0..127 only.** Beyond that, ranges are unchecked.
- **Not in the repository.** Templates, recorded activations and program sets stay local
  scratch. Qwen3-0.6B's program set is 827 MB, and its templates are 49 Pulsar2 builds
  (four per fused / Add shape, one per linear shape; llama-160m: 29).
- **Layouts `4` to `6` and `8` need a per-head q / k norm** (`qkf`): llama-160m stops at layout `3`
  (`7` is layout 3 with `aon2` there).
- **Layout 8 (the Qwen3 default) needs the exact Add offsets.** Without them its mean
  logit error is layout 7's, 0.07 / 0.10 points above layout 6's, and it misses one more
  near tie. The last layer keeps two programs (`aon2g` + `dn`, the final norm), and the
  first set's split layers (0-2) and the layer after them keep their own programs
  (layouts 9-11 merge them). Its `alyrqkr2048` templates are eight Pulsar2 builds,
  about 0.9 GB.
- **Layouts 9-11 are opt-in.** The first set's last layer keeps `aon2g` + `dn` (its
  residual Add shifts 13). Layer 0's first norm (`n1qk`, 0.57 ms) stays its own program:
  it writes layer 0's K / V row, which the attention after it reads from the cache.
- **Even tables only.** Pulsar2's uneven Sigmoid / Silu tables (above) would be more
  accurate on wide ranges; generating them is not done.
- **Layouts 6 and 8 keep layout 5's first-set programs after a split layer.** The first
  set's layers 1-3 keep a `qkv5` of 7168 columns and `qk`.
- **Not ported from the scratch run.** The per-program card-vs-host diagnostics and the
  attention calibration-margin experiment (`m = 1.0 / 1.2 / 1.414` on recorded inputs) are
  left out; the margin in use is sqrt(2).
