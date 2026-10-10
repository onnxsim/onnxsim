# Better int4 weights for a compiled `llm_build --weight_type s4` model

`pulsar2 llm_build --weight_type s4` rounds every weight to the nearest of 15 or 16 levels
per output row. On Qwen3-0.6B that changes answers the float model is confident about
(`docs/axera-llm-rpc-decode.md`). `scripts/axera/llm_int4_requant.py` replaces those codes
with activation-aware ones **inside the compiled files**, without recompiling: same files,
same size, same NPU program, only the weight-block bytes differ.

```
python scripts/axera/llm_int4_requant.py --src OUT_S4_DIR --checkpoint CHECKPOINT_DIR \
    --dst NEW_DIR --method gptq --calibration calib.json notes.txt "some literal text"
```

`--calibration` items are JSON files of token-id lists (`[[1, 2], ...]` or
`[{"ids": [1, 2]}, ...]`, used as they are, so chat-templated sequences keep their
template), text files (tokenized with the checkpoint's `tokenizer.json`, cut into
`--seq-len` tokens), or literal text. The result runs like any other `llm_build` directory
(`run_llm_rpc.py --local-model-dir NEW_DIR ...`). The worker caches models by path, mtime
and size: give the copy a new path on the worker, or `unload` first.

Tests: `tests/test_axera_llm_int4_requant.py`, offline, on about 277 KB of committed slices
of our own builds (`scripts/axera/fixtures/llm_int4_requant/`) and a synthetic compiled
directory.

## Byte map

Everything below was read from our own builds: SmolLM2-135M and Qwen3-0.6B compiled with
`--weight_type s4 --hidden_state_type bf16`, and the tiny 256-hidden Llama of
`docs/axera-llm-build-dtype-analysis.md`.

### What the real models changed

| | tiny 256-hidden model | SmolLM2-135M | Qwen3-0.6B |
| --- | --- | --- | --- |
| quantizer | rule A: `s = -w[argmax\|w\|]/8`, `q = floor(w/s + 0.5)`, codes in [-8, 7] | rule A. 3,535 of 106,168,320 codes differ: every one is an exact .5 tie that the compiler rounded down. Its tie rule is not identified | **rule B**: `s = max\|w\|/7` (always positive), `q = rint(w/s)` (half to even), codes in [-7, 7]. 0 of 440,401,920 codes differ, every scale is bit-equal |
| column parts | 256, 512: one part; 2048: 416, 544 x3 | 576 is **one** part; 1536: 448, 544, 544 | 1024: 480, 544; 2048: 416, 544 x3; 3072: 352, 544 x5 |
| row blocks | 32 rows | o/gate/up/down: 32 rows. **q/k/v: 64 rows** | 32 rows everywhere |
| post model | not looked at | **s8 whatever `--weight_type` says**: its `npu_params` is byte-identical to the s8 build's | the same |

What selects rule A or rule B, and 32 or 64 rows, is not known. The script reads both from
the file: the row-block size from which layout validates, the rule from the codes
(`detect_rule`: under A every row holds code -8, its peak; under B no code is -8 and every
scale is positive).

**Column parts.** The first part holds up to 576 columns (16 chunks of 36) and every later
part exactly 544; there are as few 544-wide parts as leave the first one at most 576 wide:
`n = max(0, ceil((cin - 576) / 544))`, widths `[cin - 544 n] + [544] * n`. This fits every
width built so far (256, 512, 576, 1024, 1536, 2048, 3072, and 4096 for s8). Only
`cin = 576` separates it from the earlier "split once `cin > 544`" reading, which gave
32 + 544 there. No width with `cin mod 544` between 1 and 31 has been built.

The embedding is in no compiled file: the host looks it up from the checkpoint.

### A row block

A Linear `[rows, cin]` is `rows / RB` row blocks, `RB` = 32 or 64, each the concatenation of
its column parts. One column part of width `w`, with `ch = ceil(w / 36)` chunks:

| bytes | content |
| --- | --- |
| `RB * 18 * ch` | nibble codes `q + 8`; byte = `(code[2j+1] << 4) \| code[2j]`; columns from `w` up to `36 * ch` hold 8 |
| `4 * RB` | `RB` int32, Q16.16 of `-sum(q)/2` over **this part's** columns |
| `4 * RB` | zero |
| `4 * RB` | `RB` float32 scales, the same in every part of the row |

The code bytes are units of `36 * RB / 32` bytes. Unit `(k, c)`, for row pair `k` in 0..15
and chunk `c`, holds 18 bytes each of chunk `c` of rows `2k`, `2k + 1` and, when `RB` is 64,
`2k + 32`, `2k + 33`. For `RB` = 32 this is the layout of the dtype-analysis doc.

### A layer file

The tensors come in the order v, k, q, o, gate, up, down, in every layer of all three
models. SmolLM2 (`npu_params` 1,943,428 B, of which 1,869,696 B are weight blocks):

| bytes | content |
| --- | --- |
| [0, 5636) | 256 x `1/sqrt(64)`, `input_layernorm` weight (float32, at 1024), 2304 zero bytes, float32 32767.0 |
| [5636, 63236) | `v_proj`, 3 blocks of 19,200 B (64 rows) |
| [63236, 82436) | `k_proj` block 0 |
| [82436, 147716) | RoPE table (255 x 64 float32), identical in every layer |
| [147716, 186116) | `k_proj` blocks 1-2 |
| [186116, 358916) | `q_proj`, 9 blocks of 19,200 B |
| [358916, 531716) | `o_proj`, 18 blocks of 9,600 B |
| [531716, 534020) | `post_attention_layernorm` weight |
| [534020, 994820) | `gate_proj`, 48 x 9,600 B |
| [994820, 1455620) | `up_proj`, 48 x 9,600 B |
| [1455620, 1942916) | `down_proj`, 18 x 27,072 B (3 column parts) |
| [1942916, 1943428) | 128 x `1/sqrt(64)` |

Qwen3 (`npu_params` 8,939,524 B, weight blocks 8,663,040 B): constants and
`input_layernorm` in [0, 9220); v [9220, 586756); k [586756, 658948) and
[921092, 1426436), with the `k_norm` weight and the RoPE tables between; q
[1426436, 1498628) and [1499140, 2582020) with the `q_norm` weight between; o
[2582020, 3737092); `post_attention_layernorm` [3737092, 3741188); gate
[3741188, 5473796); up [5473796, 7206404); down [7206404, 8939012); 512 B of
`1/sqrt(128)`. In both models the first head of k (and of q on Qwen3) is separated from
the rest by non-weight constants.

`discover_layout` does not use these offsets. It takes the seven shapes from the
checkpoint's config and walks the table in tensor order: a block starts where the previous
one ended or, failing that, at the next offset whose tail has its zero bytes; a block
counts only if its zero bytes, pad nibbles and row sums check. Both row-block sizes are
tried and the one whose first block comes first wins. On all 30 + 28 layer files this
gives the offsets found earlier by searching for each row's scale bytes. The stored scales
must then equal the plain-rule scales of the checkpoint bit for bit, or the directory is
refused. A build with deduplicated (byte-identical) blocks would be refused too: none of
these files has any.

### What a re-quantization has to rewrite

Per column part: the codes, the row sums, the scales. Nothing else depends on them:

- **Other `npu_params` bytes.** Between two layers of one model only the norm weights
  differ outside the weight blocks, although every scale differs. No constant derived from
  a weight scale lives there.
- **MCode.** Decoded with `short_unit_codec`, all 15 segments of both subgraphs have the
  same record streams in layers 0, 1, a middle one and the last (Qwen3: all identical;
  SmolLM2: only `a8/0170` and `a8/0230` differ, the two streams the dtype-analysis doc
  lists as rebuild noise). The raw MCode of SmolLM2's layers 0 and 1 differs in one byte,
  the layer index in a name.
- **Post model.** s8; copied.

## Quantizers

All keep the format: one float32 scale per output row, codes in the range the build already
uses (rule A: [-8, 7] with the scale's sign opposite to the row peak's; rule B: [-7, 7]
with positive scales).

- **`plain`**: the compiler's rule, from the checkpoint. On a rule-B build it reproduces
  the compiled files; on a rule-A build it differs at the ties above.
- **`gptq`**: per Linear, `H = X^T X` of its input over the calibration tokens. Columns
  are rounded one at a time with GPTQ/OBS error compensation (1% damping, blocks of 128
  columns) at seven per-row scale candidates (1.0 down to 0.37 of the plain scale); the
  scale is then refit in closed form, `s = (q H w) / (q H q)`; each row keeps the candidate
  with the lowest H-weighted error.
- **`gptqa`**: the same, applied to a re-targeted weight. `W' = W (X_f^T X_q + lam I)
  (X_q^T X_q + lam I)^-1` makes the Linear reproduce the *float* path's output from the
  *quantized* path's input; `lam = ridge * mean diag(H)`, `--ridge` 0.3 by default. It is a
  regression with up to 3072 inputs per row fitted on a few thousand tokens, so it can
  overfit a small calibration set. `gptq` is the conservative one.
- **`identity`**: writes the decoded codes back. The output must equal the input byte for
  byte; it is a self-check of the layout and the encoder.

Calibration is layer-sequential and sequential inside a layer (q/k/v, then o, then
gate/up, then down). Every `H` comes from the numpy reference (`llm_reference.py`) running
through the Linears quantized so far, with bf16 rounding of the hidden state at layer
boundaries.

Tried and not ported: per-row MSE clipping without activations (worse KL than plain on
both models, below). Not implemented: AWQ. The format has no per-column scale, so plain AWQ
does not fit. Two foldings would fit and were **not implemented**: (i) column scales of
`o_proj` / `down_proj` absorbed into the float row scales of `v_proj` / `up_proj` (for
`o_proj` the scale must be shared by the query heads of one KV head); (ii) column scales
of q/k/v and gate/up folded into the float32 layernorm weights in `npu_params`, which
means rewriting bytes outside the weight blocks.

## What the patcher guarantees

`requantize_directory` copies the directory, rewrites the layer files, and reads each one
back from disk. It raises unless the file parses, has the same size, differs from the
source only inside the weight blocks (so the MCode and every other constant are
identical), and decodes to exactly the intended codes and scales with consistent zero
bytes, pad nibbles and row sums.

Checked offline against the scratch runs this was ported from:

| run | result |
| --- | --- |
| `identity`, SmolLM2 (30 layer files) and Qwen3 (28) | byte-identical to the compiled directories |
| `plain`, Qwen3 | byte-identical to the compiled directory (rule B is exact) |
| `plain`, SmolLM2 | 7,429 bytes differ over 30 files: the 3,535 tie codes and their row sums |
| `gptq`, SmolLM2, the 5,212-token calibration set | byte-identical to the earlier patched directory, in a second process on the same machine (228 s) |

The Qwen3 `gptq` directory was regenerated with the ported script (25 minutes, 2.6 GB peak) and
is byte-identical, all 29 files, to the directory that was run on the device. The two `gptqa`
directories were not regenerated with the ported script.

## Evaluation on an emulator (not the device)

The emulator (`emulator`, `emulate_directory`) is the numpy reference with every Linear
replaced by `q * s`, bf16 rounding at layer boundaries and the s8 `lm_head`. Arithmetic
inside a layer is float32.

### How close the emulator is to the device

Device outputs are the plain-`s4` ones already in `docs/axera-llm-rpc-decode.md`.

| case | device | emulator with Pulsar2's own codes |
| --- | --- | --- |
| Qwen3, chat, "What is the capital of France?" | "The capital of France is **Lille**. ...", diverges from float at the 7th new token | the same text start, first divergence at token 7 |
| Qwen3, chat, "Write one sentence about the ocean." | diverges at the 8th new token; ends "...of life and life, where life and life are ever-changing." | diverges at token 8; the same sentence |
| SmolLM2, "The capital of France is" | diverges at the 2nd new token | diverges at the 2nd new token with the same token; tokens 1-4 equal the device's, **token 5 differs** |

So the emulator matched the device on the two Qwen3 prompts and only partly on SmolLM2.
At SmolLM2's token 5 the device's choice is the emulator's rank 9, 1.7 logits down: not a
tie, the device's arithmetic inside a layer is not float32. The emulator's own SmolLM2
tokens also change with the BLAS thread count. Treat its SmolLM2 token-level predictions
as indicative only.

### Held-out prompts, 32 new tokens, greedy

**Emulator numbers.** Teacher is the float32 reference. "ident": prompts whose greedy
tokens equal the float model's. "1st div": median position of the first differing new
token (33 when identical). "match": position-wise token agreement of the free-running
output. "top-1" and KL(float || variant) are teacher-forced on the float path. No held-out
prompt is in the calibration set.

SmolLM2-135M, 24 text-completion prompts; calibration 60 sequences, 5,212 tokens (28 prose
sequences, 32 float-model continuations of other stems):

| variant | ident | 1st div | match | top-1 | mean KL | "The capital of France is" |
| --- | --- | --- | --- | --- | --- | --- |
| float + bf16 boundaries + s8 head | 9/24 | 16.5 | 0.595 | 0.971 | 0.0015 | " the capital of the country. ..." (= float) |
| s8 (formula) | 9/24 | 15.5 | 0.570 | 0.961 | 0.0034 | = float |
| plain s4 (Pulsar2's codes) | 0/24 | 1 | 0.061 | 0.697 | 0.479 | " the of the French people. It is the seat of the French parliament, ..." |
| mse (not ported) | 0/24 | 1.5 | 0.052 | 0.552 | 0.938 | " called the ______. A. France B. ..." |
| **gptq** | 1/24 | 2.5 | 0.147 | 0.831 | 0.128 | " the country's largest city. The capital of France is Paris. ..." |
| **gptqa** (ridge 0.3) | 1/24 | 2 | 0.152 | 0.833 | 0.101 | " the city of Paris. The city is the largest city in the country ..." |

Qwen3-0.6B, 26 chat prompts (thinking off; 20 factual, 6 open), stopping at end of turn;
calibration 68 sequences, 4,590 tokens (20 prose sequences, 48 chat-templated questions
with the float model's answers):

| variant | ident | 1st div | match | top-1 | mean KL | "What is the capital of France?" |
| --- | --- | --- | --- | --- | --- | --- |
| float + bf16 boundaries + s8 head | 24/26 | 33 | 0.968 | 0.998 | 0.0003 | "The capital of France is **Paris**." |
| s8 (formula) | 20/26 | 33 | 0.883 | 0.987 | 0.0033 | "The capital of France is **Paris**." |
| plain s4 (Pulsar2's codes) | 0/26 | 3.5 | 0.279 | 0.659 | 1.024 | "The capital of France is **Lille**. ..." |
| mse (not ported) | 1/26 | 5.5 | 0.224 | 0.677 | 1.056 | "The capital of France is Paris." |
| **gptq** | 5/26 | 8 | 0.522 | 0.867 | 0.159 | "The capital of France is **Paris**." (identical to float) |
| **gptqa** (ridge 0.3) | 7/26 | 12 | 0.628 | 0.889 | 0.126 | "The capital of France is **Paris**." (identical to float) |

The s8 rows use the formula quantizer, not codes decoded from the s8 builds. Typical
H-weighted relative output error per Linear: 0.01-0.05 for plain, 0.001-0.006 for gptq.
The difference between plain and gptq is large on this set; differences among the gptq
variants (gptqa; more compensation passes; other damping or ridge values) are not resolved
by 24 or 26 prompts.

## On the device

AX8850, AXCL V3.6.5, 2026-10-10. Qwen3-0.6B `s4` over RPC (`run_llm_rpc.py`) with the chat
template, thinking off. Eight prompts, each compared with the float32 reference.

| prompt | plain `s4` (Pulsar2's codes) | `gptq`-patched `s4` |
| --- | --- | --- |
| capital of France | "Lille" | "The capital of France is **Paris**." -- identical to the float reference |
| 12 times 12 | answered after switching to Arabic text | "12 times 12 is 144." |
| author of Romeo and Juliet | a garbled non-answer | "Romeo and Juliet was written by Shakespeare." |
| boiling point of water | "100.01°C (68.01°F)" | "100°C at standard atmospheric pressure." |
| "good morning" in Spanish | "Buenas tardes" | "Buenos días." |
| one sentence about the ocean | degenerates | a fluent sentence, diverging from float at the 13th new token |
| mixing blue and yellow | "it's red" | answered like the float model (which also says "blue-yellow") |
| largest planet | "Mars" | **wrong**: "Saturn" (float: Jupiter) |

- Plain `s4` is wrong or degenerate on all eight and rarely stops at the end-of-turn
  token. The patched model has one wrong answer, and every answer ends at the end-of-turn
  token.
- Same files, same speed, same size (411 MB against 636 MB for `s8`). The patched files
  differ from the compiled ones only inside the weight blocks.
- The coherent output confirms the row-sum convention (`-sum(q)/2` per column part,
  recomputed for the new codes). Had it been misread, every row would carry a bias-like
  error from layer 0 on.

### Limits

- Eight prompts. This is not an accuracy measurement.
- The patched model is still clearly below `s8`.
- The calibration set is about 4,600 tokens, partly generated by the float model on
  question styles similar to the prompts.
- Eight prompts cannot separate `gptq` from `gptqa` (below), and both get the planet wrong.
- The two AWQ-style foldings are not implemented.
- Not verifiable offline: SmolLM2's compiled scales happen to be bf16-representable, the
  new ones are arbitrary float32. If that kernel read scales at lower precision the result
  would shift by up to 0.4% per row. Qwen3's compiled scales are already full float32.

### `gptqa` on Qwen3, and SmolLM2

The scratch-generated directories, same device session and prompts:

| prompt | float | `gptq` | `gptqa` |
| --- | --- | --- | --- |
| capital of France | Paris | Paris (identical) | Paris (identical) |
| one sentence about the ocean | fluent | fluent, diverges at the 13th new token | fluent, diverges at the 8th |
| 12 times 12 | **144** | 144 | 144 |
| who wrote Romeo and Juliet | William Shakespeare | Shakespeare | William Shakespeare |
| boiling point of water | 100°C | 100°C | 100°C |
| largest planet | Jupiter | **Saturn** | **Mercury** |
| mixing blue and yellow | "blue-yellow" | "blue-yellow", like float | green |
| "good morning" in Spanish | Buenos días | Buenos días. | Buenos días (identical) |

SmolLM2-135M (`llama` prefix, 64-row q/k/v blocks), three completion prompts, 24 new
tokens; none of the three variants is identical to float on any prompt:

| prompt | plain s4 | `gptq` | `gptqa` |
| --- | --- | --- | --- |
| "The capital of France is" | "the of the French capital. ..." | "the largest city in the country, ... Paris is the cultural and political center" | "the city of Paris. ..." |
| "Water boils at a temperature of" | "about 100 degrees Celsius." | "100 degrees Celsius." | "100 degrees Celsius." |
| "The largest planet in the solar system is" | "the planet Mercury." | "Jupiter." | "Jupiter." |

So the patched files also run correctly on the 64-row layout, and both methods are better
than plain on SmolLM2 as well. All three repeat themselves after the first sentence.
