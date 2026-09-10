# nanochat × onnxsim — captured results

Result of running `simplify_nanochat.py` at nanochat's own depth-based sizes
(`config_for_depth()`, reproducing `scripts/base_train.py`'s
`build_model_meta()`), plus the tiny fixed config used for
`nanochat_wasm_demo.onnx`. Each export/simplify passed onnxsim's `check_n=3`
numerical equivalence check.

## Environment

| package | version |
|---|---|
| onnxsim | 0.7.3 |
| onnx | 1.22.0 |
| onnxruntime | 1.29.0 |
| torch | 2.14.0+cpu |
| Python | 3.11.15 (Linux, CPU) |

Export opset 17, static batch size 1, random weights (structure is identical
across weight values, same reasoning as `scripts/yolo`/`scripts/rfdetr`).

## Results

| Config | n_layer | n_embd | n_head | n_kv_head | seq_len | vocab | Params | Nodes before → after | Reduction | `check_ok` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|:---:|
| `--depth 4` | 4 | 256 | 2 | 2 (MHA) | 64 | 2048 | 4.2M | 458 → 274 | 40% | ✅ |
| `--depth 4 --n-kv-head 1` | 4 | 256 | 2 | 1 (GQA) | 64 | 2048 | 3.9M | 586 → 302 | 48% | ✅ |
| `--depth 12` | 12 | 768 | 6 | 6 (MHA) | 64 | 2048 | 88.1M | 1314 → 786 | 40% | ✅ |
| `--tiny` (wasm demo) | 1 | 16 | 2 | 1 (GQA) | 8 | 32 | 3,840 | 169 → 89 | 47% | ✅ |
| `--checkpoint` (trained, `train_tiny.py`) | 2 | 64 | 4 | 2 (GQA) | 64 | 59 | 98,304 | 308 → 160 | 48% | ✅ |

`--depth 12` is nanochat's own reference size ("Our reference model is d12,
this is where a lot of hyperparameters are tuned"); `--depth 4` is the size
used in `base_train.py`'s own quick smoke-test invocation
(`--depth=4 --max-seq-len=512 ... --num-iterations=20`). The real
`--depth 20` default (base pretraining size, ~561M matmul params before the
embeddings) also exports and simplifies the same way -- structurally
identical, just larger -- but takes several minutes for onnxsim's
`onnxruntime`-based `check_n=3` pass at that parameter count, so it isn't
included in this table.

**5 / 5 configurations simplify and pass onnxsim's numerical check.**

## What onnxsim removes

Consistently ~40–48% fewer nodes across every size and GQA/MHA combination.
The reduction comes from:

- **Rotary embeddings.** `cos`/`sin` are model buffers sliced to the current
  sequence length every forward pass (`self.cos[:, :T]`) -- onnxsim's
  constant folding resolves the `Slice` against the buffer once, and folds
  the `-sin`/concat arithmetic in `apply_rotary_emb` down to the pieces that
  actually depend on the input.
- **RMSNorm's `Pow`/`ReduceMean`/`Add`/`Sqrt` chain** and the **logit
  softcap's** `Div`/`Tanh`/`Mul` -- the divisor and epsilon constants fold
  in, and shape-inference resolves the broadcasted keepdim shapes.
- **The causal mask.** `torch.triu(..., float("-inf"))` is itself a model
  buffer, sliced to `[:T, :T]` the same way as the rotary buffers -- folds to
  a single constant per exported sequence length.
- **GQA's `repeat_interleave`** exports as `Unsqueeze`+`Tile`+`Reshape`
  around the attention matmuls; onnxsim's shape inference collapses the
  static-shape bookkeeping around it once the tile count is known.

The `--n-kv-head 1` (GQA) row consistently reduces slightly more than the
matching MHA row (48% vs. 40% at depth 4) -- GQA's extra `repeat_interleave`
scaffolding is exactly the kind of static-shape arithmetic onnxsim folds
away, so it starts with more foldable nodes per layer.

## Notes

- **`onnxruntime` is required for the check.** Without it, onnxsim falls
  back to onnx's slower Python reference evaluator for `check_n`.
- **The trained checkpoint simplifies identically to a random-weight model
  of the same shape** (308 → 160, 48%, same as the `--n-kv-head 1` depth-4
  row scaled down) -- confirming that, as with YOLO/RF-DETR, the structural
  reduction onnxsim performs here doesn't depend on the weight values, only
  the graph shape.
- See [`README.md`](README.md) for what's reimplemented vs. upstream
  nanochat, the training + loss-curve demo, and how to load
  `nanochat_wasm_demo.onnx` directly in the browser wasm UI.
