# Decoding an `llm_build` LLM from the host, through the onnx-remote AXCL worker

`pulsar2 llm_build` compiles a llama-family model (llama, qwen3; see
"Supported architectures") into one `.axmodel` per
transformer layer plus a "post" model (final norm and `lm_head`). Everything
between those calls is the host's job: the embedding lookup, the KV caches,
the attention mask and the argmax. This note describes that host loop and the
worker features that let it run over the onnx-remote RPC.

| file | what |
| --- | --- |
| `scripts/axera/llm_layer_loop.py` | bf16 helpers, file discovery, `IOSpec`, `Host` (embedding, caches, mask), `decode()`, the backends, `compare_layers()` / `compare_decode()` |
| `scripts/axera/llm_reference.py` | numpy float32 reference of the same checkpoint, config-driven, with its own memory-mapped safetensors reader |
| `scripts/axera/run_llm_rpc.py` | CLI: decode a prompt (text, chat template or token ids) against a running worker |
| `tools/onnx-remote/python/onnx_remote_client.py` | numpy-only client for the onnx-remote v5 wire protocol |
| `tools/onnx-remote/remote_axcl_worker.cpp` | the worker: model cache, `io_info`, `unload`, resident inputs (`run_resident`, `reset_state`) |
| `tests/test_axera_llm_layer_loop.py` | offline tests (no device, no real checkpoint) |

## Layer I/O

Only the decode subgraph of a layer file is used: one token per call. With
H = hidden size, D = KV heads x head dim, S = `--kv_cache_len`, V = vocabulary,
in engine order:

| | name | type | shape | the host puts / gets |
| --- | --- | --- | --- | --- |
| in 0 | `K_cache` | bf16 | [1, S, D] | row `s` = `K_cache_out` of the token at position `s`; unused rows zero |
| in 1 | `V_cache` | bf16 | [1, S, D] | same for V |
| in 2 | `indices` | uint32 | [1, 1] | absolute position of the current token, 0..S-1 (RoPE is inside the layer) |
| in 3 | `input` | bf16 | [1, 1, H] | embedding (layer 0) or the previous layer's `output` |
| in 4 | `mask` | bf16 | [1, 1, S+1] | additive: 0.0 at `[0..pos-1]` and at `[S]` (the current token), -65536.0 elsewhere |
| out 0 | `K_cache_out` | bf16 | [1, 1, D] | this token's K row, copied to `K_cache[0, pos]` |
| out 1 | `V_cache_out` | bf16 | [1, 1, D] | this token's V row, copied to `V_cache[0, pos]` |
| out 2 | `output` | bf16 | [1, 1, H] | the next layer's `input` |

The post model takes `input` bf16 [1, 1, H] (the last layer's raw output; the
final RMSNorm is inside it) and returns `output` bf16 [1, 1, V], the logits.

None of the four sizes is written into the loop. `IOSpec` is read from the
compiled models: `IOSpec.from_io()` takes one layer's and the post model's
`io_info` (what `RpcBackend` does when no spec is given),
`IOSpec.from_axmodels()` / `from_model_dir()` read the same from the files'
protobuf (`axmodel_io()`: the first `neu mode` node of a layer file is the
decode subgraph, the second the prefill one). S, and with it the context
limit, is the row count of `K_cache`. The checkpoint's `config.json` gives
the layer count and is checked against the compiled sizes
(`IOSpec.check_model`), so a checkpoint of another shape is refused before
anything runs.

| build (`--prefill_len 128 --kv_cache_len 255`) | files | layers | H | D | S | V |
| --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | `llama_p128_l{i}_together.axmodel`, `llama_post.axmodel` | 30 | 576 | 192 (3 x 64) | 255 | 49152 |
| Qwen3-0.6B | `qwen3_p128_l{i}_together.axmodel`, `qwen3_post.axmodel` | 28 | 1024 | 1024 (8 x 128) | 255 | 151936 |
| JackFram/llama-160m | `llama_p128_l{i}_together.axmodel`, `llama_post.axmodel` | 12 | 768 | 768 (12 x 64) | 255 | 32000 |

The file names follow `{model_type}_p{prefill_len}_l{i}_together.axmodel` and
`{model_type}_post.axmodel`. They are discovered, not assumed:
`discover_files()` takes a directory listing and returns the prefix, the
prefill length and the layer count (and refuses an incomplete set, e.g. a
build that is still running); `DeviceBackend` lists its local directory.
A worker's directory cannot be listed over the protocol, so `run_llm_rpc.py`
lists a local copy (`--local-model-dir`, or `--model-dir` itself when that
path exists on the host too) and otherwise probes the worker with `io_info`
for `{model_type}_p{N}_l0_together.axmodel` over the usual prefill lengths.

numpy has no bfloat16, so the loop carries bf16 as `uint16` bit patterns: the
top 16 bits of the float32 pattern, rounded to nearest even (`f32_to_bf16`;
`bf16_to_f32` is exact). -65536.0 is `0xC780`, bytes `80 c7`.

## The host loop

`Host.step(token, backend)` feeds one token: look up its embedding row, build
the mask for position `pos`, call each layer with that layer's cache and the
running hidden state, write the returned K/V rows into slot `pos`, and call
the post model if logits are wanted. `decode()` prefills the prompt one token
at a time (post only on the last prompt token), then picks the argmax and
feeds it back. At most S tokens (prompt plus generated) fit; `decode()`
refuses a longer run before calling anything.

A backend is anything with `run_layer(i, feeds)` and `run_post(feeds)`:

- `NumpyBackend`: a float32 model of the contract above, taking and returning
  the same raw tensors. It attends over all S+1 slots and relies on the mask
  alone, so a wrong mask or cache slot gives wrong tokens offline too.
- `DeviceBackend(load, run)`: two caller-supplied callables,
  `load(path) -> handle` and `run(handle, arrays) -> arrays`, with an optional
  LRU of handles.
- `RpcBackend(client, model_dir)`: an onnx-remote AXCL worker. Paths are the
  worker's. When the worker advertises `resident_state` it keeps the caches
  on the worker (next section); `Host.K` / `Host.V` are then not the caches.

`compare_layers()` is teacher-forced: every layer call gets the reference's
hidden state and cache, so an error at (token, layer) is that call's own.
`compare_decode()` is free-running: it reports the tokens, the first
divergence with the reference's top-2 logit gap there, and the chained
per-layer hidden-state error. It also prints the reference's tokens with
every layer boundary rounded to bf16, and the smallest top-2 logit gap along
the reference's greedy path. Both matter when reading a mismatch: bf16 logits
near 16 are 0.125 apart, and rounding the hidden state to bf16 between layers
is by itself enough to change a token when the float32 gap is smaller than
that. Qwen3-0.6B on "The capital of France is" does exactly this offline: the
float32 reference continues " Paris. The capital of Italy is Rome ...", the
same weights with bf16 boundaries " Paris. The capital of France is also
...", the two candidates being 0.024 apart at the sixth new token. Prompts
with wide gaps are the ones to use for a token-identity check.

### What crosses the wire per token

The original request form has no tensor that stays on the worker: every layer
call sends that layer's two caches in full, 2 x S x D bf16 values, whatever
the position, and gets the token's K and V rows back so the host can store
them. With **resident KV caches** the caches live in the worker's device
buffers, which a cached model keeps allocated anyway. A layer call then sends
`indices`, `input` and `mask`, the worker copies `K_cache_out` / `V_cache_out`
into row `indices` of `K_cache` / `V_cache` itself, and only `output` comes
back.

Tensor payload per fed token (all L layer calls plus one post call; message
framing adds about 0.3 kB per call), computed from the shapes:

| build | | sent, host to worker | received |
| --- | --- | --- | --- |
| SmolLM2-135M (L 30, S 255, D 192, H 576) | caches in full | 5,926,392 bytes (197,508 per layer call) | 155,904 |
| | resident | 51,192 bytes (1,668 per layer call) | 132,864 |
| Qwen3-0.6B (L 28, S 255, D 1024, H 1024) | caches in full | 29,319,280 bytes (1,047,044 per layer call) | 475,904 |
| | resident | 73,840 bytes (2,564 per layer call) | 361,216 |

Per layer call that is 4 + 2H + 2(S+1) bytes instead of 4·S·D more, and 2H
bytes back instead of 2H + 4D. What is left of "received" is almost all the
post model's logits (2V: 98,304 and 303,872 bytes). The same bytes that no
longer cross the network also no longer cross from the worker to the card.

**Device timings** (per decoded token):

| build | caches in full (measured) | resident |
| --- | --- | --- |
| SmolLM2-135M | 118 ms per token (8.5 tokens/s) | 76 ms per token (13.2 tokens/s) |
| Qwen3-0.6B | 215 ms per token (4.7 tokens/s; 205.7 ms in layer calls) | 98 ms per token (10.2 tokens/s; 89.9 ms in layer calls) |

Measured on an AX8850 (AXCL V3.6.5, LXD VM, 2026-10-10) with this worker built against the
real SDK, both columns in the same session with the same binary; decode steps only. In both
modes the tokens are identical to the float32 reference (SmolLM2: 32 new tokens; Qwen3 with
the chat template: 28). The row write-back is the device-to-device copy at an offset, which
the runtime accepts. Also run: an fp16 hidden-state build in resident mode (identical, 13.3
tokens/s), the `--row-copy host` fallback (identical), and the older request forms against
the same binary. One prompt per model; these are not steady-state throughput figures.

How it works, end to end (the protocol is in the "AXCL worker" section of
`tools/onnx-remote/README.md`):

- **Protocol.** `run_resident` is `run` plus a text control in the request's
  `artifact` bytes. An input sent with no elements keeps its device buffer
  (the Hexagon worker's idiom for recurrent state); `write <out> <in> <slot>`
  copies an output into an input at `slot` x the output's byte size after
  execution; `omit <out>` leaves an output on the worker; `state <id>` names
  the state the request continues. `reset_state` zero-fills a model's inputs
  and returns a new id. The v5 wire format and the old request forms are
  unchanged.
- **Worker.** The caches are the cached model's own input buffers. The row
  copy is a device-to-device `axclrtMemcpy` into the input buffer at the
  row's offset (`--row-copy host` goes through the worker's memory instead,
  with whole-buffer copies only). Whatever ends the cache entry, or
  overwrites its inputs, ends the state, and the next request that refers to
  it is refused with `resident state lost` rather than run.
- **Loop.** `RpcBackend` uses the resident form when the worker's
  capabilities advertise `resident_state` (`resident=False`, or
  `run_llm_rpc.py --no-resident-kv`, keeps the original form). `Host.step`
  resets the worker's caches at position 0 and afterwards sends no K/V. The
  backend tracks the state id and the number of rows the worker holds per
  layer, and which conversation they belong to.
- **Lost state.** When the worker (or the backend's own bookkeeping) reports
  the state lost, the step rebuilds the caches and uploads them in full once:
  from the host's copy if it is complete, otherwise by feeding the tokens so
  far again through full-upload calls, which costs what those tokens cost
  without resident caches. After three losses without 16 clean tokens in
  between (a worker whose `--max-loaded` is below the layer count loses a
  layer on every token) the backend stops using the resident form.
- **Diagnostics.** `compare_layers()` sets every call's cache explicitly, so
  it always uses the full-upload form (which also ends the worker's state for
  that layer). To see the rows during a resident decode, set
  `Host.mirror_kv = True` (`--mirror-kv`): the rows are returned as well as
  written on the worker (2 x 2D more bytes per layer call), `Host.K` /
  `Host.V` stay complete, and a lost state is re-uploaded without replay.
- **fp16 hidden state.** A row written on the worker never leaves the
  device's format, so it needs no conversion; the kept inputs are sent as
  empty tensors under the dtype the worker reported. This is also a small
  numerical difference to the full-upload path on an fp16 build: there the
  host stores each returned row as bf16 (7 fraction bits against the 10 of the
  IEEE half the device produced) and converts it back on every call, so the
  device attends over rounded rows; resident rows stay exact. On a bf16
  build both paths hold the same bits. A replay after a lost state rebuilds
  the rows through the full-upload path.

Independently of all that, the host side adds no copies: a cache that is
sent goes to the client as it is (no conversion: it is already little-endian
uint16), and the client writes each large tensor to the socket from the
array's own memory (`encode_request_parts()` / `send_parts()`) where it used
to join the whole message into one buffer, which copied every payload three
times. On a local socket pair that takes one full Qwen3 layer request from
0.30 ms to 0.06 ms of host time (0.04 to 0.03 ms at SmolLM2's size). The
embedding table of a BF16 checkpoint is likewise used as stored, from the
file mapping, one row copied per token.

## Supported architectures

The loop itself has no architecture in it. Everything that distinguishes
these models happens inside the compiled layer, so on the host side an
architecture is its sizes (read from the compiled models, above), its file
name prefix, and its tokenizer and chat template.

| | `llama` (SmolLM2-135M, llama-160m) | `qwen3` (Qwen3-0.6B) |
| --- | --- | --- |
| attention | grouped-query when `num_key_value_heads` < heads (SmolLM2: 9 / 3); llama-160m is 12 / 12 | grouped-query, 16 / 8 |
| head dim | hidden / heads | `head_dim` from the config (128); heads x head_dim = 2048 is not the hidden size |
| q/k norm | none | RMSNorm over each head's `head_dim` on q and on k, before RoPE (`q_norm`, `k_norm`) |
| RoPE theta | `rope_theta` (SmolLM2 1e5; llama-160m has none in its config, HF's default 1e4) | 1e6 |
| `lm_head` | tied to the embedding (SmolLM2) or its own tensor (llama-160m) | tied; the file stores both |
| stored as | BF16 (SmolLM2), F32 (llama-160m) | BF16 |
| end of sequence | `eos_token_id` | two ids, ChatML's end of turn (151645) and end of text (151643); `generation_config.json` lists both |
| chat template | none in these two checkpoints | ChatML with an `enable_thinking` switch |

Where each of these lives:

- **In the compiled layer**: both layer norms, the q/k/v/o projections,
  q_norm/k_norm, RoPE, the attention itself and the MLP. The host passes the
  absolute position in `indices` and nothing else. This was checked in the
  compiled output, not assumed: the `npu_params` blob of a Qwen3-0.6B layer
  file (layers 0 and 27 looked at) contains, as float32, that layer's
  `q_norm.weight` and `k_norm.weight` (128 values each), both layer norm
  weights, and two 255 x 128 tables that equal
  `[cos, -sin]` and `[sin, cos]` of `pos * inv_freq` at theta 1e6 to 3e-8
  (in 32-wide blocks; at theta 1e4 or 1e5 they do not match at all). The
  same search finds the per-position rows at theta 1e5 in a SmolLM2 layer
  and at theta 1e4 in a llama-160m layer. That shows the weights and tables are in
  the layer; that the layer applies them correctly is what the device run
  against the reference shows.
- **In the post model**: the final RMSNorm and `lm_head`, tied or not.
- **On the host**: the embedding row (rounded to bf16 if the checkpoint is
  not BF16), the caches (D wide, whatever D is), the mask, the argmax, the
  stop ids, and the prompt.

`llm_reference.py` is the float32 model of all of it and is driven by
`config.json` and by which tensors the file has: `num_key_value_heads`
(default: heads), `head_dim` (default: hidden / heads), `rope_theta`
(default 1e4), `rms_norm_eps`, q_norm/k_norm when present, biases when
present, `lm_head.weight` when present and otherwise the embedding if
`tie_word_embeddings`. A layer tensor it does not implement is refused, as
are `rope_scaling` and an activation other than SiLU. The safetensors file
is memory-mapped: F32 tensors are used from the mapping, BF16 ones are
widened when first used and kept in an LRU bounded by `cache_bytes` (3 GiB
by default, which holds all of Qwen3-0.6B: about 2.4 GB once every layer
has run; 0 keeps nothing).

Reference against transformers 5.17 on CPU, float32, 16 greedy tokens
(`python scripts/axera/llm_reference.py CHECKPOINT --torch`):

| checkpoint | prompt | max logit difference | greedy tokens |
| --- | --- | --- | --- |
| SmolLM2-135M | "The capital of France is" | 2.5e-4 (logits up to 25) | identical |
| JackFram/llama-160m | ids `1,450,7483,310,3444,338` | 1.3e-4 (up to 21) | identical |
| Qwen3-0.6B | "The capital of France is" | 3.1e-5 (up to 19) | identical |

Offline, `NumpyBackend` with the `IOSpec` read from each real `llm_build`
output directory reproduces the reference's greedy tokens for all three
(for Qwen3 on prompts with wide logit gaps, see above).

### Prompts

`--prompt TEXT` is tokenized as it is. `--chat` first wraps it in the
checkpoint's chat template (`chat_template` in `tokenizer_config.json`, or
`chat_template.jinja`) as one user turn, with `--system TEXT` if given. The
template is rendered with jinja2 when that is installed, with the settings
transformers uses; without it, two shapes are written out by hand, Qwen3's
and plain ChatML with the template's default system message. For Qwen3 the
hand-written form, the jinja2 rendering and
`tokenizer.apply_chat_template(..., enable_thinking=False)` give the same
text and ids.

Qwen3's template has a thinking switch. `--chat` renders with
`enable_thinking=False`: the prompt then ends with
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, an empty reasoning
block, and the model answers directly. `--thinking` leaves that block out
(`enable_thinking=True`), and the model writes its reasoning between
`<think>` and `</think>` first; that takes far more than 16 new tokens, and
prompt plus output have to fit the 255-token context.

`--prompt-ids 1,2,3` takes token ids and needs no tokenizer; llama-160m's
checkpoint has none. `--stop-at-eos` stops at any of the checkpoint's end of
sequence ids. Decoding is greedy.

## The worker: model cache and `io_info`

A 30-layer model needs 31 model executions per token. The worker used to load
the model from its file on every request; it now keeps models loaded
(`--max-loaded N`, default 64, least recently used evicted first; `--no-cache`
for the old behaviour) with their context, IO object and device buffers, and
reloads one only when its file's modification time or size changes. `unload`
drops a model explicitly. The details are in the "AXCL worker" section of
`tools/onnx-remote/README.md`.

`io_info` returns, for a model path or artifact ID, the engine's input and
output names, ONNX dtypes, shapes and byte sizes as JSON, in engine order.
`RpcBackend` asks for it once per model (which also loads the model, so
`preload()` moves all 31 loads ahead of the first token) and sends each tensor
under the dtype the worker reported for it. The engine does not report the
bf16 tensors uniformly: the K/V/hidden tensors come back with an unknown type
code that the worker reads as FLOAT16, `mask` as BFLOAT16. The bytes are the
same 16-bit patterns either way; only the label on the wire differs, and the
worker refuses a tensor whose label is not the one it expects.

```sh
onnx-remote-axcl-worker --port 39501                    # where the card is
python scripts/axera/run_llm_rpc.py --host HOST --port 39501 \
    --model-dir /path/on/the/worker/out --checkpoint ~/models/SmolLM2-135M \
    --prompt "The capital of France is" --max-new-tokens 16 --compare-reference
python scripts/axera/run_llm_rpc.py --host HOST --port 39501 \
    --model-dir /path/on/the/worker/out --local-model-dir ~/builds/qwen3/out \
    --checkpoint ~/models/Qwen3-0.6B --chat --prompt "What is the capital of France?" \
    --max-new-tokens 32 --stop-at-eos --compare-reference
```

`--model-dir` is read by the worker. `--checkpoint` is read by the script (it
needs the embedding table, and the tokenizer unless `--prompt-ids` is given).
The script prints what it found (file names and how, the compiled sizes, the
context limit), the tokens and text, and a timing summary: per fed token the
mean, median, minimum and maximum wall time, separately for the prompt
(prefilled one token per step) and for the generated tokens, the split
between layer calls and post calls, and the time to the first new token.
With `--compare-reference` the post model runs on every prompt token too
(the comparison wants all logits), so prefill times are higher than in a
plain run. `--compare-layers` runs the teacher-forced per-layer check on the
prompt instead of decoding.

## Results

Offline (`tests/test_axera_llm_layer_loop.py`, random 2-layer checkpoints):
the loop with `NumpyBackend` gives the reference's greedy tokens; leaving the
mask at zero, or writing the cache rows one slot off, changes them; the same
decode through `RpcBackend` over a socket, against an in-process worker that
answers `io_info` and enforces the dtype labels, gives the same tokens; and
the Python client round-trips every transport dtype through the C++ reference
worker. With the caches resident on that worker (it implements
`run_resident` / `reset_state` with the real worker's semantics) the tokens
are the full-upload path's for the tiny llama and qwen3 configs, the bytes
per layer call are exactly the table's formula, an evicted layer mid-decode
is recovered (by replay, or by upload alone when the rows are mirrored), a
worker that loses the state before every token ends in full-upload mode, and
the old request forms answer as before. The same holds for a qwen3-style checkpoint (grouped-query, q_norm
and k_norm, `head_dim` 16 with hidden 32 and 4 heads, tied, theta 1e6) and a
llama-style one with one KV head for four heads, float32 storage and its own
`lm_head`; for both, sizes and file names are taken from files with the
structure of an `llm_build` output (two `neu mode` nodes per layer) or from
the worker by probing, and the CLI runs end to end against the in-process
worker. Reading the q_norm weights away, or the `lm_head`, changes the
tokens.

On the device, through the non-RPC session path (AX8850, AXCL V3.6.5, LXD
VM), SmolLM2-135M built with
`pulsar2 llm_build --prefill_len 128 --kv_cache_len 255` (30 layers + post),
prompt "The capital of France is":

- 16 greedy tokens, identical to the float32 reference.
- Teacher-forced, each layer's output is within 1-4% relative L2 of the
  reference, except layer 29 at token 0 (27%).
- Free-running, the chained error grows to about 18% by layer 28 and 70% at
  layer 29, while the tokens still match.

Through the RPC path (this worker built in the VM with the real AXCL SDK, `run_llm_rpc.py`
on the host, 2026-10-10): the same prompt gave the same 16 tokens, identical to the float32
reference and to the session-path run. The 31 models loaded once in 2.37 s; 620 worker
calls took 2.69 s, 134.5 ms per fed token (30 layers and the post model), against about
380 ms per token through the session path, whose cost is mostly per-call file transfer.
Every tensor of this build is reported by the engine as BF16 except `indices` (UINT32).
The worker's older request forms were rechecked on the device against the same binary: a
float32 model addressed by path (first call and cached call bit-exact) and an llm_build
layer whose fp16 tensors the engine reports with no dtype.

One prompt and 16 tokens only; nothing here measures accuracy or steady-state speed.

### Qwen3-0.6B on the device

Compiled with `pulsar2 llm_build --prefill_len 128 --kv_cache_len 255` (28 layers + post,
514 s, 636 MB). Run over RPC on the AX8850 (2026-10-10) with `--chat` (thinking off) and the
prompt "What is the capital of France?":

- 9 new tokens then the end-of-turn token, `The capital of France is **Paris**.<|im_end|>`,
  identical to the float32 reference and to the reference with bf16 layer boundaries
  (smallest top-2 logit gap on the path 2.91).
- Free-running hidden-state error against float: 3% after layer 1, 11% by layer 27, 23% at
  the last layer.
- 224 ms per fed token (213.7 ms in the 28 layer calls, 10.5 ms in the post model), 4.5
  tokens/s for prefill and decode alike.

### llama-160m on the device

Compiled the same way (12 layers + post, 124 s, 151 MB). The checkpoint has no tokenizer,
so the prompt is given as ids (`--prompt-ids`, a count from 1 to 6):

- 16 new tokens identical to the float32 reference (the count continues: 7, 8, 9, 10, 11);
  smallest top-2 logit gap 2.79.
- Free-running hidden-state error: 1.5% after layer 1, 4.7% at layer 11.
- 78 ms per fed token (76.2 ms in the 12 layer calls, 2.8 ms in the post model), 12.8
  tokens/s.

### SmolLM2-135M again, and other dtypes

With this change's client (large tensors sent from the array's memory), SmolLM2-135M at the
default dtypes gives the same tokens at 110 ms per decoded token (9.1 tokens/s).

`llm_build` variants of SmolLM2-135M, same prompt, 32 new tokens, against the float32
reference:

| `--weight_type` | `--hidden_state_type` | size | tokens vs float | chained hidden error | speed |
| --- | --- | --- | --- | --- | --- |
| `s8` (default) | `bf16` (default) | 140 MB | identical (16 tokens) | 3% rising to 18% by layer 28 | 9.1 tokens/s |
| `s4` | `bf16` | 88 MB | diverges at the 2nd new token (float top-2 gap there is 0.17); text stays fluent | 14% after layer 1, peaks of 31% and 60% at layers 10 and 11 | 9.4 tokens/s |
| `s8` | `fp16` | 140 MB | identical | 1% to 3% throughout | 8.8 tokens/s |
| `bf16` | `bf16` | 438 MB | identical | 2% rising to about 15% | 8.3 tokens/s |
| `fp16` | `bf16` | 438 MB | identical | same as `bf16` weights to four digits | 8.0 tokens/s |
| `fp8_e4m3` | `bf16` | 438 MB | diverges at the 2nd new token (the same 0.17 near-tie); fluent, then repeats | 4% rising to 15%, bump to 12% at layers 10 and 11 | 7.9 tokens/s |
| `fp8_e5m2` | `bf16` | 438 MB | diverges at the 1st new token (float top-2 gap 1.03); fluent, repeats | 5% and 10% early, 34% and 70% at layers 10 and 11, about 30% after | 8.1 tokens/s |
| `s4` | `fp16` | 88 MB | diverges at the 2nd new token (the 0.17 near-tie); fluent | 14% and 26% early, 26% and 46% at layers 10 and 11, under 5% after | 8.6 tokens/s |

An fp16 hidden state needs real IEEE halves on the wire. The engine reports those tensors
with no dtype, the worker labels them FLOAT16, and the loop converts its bf16 patterns at
the RPC boundary. Before that conversion existed the same build produced infinities.

The four float weight types all cost 4 bytes per weight on the card (see
`docs/axera-llm-build-dtype-analysis.md`), so `fp8_*` loses precision without saving memory.

**`s4` on Qwen3-0.6B** (411 MB against 636 MB for `s8`), `--chat`, thinking off:

| prompt | `s8` (default) | `s4` |
| --- | --- | --- |
| "What is the capital of France?" | identical to float: "The capital of France is **Paris**." | "The capital of France is **Lille**. ..." -- diverges at the 7th new token, where the float model's top-2 gap is 4.47, and does not stop |
| "Write one sentence about the ocean." | identical to float (28 tokens) | diverges at the 8th new token (gap 1.37); ends "...of life and life, where life and life are ever-changing." |

`s4` hidden-state error is 31% and 35% after the first two layers and 10% to 20% after
(`s8`: 3% to 11%). It is faster, 5.2 against 4.6 tokens/s. So on this model plain `s4` from
`llm_build` changes answers the float model is confident about; `s8` did not on these two
prompts.

One or two prompts per model. Nothing here measures accuracy; the SmolLM2 `s4` and
`fp8_e4m3` rows are a single divergence at a near-tie, the Qwen3 `s4` rows are not.

The worker binary used for these runs is the one built in the VM against the real AXCL SDK,
before resident KV caches existed: every number above is the full-upload path's.

### Resident KV caches: what was and was not run

`remote_axcl_worker.cpp` with `run_resident` / `reset_state` was syntax-checked against the
stub header and against the AXCL SDK's headers, and linked against a scratch fake AXCL (not
in the tree: host memory as device memory, outputs a hash of all input bytes). Against that
binary the Python client ran 36 resident calls with SmolLM2-135M's layer shapes, bit-identical
to the full-upload form in both `--row-copy` modes, every lost-state case, and a `Host` /
`RpcBackend` decode with one layer unloaded mid-way. **Nothing of it has run on the card**:
the device-to-device copy into an offset of an input buffer, the timings and the token
identity on real layers are still to be checked there (`run_llm_rpc.py` with and without
`--no-resident-kv` on the same prompt; `--row-copy host` on the worker if the offset copy is
refused).
