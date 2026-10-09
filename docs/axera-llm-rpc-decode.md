# Decoding an `llm_build` LLM from the host, through the onnx-remote AXCL worker

`pulsar2 llm_build` compiles a llama-style model into one `.axmodel` per
transformer layer plus a "post" model (final norm and `lm_head`). Everything
between those calls is the host's job: the embedding lookup, the KV caches,
the attention mask and the argmax. This note describes that host loop and the
worker features that let it run over the onnx-remote RPC.

| file | what |
| --- | --- |
| `scripts/axera/llm_layer_loop.py` | bf16 helpers, `IOSpec`, `Host` (embedding, caches, mask), `decode()`, the backends, `compare_layers()` / `compare_decode()` |
| `scripts/axera/llm_reference.py` | numpy float32 reference of the same checkpoint, with its own safetensors reader |
| `scripts/axera/run_llm_rpc.py` | CLI: decode a prompt against a running worker |
| `tools/onnx-remote/python/onnx_remote_client.py` | numpy-only client for the onnx-remote v5 wire protocol |
| `tools/onnx-remote/remote_axcl_worker.cpp` | the worker: model cache, `io_info`, `unload` |
| `tests/test_axera_llm_layer_loop.py` | offline tests (no device, no real checkpoint) |

## Layer I/O

Only the decode subgraph of a layer file is used: one token per call. With
H = hidden size, D = KV heads x head dim, S = `--kv_cache_len`, V = vocabulary
(SmolLM2-135M at `--kv_cache_len 255`: 576, 192, 255, 49152), in engine order:

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
  worker's.

`compare_layers()` is teacher-forced: every layer call gets the reference's
hidden state and cache, so an error at (token, layer) is that call's own.
`compare_decode()` is free-running: it reports the tokens, the first
divergence with the reference's top-2 logit gap there, and the chained
per-layer hidden-state error.

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
```

`--model-dir` is read by the worker. `--checkpoint` is read by the script (it
needs the embedding table, and the tokenizer unless `--prompt-ids` is given).

## Results

Offline (`tests/test_axera_llm_layer_loop.py`, a random 2-layer checkpoint):
the loop with `NumpyBackend` gives the reference's greedy tokens; leaving the
mask at zero, or writing the cache rows one slot off, changes them; the same
decode through `RpcBackend` over a socket, against an in-process worker that
answers `io_info` and enforces the dtype labels, gives the same tokens; and
the Python client round-trips every transport dtype through the C++ reference
worker.

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

The rebuilt worker itself has not been compiled against the AXCL SDK yet;
on a host without the SDK it was only syntax-checked against
`tools/onnx-remote/test/axcl_stub/axcl.h`.
