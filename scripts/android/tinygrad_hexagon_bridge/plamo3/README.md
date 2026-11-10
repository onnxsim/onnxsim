# PLaMo 3 Translate 2B on the Hexagon cDSP (onnx-remote, tinygrad-generated HMX programs)

`pfnet/plamo-3-translate-2b` decodes on a Xiaomi 12S (Snapdragon 8+ Gen 1, V69 cDSP). The decoder layers run on the DSP as
six 4-layer programs, each with its KV caches resident on the phone. The host keeps the embedding table, the final norm and
the LM head. Programs are built by the tinygrad fork (`onnxsim/tinygrad`, `examples/dsp_graph_onnx.py`) through
`../onnx_remote/compile_v69.sh`, and run by `onnx-remote-hexagon-worker` (`tools/onnx-remote`).

```sh
# int4 (group 128) weights, 32-row blocks, 256 cache slots, device-resident caches
export_chunks.py CHECKPOINT blk4st256 --chunk 4 --cache-len 256 --rows 32 --int4 --split qkv:2,gu:8,d:4 --state
compile_blocks.sh blk4st256 tghb4st256           # HMX fp16 GEMMs; needs TINYGRAD_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN
../openpilot_v65/build_worker.sh worker/          # Android aarch64 worker
run_phone_spec.sh worker/onnx-remote-hexagon-worker CHECKPOINT x tghb4st256 --ids 1,16,... --cache-len 256 --max-new 40 --profile
```

`run_phone_block.py` is the plain one-token-per-pass driver. `run_phone_spec.py` is the faster one:

- **Prompt-lookup speculative decoding.** A pass costs the same whatever the number of real rows, since the device always
  computes all 32. So each pass feeds the next token plus up to 31 drafts: the tokens that followed the latest earlier
  occurrence of the longest matching suffix (4 down to 1 tokens). The longest prefix of drafts that matches the per-row
  greedy choices is accepted, plus a bonus token. The output is exactly plain greedy decoding
  (`tests/test_axera_plamo3.py::test_speculative_decode_matches_greedy` checks this on a random model). KV rows of rejected
  drafts are masked and later overwritten, so nothing is rolled back. It pays off only when the output repeats earlier text.
  `--no-spec` turns it off.
- **One chained request per pass.** The worker's `run_compiled` takes `a+b+...` and feeds each program's hidden output to
  the next on the phone (`tools/onnx-remote/README.md`). Six requests per pass become one. `--no-chain` turns it off.

## Measured (Xiaomi 12S, 2026-11-10)

| run | s/token | passes | host<->phone transfer per pass |
|---|---|---|---|
| 91-token prompt, greedy (`--no-spec`) | 1.13 | 39 | 0.030 s |
| 91-token prompt, speculative, `--no-chain` | 0.88 | 28 | 0.140 s |
| 91-token prompt, speculative (default) | **0.82** | 28 | 0.031 s |
| 41-token prompt, greedy | 1.10 | 27 | 0.029 s |

All runs emitted identical tokens. Speculative decoding accepted 27 of 35 drafts on the 91-token prompt, which repeats its
own text, and 0 of 8 on the 41-token prompt. Per pass, device time is ~1.10 s (six programs at ~184 ms); the LM head is 0.02-0.03 s.

**Run-to-run drift.** In some runs, one or more programs run at ~240 ms instead of ~184 ms for the whole run. Which programs
and how often changes with the phone's state over time, not with the binaries. Ruled out on the device:
- VTCM refusal: always granted
- physically fragmented or different memory: every page contiguous, the same physical addresses in fast and slow runs
- core clock: 1401 MHz in slow runs too
- the bus-corner vote: it coincided with a fast evening, and the same binaries were slow the next morning

The slow programs' extra time is in the latency-bound kernels. Small HVX loops take 10-100x longer and HMX GEMMs ~4x, while
the int4 weight dequant is unchanged. Compare changes with interleaved A/B runs in one session, not against an earlier day.
