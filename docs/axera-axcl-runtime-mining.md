# axcl-runtime source and device trace logging: two MCode leads checked

Two leads for decoding the AX650 NPU's compiled MCode that this project had not
tried: the public source of `AXERA-TECH/axcl-runtime`, and the AXCL runtime's
trace-level logging (`log.level=0` in `axcl.json`, noted as untested in
[`axera-teng2-toolchain-mining.md`](axera-teng2-toolchain-mining.md)).

## Lead 1: axcl-runtime source -- the host never parses the model

Checked `AXERA-TECH/axcl-runtime` at commit `c0a009b` (shallow clone, outside
this repository). The host-side NPU and engine code is a pure RPC proxy.

- `axclrtEngineLoadFromMem(model, size, id)` (`host/runtime/engine/engine_load_unload.cpp`)
  checks for a null pointer and a zero size, packs `(address, size)` into a
  protobuf request and returns the handle the card sends back. The native path
  (`host/native/npu/npu_handle.cpp`, `AX_ENGINE_CreateHandle[V2]`) does the same.
  No code reads the `.axmodel` protobuf, the `npu_graph_info`/`dotneus` JSON,
  the FlatBuffers tail, the segment table or any MCode byte.
- `protocol/proto/protocol/runtime/engine/type.proto` lists the whole engine
  RPC surface: `InitNpu`, `LoadModel` (address, size -> handle),
  `GetModelVersion`, `GetModelType`, `GetMemoryUsage`, `CreateContext`,
  `GetModelIoInfo`, `RunModel`, affinity calls and a P2P test. Parsing,
  memory allocation and any address relocation all happen on the card, in the
  closed `libax_engine` there.
- `drv/` is PCIe transport only (DMA, message channel, MMB allocator, P2P, a
  virtual NIC). There is no NPU driver source.
- `toolkit/` holds generic helpers (FIFOs, ring buffer, thread pool, DMA
  buffer wrapper), not inspection or dump tools.
- A whole-tree grep (excluding `3rdparty/`) for `mcode`, `neu`, `wbt`, `sdma`,
  `ringbuf`, `flatbuf`, `segment` and `relocat` as whole words finds nothing
  NPU-related. (`teng` only matches inside `Engine`.)

So the source names none of the structures `scripts/axera/mcode.py` decoded
(`tail_vector`, `tail_tables`, `segments`, the per-segment word-count tables),
and it contains no relocation logic that would prove which operand slots are
addresses.

What the public headers do add, as names rather than MCode decode:

| name | where | relevance |
| --- | --- | --- |
| `ModelKind { NPU1, NPU2, NPU3 }` | `type.proto` | the three `npu_mode` values Pulsar2 configs use |
| `NpuKind { Disable, Split, LittleBig, BigLittle }` / `AX_ENGINE_VIRTUAL_NPU_*` | `type.proto`, `ax_engine_type.h` | virtual-NPU partitioning modes |
| `AX_ENGINE_MT_OCM = 2` | `ax_engine_type.h` | "OCM" is the runtime's own name for on-chip memory, as this project already uses it |
| `AX_ENGINE_IO_SETTING_T.nWbtIndex` | `ax_engine_type.h` | a run can select one of several weight tables ("Wbt"), matching the AX620A `pWbtNames`/`nWbtNum` naming in `scripts/axera/README.md` |
| `AX_ENGINE_CMM_INFO.nCMMSize` | `ax_engine_type.h` | model memory is reported as one CMM total, not split by mcode/Wbt/ringbuffer as the AX620A struct was |

Verdict: closed. No MCode or tail field names, no relocation logic.

## Lead 2: trace-level logging -- no effect, but the device kernel log already has NPU fault dumps

**`axcl.json` trace level: no effect.** `axcl.json` lives at `/usr/bin/axcl/axcl.json`
on `axcl-vm`, with separate `log.host.level` and `log.device.level` fields (default
2, info). I left it unmodified. I passed a private copy with both levels at 0
(trace) and a new host log path through `axcl_run_model -c`, and ran the
`relu_1x64x56x56` fixture: control run (default config), trace-config run, then
health run, all under the device lock. All three succeeded (0.48-0.53 ms). The
trace run changed nothing observable. It created no log at the configured path.
The existing host log (`/tmp/axcl/axcl_logs.txt`) gained only `[I]`/`[E]`/`[C]`
lines, and the device's AXCL daemon and worker logs, dumped with
`axcl-smi log`, contain only `[I]` lines. Those are PCIe and RPC plumbing in any
case (handshakes, DMA buffers, packet headers), not NPU execution.

**The device kernel log is where the NPU is visible.** `axcl-smi log -o DIR`
(documented; bitmask `-t` selects daemon, worker, syslog and kernel logs)
returns a tarball whose `AXSyslog/syslog/*.log` holds the card's kernel log.
At default log levels it already contains the NPU driver's fault diagnostics.
Every faulting or hanging model leaves a burst like this, which the host never
reports (`axcl_run_model` prints only `Run model failed{0x8030070C}`):

```
[NPU][Error][npu_check_error_id 821]: EU[6] error
[NPU][Error][npu_irq_err_handle 880]: sync manager INTR_ERROR_ID is 0x40
[NPU][Error][npu_warp_check_error_id 455]: CV EU[6]: AXI0 Read Response Error.
[NPU][Error][npu_warp_clear_interrupt 499]: CV EU[6] CTRL_INT_VEC is 0xf001
[NPU][Error][print_vnpu_ocm_1k_contents 2041]: vnpu 0 1k ocm data offset[0x0000]: ...
[NPU][Error][irq_get_err_code 640]: npu hard error, please reset npu
```

For a hang (`Timeout waiting for NPU to finish running`, `wait time is 10401 ms`),
the dump instead shows every execution unit's queue position and the command it
was executing:

```
[NPU][Error][npu_dev_show_eu_status 1029]: EU[1] q_head=3 q_paddr=0xa493900
[NPU][Error][npu_dev_show_eu_status 1030]: EU[1] Run CMD: cmd_l:0x000000a2, cmd_h:0x00300023
```

What these dumps establish:

- **The hardware command is `mcode.py`'s 8-byte verb record.** `cmd_l`/`cmd_h`
  read as two little-endian words give `a2 00 00 00 23 00 30 00`: verb, field,
  bank, a zero byte, then a 4-byte operand. That is the byte layout `mcode.decode`
  already reports for `V` records (for example `a2 00 00 00 23 00 20 00` at
  offset 305 of the Relu fixture). This is the first confirmation of the
  tokenizer's verb boundary from the hardware side rather than from compiler
  diffs.
- **Engines are named per execution unit.** The driver's per-engine handlers
  name `CONV EU[0]`, `CONV EU[1]` (the "potato" handler) and `CV EU[6]` (the
  "warp" handler), and the sync-manager error ID is a bitmask with bit *n* for
  EU *n* (0x1, 0x2, 0x40 seen). This matches the hardware unit names the
  toolchain mining found in driver strings (TENG, CONV, CV, DMA, MAU, SDMA,
  WARP; [`axera-teng2-toolchain-mining.md`](axera-teng2-toolchain-mining.md)).
- **Decoded fault causes, with raw interrupt vectors:**

| engine | cause | `CTRL_INT_VEC` |
| --- | --- | --- |
| CV | AXI0 Read Response Error | 0xf001 |
| CV | AXI1 Read Response Error | 0xf002 |
| CV | AXI0 Write Response Error | 0xf003 |
| CV | Read0 Trigger Error | 0xf004 |
| CV | Write Trigger Error | 0xf006 |
| CV | Calc Triger Error (sic) | 0xf007 |
| CONV | Iftr Rdma Ch7 Cmd Error | 0xf00e |
| CONV | Undefine Error:ffff | 0xffff |

  An AXI read or write *response* error means the engine issued a bus access to
  a bad address. A patched byte that produces one is therefore very likely part
  of an address operand. That is the evidence `mcode.py`'s docstring says is
  missing: its "operand slots hold allocator output" claim was never proved.

**Segment 3 runs on the `CV` engine (EU6), at least for `Add`.** The device log
held about 120 fault events from 06:38-07:04 device time. A concurrent fork was
fault-injecting a standalone `Add` model then, sweeping its segment 3 (`cv3`)
first, and it held the device lock until my run started. In every one of those
events the same five EUs had advanced queues: 0, 1, 6, 9 and 13. That is one per
segment of the five-segment `Add` program. EU0, EU1, EU9 and EU13 sat at the
same `q_head` (4, 3, 205, 44) and the same command in every hang. Only CV EU6
moved, with `q_head` from 32 to 68 and a different command each time. Its
explicit errors were all `CV EU[6]`. So patches to segment 3 break the CV unit.
This gives hardware backing to the name "cv3" that
[`axera-step-attribution.md`](axera-step-attribution.md) inferred with a modest
margin. EU9, with the longest queue (205 entries), fits `sdma4`, the segment
with the most records. That part is an inference.

**The engine parses a variable-length stream.** Several hung EU6 commands are not
aligned to an 8-byte verb record: `00 a2 00 00 00 12 00 30`, `22 00 30 c4 09 a3 0b 0b`.
Once a patch changes how the stream decodes, the engine's fetch drifts off the
record boundaries. The engine therefore walks a variable-length encoding in
sequence, as `mcode.py`'s mixed record kinds (8-byte `V` plus shorter `S`/`B`
units) imply, rather than a table of fixed-width words.

**A further lever, not tried.** `/proc/ax_proc/logctl` on the device (read with
`axcl-smi sh /bin/cat /proc/ax_proc/logctl`) lists per-module user and kernel log
levels, including `NPU` (id 6) and `ENGINE` (id 29), all at 4. Raising `NPU` or
`ENGINE` may make the driver log non-fault execution too. I did not change it,
because another fork was using the device and it is shared state. Reading
`/proc/ax_proc/npu` through `axcl-smi sh` failed (`0x8030090c`).

## What to do with this

- **Fault-injection sweeps should collect the device log.** After each patched
  model faults or hangs, run `axcl-smi log -t 0x10 -o DIR` (syslog only) and
  parse the new events with `scripts/axera/npu_fault_log.py`. Each patched byte
  then gets an engine, a cause (address fault, trigger fault, command error) and,
  for hangs, the `q_head` and executing command. That turns
  INERT/FAULT/FUNCTIONAL into a map of which bytes are addresses and which are
  command fields, and which engine consumes each segment.
- `scripts/axera/npu_fault_log.py SYSLOG` groups the driver lines into events;
  `tests/test_axera_npu_fault_log.py` covers the formats (copied from a real dump)
  and checks that the command words match the `mcode.py` `V` layout on the
  committed Relu fixture.

Nothing here bypasses a protection. Everything was read through documented
interfaces (`axcl-smi log`, `axcl-smi sh`, `axcl_run_model -c`), and
`/usr/bin/axcl/axcl.json` was never modified.
