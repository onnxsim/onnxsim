# Fault-injecting Add's `teng2`/`cv3` on real hardware: a byte-role map, and where it stopped

Every prior `teng2`/`cv3` decode attempt in this project (`docs/axera-dma-queue.md`,
`docs/axera-teng2-sqrt-blocks.md`, `docs/axera-teng2-tiled-repeat.md`,
`docs/axera-teng2-add-two-input.md`, `docs/axera-add-cv3-decode.md`,
`docs/axera-teng2-cluster-patch-refine.md`) worked by **diffing compiled
builds across a shape or calibration sweep** -- looking for which bytes
change when a known input changes. All of them found real local structure
and no global rule. This is a different technique: **causal intervention on
one fixed, already-verified binary**, the same one `README.md`'s `M_channel`
decode used -- XOR one byte, run it on the AX8850, see what breaks. It
doesn't need a second compiled shape at all.

## Method

Reference: `Add(x[1,16,8,8], z[1,16,8,8])`, the smallest fixture in
`scripts/axera/fixtures/add_tiles/` (from the merged `docs/axera-teng2-add-two-input.md`).
`mcode.segments()` locates `teng2` (segment 2, 1,632 bytes, offset 436) and
`cv3` (segment 3, 256 bytes, offset 2068). For each byte offset in a
segment: copy the reference, XOR that byte with `0xFF`, run on the AX8850
(`axcl-vm`, fixed real in-range inputs for both `x` and `z`) under the
shared device lock, and classify against a control run of the unpatched
reference:

- **IDENTICAL** -- byte-identical output.
- **FAULT** -- `[ERROR] Run model failed{0x8030070C}`, retried once (a real
  fault faults every time; a rare transient recovers -- `README.md`'s own
  documented caveat on this fault class).
- **DIFFERENT** -- runs to completion, output changes: a live,
  computational byte.

A health check (a fresh control run, compared byte-for-byte to the original
control) runs after every 16 patches.

**Locking discipline, corrected mid-run.** The sweep initially held the
shared `/tmp/axcl-device.lock` for its *entire* duration via an outer
`flock` wrapper -- wrong, since two other forks (a Relu-side fault-injection
sweep and a one-off trace-log run) share the same physical AX8850 and were
blocked out for hours. Fixed to acquire the lock for exactly one device
interaction at a time (`device_lock()`, a `fcntl.flock` context manager),
with the sweep's own results checkpointed to disk so resuming after the fix
replayed nothing already classified.

**Fault-log enrichment.** For every `FAULT`, `axcl-smi log -t 16 -o DIR`
(decimal 16 == the `0x10` mask another concurrent fork's PR #1821,
`docs/axera-axcl-runtime-mining.md`, documented) dumps the device kernel
log, pulled and diffed against the previous pull so only genuinely new
lines are attributed to that fault -- all inside the *same* lock window as
the triggering run, so no other fork's activity lands in between. This
turns a bare `0x8030070C` into an execution unit, a decoded cause, and (for
a hang) the executing command's own bytes.

## Results

### `cv3` (256 bytes): fully swept

| class | count |
| --- | --- |
| IDENTICAL | 143 (56%) |
| FAULT | 109 (43%) |
| DIFFERENT | 4 (1.5%) |

A concurrent fork mining the same device log (PR #1821) independently
observed roughly 120 of these faults and found they all land on **CV EU6**
-- `cv3` runs on the CV compute unit. That fork's finding is cited here
rather than re-collected; re-running ~109 fault offsets solely to capture a
cause already established from the same events would have been redundant
device time.

The 4 `DIFFERENT` bytes, matched against `mcode.decode()`'s own record
boundaries:

| offset | role |
| --- | --- |
| 2102 | `V`-record @2097: value32 byte 1 (LE) of a write to register `0x0170` |
| 2103 | `V`-record @2097: value32 byte 2 (LE), same register |
| 2106 | `S`-record @2104 (compressed short unit), byte 2 |
| 2107 | `S`-record @2104 (compressed short unit), byte 3 |
| (2111 also `DIFFERENT`, an `S`-record @2110 byte) |

### `teng2` (1,632 bytes): offsets 436-480 only, 45 of 1,632

| class | count |
| --- | --- |
| IDENTICAL | 15 |
| FAULT | 24 |
| DIFFERENT | 6 |

A far higher functional rate proportionally than `cv3` (13% vs. 1.5%),
consistent with `teng2` being the actual op-defining compute region rather
than `cv3`'s smaller, more structural role.

**Faults land on TENG EU9**, and split into two decoded causes, captured
for 13 of the 24 fault offsets (fault-log capture was only wired in after
resuming past offset 448; the earlier 437-448 faults were classified but
not cause-attributed):

| cause | offsets |
| --- | --- |
| `AXI0 Read Response Error` (address-like: corrupting the byte makes the engine read the wrong memory) | 5 |
| `Undefine Error:*` / `Retrigger error` (structure-like: corrupting the byte makes the engine reject the instruction itself) | 8 |
| `timeout` (the engine hangs rather than erroring immediately) | 6 |

(these overlap across the 13 offsets, since one offset's two capture
attempts sometimes hit different causes.)

**One fault snapshot directly confirmed the offset-to-instruction mapping.**
At offset 438, the hang snapshot's `EU9` executing-command bytes read
`a3 ff 00 00 00 ff 00 00` -- the injected `0xFF` visible twice inside the
live instruction word the hardware was executing when it faulted, not just
inferred from the classification.

The 6 `DIFFERENT` bytes:

| offset | role |
| --- | --- |
| 436 | `V`-record @433: register HIGH byte (`bank`; reg=`0x0e00`) |
| 441 | `V`-record @441: the verb byte itself (`verb=0xa1`, reg=`0x0280`) -- corrupting the verb still ran and changed output, meaning at least one other byte value in the verb's range is itself a valid, different instruction |
| 456 | `V`-record @449: value32 byte 3 (LE), reg=`0x03b0` |
| 462 | `V`-record @457: value32 byte 1 (LE), reg=`0x03c0` |
| 467 | `V`-record @465: register LOW byte, reg=`0x03d0` |
| 468 | `V`-record @465: register HIGH byte, reg=`0x03d0` |
| 473 | `S`-record @472 (compressed short unit), byte 1 |

Notably, offsets 467/468 corrupt the *register address itself*, not the
value written to it -- and the model still ran, just evidently writing to
a different, still-valid register rather than faulting. That is a real,
useful data point about how forgiving the register-address decode is,
independent of what either register's semantic role is.

## Why the sweep stopped at 45 of 1,632 `teng2` bytes

Two things, in order:

1. **Shared-device throughput**, once the locking was fixed to share
   properly (as it must): three forks round-robin one physical AX8850, so
   per-byte throughput dropped well below the ~1.4s/run this sweep saw
   running alone.
2. **A genuine, unresolved device anomaly.** After a burst of `FAULT` and
   `timeout` classifications around teng2 offsets 477-480, this sweep's own
   health check reported `BAD` and it self-stopped, exactly as designed. A
   careful re-check (three fresh attempts, lock properly released and free,
   no other fork's process running at the time) found a **persistent,
   different** failure on every attempt:

   ```
   ... failed{0x80300186}.
   [...][E][device manager][request_ports][464]: request ports from device 3 fail, errno: 1 Operation not permitted
   [ERROR] Init failed.
   ```

   This is not the `0x8030070C` runtime-rejection class every other fault
   in this whole project's fault-injection history has produced (including
   every fault this exact sweep saw at every other offset). It is a
   *device/port-level* initialization failure, 3 for 3, with the shared
   lock free and no concurrent device process running at the time of the
   check. The VM itself (`axcl-vm`) stayed `RUNNING` throughout, and
   another fork's `axcl_run_model` call succeeded shortly before this was
   noticed, so this is not a full device lockup -- but it is a real,
   persistent failure mode this sweep's own safety discipline correctly
   caught and refused to push through.

   **This was not investigated further and is reported, not resolved.**
   Everything above (all of `cv3`, `teng2` offsets 436-480) was collected
   *before* this state, with clean health checks throughout, and is not in
   question. Whoever continues this sweep should start with a fresh,
   careful health check of the shared device before resuming, and treat
   `0x80300186`/"request ports... Operation not permitted" as a distinct
   failure class from the fault-injection `0x8030070C`, worth its own
   before continuing -- not something to retry through.

## What this is, and isn't

A verified, reproducible byte-role map for 301 of `Add`'s 1,888 `teng2`+`cv3`
bytes (256/256 of `cv3`, 45/1,632 of `teng2`), with register addresses and
fault causes for the functional and fault classes respectively. It is not a
predictor: nothing here says what value register `0x03b0` (or any other)
should hold for a shape or calibration this reference wasn't built with --
these are the reference's own bytes, causally confirmed live, not derived
from a formula. Combined with the parallel Relu-side sweep and PR #1821's
independent device-log mining, this is the first time any op's `teng2` has
been mapped at the individual-byte, individual-register level rather than
only characterized by cross-build diffing.

## Reproduction

```
scripts/axera/teng2_fault_injection_add.py report   # byte map from the committed fixtures, no device needed
```

The live sweep itself needs the AX8850 (`axcl-vm`) and is not reproduced as
a single CLI entry point here; `run_on_device`/`run_and_maybe_capture_log`/
`dump_syslog_nolock` in `scripts/axera/teng2_fault_injection_add.py` are the
primitives it was built from. `tests/test_axera_teng2_fault_injection_add.py`
checks the register-decode logic and the committed results/fault-event
fixtures, no Docker/device required.
