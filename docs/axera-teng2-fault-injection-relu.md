# Single-byte fault injection on `teng2`: a first causal-intervention pass on a real elementwise compute segment

Every prior `teng2` decode attempt in this project (`docs/axera-teng2-sqrt-blocks.md`,
`docs/axera-teng2-tiled-repeat.md`, `docs/axera-teng2-add-two-input.md`,
`docs/axera-teng2-calibration-isolation.md`, `docs/axera-teng2-cluster-patch-refine.md`)
compared *different compiler builds* against each other -- across shapes, across
calibration ranges, across clusters -- and looked for a byte-level rule that
predicts one build's mcode from another's. None found a global rule; the
compute segment (`teng2`, `mcode.segments()[2]` for a standalone elementwise op)
consistently shows real local structure and no shape-to-bytes formula.

This is a different kind of attempt: no rebuild at all. `README.md`'s own
"bit-flip probe" sections used single-byte fault injection directly on real,
already-compiled hardware to classify Conv-heavy mcode bytes as INERT, FAULT
(`0x8030070C`), or FUNCTIONAL (a genuine, observable effect on the actual
computation). This does that, on the already device-verified
`Relu(x[1,64,56,56])` reference from
`scripts/axera/fixtures/dma_tiles/relu_1x64x56x56.axmodel.gz`
(`docs/axera-dma-queue.md`'s own reference build), enriched two ways over the
README's original technique using two concurrent forks' findings:

- **Register-level targeting.** `docs/axera-pulsar-v1-mining.md` (PR #1818)
  decoded `mcode.py`'s `V` records as 8-byte register writes -- `[verb][unit]
  [reg_lo][reg_hi][value32]`, with `field`+`bank` forming a 16-bit register
  address. Every byte of every `V` record is tested individually here, so a
  result reports *which register, which byte of its value* rather than a bare
  offset.
- **Fault-cause attribution.** `docs/axera-axcl-runtime-mining.md` (PR #1821)
  found the AX650's device kernel log (`axcl-smi log -t 16 -o DIR`, a
  `dev*_log_*.tar.gz` whose `AXSyslog/syslog/*.log` holds it) carries the NPU
  driver's own fault diagnostics: which execution unit faulted, a decoded
  cause (an AXI *response* error means a bad bus address; a trigger/command
  error means a structural problem), and on a hang, every EU's queue position
  and executing command. Every `FAULT` result here collects and diffs this log
  (inside the same device-lock window as the faulting run, so no other
  process's fault gets misattributed) and records the engine, EU, and cause.

## Method

`scripts/axera/teng2_fault_injection.py`:

1. Load the reference, locate segment 2: absolute byte range `[380, 1788)`.
   The project's known 301-325 compiler-noise window sits entirely inside
   segment 0, not segment 2, so no exclusion is needed. `mcode.decode()`
   parses the segment's real (untrimmed) span -- trimming trailing zero
   bytes before decoding, the way `mcode.segment_coverage()` does for
   coverage reporting, would clip a genuine record whose own trailing bytes
   are legitimately zero (a live register write with value 0 is not
   padding); one record at the segment's very end is excluded instead, since
   it is actually the *next* segment's opening `a7` marker straddling
   backward (per `mcode.py`'s own documented boundary rule).
2. `candidates()` enumerates one entry per record: every byte of a `V`
   record (verb, unit, reg_lo, reg_hi, up to 4 value bytes -- 613 candidates
   total, 236 of them across 30 `V` records / 24 distinct registers, 377 the
   first byte of every other record kind).
3. Each candidate is patched with `original ^ 0xFF` and run on the AX650N
   against a fixed, real, in-range input (`x` uniform in ±0.8, seed 0),
   classified against the unpatched control's own output as INERT, FAULT,
   FUNCTIONAL, or ERROR (a failure that isn't the known fault code).
4. On FAULT, the device kernel log is collected and diffed against a
   baseline snapshot taken once at sweep start (not against an empty
   string -- the log is the device's own persistent, cross-process kernel
   log, so diffing from `""` on the very first fault would attribute
   whatever any other fork last faulted, not this sweep's own run).
5. A retried health-check control run follows every patch (a single mismatch
   retries once before the sweep stops -- `README.md` documents a rare,
   non-reproducible `0x8030070C` on a genuinely valid model; a transient
   recovers on retry, a real problem reproduces).

Device calls are serialized per-call (not around the whole sweep -- an
earlier version of this script held the lock for the entire run, which
another concurrent fork correctly flagged as starving it) via
`flock /tmp/axcl-device.lock`, routed through the `axcl-vm` LXD guest.

## Results: 27 of 613 candidates, stopped by a genuinely wedged device

The sweep ran cleanly for its first 27 candidates (all real device calls,
persisted incrementally, resumable), then a health check failed twice in a
row and stopped itself -- exactly the designed safety behaviour. Investigating
directly (not assumed): the device was not merely contended, it was wedged --
`axcl-smi` itself, queried directly with no model involved, timed out after
40 seconds. A stuck `axcl_run_model` process was found inside the guest in
uninterruptible I/O wait. This matches `pulsar2_docker.py`'s own documented
AXCL host-driver-crash failure mode, whose established recovery is
`lxc restart --force` on the guest -- but that would kill every other fork's
in-progress device work sharing this VM, not just this sweep's, so it was
**not** attempted unilaterally here. Per this task's own instruction ("if the
device stops responding, STOP immediately and report"), device work stopped
at this point. The 27 collected results and the raw sweep state are committed
at `scripts/axera/fixtures/teng2_fault_injection/relu_sweep_partial.json`;
`scripts/axera/teng2_fault_injection.py sweep OUT.json --resume` continues
from there once the device recovers.

**27 of 613 candidates is 4.4% coverage** -- far short of a full map. What
follows is real, but drawn from a small, non-random-looking prefix (the
sweep walks candidates in stream order, so this is segment 2's first ~400
bytes: the opening raw header and the first four `V` records), not a
representative sample of the whole segment.

| classification | count | share |
| --- | --- | --- |
| FAULT | 14 | 51.9% |
| INERT | 12 | 44.4% |
| FUNCTIONAL | 1 | 3.7% |

**Segment 2 (`teng2`) runs on the engine named `TENG`, execution unit 9 --
confirmed directly from the device's own fault log, not inferred.** Every
fault whose cause was captured (7 of 14 -- the log collection did not
reliably capture all of them; see below) names `TENG EU9`. Prior work in this
project (`docs/axera-step-attribution.md`) inferred segment identities from
node counts and queue lengths (a Relu-only guess never made); this is the
first time an mcode segment's owning engine has been read directly off the
hardware for any standalone elementwise op.

**Fault causes split into the two classes `docs/axera-axcl-runtime-mining.md`
predicts, both observed on real segment-2 bytes:**

| cause | class | count |
| --- | --- | --- |
| AXI0 Read Response Error | address | 4 |
| AXI Write Response Error | address | 1 |
| Retrigger error | trigger/cmd | 2 |
| (log collection didn't capture a cause) | -- | 7 |

An AXI response error means the engine issued a bus access to a bad address
-- direct, causal evidence that byte is part of an address operand, the exact
claim `mcode.py`'s own docstring says was "never proved." This is now proved,
for at least five bytes.

**Register-level sensitivity is uneven within a single 8-byte `V` record**,
not uniform across it:

| register | classifications (of the bytes tested so far) |
| --- | --- |
| `0x0280` | 7 FAULT, 1 INERT |
| `0x03b0` | 1 FAULT, 7 INERT |
| `0x03c0` | 2 FAULT, 4 INERT |

`0x0280` (the first `V` record, at absolute offset 385) is almost entirely
load-bearing; `0x03b0` (the second, at offset 393) is almost entirely
tolerant of a full byte flip. Both are real, not an artifact of one
unrepresentative record -- they are two different registers with two
different apparent roles.

**The one FUNCTIONAL byte**: offset 380, the segment's very first byte (a
`raw`-kind byte before the first decoded `V` record), original value `0x0A`.
Flipping it to `0xF5` changed the output with `max_abs_diff = 0.798` against
the unpatched control -- a genuinely live, non-opcode, non-address byte with
an observable numeric effect on the actual Relu computation. Per `README.md`'s
own resnet18d probe, this raw delta is shaped by the model's own int8
dequantization step and is not the byte's encoding directly; only one value
was tried, so nothing beyond "this byte matters and changes the result" is
established -- the finer, multi-value bisection `README.md`'s Conv probes used
was not reached in the time available.

## Two known gaps in the fault-log attribution, stated precisely

- **Only 7 of 14 faults got a cause.** The other 7 show no captured fault
  event, most likely a timing race (the kernel log write may lag slightly
  behind `axcl_run_model` returning its own error) rather than a systematic
  gap -- not confirmed either way in the time available.
- **The `-t` flag needs a decimal mask, not the hex string `axcl-smi`'s own
  help text style would suggest** (`-t 16`, not `-t 0x10`) -- found by direct
  trial after the first several attempts silently returned nothing. Anyone
  reusing this pattern should confirm the flag value against their own
  `axcl-smi` build rather than assume.

## What this does and does not establish

This classifies *whether* a byte matters and, where the device log worked,
*which engine and which class of fault it causes* -- not the byte's exact
encoding. 96% of segment 2 (586 of 613 candidates) was not reached at all.
The method itself -- direct hardware fault injection with device-log
attribution, at register granularity -- is validated and ready to continue
once the device recovers; what it has not yet done is decode enough of
segment 2 to reconstruct or predict its bytes.

## Reproduction

```
scripts/axera/teng2_fault_injection.py sweep OUT.json [--limit N] [--resume]
scripts/axera/teng2_fault_injection.py finesweep OUT.json OFFSET [--values 0x01,0x02,...]
```

`tests/test_axera_teng2_fault_injection.py` covers the patching, candidate
enumeration, classification, and fault-log-diffing logic without needing
Docker or a device (18 tests). The fault-log integration additionally needs
`scripts/axera/npu_fault_log.py` from PR #1821 (`codex/axera-axcl-runtime-mining`,
not yet merged when this was written); this module imports it as optional
(`try/except ImportError`) so the classification sweep itself still runs
without it, only skipping the engine/cause enrichment.
