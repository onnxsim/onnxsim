# ResNet-18 DSP: what is measured, and what is left

## Where the phone says the time goes

Xiaomi 12S (SM8475, taro, V69), whole graph as one FastRPC call, 10 iters, `HMX_VTCM_KB=4096`,
0/25088 off ORT. Baseline **22,208 us**.

| shape | us | share | n |
|---|---:|---:|---:|
| `r_16_..._3_3_16` (layer4 3x3 convs) | 7708 | **34.7%** | 3 |
| `r_2_202_..._4_4` (the stem) | 4499 | 20.2% | 1 |
| `E_13276_8_2_2` | 1636 | 7.4% | 1 |
| `r_2_51_..._3_3_2` | 1573 | 7.1% | 4 |
| `r_2_6464_32_3_3` | 1502 | 6.8% | 1 |

The simulator's per-kernel shares are wrong by up to 1.5x and the error points the wrong way:
it put the layer4 convs at 23.4% and the stem at 26.4%. Two earlier attempts targeted the stem,
i.e. the second-biggest item. The phone's breakdown comes from `client <uri> <case> 10 prof`,
which needed a small change to `dsp_graph` (see PHONE_PROFILE.md).

## What each mechanism is worth

Full rebuild + phone run each:

| build | us | vs baseline |
|---|---:|---:|
| baseline | 22,208 | - |
| `HMX_I8_CACHE=0` | 29,878 | **+34.5%** (the VTCM tile cache is worth 6.9 ms) |
| `HMX_I8_QUAD=0` | 23,739 | +6.9% |

Both on by default, so neither is available as a win. They size what a new mechanism has to beat.

`HMX_RQ=0` is **not** a usable measurement: without the requant the kernel falls back to a scalar
path and measures 526,475 us (24x slower), which says nothing about the requant's cost.

## Measurement traps, both mine

- `HMX_VTCM_KB` defaults to 256 KB; the phone grants **4 MB**. A build with the default measures
  25,558 us - 14.6% slower - and the client says so in one field (`vtcm 262144`). My first
  rebuild looked like a regression for this reason.
- The `dsp_graph` per-call timing initially reported 0.0 us because the total's start timestamp
  and `G_PROF`'s per-call cursor were the same variable. Fixed; verified against the known
  22,238 us.

## The requant trade, undecided

`__hmx_rqwf` is ~31 vector ops per 32-lane group, 1.16 cycles/output against the hand kernel's
0.84, and most of that is per-lane normalization (a shift and exponent from each lane's own
`|acc|` via `vcl0w`) that cannot be hoisted. The hand kernel's `QC_FAST` removes all of it with a
host-built per-column table and one `:after:cm:sat.ub` store, at the cost of ~5-8% of outputs
landing 1 off - the same error class as QNN's own HTP (5.6-7.7%).

**Not decided, and not mine to decide**: it trades exactness against ORT for speed, and the
0.045-of-float64 contract is the family's. The 5-8% figure is the hand kernel's documentation,
not something measured here - emulating the `:cm.ub` convert in numpy gave a max error of 186,
far outside the expected class, so the emulation is wrong. It needs a real measurement on the
25088-output comparison, which needs a skel build.

## The stem: two attempts, no speedup, and a correction

- The stem does 3.59x the MACs it needs (3 channels padded to 8, so 12 of 32 K lanes carry data).
  Re-lane-packing to use all 32 was **bit-exact and 2.9x slower** (68.9M pcycles): the wasted
  lanes could only be filled from other grid rows, which cost 46.4M pcycles in a separate gather
  kernel. The invariant is that the A pack must stay a flat contiguous 2 KB.
- A third attempt (crouton) hit its agent turn limit with no result. The hand kernel's crouton
  is already the A tile shape, so the real fix is a graph-level layout change, not a re-lane-packing.
- **The correction**: HMX MACs are ~20k of 682k pcycles in the 3x3 family - about 3%. Requant is
  ~330k, activation packing ~150k. Two attempts went after the smallest term.

## Open, in order of value

1. The layer4 3x3 convs, 34.7% of phone time, not yet touched.
2. Decide the `QC_FAST` accuracy trade, then measure it properly.
3. The five quarantined HMX oracle files. Root cause is **not** a kernel bug: the captured 1x1
   and 3x3 kernels both complete when run directly (284k pcycles in 12.7 s; 254M insns, 97 s), and
   the whole test body completes outside pytest. Only under pytest does the simulator child die,
   in Python's `subprocess` frame inside `hexsim._sim`. Needs a core dump of the parent;
   `kernel.core_pattern` pipes to apport and leaves no file. See ORACLE_QUARANTINE.md.
4. MCC: goldens are real device captures and committed; the test is still the original xfail
   because three agent runs hit their turn limit with the skel bugs unfixed (in-place restore
   wiping the result before copy-out; GELU's output never returned).
