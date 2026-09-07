# AXCL host driver (v2.25.0) — kernel crash fixes

Local fixes to Axera's out-of-tree AXCL host PCIe driver, made 2026-09-07 while
chasing "kernel crash when running `axcl-smi`" on an **AX650N** attached over
**Thunderbolt/USB4** (ASMedia 246x bridge → PCIe bus 03, device `[1f4b:0650]`).

- Host: GMKtec NucBox EVO-X2, Ubuntu, kernel **7.0.0-31-generic**
- Driver: `axclhost_x86_64_8G_V2.25.0_20250207.deb`, built via **DKMS** from
  `/usr/src/axcl-2.25.0` (`dkms status` → `axcl/2.25.0`)
- Loaded modules: `ax_pcie_host_dev`, `ax_pcie_msg`, `ax_pcie_mmb`, `axcl_host`

Three separate bugs turned out to be hiding behind one symptom, plus a fourth
change layering auto-recovery on top. All four are applied and **confirmed on
real hardware**.

These patches are host-side kernel driver fixes, not onnxsim code. They live
here because keeping this AX650N reachable is a prerequisite for everything
else under `scripts/axera/` that touches the real device (`axcl_run_model`,
`--profile` traces, the on-device mcode probes) -- before fix 3, a Thunderbolt
hiccup mid-experiment reset the whole machine.

## Baseline: pre-existing local edits (not mine)

`/usr/src/axcl-2.25.0` was seeded from `/usr/src/axcl/driver/axcl/`, which
already carried three local edits needed to build/run on kernel 7.0. The patches
in `patches/` are diffed **against that tree**, so they do not include these:

| File | Edit | Why |
| --- | --- | --- |
| `axcl_pcie_host.c` | `+#include <linux/vmalloc.h>` | build fix, newer kernels |
| `ax_pcie_msg_usrdev.c` | `+#include <linux/vmalloc.h>` | build fix, newer kernels |
| `ax_pcie_dev.h` | `SUPPORT_PCI_NET 1` → `0` | PCIe-net feature disabled |

## Fix 1 — `ax_mmb`: 4MB contiguous `kmalloc` WARN storms

**Symptom.** Running `axcl-smi` sprayed `page allocation failure: order:10`
WARNs with full stack traces + meminfo into the log. Alarming, but *not* fatal.

**Evidence.** 9 occurrences on the 2026-09-04→06 boot, e.g. 09-04 22:52:54,
`Comm: axcl-smi`, trace `ax_mmb_ioctl` → `__kmalloc_noprof` → `warn_alloc`,
followed by `[ax_sglist_alloc_memory, 296]: kmalloc 400000 memory failed`.

**Cause.** `ax_scatterlist_alloc()` starts at `MAX_SG_ALLOC_SIZE = SZ_4M`
(order-10) and halves toward 4K on failure. Order-10 is far above
`PAGE_ALLOC_COSTLY_ORDER` (order-3/32K), so plain `GFP_KERNEL` drives direct
reclaim/compaction hunting for 1024 physically-contiguous pages and WARNs when
it fails. The box had ~4GB free — just fragmented after days of uptime.

**Fix.** `patches/ax_mmb.c.patch` — add `__GFP_NORETRY | __GFP_NOWARN`. The
existing halving loop already degrades gracefully; this just makes oversized
attempts fail *fast* (no expensive reclaim stall) and *quietly* (the driver
already logs the failure itself).

**Verified.** WARNs gone; no recurrence.

## Fix 2 — `axcl_pcie_port_manage`: unvalidated user input + NULL deref

**Symptom.** Hard crash (kdump-captured Oops) when running `axcl-smi`.

**Evidence.** `/var/crash/202609070547/` —
`BUG: kernel NULL pointer dereference, address: 0000000000000000`,
`RIP: axcl_pcie_ioctl+0x842/0xef0 [axcl_host]`, `CR2=0`, `RDI=0`,
`Comm: axcl-smi`, faulting instruction `mov (%rdi),%rdi`. The ioctl was
`0xc0284101` = `IOC_AXCL_PORT_MANAGE` (magic `'A'`, nr 1, 40 bytes).

**Cause.** `axcl_pcie_port_manage()` takes `target = devinfo->device`
**straight from the `copy_from_user`'d ioctl argument with zero validation**,
then dereferences `port_handle[target][port]->pci_handle`. Two problems:

1. `target` indexes `port_handle[AXERA_MAX_MAP_DEV][MAX_MSG_PORTS]` — an
   out-of-range value is an out-of-bounds access from an unprivileged ioctl.
2. Even in range, the slot is NULL until a port is opened. This device never
   completed its handshake (`axcl wait dev 3 handshake...`), so it was NULL —
   and the code dereferenced it unconditionally.

**Fix.** In `patches/axcl_pcie_host.c.patch` — bounds-check `target`, NULL-check
`port_handle[target][port]` before use, and guard the identical
`port_handle[target][port]->pci_handle` pattern in `axcl_pcie_release()`.

**Verified.** `axcl-smi` now gets a clean error return instead of crashing:
`[axcl_pcie_port_manage, 719]: Recv port ack timeout.` +
`[axcl_pcie_ioctl, 1111]: axcl pcie req port failed.`

## Fix 3 — heartbeat thread reads unmapped MMIO after hot-unplug ← the real one

**Symptom.** Whole machine resets seconds after the Thunderbolt link drops.
**No `BUG:`, no `Oops`, no kdump vmcore** — the log simply stops mid-line.

**Evidence.** Boot ending 06:09:15 — Thunderbolt disconnect at 06:09:07
(`retimer disconnected`, `Slot(0): Link Down / Card not present`,
`thunderbolt 0-2: device disconnected`), reconnect 06:09:12, then
`[heartbeat_recv_thread, 573]: device 3: dead!` at 06:09:15 and the log ends.

**Cause.** `heartbeat_recv_thread()` resolves the device **once, before its
loop**:

```c
axdev = g_pcie_opt->slot_to_axdev(target);
hbeat = (struct device_heart_packet *)axdev->shm_base_virt;  /* BAR-mapped */
do { ... axcl_heartbeat_recv_timeout(hbeat, ...) ... } while (1);
```

and `axcl_heartbeat_recv_timeout()` polls **through `hbeat`** in a `while(1)` +
`msleep(1000)` loop for up to 50s. On unplug, `ax_pcie_dev_remove()`
`pci_iounmap()`s those BARs and `kfree()`s the `axera_dev` — so the thread keeps
reading a **torn-down ioremap window once per second**. That is why nothing is
ever logged: it is not a heap use-after-free the kernel can trap and report, it
is MMIO access into an unmapped PCIe window of a device that is physically gone.

Nothing told `axcl_host` about the removal at all. Its `port_handle[]` and
per-target state are populated **once**, at `module_init`, and
`ax_pcie_dev_remove()`/`ax_pcie_dev_probe()` only touch their own module's
bookkeeping. So every cached pointer became dangling on unplug.

**Fix** (`scripts/fix_axcl_hotplug_p1.sh`, 13 sites across 3 files):

1. **Hotplug notifier** (`ax_pcie_dev.h`, `ax_pcie_dev_host.c`):
   `ax_pcie_register_hotplug_notify()`, called from `ax_pcie_dev_remove()`
   **before** any teardown (so consumers can drop cached pointers while the
   mappings are still valid) and from `ax_pcie_dev_probe()` on arrival. The
   registration mutex is held across the callback so an unregister cannot
   complete mid-call and leave a pointer into unloaded module text.
2. **Heartbeat thread**: no longer caches `axdev`/`hbeat` across the loop —
   re-resolves each iteration (after the DEAD wait, which can park it
   arbitrarily long) and exits if the device is gone.
3. **`dev_offline[]` flag** checked inside the 50s poll loop, so a thread
   *already inside* it bails within ~1s instead of polling a dead window.
4. **`axcl_pcie_device_offline()`**: clears all `port_handle[target][*]` /
   `port_info[target][*]`, marks the device DEAD, stops that target's heartbeat
   thread.
5. **Indexing bug found on the way**: `heartbeat[]` was indexed by *enumeration
   order* while `heart_waitqueue[]`/`htcondition[]`/`port_handle[]` are indexed
   by *slot index* — and `ax_pcie_dev_remove()` compacts the enumeration array
   on every unplug, scrambling the association. Now all target-indexed, and the
   thread gets its target from stable storage (`heartbeat_target[]`) rather than
   a pointer into the freeable `axera_dev`.

**Verified 2026-09-07 06:35** — first hot-unplug this hardware has survived:

```
06:35:23  thunderbolt 0-0:2.1: retimer disconnected
06:35:23  pciehp: Slot(0): Link Down / Card not present
06:35:24  [heartbeat_recv_thread, 586]: device 3: dead!
06:35:24  [axcl_pcie_device_offline, 1382]: dev 3 offline: cached handles dropped
06:35:28  pciehp: Slot(0): Card present / Link Up
06:35:28  [axcl_pcie_hotplug_notify, 1396]: dev 3 is back; ...
```

Uptime unbroken across the event.

## Fix 4 — automatic bring-up on reconnect

`scripts/fix_axcl_hotplug_p2.sh`. After fix 3 a reconnected device is *safe* but
still unusable until `axcl_host` is reloaded. This adds
`axcl_pcie_device_online(target)` — firmware load → port creation → RC/EP
handshake → timestamp sync → heartbeat thread, for one target — dispatched from
an **ordered workqueue**, because that sequence pushes ~150MB of firmware
(~10s) and `ax_pcie_msg_check_remote()` can block for ~120s, so it must not run
inside the PCI `.probe()` callback.

Deliberately **does not** refactor `axcl_pcie_host_init()`'s phased loops to
share this code: init starts every device's firmware before waiting on any
handshake, which lets multiple cards boot concurrently. Keeping the phases
intact preserves that (and leaves the verified init path untouched) at the cost
of ~30 lines that mirror it.

Incidental correctness note: init's original handshake block does
`if (ret < 0) { set DEAD } ; set ALIVE;` — a failed handshake gets marked ALIVE
anyway. `axcl_pcie_device_online()` uses `goto dead` and does not repeat that.

**Verified 2026-09-07 06:54** — applied, `axcl_host` reloaded on its own
(only that module changed), init reached the handshake and returned without
error, heartbeat thread up, uptime unbroken. On a reconnect the log now reads
`dev N is back, bring-up scheduled` → `dev N back online` instead of asking for
a module reload.

### Known remaining race

If the device is unplugged **again during bring-up**, `ax_pcie_msg_check_remote()`
(in `ax_pcie_msg`, a different module) can still poll shared memory that is
being torn down. `axcl_pcie_device_online()` checks `dev_offline[]` between
phases, which bounds but does not eliminate the window. Closing it properly
needs offline-awareness in the transport layer. Avoid unplugging within
~2 minutes of a reconnect.

Also: `destroy_workqueue()` on module unload waits for an in-flight bring-up, so
`rmmod axcl_host` can block up to ~2 minutes if it is mid-handshake.

## Not fixed: device-side handshake timeout

Separate, **not** a kernel bug and not addressed here. `axcl-smi` still hangs
with **no output at all**: it retries `IOC_AXCL_PORT_MANAGE` forever, and each
attempt times out after 50s (`AXCL_RECV_TIMEOUT`) waiting for an ack from the
AX650N's own onboard software. The low-level RC/EP handshake succeeds and
firmware loads fine (`ATF`/`KERNEL`/`ROOTFS` all `SUCCESS`), so the gap is one
level up — the device's own agent not answering port-open requests. Candidates:
device still booting, its agent crashed, or a firmware/driver version mismatch.

The root *hardware* problem is also still open: **the Thunderbolt link drops on
its own** (`tbtacl` failure + `boltd` probe timeout at the 06:09 disconnect,
which nobody physically triggered). Fix 3/4 make that survivable; they don't
make it stop happening. Worth trying a different cable/port/dock.

## Applying / re-applying

The scripts are idempotent and patch `/usr/src/axcl-2.25.0` in place, then
`dkms build` + `dkms install`:

```sh
sudo bash scripts/fix_axcl_hotplug_p1.sh    # fix 3
sudo bash scripts/fix_axcl_hotplug_p2.sh    # fix 4 (requires p1)
```

Fixes 1 and 2 predate these scripts and their originals were lost to a `/tmp`
wipe on reboot — `patches/*.patch` is the authoritative record for those (and
for everything else). To apply from the patch files instead:

```sh
# headers are in a/ b/ form, so -p1 with -d resolves them regardless of cwd
sudo patch -p1 -d /usr/src/axcl-2.25.0 --dry-run < patches/ax_mmb.c.patch   # drop --dry-run to apply
# all four, then rebuild:
for p in patches/*.patch; do sudo patch -p1 -d /usr/src/axcl-2.25.0 < "$p"; done
sudo dkms build axcl/2.25.0 --force && sudo dkms install axcl/2.25.0 --force
```

Verified 2026-09-07: all four patches reverse-apply cleanly against the
patched `/usr/src/axcl-2.25.0` and forward-apply cleanly against the pristine
vendor tree `/usr/src/axcl/driver/axcl`, i.e. they are exactly the delta.

**After any driver package upgrade or `dkms remove`, all of this is lost** —
`/usr/src/axcl-2.25.0` gets replaced. Re-apply from `patches/`.

Reloading: fixes touching only `axcl_host` need
`modprobe -r axcl_host && modprobe axcl_host`. Anything touching
`ax_pcie_host_dev` needs the whole stack down in dependency order:

```sh
sudo modprobe -r axcl_host ax_pcie_msg ax_pcie_mmb ax_pcie_host_dev
sudo modprobe ax_pcie_host_dev && sudo modprobe ax_pcie_msg && \
  sudo modprobe ax_pcie_mmb && sudo modprobe axcl_host
```

Note that reloading `axcl_host` re-pushes firmware and re-runs the handshake,
i.e. it power-cycles the device's software.

## Debugging notes for next time

- `/var/crash/<ts>/dmesg.<ts>` (root-only) holds the kdump-captured panic log —
  the only place fix 2's trace existed. `journalctl -k -b -1` had nothing,
  because an abrupt crash never flushes to the persistent journal.
- **A crash with no `BUG:`/`Oops` and no vmcore is a signal, not a dead end** —
  it points away from ordinary kernel faults toward unmapped-MMIO access or a
  hardware-level reset, which is exactly what fix 3 turned out to be.
- `journalctl -k -b -N | grep "Comm: axcl-smi"` finds every crash attributable
  to the tool across boots.
- The driver logs success **silently**: `ax_pcie_msg_check_remote()` only prints
  on failure, so "wait dev N handshake..." with nothing after it means it
  *worked*.
