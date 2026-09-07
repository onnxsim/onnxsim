# AXCL host driver (v2.25.0) — kernel crash fixes

Local fixes to Axera's out-of-tree AXCL host PCIe driver, made 2026-09-07 while
chasing "kernel crash when running `axcl-smi`" on an **AX650N** attached over
**Thunderbolt/USB4** (ASMedia 246x bridge → PCIe bus 03, device `[1f4b:0650]`).

- Host: GMKtec NucBox EVO-X2, Ubuntu, kernel **7.0.0-31-generic**
- Driver: `axclhost_x86_64_8G_V2.25.0_20250207.deb`, built via **DKMS** from
  `/usr/src/axcl-2.25.0` (`dkms status` → `axcl/2.25.0`)
- Loaded modules: `ax_pcie_host_dev`, `ax_pcie_msg`, `ax_pcie_mmb`, `axcl_host`

Three separate bugs turned out to be hiding behind one symptom, plus a fourth
change layering auto-recovery on top, a fifth that lets the driver run with
the IOMMU on, a sixth that closes the bring-up/removal race, and a
seventh that lets the card work when its reported id differs from its PCI bus
number. All seven are applied and **confirmed on real hardware** (six: by
construction plus a clean host bring-up).

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

## Fix 5 -- `ax_mmb` hands the card raw physical addresses (breaks with the IOMMU on)

**Symptom.** With `amd_iommu=off` removed from the kernel command line, the
host driver loads, pushes firmware and reports `dev 3 back online`, then the
card hits `AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0003 address=0x15cb400000 ...]`
(ten events in the first minutes) and `axcl-smi` hangs. Reproduced 2026-09-07
07:37 on kernel 7.0.0-31 right after the reboot that enabled the IOMMU.

**Cause.** A Thunderbolt-attached device is untrusted, so the kernel keeps it
in a translated `DMA` domain even with `iommu=pt`
(`/sys/bus/pci/devices/0000:03:00.0/iommu_group/type` = `DMA`). Every address
the driver hands the card must therefore be IOMMU-mapped for *that* device.
`ax_mmb.c` was not doing that in either live allocation path:

- `ax_mmb_alloc_mem()` called `dma_alloc_coherent()` on the module's own
  **misc device**, which has no IOMMU domain, so the "DMA address" it returned
  was a raw host physical address.
- `ax_sglist_alloc_memory()` did `kmalloc()` + `virt_to_phys()` -- raw
  physical again.

(`axcl_pcie_host.c`'s firmware-push buffer already used `dma_alloc_coherent()`
on the card's `pci_dev`, which is why the push worked; the per-device pool
under `MEM_LIST_PARTITION` is compiled out.) The card then DMAs to unmapped
addresses, faults, and its runtime never comes up. This is almost certainly why
`amd_iommu=off` had been put on the command line.

**Fix** (`scripts/fix_axcl_iommu_p5.sh`, `patches/ax_mmb.c.iommu.patch`):
allocate and map every card-visible buffer against the card's own `pci_dev`
(`g_axera_dev_map[0]->pdev->dev`, whose probe already sets 64-bit DMA masks):
`dma_alloc_coherent()` on it for the coherent buffers, `dma_map_single()` on
page-aligned `kmalloc` chunks for the scatterlist (page alignment keeps an
untrusted-device mapping from bouncing through swiotlb). `mmap` no longer
treats the card-visible address as a CPU page: scatter chunks map by
`virt_to_phys()` of the kernel buffer, coherent buffers through
`dma_mmap_coherent()` (which also handles IOMMU-backed, physically
non-contiguous coherent memory). With the IOMMU off the DMA API degenerates
to the identity mapping the old code assumed, so behaviour there is
unchanged. Single-card assumption: all buffers map for the first device.
`ax_pcie_mmb` now depends on `ax_pcie_host_dev` (symbol dependency), so
reload the stack in dependency order.

**Verified 2026-09-07 19:24, host, kernel 7.0.0-31, IOMMU on** (`amd_iommu=off`
removed, `iommu=pt`, card in a translated `DMA` domain): after `dkms
build/install` and a reload in dependency order (`lsmod` now shows
`ax_pcie_mmb` among `ax_pcie_host_dev`'s users), firmware push and handshake
completed, `axcl-smi` listed the card, and the kernel log stayed free of
`IO_PAGE_FAULT` -- every fault on record predates the reload (the old module
faulted at fresh 2 MB-aligned `kmalloc` chunks, e.g. `0x1838c00000`, each
time the runtime library allocated a buffer for a request). Also compiles
cleanly under DKMS on 6.8.0-138 in the LXD guest. Full inference through the
mapped buffers was then exercised by the repo's on-device tests (an
`llm_build` layer with real K/V-cache inputs and pulled outputs).

## Fix 6 -- bring-up torn down under itself (the "unplugged again during bring-up" race)

**Symptom.** Whole-host panic, kdump-captured (`/var/crash/202609071939`):
`BUG: unable to handle page fault for address: ffffd4c4f380000c` (supervisor
write, not-present), `RIP: axcl_firmware_load.cold`, called from
`axcl_pcie_device_online_work` on the hotplug workqueue. One millisecond
earlier, on another thread: `[axcl_pcie_device_offline, 1390]: dev 3 offline:
cached handles dropped`. The log shows the bring-up mid-push (UBOOT, DTB, ATF
`SUCCESS`, KERNEL header just printed) when the device was removed.

**How it was triggered.** An LXD VM had the card via VFIO while the host AXCL
modules were still loaded (the blacklist had been undone to test fix 5). The
guest driver's SoC reset made the card re-enumerate; LXD's per-device
`driver_override` went with the old instance; the host's `ax_pcie_dev_host`
probed the new one and fix 4 scheduled a bring-up. A second re-enumeration
during that push ran `ax_pcie_dev_remove()` -> offline -> BAR teardown, and
the push's next write into the ioremap window faulted. This is exactly the
race fix 4's notes called "known remaining"; a Thunderbolt drop during a
normal host bring-up reaches the same window.

**Fix** (`scripts/fix_axcl_online_race_p6.sh`,
`patches/axcl_pcie_host.c.online_race.patch`; needs fixes 3 and 4):
`axcl_pcie_device_online()` marks the target busy for its whole duration and
wakes a waitqueue on every exit; the firmware chunk loop checks
`dev_offline[]` before every chunk it writes (`[STATUS]: ABORTED, device
went offline`) and the completion poll checks it on every iteration, so an
in-flight push exits within one chunk timeout; `axcl_pcie_device_offline()`,
which `ax_pcie_dev_remove()` calls *before* teardown, sets the flag and then
waits (bounded, 20 s) for the busy bring-up to bail. Nothing in the driver
follows a stale BAR mapping after that.

**Verified.** Applies with `patch -p1` on the fix 1-5 tree; compiles cleanly
under DKMS on 6.8.0-138 in the LXD guest and on the host (see below). The
crash itself is not re-provoked on purpose -- doing so needs a card removal
mid-push -- so the guard is verified by construction plus the normal host
bring-up path still completing.

**Operational rule that makes the trigger impossible:** never start the VM
with the host AXCL modules loaded; `scripts/axera/vm/create_vm.sh` now refuses
unless `host_bind_vfio.sh`'s blacklist is in place.

## Fix 7 -- the target id must be allowed to differ from the PCI bus number

**Symptom.** Behind VFIO passthrough into a VM, the AX650N boots perfectly --
firmware push, RC/EP handshake and `axcl-smi`'s port requests all reach the
card -- and is then declared dead 50 s later: `[heartbeat_recv_thread]:
device 7: dead!`, no heartbeat ever accepted, `Recv port ack timeout` for
every request. This was the "device-side handshake timeout" this file listed
as an open, device-side problem. It is neither device-side nor a timeout.

**Evidence.** A debug print of the heartbeat packet at the point of the
timeout, in the guest:

```
HBDEBUG: target=7 want_count=1 pkt{dev=3 int=9000 cnt=16} raw=00000003 00002328 00000010 00000000
```

The card had already sent **16** heartbeats, every one discarded.
`axcl_heartbeat_status()` accepts a packet only `if (hbeat->device ==
target)`, and `target` is `ax_dev->slot_index`, which `ax_get_slot_index()`
derives from `pdev->bus->number`. The card reports its own fixed id, 3. On
this host the card happens to enumerate on **bus 3**, so the two match by
coincidence of topology; in the guest it lands on bus 7 and nothing matches.
Any host that enumerates the card on a different bus hits the same wall.

**Fix** (`scripts/fix_axcl_slot_index_p7.sh`,
`patches/ax_pcie_dev_host.c.slot_index.patch`,
`patches/ax_pcie_dev.h.slot_index.patch`): a `slot_index_force` module
parameter on `ax_pcie_host_dev`. Default `AX_SLOT_INDEX_AUTO` keeps the
existing bus-number behaviour, so the host is unaffected; pass
`slot_index_force=3` where the card's reported id and its bus number differ.
The pinned value is logged once at probe.

**Verified 2026-09-07 in the LXD guest**: with
`modprobe ax_pcie_host_dev slot_index_force=3`, `[ax_pcie_dev_probe]: slot
index pinned to 3 (bus 7)`, zero missed heartbeats, and `axcl-smi` lists the
card with live temperature and utilisation. A real model then compiled on the
host and ran on the card inside the guest through the harness's
`AXCL_LXD_VM` mode (0.18 ms, correct output tensor returned).

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
cd /usr/src/axcl-2.25.0
sudo patch -p0 --dry-run < .../patches/ax_mmb.c.patch    # drop --dry-run to apply
```

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
