# Running the AX650N inside an LXD virtual machine

The AXCL host driver is an out-of-tree kernel module, and its faults have taken
this host down (see `../host-driver-patches/NOTES.md`). Putting the card into a
KVM guest via VFIO passthrough bounds the blast radius: a driver fault kills the
guest, and `lxc restart --force <vm>` bus-resets the card without a host reboot.
This must be an LXD **virtual machine** (`--vm`); a container shares the host
kernel and gives no protection.

Everything that only *compiles* (Docker, `pulsar2 build`/`llm_build`) stays on
the host. Only `axcl_run_model` moves into the guest, and the harness routes
it there transparently when `AXCL_LXD_VM=<vm name>` is set:
`pulsar2_docker.axcl_available()`, `run_on_device()` and
`run_on_device_with_inputs()` push the `.axmodel` and input folders with
`lxc file push`, run the tool with `lxc exec`, and pull outputs back. Every
existing device test runs unchanged.

## One-time host prerequisites (sudo)

1. The IOMMU must be on. This host shipped with `amd_iommu=off` on the kernel
   command line; removing it (optionally adding `iommu=pt`) and rebooting gave
   38 IOMMU groups, with the card in a group of its own plus its ASMedia
   downstream bridge -- the friendly case (bridges stay on the host). Check:
   `ls /sys/kernel/iommu_groups/*/devices/ | grep 03:00.0`.
2. Hand the card to `vfio-pci` and keep the host from loading the AXCL stack:
   `sudo bash scripts/axera/vm/host_bind_vfio.sh` (`--undo` reverses it). It
   writes `modprobe.d` overrides, rebuilds the initramfs, and tries to rebind
   immediately; a reboot makes it certain.

**Why the host can no longer run the card once the IOMMU is on** (confirmed
2026-09-07): a Thunderbolt-attached device is treated as untrusted, so the
kernel keeps it in a translated `DMA` domain even with `iommu=pt`
(`/sys/bus/pci/devices/0000:03:00.0/iommu_group/type` reads `DMA`), and the
AXCL driver hands the card raw host physical addresses for its shared memory
-- the card's DMA then faults (`AMD-Vi: Event logged [IO_PAGE_FAULT
domain=0x0003 ...]` from `ax_pcie_dev_host 0000:03:00.0`) and `axcl-smi`
hangs. That is almost certainly why `amd_iommu=off` was on the command line.
Inside a VFIO guest the same driver works, because VFIO installs the guest's
memory map in the IOMMU and the addresses the guest driver hands the card are
exactly the ones mapped. That was fixed the same day by driver fix 5 (`../host-driver-patches/`,
`fix_axcl_iommu_p5.sh`): every card-visible buffer is now allocated and
mapped against the card's own `pci_dev`, and the host runs the card with the
IOMMU on and zero faults. The VM is therefore optional again -- its value is
containment of driver faults, not access to the card.

## Create the guest (no sudo)

```sh
bash scripts/axera/vm/create_vm.sh            # ubuntu:24.04, 8 cores, 16 GiB, the card, /mnt/share
lxc exec axcl-vm -- bash /mnt/share/guest_install_axcl.sh   # deb + DKMS + the four driver fixes
lxc exec axcl-vm -- axcl-smi
```

`create_vm.sh` stages the `axclhost` deb (`AXCL_DEB=` to point at it), the
`host-driver-patches/` tree and the guest installer into
`scripts/axera/vm/share/` (git-ignored), mounted at `/mnt/share` in the guest.
The guest installer registers the driver with DKMS exactly as on the host and
applies the four fixes with `patch -p1`, so a Thunderbolt hiccup inside the
guest is survivable there too.

## Use it

```sh
export AXCL_LXD_VM=axcl-vm
python -m pytest tests/test_axera_mcode_structure.py -k on_device
```

If the guest driver wedges: `lxc restart --force axcl-vm` (the card gets a
secondary bus reset on the way), then re-run. The host is never involved.

## A host crash to never repeat (2026-09-07 19:39, kdump `202609071939`)

Starting the VM while the host AXCL modules were still loaded (the blacklist
had been undone to test driver fix 5) took the host down. LXD steals the card
with a per-device `driver_override`; when the guest driver reset the card's
SoC, the card re-enumerated, the override went with the old device instance,
the host's `ax_pcie_dev_host` probed the new one, and driver fix 4's automatic
bring-up pushed firmware into a card the guest was driving -- panic in
`axcl_firmware_load` (`axcl_pcie_device_online_work`). `create_vm.sh` now
refuses to run unless the stack is unloaded *and* the blacklist file exists.
The guest side has a rule too: a VFIO bus reset does not reset the card's
SoC, only the driver's unload path does, so after `lxc restart --force`
always `modprobe -r axcl_host && modprobe axcl_host` inside the guest before
expecting a firmware push to succeed.

## Where the guest path stands (2026-09-07 evening)

Two guest-side configuration bugs found and fixed, one blocker left.

**Fixed: the guest's virtual IOMMU was remapping the card's DMA.** Passing
`-device intel-iommu,intremap=on,caching-mode=on` with a split irqchip (both
in `raw.qemu`/`raw.qemu.conf`, already set on the VM) brings interrupt
remapping up, but it also turns on DMA remapping, and the card's group in the
guest came up as a translated `DMA` domain: the guest driver handed the card
guest IOVAs that VFIO had never mapped, and the host logged
`vfio-pci ... IO_PAGE_FAULT domain=0x0009 address=0xffc00000`. Adding
`iommu=pt` to the *guest's* kernel command line (`/etc/default/grub.d/
99-axcl-iommu.cfg`) gives the card an `identity` domain, so it gets
guest-physical addresses that VFIO's mapping covers. After that: no host
faults from the guest's DMA, and `DMAR-IR: Enabled IRQ remapping in x2apic
mode` in the guest.

**Fixed: a VFIO bus reset does not reset the card's SoC.** Only the driver's
unload path does (`start reset slave`). After `lxc restart --force`, always
`modprobe -r axcl_host && modprobe axcl_host` inside the guest before
expecting a firmware push to succeed.

**Blocker: the card's onboard software never comes up in the guest.** The
boot ROM works -- all five firmware stages report `[STATUS]: SUCCESS`, which
means the guest's MMIO writes reach the card *and* the card's DMA/status
writes come back. But after `start_devices()` nothing follows: BAR0's shared
window stays all zeros, the card raises no interrupt at all (VFIO's four host
IRQs `vfio-msi[0..3]` stay at count 0 while the guest's MSI capability is
enabled with the vIOMMU's remapped address `0xfee00918`), the RC/EP handshake
times out, and `axcl-smi` reports `Recv port ack timeout`. On the host the
same card raises hundreds of interrupts on its `endpoint-msi-irq` line. So
the card's own Linux either does not finish booting or cannot signal from
inside the guest; that is the next thing to chase, e.g. by capturing the
card's own boot log (`/proc/ax_proc/pcie/sysdump`) after a failed bring-up.

## Known limits

- A Thunderbolt link drop while the card is assigned reaches the *guest*
  driver as a hot-unplug -- exactly the path fix 3 hardens -- and VFIO on the
  host handles the surprise removal; untested until it happens.
- `lxc file push` adds ~0.1 s per run for a small model and scales with the
  weight table (a 12 MB resnet18d `.axmodel` is fine; a 250 MB LLM head is
  noticeable). Keep large models in `/mnt/share` and pass that path when it
  matters.
