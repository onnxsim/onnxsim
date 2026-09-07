#!/usr/bin/env bash
# Fix 6 -- close the "unplugged again during bring-up" race in fix 4. The
# automatic bring-up (axcl_pcie_device_online, on the hotplug workqueue)
# pushes ~150MB of firmware through the card's BAR window and DMA buffer; if
# the device is removed meanwhile, ax_pcie_dev_remove() unmaps the BARs
# under it and the next write faults -- confirmed host panic 2026-09-07
# 19:39 (kdump 202609071939: axcl_firmware_load.cold via
# axcl_pcie_device_online_work, one millisecond after
# axcl_pcie_device_offline ran for the same device). Idempotent; rebuilds
# via DKMS. Requires fixes 3 and 4 (fix_axcl_hotplug_p1.sh / p2.sh).
#
#   sudo bash scripts/fix_axcl_online_race_p6.sh
#   sudo modprobe -r axcl_host && sudo modprobe axcl_host
set -euo pipefail
SRCDIR=${SRCDIR:-/usr/src/axcl-2.25.0}
KVER="$(uname -r)"
if [ "$(id -u)" -ne 0 ]; then echo "Run this with sudo: sudo bash $0" >&2; exit 1; fi
if ! grep -q 'axcl_hotplug_wq' "$SRCDIR/axcl_pcie_host.c"; then echo "ERROR: fix 4 is not applied -- run fix_axcl_hotplug_p2.sh first." >&2; exit 1; fi

if grep -q 'online_busy' "$SRCDIR/axcl_pcie_host.c"; then
  echo "Patch already applied, skipping edits."
else
  echo "Patching $SRCDIR/axcl_pcie_host.c ..."
  python3 - "$SRCDIR" <<'PYEOF'
import sys, os
path = os.path.join(sys.argv[1], "axcl_pcie_host.c")
content = open(path).read()
def patch(old, new, count=1):
    global content
    n = content.count(old)
    if n != count:
        print(f"ERROR: expected {count} match(es), found {n} for:\n{old[:100]!r}", file=sys.stderr); sys.exit(1)
    content = content.replace(old, new)

# 1. state: a bring-up in flight per target, and a waitqueue for offline to wait on
patch("static volatile bool dev_offline[AXERA_MAX_MAP_DEV];\n",
      "static volatile bool dev_offline[AXERA_MAX_MAP_DEV];\n"
      "/* a bring-up (axcl_pcie_device_online) is in flight for this target; the\n"
      " * offline path waits for it to bail before the device is torn down */\n"
      "static volatile bool online_busy[AXERA_MAX_MAP_DEV];\n"
      "static DECLARE_WAIT_QUEUE_HEAD(online_wq);\n")

# 2. the completion poll bails as soon as the device goes offline
patch("""\twhile (1) {
\t\treg = pcie_dev_readl(ax_dev, AX_PCIE_ENDPOINT_STATUS);
\t\tif (reg != 0) {""",
"""\twhile (1) {
\t\tif (ax_dev->slot_index < AXERA_MAX_MAP_DEV &&
\t\t    dev_offline[ax_dev->slot_index])
\t\t\treturn -ENODEV;
\t\treg = pcie_dev_readl(ax_dev, AX_PCIE_ENDPOINT_STATUS);
\t\tif (reg != 0) {""")

# 3. the firmware chunk loop checks before every chunk it writes
patch("""\t\twhile (imgsize) {
\t\t\tsize = min_t(uint, imgsize, MAX_TRANSFER_SIZE);
\t\t\tmemcpy(dma_virt_addr, imgaddr + sent, size);""",
"""\t\twhile (imgsize) {
\t\t\tif (ax_dev->slot_index < AXERA_MAX_MAP_DEV &&
\t\t\t    dev_offline[ax_dev->slot_index]) {
\t\t\t\tprintk("[STATUS]: ABORTED, device %x went offline\\n",
\t\t\t\t       ax_dev->slot_index);
\t\t\t\tret = -ENODEV;
\t\t\t\tgoto out;
\t\t\t}
\t\t\tsize = min_t(uint, imgsize, MAX_TRANSFER_SIZE);
\t\t\tmemcpy(dma_virt_addr, imgaddr + sent, size);""")

# 4. bring-up marks itself busy for its whole duration
patch("""\tdev_offline[target] = false;

\t/* 1. firmware loading */""",
"""\tdev_offline[target] = false;
\tonline_busy[target] = true;

\t/* 1. firmware loading */""")
patch("""\taxcl_trace(AXCL_ERR, "dev %x back online", target);
\treturn 0;

gone:""",
"""\taxcl_trace(AXCL_ERR, "dev %x back online", target);
\tonline_busy[target] = false;
\twake_up(&online_wq);
\treturn 0;

gone:""")
patch("""dead:
\taxcl_devices_heartbeat_status_set(target, AXCL_HEARTBEAT_DEAD);
\treturn ret;
}""",
"""dead:
\taxcl_devices_heartbeat_status_set(target, AXCL_HEARTBEAT_DEAD);
\tonline_busy[target] = false;
\twake_up(&online_wq);
\treturn ret;
}""")

# 5. offline waits (bounded) for an in-flight bring-up to bail before the
#    caller (ax_pcie_dev_remove) unmaps the BARs and frees the device
patch("""\t/* make the heartbeat poll loop bail out of its msleep(1000) cycle */
\tdev_offline[target] = true;
""",
"""\t/* make the heartbeat poll loop bail out of its msleep(1000) cycle */
\tdev_offline[target] = true;

\t/*
\t * A bring-up in flight for this target is still writing through the
\t * BAR window and the DMA buffer the caller is about to tear down. It
\t * checks dev_offline before every firmware chunk and inside every
\t * completion poll, so it exits within one chunk timeout; wait for it.
\t */
\tif (online_busy[target]) {
\t\taxcl_trace(AXCL_ERR, "dev %x offline: waiting for bring-up to bail",
\t\t\t   target);
\t\tif (!wait_event_timeout(online_wq, !online_busy[target],
\t\t\t\t\tmsecs_to_jiffies(20000)))
\t\t\taxcl_trace(AXCL_ERR,
\t\t\t\t   "dev %x offline: bring-up did not bail in 20s",
\t\t\t\t   target);
\t}
""")
open(path, "w").write(content)
print("  axcl_pcie_host.c: patched")
PYEOF
fi
echo "Rebuilding axcl/2.25.0 dkms module for $KVER ..."
dkms build axcl/2.25.0 -k "$KVER" --force
dkms install axcl/2.25.0 -k "$KVER" --force
echo "Done. Only axcl_host changed:  sudo modprobe -r axcl_host && sudo modprobe axcl_host"
