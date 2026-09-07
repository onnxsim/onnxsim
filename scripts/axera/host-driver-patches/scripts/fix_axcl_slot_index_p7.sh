#!/usr/bin/env bash
# Fix 7 -- the device's target id must be able to differ from its PCI bus
# number. ax_get_slot_index() returns pdev->bus->number and that value is
# used everywhere as the device's "target": port_handle[target], the
# heartbeat arrays, and the id the AX650N's own software reports back in its
# heartbeat packet. The card reports a fixed id (3 on this hardware), which
# matches the host only by coincidence of topology -- on this host the card
# enumerates on bus 3. Behind VFIO passthrough into a VM it lands on another
# bus (7), every heartbeat is then rejected by
#
#     if (hbeat->device == target)          /* 3 == 7 -> never */
#
# and the driver declares a perfectly healthy device dead after 50s, even
# though the firmware push, the RC/EP handshake and the card's heartbeats all
# work. Confirmed 2026-09-07 in an LXD VM: "HBDEBUG: target=7 want_count=1
# pkt{dev=3 int=9000 cnt=16}" -- 16 heartbeats received and discarded.
#
# Adds a module parameter so the id can be pinned; the default keeps the
# existing bus-number behaviour, so the host is unaffected.
#
#   sudo bash scripts/fix_axcl_slot_index_p7.sh
#   # host (unchanged behaviour):
#   sudo modprobe ax_pcie_host_dev
#   # guest, where the card enumerates elsewhere but reports id 3:
#   sudo modprobe ax_pcie_host_dev slot_index_force=3
set -euo pipefail
SRCDIR=${SRCDIR:-/usr/src/axcl-2.25.0}
KVER="$(uname -r)"
if [ "$(id -u)" -ne 0 ]; then echo "Run this with sudo: sudo bash $0" >&2; exit 1; fi

if grep -q 'slot_index_force' "$SRCDIR/ax_pcie_dev_host.c"; then
  echo "Patch already applied, skipping edits."
else
  echo "Patching $SRCDIR/ax_pcie_dev_host.c ..."
  python3 - "$SRCDIR" <<'PYEOF'
import sys, os
path = os.path.join(sys.argv[1], "ax_pcie_dev_host.c")
content = open(path).read()

def patch(old, new, count=1):
    global content
    n = content.count(old)
    if n != count:
        print(f"ERROR: expected {count} match(es), found {n} for:\n{old[:110]!r}", file=sys.stderr)
        sys.exit(1)
    content = content.replace(old, new)

# the parameter, next to the existing shm_* ones
patch("module_param(shm_phys_addr, ulong, S_IRUGO);\n",
      "module_param(shm_phys_addr, ulong, S_IRUGO);\n"
      "\n"
      "/*\n"
      " * The device's target id. Defaults to the PCI bus number, which is what\n"
      " * this driver has always used, but that only works while the number the\n"
      " * card's own software reports in its heartbeat packet happens to equal\n"
      " * the bus it enumerated on. Pin it when they differ -- e.g. under VFIO\n"
      " * passthrough, where the card lands on a different bus in the guest but\n"
      " * still reports its own fixed id.\n"
      " */\n"
      "unsigned int slot_index_force = AX_SLOT_INDEX_AUTO;\n"
      "module_param(slot_index_force, uint, S_IRUGO);\n")

# honour it where the id is derived
patch("\tax_dev->slot_index = ax_get_slot_index(pdev);\n",
      "\tax_dev->slot_index = (slot_index_force != AX_SLOT_INDEX_AUTO)\n"
      "\t\t\t     ? slot_index_force : ax_get_slot_index(pdev);\n"
      "\tif (slot_index_force != AX_SLOT_INDEX_AUTO)\n"
      "\t\taxera_trace(AXERA_ERR, \"slot index pinned to %u (bus %u)\",\n"
      "\t\t\t    ax_dev->slot_index, ax_get_slot_index(pdev));\n")
open(path, "w").write(content)
print("  ax_pcie_dev_host.c: patched")

path = os.path.join(sys.argv[1], "ax_pcie_dev.h")
content = open(path).read()
old = "extern struct ax_pcie_operation *g_pcie_opt;\n"
if "AX_SLOT_INDEX_AUTO" not in content:
    assert content.count(old) == 1
    content = content.replace(old, "#define AX_SLOT_INDEX_AUTO\t0xffffffff\nextern unsigned int slot_index_force;\n" + old)
    open(path, "w").write(content)
    print("  ax_pcie_dev.h: patched")
PYEOF
fi

echo "Rebuilding axcl/2.25.0 dkms module for $KVER ..."
dkms build axcl/2.25.0 -k "$KVER" --force
dkms install axcl/2.25.0 -k "$KVER" --force
echo "Done. Reload the whole stack in dependency order; add slot_index_force=<id> where needed:"
echo "  sudo modprobe -r axcl_host ax_pcie_msg ax_pcie_mmb ax_pcie_host_dev"
echo "  sudo modprobe ax_pcie_host_dev [slot_index_force=3] && sudo modprobe ax_pcie_msg && sudo modprobe ax_pcie_mmb && sudo modprobe axcl_host"
