#!/usr/bin/env bash
# Create the LXD virtual machine that owns the AX650N. No sudo needed (the
# user must be in the `lxd` group). Idempotent: re-running reconfigures.
#
#   bash scripts/axera/vm/create_vm.sh [vm-name]
set -euo pipefail
VM=${1:-axcl-vm}
ADDR=${AXCL_PCI_ADDR:-0000:03:00.0}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SHARE=${AXCL_VM_SHARE:-$REPO/scripts/axera/vm/share}
mkdir -p "$SHARE"
# stage what the guest installer needs
cp -n "${AXCL_DEB:-$HOME/dev/archives/axclhost_x86_64_8G_V2.25.0_20250207.deb}" "$SHARE"/ 2>/dev/null || echo "note: put the axclhost .deb into $SHARE (AXCL_DEB=... to point at it)"
rm -rf "$SHARE/host-driver-patches" && cp -r "$REPO/scripts/axera/host-driver-patches" "$SHARE"/
cp "$REPO/scripts/axera/vm/guest_install_axcl.sh" "$SHARE"/

if ! lxc info "$VM" >/dev/null 2>&1; then
  lxc launch ubuntu:24.04 "$VM" --vm -c limits.cpu=8 -c limits.memory=16GiB -c security.secureboot=false
  echo "waiting for the guest agent..."; for i in $(seq 1 60); do lxc exec "$VM" -- true 2>/dev/null && break; sleep 2; done
fi
lxc stop "$VM" --force 2>/dev/null || true
# the card, and a shared folder for .axmodel files / patches / the deb
lxc config device add "$VM" ax650 pci address="$ADDR" 2>/dev/null || lxc config device set "$VM" ax650 address="$ADDR"
lxc config device add "$VM" share disk source="$SHARE" path=/mnt/share 2>/dev/null || true
lxc start "$VM"
for i in $(seq 1 60); do lxc exec "$VM" -- true 2>/dev/null && break; sleep 2; done
echo "guest sees:"; lxc exec "$VM" -- lspci -nn | grep -i 1f4b || echo "  (card not visible in guest -- is it bound to vfio-pci on the host?)"
echo "next: copy the AXCL deb and host-driver-patches into $SHARE, then"
echo "      lxc exec $VM -- bash /mnt/share/guest_install_axcl.sh"
