#!/usr/bin/env bash
# Hand the AX650N (PCI 1f4b:0650) to vfio-pci on the host so an LXD virtual
# machine can own it. Run with sudo; takes effect at the next boot (initramfs
# is rebuilt) and, if the axcl stack is not currently in use, immediately.
#
#   sudo bash scripts/axera/vm/host_bind_vfio.sh          # bind to vfio-pci
#   sudo bash scripts/axera/vm/host_bind_vfio.sh --undo   # give it back to the host driver
set -euo pipefail
ADDR=${AXCL_PCI_ADDR:-0000:03:00.0}
IDS=1f4b:0650
if [ "$(id -u)" -ne 0 ]; then echo "run with sudo" >&2; exit 1; fi

if [ "${1:-}" = "--undo" ]; then
  rm -f /etc/modprobe.d/vfio-axcl.conf /etc/modprobe.d/blacklist-axcl-host.conf
  update-initramfs -u
  if [ -e /sys/bus/pci/devices/$ADDR/driver ]; then echo "$ADDR" > /sys/bus/pci/devices/$ADDR/driver/unbind || true; fi
  echo > /sys/bus/pci/devices/$ADDR/driver_override
  modprobe ax_pcie_host_dev && modprobe ax_pcie_msg && modprobe ax_pcie_mmb && modprobe axcl_host
  echo "$ADDR returned to the host driver"; exit 0
fi

# 1. vfio-pci claims the card at boot, before the axcl driver can.
cat > /etc/modprobe.d/vfio-axcl.conf <<CONF
options vfio-pci ids=$IDS
softdep ax_pcie_host_dev pre: vfio-pci
CONF
# 2. the host never loads the axcl stack again (the VM runs it).
cat > /etc/modprobe.d/blacklist-axcl-host.conf <<CONF
blacklist axcl_host
blacklist ax_pcie_mmb
blacklist ax_pcie_msg
blacklist ax_pcie_host_dev
CONF
update-initramfs -u

# 3. try to switch right now (harmless if the axcl stack is busy or absent).
modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_host_dev 2>/dev/null || true
modprobe vfio-pci
echo vfio-pci > /sys/bus/pci/devices/$ADDR/driver_override
if [ -e /sys/bus/pci/devices/$ADDR/driver ]; then echo "$ADDR" > /sys/bus/pci/devices/$ADDR/driver/unbind || true; fi
echo "$ADDR" > /sys/bus/pci/drivers_probe
echo "driver now: $(basename "$(readlink /sys/bus/pci/devices/$ADDR/driver)" 2>/dev/null || echo none)"
echo "IOMMU group: $(basename "$(readlink /sys/bus/pci/devices/$ADDR/iommu_group)" 2>/dev/null || echo NONE -- is the IOMMU on?)"
