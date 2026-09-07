#!/usr/bin/env bash
# Inside the guest: install the AXCL host package, register its driver with
# DKMS, apply the host-driver-patches, load the stack, and check the card.
# Expects /mnt/share/axclhost_*.deb and /mnt/share/host-driver-patches/.
set -euo pipefail
DEB=$(ls /mnt/share/axclhost_*.deb | head -1)
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dkms build-essential "linux-headers-$(uname -r)" python3 patch >/dev/null
dpkg -i "$DEB" || apt-get install -f -y -qq
SRC=/usr/src/axcl-2.25.0
if [ ! -d "$SRC" ]; then
  mkdir -p "$SRC" && cp -r /usr/src/axcl/driver/axcl/. "$SRC"/
  cat > "$SRC/dkms.conf" <<CONF
PACKAGE_NAME="axcl"
PACKAGE_VERSION="2.25.0"
AUTOINSTALL="yes"
CLEAN="make -C /lib/modules/\${kernelver}/build M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build clean"
MAKE[0]="make -C /lib/modules/\${kernelver}/build M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build -j\$(nproc) modules"
BUILT_MODULE_NAME[0]="ax_pcie_host_dev"
DEST_MODULE_LOCATION[0]="/updates/dkms"
BUILT_MODULE_NAME[1]="ax_pcie_mmb"
DEST_MODULE_LOCATION[1]="/updates/dkms"
BUILT_MODULE_NAME[2]="ax_pcie_msg"
DEST_MODULE_LOCATION[2]="/updates/dkms"
BUILT_MODULE_NAME[3]="axcl_host"
DEST_MODULE_LOCATION[3]="/updates/dkms"
CONF
fi
# the three pre-existing kernel-7 build edits (see host-driver-patches/NOTES.md)
grep -q 'linux/vmalloc.h' "$SRC/axcl_pcie_host.c" || sed -i '0,/#include/s//#include <linux\/vmalloc.h>\n#include/' "$SRC/axcl_pcie_host.c"
grep -q 'linux/vmalloc.h' "$SRC/ax_pcie_msg_usrdev.c" || sed -i '0,/#include/s//#include <linux\/vmalloc.h>\n#include/' "$SRC/ax_pcie_msg_usrdev.c"
sed -i 's/SUPPORT_PCI_NET 1/SUPPORT_PCI_NET 0/' "$SRC/ax_pcie_dev.h"
# the four fixes
for p in /mnt/share/host-driver-patches/patches/*.patch; do
  if patch -p1 -d "$SRC" -R --dry-run -f < "$p" >/dev/null 2>&1; then echo "already applied: $(basename "$p")"; else patch -p1 -d "$SRC" < "$p"; fi
done
dkms add axcl/2.25.0 2>/dev/null || true
dkms build axcl/2.25.0 --force && dkms install axcl/2.25.0 --force
# The card reports a fixed device id (3); the driver otherwise derives the id
# from the PCI bus number, which differs in the guest -- see driver fix 7.
SLOT=${AXCL_SLOT_INDEX:-3}
modprobe ax_pcie_host_dev slot_index_force=$SLOT || modprobe ax_pcie_host_dev
modprobe ax_pcie_msg && modprobe ax_pcie_mmb && modprobe axcl_host
echo "waiting for the handshake (up to ~2 min)..."; sleep 90
timeout 30 axcl-smi || echo "axcl-smi did not answer yet; check dmesg for 'handshake'"
