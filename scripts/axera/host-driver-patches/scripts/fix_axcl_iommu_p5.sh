#!/usr/bin/env bash
# Fix 5 -- ax_mmb hands the AX650N raw host physical addresses, which the
# card cannot DMA to once the IOMMU is on (AMD-Vi IO_PAGE_FAULT storms, the
# runtime never comes up; this is why `amd_iommu=off` was on the command
# line). Allocate and map every card-visible buffer against the card's own
# pci_dev through the DMA API instead. Idempotent; rebuilds via DKMS.
#
#   sudo bash scripts/fix_axcl_iommu_p5.sh
#   sudo modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_host_dev
#   sudo modprobe ax_pcie_host_dev && sudo modprobe ax_pcie_msg && \
#     sudo modprobe ax_pcie_mmb && sudo modprobe axcl_host
set -euo pipefail
SRCDIR=${SRCDIR:-/usr/src/axcl-2.25.0}
KVER="$(uname -r)"
if [ "$(id -u)" -ne 0 ]; then echo "Run this with sudo: sudo bash $0" >&2; exit 1; fi

if grep -q 'ax_mmb_dma_device' "$SRCDIR/ax_mmb.c"; then
  echo "Patch already applied, skipping edits."
else
  echo "Patching $SRCDIR/ax_mmb.c ..."
  python3 - "$SRCDIR" <<'PYEOF'
import sys, os
path = os.path.join(sys.argv[1], "ax_mmb.c")
content = open(path).read()

def patch(old, new, count=1):
    global content
    n = content.count(old)
    if n != count:
        print(f"ERROR: expected {count} match(es), found {n} for:\n{old[:120]!r}", file=sys.stderr)
        sys.exit(1)
    content = content.replace(old, new)

# 1. pci_dev is needed for the card's DMA device
patch("#include <linux/dma-mapping.h>\n",
      "#include <linux/dma-mapping.h>\n#include <linux/pci.h>\n")

# 2. remember which device a buffer was allocated/mapped for
patch("\tunsigned long dma_buf_size;\n};\n\nstruct ax_mem_info {",
      "\tunsigned long dma_buf_size;\n\tstruct device *dma_dev;\n};\n\nstruct ax_mem_info {")

# 3. the device to allocate against, plus IOMMU-aware scatterlist free
patch("""static void ax_sglist_free_memory(int count, struct ax_scatterlist *scatterlist)
{
\tint i;

\tfor (i = 0; i < count; i++) {
\t\tif ((void *)(scatterlist->sg_list[i].virtaddr) != NULL)
\t\t\tkfree((void *)(scatterlist->sg_list[i].virtaddr));
\t}
}
""",
"""/*
 * Every address this module hands the AX650N must be one the card can DMA
 * to *as seen through the IOMMU*. The misc device this module used to
 * allocate against has no IOMMU domain, so dma_alloc_coherent() on it (and
 * virt_to_phys() on kmalloc memory) yield raw host physical addresses; with
 * the IOMMU on, the card's DMA then faults (AMD-Vi IO_PAGE_FAULT) and the
 * runtime never comes up. Allocate and map against the card's own pci_dev
 * instead: with the IOMMU off that is the identity mapping the old code
 * assumed, with it on it is the mapping the card actually needs.
 * Single-card assumption: every buffer is mapped for the first device.
 */
static struct device *ax_mmb_dma_device(void)
{
\tif (g_pcie_opt && g_pcie_opt->remote_device_number > 0 &&
\t    g_axera_dev_map[0] && g_axera_dev_map[0]->pdev)
\t\treturn &g_axera_dev_map[0]->pdev->dev;
\treturn ax_mmb_miscdev.this_device;
}

static void ax_sglist_free_memory(struct device *dev, int count,
\t\t\t\t  struct ax_scatterlist *scatterlist)
{
\tint i;

\tfor (i = 0; i < count; i++) {
\t\tif ((void *)(scatterlist->sg_list[i].virtaddr) != NULL) {
\t\t\tdma_unmap_single(dev, scatterlist->sg_list[i].phyaddr,
\t\t\t\t\t PAGE_ALIGN(scatterlist->sg_list[i].size),
\t\t\t\t\t DMA_BIDIRECTIONAL);
\t\t\tkfree((void *)(scatterlist->sg_list[i].virtaddr));
\t\t}
\t}
}
""")

# 4. scatterlist chunks: kmalloc, then map for the card (page-aligned so an
#    untrusted-device mapping never bounces through swiotlb)
patch("static int ax_sglist_alloc_memory(int size, unsigned long *virtaddr, unsigned long *phyaddr)\n{",
      "static int ax_sglist_alloc_memory(struct device *dev, int size,\n\t\t\t\t  unsigned long *virtaddr, unsigned long *phyaddr)\n{")
patch("\ttmpvirtaddr = kmalloc(size, GFP_KERNEL | __GFP_NORETRY | __GFP_NOWARN);\n",
      "\ttmpvirtaddr = kmalloc(PAGE_ALIGN(size), GFP_KERNEL | __GFP_NORETRY | __GFP_NOWARN);\n")
patch("""\ttmpphyaddr = virt_to_phys(tmpvirtaddr);
\tif (tmpphyaddr < 0) {
\t\tPCIe_PRINT(PCIe_ERR, "%x dmabuf virt to phys failed\\n", size);
\t\tkfree(tmpvirtaddr);
\t\treturn -1;
\t}
""",
"""\ttmpphyaddr = dma_map_single(dev, tmpvirtaddr, PAGE_ALIGN(size),
\t\t\t\t    DMA_BIDIRECTIONAL);
\tif (dma_mapping_error(dev, tmpphyaddr)) {
\t\tPCIe_PRINT(PCIe_ERR, "%x dmabuf dma_map_single failed\\n", size);
\t\tkfree(tmpvirtaddr);
\t\treturn -1;
\t}
""")
patch("\tmemset((void *)ax_mmb->scatterlist, 0, sizeof(struct ax_scatterlist));\n",
      "\tmemset((void *)ax_mmb->scatterlist, 0, sizeof(struct ax_scatterlist));\n\tax_mmb->dma_dev = ax_mmb_dma_device();\n")
patch("\t\tret = ax_sglist_alloc_memory(alloc_size, &virtaddr, &phyaddr);\n",
      "\t\tret = ax_sglist_alloc_memory(ax_mmb->dma_dev, alloc_size, &virtaddr, &phyaddr);\n")
patch("ax_sglist_free_memory(count, ax_mmb->scatterlist);",
      "ax_sglist_free_memory(ax_mmb->dma_dev, count, ax_mmb->scatterlist);", count=2)
patch("ax_sglist_free_memory(ax_mmb->scatterlist->num, ax_mmb->scatterlist);",
      "ax_sglist_free_memory(ax_mmb->dma_dev, ax_mmb->scatterlist->num, ax_mmb->scatterlist);")

# 5. coherent buffers: allocate on the card's device, free on the same one
patch("\tax_mmb->ax_mmb_buf =\n\t    dma_alloc_coherent(ax_mmb_miscdev.this_device, dma_buf_size,",
      "\tax_mmb->dma_dev = ax_mmb_dma_device();\n\tax_mmb->ax_mmb_buf =\n\t    dma_alloc_coherent(ax_mmb->dma_dev, dma_buf_size,")
patch("\t\t\tdma_free_coherent(ax_mmb_miscdev.this_device,\n",
      "\t\t\tdma_free_coherent(ax_mmb->dma_dev,\n")

# 6. mmap: the value the card sees is no longer a CPU-physical page. Scatter
#    chunks map by the kernel buffer's real physical page; coherent buffers
#    through dma_mmap_coherent (which also handles IOMMU-backed, physically
#    non-contiguous coherent memory).
patch("\t\t\tpfn_start = ax_mmb->scatterlist->sg_list[i].phyaddr >> PAGE_SHIFT;\n",
      "\t\t\tpfn_start = virt_to_phys((void *)ax_mmb->scatterlist->sg_list[i].virtaddr) >> PAGE_SHIFT;\n")
patch("""\t\tret =
\t\t    remap_pfn_range(vma, vma->vm_start, pfn_start, size,
\t\t\t\t    vma->vm_page_prot);
""",
"""\t\tif (ax_mmb->ax_mmb_buf)
\t\t\tret = dma_mmap_coherent(ax_mmb->dma_dev, vma,
\t\t\t\t\t\tax_mmb->ax_mmb_buf,
\t\t\t\t\t\tax_mmb->dma_phy_addr,
\t\t\t\t\t\tax_mmb->dma_buf_size);
\t\telse
\t\t\tret = remap_pfn_range(vma, vma->vm_start, pfn_start,
\t\t\t\t\t      size, vma->vm_page_prot);
""")

open(path, "w").write(content)
print("  ax_mmb.c: patched")
PYEOF
fi

echo "Rebuilding axcl/2.25.0 dkms module for $KVER ..."
dkms build axcl/2.25.0 -k "$KVER" --force
dkms install axcl/2.25.0 -k "$KVER" --force
echo "Done. ax_pcie_mmb now depends on ax_pcie_host_dev; reload the whole stack in dependency order:"
echo "  sudo modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_host_dev"
echo "  sudo modprobe ax_pcie_host_dev && sudo modprobe ax_pcie_msg && sudo modprobe ax_pcie_mmb && sudo modprobe axcl_host"
