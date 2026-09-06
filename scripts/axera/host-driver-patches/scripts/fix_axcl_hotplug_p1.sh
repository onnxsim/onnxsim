#!/usr/bin/env bash
set -euo pipefail

SRCDIR=/usr/src/axcl-2.25.0
KVER="$(uname -r)"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo: sudo bash $0" >&2
  exit 1
fi

if grep -q 'ax_pcie_register_hotplug_notify' "$SRCDIR/ax_pcie_dev.h"; then
  echo "Patch already applied, skipping edits."
else
  echo "Patching $SRCDIR ..."
  python3 - "$SRCDIR" <<'PYEOF'
import sys, os
srcdir = sys.argv[1]

def patch(fname, pairs):
    path = os.path.join(srcdir, fname)
    with open(path) as f:
        content = f.read()
    for i, (old, new) in enumerate(pairs, 1):
        n = content.count(old)
        if n != 1:
            print(f"ERROR: {fname} patch {i}: expected 1 match, found {n}", file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  {fname}: {len(pairs)} site(s) patched")

# ---------------------------------------------------------------- ax_pcie_dev.h
# Hotplug notification API, owned by ax_pcie_dev_host (the module bound to the
# PCI device, and the first one loaded). Higher-level modules that cache
# pointers into a device's BAR mappings register here so they can drop those
# pointers while the mappings are still valid.
patch("ax_pcie_dev.h", [(
"""extern struct ax_pcie_operation *g_pcie_opt;
extern struct axera_dev *g_axera_dev_map[MAX_PCIE_DEVICES];
""",
"""extern struct ax_pcie_operation *g_pcie_opt;
extern struct axera_dev *g_axera_dev_map[MAX_PCIE_DEVICES];

#define AX_PCIE_DEV_REMOVED	0
#define AX_PCIE_DEV_ADDED	1
typedef void (*ax_pcie_hotplug_notify_t) (unsigned int target, int event);
extern void ax_pcie_register_hotplug_notify(ax_pcie_hotplug_notify_t cb);
""")])

# ----------------------------------------------------------- ax_pcie_dev_host.c
patch("ax_pcie_dev_host.c", [
# 1. the notifier itself
("""static int ax_pcie_dev_probe(struct pci_dev *pdev,""",
"""static ax_pcie_hotplug_notify_t g_hotplug_notify_cb;
static DEFINE_MUTEX(g_hotplug_notify_lock);

void ax_pcie_register_hotplug_notify(ax_pcie_hotplug_notify_t cb)
{
	mutex_lock(&g_hotplug_notify_lock);
	g_hotplug_notify_cb = cb;
	mutex_unlock(&g_hotplug_notify_lock);
}
EXPORT_SYMBOL(ax_pcie_register_hotplug_notify);

/*
 * The lock is held across the callback so that an unregister (the callback
 * owner's module_exit) cannot complete while a callback is in flight -- it
 * would otherwise be possible to call into unloaded module text.
 */
static void ax_pcie_hotplug_notify(unsigned int target, int event)
{
	mutex_lock(&g_hotplug_notify_lock);
	if (g_hotplug_notify_cb)
		g_hotplug_notify_cb(target, event);
	mutex_unlock(&g_hotplug_notify_lock);
}

static int ax_pcie_dev_probe(struct pci_dev *pdev,"""),
# 2. announce a newly probed device once it is fully set up
("""	/* return success */
	return 0;
""",
"""	ax_pcie_hotplug_notify(ax_dev->slot_index, AX_PCIE_DEV_ADDED);

	/* return success */
	return 0;
"""),
# 3. announce removal *before* tearing anything down, so consumers can drop
# cached pointers into the BAR mappings while those mappings still exist.
("""	if (!ax_dev) {
		axera_trace(AXERA_ERR, "device is not non exist");
		return;
	}

	for (bar = BAR_0; bar <= BAR_5; bar++) {""",
"""	if (!ax_dev) {
		axera_trace(AXERA_ERR, "device is not non exist");
		return;
	}

	ax_pcie_hotplug_notify(ax_dev->slot_index, AX_PCIE_DEV_REMOVED);

	for (bar = BAR_0; bar <= BAR_5; bar++) {"""),
])

# ------------------------------------------------------------ axcl_pcie_host.c
patch("axcl_pcie_host.c", [
# 1. per-target offline flag, and index heartbeat[] by target (slot index) so
# it agrees with heart_waitqueue[]/htcondition[]/port_handle[]. It was indexed
# by enumeration order, which ax_pcie_dev_remove() reshuffles on every unplug.
("""static struct task_struct *heartbeat[32];
""",
"""static struct task_struct *heartbeat[AXERA_MAX_MAP_DEV];
/* stable thread argument: the axera_dev a slot_index lives in can be freed */
static unsigned int heartbeat_target[AXERA_MAX_MAP_DEV];
/* set before a device's BAR mappings are torn down; polling loops must bail */
static volatile bool dev_offline[AXERA_MAX_MAP_DEV];
"""),
# 2. the heartbeat poll loop reads through shared memory once a second for up
# to 50s -- it has to notice a device that went away mid-wait.
("""	ktime_get_ts64(&tv_start);
	while (1) {
		ret = axcl_heartbeat_status(hbeat, target, count);
""",
"""	ktime_get_ts64(&tv_start);
	while (1) {
		if (target < AXERA_MAX_MAP_DEV && dev_offline[target])
			return -2;
		ret = axcl_heartbeat_status(hbeat, target, count);
"""),
# 3. do not cache axdev/hbeat across the loop: on a surprise hot-unplug
# ax_pcie_dev_remove() pci_iounmap()s the BARs and kfree()s the axera_dev, so a
# pointer resolved before the loop becomes a torn-down ioremap window. Reading
# it faults unrecoverably -- the box resets without even logging an oops.
("""	axcl_trace(AXCL_DEBUG, "target 0x%x thread running", target);

	axdev = g_pcie_opt->slot_to_axdev(target);
	if (!axdev) {
		axcl_trace(AXCL_ERR, "Get axdev is failed");
		return -1;
	}

	hbeat = (struct device_heart_packet *)axdev->shm_base_virt;

	do {""",
"""	axcl_trace(AXCL_DEBUG, "target 0x%x thread running", target);

	do {"""),
# 4. ...resolve it fresh each iteration instead, after the DEAD wait (which can
# park the thread for an arbitrarily long time) and before it is dereferenced.
("""		ret = axcl_heartbeat_recv_timeout(hbeat, target, count, timeout);
""",
"""		if (target >= AXERA_MAX_MAP_DEV || dev_offline[target]) {
			axcl_trace(AXCL_ERR,
				   "dev %x is gone, heartbeat thread exiting",
				   target);
			break;
		}
		axdev = g_pcie_opt->slot_to_axdev(target);
		if (!axdev || !axdev->shm_base_virt) {
			axcl_trace(AXCL_ERR,
				   "dev %x has no mapping, heartbeat thread exiting",
				   target);
			break;
		}
		hbeat = (struct device_heart_packet *)axdev->shm_base_virt;

		ret = axcl_heartbeat_recv_timeout(hbeat, target, count, timeout);
"""),
# 5. offline/online handling + notifier registration
("""static int __init axcl_pcie_host_init(void)
""",
"""/*
 * Called from ax_pcie_dev_remove() while the device's mappings are still
 * valid. Drops every pointer this module cached into them and stops the
 * heartbeat thread, so nothing follows a stale pointer once the BARs are
 * unmapped and the axera_dev is freed.
 */
static void axcl_pcie_device_offline(unsigned int target)
{
	int port;

	if (target >= AXERA_MAX_MAP_DEV)
		return;

	/* make the heartbeat poll loop bail out of its msleep(1000) cycle */
	dev_offline[target] = true;

	mutex_lock(&ioctl_mutex);
	axcl_devices_heartbeat_status_set(target, AXCL_HEARTBEAT_DEAD);
	for (port = 0; port < MAX_MSG_PORTS; port++) {
		port_handle[target][port] = NULL;
		port_info[target][port] = 0;
	}
	mutex_unlock(&ioctl_mutex);

	/* release a thread parked on the DEAD wait so it can observe the flag */
	htcondition[target] = true;
	wake_up(&heart_waitqueue[target]);

	if (heartbeat[target]) {
		kthread_stop(heartbeat[target]);
		heartbeat[target] = NULL;
	}

	axcl_trace(AXCL_ERR, "dev %x offline: cached handles dropped", target);
}

static void axcl_pcie_hotplug_notify(unsigned int target, int event)
{
	if (event == AX_PCIE_DEV_REMOVED) {
		axcl_pcie_device_offline(target);
	} else {
		/*
		 * The device is back, but this module's per-device bring-up
		 * (firmware load, port creation, RC/EP handshake) only runs at
		 * module init, so it is not usable again until axcl_host is
		 * reloaded.
		 */
		axcl_trace(AXCL_ERR,
			   "dev %x is back; reload axcl_host to use it again",
			   target);
	}
}

static int __init axcl_pcie_host_init(void)
"""),
# 5b. create the thread under its target index, with a stable argument
("""		init_waitqueue_head(&heart_waitqueue[target]);
		htcondition[target] = true;
		heartbeat[i] =
		    kthread_create(heartbeat_recv_thread,
				   &g_axera_dev_map[i]->slot_index,
				   "heartbeat_kthd");
		if (IS_ERR(heartbeat[i]))
			return PTR_ERR(heartbeat[i]);
		wake_up_process(heartbeat[i]);
""",
"""		if (target >= AXERA_MAX_MAP_DEV) {
			axcl_trace(AXCL_ERR, "invalid target device: %u", target);
			continue;
		}

		init_waitqueue_head(&heart_waitqueue[target]);
		htcondition[target] = true;
		dev_offline[target] = false;
		heartbeat_target[target] = target;
		heartbeat[target] =
		    kthread_create(heartbeat_recv_thread,
				   &heartbeat_target[target],
				   "heartbeat_kthd");
		if (IS_ERR(heartbeat[target]))
			return PTR_ERR(heartbeat[target]);
		wake_up_process(heartbeat[target]);
"""),
# 5c. ...and stop it by the same index on unload
("""		target = g_axera_dev_map[i]->slot_index;
		if (heartbeat[i]) {
			htcondition[target] = true;
			wake_up(&heart_waitqueue[target]);
			kthread_stop(heartbeat[i]);
		}
""",
"""		target = g_axera_dev_map[i]->slot_index;
		if (target < AXERA_MAX_MAP_DEV && heartbeat[target]) {
			htcondition[target] = true;
			wake_up(&heart_waitqueue[target]);
			kthread_stop(heartbeat[target]);
			heartbeat[target] = NULL;
		}
"""),
# 6. register the notifier once this module's own state is up
("""	misc_register(&axcl_usrdev);

	return 0;
}
""",
"""	misc_register(&axcl_usrdev);

	ax_pcie_register_hotplug_notify(axcl_pcie_hotplug_notify);

	return 0;
}
"""),
# 7. unregister first thing on unload -- ax_pcie_dev_host outlives this module
("""static void __exit axcl_pcie_host_exit(void)
{
	int i;
	unsigned int target;
""",
"""static void __exit axcl_pcie_host_exit(void)
{
	int i;
	unsigned int target;

	ax_pcie_register_hotplug_notify(NULL);
"""),
])
PYEOF
fi

echo "Rebuilding axcl/2.25.0 dkms module for $KVER ..."
dkms build axcl/2.25.0 -k "$KVER" --force
dkms install axcl/2.25.0 -k "$KVER" --force

echo
echo "Modules rebuilt. NOT reloading automatically -- ax_pcie_host_dev is the"
echo "module that changed and axcl_host/ax_pcie_msg depend on it, so the whole"
echo "stack has to come down in order. Run this when you are ready:"
echo
echo "  sudo modprobe -r axcl_host ax_pcie_msg ax_pcie_mmb ax_pcie_host_dev"
echo "  sudo modprobe ax_pcie_host_dev && sudo modprobe ax_pcie_msg && sudo modprobe ax_pcie_mmb && sudo modprobe axcl_host"
echo
echo "...or just reboot, which is cleaner."
