#!/usr/bin/env bash
set -euo pipefail

SRCDIR=/usr/src/axcl-2.25.0
KVER="$(uname -r)"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo: sudo bash $0" >&2
  exit 1
fi

if ! grep -q 'ax_pcie_register_hotplug_notify' "$SRCDIR/ax_pcie_dev.h"; then
  echo "ERROR: patch 1 is not applied -- run fix_axcl_hotplug_p1.sh first." >&2
  exit 1
fi

if grep -q 'axcl_hotplug_wq' "$SRCDIR/axcl_pcie_host.c"; then
  echo "Patch already applied, skipping edits."
else
  echo "Patching $SRCDIR/axcl_pcie_host.c ..."
  python3 - "$SRCDIR" <<'PYEOF'
import sys, os
srcdir = sys.argv[1]
path = os.path.join(srcdir, "axcl_pcie_host.c")
with open(path) as f:
    content = f.read()

pairs = [
# 1. workqueue for deferred bring-up
("""/* set before a device's BAR mappings are torn down; polling loops must bail */
static volatile bool dev_offline[AXERA_MAX_MAP_DEV];
""",
"""/* set before a device's BAR mappings are torn down; polling loops must bail */
static volatile bool dev_offline[AXERA_MAX_MAP_DEV];
/* bringing a reconnected device up blocks for a long time; keep it off the
 * PCI probe path. Ordered so two devices are never brought up concurrently. */
static struct workqueue_struct *axcl_hotplug_wq;

struct axcl_online_work {
	struct work_struct work;
	unsigned int target;
};
"""),

# 2. per-target bring-up + its work item, defined before the notify callback
# that queues them.
("""static void axcl_pcie_hotplug_notify(unsigned int target, int event)
{""",
"""/*
 * Bring one reconnected device up: the same sequence axcl_pcie_host_init()
 * performs, for a single target. Kept separate from init's phased loops so
 * that init still starts every device's firmware before waiting on any
 * handshake (which lets multiple cards boot concurrently).
 *
 * Only ever called from the hotplug workqueue -- axcl_firmware_load() pushes
 * ~150MB over PCIe and ax_pcie_msg_check_remote() can block for two minutes,
 * so this must not run in a PCI probe callback.
 */
static int axcl_pcie_device_online(unsigned int target)
{
	struct axera_dev *ax_dev;
	int ret;

	if (target >= AXERA_MAX_MAP_DEV)
		return -1;

	ax_dev = g_pcie_opt->slot_to_axdev(target);
	if (!ax_dev) {
		axcl_trace(AXCL_ERR, "dev %x has no axdev, cannot bring up",
			   target);
		return -1;
	}

	dev_offline[target] = false;

	/* 1. firmware loading */
	ret = axcl_firmware_load(ax_dev);
	if (ret < 0) {
		axcl_trace(AXCL_ERR, "Device %x firmware load failed.", target);
		goto dead;
	}
	if (dev_offline[target])
		goto gone;

	/* 2. create port */
	ret = axcl_common_prot_create(target);
	if (ret < 0)
		goto dead;
	if (dev_offline[target])
		goto gone;

	/* 3. rc and ep handshake */
	axcl_trace(AXCL_ERR, "axcl wait dev %x handshake...", target);
	ret = ax_pcie_msg_check_remote(target);
	if (ret < 0) {
		axcl_trace(AXCL_ERR, "axcl pcie check remote device %x failed",
			   target);
		goto dead;
	}
	if (dev_offline[target])
		goto gone;
	axcl_devices_heartbeat_status_set(target, AXCL_HEARTBEAT_ALIVE);

	/* 4. timestamp sync */
	axcl_timestamp_sync(target);

	/* 5. heartbeat recv thread */
	init_waitqueue_head(&heart_waitqueue[target]);
	htcondition[target] = true;
	heartbeat_target[target] = target;
	heartbeat[target] = kthread_create(heartbeat_recv_thread,
					   &heartbeat_target[target],
					   "heartbeat_kthd");
	if (IS_ERR(heartbeat[target])) {
		ret = PTR_ERR(heartbeat[target]);
		heartbeat[target] = NULL;
		goto dead;
	}
	wake_up_process(heartbeat[target]);

	axcl_trace(AXCL_ERR, "dev %x back online", target);
	return 0;

gone:
	axcl_trace(AXCL_ERR, "dev %x went away again during bring-up", target);
	ret = -1;
dead:
	axcl_devices_heartbeat_status_set(target, AXCL_HEARTBEAT_DEAD);
	return ret;
}

static void axcl_pcie_device_online_work(struct work_struct *work)
{
	struct axcl_online_work *ow =
	    container_of(work, struct axcl_online_work, work);
	unsigned int target = ow->target;

	kfree(ow);
	axcl_pcie_device_online(target);
}

static void axcl_pcie_hotplug_notify(unsigned int target, int event)
{"""),

# 3. the reconnect branch now schedules a real bring-up
("""	} else {
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
""",
"""	} else {
		struct axcl_online_work *ow;

		if (!axcl_hotplug_wq)
			return;

		ow = kmalloc(sizeof(*ow), GFP_KERNEL);
		if (!ow) {
			axcl_trace(AXCL_ERR,
				   "dev %x: no memory to schedule bring-up",
				   target);
			return;
		}
		INIT_WORK(&ow->work, axcl_pcie_device_online_work);
		ow->target = target;
		queue_work(axcl_hotplug_wq, &ow->work);
		axcl_trace(AXCL_ERR, "dev %x is back, bring-up scheduled",
			   target);
	}
"""),

# 4. create the queue before the notifier that feeds it
("""	ax_pcie_register_hotplug_notify(axcl_pcie_hotplug_notify);
""",
"""	axcl_hotplug_wq = alloc_ordered_workqueue("axcl_hotplug", 0);
	if (!axcl_hotplug_wq) {
		axcl_trace(AXCL_ERR, "alloc hotplug workqueue failed");
		return -ENOMEM;
	}

	ax_pcie_register_hotplug_notify(axcl_pcie_hotplug_notify);
"""),

# 5. ...and tear it down after the notifier is gone, so nothing can requeue.
# destroy_workqueue() waits for an in-flight bring-up, which can take a
# couple of minutes if it is sitting in the handshake.
("""	ax_pcie_register_hotplug_notify(NULL);
""",
"""	ax_pcie_register_hotplug_notify(NULL);

	if (axcl_hotplug_wq) {
		destroy_workqueue(axcl_hotplug_wq);
		axcl_hotplug_wq = NULL;
	}
"""),
]

for i, (old, new) in enumerate(pairs, 1):
    n = content.count(old)
    if n != 1:
        print(f"ERROR: patch {i}: expected 1 match, found {n}", file=sys.stderr)
        sys.exit(1)
    content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)
print(f"  axcl_pcie_host.c: {len(pairs)} site(s) patched")
PYEOF
fi

echo "Rebuilding axcl/2.25.0 dkms module for $KVER ..."
dkms build axcl/2.25.0 -k "$KVER" --force
dkms install axcl/2.25.0 -k "$KVER" --force

echo "Reloading axcl_host (only that module changed this time) ..."
modprobe -r axcl_host
modprobe axcl_host

echo "Done."
lsmod | grep -E '^ax_pcie_mmb|^axcl_host|^ax_pcie_host_dev|^ax_pcie_msg'
modinfo axcl_host | grep -E 'filename|srcversion'
