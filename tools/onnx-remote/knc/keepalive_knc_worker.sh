#!/usr/bin/env bash
# Keep the k1om worker running on the card: run from a service on the host VM (see README.md). Waits for the card
# to be online (micctrl -s) and redeploys the worker after a card reboot or a crash.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-/root/onnx-remote/knc/onnx-remote-worker-k1om}
while true; do
  state=$(micctrl -s 2>/dev/null || true)
  # After a reset the card stays in 'ready' until it is booted (the MPSS daemon does not boot it by itself).
  if echo "$state" | grep -q 'mic0: ready'; then micctrl -b mic0 >/dev/null 2>&1 || true; fi
  if echo "$state" | grep -q 'mic0: online'; then
    "$here/deploy_knc_worker.sh" "$bin" || echo "keepalive: deploy failed, retrying"
  fi
  sleep 30
done
