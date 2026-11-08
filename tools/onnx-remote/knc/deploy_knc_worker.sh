#!/usr/bin/env bash
# Copy the k1om worker to the card and start it on port $PORT (default 39501), unless it already answers.
# CARD (default root@172.31.1.1) is reached with SSH key $KEY (default /root/.ssh/id_mic, from micctrl --sshkeys).
set -euo pipefail
bin=${1:?usage: deploy_knc_worker.sh path/to/onnx-remote-worker-k1om}
card=${CARD:-root@172.31.1.1}; key=${KEY:-/root/.ssh/id_mic}; port=${PORT:-39501}
ssh_opts=(-i "$key" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
if ssh "${ssh_opts[@]}" "$card" "grep -qa '[o]nnx-remote-worker-k1om' /proc/[0-9]*/cmdline"; then echo "worker already running on $card"; exit 0; fi
scp "${ssh_opts[@]}" -q "$bin" "$card:/tmp/onnx-remote-worker-k1om"
ssh "${ssh_opts[@]}" "$card" "chmod +x /tmp/onnx-remote-worker-k1om; nohup /tmp/onnx-remote-worker-k1om --port $port >/tmp/worker.log 2>&1 &"
sleep 2
ssh "${ssh_opts[@]}" "$card" "grep -qa '[o]nnx-remote-worker-k1om' /proc/[0-9]*/cmdline" && echo "worker started on $card:$port"
