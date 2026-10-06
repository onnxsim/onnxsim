# Flashing a Jetson Orin Nano (JetPack 7.2.1) with SDK Manager in Docker

How the Jetson Orin Nano Super Developer Kit was flashed headlessly from an Ubuntu host
using NVIDIA SDK Manager inside a container, what broke, and the workarounds. Nothing here
is onnxsim code; it is the bring-up recipe for the board used by the Jetson/TensorRT
experiments.

Result: JetPack 7.2.1 (L4T R39.2.1, Ubuntu 24.04), root filesystem on NVMe, QSPI firmware
updated, a preconfigured user (no first-boot wizard), reachable over SSH on the USB
network link.

## Host and container

- Host: Ubuntu, Docker, board connected over USB-C (data port) to the host.
- Container: `jetson-sdkmanager-u2204-nfsfix`, an Ubuntu 22.04 SDK Manager image running
  `sdkmanager --cli`, with these mounts: the host's `/dev` (so the USB device nodes are visible), the host's
  `/run/rpcbind.sock`, and `/home/takecheeze/jetson-sdkm-share/home/nvidia` mapped to the
  container's `/home/nvidia` (so downloads and flash logs persist on the host).
- Enter it with `docker exec -it --user nvidia jetson-sdkmanager-u2204-nfsfix bash`.
- The BSP lives at
  `/home/nvidia/nvidia/nvidia_sdk/JetPack_7.2.1_Linux_JETSON_ORIN_NANO_TARGETS/Linux_for_Tegra`
  (called `L4T` below).

The container needs extra setup that the stock SDK Manager image does not do for initrd
flashing:

1. **NFS server running inside the container.** The flash initrd on the board mounts the
   kernel-flash `tmp` and `images` directories over NFS (IPv6, `fc00:1:1::/48`). Start
   `nfs-kernel-server`, with the host's rpcbind socket reachable, and put this in the
   container's `/etc/exports` (`$L4T` is the path above):

   ```
   $L4T/tools/kernel_flash/tmp    fc00:1:1::/48(rw,nohide,insecure,no_subtree_check,async,no_root_squash)
   $L4T/tools/kernel_flash/images fc00:1:1::/48(rw,nohide,insecure,no_subtree_check,async,no_root_squash)
   ```

   Verify with `rpcinfo -p` (an `nfs` program must be listed) and `showmount -e`.
2. **Marker file** `/run/nvidia_initrd_flash/docker_host_network` (tells NVIDIA's scripts
   the host network is shared).
3. **USB IPv6 watcher**, `/usr/local/sbin/jetson-usb-ipv6-watch`, left running in the
   background: when the board re-enumerates in initrd mode (USB product `7035`) it assigns
   `fc00:1:1::1/48` to the new USB NCM interface; the target uses `fc00:1:1::2`. Without it
   the flash hangs at "Waiting for target to boot-up".
4. **`USER=root`.** The helper scripts check for root; the container's environment had
   `USER=nvidia`, which made the check fail even under `sudo`. Run with `USER=root sudo -E`.

`/run` is cleared on container restart: re-create the marker, exports, NFS service and
watcher before every flash session.

## Pitfall: the SDK Manager CLI screen is not a progress indicator

SDK Manager's CLI sat on a spinner with an empty panel, its log stopped after the
downloads, and no flash worker was running. Do not trust it. The flash that worked was run
by calling NVIDIA's bundled helper directly (below). SDK Manager is still useful for
downloading the BSP and rootfs and creating `Linux_for_Tegra`.

## Putting the board in recovery

Power off, jumper `FC REC` to `GND`, power on with USB connected. `lsusb` must show
`0955:7523` (APX). A board running L4T shows `0955:7020` and cannot be flashed.

## Pre-create the user (avoids the OEM wizard)

The default "runtime OEM config" wizard only runs on a display console. The USB serial
console (`/dev/ttyACM*`, `ttyGS0` on the board) shows just a login prompt, so a headless
board is stuck at first boot. Create the user in the rootfs before flashing:

```sh
cd "$L4T"
sudo ./tools/l4t_create_default_user.sh -u <user> -p <password> -n <hostname> --accept-license
```

- Do not pass `-a` unless you want passwordless autologin on the serial console and GDM.
- The script's "autologin off" path removes the wrong `serial-getty@tty*` path; if you ran
  it with `-a` once, delete `rootfs/etc/systemd/system/serial-getty@tty*.service.d/` by
  hand.
- It needs `qemu-user-static` in the container.

## Flash

From `$L4T`, with the board in recovery and the container prerequisites in place:

```sh
USER=root sudo -E ./nvsdkmanager_flash.sh --storage nvme0n1p1 > /home/nvidia/reflash.log 2>&1
```

This wraps `tools/kernel_flash/l4t_initrd_flash.sh` and flashes the NVMe rootfs plus the QSPI
firmware. It takes roughly 10 minutes after the images are built. Success is
`Successfully flashed the external device`, `Successfully flashed the QSPI`, then
`Flash is successful`. Without `--nv-auto-config --username ...` the user comes from the
rootfs prepared above; with them SDK Manager generates a pre-config file instead (the first
flash used that and still ended at the wizard on this board).

Harmless log noise: `File rcm_state open failed`, `failed to import T264 module`,
`ipv6: address already assigned` (the watcher got there first), and a "backup GPT table is
corrupt" warning while the new partition table is written.

The per-device log is `L4T/initrdlog/flash_<usb-port>_*.log`.

## After the flash

The board boots, appears on the host as a USB network interface (`192.168.55.100/24` on
the host, board at `192.168.55.1`), and SSH answers within a couple of minutes:

```sh
ssh <user>@192.168.55.1
```

Then copy in an SSH key and change the password. Further setup (JetPack components, Tailscale)
is plain apt on the board; the flash installs only the base L4T packages, so
`sudo apt update && sudo apt install nvidia-jetpack` is needed for CUDA/TensorRT.

## Host gotchas seen on the way

- ModemManager probes new `/dev/ttyACM*` devices and writes junk to them; the node also
  renumbers (`ttyACM0` to `ttyACM1`) after a reboot.
- Do not run `pkill -f <name>` from a tool shell: it matches the shell's own command line.
- Do not rerun the flash unless a reflash is intended; it erases the NVMe.

## Running TensorRT Edge-LLM on the board (verified)

Full TensorRT-LLM is not the supported path on Orin with JetPack 7.2; NVIDIA's route is
TensorRT Edge-LLM (FP16/INT8/INT4 only on Orin). Verified on the flashed board:

1. `sudo apt install nvidia-jetpack cmake build-essential git` (CUDA 13.2, TensorRT 10.16).
2. Clone `NVIDIA/TensorRT-Edge-LLM` at the same tag as the host that exports the ONNX
   (v0.10.1 here), `git submodule update --init --recursive`, then
   `cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL`
   and `cmake --build . -j3` (on-device, 8 GB RAM, about 15 minutes).
3. Export the ONNX on an x86 GPU host, `scp` it over, then on the board
   `llm_build --onnxDir <onnx>/llm --engineDir <engines> --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 4096`
   (Qwen3-0.6B FP16: about 3 minutes). Never copy engines between JetPack versions.
4. `llm_inference` / `llm_bench` run the engine. Qwen3-0.6B FP16, board in the 25W mode:
   prefill 4.5k tok/s (128 tokens) and 8.0k tok/s (512), decode about 49 tok/s at 128
   past tokens and 45 tok/s at 1024.

### onnxsim-simplified exports on the Orin (Edge-LLM v0.10.1, greedy, 3 prompts)

Both simplified Qwen3-0.6B exports (`edgellm_simplify.py` output) pass `llm_build` and
`llm_inference` on JetPack 7.2 unchanged (engines within 0.05% of the original's size,
about 1.2 GB). The 2.6 GB `qwen3_sim` export produced text identical to the original's
engine on all 3 prompts; `qwen3_sim2` matched on 1/3 and diverged late (char 433 and 91)
on the others, the kind of drift expected from FP16 argmax near-ties, not a build failure.
Host (RTX 5050) and board outputs also differ from each other for the same reason.

### INT4 (W4A16 AWQ) vs FP16, and the memory-bandwidth ideal (Qwen3-0.6B, Orin Nano Super)

Recipe: on the x86 host, `tensorrt-edgellm-quantize llm --quantization int4_awq
--lm_head_quantization int4_awq`, then `tensorrt-edgellm-export`, then `llm_build` on the
board. On an 8 GB GPU the quantizer OOMs in the lm_head AWQ search because
`quantization/quantize.py` hard-codes calibration `batch_size = 16` for `int4_awq`
(16 x 512 x 151936 fp32 logits); a scratch copy of the package with batch size 2
and `--num_samples 64` works. The INT4 engine is 322 MB (FP16: 1.2 GB).

Clocks locked with `nvpmodel -m 2` (MAXN_SUPER) + `jetson_clocks` (GPU 1020 MHz, EMC
3199 MHz), then restored to 25W. Measured read bandwidth 80 GB/s with a plain CUDA read
kernel; LPDDR5 peak is 102.4 GB/s. Per-token decode traffic: 596M weight parameters
(28 layers x 15.7M + lm_head 155.6M; the embedding is a lookup and not read) plus
115 KB of KV cache per past token.

| decode, batch 1 | FP16 | INT4 |
|---|---|---|
| bytes per token, 128 past tokens | 1207 MB | 322 MB |
| measured | 14.1 ms (71 tok/s) | 6.29 ms (159 tok/s) |
| ideal at 102.4 GB/s | 11.8 ms | 3.1 ms |
| achieved bandwidth | 86 GB/s (84% of peak) | 51 GB/s (50% of peak) |
| measured, 1024 past tokens | 15.1 ms (66 tok/s) | 7.23 ms (138 tok/s) |

FP16 decode is already at the memory roofline. INT4 is 2.2x faster, against 3.75x fewer
bytes, so it reaches about 60% of its own roofline; the remaining ~2.5 ms is not
bandwidth. Cause not profiled (candidates: per-layer launch/plugin overhead, small W4A16
kernels at M=1).

Prefill (compute-bound; ideal assumes about 17 TFLOPS dense FP16, derived from the 67 sparse
INT8 TOPS spec, so approximate): 128 tokens: FP16 21.4 ms, INT4 19.3 ms, ideal about 7 ms;
512 tokens: FP16 64.7 ms, INT4 68.0 ms (INT4 is slower here), ideal about 27 ms. That is
roughly 30-40% of peak. The earlier 25W numbers (49 tok/s FP16 decode) were with default
DVFS clocks and EMC at 2133 MHz.

INT4 output text is fluent but differs from FP16 from the first characters on all 3
prompts; no accuracy metric was run.

### Qwen3-4B INT4 (official `Qwen/Qwen3-4B-AWQ`) on the Orin Nano

A supported pre-quantized checkpoint, so no local quantization: `tensorrt-edgellm-export
<snapshot> OUT` on the host (CPU is enough), copy `OUT/llm` (3.3 GB) over, then `llm_build`
on the board (same flags, about 10 minutes). Engine 2.67 GB (INT4 body 1.87 GB plus the
fp16 lm_head, 0.78 GB); it loads and runs in the board's 8 GB. The three sample prompts
give fluent, correct answers ("The capital of Japan is Tokyo").

Locked MAXN_SUPER clocks, batch 1, bytes per token = engine size plus 147 KB of KV cache
per past token:

| | measured | ideal at 102.4 GB/s | achieved bandwidth |
|---|---|---|---|
| decode, 128 past tokens (2.69 GB) | 34.3 ms (29.1 tok/s) | 26.2 ms | 78 GB/s (76% of peak) |
| decode, 1024 past tokens (2.82 GB) | 36.4 ms (27.5 tok/s) | 27.5 ms | 77 GB/s (76%) |
| prefill 128 tokens | 121 ms (1054 tok/s) | about 55 ms | about 45% of compute peak |
| prefill 512 tokens | 437 ms (1171 tok/s) | about 219 ms | about 50% |

Decode efficiency is much better than for the 0.6B INT4 engine (50% of peak) because a
roughly fixed per-token cost of about 2.5 ms matters less against 34 ms. The fp16 lm_head
is 29% of the bytes read per token, so quantizing it as well should cut decode time by
about a quarter (not tried; the official checkpoint leaves it fp16). Note: do not use
`jetson_clocks --store` here; it hung once after an earlier failed `--restore`. Switching
back with `nvpmodel -m 1` restored normal clocks.

### Qwen3-8B INT4 (official `Qwen/Qwen3-8B-AWQ`) on the Orin Nano

It fits, but only with `--externalize-weights int4_ffn` at export. A plain export builds a
4.8 GB weight blob and `llm_build` dies with `OutOfMemory (Requested size was 4294967296
bytes)` in `globWriter`: peak RAM use is about 5 GB, then it asks for another 4 GiB. Dropping
the page cache and adding a 16 GB swap file did not help (the GPU allocation is not
swappable; the swap file stayed unused). With `int4_ffn` externalized the FFN weights
(2.8 GB) live in `external_int4_ffn_weights.safetensors` next to the 2.04 GB engine, and
the build succeeds with `--maxInputLen 512 --maxKVCacheCapacity 2048`. Copy the whole
`llm` directory (8.3 GB, of which `model.onnx.data` is not needed at runtime).

Locked MAXN_SUPER clocks, batch 1; bytes per token = 2.04 GB engine + 2.80 GB external
weights (INT4 body 3.6 GB plus the fp16 lm_head, 1.25 GB) plus 147 KB of KV per past token:

| | measured | ideal at 102.4 GB/s | achieved bandwidth |
|---|---|---|---|
| decode, 128 past tokens (4.86 GB) | 60.2 ms (16.6 tok/s) | 47.5 ms | 81 GB/s (79% of peak) |
| decode, 1024 past tokens (4.99 GB) | 61.8 ms (16.2 tok/s) | 48.8 ms | 81 GB/s (79%) |
| prefill 128 tokens | 205 ms (624 tok/s) | about 105 ms | about 51% of compute peak |
| prefill 512 tokens | 759 ms (675 tok/s) | about 419 ms | about 55% |

**The generated text is broken**, so these timings are for an engine that does not produce
correct output. On the three sample prompts (greedy, `input.json`) it repeats itself
("a type of a type of a type of ..."), answers the Fibonacci prompt with one restated
sentence, and drifts into "Mind is mind is mind is ..." on the Japan prompt. The 4B engine
built the same way (but without externalized weights) answers all three correctly. The
runtime log shows all 216 external weight tensors loaded and validated against the engine,
so the cause is not a missing file. Not yet isolated; untested candidates: the externalized
weight path itself, the reduced `--maxInputLen 512 --maxKVCacheCapacity 2048`, and the
Qwen3-8B-AWQ checkpoint under the Edge-LLM INT4 plugin. Do not quote the 8B speed as a
working configuration until the output is checked against a reference.

For what the timing is worth: decode is 79% of the bandwidth peak (0.6B: 50%, 4B: 76%),
consistent with a small fixed per-token cost, and free memory stayed above 5 GB during the
benchmark, so build-time memory, not runtime, is the limit at this size.

Follow-up: `--externalize-weights all` (adds the fp16 lm_head, 1.24 GB, as an external file;
the engine shrinks to 795 MB and builds in about 90 seconds instead of 8 minutes) gives
byte-identical garbage to the `int4_ffn` export, so externalization is not the cause of the
broken 8B text. At the default context (`--maxInputLen 1024 --maxKVCacheCapacity 4096`) the
fully externalized engine builds but `llm_inference` fails with `cudaMalloc ... out of
memory` (`NvMapMemAlloc ... error 12`) at runtime, even with 5.7 GB reported free and after
dropping caches and compacting memory; `512 / 2048` runs. Remaining suspects: the
Qwen3-8B-AWQ checkpoint under the Edge-LLM INT4 plugin, the reduced context, or an
Orin-specific kernel issue; a host (RTX 5050) run of the same export would separate the
last from the first two.

## Scripts

`scripts/nvidia/jetson/edgellm_bench.sh ENGINE_DIR...` produced the clock-locked numbers
above (it also prints the ideal decode time and achieved bandwidth);
`scripts/nvidia/jetson/read_bandwidth.cu` measured the read bandwidth;
`scripts/nvidia/jetson/quantize_int4_small_gpu.sh` made the 0.6B INT4 checkpoint on the
8 GB host GPU. The TensorRT RPC worker is `tools/onnx-remote/remote_tensorrt_worker.cpp`.
