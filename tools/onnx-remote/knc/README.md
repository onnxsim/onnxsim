# onnx-remote on Intel Knights Corner (k1om)

The reference worker (`remote_worker.cpp`, the element-wise ops of `README.md` one level up) runs on a Xeon Phi
coprocessor 3120 series card, and a host client sends it RUN requests over TCP. Profiling events come back in the
response, as for any worker.

## Requirements

- **k1om toolchain** at `NFCXX_K1OM_ROOT` (default `/opt/k1om`): Intel MPSS 3.8.6 binutils 2.22 and GCC 5.1.1 for
  `k1om-mpss-linux`, rebuilt for a modern host. nfcxx's `--target=linux_k1om` documents the same toolchain
  (`docs/notes/k1om.md`).
- **Card running** under MPSS, with its network configured (`micctrl`, `mic0`, `172.31.1.0/24`; the host side
  `172.31.1.254` on `mic0`) and root SSH key access (`micctrl --sshkeys=root`).

## Build and deploy

```sh
tools/onnx-remote/knc/build_knc_worker.sh                       # -> build/knc/onnx-remote-worker-k1om (static)
tools/onnx-remote/knc/deploy_knc_worker.sh build/knc/onnx-remote-worker-k1om   # copy to the card, start on :39501
```

`deploy_knc_worker.sh` does nothing when the worker is already running. Its settings are environment variables:
`CARD` (default `root@172.31.1.1`), `KEY` (default `/root/.ssh/id_mic`), `PORT` (default `39501`).

The card's root filesystem is in RAM, so the worker is copied again after each card boot. To keep it running on
the host, run `keepalive_knc_worker.sh` as a service: it waits until `micctrl -s` reports `mic0: online` and then
deploys the worker, and it repeats every 30 seconds, so it also covers a card reboot or a crash. `BIN` sets the
worker path.

## Check and profile

```sh
g++ -std=c++17 -O2 -w -I tools/onnx-remote tools/onnx-remote/knc/check_knc.cpp \
    tools/onnx-remote/remote_transport.cpp tools/onnx-remote/remote_profile.cpp -o check_knc
./check_knc 172.31.1.1 39501       # relu, add (broadcast), tanh: PASS/FAIL per op
g++ -std=c++17 -O2 -w -I tools/onnx-remote tools/onnx-remote/knc/prof_knc.cpp \
    tools/onnx-remote/remote_transport.cpp tools/onnx-remote/remote_profile.cpp -o prof_knc
./prof_knc 172.31.1.1 39501        # detailed profiling of 1M-element relu/add/tanh/exp
```

`prof_knc` requests `ProfilingLevel::Detailed`. Each response carries one event per operation: the worker's own
start and duration in microseconds, measured with `steady_clock` on the card, next to the host's round-trip time.

Measured on the card (1,048,576 float32 elements, three repetitions each, the 3120 series card in this setup):

| op   | worker duration | host round trip |
|------|-----------------|-----------------|
| relu | 26 ms           | 106-125 ms      |
| add  | 711 ms          | 845-871 ms      |
| tanh | 1010 ms         | 1094-1099 ms    |
| exp  | 824 ms          | 912-914 ms      |

The worker is single-threaded scalar code, so these are scalar-loop times, not the card's peak. The gap between
worker and host times is transfer and transport cost (about 100 ms for 4 MB each way over the `mic0` link).

## Limits

- Element-wise reference ops only. No ONNX graph execution, no ORT, no vector code (the reference worker is
  scalar).
- Tested only on this one card, with MPSS 3.8.6 and the k1om GCC 5.1.1 toolchain.
