#!/usr/bin/env python3
"""PLaMo 3 decode on the phone: the host embeds, the phone runs each chunk program, the host keeps the KV caches.

Reads the chunk programs written by export_chunks.py (compiled to .tghx artifacts by compile_v69.sh) and checks every
step's logits against the NumPy reference (plamo3_reference). Talks to onnx-remote-hexagon-worker on RUNNER_PORT.

  run_phone.py CHECKPOINT CHUNK_DIR TGHX_DIR [--port 39520] [--cache-len 15] [--ids 5,17,200,9001] [--steps 4]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..", "..")
sys.path.insert(0, os.path.join(REPO, "tools", "onnx-remote", "python"))
sys.path.insert(0, os.path.join(REPO, "scripts", "axera"))

import onnx_remote_client as C  # noqa: E402
import plamo3_reference as R  # noqa: E402


def chunk_ranges(chunk_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.onnx"))):
        a, b = os.path.basename(p)[len("chunk_") : -len(".onnx")].split("_")
        out.append((int(a), int(b), p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("chunk_dir")
    ap.add_argument("tghx_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39520)
    ap.add_argument("--cache-len", type=int, default=15)
    ap.add_argument("--ids", default="5,17,200,9001")
    ap.add_argument(
        "--steps",
        type=int,
        default=4,
        help="decode steps after the prompt (each checked)",
    )
    ap.add_argument(
        "--fresh-worker",
        action="store_true",
        help="restart the worker before every chunk (debugging only: programs share one worker safely since the skels hide their symbols)",
    )
    ap.add_argument(
        "--worker-binary-on-phone",
        default="/data/local/tmp/onnx-remote-hexagon/onnx-remote-hexagon-worker",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="at position 0, compare the hidden state after every chunk with the reference",
    )
    a = ap.parse_args()

    ref = R.Plamo3Reference.from_checkpoint(a.checkpoint)
    emb = ref.w("model.embed_tokens.weight")
    nkv, hd, H, C_ = ref.nkv, ref.hd, ref.H, a.cache_len
    chunks = []
    for lo, hi, path in chunk_ranges(a.chunk_dir):
        tg = os.path.join(a.tghx_dir, f"chunk_{lo:02d}_{hi:02d}.tghx")
        with open(tg, "rb") as f:
            chunks.append((lo, hi, os.path.basename(tg), f.read()))
    final_gain = ref.w("model.norm.weight")
    lm_head_T = ref.w("lm_head.weight").T.copy()
    cli = C.Client(a.host, a.port, io_timeout=3600.0)

    def restart_worker():
        import subprocess

        serial = os.environ.get("DEVICE_SERIAL", "239dbd8f")
        subprocess.run(
            [
                "adb",
                "-s",
                serial,
                "shell",
                "kill $(pidof onnx-remote-hexagon-worker) 2>/dev/null; true",
            ],
            check=False,
        )
        time.sleep(1)
        subprocess.Popen(
            [
                "adb",
                "-s",
                serial,
                "shell",
                f"cd {os.path.dirname(a.worker_binary_on_phone)} && exec ./onnx-remote-hexagon-worker --port {a.port} --cache-dir {os.path.dirname(a.worker_binary_on_phone)}/cache",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        for _ in range(120):
            try:
                cli.capabilities()
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("worker did not come back after restart")

    if not a.fresh_worker:
        for _, _, name, blob in chunks:
            t0 = time.time()
            cli.load_compiled(name, blob)
            print("loaded", name, f"{time.time() - t0:.1f}s", flush=True)

    def run(name, blob, inputs):
        if a.fresh_worker:
            restart_worker()
            cli.load_compiled(name, blob)
            return cli.run_compiled(name, inputs)
        try:
            return cli.run_compiled(name, inputs)
        except C.RemoteError as e:
            print("  reload", name, "after:", str(e)[:120], flush=True)
            cli.load_compiled(name, blob)
            return cli.run_compiled(name, inputs)

    ids = [int(x) for x in a.ids.split(",")]
    kc = {i: np.zeros((nkv, C_, hd), np.float32) for i in range(ref.L)}
    vc = {i: np.zeros((nkv, C_, hd), np.float32) for i in range(ref.L)}
    seq = list(ids)
    worst = 0.0
    h_ref = emb[ids[0] : ids[0] + 1].astype(np.float32).copy()
    for pos in range(len(ids) + a.steps):
        tok = seq[pos]  # prompt tokens first, then the phone's greedy tokens
        hidden = emb[tok : tok + 1].astype(np.float32).copy()
        mask = np.full((1, C_ + 1), -1e9, np.float32)
        mask[0, :pos] = 0
        mask[0, C_] = 0
        t0 = time.time()
        for lo, hi, name, blob in chunks:
            inputs = [hidden, np.array([pos], np.int64), mask]
            for i in range(lo, hi + 1):
                inputs += [kc[i], vc[i]]
            outs = run(name, blob, inputs)
            hidden = outs[0].reshape(1, H).astype(np.float32)
            if a.trace and pos == 0:
                for i in range(lo, hi + 1):
                    h_ref = ref.layer(i, h_ref)
                d = float(
                    np.max(np.abs(hidden - h_ref)) / (np.max(np.abs(h_ref)) + 1e-9)
                )
                print(
                    f"trace after layer {hi}: rel-max-diff {d:.3e} (ref max {np.max(np.abs(h_ref)):.2f})",
                    flush=True,
                )
            for j, i in enumerate(range(lo, hi + 1)):
                kc[i][:, pos] = outs[1 + 2 * j].reshape(nkv, hd)
                vc[i][:, pos] = outs[2 + 2 * j].reshape(nkv, hd)
        # final norm + LM head on the host: its 1.1 GB program is over the 512 MB wire limit
        logits = ref.rms(hidden, final_gain, 1.0) @ lm_head_T
        logits = logits.reshape(-1)
        dt = time.time() - t0
        if pos + 1 >= len(ids):
            ref_logits = ref.logits(seq[: pos + 1])[-1]
            diff = float(
                np.max(np.abs(logits - ref_logits))
                / (np.max(np.abs(ref_logits)) + 1e-9)
            )
            worst = max(worst, diff)
            nxt = int(np.argmax(logits))
            print(
                f"pos {pos}: argmax {nxt} (ref {int(np.argmax(ref_logits))}) rel-max-diff {diff:.3e} step {dt:.1f}s",
                flush=True,
            )
            seq.append(nxt)
    print("worst rel diff", f"{worst:.3e}", "tokens", seq, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
