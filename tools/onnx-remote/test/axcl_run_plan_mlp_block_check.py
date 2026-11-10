"""On-card check of PLaMo layer 0's MLP block as two run_plan requests, with the
host product between them:

  plan A: pre-MLP RMS (rms2560, real gain) -> gate and up linears (k2560n10240,
          real weights; up is W_up/4) -> silu(gate) with up = 0.25 (silu_q)
  host:   unit RMS of silu(gate) and of up (the device crms10240 template clips
          its output grid, so the unit norms stay on the host), then a = u_gate * u_up
  plan B: split-K down projection (four k2560n2560 partials + three Adds)

Every device stage is compared with the float reference of the PLaMo layer
(the reference .npy files are produced with the emission scripts in the
PLaMo decomposition work; see the README). The device chain is not compared
with a host chain here: the split check already covers that.

Environment:
  ONNX_REMOTE_HOST / ONNX_REMOTE_PORT   the worker (default 127.0.0.1:39501)
  ONNX_REMOTE_MLP_DIR                   directory on the worker that holds the
                                        models: mlp_rms_real, mlp_gate, mlp_up_q,
                                        mlp_silu_q (.axmodel), plamo_l0_part0..3 and
                                        plamo_l0_add1..3 (.axmodel)
  ONNX_REMOTE_MLP_REF_DIR               host directory with mlp_hm.npy [>=1, 2560]
                                        (the block input), mlp_gain.npy [2560] (the
                                        RMS gain, 1 + w), mlp_ref_gate.npy,
                                        mlp_ref_up.npy, mlp_ref_down.npy (float
                                        references, [*, 10240] / [*, 2560])
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)
import onnx_remote_client as C  # noqa: E402

HOST = os.environ.get("ONNX_REMOTE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ONNX_REMOTE_PORT", "39501"))
MODELS = os.environ["ONNX_REMOTE_MLP_DIR"].rstrip("/") + "/"
REF = os.environ["ONNX_REMOTE_MLP_REF_DIR"].rstrip("/") + "/"

H, FF, K = 2560, 10240, 2560
HB, FB, KB = 4 * H, 4 * FF, 4 * K  # float32 bytes of one row (hidden, FFN, K)

# measured on the card: 1.2%, 1.5%, 1.1%, 1.5%, 0.27%, 0.86%; limits are about 2x
LIMITS = {
    "gate": 0.03,
    "up": 0.03,
    "unit(silu gate)": 0.03,
    "unit(up)": 0.03,
    "product": 0.01,
    "down": 0.02,
}


def unit(x):
    return x / np.sqrt(np.mean(x**2, -1, keepdims=True) + 1e-12)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def plan_a():
    m = MODELS
    return (
        "\n".join(
            [
                f"model 1 {m}mlp_rms_real.axmodel",
                f"model 2 {m}mlp_gate.axmodel",
                f"model 3 {m}mlp_up_q.axmodel",
                f"model 4 {m}mlp_silu_q.axmodel",
                f"buffer 1 {HB}",
                f"buffer 2 {HB}",
                f"buffer 3 {HB}",
                *[f"buffer {b} {FB}" for b in range(4, 12)],
                f"write 1 0 {HB} 0",
                f"write 2 0 {HB} 1",
                f"write 11 0 {FB} 2",
                "bind 1 I 0 1 0",
                "bind 1 I 1 2 0",
                "bind 1 O 0 3 0",
                "exec 1",
                "bind 2 I 0 3 0",
                "bind 2 O 0 4 0",
                "exec 2",
                "bind 3 I 0 3 0",
                "bind 3 O 0 5 0",
                "exec 3",
                "bind 4 I 0 4 0",
                "bind 4 I 1 11 0",
                "bind 4 O 0 6 0",
                "exec 4",
                f"read 4 0 {FB}",
                f"read 5 0 {FB}",
                f"read 6 0 {FB}",
                "free",
            ]
        )
        + "\n"
    )


def plan_b():
    m = MODELS
    lines = [f"model {i} {m}plamo_l0_part{i - 1}.axmodel" for i in range(1, 5)]
    lines += [f"model {4 + k} {m}plamo_l0_add{k}.axmodel" for k in (1, 2, 3)]
    lines += [f"buffer 1 {FB}"]
    lines += [f"buffer {10 + i} {KB}" for i in range(4)]
    lines += [f"buffer {20 + i} {KB}" for i in range(4)]
    lines += [f"buffer {30 + i} {KB}" for i in range(3)]
    lines.append(f"write 1 0 {FB} 0")
    for i in range(4):
        lines += [
            f"copy {10 + i} 0 1 {i * KB} {KB}",
            f"bind {i + 1} I 0 {10 + i} 0",
            f"bind {i + 1} O 0 {20 + i} 0",
            f"exec {i + 1}",
        ]
    lines += [
        "bind 5 I 0 20 0",
        "bind 5 I 1 21 0",
        "bind 5 O 0 30 0",
        "exec 5",
        "bind 6 I 0 30 0",
        "bind 6 I 1 22 0",
        "bind 6 O 0 31 0",
        "exec 6",
        "bind 7 I 0 31 0",
        "bind 7 I 1 23 0",
        "bind 7 O 0 32 0",
        "exec 7",
        f"read 32 0 {KB}",
        "free",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    c = C.Client(HOST, PORT)
    hm = np.load(REF + "mlp_hm.npy")[:1].astype(np.float32)
    gain = np.load(REF + "mlp_gain.npy").astype(np.float32)
    ref_gate = np.load(REF + "mlp_ref_gate.npy").astype(np.float64)
    ref_up = np.load(REF + "mlp_ref_up.npy").astype(np.float64)
    ref_down = np.load(REF + "mlp_ref_down.npy").astype(np.float64)
    quarter = np.full((1, FF), 0.25, np.float32)  # silu_q: up = 0.25

    ra = c.run_plan(plan_a(), [hm, gain, quarter])
    gate = ra.outputs[0].view(np.float32).reshape(1, FF).astype(np.float64)
    up = ra.outputs[1].view(np.float32).reshape(1, FF).astype(np.float64) * 4
    sg4 = ra.outputs[2].view(np.float32).reshape(1, FF).astype(np.float64)
    # scale-invariant: sg4 (silu of gate with up = 0.25) and the quartered up
    # give the same unit norms as the unquartered values
    ug, uu = unit(sg4), unit(up)
    ref_sg = ref_gate / (1 + np.exp(-ref_gate))
    ref_a = unit(ref_sg) * unit(ref_up)
    a = (ug * uu).astype(np.float32)

    rb = c.run_plan(plan_b(), [a])
    down = rb.outputs[0].view(np.float32).reshape(1, K).astype(np.float64)

    errors = {
        "gate": rel(gate, ref_gate),
        "up": rel(up, ref_up),
        "unit(silu gate)": rel(ug, unit(ref_sg)),
        "unit(up)": rel(uu, unit(ref_up)),
        "product": rel(a.astype(np.float64), ref_a),
        "down": rel(down, ref_down),
    }
    failed = []
    for name, err in errors.items():
        ok = err <= LIMITS[name]
        print(f"{name:18s} {err:.4f} (limit {LIMITS[name]}) {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    print("MLP block matches the float reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
