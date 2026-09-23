#!/usr/bin/env python3
"""Turn capture_fpn_out.py's fpn_out_real.npz into the flat binaries fpn_client.c reads, per FPN
level k (0..3 = P5, P4, P3, P2): `lvlK_chw.bin` (ORT's real NCHW fp32 map -- what RoiAlign reads
today), `lvlK_apad.bin` (the output conv's uint8 input, channels-last, padded with its zero point),
`lvlK_wp.bin` (pack_weight_3x3()), `lvlK_bias.bin` (int32, zero point folded), plus `levels.txt`.

    python gen_fpn_test_data.py fpn_out_real.npz OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hex_conv3x3_fpnout_kernel import pad_input_zp  # noqa: E402
from hex_conv3x3_kernel import pack_weight_3x3  # noqa: E402

d = np.load(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lines = []
for k, lvl in enumerate(["P5", "P4", "P3", "P2"]):
    a = d[f"{lvl}_in"]
    zp_in = int(d[f"{lvl}_zps"][0])
    np.ascontiguousarray(d[f"{lvl}_ort_f"], dtype=np.float32).tofile(out / f"lvl{k}_chw.bin")
    np.ascontiguousarray(pad_input_zp(a.transpose(1, 2, 0), zp_in), dtype=np.uint8).tofile(out / f"lvl{k}_apad.bin")
    np.ascontiguousarray(pack_weight_3x3(d[f"{lvl}_w"]), dtype=np.uint8).tofile(out / f"lvl{k}_wp.bin")
    np.ascontiguousarray(d[f"{lvl}_bias"], dtype=np.int32).tofile(out / f"lvl{k}_bias.bin")
    lines.append(f"{a.shape[1]} {a.shape[2]}")
(out / "levels.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
