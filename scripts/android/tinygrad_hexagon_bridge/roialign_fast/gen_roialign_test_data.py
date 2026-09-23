#!/usr/bin/env python3
"""Turn dump_real_roialign_io.py's captured real RoiAlign inputs/outputs (one COCO image through the
full MaskRCNN-12-qdq model in ONNX Runtime) into flat binaries for roialign_host_check.c,
roialign_qemu.c and roialign_client.c: per call `callN_feat.bin` (channels-last H*W*C fp32),
`callN_rois.bin` (R*4 fp32), `callN_ref.bin` (ORT's output, transposed to R*OH*OW*C), plus
`calls.txt` with one `H W C R OH OW sr scale` line per call."""
import json
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
out.mkdir(parents=True, exist_ok=True)
d = np.load(src / "roi_real.npz")
meta = json.load(open(src / "roi_meta.json"))
lines = []
for i, n in enumerate(meta["nodes"]):
    feat, rois, bidx = (d[k] for k in n["inputs"])
    ref = d[n["output"]]
    assert (bidx == 0).all() and feat.shape[0] == 1
    _, C, H, W = feat.shape
    a = n["attrs"]
    np.ascontiguousarray(feat[0].transpose(1, 2, 0), dtype=np.float32).tofile(out / f"call{i}_feat.bin")
    np.ascontiguousarray(rois, dtype=np.float32).tofile(out / f"call{i}_rois.bin")
    np.ascontiguousarray(ref.transpose(0, 2, 3, 1), dtype=np.float32).tofile(out / f"call{i}_ref.bin")
    lines.append(f"{H} {W} {C} {rois.shape[0]} {a['output_height']} {a['output_width']} {a['sampling_ratio']} {a['spatial_scale']!r}")
(out / "calls.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
