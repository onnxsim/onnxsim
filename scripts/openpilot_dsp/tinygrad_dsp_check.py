"""tinygrad DEV=DSP (MOCKDSP=1: qemu-hexagon, hexagonv65 code) vs ORT on real DM frames, fp32 model."""

import sys
import time
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner

d = np.load("./inputs_seg8.npz")
ref = np.load("./ref_dm_fp32_s8.npy")
r = OnnxRunner(Path(sys.argv[1]))
calib = np.array([[-0.00028, 0.16415, 0.00528]], np.float32)
for i in (0, 300):
    t = time.time()
    out = r({"input_img": Tensor(d["driver"][i]), "calib": Tensor(calib)})[
        "outputs"
    ].numpy()[0]
    print(
        i,
        "max|diff| %.2e" % np.abs(out - ref[i]).max(),
        "ref absmax %.2f" % np.abs(ref[i]).max(),
        "%.1fs" % (time.time() - t),
        flush=True,
    )
