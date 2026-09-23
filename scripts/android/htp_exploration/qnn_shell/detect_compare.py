"""Detection-level check: HTP backbone outputs -> rest.onnx (host ORT) vs. all-ORT.

Same matching as scripts/android/maskrcnn_e2e/README.md (score > 0.5, label + box IoU > 0.5).
usage: python detect_compare.py backbone.onnx rest.onnx input.bin out_htp
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maskrcnn_e2e"))
from eval_common import compare  # noqa: E402


def main(backbone: str, rest: str, input_bin: str, prefix: str) -> None:
    bb = ort.InferenceSession(backbone, providers=["CPUExecutionProvider"])
    x = np.fromfile(input_bin, np.float32).reshape(3, 800, 1088)
    ref_feats = bb.run(None, {bb.get_inputs()[0].name: x})
    names = [o.name for o in bb.get_outputs()]
    htp_feats = [
        np.fromfile(f"{prefix}_o{i}.bin", np.float32).reshape(r.shape)
        for i, r in enumerate(ref_feats)
    ]
    rs = ort.InferenceSession(rest, providers=["CPUExecutionProvider"])
    ref = rs.run(None, dict(zip(names, ref_feats)))
    htp = rs.run(None, dict(zip(names, htp_feats)))
    print(compare(ref, htp))


if __name__ == "__main__":
    main(*sys.argv[1:5])
