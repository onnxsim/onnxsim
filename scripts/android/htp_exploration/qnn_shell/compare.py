"""Compare qnn_run's pulled outputs (out_<mode>_o<i>.bin) against host ONNX Runtime CPU.

usage: python compare.py backbone.onnx input.bin out_htp   (after adb-pulling out_htp_o*.bin)
"""

import sys

import numpy as np
import onnxruntime as ort


def main(model: str, input_bin: str, prefix: str) -> None:
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.fromfile(input_bin, np.float32).reshape(
        [d if isinstance(d, int) else 1 for d in inp.shape]
    )
    refs = sess.run(None, {inp.name: x})
    for i, (o, r) in enumerate(zip(sess.get_outputs(), refs)):
        a = np.fromfile(f"{prefix}_o{i}.bin", np.float32).reshape(r.shape)
        d = np.abs(a - r)
        print(
            f"o{i:<2} {o.name[:22]:22} {str(r.shape):20} maxabs={d.max():.4g} meanabs={d.mean():.3g} "
            f"exact={(d < 1e-6).mean() * 100:.1f}% range=[{r.min():.3g},{r.max():.3g}]"
        )


if __name__ == "__main__":
    main(*sys.argv[1:4])
