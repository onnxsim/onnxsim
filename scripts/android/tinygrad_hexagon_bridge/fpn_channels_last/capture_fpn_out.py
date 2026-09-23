#!/usr/bin/env python3
"""Capture the real inputs, parameters and outputs of the four FPN output convs (the 3x3 convs on
P5/P4/P3/P2 whose dequantized outputs `391`/`423`/`455`/`487` are the feature maps every RoiAlign
node reads), by running `backbone.onnx` (from ../../maskrcnn_e2e/prepare.py) in ONNX Runtime on
one real image.

Per level it saves: the conv's uint8 input (NCHW), int8 weight (OIHW), int32 bias, the scales /
zero points, ORT's uint8 conv output (`*_quantized`, what QuantizeLinear produced) and ORT's fp32
output (the backbone output RoiAlign reads). It then checks that an integer-only reimplementation
of the conv + requantize + dequantize epilogue (exact int32 accumulation with the input zero point
folded into the bias, then ORT/MLAS-style `round_half_even(acc * (s_in*s_w/s_out)) + zp`) matches
ORT bit-for-bit -- the semantics hex_conv3x3_fpnout_kernel.py implements on the DSP.

    python capture_fpn_out.py --backbone backbone.onnx --image 000000000139.jpg --out DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maskrcnn_e2e"))
from eval_common import canvas  # noqa: E402

# level -> (conv uint8 input, weight, bias, fp32 output, output scale, output zero point,
#           input scale, input zero point); traced from backbone.onnx (see ../README.md).
LEVELS = {
    "P5": ("388_quantized", "389", "390", "391", "391_scale", "419_zero_point", "388_scale", "57_zero_point"),
    "P4": ("420_422_QuantizeLinear", "421", "422", "423", "423_scale", "151_zero_point", "420_scale", "107_zero_point"),
    "P3": ("452_454_QuantizeLinear", "453", "454", "455", "455_scale", "509_zero_point", "452_scale", "452_zero_point"),
    "P2": ("484_486_QuantizeLinear", "485", "486", "487", "487_scale", "383_zero_point", "484_scale", "107_zero_point"),
}


def requant_ort(acc: np.ndarray, m: np.float32, zp: int) -> np.ndarray:
    x = acc.astype(np.float32) * m  # float32 multiply, like MLAS's requantize
    return np.clip(np.rint(x) + zp, 0, 255).astype(np.uint8)  # np.rint = round half to even


def conv_int(a_chw_u8: np.ndarray, zp_in: int, w: np.ndarray) -> np.ndarray:
    """Exact int32 3x3/pad1 conv on raw uint8 (border padded with the input zero point), no zp
    correction: returns (H, W, Cout) int64."""
    c, h, wd = a_chw_u8.shape
    a = np.pad(a_chw_u8.transpose(1, 2, 0).astype(np.int64), ((1, 1), (1, 1), (0, 0)), constant_values=zp_in)
    out = np.zeros((h, wd, w.shape[0]), np.int64)
    for kh in range(3):
        for kw in range(3):
            out += a[kh:kh + h, kw:kw + wd] @ w[:, :, kh, kw].astype(np.int64).T
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--check-against", type=Path, help="roi_real.npz from ../roialign_fast/dump_real_roialign_io.py")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = onnx.load(args.backbone)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    have = {o.name for o in m.graph.output}
    want = []
    for inp, _, _, out, *_ in LEVELS.values():
        want += [inp, out + "_quantized", out]
    for nm in want:
        if nm not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    img = canvas(Path(args.image), 800, 1088)
    vals = dict(zip(want, sess.run(want, {sess.get_inputs()[0].name: img})))
    ref_npz = np.load(args.check_against) if args.check_against else None

    save = {}
    for lvl, (inp, w, b, out, osc, ozp, isc, izp) in LEVELS.items():
        a = vals[inp][0]
        wq, bq = init[w + "_quantized"], init[b + "_quantized"]
        s_in, s_w, s_out = np.float32(init[isc]), np.float32(init[w + "_scale"]), np.float32(init[osc])
        zp_in, zp_out = int(init[izp]), int(init[ozp])
        assert int(init["ConvMulFusion_W_2_zero_point"]) == 0
        assert np.isclose(np.float32(init[b + "_scale"]).ravel()[0], s_in * s_w, rtol=1e-6)
        mult = np.float32(np.float32(s_in * s_w) / s_out)
        # fold the input zero point: sum (a - zp) w = sum a w - zp * sum w
        bias_fold = bq.astype(np.int64) - zp_in * wq.astype(np.int64).sum(axis=(1, 2, 3))
        acc = conv_int(a, zp_in, wq) + bias_fold  # (H, W, C)
        q = requant_ort(acc, mult, zp_out)
        ort_q = vals[out + "_quantized"][0].transpose(1, 2, 0)
        deq = ((q.astype(np.int32) - zp_out).astype(np.float32) * s_out)
        ort_f = vals[out][0].transpose(1, 2, 0)
        nq = int((q != ort_q).sum())
        print(f"{lvl}: in {a.shape} zp_in={zp_in} out zp={zp_out} mult={mult!r}  uint8 mismatches vs ORT: "
              f"{nq}/{q.size} (max |d|={int(np.abs(q.astype(int) - ort_q.astype(int)).max())}); "
              f"fp32 bit-exact vs ORT: {np.array_equal(deq, ort_f)}")
        if ref_npz is not None:
            print(f"    == roi_real.npz[{out}] (the RoiAlign input): {np.array_equal(vals[out], ref_npz[out])}")
        save.update({f"{lvl}_in": a, f"{lvl}_w": wq, f"{lvl}_bias": bias_fold.astype(np.int32),
                     f"{lvl}_ort_q": vals[out + "_quantized"][0], f"{lvl}_ort_f": vals[out][0],
                     f"{lvl}_scalars": np.array([mult, s_out], np.float32),
                     f"{lvl}_zps": np.array([zp_in, zp_out], np.int32)})
    np.savez(args.out / "fpn_out_real.npz", **save)
    print("wrote", args.out / "fpn_out_real.npz")


if __name__ == "__main__":
    main()
