"""Exact integer reference of a QDQ conv layer as ORT CPU computes it (QLinearConv after its QDQ fusion):
acc = sum((xq - zx) * wq) + bq (int32), y = clamp(rne(fp32(acc) * fp32(sx * sw[c] / sy)) + zy, lo, 255)
with lo = zy under a fused Relu. Checked against ORT's own output by qdq_layer.py's caller."""

import numpy as np


def conv_acc(xq, wq, bq, zx, k, stride):
    """xq uint8 [1, C, H, W], wq int8 [O, C, k, k] -> int64 acc [O, Ho, Wo] (zero padding in the real domain =
    zero-point padding in the quantized one)"""
    _, c, h, w = xq.shape
    p = k // 2
    x = np.pad(xq[0].astype(np.int64) - int(zx), ((0, 0), (p, p), (p, p)))
    ho, wo = (h + 2 * p - k) // stride + 1, (w + 2 * p - k) // stride + 1
    acc = np.zeros((wq.shape[0], ho, wo), np.int64)
    for dy in range(k):
        for dx in range(k):
            patch = x[
                :, dy : dy + stride * ho : stride, dx : dx + stride * wo : stride
            ].reshape(c, -1)
            acc += (wq[:, :, dy, dx].astype(np.int64) @ patch).reshape(-1, ho, wo)
    return acc + bq.astype(np.int64)[:, None, None]


def requant(acc, sx, sw, sy, zy, relu):
    m = (np.float32(sx) * sw.astype(np.float32) / np.float32(sy)).astype(np.float32)
    v = acc.astype(np.float32) * m[:, None, None]
    y = np.rint(v).astype(np.int64) + int(zy)
    return np.clip(y, int(zy) if relu else 0, 255).astype(np.uint8)
