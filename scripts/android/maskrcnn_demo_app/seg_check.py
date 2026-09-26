"""Host check of the YOLO-seg decode the app does (native/yolo_engine.cpp, here in numpy):
int8 HTP graph vs fp32 on the deploy spec's 20 eval images (matched detections, mask IoU), and with
--ultralytics the fp32 decode vs Ultralytics' own predict masks (needs ultralytics + the .pt weights).

    seg_check.py <deploy work dir> <yolo26n-seg|yolo11n-seg> [--ultralytics]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

name = sys.argv[2]
W = Path(sys.argv[1]) / name
IMGS = Path(sys.argv[1]) / "_images"  # the fetch stage's COCO images
EVAL = [
    7088,
    7108,
    7278,
    7281,
    7386,
    7511,
    7574,
    7784,
    7795,
    7816,
    7818,
    7888,
    7977,
    7991,
    8021,
    8211,
    8277,
    8532,
    8629,
    8690,
]
e2e = name.startswith("yolo26")


def letterbox(img):
    h, w = img.shape[:2]
    r = min(640 / h, 640 / w)
    fw, fh = round(w * r), round(h * r)
    out = np.full((640, 640, 3), 114, np.uint8)
    left, top = (640 - fw) // 2, (640 - fh) // 2
    out[top : top + fh, left : left + fw] = cv2.resize(
        img, (fw, fh), interpolation=cv2.INTER_LINEAR
    )
    return out


def decode(o0, protos, conf=0.25, iou_t=0.7, maxdet=300):
    """the app's post: boxes (x1,y1,x2,y2 in 640 px), class, score, 32 coefs -> per-det 160x160 masks cropped to the box"""
    h = o0[0]  # (116, N)
    nc = h.shape[0] - 4 - 32
    sc, cf = h[4 : 4 + nc], h[4 + nc :]
    dets = []
    if e2e:
        best = sc.max(0)
        k = min(maxdet, best.size)
        idx = np.argsort(-best, kind="stable")[:k]
        cand = sc[:, idx].T  # (k, nc)
        order = np.argsort(-cand.ravel(), kind="stable")[:k]
        for j in order:
            a, c = idx[j // nc], j % nc
            s = cand.ravel()[j]
            if s < conf:
                break
            dets.append((h[0, a], h[1, a], h[2, a], h[3, a], s, c, cf[:, a]))
    else:
        cands = []
        for c in range(nc):
            for a in np.nonzero(sc[c] > conf)[0]:
                cx, cy, w, hh = h[:4, a]
                cands.append(
                    (
                        cx - w / 2,
                        cy - hh / 2,
                        cx + w / 2,
                        cy + hh / 2,
                        sc[c, a],
                        c,
                        cf[:, a],
                    )
                )
        cands.sort(key=lambda d: -d[4])
        for d in cands:
            if all(k[5] != d[5] or box_iou(k, d) <= iou_t for k in dets):
                dets.append(d)
            if len(dets) >= maxdet:
                break
    P = protos[0].reshape(32, -1)
    out = []
    for d in dets:
        m = 1 / (1 + np.exp(-(d[6] @ P).reshape(160, 160)))
        x1, y1, x2, y2 = [v / 4 for v in d[:4]]
        yy, xx = np.mgrid[0:160, 0:160] + 0.5
        m = m * ((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2))
        out.append((d[:6], m > 0.5))
    return out


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


f32 = ort.InferenceSession(
    str(W / "simplify/model.onnx"), providers=["CPUExecutionProvider"]
)
i8 = ort.InferenceSession(
    str(W / f"pipe/{name}.onnx"), providers=["CPUExecutionProvider"]
)
ious, matched, total = [], 0, 0
for i in EVAL:
    img = cv2.cvtColor(cv2.imread(str(IMGS / f"coco_{i:012d}.jpg")), cv2.COLOR_BGR2RGB)
    lb = letterbox(img)
    a = decode(
        *f32.run(
            None, {"images": (lb.transpose(2, 0, 1)[None] / 255).astype(np.float32)}
        )
    )
    b = decode(*i8.run(None, {"images_u8": lb[None]}))
    for da, ma in a:
        total += 1
        best = max(
            ((box_iou(da, db), mb) for db, mb in b if db[5] == da[5]),
            key=lambda t: t[0],
            default=(0, None),
        )
        if best[0] > 0.5:
            matched += 1
            inter, union = (ma & best[1]).sum(), (ma | best[1]).sum()
            ious.append(inter / max(union, 1))
print(
    f"{name}: int8 vs fp32 on {len(EVAL)} eval images: {matched}/{total} fp32 detections matched (class, box IoU > 0.5); "
    f"mask IoU mean {np.mean(ious):.3f}, median {np.median(ious):.3f}, 10th pct {np.percentile(ious, 10):.3f}"
)

# decode check: fp32 ONNX + this decode vs Ultralytics predict on the same letterboxed image (first eval image)
if "--ultralytics" in sys.argv:
    from ultralytics import YOLO

    img = cv2.cvtColor(
        cv2.imread(str(IMGS / f"coco_{EVAL[0]:012d}.jpg")), cv2.COLOR_BGR2RGB
    )
    lb = letterbox(img)
    mine = decode(
        *f32.run(
            None, {"images": (lb.transpose(2, 0, 1)[None] / 255).astype(np.float32)}
        )
    )
    r = YOLO(str(Path.home() / f".cache/onnxsim-deploy/_weights/{name}.pt"))(
        lb, imgsz=640, conf=0.25, verbose=False
    )[0]
    um = r.masks.data.cpu().numpy() > 0.5  # (n, 640, 640) at the input size
    ub = r.boxes.xyxy.cpu().numpy()
    print(f"decode check: mine {len(mine)} dets, ultralytics {len(ub)}")
    for d, m in mine[:8]:
        j = int(np.argmax([box_iou(d, u) for u in ub])) if len(ub) else -1
        mu = (
            cv2.resize(um[j].astype(np.uint8), (160, 160), interpolation=cv2.INTER_AREA)
            > 0
            if j >= 0
            else None
        )
        mi = (m & mu).sum() / max((m | mu).sum(), 1) if j >= 0 else 0
        print(
            f"  cls {d[5]} score {d[4]:.2f} box IoU vs ultralytics {box_iou(d, ub[j]):.3f}, mask IoU {mi:.3f}"
        )
