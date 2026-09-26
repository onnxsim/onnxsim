"""Detection agreement of YOLO head outputs against a reference (fp32 ONNX Runtime by default).

  python compare_yolo.py --ref ort --onnx yolo11n.onnx --inputs eval/*.bin --run tg:out_tg/out0_{i}.bin --run qnn:out_qnn/{stem}.bin

Each input is a uint8 NHWC 640x640 letterboxed image (.bin). Every run's (1, 84, 8400) float head is decoded the way
the app's YOLO engine does: post=nms (YOLO11: cx,cy,w,h + class scores, per-class greedy NMS iou 0.7, conf 0.25, max
300) or post=end2end (YOLO26: x1,y1,x2,y2, best class per anchor, top 300 above conf; the app's two-stage top-k)
and matched to the reference detections (same class, IoU >= 0.5, greedy by score). Prints per run: matched / reference
boxes, extra boxes, and the max |score| difference over matched pairs.
"""
import argparse
from pathlib import Path
import numpy as np

def decode(head, rows, post="nms", conf=0.25, iou=0.7, max_det=300):
  head = head.reshape(rows, -1)
  boxes, scores = head[:4].T, head[4:84].T  # rows 84.. are the -seg models' mask coefficients
  if post == "end2end":  # YOLO26's one-to-one head: x1,y1,x2,y2, no NMS (best class per anchor, top max_det)
    cls, sc = scores.argmax(1), scores.max(1)
    order = np.argsort(-sc)[:max_det]
    return [(boxes[i], sc[i], cls[i]) for i in order if sc[i] > conf]
  cls, sc = scores.argmax(1), scores.max(1)
  keep = sc > conf
  b, s, c = boxes[keep], sc[keep], cls[keep]
  xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
  out = []
  for k in np.unique(c):
    idx = np.where(c == k)[0][np.argsort(-s[c == k])]
    while len(idx):
      i = idx[0]; out.append((xyxy[i], s[i], k))
      idx = idx[1:][box_iou(xyxy[i], xyxy[idx[1:]]) <= iou]
  out.sort(key=lambda d: -d[1])
  return out[:max_det]

def box_iou(a, b):
  b = np.atleast_2d(b)
  lt, rb = np.maximum(a[:2], b[:, :2]), np.minimum(a[2:], b[:, 2:])
  inter = np.prod(np.clip(rb - lt, 0, None), 1)
  area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
  return inter / (area(a) + area(b) - inter + 1e-9)

def match(ref, det):
  used, m, dsc = set(), 0, []
  for rb, rs, rc in ref:
    best, bj = 0.5, -1
    for j, (db, ds, dc) in enumerate(det):
      if j in used or dc != rc: continue
      v = box_iou(rb, db[None])[0]
      if v >= best: best, bj = v, j
    if bj >= 0: used.add(bj); m += 1; dsc.append(abs(det[bj][1] - rs))
  return m, len(det) - len(used), max(dsc, default=0.0)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True, help="fp32 model, uint8 NHWC input is converted to /255 NCHW for it")
  ap.add_argument("--inputs", nargs="+", required=True)
  ap.add_argument("--post", default="nms", choices=["nms", "end2end"])
  ap.add_argument("--rows", type=int, default=84, help="head rows: 84, or 116 for the -seg models")
  ap.add_argument("--run", action="append", default=[], help="name:path pattern, {i} = input index, {stem} = input stem")
  args = ap.parse_args()
  import onnxruntime as ort
  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  runs = [r.split(":", 1) for r in args.run]
  tot = {n: [0, 0, 0.0] for n, _ in runs}
  nref = 0
  for i, p in enumerate(args.inputs):
    x = np.fromfile(p, np.uint8).reshape(1, 640, 640, 3).astype(np.float32).transpose(0, 3, 1, 2) / 255
    ref = decode(sess.run(None, {sess.get_inputs()[0].name: x})[0], args.rows, args.post)
    nref += len(ref)
    for n, pat in runs:
      det = decode(np.fromfile(pat.format(i=i, stem=Path(p).stem), np.float32, count=args.rows * 8400), args.rows, args.post)
      m, extra, ds = match(ref, det)
      tot[n][0] += m; tot[n][1] += extra; tot[n][2] = max(tot[n][2], ds)
  print(f"{len(args.inputs)} images, {nref} reference (fp32 ORT) detections")
  for n, (m, extra, ds) in tot.items():
    print(f"  {n:10s} matched {m}/{nref}, extra {extra}, max |score diff| on matches {ds:.3f}")

if __name__ == "__main__": main()
