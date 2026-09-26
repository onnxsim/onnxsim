#!/usr/bin/env python3
"""Pinned YOLO26 export for the `fetch` stage (`fetch: {script: export_yolo26.py, args: [...]}`).

    export_yolo26.py <out.onnx> <weights.pt name> <sha256> [head|end2end]

Also exports the -seg models (yolo26n-seg.pt, yolo11n-seg.pt): output0 then carries 32 mask
coefficients after the class scores, and output1 is the (1, 32, 160, 160) prototypes.

Downloads the official Ultralytics weights (github.com/ultralytics/assets, release v8.4.0),
checks the sha256, and exports ONNX (opset 17, 640x640, static, not simplified -- the `simplify`
stage runs onnxsim):

  head      (default) the one-to-one head's decoded output, (1, 4 + 80, 8400): x1,y1,x2,y2 in
            input pixels + sigmoid class scores. The NMS-free top-k selection that Ultralytics
            appends is left out and rebuilt on the CPU by `postprocess: {kind: yolo_end2end}`
            (stages/post.py), so the HTP graph is plain conv net + head.
  end2end   Ultralytics' own end-to-end export, (1, 300, 6): x1,y1,x2,y2,score,class after TopK
            and GatherElements (for measuring TopK on the HTP; `postprocess: {kind: none}`).

Needs `ultralytics` (+ torch). If the deploy python lacks it, point ULTRALYTICS_PYTHON at an
interpreter that has it (e.g. a venv) and this script re-executes itself there.
Ultralytics YOLO26 weights are AGPL-3.0 (https://ultralytics.com/license), as YOLO11's are.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"


def main() -> None:
    out, weights, sha = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "head"
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        py = os.environ.get("ULTRALYTICS_PYTHON")
        if not py:
            raise SystemExit("export_yolo26: needs ultralytics; set ULTRALYTICS_PYTHON to a python that has it")
        os.execv(py, [py, __file__, *sys.argv[1:]])
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "onnxsim-deploy" / "_weights"
    cache.mkdir(parents=True, exist_ok=True)
    pt = cache / weights
    if not pt.exists() or hashlib.sha256(pt.read_bytes()).hexdigest() != sha:
        urllib.request.urlretrieve(URL.format(weights), pt)
    got = hashlib.sha256(pt.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"export_yolo26: {weights} sha256 {got} != {sha}")
    if mode == "head":
        # Detect.forward: y = _inference(one2one preds) -> (B, 84, N); end2end then calls
        # postprocess(y.permute(0, 2, 1)) for the top-k. Undo the permute instead.
        Detect.postprocess = lambda self, p: p.permute(0, 2, 1)
    elif mode != "end2end":
        raise SystemExit(f"export_yolo26: unknown mode {mode}")
    with tempfile.TemporaryDirectory(dir=cache) as td:  # not /tmp (tmpfs on the dev box)
        m = YOLO(str(shutil.copy(pt, Path(td) / weights)))
        f = m.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, nms=False)
        shutil.move(f, out)
    print(f"  exported {weights} ({mode}) -> {out}")


if __name__ == "__main__":
    main()
