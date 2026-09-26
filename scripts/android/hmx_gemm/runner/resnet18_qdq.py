#!/usr/bin/env python3
"""ResNet-18 backbone (torchvision weights, 224x224, stem .. layer4) -> fp32 ONNX -> onnxsim full_qdq (uint8
activations, per-channel int8 weights, uint8 graph I/O NHWC) + a calibration/eval input and ORT CPU's output.

usage: resnet18_qdq.py <resnet18-f37072fd.pth> <coco images dir> <outdir>
writes backbone_fp32.onnx, backbone_qdq.onnx (input "x" uint8 [1,224,224,3], output "y" uint8 [1,512,7,7]),
io.json (scales / zero points), input.bin (uint8 NHWC of the first eval image), ref.bin (ORT CPU output)."""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + idt)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for i, (a, b, s) in enumerate(cfg):
            setattr(
                self, f"layer{i + 1}", nn.Sequential(Block(a, b, s), Block(b, b, 1))
            )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


MEAN, STD = (
    np.array([0.485, 0.456, 0.406], np.float32),
    np.array([0.229, 0.224, 0.225], np.float32),
)


def image(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    im = im.crop(
        ((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224)
    )
    a = (np.asarray(im, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)[None].copy()


def main():
    pth, imgs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    net = Backbone().eval()
    sd = {
        k: v
        for k, v in torch.load(pth, map_location="cpu").items()
        if not k.startswith("fc.")
    }
    net.load_state_dict(sd)
    torch.onnx.export(
        net,
        torch.zeros(1, 3, 224, 224),
        out / "backbone_fp32.onnx",
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamo=False,
    )
    import onnxsim
    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    m, ok = onnxsim.simplify(onnx.load(out / "backbone_fp32.onnx"))
    assert ok
    onnx.save(m, out / "backbone_fp32.onnx")
    files = sorted(imgs.glob("*.jpg"))
    calib = [{"x": image(p)} for p in files[:32]]
    q = quantize_full_qdq(m, calib, method="minmax")
    q, info = quantized_io(q, nhwc_inputs=["x"])
    onnx.save(q, out / "backbone_qdq.onnx")
    (out / "io.json").write_text(json.dumps(info, indent=1))
    xs, zx = info["x"]["scale"], info["x"]["zero_point"]
    x = image(files[100])[0].transpose(1, 2, 0)
    xq = np.clip(np.rint(x / xs) + zx, 0, 255).astype(np.uint8)[None]
    xq.tofile(out / "input.bin")
    y = ort.InferenceSession(
        str(out / "backbone_qdq.onnx"), providers=["CPUExecutionProvider"]
    ).run(None, {"x": xq})[0]
    y.tofile(out / "ref.bin")
    print(
        "qdq model",
        out / "backbone_qdq.onnx",
        "io",
        info,
        "y",
        y.shape,
        y.dtype,
        y.min(),
        y.max(),
    )


if __name__ == "__main__":
    main()
