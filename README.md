# ONNX Simplifier

[![PyPI version](https://img.shields.io/pypi/v/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI license](https://img.shields.io/pypi/l/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/daquexian/onnx-simplifier/pulls)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=flat&logo=discord&logoColor=white)](https://discord.gg/W3ht33v4)

_ONNX is great, but sometimes too complicated._

## Background

One day I wanted to export the following simple reshape operation to ONNX:

```python
import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
```

The input shape in this model is static, so what I expected is

![simple_reshape](imgs/simple_reshape.png)

However, I got the following complicated model instead:

![complicated_reshape](imgs/complicated_reshape.png)

## Our solution

ONNX Simplifier is presented to simplify the ONNX model. It infers the whole computation graph
and then replaces the redundant operators with their constant outputs (a.k.a. constant folding).

### Web version

We have published ONNX Simplifier on [GitHub pages](https://onnxsim.github.io/onnxsim/). It works out of the box and **doesn't need any installation**. Note that it runs in the browser locally and your model is completely safe.

### Python version


```
pip3 install -U pip && pip3 install onnxsim
```

Then

```
onnxsim input_onnx_model output_onnx_model
```

For more advanced features, try the following command for help message

```
onnxsim -h
```

## Demonstration

An overall comparison between
[a complicated model](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)
and its simplified version:

![Comparison between old model and new model](imgs/comparison.png)

## In-script workflow

If you would like to embed ONNX simplifier python package in another script, it is just that simple.

```python
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(filename)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
```

You can see more details of the API in [onnxsim/onnx_simplifier.py](onnxsim/onnx_simplifier.py)

## Projects Using ONNX Simplifier

* [MXNet](https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/deploy/export/onnx.html#Simplify-the-exported-ONNX-model)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [ncnn](https://github.com/Tencent/ncnn)
* ...

## Chat

We created a Chinese QQ group for ONNX!

ONNX QQ Group (Chinese): 1021964010, verification code: nndab. Welcome to join!

For English users, I'm active on the [ONNX Slack](https://github.com/onnx/onnx#discuss). You can find and chat with me (daquexian) there.
