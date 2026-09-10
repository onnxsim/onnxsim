"""Scraped from Voyager SDK's docs/reference/compiler/onnx-support.md and
onnx-opset17-support.md (opset 17, the compiler's own
recommended default -- see scrape_onnx_support_docs.py's docstring for why
only one opset is captured). Auto-generated -- do not hand-edit; re-run
scrape_onnx_support_docs.py against a voyager-sdk checkout instead."""

VOYAGER_OP_SUPPORT = {
    "Add": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "A.shape == B.shape and not A.is_constant and not B.is_constant",
            "len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)",
            "len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)",
            "len(A.shape)==4 and A.shape[1]!=1 and (B.shape==(1, A.shape[1], 1, 1) or B.shape==())",
            "len(B.shape)==4 and B.shape[1]!=1 and (A.shape==(1, B.shape[1], 1, 1) or A.shape==())",
        ],
        "notes": "Given an operand with shape [N, C, H, W], Addition is supported with other operands with shape [N, C, H, W], [1, C, 1, 1], and scalars. If operands have the same shapes, they must be non-constant.",
    },
    "AveragePool": {
        "level": "Constrained",
        "rules": [
            'auto_pad=="NOTSET"',
            "pads is None or len(pads)==4",
            "pads is None or (pads[0]==pads[2] and pads[1]==pads[3] and pads[0]<=0.5*kernel_shape[0] and pads[1]<=0.5*kernel_shape[1]) or (pads[0]!=pads[2]) or (pads[1]!=pads[3])",
            "pads is None or (pads!=[0, 0, 0, 0] and count_include_pad==1) or (pads==[0, 0, 0, 0])",
        ],
        "allow_config": [],
        "notes": 'Only AveragePool operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported. Moreover, due to torch runtime constraints, symmetric padding along each dimension must be at most half of the kernel size along the same dimension. Lastly, note that count_include_pad is only supported equal to 1. If padding is specified for this operator, count_include_pad !=1 may lead to wrong results.',
    },
    "BatchNormalization": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "Clip": {
        "level": "Constrained",
        "rules": [],
        "allow_config": ["min==0 and max==6", "min==-1 and max==1"],
        "notes": "Only Clip operators implementing ReLU6 (min=0, max=6) and HardTanh (min=-1, max=1) are currently supported.",
    },
    "Concat": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "all([len(x.shape) == 4 for x in inputs]) and axis in [-3, -2, -1, 1, 2, 3]"
        ],
        "notes": "Concatenation is supported along the C, H, or W dimension for 4D feature maps.",
    },
    "Conv": {
        "level": "Constrained",
        "rules": [
            'auto_pad == "NOTSET"',
            "group == 1 or (W.shape[2] == W.shape[3] and (dilations is None or dilations[0] == dilations[1]) and (strides is None or strides[0] == strides[1]))",
            "group == 1 or (dilations is not None and ((W.shape[2] - 1) * dilations[0] + 1) * ((W.shape[3] - 1) * dilations[1] + 1) < 128) or (dilations is None and W.shape[2] * W.shape[3] < 128)",
        ],
        "allow_config": [],
        "notes": 'Only Conv operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported. Only symmetric kernels/strides/dilations with kernel_h * kernel_w < 128 are supported for grouped and depthwise convolutions. The kernel dimensions are read from the weight tensor W (shape [M, C/group, kH, kW]) rather than the optional "kernel_shape" attribute, which ONNX exporters frequently omit (leaving it None and inferring it from W).',
    },
    "ConvTranspose": {
        "level": "Constrained",
        "rules": [
            'auto_pad == "NOTSET"',
            "group == 1",
            "pads is None or pads[:len(pads)//2] == pads[len(pads)//2:]",
            "output_shape is None",
        ],
        "allow_config": [],
        "notes": 'Only ConvTranspose operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported. Grouped convolutions and "output_shape" attribute are not supported.',
    },
    "Flatten": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [],
        "notes": "The Flatten operations is only supported in specific scenario, such as before a Gemm layer, or at the end of a model. Note that, even in the cases where it is supported, Flatten should have axes specified as >= 0.",
    },
    "Gemm": {
        "level": "Constrained",
        "rules": ["transA == 0"],
        "allow_config": [],
        "notes": "Gemm with automatic transposition of the first operand (i.e., transA == 1) is not supported.",
    },
    "GlobalAveragePool": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "GlobalMaxPool": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "HardSigmoid": {
        "level": "Constrained",
        "rules": ["abs(alpha - 0.166667) < 5e-6", "abs(beta - 0.5) < 5e-6"],
        "allow_config": [],
        "notes": "Only HardSigmoid operators with pytorch-like parameters (alpha=1/6, beta=0.5) are supported.",
    },
    "HardSwish": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "LeakyRelu": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "MatMul": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [],
        "notes": "Matmul is supported only when it is part of the attention block used by the YOLO11 family of networks.",
    },
    "MaxPool": {
        "level": "Constrained",
        "rules": ['auto_pad=="NOTSET"', "storage_order==0"],
        "allow_config": [],
        "notes": 'Only MaxPool operators with explicit padding (i.e., auto_pad = "NOTSET") and row major order (i.e. storage_order=0) are currently supported.',
    },
    "Mul": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "A.shape == B.shape and not A.is_constant and not B.is_constant",
            "(A.shape==() and len(B.shape)==4) or (B.shape==() and len(A.shape)==4)",
            "len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)",
            "len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)",
            "len(A.shape)==4 and A.shape[1]!=1 and B.shape==(1, A.shape[1], 1, 1)",
            "len(B.shape)==4 and B.shape[1]!=1 and A.shape==(1, B.shape[1], 1, 1)",
        ],
        "notes": "Given an operand with shape [N, C, H, W], Multiplication is supported with other operands with shape [N, C, H, W], [1, C, 1, 1], and scalars. Multiplication cannot be performed when the left and right operands are the same node.",
    },
    "PRelu": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "slope.size == 1",
            "np.array_equal([x for x in slope.shape if x != 1], [X.shape[1]])",
        ],
        "notes": "Due to torch runtime constraints, Prelu is supported with either scalar or per-channel slope parameters.",
    },
    "Pad": {
        "level": "Constrained",
        "rules": [
            "constant_value is None or constant_value == 0.0",
            'mode not in ["reflect", "edge"]',
            "len(pads) == 2 * len(data.shape) and pads[0] == pads[len(data.shape)] == 0",
            "len(pads) == 2 * len(data.shape) and pads[1] % 64 == pads[1 + len(data.shape)] % 64 == 0",
        ],
        "allow_config": [],
        "notes": 'Pad is currently supported only for the "constant" mode. Moreover, padding along the batch dimension is not supported, and should be specified as 0 in the pads parameter. Padding along the channel dimension should be a multiple of 64.',
    },
    "Relu": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "Reshape": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "allowzero == 0 and np.array_equal(shape, data.shape)",
            "allowzero == 0 and len(data.shape) >= 2 and len(shape) == 2 and shape[0] == data.shape[0] and (shape[1] == data.shape[1] or shape[1] == -1)",
        ],
        "notes": "Reshape is supported in two cases: When used within attention blocks in Yolo11 networks and for trivial operations where the shape parameter equals the input shape (no-op) or acts like a squeeze operation (e.g., [1, N, 1, 1] → [1, N]).",
    },
    "Resize": {
        "level": "Constrained",
        "rules": [
            "roi is None",
            'mode in ["nearest", "linear"]',
            'coordinate_transformation_mode in ["half_pixel", "pytorch_half_pixel", "asymmetric"]',
            'nearest_mode in ["round_prefer_floor", "round_prefer_ceil", "floor"]',
            '(mode == "linear" and coordinate_transformation_mode in ["half_pixel", "pytorch_half_pixel"]) or (mode == "nearest" and coordinate_transformation_mode in ["half_pixel", "pytorch_half_pixel"] and nearest_mode in ["round_prefer_floor", "round_prefer_ceil"]) or (mode == "nearest" and coordinate_transformation_mode in ["asymmetric"] and nearest_mode in ["floor"])',
            '(mode == "linear" and Y.shape[-2] // X.shape[-2] == Y.shape[-1] // X.shape[-1]) or mode == "nearest"',
            '(mode == "linear" and Y.shape[-2] % X.shape[-2] == 0) or (mode == "nearest" and (Y.shape[-2] % X.shape[-2] <= 1 or Y.shape[-2] % X.shape[-2] == X.shape[-2] - 1))',
            '(mode == "linear" and Y.shape[-1] % X.shape[-1] == 0) or (mode == "nearest" and (Y.shape[-1] % X.shape[-1] <= 1 or Y.shape[-1] % X.shape[-1] == X.shape[-1] - 1))',
        ],
        "allow_config": [],
        "notes": 'Three coordinate transformation modes are supported: "half_pixel", "pytorch_half_pixel", and "asymmetric". For nearest mode, available options are "round_prefer_floor", "round_prefer_ceil", or "floor". Linear resizing is limited to symmetric, integer scaling factors, while nearest resizing supports any integer scaling factors. Note that the "floor" nearest mode can only be used with asymmetric coordinate transformation.',
    },
    "Selu": {
        "level": "Constrained",
        "rules": ["abs(alpha - 1.67326) < 5e-6", "abs(gamma - 1.0507) < 5e-6"],
        "allow_config": [],
        "notes": "Selu operators are supported if the alpha and gamma parameters are set to the defaults of 1.67326 and 1.0507, respectively.",
    },
    "Sigmoid": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "Slice": {
        "level": "Constrained",
        "rules": [
            "axes is not None and len(axes) == 1",
            "steps is None or (len(steps) == 1 and (steps == 1 or axes == 1))",
            "(axes[0] != 1 and data.shape[1] % 64 == 0) or axes[0] == 1",
        ],
        "allow_config": [],
        "notes": "Slice is supported along one axis only, which must be specified as input to the operator. Stepped slice is currently supported only on the channel dimension. Slicing along any axis other than the channel axis requires the number of channels to be a multiple of 64",
    },
    "Softmax": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [],
        "notes": "Softmax is supported only when it is part of the attention block used by the YOLO11 family of networks.",
    },
    "Split": {
        "level": "Constrained",
        "rules": ["axis > 0"],
        "allow_config": [],
        "notes": "Split is not supported as the first operator in a model. Split is supported for axis different than 0. Negative axis values are not supported.",
    },
    "Squeeze": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [],
        "notes": "The Squeeze operations is only supported in specific scenario, such as before a Gemm layer, or at the end of a model.",
    },
    "Sub": {
        "level": "Constrained",
        "rules": [],
        "allow_config": [
            "(A.shape==() and len(B.shape)==4) or (B.shape==() and len(A.shape)==4)",
            "len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)",
            "len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)",
            "len(A.shape)==4 and A.shape[1]!=1 and B.shape==(1, A.shape[1], 1, 1)",
            "len(B.shape)==4 and B.shape[1]!=1 and A.shape==(1, B.shape[1], 1, 1)",
        ],
        "notes": "Given an operand with shape [N, C, H, W], Subtraction is supported with other operands with shape [1, C, 1, 1] or scalars.",
    },
    "Tanh": {
        "level": "Supported",
        "rules": [],
        "allow_config": [],
        "notes": "Operator is supported in any configurations.",
    },
    "Transpose": {
        "level": "Constrained",
        "rules": ["perm == [0, 1, 2, 3]"],
        "allow_config": [],
        "notes": "Transpose operations that are not the no-op, trivial case (i.e. perm=[0, 1, 2, 3]), are not supported. Transpose with perm=[0, 1, 3, 2] is supported only when it is part of the attention block used by the YOLO11 family of networks.",
    },
}
