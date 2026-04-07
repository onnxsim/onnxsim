import argparse

import copy
import os
import sys
import re
import tempfile
from typing import List, Literal, Dict, Union, Optional, Tuple, Sequence
from rich.text import Text
from rich import print
import numpy as np
from google.protobuf.message import EncodeError

import onnx  # type: ignore
import onnx.checker  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
try:
    import onnxruntime as rt  # type: ignore
except ImportError:
    command = [sys.executable, '-m', 'pip', 'install', 'onnxruntime']
    print(Text(f"Installing onnxruntime by `{' '.join(command)}`, please wait for a moment..", style="bold magenta"))
    import subprocess
    subprocess.check_call(command)
    import onnxruntime as rt


import onnxsim.onnxsim_cpp2py_export as C
from . import model_info
from . import model_checking
from . import version


TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]
Unit = Literal["B", "KB", "MB", "GB", "TB"]

UNIT_MAP: dict[Unit, int] = {
    "B": 1,
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
    "TB": 1 << 40,
}

def get_output_names(model: onnx.ModelProto) -> List[str]:
    output_names = [opt.name for opt in model.graph.output]
    return output_names


def remove_unused_output(
    model: onnx.ModelProto, unused_output: Sequence[str]
) -> onnx.ModelProto:
    unused_output_names = unused_output
    output_names = get_output_names(model)
    for unused_output_name in unused_output_names:
        if unused_output_name not in output_names:
            raise RuntimeError(
                f'The model doesn\'t have output named "{unused_output_name}"'
            )
    for graph_output in copy.deepcopy(model.graph.output):
        if graph_output.name in unused_output_names:
            model.graph.output.remove(graph_output)
    return model


def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
    initializer_names = [x.name for x in model.graph.initializer]
    for graph_input in copy.deepcopy(model.graph.input):
        if graph_input.name in initializer_names:
            model.graph.input.remove(graph_input)
    return model


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: Optional[TensorShapesWithOptionalKey]) -> Optional[TensorShapes]:
    if input_shapes is None:
        return None

    def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = [ipt.name for ipt in get_inputs(model)]
        return input_names

    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(x))

    return input_shapes  # type: ignore


# A very very large threshold
DEFAULT_TENSOR_SIZE_THRESHOLDHOLD = '1.5GB'


def simplify(
    model: Union[str, onnx.ModelProto],
    check_n: int = 0,
    perform_optimization: bool = True,
    skip_fuse_bn: bool = False,
    overwrite_input_shapes=None,
    test_input_shapes=None,
    skipped_optimizers: Optional[List[str]] = None,
    skip_constant_folding=False,
    skip_shape_inference=False,
    input_data=None,
    dynamic_input_shape: bool = False,
    custom_lib: Optional[str] = None,
    include_subgraph: bool = False,
    unused_output: Optional[Sequence[str]] = None,
    tensor_size_threshold: str = DEFAULT_TENSOR_SIZE_THRESHOLDHOLD,
    mutable_initializer: bool = False,
    *,
    input_shapes=None,
) -> Tuple[onnx.ModelProto, bool]:
    """
    :param model: onnx ModelProto object or file path
    :param check_n: The simplified model will be checked for `check_n` times by random inputs
    :param perform_optimization: Whether to run onnx optimizer on the model
    :param skip_fuse_bn: Skip fuse_bn_into_conv onnx optimizer
    :param overwrite_input_shapes: If the model has dynamic input shape, user must pass a fixed input shape
            for generating random inputs and checking equality.
    :param test_input_shapes: If the model has dynamic input shape, user must pass a fixed input shape
            for generating random inputs and checking equality.
    :param skipped_optimizers: Skip some specific onnx optimizers
    :param skip_constant_folding: Skip constant folding
    :param skip_shape_inference: Skip shape inference (sometimes shape inference will crash)
    :param input_data: Feed custom input data for checking if needed
    :param dynamic_input_shape: Deprecated. Not needed anymore.
    :param custom_lib: onnxruntime custom ops's shared library
    :param include_subgraph: Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph
    :param unused_output: name of unused outputs that will be eliminated from the model
    :param input_shapes: Deprecated. Please use `overwrite_input_shapes` and/or `test_input_shapes` instead.
    :return: A tuple (simplified model, success(True) or failed(False))
    """
    if dynamic_input_shape:
        print(
            Text(
                "WARNING: The argument `dynamic_input_shape=True` is not needed any more, onnxsim can now support dynamic input shapes natively, please refer to the latest documentation. An error will be raised in the future.",
                style="bold red",
            )
        )
    if input_shapes is not None:
        print(
            Text(
                "WARNING: The argument `input_shapes` is deprecated. Please use `overwrite_input_shapes` and/or `test_input_shapes` instead. An error will be raised in the future.",
                style="bold red",
            )
        )
        overwrite_input_shapes = input_shapes
        test_input_shapes = input_shapes

    if not perform_optimization:
        # None means skip all optimizers
        skipped_optimizers = None
    elif skipped_optimizers is None:
        skipped_optimizers = []

    if skip_fuse_bn and skipped_optimizers is not None:
        skipped_optimizers.append("fuse_bn_into_conv")
    if isinstance(model, str):
        model = onnx.load(model)
    if overwrite_input_shapes is None:
        overwrite_input_shapes = {}
    overwrite_input_shapes = check_and_update_input_shapes(
        model, overwrite_input_shapes)
    test_input_shapes = check_and_update_input_shapes(
        model, test_input_shapes)

    for name, input_shape in overwrite_input_shapes.items():
        for ipt in model.graph.input:
            if ipt.name == name:
                for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                    dim.dim_value = input_shape[i]
    if unused_output is not None:
        model = remove_unused_output(model, unused_output)
    if not mutable_initializer and model.ir_version >= 4:
        model = remove_initializer_from_input(model)

    def parse_size(size: str) -> int:
        m = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", size.strip(), re.I)
        if not m:
            raise ValueError(size)
        number: float = float(m.group(1))
        unit: Unit = m.group(2).upper()  # type: ignore
        return int(number * UNIT_MAP[unit])

    tensor_size_threshold = parse_size(tensor_size_threshold)
    if tensor_size_threshold > 2**31 - 9999:
        raise ValueError("tensor_size_threshold should be less than 2GB")

    try:
        model_bytes = model.SerializeToString()
        model_opt_bytes = C.simplify(
            model_bytes,
            skipped_optimizers,
            not skip_constant_folding,
            not skip_shape_inference,
            tensor_size_threshold,
        )
        if len(model_opt_bytes) == 0:
            raise ValueError("Simplified model larger than 2GB")
        model_opt = onnx.load_from_string(model_opt_bytes)
        check_ok = model_checking.compare(
            model_opt, model, check_n, test_input_shapes, input_data, custom_lib
        )
    except (EncodeError, ValueError, onnx.onnx_cpp2py_export.checker.ValidationError):
        print("[bold magenta]Simplified model larger than 2GB. Trying to save as external data...[/bold magenta]")
        # large models try to convert through a temporary file
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx.save(
                copy.deepcopy(model),
                os.path.join(tmpdirname, 'model.onnx'),
                save_as_external_data=True,
            )
            check_ok = C.simplify_path(
                os.path.join(tmpdirname, 'model.onnx'),
                os.path.join(tmpdirname, 'opt.onnx'),
                skipped_optimizers,
                not skip_constant_folding,
                not skip_shape_inference,
                tensor_size_threshold,
            )
            check_ok = model_checking.compare(
                os.path.join(tmpdirname, 'opt.onnx'),
                os.path.join(tmpdirname, 'model.onnx'),
                check_n, test_input_shapes, input_data, custom_lib
            )
            model_opt = onnx.load(os.path.join(tmpdirname, 'opt.onnx'))
    return model_opt, check_ok


class PyModelExecutor(C.ModelExecutor):
    def Run(self, model_str: str, inputs_str: List[str]):
        model = onnx.ModelProto()
        model.ParseFromString(model_str)

        def deserialize_tp(tp_str):
            tp = onnx.TensorProto()
            tp.ParseFromString(tp_str)
            return tp

        input_tps = map(deserialize_tp, inputs_str)
        input_arrs = map(onnx.numpy_helper.to_array, input_tps)
        input_names = [x.name for x in model.graph.input]
        inputs = dict(zip(input_names, input_arrs))
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        sess = rt.InferenceSession(
            model.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        output_names = [x.name for x in sess.get_outputs()]
        run_options = rt.RunOptions()
        run_options.log_severity_level = 3
        output_arrs = sess.run(output_names, inputs, run_options=run_options)
        return [
            onnx.numpy_helper.from_array(x).SerializeToString() for x in output_arrs
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")
    parser.add_argument(
        "check_n",
        help="Check whether the output is correct with n random inputs",
        nargs="?",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable-fuse-bn",
        help="This option is deprecated. Fusing bn into conv is enabled by default.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-fuse-bn", help="Skip fusing batchnorm into conv.", action="store_true"
    )
    parser.add_argument(
        "--skip-optimization",
        help="Skip all ONNX optimizers or some of them. To skip all optimizers, use `onnxsim a.onnx b.onnx --skip-optimization`. To skip some of optimizers, use something like `onnxsim a.onnx b.onnx --skip-optimization fuse_bn_into_conv fuse_pad_into_pool`.",
        type=str,
        nargs="*",
    )
    parser.add_argument("--skip-constant-folding", help="Skip constant folding", action="store_true")
    parser.add_argument(
        "--input-shape",
        help="This argument has been renamed to --overwrite-input-shape, please refer to it",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--overwrite-input-shape",
        help='Overwrite the input shape. The format is "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--test-input-shape",
        help='The input shape to generated random inputs for test, useful when the input shape is dynamic. The format is "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-optimizer",
        help="Deprecated. Refer to --skip-optimization",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-shape-inference", help="Skip shape inference", action="store_true"
    )
    parser.add_argument(
        "--enable-onnxruntime-optimization",
        help="Enable ONNX Runtime's ORT_ENABLE_BASIC level optimization.",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic-input-shape",
        help="Deprecated. Not needed any more.",
        action="store_true",
    )
    parser.add_argument(
        "--input-data-path",
        help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--custom-lib", help="Deprecated. Not needed any more.", type=str
    )
    parser.add_argument(
        "--include-subgraph",
        help='Experimental feature. Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph',
        action="store_true",
    )
    parser.add_argument(
        "--unused-output",
        help="Name of unused outputs that will be eliminated from the model",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--no-large-tensor",
        help="Some ops like Tile and ConstantOfShape can produce large tensor and make the model size much larger. Specifying this flag to skip folding these ops, with loss of some optimization chances. It can be followed with a threshold, for example, --no-large-tensor 1M or --no-large-tensor 100KB. A simple '--no-large-tensor' means '--no-large-tensor 1KB'.",
        type=str,
        const='1KB',
        default=DEFAULT_TENSOR_SIZE_THRESHOLDHOLD,
        nargs="?",
        dest="tensor_size_threshold",
    )
    parser.add_argument(
        "--mutable-initializer",
        help="By ONNX specification, initializers can also serve as inputs. This allows users to overwrite their values during runtime, but some useful optimizations like fuse-conv-and-bn will not be applicable anymore. In almost all cases, having an initializer that is also an input is unintended (usually caused by a out-dated PyTorch). So onnxsim treats all initializers immutable to enabling all optimizations. If it is not wanted, you can specify '--mutable-initializer' to disable this behavior.",
        action="store_true",
        )
    parser.add_argument(
        "--save-as-external-data",
        help="Save parameters as external data. This will make the .onnx file much smaller, but the .onnx file will depend on the external data file (.data).",
        action="store_true",
        )
    parser.add_argument('-v', '--version', action='version', version='onnxsim ' + version.version)

    class ListOptimizers(argparse.Action):
        def __call__(self, parser, ns, v, option_string=None):
            for p in C._list_optimizers():
                print(p)
            parser.exit()

    parser.add_argument("--list-default-optimizers", help="List default optimizer pass names", nargs=0, action=ListOptimizers)

    args = parser.parse_args()

    if args.enable_fuse_bn:
        print(
            Text(
                'WARNING: "--enable-fuse-bn" is not needed any more, because fuse bn is enabled by default. "--enable-fuse-bn" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    if args.dynamic_input_shape:
        print(
            Text(
                'WARNING: "--dynamic-input-shape" is not needed any more, onnxsim v0.4 now handles dynamic input shapes automatically. "--dynamic-input-shape" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    assert not (args.input_shape is not None and args.overwrite_input_shape is not None)
    if args.input_shape:
        print(
            Text(
                'WARNING: "--input-shape" is renamed to "--overwrite-input-shape". Please use it instead.',
                style="bold red",
            )
        )
        args.overwrite_input_shape = args.input_shape
    if args.include_subgraph:
        print(
            Text(
                "WARNING: subgraph optimization is not supported in v0.4 for now.",
                style="bold red",
            )
        )
    assert not (args.skip_optimizer is not None and args.skip_optimization is not None)
    if args.skip_optimizer:
        print(
            Text(
                'WARNING: "--skip-optimizer" is renamed to "--skip-optimization". Please use it instead.',
                style="bold red",
            )
        )
        args.skip_optimization = args.skip_optimizer
    if args.skip_optimization is None:
        # user doesn't specify --skip-optimization
        args.skip_optimization = []
    elif len(args.skip_optimization) == 0:
        # user specify --skip-optimization without any certain optimizer name
        # set it to None means skip all optimizations
        args.skip_optimization = None
    if args.skip_fuse_bn and args.skip_optimization is not None:
        args.skip_optimization.append("fuse_bn_into_conv")

    perform_optimization = False if args.skip_optimization is None else True

    def parse_shapes(shapes_arg):
        shapes = {}
        if shapes_arg is not None:
            for x in shapes_arg:
                if ':' not in x:
                    shapes[None] = list(map(int, x.split(',')))
                else:
                    pieces = x.split(':')
                    # for the input name like input:0
                    name, shape = ':'.join(
                        pieces[:-1]), list(map(int, pieces[-1].split(',')))
                    shapes.update({name: shape})
        return shapes

    test_input_shapes = parse_shapes(args.test_input_shape)
    overwrite_input_shapes = parse_shapes(args.overwrite_input_shape)

    if args.enable_onnxruntime_optimization:

        tmp_file = tempfile.NamedTemporaryFile()
        sess_options = rt.SessionOptions()
        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # To enable model serialization after graph optimization
        sess_options.optimized_model_filepath = tmp_file.name
        _ = rt.InferenceSession(args.input_model, sess_options, providers=["CPUExecutionProvider"])

        model = onnx.load(tmp_file.name)
    else:
        model = onnx.load(args.input_model)

    if args.tensor_size_threshold == DEFAULT_TENSOR_SIZE_THRESHOLDHOLD:
        for node in model.graph.node:
            if node.op_type in ["Tile", "ConstantOfShape"]:
                print(
                    Text(
                        'Your model contains "Tile" ops or/and "ConstantOfShape" ops. Folding these ops can make the simplified model much larger. If it is not expected, please specify "--no-large-tensor" (which will lose some optimization chances)',
                        style="bold magenta",
                    )
                )
                break

    if not args.mutable_initializer:
        initializer_names = set([x.name for x in model.graph.initializer])
        input_names = set([x.name for x in model.graph.input])
        if len(initializer_names.intersection(input_names)) > 0:
            print(
                Text(
                    'Your model contains initializers that are also inputs. This is usually caused by an out-dated PyTorch. onnxsim treats all initializers immutable to enabling all optimizations. If it is not wanted, please specify "--mutable-initializer" to disable this behavior.',
                    style="bold magenta",
                )
            )

    input_tensors = None
    if args.input_data_path is not None:
        input_tensors = {}
        for x in args.input_data_path:
            pieces = x.split(':')
            name, data = ':'.join(pieces[:-1]), pieces[-1]
            input_tensors.update({name: np.load(data)})

    print("Simplifying...")

    model_opt, check_ok = simplify(
        model,
        args.check_n,
        perform_optimization,
        False,
        overwrite_input_shapes,
        test_input_shapes,
        args.skip_optimization,
        args.skip_constant_folding,
        args.skip_shape_inference,
        input_tensors,
        False,
        args.custom_lib,
        args.include_subgraph,
        args.unused_output,
        args.tensor_size_threshold,
        args.mutable_initializer,
    )

    try:
        if not args.save_as_external_data:
            onnx.save(model_opt, args.output_model)
        else:
            raise ValueError("save_as_external_data")
    except ValueError:
        # large models (>2GB) which onnx.save doesn't support,
        # or explicitly specified --save-as-external-data
        external_data_path = os.path.basename(args.output_model) + '.data'
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
        onnx.save(
            copy.deepcopy(model_opt),
            args.output_model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
        )

    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_opt)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        sys.exit(1)
