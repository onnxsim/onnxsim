import copy
from typing import Dict, Iterable, List, Optional, Set, Union

import numpy as np
import onnx
import onnx.checker
import onnx.defs

from . import backend

Tensors = Dict[str, np.ndarray]
TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

# How the values of the random test inputs are generated when the user does not
# supply their own ``input_data``. See ``compare``'s ``input_fill`` parameter.
INPUT_FILL_CHOICES = ("random", "ones", "zeros", "arange")


def _fill_array(shape: TensorShape, np_type: type, input_fill: str) -> np.ndarray:
    """Create an array of ``shape`` / ``np_type`` filled according to ``input_fill``.

    ``random`` draws from a uniform ``[0, 1)`` distribution (the historical
    default), ``ones``/``zeros`` fill with the constant, and ``arange`` fills with
    ``0, 1, 2, ...`` in row-major order, which is handy for reproducible,
    easy-to-eyeball checks.
    """
    if input_fill == "random":
        values = np.random.rand(*shape)
    elif input_fill == "ones":
        values = np.ones(shape)
    elif input_fill == "zeros":
        values = np.zeros(shape)
    elif input_fill == "arange":
        values = np.arange(int(np.prod(shape, dtype=np.int64))).reshape(shape)
    else:
        raise ValueError(
            'Unknown input_fill "{}", expected one of {}.'.format(
                input_fill, ", ".join(INPUT_FILL_CHOICES)
            )
        )
    return np.array(values, dtype=np_type)


def _iter_graph_nodes(graph: onnx.GraphProto) -> Iterable[onnx.NodeProto]:
    """Yield every node in ``graph``, recursing into subgraph attributes."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _iter_graph_nodes(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_graph_nodes(subgraph)


# Ops whose output is intentionally non-deterministic per the ONNX spec, but
# which all accept an optional FLOAT ``seed`` attribute to pin that output.
_RANDOM_OP_TYPES = frozenset(
    {
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Multinomial",
    }
)


def _is_unseeded_random_node(node: onnx.NodeProto) -> bool:
    return node.op_type in _RANDOM_OP_TYPES and not any(
        attr.name == "seed" for attr in node.attribute
    )


def _has_unseeded_random_ops(model: Union[str, onnx.ModelProto]) -> bool:
    # Structure-only load (no external data / initializers) is enough: op_type
    # and attributes don't depend on tensor weights.
    m = onnx.load(model, load_external_data=False) if isinstance(model, str) else model
    return any(_is_unseeded_random_node(node) for node in _iter_graph_nodes(m.graph))


def _with_fixed_random_seeds(
    model: Union[str, onnx.ModelProto], seed: float
) -> onnx.ModelProto:
    """Return a copy of ``model`` with every unseeded Random* op pinned to ``seed``.

    Random* ops (RandomNormal[Like], RandomUniform[Like], Multinomial) draw
    fresh noise on every onnxruntime session. ``compare`` creates a brand-new
    session per model per check_n trial, so comparing the original and
    simplified outputs is otherwise meaningless regardless of whether
    simplification changed anything -- confirmed in practice on VOICEVOX's
    predict_sing_f0.onnx, a flow-matching pitch predictor whose unseeded
    RandomNormalLike nodes made every check_n trial report a false failure.
    This only ever mutates a throwaway copy used for one comparison trial; the
    model returned to the caller is never touched.
    """
    model = onnx.load(model) if isinstance(model, str) else copy.deepcopy(model)
    for node in _iter_graph_nodes(model.graph):
        if _is_unseeded_random_node(node):
            node.attribute.append(onnx.helper.make_attribute("seed", seed))
    return model


def _custom_default_domain_ops(model: onnx.ModelProto) -> Set[str]:
    """Return op types in the default ONNX domain that have no ONNX schema.

    These are custom operators such as TensorRT plugins (e.g. ``BatchedNMS_TRT``)
    that were exported into the default domain. ``onnx.checker.check_model``
    rejects such a model with "No Op registered for <op> with domain_version of
    <n>" (GitHub issues #107, #220). onnxsim preserves these ops unchanged, so
    the checker error about them is expected and tolerated.
    """
    try:
        known = {
            (schema.domain, schema.name)
            for schema in onnx.defs.get_all_schemas_with_history()
        }
    except Exception:
        known = set()
    custom_ops = set()
    for node in _iter_graph_nodes(model.graph):
        if node.domain in ("", "ai.onnx") and ("", node.op_type) not in known:
            custom_ops.add(node.op_type)
    return custom_ops


def _has_one_sided_nan(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a NaN sits in one array where the other holds a non-NaN value.

    Used only to explain a reported mismatch: ``np.max(np.abs(a - b))`` is NaN
    whenever either side has one, so the printed "max diff" cannot by itself
    distinguish a real NaN-vs-number regression from the both-sides-NaN case
    that ``compare`` deliberately tolerates.
    """
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        return False  # integer/bool outputs cannot hold NaN
    return bool(np.any(np.isnan(a) != np.isnan(b)))


def compare(
    model_opt: Union[str, bytes, onnx.ModelProto],
    model_ori: Union[str, onnx.ModelProto],
    n_times: int = 5,
    input_shapes: Optional[TensorShapes] = None,
    input_data: Optional[Tensors] = None,
    custom_lib: Optional[str] = None,
    verbose=True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    input_fill: str = "random",
) -> bool:
    """
    :param model_opt: The simplified ONNX model. May be given as already-serialized
        ``bytes`` or a file path instead of a loaded ``ModelProto`` when ``n_times
        == 0`` -- ``onnx.checker.check_model`` accepts either directly (no
        re-serialization needed), and nothing else touches ``model_opt`` in that
        case. Passing bytes/a path with ``n_times > 0`` is not supported: the
        per-trial inference loop needs a real ``ModelProto``.
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    :param input_shapes: Shapes of generated random inputs
    :param input_data: User-given data instead of random generated data
    :param custom_lib: ONNX Runtime custom lib for custom ops
    :param rtol: Relative tolerance for ``numpy.allclose`` when comparing the
        original and simplified outputs. Increase it for very deep models whose
        (correct) op reordering accumulates floating-point error beyond the
        default -- see the RF-DETR XLarge case in ``scripts/rfdetr``.
    :param atol: Absolute tolerance for ``numpy.allclose`` (see ``rtol``).
        Outputs are compared with ``equal_nan=True``: a NaN the original model
        already produces for a given random input (a Div by zero, say) is not
        counted as a change when the simplified model produces a NaN in the
        same position. A NaN on only one side remains a mismatch.
    :param input_fill: How to fill the generated test inputs when ``input_data``
        is not given. One of ``"random"`` (uniform ``[0, 1)``, the default),
        ``"ones"``, ``"zeros"`` or ``"arange"`` (``0, 1, 2, ...`` in row-major
        order).
    """
    if input_fill not in INPUT_FILL_CHOICES:
        raise ValueError(
            'Unknown input_fill "{}", expected one of {}.'.format(
                input_fill, ", ".join(INPUT_FILL_CHOICES)
            )
        )

    def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_value_info_all(
        m: onnx.ModelProto, name: str
    ) -> Optional[onnx.ValueInfoProto]:
        for v in m.graph.value_info:
            if v.name == name:
                return v

        for v in m.graph.input:
            if v.name == name:
                return v

        for v in m.graph.output:
            if v.name == name:
                return v

        return None

    def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
        """
        Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
        """
        v = get_value_info_all(m, name)
        if v is not None:
            return get_shape_from_value_info_proto(v)
        raise RuntimeError('Cannot get shape of "{}"'.format(name))

    def get_elem_type(m: onnx.ModelProto, name: str) -> int:
        v = get_value_info_all(m, name)
        if v is not None:
            return v.type.tensor_type.elem_type
        raise RuntimeError('Cannot get element type of "{}"'.format(name))

    def get_np_type_from_elem_type(elem_type: int) -> type:
        sizes = (
            None,
            np.float32,
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.int32,
            np.int64,
            str,
            bool,
            np.float16,
            np.double,
            np.uint32,
            np.uint64,
            np.complex64,
            np.complex128,
            np.float16,
        )
        assert len(sizes) == 17
        size = sizes[elem_type]
        assert size is not None
        return size

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = list(
            set([ipt.name for ipt in model.graph.input])
            - set([x.name for x in model.graph.initializer])
        )
        return input_names

    def generate_rand_input(
        model: Union[str, onnx.ModelProto], input_shapes: Optional[TensorShapes] = None
    ):
        if input_shapes is None:
            input_shapes = {}
        if isinstance(model, str):
            model = onnx.load(model, load_external_data=False)
        input_names = get_input_names(model)
        full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
        assert None not in input_shapes
        full_input_shapes.update(input_shapes)  # type: ignore
        for name, shape in full_input_shapes.items():
            if any([dim <= 0 for dim in shape[1:]]):
                raise RuntimeError(
                    'The shape of input "{}" has dynamic size, '
                    "please set an input shape manually with --test-input-shape".format(
                        name
                    )
                )
            if len(shape) > 0 and shape[0] <= 0:
                print(
                    f'shape[0] of input "{name}" is dynamic, we assume it presents batch size and set it as 1 when testing. If it is not wanted, please set the it manually by --test-input-shape (see `onnxsim -h` for the details).'
                )
                shape[0] = 1

        inputs = {
            ipt: _fill_array(
                full_input_shapes[ipt],
                get_np_type_from_elem_type(get_elem_type(model, ipt)),
                input_fill,
            )
            for ipt in input_names
        }
        return inputs

    def forward(
        model: Union[str, onnx.ModelProto],
        inputs: Tensors,
        custom_lib: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        return backend.run_model(model, inputs, custom_lib=custom_lib)

    if input_shapes is None:
        input_shapes = {}
    try:
        onnx.checker.check_model(model_opt)
    except onnx.checker.ValidationError as e:
        # A model containing a custom op in the default ONNX domain (e.g. a
        # TensorRT plugin like BatchedNMS_TRT) fails validation with "No Op
        # registered for <op> ...". onnxsim preserves such ops unchanged, so
        # tolerate that specific error instead of failing (GitHub issues #107,
        # #220). Any other validation error is a genuine problem and re-raised.
        # This is the one spot that needs a real ModelProto to walk the graph --
        # rare enough (only reached on a validation failure) that materializing
        # one here, if the caller instead handed us bytes/a path to save the
        # checker a re-serialize on the common (no-error) path, costs nothing
        # that matters.
        if isinstance(model_opt, onnx.ModelProto):
            model_opt_proto = model_opt
        elif isinstance(model_opt, bytes):
            model_opt_proto = onnx.load_from_string(model_opt)
        else:
            model_opt_proto = onnx.load(model_opt)
        custom_ops = _custom_default_domain_ops(model_opt_proto)
        message = str(e)
        if custom_ops and any(op in message for op in custom_ops):
            print(
                "The model contains custom operator(s) {} in the default ONNX "
                "domain (e.g. a TensorRT plugin). They are preserved unchanged "
                "and skipped during model checking.".format(sorted(custom_ops))
            )
        else:
            raise
    # Only models with Random* ops pay for the extra per-trial copy below; the
    # overwhelming majority of models have none and this check is a cheap,
    # structure-only scan. Skipped entirely when n_times == 0: the loop below
    # never runs, and model_ori may already be None at that point (the caller
    # frees it once check_n == 0 makes it unnecessary).
    needs_seeding = n_times > 0 and (
        _has_unseeded_random_ops(model_opt) or _has_unseeded_random_ops(model_ori)
    )
    for i in range(n_times):
        print(f"Checking {i}/{n_times}...")
        if input_data is None:
            inputs = generate_rand_input(model_opt, input_shapes=input_shapes)
        else:
            inputs = input_data
        if needs_seeding:
            trial_ori = _with_fixed_random_seeds(model_ori, float(i))
            trial_opt = _with_fixed_random_seeds(model_opt, float(i))
        else:
            trial_ori = model_ori
            trial_opt = model_opt
        res_ori = forward(trial_ori, inputs, custom_lib)
        res_opt = forward(trial_opt, inputs, custom_lib)

        for name in res_opt.keys():
            # equal_nan=True: a NaN produced at the same position by *both*
            # models is not evidence that anything changed -- it just means the
            # model itself is undefined at this (random) input, e.g. a Div by
            # zero. Without it, NaN != NaN made such inputs report a bogus
            # "changes after optimization" (GitHub issue #1285). It only ever
            # equates NaN with NaN: a NaN in one output where the other holds a
            # number is still a mismatch, and so is +Inf vs -Inf or Inf vs any
            # finite value (np.allclose compares infinities exactly by default,
            # which is already the behaviour we want for them).
            if not np.allclose(
                res_opt[name], res_ori[name], rtol=rtol, atol=atol, equal_nan=True
            ):
                if verbose:
                    print(
                        "Tensor {} changes after optimization. The max diff is {}.".format(
                            name, np.max(np.abs(res_opt[name] - res_ori[name]))
                        )
                    )
                    if _has_one_sided_nan(res_opt[name], res_ori[name]):
                        # The max diff is NaN in this case, which reads exactly
                        # like the (now tolerated) both-sides-NaN case. Say
                        # which one it actually is.
                        print(
                            "(The max diff is NaN because NaN appears in one "
                            "model's output where the other's is a number. "
                            "Matching NaNs on both sides are not reported.)"
                        )
                    print("After optimization:")
                    print(res_opt[name])
                    print("Before optimization:")
                    print(res_ori[name])
                    print("----------------")
                return False
    return True
