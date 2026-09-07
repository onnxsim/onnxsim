"""Inference backend used for constant folding and correctness checking.

onnxruntime is preferred when it is available. If it is not installed,
onnxsim falls back to onnx's built-in reference evaluator so that
onnxruntime becomes an optional dependency (installing onnxruntime is
sometimes harmful, see https://github.com/onnxsim/onnxsim/issues/441).
"""

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import onnx

try:
    import onnxruntime as rt  # type: ignore

    _HAS_ONNXRUNTIME = True
except ImportError:
    rt = None  # type: ignore
    _HAS_ONNXRUNTIME = False


# An execution provider is either the provider name (e.g.
# ``"CUDAExecutionProvider"``) or a ``(name, options_dict)`` tuple as accepted
# by ``onnxruntime.InferenceSession``. onnxruntime tries the providers in order
# and falls back to the next one for operators a provider cannot run, so the CPU
# provider is normally kept last as a catch-all.
Provider = Union[str, Tuple[str, Dict[str, object]]]

# Constant folding runs on CPU unless the caller asks otherwise. CPU is always
# available and deterministic, which keeps folding results stable regardless of
# the machine onnxsim happens to run on.
DEFAULT_PROVIDERS: List[str] = ["CPUExecutionProvider"]


def has_onnxruntime() -> bool:
    """Whether onnxruntime is available as the inference backend."""
    return _HAS_ONNXRUNTIME


def _ort_profile_prefix() -> Optional[str]:
    """The file prefix for onnxruntime's built-in session profiler, or ``None``
    when it is disabled.

    Constant folding runs each fold group through an ``onnxruntime`` session;
    setting ``ONNXSIM_ORT_PROFILE`` turns on onnxruntime's own per-operator
    profiler (``SessionOptions.enable_profiling``) for those sessions, which is
    finer-grained than onnxsim's ``OrtSession`` span. The variable names a file
    *prefix* -- onnxruntime writes one ``<prefix>_<timestamp>.json`` Chrome trace
    per session -- and mirrors ``ONNXSIM_PROFILE``: the truthy shorthands
    ``1``/``true``/``on``/``yes`` (and the empty string, as set by the
    ``ort_profile=""`` API default) select the default prefix.
    """
    value = os.environ.get("ONNXSIM_ORT_PROFILE")
    if value is None:
        return None
    if value.lower() in ("", "1", "true", "on", "yes"):
        return "onnxsim_ort_profile"
    return value


def _provider_name(provider: Provider) -> str:
    """The provider name whether ``provider`` is a bare string or a
    ``(name, options)`` tuple."""
    return provider[0] if isinstance(provider, (tuple, list)) else provider


def _check_providers_available(providers: Sequence[Provider]) -> None:
    """Raise a helpful error if any requested provider is not built into the
    installed onnxruntime.

    onnxruntime otherwise only logs a warning and silently drops an unavailable
    provider (e.g. ``CUDAExecutionProvider`` when the CPU-only wheel is
    installed), so a user who asked to fold on the GPU would quietly get CPU
    execution instead. Failing loudly makes that misconfiguration obvious.
    """
    available = set(rt.get_available_providers())
    missing = [
        _provider_name(p) for p in providers if _provider_name(p) not in available
    ]
    if missing:
        raise ValueError(
            "The following execution provider(s) are not available in the "
            f"installed onnxruntime: {missing}. Available providers: "
            f"{sorted(available)}. For CUDA, install the GPU build with "
            "`pip install onnxruntime-gpu`."
        )


def validate_providers(providers: Optional[Sequence[Provider]]) -> None:
    """Validate a requested execution-provider list, raising if it cannot be
    honoured by the current backend.

    Callers use this to fail fast *before* constant folding starts. onnxsim's
    folding loop catches per-op executor errors and simply leaves the op
    unfolded, so an unavailable provider raised deep inside a fold would be
    swallowed and silently degrade to no folding rather than surfacing to the
    user. Checking here instead turns a misconfigured provider into an
    immediate, actionable error.

    ``None`` (fold on CPU) is always valid.
    """
    if providers is None:
        return
    if _HAS_ONNXRUNTIME:
        _check_providers_available(providers)
        return
    # Without onnxruntime only the pure-Python reference evaluator is available,
    # which runs on the CPU and cannot honour any other provider.
    non_cpu = [
        _provider_name(p)
        for p in providers
        if _provider_name(p) != "CPUExecutionProvider"
    ]
    if non_cpu:
        raise ValueError(
            "Execution providers other than CPUExecutionProvider require "
            "onnxruntime. Please install it (e.g. `pip install onnxruntime-gpu` "
            f"for CUDA). Requested providers: {non_cpu}."
        )


def _run_with_onnxruntime(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]],
    custom_lib: Optional[str],
    providers: Optional[Sequence[Provider]] = None,
    single_threaded: bool = False,
    deterministic: bool = False,
) -> "OrderedDict[str, np.ndarray]":
    if providers is None:
        providers = DEFAULT_PROVIDERS
    validate_providers(providers)
    sess_options = rt.SessionOptions()
    if custom_lib is not None:
        if os.path.exists(custom_lib):
            sess_options.register_custom_ops_library(custom_lib)
        else:
            raise ValueError("No such file '{}'".format(custom_lib))
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    # Every session created here runs exactly once (one Run() call), so
    # onnxruntime's memory-pattern optimizer -- which spends time up front
    # planning buffer reuse across *repeated* Run() calls -- pays for itself
    # never. Disabling it removes that planning cost from every session.
    sess_options.enable_mem_pattern = False
    if deterministic:
        # onnxruntime's own documented switch for exactly this: steers CPU/
        # CUDA/ROCM kernels (MatMul, reductions, etc.) away from any
        # algorithm whose result can depend on the host's available CPU/GPU
        # instruction set (e.g. MLAS choosing an AVX-512 vs. AVX2 vs. SSE
        # kernel), at some performance cost. Without this, two hosts that
        # differ only in SIMD width can silently diverge in a
        # graph_optimization_level=0 session too, since that level only
        # controls *graph rewrites*, not which low-level kernel a node's op
        # dispatches to. Combine with single_threaded=True (below) to also
        # remove thread-partitioning as a source of divergence -- both matter
        # for a *measurement* like :func:`onnxsim.measure_accuracy_drop`,
        # which is only meaningful if it reproduces regardless of which
        # machine runs it.
        sess_options.use_deterministic_compute = True
    if single_threaded:
        # Constant folding creates one throwaway session per fold-group, often
        # hundreds of times per model (once per batch of foldable nodes, per
        # fixed-point round -- see ``RunOps`` in onnxsim.cpp). Each session
        # otherwise spins up a fresh intra-op thread pool sized to the machine's
        # CPU count purely to run, and then discard, a handful of shape/index
        # ops on tiny tensors; that thread-pool spin-up/join is pure overhead
        # for graphs this small, and it is repeated at every one of those
        # session creations. Comparable to onnxsim issue observations that
        # ``OrtSessionInit`` (not the actual op execution) is usually the
        # dominant cost of a fold session. Running single-threaded skips it.
        # Not applied to the ``model_checking`` correctness-check path, which
        # runs the full (potentially large) model and can benefit from real
        # parallelism.
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
    # Optionally turn on onnxruntime's own per-operator session profiler for this
    # folding session (separate from onnxsim's span profiler; see
    # ``_ort_profile_prefix``). onnxruntime writes one Chrome trace JSON per
    # session when the session ends.
    ort_profile_prefix = _ort_profile_prefix()
    if ort_profile_prefix is not None:
        sess_options.enable_profiling = True
        # onnxruntime appends "_<timestamp>.json" to the prefix. Guard the
        # attribute: it was added in newer onnxruntime, and without it the
        # default prefix ("onnxruntime_profile_") is used instead.
        if hasattr(sess_options, "profile_file_prefix"):
            sess_options.profile_file_prefix = ort_profile_prefix
    if isinstance(model, onnx.ModelProto):
        model = model.SerializeToString()
    sess = rt.InferenceSession(
        model,
        sess_options=sess_options,
        providers=list(providers),
    )
    if output_names is None:
        output_names = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    outputs = sess.run(list(output_names), inputs, run_options=run_options)
    if ort_profile_prefix is not None:
        # Flush the per-operator trace to disk and stop profiling for this
        # session (otherwise the file is only written when the session is later
        # garbage-collected).
        sess.end_profiling()
    return OrderedDict(zip(output_names, outputs))


def _has_subgraphs(graph: onnx.GraphProto) -> bool:
    """Whether any node in ``graph`` carries a control-flow subgraph (If /
    Loop / Scan), recursing into nested subgraphs. Such a subgraph's body can
    reference an enclosing value by name at any point during its own
    execution (`OpRun.need_context`), so :func:`_run_reference_pruned`'s
    liveness analysis -- which only tracks *direct* top-level consumption --
    cannot safely drop anything early once one of these exists anywhere in
    the model; the caller falls back to the plain, always-correct
    ``ReferenceEvaluator.run``.
    """
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                return True
            if len(attr.graphs) > 0:
                return True
    return False


def _last_use_indices(graph: onnx.GraphProto) -> Dict[str, int]:
    """The index of the last top-level node that consumes each value name, as
    an input. A value with no entry is either never consumed (e.g. a graph
    output nothing downstream reads) or not produced by a node at all.
    """
    last_use: Dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for name in node.input:
            if name:
                last_use[name] = i
    return last_use


def _run_reference_pruned(
    sess: Any,  # onnx.reference.ReferenceEvaluator, imported lazily by the caller
    graph: onnx.GraphProto,
    output_names: List[str],
    feed_inputs: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """Drive ``sess`` the same way ``ReferenceEvaluator.run`` does, but drop a
    value from the live-results dict as soon as the last top-level node that
    needs it has run, instead of keeping every intermediate (and every
    initializer) alive for the whole graph -- ``ReferenceEvaluator.run``'s own
    ``results`` dict never frees anything until the call returns, so its peak
    memory is the *naive*, no-reuse total :func:`onnxsim.plan_activation_memory`
    reports as ``naive_bytes``. This gets closer to that call's
    ``arena_bytes`` bound without computing an actual offset plan: dropping a
    Python reference early just lets it get garbage-collected, no shapes or
    byte sizes needed, so this works even on models with dynamic shapes that
    :func:`onnxsim.plan_activation_memory` itself could not fully plan.

    Only called when :func:`_has_subgraphs` is False for the whole model --
    see its docstring for why a control-flow subgraph makes this unsafe.
    Mirrors the internals ``ReferenceEvaluator.run`` itself uses
    (``rt_nodes_``, ``rt_inits_``, ``need_context``,
    ``has_linked_attribute``), so it stays exact if a value is ever consumed
    outside the ways this scans for.
    """
    last_use = _last_use_indices(graph)
    keep = set(output_names)  # requested outputs must survive to the end

    results: Dict[str, Any] = {"": None}
    results.update(sess.rt_inits_)
    results.update(feed_inputs)
    for i, node in enumerate(sess.rt_nodes_):
        node_inputs = [results[name] for name in node.input]
        linked_attributes: Dict[str, Any] = {}
        if getattr(node, "has_linked_attribute", False):
            linked_attributes["linked_attributes"] = {}
        if node.need_context():
            node_outputs = node.run(*node_inputs, context=results, **linked_attributes)
        else:
            node_outputs = node.run(*node_inputs, **linked_attributes)
        for name, value in zip(node.output, node_outputs):
            results[name] = value
        for name in node.input:
            if name and name not in keep and last_use.get(name) == i:
                results.pop(name, None)

    return [results[name] for name in output_names]


def _run_with_reference(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]],
    custom_lib: Optional[str],
    providers: Optional[Sequence[Provider]] = None,
) -> "OrderedDict[str, np.ndarray]":
    if custom_lib is not None:
        raise ValueError("custom_lib is only supported when onnxruntime is installed")
    # The reference evaluator runs in pure Python on the CPU and has no notion of
    # execution providers. Asking for a non-CPU provider (e.g. CUDA) without
    # onnxruntime installed cannot be honoured, so surface that instead of
    # silently ignoring the request.
    validate_providers(providers)
    from onnx.reference import ReferenceEvaluator

    if isinstance(model, str):
        model = onnx.load(model)
    elif isinstance(model, bytes):
        model = onnx.load_from_string(model)
    sess = ReferenceEvaluator(model)
    if output_names is None:
        output_names = list(sess.output_names)
    output_names = list(output_names)
    # `model` is always a ModelProto here (the str/bytes branches above
    # normalize it), so its graph is always available for the liveness scan.
    if not _has_subgraphs(model.graph):
        outputs = _run_reference_pruned(sess, model.graph, output_names, inputs)
    else:
        # intermediate defaults to False, so this is always a list -- ReferenceEvaluator.run's
        # declared return type is the wider dict-or-list Union covering both.
        outputs = cast(List[np.ndarray], sess.run(output_names, inputs))
    return OrderedDict(zip(output_names, outputs))


def run_model(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]] = None,
    custom_lib: Optional[str] = None,
    providers: Optional[Sequence[Provider]] = None,
    single_threaded: bool = False,
    deterministic: bool = False,
) -> "OrderedDict[str, np.ndarray]":
    """Run ``model`` on ``inputs`` and return an ordered ``{name: array}`` map.

    :param model: onnx ModelProto, serialized bytes, or a file path
    :param inputs: mapping from input name to numpy array
    :param output_names: outputs to fetch, ``None`` means all model outputs
    :param custom_lib: onnxruntime custom ops's shared library (onnxruntime only)
    :param providers: onnxruntime execution providers to run with, in priority
            order (e.g. ``["CUDAExecutionProvider", "CPUExecutionProvider"]``).
            ``None`` means CPU only. Non-CPU providers require onnxruntime.
    :param single_threaded: Run the onnxruntime session with a single intra-/
            inter-op thread instead of onnxruntime's default (one per CPU core).
            Used by constant folding, which creates many small throwaway
            sessions where thread-pool spin-up dwarfs the tiny amount of actual
            work; leave this ``False`` for a full-size model. Ignored by the
            pure-Python reference-evaluator fallback (no onnxruntime installed),
            which has no thread pool to configure.
    :param deterministic: Set onnxruntime's ``use_deterministic_compute``
            session option, which steers kernels (CPU, CUDA, ROCM) away from
            any algorithm whose numerical result depends on the host's SIMD
            capabilities (e.g. AVX-512 vs. AVX2 vs. SSE) rather than the model
            and inputs alone -- at some performance cost. Used by
            :func:`onnxsim.measure_accuracy_drop`, a *measurement* that should
            reproduce across hosts; combine with ``single_threaded=True`` to
            also remove thread-partitioning as a source of divergence. Ignored
            by the pure-Python reference-evaluator fallback (already fully
            deterministic, no SIMD dispatch of its own).
    """
    if _HAS_ONNXRUNTIME:
        return _run_with_onnxruntime(
            model,
            inputs,
            output_names,
            custom_lib,
            providers,
            single_threaded,
            deterministic,
        )
    return _run_with_reference(model, inputs, output_names, custom_lib, providers)


# Which onnxruntime ``OrtValue`` device an execution provider keeps its tensors
# on, for :meth:`Runner.bind_loop`. The strings are the ones onnxruntime's own
# ``get_ort_device_type`` accepts; a provider missing from this table is bound
# on the CPU, which is always *correct* -- onnxruntime then inserts the same
# host-to-device copy the unbound path pays -- just not always the fastest
# place. The ROCm/MIGraphX entries reuse the CUDA device enum, which is what
# onnxruntime's ROCm build itself does; if that guess is ever wrong the
# allocation raises and :meth:`Runner.bind_loop` degrades to the unbound path,
# so it cannot produce a wrong answer.
_PROVIDER_DEVICES: Dict[str, str] = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "cuda",
    "ROCMExecutionProvider": "cuda",
    "MIGraphXExecutionProvider": "cuda",
    "CANNExecutionProvider": "cann",
    "DmlExecutionProvider": "dml",
    "WebGpuExecutionProvider": "webgpu",
}

# onnxruntime reports an output's type as a string like ``"tensor(float)"``.
# Only tensor element types with a numpy equivalent can be pre-allocated as a
# bound output buffer; anything else (a sequence, a map, a type numpy has no
# dtype for) makes :meth:`Runner.bind_loop` decline.
_ORT_TYPE_TO_NUMPY: Dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _binding_device(providers: Optional[Sequence[Provider]]) -> Tuple[str, int]:
    """The ``(device_type, device_id)`` to allocate bound tensors on for
    ``providers``.

    The *first* provider decides: onnxruntime tries the list in priority order
    and only falls back for operators the leading provider cannot run, so its
    device is where a step graph's tensors want to live. An unrecognized
    provider, or one whose options do not name a device, gets ``("cpu", 0)`` --
    see :data:`_PROVIDER_DEVICES` for why that is safe.
    """
    if not providers:
        return "cpu", 0
    provider = providers[0]
    device = _PROVIDER_DEVICES.get(_provider_name(provider), "cpu")
    device_id = 0
    if isinstance(provider, (tuple, list)) and len(provider) > 1:
        options = provider[1]
        if isinstance(options, dict):
            requested = options.get("device_id", 0)
            if isinstance(requested, (int, str)):
                try:
                    device_id = int(requested)
                except ValueError:
                    device_id = 0
    return device, device_id


def _static_output_spec(meta: Any) -> Optional[Tuple[List[int], Any]]:
    """``(shape, numpy dtype)`` for an onnxruntime output whose buffer can be
    allocated up front, or ``None`` when it cannot.

    A bound output needs a buffer before the run that fills it, so every
    dimension has to be a concrete integer -- onnxruntime reports a symbolic or
    unknown dimension as a string or ``None`` -- and the element type has to be
    one numpy can hold. Returning ``None`` is not an error: it is how
    :meth:`Runner.bind_loop` decides that this model is not one it can bind,
    and the caller keeps using the ordinary feed-per-call path.
    """
    ort_type = getattr(meta, "type", None)
    if not isinstance(ort_type, str):
        return None
    dtype = _ORT_TYPE_TO_NUMPY.get(ort_type)
    if dtype is None:
        return None
    shape = getattr(meta, "shape", None)
    if shape is None:
        return None
    dims: List[int] = []
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            return None
        dims.append(dim)
    return dims, dtype


class Runner:
    """A model prepared once and run many times.

    :func:`run_model` creates a fresh session per call, which is right for
    constant folding (every fold group is a different throwaway sub-model) and
    wrong for an optimization loop, where the *same* small graph is run
    hundreds of times with different tensor values -- there, session creation
    would dominate and ``enable_mem_pattern``'s cross-run buffer planning,
    disabled in :func:`run_model` because it can never pay for itself in a
    single ``Run()``, is exactly what should be on.

    This is the Python counterpart of the converter page's own
    ``makeOrtRunner`` (``scripts/convertmodel/ort_executor.mjs``): bind a model
    and an execution-provider list once, then call it per step. See
    :mod:`onnxsim.qat_graph` for the loop it exists for.
    """

    def __init__(
        self,
        model: Union[str, bytes, onnx.ModelProto],
        output_names: Optional[Sequence[str]] = None,
        providers: Optional[Sequence[Provider]] = None,
    ) -> None:
        self._providers = providers
        if _HAS_ONNXRUNTIME:
            if providers is None:
                providers = DEFAULT_PROVIDERS
            validate_providers(providers)
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
            sess_options.log_severity_level = 3
            if isinstance(model, onnx.ModelProto):
                model = model.SerializeToString()
            self._sess: Any = rt.InferenceSession(
                model, sess_options=sess_options, providers=list(providers)
            )
            self._output_names = list(
                output_names
                if output_names is not None
                else [o.name for o in self._sess.get_outputs()]
            )
            self._run_options = rt.RunOptions()
            self._run_options.log_severity_level = 3
            self._graph = None
        else:
            validate_providers(providers)
            from onnx.reference import ReferenceEvaluator

            if isinstance(model, str):
                model = onnx.load(model)
            elif isinstance(model, bytes):
                model = onnx.load_from_string(model)
            self._sess = ReferenceEvaluator(model)
            self._output_names = list(
                output_names
                if output_names is not None
                else list(self._sess.output_names)
            )
            self._graph = None if _has_subgraphs(model.graph) else model.graph

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    def __call__(self, inputs: Dict[str, np.ndarray]) -> "OrderedDict[str, np.ndarray]":
        if _HAS_ONNXRUNTIME:
            outputs = self._sess.run(
                self._output_names, inputs, run_options=self._run_options
            )
        elif self._graph is not None:
            outputs = _run_reference_pruned(
                self._sess, self._graph, self._output_names, inputs
            )
        else:
            outputs = cast(List[np.ndarray], self._sess.run(self._output_names, inputs))
        return OrderedDict(zip(self._output_names, outputs))

    def bind_loop(
        self,
        constants: Dict[str, np.ndarray],
        state: Dict[str, Tuple[str, np.ndarray]],
    ) -> Optional["BoundStepLoop"]:
        """Prepare a device-resident, state-threading loop over this model, or
        return ``None`` if this backend cannot provide one.

        :meth:`__call__` re-sends every input on every call, because that is
        what ``InferenceSession.run`` takes: a fresh ``{name: numpy array}``
        dict. For an optimization loop that is the wrong shape of API. The
        constants (calibration activations, a reconstruction target) never
        change, and the state (the parameter being trained, Adam's moments) is
        produced by the previous step -- so on a non-CPU provider every step
        pays a host-to-device copy of *everything*, plus a device-to-host copy
        back, for tensors that never needed to leave the device. Over a few
        hundred steps that transfer, not the arithmetic, is the loop.

        ``IOBinding`` is onnxruntime's answer: bind a tensor once, run against
        the binding. This method uploads the constants once, allocates the
        state on the session's device, and hands back a
        :class:`BoundStepLoop` that runs step after step with only the
        per-step scalars going up and only the explicitly fetched outputs
        coming down.

        :param constants: inputs whose value is the same on every step. Copied
                to the device once. The copy is defensive: a caller's array
                would otherwise stay aliased by the binding for the loop's
                whole life.
        :param state: ``{input name: (output name, initial value)}`` -- the
                inputs each step re-supplies from the previous step's output.
                The output is required to have the same shape and dtype as the
                input it feeds; that is the step-graph contract
                (:func:`onnxsim.qat_graph.make_step_graph` builds exactly this)
                and it is verified here against the session's own metadata.

        Returns ``None`` -- meaning "run the ordinary way" -- whenever binding
        cannot be set up: no onnxruntime (the reference evaluator has no
        binding and no devices), an onnxruntime too old for ``io_binding``, an
        output whose buffer cannot be pre-allocated (dynamic shape, or a type
        numpy cannot hold), a shape that contradicts the model, or an
        allocation the provider refuses. Binding is a pure optimization, so
        every one of those is a reason to fall back rather than to fail: the
        caller gets the same numbers either way.
        """
        if not _HAS_ONNXRUNTIME or not hasattr(self._sess, "io_binding"):
            return None
        # Everything below is best-effort. A broad except is deliberate here:
        # any failure at all -- an unavailable device string, an allocator that
        # refuses, an onnxruntime whose binding API differs -- must land the
        # caller on the unbound path, never on a traceback, because binding
        # changes nothing about the answer.
        try:
            device_type, device_id = _binding_device(self._providers)
            metadata = {o.name: o for o in self._sess.get_outputs()}
            state_outputs = {name: out for name, (out, _) in state.items()}
            produced = set(state_outputs.values())
            if not produced <= set(metadata):
                return None

            buffers: Dict[str, List[Any]] = {}
            for name, (out, value) in state.items():
                array = np.ascontiguousarray(value)
                spec = _static_output_spec(metadata[out])
                if spec is None:
                    return None
                shape, dtype = spec
                if tuple(shape) != array.shape or np.dtype(dtype) != array.dtype:
                    return None
                # Two buffers per state tensor, alternated by
                # :meth:`BoundStepLoop.step` -- see its comment for why one
                # would not be safe. Both are allocated by onnxruntime rather
                # than wrapped around the caller's numpy array: on the CPU
                # ``ortvalue_from_numpy`` aliases the array it is given, so a
                # single buffer would make the caller's own initial state the
                # loop's scratch space.
                pair = [
                    rt.OrtValue.ortvalue_from_shape_and_type(
                        list(array.shape), dtype, device_type, device_id
                    )
                    for _ in range(2)
                ]
                # The one copy of the state the loop pays: the initial value
                # into the buffer onnxruntime allocated for it.
                pair[0].update_inplace(array)
                buffers[name] = pair

            binding = self._sess.io_binding()
            # onnxruntime requires *every* model output to be bound before
            # ``run_with_iobinding``, not only the ones this Runner fetches, so
            # each non-state output gets a host buffer whether the caller reads
            # it or not.
            host_outputs: "OrderedDict[str, Any]" = OrderedDict()
            for name, meta in metadata.items():
                if name in produced:
                    continue
                spec = _static_output_spec(meta)
                if spec is None:
                    return None
                shape, dtype = spec
                value = rt.OrtValue.ortvalue_from_shape_and_type(shape, dtype, "cpu", 0)
                host_outputs[name] = value
                binding.bind_ortvalue_output(name, value)

            # The constants: copied to the device once, bound once, never
            # touched again. This is the whole point of the exercise.
            resident: List[Any] = []
            for name, value in constants.items():
                array = np.array(value, copy=True, order="C")
                ort_value = rt.OrtValue.ortvalue_from_numpy(
                    array, device_type, device_id
                )
                resident.append(ort_value)
                binding.bind_ortvalue_input(name, ort_value)

            return BoundStepLoop(
                self._sess,
                self._run_options,
                binding,
                device_type,
                buffers,
                state_outputs,
                host_outputs,
                resident,
            )
        except Exception:
            return None


class BoundStepLoop:
    """One step of a state-threading loop, run against onnxruntime tensors that
    stay where the execution provider put them.

    Created by :meth:`Runner.bind_loop`, never directly. The loop it serves is
    the one :func:`onnxsim.qat_graph.run_step_graph` runs: a pure
    ``(constants, state, per-step scalars) -> (next state, loss)`` function
    applied over and over, each step's state outputs becoming the next step's
    state inputs. Expressed through ``InferenceSession.run`` that costs a full
    round trip of the state per step; expressed through ``IOBinding`` the state
    never leaves the device at all, and what crosses the bus is a handful of
    scalars up and (only if the caller asked for it) a scalar loss down.

    The buffers are owned here and reused, so an instance is single-threaded
    and stateful by construction: :meth:`step` advances it, :meth:`state` reads
    where it got to.
    """

    def __init__(
        self,
        sess: Any,
        run_options: Any,
        binding: Any,
        device_type: str,
        buffers: Dict[str, List[Any]],
        state_outputs: Dict[str, str],
        host_outputs: "OrderedDict[str, Any]",
        resident: List[Any],
    ) -> None:
        self._sess = sess
        self._run_options = run_options
        self._binding = binding
        self._device_type = device_type
        self._buffers = buffers
        self._state_outputs = state_outputs
        self._host_outputs = host_outputs
        # Held only to keep the constants' device memory (and, on the CPU, the
        # numpy arrays onnxruntime's OrtValues alias rather than copy) alive
        # for as long as the binding refers to it.
        self._resident = resident
        self._slot = 0

    def step(self, scalars: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one step and return the host-side outputs.

        ``scalars`` are the only inputs that go up per step: they are bound
        from host memory each time (they are rank-0, so the transfer is
        nothing), while the constants stay bound from :meth:`Runner.bind_loop`
        and the state is bound from the buffers below.
        """
        for name, value in scalars.items():
            self._binding.bind_cpu_input(name, value)

        # Ping-pong: read this step's state out of one buffer, write the next
        # state into the other, then swap. The single-buffer version -- bind
        # the same OrtValue as both the state input and the state output --
        # looks natural and is a correctness trap: onnxruntime is free to begin
        # writing an output before it has finished reading every input, and
        # free to hand a bound output buffer straight back to the next
        # ``run_with_iobinding`` call, so a step could overwrite the very
        # values it is still computing from. Two buffers make each run's read
        # set and write set disjoint, which costs one extra tensor per state
        # entry and removes the question entirely.
        source, target = self._slot, 1 - self._slot
        for name, output in self._state_outputs.items():
            self._binding.bind_ortvalue_input(name, self._buffers[name][source])
            self._binding.bind_ortvalue_output(output, self._buffers[name][target])

        if self._device_type != "cpu":
            # On a real device the copies onnxruntime issues for the bound
            # inputs and outputs are asynchronous; on the CPU there is nothing
            # to wait for, so skip the call rather than pay it 400 times.
            self._binding.synchronize_inputs()
        self._sess.run_with_iobinding(self._binding, self._run_options)
        if self._device_type != "cpu":
            self._binding.synchronize_outputs()
        self._slot = target

        # The host buffers are reused across steps, so the caller sees this
        # step's values only until the next one -- which is all
        # ``run_step_graph`` needs (it turns the loss into a float immediately).
        return {name: value.numpy() for name, value in self._host_outputs.items()}

    def state(self) -> "OrderedDict[str, np.ndarray]":
        """The current state, copied back to the host as numpy arrays.

        Copied, not aliased: the buffers behind it are this loop's own scratch
        space and the next :meth:`step` would write through them.
        """
        return OrderedDict(
            (name, self._buffers[name][self._slot].numpy().copy())
            for name in self._state_outputs
        )
