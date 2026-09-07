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
