"""Export a Hugging Face ``transformers`` model straight to a simplified
ONNX deployment directory.

onnxsim has no PyTorch tracing code of its own, and does not need any: turning
a ``transformers`` model into an ONNX graph is exactly what Hugging Face's own
``optimum`` package already does, for every architecture that has an
``optimum.exporters.onnx`` ``OnnxConfig`` (hundreds of model types, including
the split multi-file encoder/decoder-with-past shape autoregressive
generation needs -- see ``tests/test_optimum_export_deploy.py``). That export
is deliberately plain -- no runtime-specific op fusion -- so there is real
simplification left on the table for onnxsim's own pipeline to find.

This is a different tool for a different job than ONNX Runtime GenAI's own
model builder (``onnxruntime_genai.models.builder``): that one covers a fixed,
curated list of decoder-only causal-LM architectures, and its output is
already fused/quantized into ORT-specific ops (e.g. ``com.microsoft::
MatMulNBits``, ``GroupQueryAttention``) meant to be consumed directly by ORT
GenAI's own generate() loop -- there is little left for a generic simplifier
to do to it, and it does not cover encoder-only, seq2seq, vision, or audio
architectures at all. ``optimum``'s export is the right shape for *this*
job instead: a plain graph, for any architecture with an ``OnnxConfig``,
handed to onnxsim to clean up.

:func:`export_transformers_model` wraps exactly the manual
export-then-simplify-then-copy-the-rest recipe
``tests/test_optimum_export_deploy.py`` exercises by hand, as a reusable
onnxsim entry point.
"""

import glob
import os
from typing import Dict, Optional, Set

import onnx
from google.protobuf.message import EncodeError
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from onnxsim.model_info import _iter_graph_tensors
from onnxsim.onnx_simplifier import simplify


def export_transformers_model(
    model_id: str,
    output_dir: str,
    task: str = "auto",
    no_post_process: bool = True,
    check_n: int = 0,
    save_as_external_data: bool = True,
    export_kwargs: Optional[Dict] = None,
    simplify_kwargs: Optional[Dict] = None,
) -> Dict[str, bool]:
    """Export ``model_id`` (a Hugging Face Hub id or local model directory) to
    ONNX via ``optimum.exporters.onnx.main_export``, then simplify every
    ``.onnx`` file it produces, in place, inside ``output_dir``.

    Needs the optional ``torch``, ``transformers``, and ``optimum`` (with the
    ``optimum-onnx`` distribution installed for ``optimum.exporters.onnx``)
    packages -- heavy, and unrelated to onnxsim's own ONNX-to-ONNX pipeline,
    so they are not normal onnxsim dependencies
    (``pip install onnxsim[transformers]``).

    :param model_id: Hugging Face Hub model id or local model directory
    :param output_dir: directory to export into. Also where the simplified
            files end up: each exported ``.onnx`` file is overwritten in
            place with its simplified version. Non-``.onnx`` files (tokenizer,
            config, ...) are left untouched, so the directory stays
            deployable exactly like a plain ``optimum`` export.
    :param task: the export task, e.g. ``"text-generation-with-past"`` or
            ``"text2text-generation-with-past"``; ``"auto"`` (the default)
            lets ``optimum`` infer it from the model's config.
    :param no_post_process: keep a multi-file encoder/decoder(-with-past)
            export split rather than merged into a single graph with an
            ``If``-node branch switch. Defaults to ``True``: as of this
            writing, simplifying a merged decoder produces a model that
            fails at runtime (see ``tests/test_optimum_export_deploy.py``'s
            docstring for the specific failure) -- the split shape simplifies
            and reloads correctly, and ``optimum``'s own runtime classes
            (e.g. ``ORTModelForSeq2SeqLM``) fall back to it automatically
            when no merged file is present.
    :param check_n: forwarded to :func:`onnxsim.simplify` for every exported
            graph -- how many random-input runs to check the simplified
            model against the freshly exported one for numerical equivalence.
    :param save_as_external_data: always save every simplified graph with its
            weights in a companion ``<filename>.data`` file, instead of
            inline. On by default here -- unlike the ``onnxsim`` CLI's own
            ``--save-as-external-data``/plain ``onnx.save``, which default off
            and only use external data as a fallback once a graph is too
            large to serialize inline at all (>2GB) -- because a real
            (non-tiny) transformers with-past export is the common case this
            function exists for, and it is multiple *independent* graphs
            (encoder/decoder/decoder-with-past, see ``no_post_process``
            above), each embedding its own full inline copy of whatever
            weights it uses -- e.g. the decoder's weights end up duplicated
            across ``decoder_model.onnx`` and ``decoder_with_past_model.onnx``
            -- and every pass in onnxsim's own optimization pipeline that
            touches the graph (shape inference, checker, each fixed-point
            round) copies those inline bytes along with it. External data
            keeps the large tensors on disk instead, so this repeated
            in-memory copying and the inline duplication across split files
            both shrink to metadata (name/offset/length) rather than the
            tensors themselves. Pass ``False`` to keep small/tiny models
            (tests, toy checkpoints) as a single self-contained ``.onnx``
            file with no companion ``.data``.
    :param export_kwargs: extra keyword arguments forwarded to
            ``optimum.exporters.onnx.main_export`` (e.g. ``opset``,
            ``device``, ``fp16``, ``trust_remote_code``).
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify` for every exported graph.
    :returns: ``{filename: check_ok}`` for every ``.onnx`` file exported,
            where ``check_ok`` is that file's :func:`onnxsim.simplify`
            numerical-equivalence check result (always ``True`` when
            ``check_n == 0``, since no check is performed).
    """
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:
        raise ImportError(
            "export_transformers_model needs the optional 'torch', "
            "'transformers', and 'optimum' (with the 'optimum-onnx' "
            "distribution) packages: pip install onnxsim[transformers]"
        ) from e

    main_export(
        model_id,
        output=output_dir,
        task=task,
        no_post_process=no_post_process,
        **(export_kwargs or {}),
    )

    results = {}
    for src in sorted(glob.glob(os.path.join(output_dir, "*.onnx"))):
        model_opt, check_ok = simplify(src, check_n=check_n, **(simplify_kwargs or {}))
        _save(model_opt, src, force_external_data=save_as_external_data)
        results[os.path.basename(src)] = check_ok
    return results


def _referenced_external_data_files(path: str) -> Set[str]:
    """Every external-data file the on-disk model at ``path`` currently points
    to, as absolute paths -- without loading the tensor data itself.

    Used by :func:`_save` to find files that become orphaned once it
    overwrites ``path``: the export this wraps (``optimum``/PyTorch's own
    ONNX exporter) picks its own external-data filenames (e.g. a UNet's
    multi-gigabyte weights land next to it as ``model.onnx_data``), which
    don't match the ``<filename>.data`` convention :func:`_save` writes under.
    Left alone, every simplified graph would carry its old, now-unreferenced
    weights file forward as dead weight -- for a real (non-tiny) model,
    gigabytes of it per graph.
    """
    try:
        raw = onnx.load(path, load_external_data=False)
    except Exception:
        return set()
    base_dir = os.path.dirname(path)
    files = set()
    for tensor in _iter_graph_tensors(raw.graph):
        if uses_external_data(tensor):
            location = ExternalDataInfo(tensor).location
            if location:
                files.add(os.path.normpath(os.path.join(base_dir, location)))
    return files


def _save(model: onnx.ModelProto, path: str, force_external_data: bool) -> None:
    # Snapshot what the pre-simplification file on disk at `path` points to
    # *before* overwriting it, so any of those files no longer referenced
    # afterwards can be cleaned up as stale rather than left behind.
    stale_candidates = _referenced_external_data_files(path)

    if not force_external_data:
        try:
            onnx.save(model, path)
            _cleanup_stale_external_data(path, stale_candidates)
            return
        except (ValueError, EncodeError):
            # Real transformers models routinely exceed onnx.save's 2GB inline
            # limit; fall back to external data next, matching the CLI's own
            # --save-as-external-data fallback (onnx_simplifier.py).
            pass

    external_data_path = os.path.basename(path) + ".data"
    full_external_data_path = os.path.join(os.path.dirname(path), external_data_path)
    if os.path.exists(full_external_data_path):
        os.remove(full_external_data_path)
    onnx.save(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path,
        # Always keep onnx's own default (1024) tensor-size cutoff, even
        # under force_external_data -- do NOT drop it to 0. A real
        # transformers export bakes in many tiny int64 scalar constants
        # (shape/index helpers); externalizing those too produces a file
        # onnxruntime fails to reload ("Cannot parse data from external
        # tensors" on one such scalar), even though plain onnx.load +
        # checker.check_model call it valid (confirmed empirically -- see
        # the identical workaround/comment on export_causal_lm_static_cache's
        # own onnx.save call below, which hit the same bug and avoids _save()
        # entirely for that reason). Tensors this small contribute nothing
        # worth saving to the in-pipeline-copying problem force_external_data
        # exists for anyway, so there is no tradeoff in leaving them inline.
        size_threshold=1024,
    )
    _cleanup_stale_external_data(path, stale_candidates)


def _cleanup_stale_external_data(path: str, stale_candidates: Set[str]) -> None:
    for stale in stale_candidates - _referenced_external_data_files(path):
        if os.path.exists(stale):
            os.remove(stale)


# --------------------------------------------------------------------------- #
# Fixed-buffer ("static") KV cache export, decoder-only causal LMs only.
# --------------------------------------------------------------------------- #


def export_causal_lm_static_cache(
    model_id: str,
    output_dir: str,
    max_cache_len: int,
    max_prompt_len: Optional[int] = None,
    check_n: int = 0,
    save_as_external_data: bool = True,
    model_kwargs: Optional[Dict] = None,
    simplify_kwargs: Optional[Dict] = None,
) -> Dict[str, bool]:
    """Export a decoder-only causal-LM ``transformers`` model to a *fixed*-size
    ("static") KV cache ONNX pair -- ``prefill.onnx`` (variable prompt length,
    empty cache) and ``decode.onnx`` (one new token, cache already partly
    filled) -- then simplify both.

    This is a different shape than :func:`export_transformers_model`'s
    ``optimum``-based export, on purpose. ``optimum``'s
    ``decoder_with_past_model.onnx`` grows its ``past_key_values``/``present``
    tensors by one position every decode step (a `Concat`), which means every
    single-token decode step reallocates and copies the *entire* cache seen
    so far -- O(n) work per step, O(n^2) over a full generation. The ONNX
    ``TensorScatter`` op (opset 24) exists specifically to let backends avoid
    that: write each new token's K/V into a pre-allocated
    ``(batch, heads, max_cache_len, head_dim)`` buffer in place, O(1) per
    step. But ``optimum``'s own export can't be rewritten into that shape
    after the fact -- HF's modern causal-mask construction
    (``transformers.masking_utils``) derives the mask's own size from the
    cache's *real* valid length, which is exactly the cache's *tensor* length
    for a growing cache, but silently wrong once the tensor is a fixed-size
    buffer with a smaller amount of real content (confirmed empirically: a
    post-hoc ``Concat``->``TensorScatter`` graph rewrite hits an ONNX Runtime
    broadcast error deep in the mask math).

    The fix is to control the export instead of patching its output:
    ``transformers.StaticCache`` (built for ``torch.compile``/``torch.export``)
    already allocates fixed-size buffers and already gets the mask math right
    -- its ``get_mask_sizes()`` reports the *buffer's* length, not the real
    valid count, which is exactly the missing piece. Tracing a
    ``StaticCache``-based forward pass turns its ``index_copy_`` cache update
    into ``ScatterND`` (not ``TensorScatter`` -- PyTorch's exporter doesn't
    know about the new op, and doesn't need to: ``ScatterND`` is the same
    "overwrite, no reduction" semantics and already runs everywhere). Verified
    against the standard ``optimum`` export: identical greedy-decoded tokens
    over a multi-step generation, via a real ``onnxruntime.InferenceSession``
    for both.

    Needs the optional ``torch`` and ``transformers`` packages (``pip install
    onnxsim[transformers]``) -- unlike :func:`export_transformers_model`, this
    does NOT go through ``optimum``: ``StaticCache``-based tracing is driven
    directly against the ``transformers`` model here, since integrating a
    custom fixed-buffer cache into ``optimum.exporters.onnx``'s own
    per-architecture ``OnnxConfig``/dummy-input-generator machinery is a much
    larger, more architecture-specific undertaking than tracing the model
    directly.

    Scope: decoder-only causal LMs with **uniform, full (non-sliding-window,
    non-hybrid) attention across every layer** -- this wrapper feeds a single
    ``max_cache_len`` to every layer via ``StaticCache``, which allocates a
    (possibly different) ``StaticSlidingWindowLayer`` per layer for hybrid
    architectures (e.g. Gemma 2/3's alternating local/global attention); this
    function does not attempt to distinguish or size those layers separately.
    Encoder-decoder/seq2seq models remain :func:`export_transformers_model`'s
    job -- their cross-attention KV cache doesn't grow at all (computed once
    from the encoder), so it never had this problem in the first place.

    :param model_id: Hugging Face Hub model id or local model directory.
    :param output_dir: directory to export into (``prefill.onnx``,
            ``decode.onnx``, plus tokenizer/config files).
    :param max_cache_len: fixed KV-cache buffer length. Must be at least as
            large as the longest prompt plus the number of tokens you intend
            to generate -- there is no bounds checking at runtime (writing
            past the end wraps around via ``TensorScatter``/``ScatterND``'s
            own modulo-free overwrite semantics, silently corrupting the
            cache instead of erroring).
    :param max_prompt_len: largest prompt length ``prefill.onnx`` accepts
            (a dynamic axis up to this bound). Defaults to ``max_cache_len``.
    :param check_n: forwarded to :func:`onnxsim.simplify` for both exported
            graphs.
    :param save_as_external_data: see :func:`export_transformers_model`.
    :param model_kwargs: extra keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained``.
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify` for both exported graphs.
    :returns: ``{"prefill.onnx": check_ok, "decode.onnx": check_ok}``.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    except ImportError as e:
        raise ImportError(
            "export_causal_lm_static_cache needs the optional 'torch' and "
            "'transformers' packages: pip install onnxsim[transformers]"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    max_prompt_len = max_prompt_len or max_cache_len

    model = AutoModelForCausalLM.from_pretrained(model_id, **(model_kwargs or {}))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = model.config.get_text_config(decoder=True)
    num_layers = config.num_hidden_layers
    num_kv_heads = (
        getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    )
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )

    class _StaticCacheStep(torch.nn.Module):
        """One forward step against explicit, plain-tensor KV-cache buffers
        (``StaticCache`` itself is stateful and Python-object-shaped, not
        something ``torch.export`` can take as an input/output directly --
        this wraps one around plain tensors for tracing, the same technique
        ``onnxsim``'s own dlpack/tensor_pool bridges use to cross a
        stateless-graph/stateful-object boundary elsewhere in this repo)."""

        def __init__(self, model, num_layers):
            super().__init__()
            self.model = model
            self.num_layers = num_layers

        def forward(self, input_ids, cache_position, attention_mask, *flat_kv):
            cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
            for i in range(self.num_layers):
                cache.layers[i].keys = flat_kv[2 * i]
                cache.layers[i].values = flat_kv[2 * i + 1]
                cache.layers[i].is_initialized = True
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=True,
            )
            new_kv = []
            for i in range(self.num_layers):
                new_kv.append(cache.layers[i].keys)
                new_kv.append(cache.layers[i].values)
            return (out.logits, *new_kv)

    wrapper = _StaticCacheStep(model, num_layers)
    dummy_kv = [
        torch.zeros(1, num_kv_heads, max_cache_len, head_dim, dtype=model.dtype)
        for _ in range(2 * num_layers)
    ]
    kv_input_names = []
    kv_output_names = []
    for i in range(num_layers):
        kv_input_names += [f"past_key.{i}", f"past_value.{i}"]
        kv_output_names += [f"present_key.{i}", f"present_value.{i}"]

    def _export(seq_len_dim, dummy_seq_len, cache_start, filename):
        input_ids = torch.zeros(1, dummy_seq_len, dtype=torch.long)
        cache_position = torch.arange(cache_start, cache_start + dummy_seq_len)
        attention_mask = torch.zeros(1, max_cache_len, dtype=torch.long)
        attention_mask[:, : cache_start + dummy_seq_len] = 1
        dynamic_shapes = (
            ({1: seq_len_dim} if seq_len_dim is not None else None),
            ({0: seq_len_dim} if seq_len_dim is not None else None),
            None,
            tuple([None] * len(dummy_kv)),
        )
        path = os.path.join(output_dir, filename)
        # torch.onnx.export's own dynamic_shapes handling (dynamo=True,
        # passed a plain nn.Module) mishandles a trailing *args group -- it
        # reports a spurious tuple/list structural mismatch that a direct
        # torch.export.export call with the exact same dynamic_shapes does
        # NOT hit (confirmed with a minimal repro). Route around it: call
        # torch.export.export ourselves first, then hand the already-traced
        # ExportedProgram to torch.onnx.export, which skips its own
        # (buggy, for this shape) capture/dynamic_shapes path entirely.
        exported_program = torch.export.export(
            wrapper,
            (input_ids, cache_position, attention_mask, *dummy_kv),
            dynamic_shapes=dynamic_shapes,
        )
        # The helper exports in memory, records each graph I/O's pytree path
        # (args/kwargs and output nesting) as metadata, then saves to ``path``.
        from onnxsim.pytree_io import export_exported_program_to_onnx

        export_exported_program_to_onnx(
            exported_program,
            path,
            input_names=["input_ids", "cache_position", "attention_mask"]
            + kv_input_names,
            output_names=["logits"] + kv_output_names,
            opset_version=18,
        )
        return path

    prompt_len_dim = torch.export.Dim("prompt_len", min=1, max=max_prompt_len)
    # dummy_seq_len must be > 1 here despite prompt_len_dim marking it
    # dynamic: some model code branches on seq_len == 1 vs > 1 internally
    # (the single-new-token decode step vs. a real multi-token prompt), and
    # tracing with a length-1 dummy specializes to that branch regardless of
    # the Dim annotation (confirmed empirically -- torch.export then reports
    # a "you marked prompt_len as dynamic but your code specialized it to a
    # constant" ConstraintViolationError). A representative multi-token
    # dummy avoids hitting that branch during tracing.
    dummy_prompt_len = min(4, max_prompt_len) if max_prompt_len > 1 else 1
    prefill_path = _export(
        prompt_len_dim,
        dummy_seq_len=dummy_prompt_len,
        cache_start=0,
        filename="prefill.onnx",
    )
    decode_path = _export(None, dummy_seq_len=1, cache_start=1, filename="decode.onnx")

    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)

    # prefill.onnx's input_ids/cache_position have a dynamic (prompt-length)
    # axis; onnxsim's own random-input equivalence check (check_n > 0) can't
    # pick a concrete size for that on its own. decode.onnx is fully static
    # and needs no such hint.
    per_file_test_input_shapes = {
        prefill_path: {
            "input_ids": [1, dummy_prompt_len],
            "cache_position": [dummy_prompt_len],
        },
        decode_path: None,
    }

    results = {}
    for path in (prefill_path, decode_path):
        loaded = onnx.load(path)
        extra_kwargs = dict(simplify_kwargs or {})
        if check_n and per_file_test_input_shapes[path] is not None:
            extra_kwargs.setdefault(
                "test_input_shapes", per_file_test_input_shapes[path]
            )
        model_opt, check_ok = simplify(loaded, check_n=check_n, **extra_kwargs)
        # Not _save(): its force_external_data path uses size_threshold=0
        # (every tensor moves out, including small helper constants), which
        # onnxsim's other exporters never trip over but the dynamo/
        # torch.export path here does -- it bakes in many tiny int64 scalar
        # constants (shape/index helpers), and pushing those to external
        # data too produces a file onnxruntime fails to load ("Cannot parse
        # data from external tensors" on one such scalar), even though plain
        # onnx.load + checker.check_model call it valid (confirmed
        # empirically). The default threshold (1024 bytes) keeps those small
        # constants inline and only moves the real weight tensors out,
        # which loads fine.
        if save_as_external_data:
            external_data_path = os.path.basename(path) + ".data"
            full_external_data_path = os.path.join(
                os.path.dirname(path), external_data_path
            )
            if os.path.exists(full_external_data_path):
                os.remove(full_external_data_path)
            onnx.save(
                model_opt,
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path,
            )
        else:
            onnx.save(model_opt, path)
        results[os.path.basename(path)] = check_ok
    return results
