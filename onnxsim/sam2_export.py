"""Export a Meta `SAM 2 <https://github.com/facebookresearch/sam2>`_
(Segment Anything 2) image model straight to a pair of simplified ONNX
files: an image encoder and a combined prompt-encoder/mask-decoder.

Like Detectron2 (see :mod:`onnxsim.detectron_export`), SAM 2 has no
``optimum``-style single-call ONNX exporter of its own, and its
``forward()`` isn't directly traceable end-to-end -- image embedding and
prompt-driven mask decoding are two separate stages meant to be called many
times per embedding (one embedding, many point/box prompts), so the
established recipe (used by Meta's own original SAM ``scripts/
export_onnx_model.py`` and carried over to SAM 2 by the community) is two
separate traced graphs: an image encoder (plain image in, embeddings out)
and a decoder wrapping the prompt encoder plus mask decoder (embeddings and
point/box/mask prompts in, masks out).

`samexporter <https://github.com/vietanhdev/samexporter>`_ already
implements and maintains that split for SAM/SAM2/SAM3 as
``SAM2ImageEncoder``/``SAM2ImageDecoder`` -- notably, its own CLI already
calls :func:`onnxsim.simplify` under its ``--simplify`` flag, so onnxsim
already sits downstream of it for real users; :func:`export_sam2_model`
just wraps that same build-model/trace/export sequence as a reusable
onnxsim entry point that always simplifies, the SAM 2 counterpart of
:func:`onnxsim.export_transformers_model` /
:func:`onnxsim.export_diffusion_model` / :func:`onnxsim.export_detectron_model`.

The real ``facebookresearch/sam2`` package is not published to PyPI under
its own name -- see :func:`export_sam2_model`'s docstring for install
instructions. A same-named ``sam2`` package that *is* on PyPI is an
unrelated, unofficial third-party upload (a different GitHub fork under a
different author) as of this writing, not Meta's own code -- this module
deliberately never suggests installing it.
"""

import io
import os
from typing import Dict, Optional, Tuple

import onnx

from onnxsim.onnx_simplifier import simplify
from onnxsim.transformers_export import _save


def export_sam2_model(
    model_type: str,
    output_dir: str,
    checkpoint: Optional[str] = None,
    opset_version: int = 18,
    multimask_output: bool = True,
    input_size: Tuple[int, int] = (1024, 1024),
    check_n: int = 0,
    save_as_external_data: bool = False,
    simplify_kwargs: Optional[Dict] = None,
) -> Dict[str, bool]:
    """Build a SAM 2 model of type ``model_type``, trace its image encoder
    and prompt/mask decoder to ONNX as two separate graphs, then simplify
    each and save them to ``output_dir``.

    Needs the optional ``torch``, ``samexporter``, and ``hydra-core``
    packages (``pip install onnxsim[sam2]``) *and* the real Meta ``sam2``
    package, which is not published to PyPI under its official name --
    install it from source instead:
    ``pip install 'git+https://github.com/facebookresearch/sam2.git'`` (see
    https://github.com/facebookresearch/sam2#installation). Do not install
    a same-named ``sam2`` package from PyPI to satisfy this -- as of this
    writing that is an unrelated, unofficial third-party upload, not Meta's
    code.

    :param model_type: one of ``samexporter.export_sam2.MODEL_CONFIGS``'s
            keys, e.g. ``"sam2.1_hiera_large"``, ``"sam2.1_hiera_tiny"``,
            ``"sam2_hiera_base_plus"`` -- selects both the Hydra model
            config bundled with ``samexporter`` and the backbone size the
            checkpoint (if any) is expected to match.
    :param output_dir: directory to save ``encoder.onnx`` and
            ``decoder.onnx`` into (created if it doesn't exist).
    :param checkpoint: path to a SAM 2 ``.pt`` checkpoint to load into the
            model. If ``None`` (the default), the model traces with its
            random initialization instead -- no checkpoint, no network
            call, enough to produce structurally-correct ONNX graphs, e.g.
            for testing.
    :param opset_version: ONNX opset to trace both graphs with. Matches
            ``samexporter``'s own CLI default (must be >=11).
    :param multimask_output: whether the decoder graph outputs SAM 2's 3
            candidate masks (``True``, the default -- matches
            ``samexporter``'s own CLI default) or a single
            stability-selected mask (``False``).
    :param input_size: ``(height, width)`` to trace the image encoder with.
            SAM 2 always internally resizes to a square
            ``sam2_model.image_size`` (1024 for the released checkpoints)
            regardless of this value's aspect ratio, so in practice this
            should stay square too unless tracing a custom-trained model
            with a different ``image_size``.
    :param check_n: forwarded to :func:`onnxsim.simplify` for both graphs --
            how many random-input runs to check each simplified model
            against its freshly traced version for numerical equivalence.
    :param save_as_external_data: save each simplified graph's weights in a
            companion ``<filename>.data`` file instead of inline. Off by
            default, like :func:`onnxsim.export_detectron_model` -- a
            single encoder or decoder graph is rarely large enough to need
            it.
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify` for both graphs.
    :returns: ``{"encoder.onnx": check_ok, "decoder.onnx": check_ok}``,
            where each ``check_ok`` is that graph's :func:`onnxsim.simplify`
            numerical-equivalence check result (always ``True`` when
            ``check_n == 0``, since no check is performed).
    """
    try:
        import torch
        from samexporter.export_sam2 import (
            MODEL_CONFIGS,
            SAM2ImageDecoder,
            SAM2ImageEncoder,
        )
    except ImportError as e:
        raise ImportError(
            "export_sam2_model needs the optional 'torch', 'samexporter', "
            "and 'hydra-core' packages: pip install onnxsim[sam2]. It also "
            "needs the real Meta 'sam2' package, which is not published to "
            "PyPI under its official name -- install it from source: "
            "pip install 'git+https://github.com/facebookresearch/"
            "sam2.git' (see https://github.com/facebookresearch/sam2"
            "#installation). Do not install a same-named 'sam2' package "
            "from PyPI for this -- as of this writing that is an "
            "unrelated, unofficial third-party upload."
        ) from e

    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of "
            f"{sorted(MODEL_CONFIGS)}"
        )

    try:
        from sam2.build_sam import build_sam2
    except ImportError as e:
        raise ImportError(
            "export_sam2_model needs the real Meta 'sam2' package, "
            "installed from source (it is not published to PyPI under its "
            "official name): pip install 'git+https://github.com/"
            "facebookresearch/sam2.git' (see https://github.com/"
            "facebookresearch/sam2#installation)"
        ) from e

    import samexporter.sam2_configs as _sam2_configs_pkg
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = os.path.dirname(os.path.abspath(_sam2_configs_pkg.__file__))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        sam2_model = build_sam2(MODEL_CONFIGS[model_type], checkpoint, device="cpu")
    sam2_model.eval()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    image = torch.randn(1, 3, input_size[0], input_size[1])
    encoder = SAM2ImageEncoder(sam2_model)
    with torch.no_grad():
        high_res_feats_0, high_res_feats_1, image_embed = encoder(image)

    encoder_raw = _trace_to_onnx(
        encoder,
        (image,),
        opset_version=opset_version,
        input_names=["image"],
        output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
    )
    encoder_opt, encoder_ok = simplify(
        encoder_raw, check_n=check_n, **(simplify_kwargs or {})
    )
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    _save(encoder_opt, encoder_path, force_external_data=save_as_external_data)
    results["encoder.onnx"] = encoder_ok

    decoder = SAM2ImageDecoder(sam2_model, multimask_output=multimask_output)
    embed_size = (
        sam2_model.image_size // sam2_model.backbone_stride,
        sam2_model.image_size // sam2_model.backbone_stride,
    )
    mask_input_size = [4 * x for x in embed_size]
    point_coords = torch.randint(
        low=0, high=input_size[1], size=(1, 5, 2), dtype=torch.float
    )
    point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    decoder_inputs = (
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
    )
    with torch.no_grad():
        decoder(*decoder_inputs)

    decoder_raw = _trace_to_onnx(
        decoder,
        decoder_inputs,
        opset_version=opset_version,
        input_names=[
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {0: "num_labels", 1: "num_points"},
            "point_labels": {0: "num_labels", 1: "num_points"},
            "mask_input": {0: "num_labels"},
            "has_mask_input": {0: "num_labels"},
        },
    )
    decoder_opt, decoder_ok = simplify(
        decoder_raw, check_n=check_n, **(simplify_kwargs or {})
    )
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    _save(decoder_opt, decoder_path, force_external_data=save_as_external_data)
    results["decoder.onnx"] = decoder_ok

    return results


def _trace_to_onnx(module, args, **export_kwargs) -> onnx.ModelProto:
    import torch

    buf = io.BytesIO()
    try:
        # Newer torch versions default torch.onnx.export to the dynamo-based
        # exporter -- pin to the classic TorchScript-based tracer explicitly
        # where that knob exists, matching what samexporter's own export
        # script (torch.onnx.utils.export) was written/tested against.
        torch.onnx.export(
            module, args, buf, export_params=True, dynamo=False, **export_kwargs
        )
    except TypeError:
        buf = io.BytesIO()
        torch.onnx.export(module, args, buf, export_params=True, **export_kwargs)
    return onnx.load_from_string(buf.getvalue())
