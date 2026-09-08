"""Export a `Detectron2 <https://github.com/facebookresearch/detectron2>`_
model (Faster/Mask R-CNN, RetinaNet, ...) straight to a simplified ONNX file.

Unlike a ``transformers``/``diffusers`` model, a Detectron2 model has no
``optimum``-style single-call ONNX exporter to build on: Detectron2's own
supported recipe for turning one into ONNX is
``detectron2.export.TracingAdapter`` wrapped around ``torch.onnx.export`` --
the same steps Detectron2's own ``tools/deploy/export_model.py`` runs by
hand, because the model's ``forward()`` takes/returns Python dicts and
``Instances`` objects that neither ``torch.jit.trace`` nor
``torch.onnx.export`` understand directly. ``TracingAdapter`` flattens that
dict-in/``Instances``-out interface into plain tensors for tracing, and (for
the ``GeneralizedRCNN`` family -- Faster/Mask/Keypoint R-CNN) skips the
non-traceable postprocessing step (resizing detections back to the original
image size), which is pure Python control flow.

That export is plain, un-fused ONNX -- there is real simplification left on
the table for onnxsim's own pipeline to find, exactly as for a transformers
or diffusion export. :func:`export_detectron_model` wraps the
build-model-then-trace-then-export recipe and feeds the result straight into
:func:`onnxsim.simplify`, the Detectron2 counterpart of
:func:`onnxsim.export_transformers_model` /
:func:`onnxsim.export_diffusion_model`.
"""

import io
import os
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.onnx_simplifier import simplify
from onnxsim.transformers_export import _save

# Detectron2's own tested/known-working ONNX opset, from
# tools/deploy/export_model.py's export_tracing(). Used only as a fallback
# when the constant isn't importable from the installed detectron2 (it has
# moved between submodules across versions) and the caller doesn't pass
# opset_version explicitly.
_FALLBACK_OPSET_VERSION = 11


def export_detectron_model(
    config_file: str,
    output_path: str,
    weights: Optional[str] = None,
    image: Optional[Union[str, "np.ndarray"]] = None,
    opset_version: Optional[int] = None,
    check_n: int = 0,
    save_as_external_data: bool = False,
    config_opts: Optional[Sequence[str]] = None,
    simplify_kwargs: Optional[Dict] = None,
) -> bool:
    """Build a Detectron2 model from ``config_file``, trace it to ONNX, then
    simplify the result and save it to ``output_path``.

    Needs the optional ``torch`` and ``detectron2`` packages. Unlike this
    package's other framework-export helpers, ``detectron2`` itself is not
    published to PyPI -- install it from source, e.g.
    ``pip install 'git+https://github.com/facebookresearch/detectron2.git'``
    (see
    https://detectron2.readthedocs.io/en/latest/tutorials/install.html);
    ``pip install onnxsim[detectron2]`` only pulls in ``torch``.

    :param config_file: path to a Detectron2 YAML config, or the name of one
            of Detectron2's built-in model zoo configs (e.g.
            ``"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"``), resolved via
            ``detectron2.model_zoo.get_config_file``.
    :param output_path: where to save the simplified ``.onnx`` file.
    :param weights: path or URL to a checkpoint to load into the model,
            overriding ``cfg.MODEL.WEIGHTS``. If ``None`` (the default), no
            checkpoint is loaded at all, *regardless* of what
            ``cfg.MODEL.WEIGHTS`` the config itself specifies -- most model
            zoo configs default it to an ImageNet-pretrained backbone URL
            (``detectron2://ImageNetPretrained/...``), which would otherwise
            mean a surprise network fetch on every call. The model then
            traces with its random initialization instead, enough to
            produce a structurally-correct ONNX graph with no network call
            at all, e.g. for testing. Pass
            ``detectron2.model_zoo.get_checkpoint_url(config_file)``
            explicitly to trace with a model zoo config's official
            pretrained COCO weights.
    :param image: a sample image to trace with -- a path (read via
            ``detectron2.data.detection_utils.read_image`` using
            ``cfg.INPUT.FORMAT``) or an already-loaded HWC ``uint8`` array in
            that format. The image only needs to be *some* valid input --
            traced control flow (e.g. FPN level count) depends on its
            resized shape, not its pixel content. If ``None``, a synthetic
            random image sized ``cfg.INPUT.MIN_SIZE_TEST`` square (800 if
            unset) is used instead.
    :param opset_version: ONNX opset to trace with. Defaults to
            ``detectron2.export.STABLE_ONNX_EXPORT_VERSION`` when
            importable, else the opset that constant has historically held.
    :param check_n: forwarded to :func:`onnxsim.simplify` -- how many
            random-input runs to check the simplified model against the
            freshly traced one for numerical equivalence.
    :param save_as_external_data: save the simplified graph's weights in a
            companion ``<output_path>.data`` file instead of inline. Off by
            default, like the ``onnxsim`` CLI's own
            ``--save-as-external-data`` (which only falls back to external
            data once a graph is too large to serialize inline at all,
            >2GB) -- unlike :func:`onnxsim.export_transformers_model` /
            :func:`onnxsim.export_diffusion_model`, a single detection
            model's weights are rarely large enough to need it by default.
    :param config_opts: extra ``key value`` pairs forwarded to
            ``cfg.merge_from_list`` (Detectron2's ``opts`` CLI convention),
            e.g. ``["MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.5"]``.
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify`.
    :returns: the numerical-equivalence check result from
            :func:`onnxsim.simplify` (always ``True`` when ``check_n == 0``,
            since no check is performed).
    """
    try:
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.data import detection_utils
        from detectron2.data import transforms as T
        from detectron2.export import TracingAdapter
        from detectron2.modeling import GeneralizedRCNN, build_model
    except ImportError as e:
        raise ImportError(
            "export_detectron_model needs the optional 'torch' and "
            "'detectron2' packages. 'pip install onnxsim[detectron2]' only "
            "pulls in torch -- detectron2 itself is not published to PyPI, "
            "install it from source, e.g. "
            "pip install 'git+https://github.com/facebookresearch/"
            "detectron2.git' (see https://detectron2.readthedocs.io/en/"
            "latest/tutorials/install.html)"
        ) from e

    try:
        from detectron2.export import STABLE_ONNX_EXPORT_VERSION
    except ImportError:
        STABLE_ONNX_EXPORT_VERSION = _FALLBACK_OPSET_VERSION

    cfg = get_cfg()
    if os.path.isfile(config_file):
        cfg.merge_from_file(config_file)
    else:
        from detectron2 import model_zoo

        cfg.merge_from_file(model_zoo.get_config_file(config_file))
    if config_opts:
        cfg.merge_from_list(list(config_opts))
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    if weights is not None:
        DetectionCheckpointer(model).load(weights)

    if image is None:
        size = cfg.INPUT.MIN_SIZE_TEST or 800
        original_image = np.random.randint(0, 256, (size, size, 3)).astype(np.uint8)
    elif isinstance(image, str):
        original_image = detection_utils.read_image(image, format=cfg.INPUT.FORMAT)
    else:
        original_image = image
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    resized_image = aug.get_transform(original_image).apply_image(original_image)
    image_tensor = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1))
    # TracingAdapter requires every flattened input to be a tensor, so
    # height/width (plain ints, only used by postprocessing) are left out --
    # exactly what Detectron2's own tools/deploy/export_model.py does before
    # constructing TracingAdapter. do_postprocess=False below means the
    # traced GeneralizedRCNN graph skips the step that would need them
    # (resizing masks/boxes back to the original image size); that resize
    # is a real limitation of this export, not something to work around
    # here -- callers doing their own postprocessing need original_image's
    # (height, width) shape from outside the traced graph.
    sample_inputs = [{"image": image_tensor}]

    inference: Optional[Callable] = None
    if isinstance(model, GeneralizedRCNN):

        def inference(model, inputs):
            instances = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": instances}]

    traceable_model = TracingAdapter(model, sample_inputs, inference)

    buf = io.BytesIO()
    with torch.no_grad():
        try:
            # Newer torch versions default torch.onnx.export to the
            # dynamo-based exporter, which isn't the path Detectron2's own
            # export recipe was ever written/tested against (TracingAdapter
            # predates it) -- pin to the classic TorchScript-based tracer
            # explicitly where that knob exists.
            torch.onnx.export(
                traceable_model,
                (image_tensor,),
                buf,
                opset_version=opset_version or STABLE_ONNX_EXPORT_VERSION,
                dynamo=False,
            )
        except TypeError:
            buf = io.BytesIO()
            torch.onnx.export(
                traceable_model,
                (image_tensor,),
                buf,
                opset_version=opset_version or STABLE_ONNX_EXPORT_VERSION,
            )
    raw_model = onnx.load_from_string(buf.getvalue())

    model_opt, check_ok = simplify(
        raw_model, check_n=check_n, **(simplify_kwargs or {})
    )
    _save(model_opt, output_path, force_external_data=save_as_external_data)
    return check_ok
