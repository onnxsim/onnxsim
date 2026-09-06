"""Genuine PyTorch end-to-end tests for the three BEV-family custom-op
rewrite passes: ``rewrite_msdeformattn_to_gridsample``,
``rewrite_deform_conv_to_gather``, ``rewrite_trt_batched_nms``.

Unlike the three per-op test files
(``tests/test_msdeformattn_to_gridsample.py``,
``tests/test_deform_conv_to_gather.py``, ``tests/test_trt_batched_nms.py``),
which build single-node graphs directly with ``onnx.parser`` and validate
each pass in isolation against a from-scratch NumPy reference, this file
builds small real ``nn.Module``s out of these custom ops (via the
``torch.autograd.Function``/``symbolic()`` pairs in
``tests/_bev_torch_custom_ops.py``), exports them through actual
``torch.onnx.export``, and checks the ``onnxsim.simplify``-decomposed graph's
output against *that same module's own eager-mode PyTorch output* -- a step
up in realism: a real exporter's custom-op emission, not an
``onnx.parser``-authored graph standing in for one.

Two small BEV-style model families are covered, both built from the same
underlying custom-op building blocks, so the passes are exercised together
in combinations no per-op test can cover:

  - ``MiniBEVFormerEncoderLayer``: a DCNv2 backbone stem, then BEVFormer's
    own two-attention encoder-layer shape -- spatial cross-attention (BEV
    queries over multi-camera/multi-level image features) and temporal
    self-attention (BEV queries over a previous-frame BEV grid) -- followed
    by a small FFN with residual/LayerNorm. Exercises
    ``ModulatedDeformConvFunction`` (DCNv2) and two independent
    ``MSDeformAttnFunction`` calls (multi-level and single-level) in one
    graph.
  - ``MiniDETR3DHead``: a DCNv1 backbone, one deformable-attention decoder
    layer (object queries over multi-camera/multi-level image features),
    small classification/bbox-regression heads, and an NMS post-processing
    stage. NOTE: real DETR3D/PETR are NMS-free set predictors -- the
    ``TRTBatchedNMSFunction`` stage here is a deliberate addition (common in
    other anchor-based BEV/3D detection heads, e.g. mmdetection3d's
    anchor-based heads) purely so this suite exercises
    ``rewrite_trt_batched_nms`` together with ``MSDeformAttnFunction`` and
    ``DeformConvFunction`` (DCNv1) in one combined end-to-end graph -- this
    is not a claim about real DETR3D's own architecture.

Both models keep every spatial size tiny (2 cameras, 8x8/4x4 feature levels,
a 6x6 BEV grid, <=10 object queries): this is a correctness test, not a
performance one.

Why the legacy (``dynamo=False``) exporter, and why gated on torch at all:
torch>=2's *default* ONNX exporter is dynamo/torch.export-based and does not
support the classic per-``autograd.Function`` ``symbolic()`` custom-op
registration mechanism this file (and mmdeploy's own real exporter) relies
on -- ``dynamo=False`` selects the legacy TorchScript-tracing exporter,
which does. torch (and, transitively, this custom-op export path) is a
heavy, optional dependency not part of onnxsim's test requirements -- same
tier as ``tests/test_mmdeploy_integration.py`` (see that file's module
docstring for the precedent) -- so this whole module is skipped outright
when torch is not installed, and runs as its own dedicated job in the
``backend-integration`` CI workflow rather than the regular build-and-test
matrix.
"""

import collections
import io
import warnings

import numpy as np
import onnx
import pytest

torch = pytest.importorskip(
    "torch", reason="genuine torch.onnx.export custom-op tests need torch"
)
nn = torch.nn

from _bev_torch_custom_ops import (  # noqa: E402
    DeformConvFunction,
    ModulatedDeformConvFunction,
    MSDeformAttnFunction,
    TRTBatchedNMSFunction,
    assert_same_detections,
)

import onnxsim  # noqa: E402

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None

_EXTRA_OPTIMIZERS = [
    "rewrite_msdeformattn_to_gridsample",
    "rewrite_deform_conv_to_gather",
    "rewrite_trt_batched_nms",
]

_CUSTOM_OP_TYPES = {
    "MMCVMultiScaleDeformableAttention",
    "MMCVDeformConv2d",
    "MMCVModulatedDeformConv2d",
    "TRTBatchedNMS",
}


def _export_and_simplify(model, inputs, input_names, output_names):
    """Exports ``model`` via the legacy tracing exporter, asserts the raw
    exported graph actually contains custom op(s) (so the numeric check
    below isn't accidentally trivial -- see this file's module docstring),
    runs ``onnxsim.simplify`` with all three rewrite passes opted in, and
    asserts every custom op type is gone afterward."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        # Two harmless, expected warnings: the legacy exporter's own
        # deprecation notice, and a shape-inference warning about the
        # missing custom-op shape function (there is none -- these are
        # custom ops with no ONNX schema).
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            inputs,
            buf,
            # >= 20, not just the >= 16 rewrite_msdeformattn_to_gridsample.h's
            # header comment asks for: that pass hardcodes GridSample's
            # mode="linear" (opset 20's renamed enum value), which is not a
            # legal value under GridSample-16's schema (needs "bilinear")
            # even though the pass's own predicate only checks opset >= 16 --
            # discovered empirically running this file against onnxruntime,
            # not from the header comment alone.
            opset_version=20,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
        )
    exported = onnx.load_model_from_string(buf.getvalue())

    exported_op_types = collections.Counter(n.op_type for n in exported.graph.node)
    found_custom = _CUSTOM_OP_TYPES & set(exported_op_types)
    assert found_custom, (
        "expected the raw torch.onnx.export output to contain a custom op "
        f"from {_CUSTOM_OP_TYPES}, got {exported_op_types}"
    )

    sim_model, _ = onnxsim.simplify(exported, extra_optimizers=_EXTRA_OPTIMIZERS)
    sim_op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    still_present = _CUSTOM_OP_TYPES & set(sim_op_types)
    assert not still_present, (
        f"custom op(s) {still_present} survived simplification: {sim_op_types}"
    )
    return exported, exported_op_types, sim_model, sim_op_types


def _run_ort_or_reference(model, feeds):
    if _ort is not None:
        sess = _ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    return evaluator.run(None, feeds)


# =========================================================================== #
# Family 1: MiniBEVFormerEncoderLayer
#
# DCNv2 backbone stem -> spatial cross-attention (multi-level/multi-camera
# image features) + temporal self-attention (previous-frame BEV grid) ->
# small FFN with residual/LayerNorm. Mirrors BEVFormer's own published
# encoder-layer structure (DCN-ResNet backbone + temporal self-attn +
# spatial cross-attn + FFN per layer), at toy scale.
# =========================================================================== #


class _MiniDCNv2(nn.Module):
    """A single DCNv2 (modulated deformable conv) layer: offset/mask are
    predicted from the same input feature map by ordinary convs, exactly how
    real DCN-ResNet blocks wire it up."""

    def __init__(
        self, cin, cout, k=3, stride=1, padding=1, dilation=1, deform_groups=2
    ):
        super().__init__()
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.deform_groups = deform_groups
        self.offset_conv = nn.Conv2d(cin, deform_groups * 2 * k * k, k, stride, padding)
        self.mask_conv = nn.Conv2d(cin, deform_groups * k * k, k, stride, padding)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k) * 0.1)
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return ModulatedDeformConvFunction.apply(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
            self.deform_groups,
        )


def _flatten_levels(feats):
    """[(1,C,H,W), ...] -> (1, sum(H*W), C)."""
    flat = [f.flatten(2).transpose(1, 2) for f in feats]
    return torch.cat(flat, dim=1)


def _make_spatial_shapes(sizes):
    shapes = torch.tensor(sizes, dtype=torch.int64)
    starts = [0]
    running = 0
    for h, w in sizes[:-1]:
        running += h * w
        starts.append(running)
    return shapes, torch.tensor(starts, dtype=torch.int64)


class MiniBEVFormerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cams = 2
        self.M, self.D = 2, 4
        self.embed_dim = self.M * self.D  # 8
        self.P = 4
        self.num_bev_h, self.num_bev_w = 6, 6
        self.nq = self.num_bev_h * self.num_bev_w  # 36

        self.stem = nn.Conv2d(3, self.embed_dim, kernel_size=1)
        self.dcn = _MiniDCNv2(self.embed_dim, self.embed_dim, deform_groups=2)
        self.down = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1
        )  # 8x8 -> 4x4

        # Spatial cross-attention: 2 cameras x 2 levels each = 4 levels.
        sca_sizes = [(8, 8), (4, 4), (8, 8), (4, 4)]
        self.L_sca = len(sca_sizes)
        shapes, starts = _make_spatial_shapes(sca_sizes)
        self.register_buffer("spatial_shapes_sca", shapes)
        self.register_buffer("level_start_index_sca", starts)

        # Temporal self-attention: single level, the previous BEV grid.
        self.L_tsa = 1
        shapes_t, starts_t = _make_spatial_shapes([(self.num_bev_h, self.num_bev_w)])
        self.register_buffer("spatial_shapes_tsa", shapes_t)
        self.register_buffer("level_start_index_tsa", starts_t)

        self.bev_queries = nn.Parameter(torch.randn(self.nq, self.embed_dim) * 0.1)

        self.sca_offsets = nn.Linear(self.embed_dim, self.M * self.L_sca * self.P * 2)
        self.sca_weights = nn.Linear(self.embed_dim, self.M * self.L_sca * self.P)
        self.tsa_offsets = nn.Linear(self.embed_dim, self.M * self.L_tsa * self.P * 2)
        self.tsa_weights = nn.Linear(self.embed_dim, self.M * self.L_tsa * self.P)

        self.combine = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.ffn1 = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.ffn2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, imgs, prev_bev):
        # imgs: (num_cams, 3, 8, 8); prev_bev: (1, nq, embed_dim).
        level_feats = []
        for cam in range(self.num_cams):
            x = self.stem(imgs[cam : cam + 1])
            l0 = self.dcn(x)  # (1, embed_dim, 8, 8)
            l1 = self.down(l0)  # (1, embed_dim, 4, 4)
            level_feats.append(l0)
            level_feats.append(l1)

        value = _flatten_levels(level_feats).reshape(1, -1, self.M, self.D)

        bs = 1
        q = self.bev_queries.unsqueeze(0)  # (1, nq, embed_dim)

        sca_loc = torch.sigmoid(self.sca_offsets(q)).reshape(
            bs, self.nq, self.M, self.L_sca, self.P, 2
        )
        sca_w = torch.softmax(
            self.sca_weights(q).reshape(bs, self.nq, self.M, self.L_sca * self.P),
            dim=-1,
        ).reshape(bs, self.nq, self.M, self.L_sca, self.P)
        sca_out = MSDeformAttnFunction.apply(
            value,
            self.spatial_shapes_sca,
            self.level_start_index_sca,
            sca_loc,
            sca_w,
        )  # (1, nq, embed_dim)

        value_t = prev_bev.reshape(bs, self.nq, self.M, self.D)
        tsa_loc = torch.sigmoid(self.tsa_offsets(q)).reshape(
            bs, self.nq, self.M, self.L_tsa, self.P, 2
        )
        tsa_w = torch.softmax(
            self.tsa_weights(q).reshape(bs, self.nq, self.M, self.L_tsa * self.P),
            dim=-1,
        ).reshape(bs, self.nq, self.M, self.L_tsa, self.P)
        tsa_out = MSDeformAttnFunction.apply(
            value_t,
            self.spatial_shapes_tsa,
            self.level_start_index_tsa,
            tsa_loc,
            tsa_w,
        )  # (1, nq, embed_dim)

        fused = self.combine(torch.cat([sca_out, tsa_out], dim=-1))
        x = self.norm1(q + fused)
        ffn_out = self.ffn2(torch.relu(self.ffn1(x)))
        x = self.norm2(x + ffn_out)
        return x  # (1, nq, embed_dim)


def test_bevformer_encoder_layer_e2e():
    torch.manual_seed(0)
    model = MiniBEVFormerEncoderLayer()
    model.eval()

    imgs = torch.randn(model.num_cams, 3, 8, 8)
    prev_bev = torch.randn(1, model.nq, model.embed_dim)

    with torch.no_grad():
        ref_output = model(imgs, prev_bev)

    exported, exported_op_types, sim_model, sim_op_types = _export_and_simplify(
        model, (imgs, prev_bev), ["imgs", "prev_bev"], ["bev_out"]
    )
    # self.dcn is called once per camera (num_cams=2), each producing its
    # own traced custom-op node.
    assert exported_op_types["MMCVModulatedDeformConv2d"] == model.num_cams, (
        exported_op_types
    )
    assert exported_op_types["MMCVMultiScaleDeformableAttention"] == 2, (
        exported_op_types
    )
    assert "GridSample" in sim_op_types, sim_op_types
    assert "GatherND" in sim_op_types, sim_op_types
    # A non-trivial number of nodes replaced two custom ops -- a meaningful
    # size assertion, not padding.
    assert len(sim_model.graph.node) > len(exported.graph.node) + 20, (
        len(sim_model.graph.node),
        len(exported.graph.node),
    )

    feeds = {
        "imgs": imgs.numpy(),
        "prev_bev": prev_bev.numpy(),
    }
    (actual,) = _run_ort_or_reference(sim_model, feeds)
    np.testing.assert_allclose(actual, ref_output.numpy(), rtol=1e-3, atol=1e-4)


# =========================================================================== #
# Family 2: MiniDETR3DHead
#
# DCNv1 backbone -> one deformable-attention decoder layer (object queries
# over multi-camera/multi-level image features) -> classification + bbox
# heads -> TRTBatchedNMS post-processing.
#
# NOTE: real DETR3D/PETR are NMS-free set predictors. The NMS stage here is
# a deliberate addition, common in other anchor-based BEV/3D detection heads
# (e.g. mmdetection3d's anchor-based heads), included purely to exercise
# rewrite_trt_batched_nms together with MSDeformAttnFunction and
# DeformConvFunction in one combined graph -- not a claim about DETR3D's own
# architecture.
# =========================================================================== #


class _MiniDCNv1(nn.Module):
    """A single DCNv1 (unmodulated deformable conv) layer -- no mask."""

    def __init__(
        self, cin, cout, k=3, stride=1, padding=1, dilation=1, deform_groups=4
    ):
        super().__init__()
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.deform_groups = deform_groups
        self.offset_conv = nn.Conv2d(cin, deform_groups * 2 * k * k, k, stride, padding)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k) * 0.1)
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x):
        offset = self.offset_conv(x)
        return DeformConvFunction.apply(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
            self.deform_groups,
        )


class MiniDETR3DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cams = 2
        self.M, self.D = 2, 4
        self.embed_dim = self.M * self.D  # 8
        self.P = 4
        self.num_queries = 10
        self.num_classes = 3

        self.stem = nn.Conv2d(3, self.embed_dim, kernel_size=1)
        self.dcn = _MiniDCNv1(self.embed_dim, self.embed_dim, deform_groups=4)
        self.down = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1
        )

        sizes = [(8, 8), (4, 4), (8, 8), (4, 4)]
        self.L = len(sizes)
        shapes, starts = _make_spatial_shapes(sizes)
        self.register_buffer("spatial_shapes", shapes)
        self.register_buffer("level_start_index", starts)

        self.queries = nn.Parameter(torch.randn(self.num_queries, self.embed_dim) * 0.1)
        self.dec_offsets = nn.Linear(self.embed_dim, self.M * self.L * self.P * 2)
        self.dec_weights = nn.Linear(self.embed_dim, self.M * self.L * self.P)

        self.cls_head = nn.Linear(self.embed_dim, self.num_classes)
        self.box_head = nn.Linear(self.embed_dim, 4)

        self.box_scale = 10.0
        self.background_label_id = -1
        self.top_k = 200
        self.keep_top_k = 5
        self.score_threshold = 0.05
        self.iou_threshold = 0.5

    def forward(self, imgs):
        level_feats = []
        for cam in range(self.num_cams):
            x = self.stem(imgs[cam : cam + 1])
            l0 = self.dcn(x)
            l1 = self.down(l0)
            level_feats.append(l0)
            level_feats.append(l1)

        value = _flatten_levels(level_feats).reshape(1, -1, self.M, self.D)

        bs = 1
        q = self.queries.unsqueeze(0)  # (1, num_queries, embed_dim)
        loc = torch.sigmoid(self.dec_offsets(q)).reshape(
            bs, self.num_queries, self.M, self.L, self.P, 2
        )
        w = torch.softmax(
            self.dec_weights(q).reshape(bs, self.num_queries, self.M, self.L * self.P),
            dim=-1,
        ).reshape(bs, self.num_queries, self.M, self.L, self.P)
        dec_out = MSDeformAttnFunction.apply(
            value, self.spatial_shapes, self.level_start_index, loc, w
        )  # (1, num_queries, embed_dim)

        scores = torch.sigmoid(self.cls_head(dec_out))  # (1, nq, num_classes)

        box_raw = torch.sigmoid(self.box_head(dec_out))  # (1, nq, 4): cx,cy,w,h
        cx, cy, bw, bh = box_raw.unbind(-1)
        x1 = (cx - bw / 2) * self.box_scale
        y1 = (cy - bh / 2) * self.box_scale
        x2 = (cx + bw / 2) * self.box_scale
        y2 = (cy + bh / 2) * self.box_scale
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (1, nq, 4)
        boxes_5d = boxes.unsqueeze(2)  # (1, nq, 1, 4) -- class-agnostic

        num_det, nms_boxes, nms_scores, nms_classes = TRTBatchedNMSFunction.apply(
            boxes_5d,
            scores,
            self.background_label_id,
            self.num_classes,
            self.top_k,
            self.keep_top_k,
            self.score_threshold,
            self.iou_threshold,
        )
        return num_det, nms_boxes, nms_scores, nms_classes


def test_detr3d_head_e2e():
    torch.manual_seed(1)
    model = MiniDETR3DHead()
    model.eval()

    imgs = torch.randn(model.num_cams, 3, 8, 8)

    with torch.no_grad():
        ref_output = model(imgs)

    exported, exported_op_types, sim_model, sim_op_types = _export_and_simplify(
        model,
        (imgs,),
        ["imgs"],
        ["num_detections", "nmsed_boxes", "nmsed_scores", "nmsed_classes"],
    )
    # self.dcn is called once per camera (num_cams=2), each producing its
    # own traced custom-op node.
    assert exported_op_types["MMCVDeformConv2d"] == model.num_cams, exported_op_types
    assert exported_op_types["MMCVMultiScaleDeformableAttention"] == 1, (
        exported_op_types
    )
    assert exported_op_types["TRTBatchedNMS"] == 1, exported_op_types
    assert "GridSample" in sim_op_types, sim_op_types
    assert "GatherND" in sim_op_types, sim_op_types
    assert "NonMaxSuppression" in sim_op_types, sim_op_types
    assert len(sim_model.graph.node) > len(exported.graph.node) + 20, (
        len(sim_model.graph.node),
        len(exported.graph.node),
    )

    feeds = {"imgs": imgs.numpy()}
    actual = tuple(_run_ort_or_reference(sim_model, feeds))
    expected = tuple(t.numpy() for t in ref_output)

    # ref_output (eager) is itself produced by the exact same
    # reference_trt_batched_nms algorithm (see
    # tests/_bev_torch_custom_ops.py's TRTBatchedNMSFunction.forward), so
    # comparing against it is equivalent in principle to comparing against
    # an independent NumPy reference -- but exact positional order across
    # near-equal scores is not guaranteed to match between the eager path
    # and the real-NonMaxSuppression-based decomposition (same caveat as
    # tests/test_trt_batched_nms.py), so compare kept detections as an
    # unordered per-batch set instead of raw positional allclose.
    assert_same_detections(actual, expected, atol=1e-3)
