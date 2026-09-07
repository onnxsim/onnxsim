"""Tests for ``onnxsim.qwen_drive_perception_reconstruct`` -- building
Qwen-Drive-1.0's BEV perception head (spine + detection/occupancy/map-seg
heads) directly from a HuggingFace-shaped checkpoint (see that module's own
docstring for the confirmed-from-source architecture and scope decisions).

Given the sheer size of this architecture (six encoder layers, six decoder
layers, a ~15-block 3D U-Net, a resnet18-style map decoder -- roughly 600
distinct parameter tensors even at this test's tiny scale), a full
independent-numpy reimplementation of the entire reference model (the rigor
``test_hf_reconstruct.py``/``test_qwen3_5_reconstruct.py`` apply) is not
attempted here. Instead:

- :func:`_build_tiny_checkpoint` synthesizes a checkpoint whose every tensor
  shape mirrors the real PyTorch modules' own ``__init__`` methods
  (confirmed against source), used to build-and-run the *entire* graph
  end to end via ``onnx.checker`` + ``onnx.reference.ReferenceEvaluator``
  -- this alone exercises every shape/wiring decision across the whole
  spine and all three heads, which is where a construction bug would
  actually show up.
- The single riskiest, most novel translation -- multi-scale deformable
  attention sampling via ONNX's native ``GridSample`` in place of the real
  CUDA kernel -- gets a real independent-numpy cross-check
  (:func:`test_ms_deform_attn_sample_matches_pytorch_reference`), reusing
  neither ``qwen_drive_perception_reconstruct.py`` nor the real repo's own
  ``multi_scale_deformable_attn_pytorch`` (confirmed identical to the CUDA
  kernel from its own source, see that module's docstring) as ground truth.
- ``denormalize_bbox``/the ``atan2`` workaround gets the same treatment.
"""

import json
import struct

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

from onnxsim.gguf_reconstruct import UnsupportedArchitectureError, _Builder
from onnxsim.qwen_drive_perception_reconstruct import (
    _IR_VERSION,
    _OPSET,
    _atan2,
    _denormalize_bbox,
    _ms_deform_attn_sample,
    reconstruct_qwen_drive_perception,
)

NUM_HEADS = 8
SCA_NUM_POINTS = 8
TSA_NUM_POINTS = 4
DECODER_NUM_POINTS = 4


def _rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.05


def _write_safetensors(path, tensors):
    """Same hand-rolled ``.safetensors`` writer as the rest of this test
    family -- see ``test_hf_reconstruct.py`` for the format."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(arr.tobytes())
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def _conv2d_w(rng, out_c, in_c, k):
    return _rand(rng, out_c, in_c, k, k)


def _conv3d_w(rng, out_c, in_c, k):
    return _rand(rng, out_c, in_c, k, k, k)


def _simple_fpn(rng, w, prefix, dim, out_channels, scale_factors):
    for i, scale in enumerate(scale_factors):
        p = f"{prefix}.stages.{i}"
        out_dim = dim
        idx = 0
        if scale == 4.0:
            w[f"{p}.{idx}.weight"] = _rand(rng, dim, dim // 2, 2, 2)
            w[f"{p}.{idx}.bias"] = _rand(rng, dim // 2)
            idx += 1
            w[f"{p}.{idx}.weight"] = _rand(rng, dim // 2)
            w[f"{p}.{idx}.bias"] = _rand(rng, dim // 2)
            idx += 2  # GELU, no params
            w[f"{p}.{idx}.weight"] = _rand(rng, dim // 2, dim // 4, 2, 2)
            w[f"{p}.{idx}.bias"] = _rand(rng, dim // 4)
            idx += 1
            out_dim = dim // 4
        elif scale == 2.0:
            w[f"{p}.{idx}.weight"] = _rand(rng, dim, dim // 2, 2, 2)
            w[f"{p}.{idx}.bias"] = _rand(rng, dim // 2)
            idx += 1
            out_dim = dim // 2
        elif scale == 0.5:
            idx += 1  # MaxPool, no params
        w[f"{p}.{idx}.weight"] = _conv2d_w(rng, out_channels, out_dim, 1)
        idx += 1
        w[f"{p}.{idx}.weight"] = _rand(rng, out_channels)
        w[f"{p}.{idx}.bias"] = _rand(rng, out_channels)
        idx += 1
        w[f"{p}.{idx}.weight"] = _conv2d_w(rng, out_channels, out_channels, 3)
        idx += 1
        w[f"{p}.{idx}.weight"] = _rand(rng, out_channels)
        w[f"{p}.{idx}.bias"] = _rand(rng, out_channels)


def _basic_block_2d(rng, w, prefix, planes, norm_names=("gn1", "gn2")):
    w[f"{prefix}.conv1.weight"] = _conv2d_w(rng, planes, planes, 3)
    w[f"{prefix}.{norm_names[0]}.weight"] = _rand(rng, planes)
    w[f"{prefix}.{norm_names[0]}.bias"] = _rand(rng, planes)
    w[f"{prefix}.conv2.weight"] = _conv2d_w(rng, planes, planes, 3)
    w[f"{prefix}.{norm_names[1]}.weight"] = _rand(rng, planes)
    w[f"{prefix}.{norm_names[1]}.bias"] = _rand(rng, planes)


def _basic_block_2d_downsample(
    rng, w, prefix, in_planes, planes, norm_names=("gn1", "gn2")
):
    w[f"{prefix}.conv1.weight"] = _conv2d_w(rng, planes, in_planes, 3)
    w[f"{prefix}.{norm_names[0]}.weight"] = _rand(rng, planes)
    w[f"{prefix}.{norm_names[0]}.bias"] = _rand(rng, planes)
    w[f"{prefix}.conv2.weight"] = _conv2d_w(rng, planes, planes, 3)
    w[f"{prefix}.{norm_names[1]}.weight"] = _rand(rng, planes)
    w[f"{prefix}.{norm_names[1]}.bias"] = _rand(rng, planes)
    w[f"{prefix}.downsample.0.weight"] = _conv2d_w(rng, planes, in_planes, 1)
    w[f"{prefix}.downsample.1.weight"] = _rand(rng, planes)
    w[f"{prefix}.downsample.1.bias"] = _rand(rng, planes)


def _aspp(rng, w, prefix, inplanes, aspp_mid):
    for i in (1, 2, 3, 4):
        w[f"{prefix}.aspp{i}.atrous_conv.weight"] = _conv2d_w(
            rng, aspp_mid, inplanes, 1 if i == 1 else 3
        )
        w[f"{prefix}.aspp{i}.bn.weight"] = _rand(rng, aspp_mid)
        w[f"{prefix}.aspp{i}.bn.bias"] = _rand(rng, aspp_mid)
    w[f"{prefix}.global_avg_pool.1.weight"] = _conv2d_w(rng, aspp_mid, inplanes, 1)
    w[f"{prefix}.global_avg_pool.2.weight"] = _rand(rng, aspp_mid)
    w[f"{prefix}.global_avg_pool.2.bias"] = _rand(rng, aspp_mid)
    w[f"{prefix}.conv1.weight"] = _conv2d_w(rng, inplanes, aspp_mid * 5, 1)
    w[f"{prefix}.bn1.weight"] = _rand(rng, inplanes)
    w[f"{prefix}.bn1.bias"] = _rand(rng, inplanes)


def _depth_net(rng, w, prefix, mid_channels, depth_dim, aspp_mid):
    w[f"{prefix}.reduce_conv.0.weight"] = _conv2d_w(rng, mid_channels, mid_channels, 3)
    w[f"{prefix}.reduce_conv.0.bias"] = _rand(rng, mid_channels)
    w[f"{prefix}.reduce_conv.1.weight"] = _rand(rng, mid_channels)
    w[f"{prefix}.reduce_conv.1.bias"] = _rand(rng, mid_channels)
    for i in range(3):
        _basic_block_2d(rng, w, f"{prefix}.depth_conv.{i}", mid_channels)
    _aspp(rng, w, f"{prefix}.depth_conv.3", mid_channels, aspp_mid)
    w[f"{prefix}.depth_conv.4.weight"] = _conv2d_w(rng, depth_dim, mid_channels, 1)
    w[f"{prefix}.depth_conv.4.bias"] = _rand(rng, depth_dim)


def _view_transform(rng, w, prefix, embed_dim, num_conv_layers=3):
    for i in range(num_conv_layers):
        w[f"{prefix}.conv_layer.{i}.0.weight"] = _conv3d_w(rng, embed_dim, embed_dim, 3)
        w[f"{prefix}.conv_layer.{i}.0.bias"] = _rand(rng, embed_dim)
        w[f"{prefix}.conv_layer.{i}.1.weight"] = _rand(rng, embed_dim)
        w[f"{prefix}.conv_layer.{i}.1.bias"] = _rand(rng, embed_dim)
        w[f"{prefix}.conv_layer.{i}.1.running_mean"] = np.zeros(
            embed_dim, dtype=np.float32
        )
        w[f"{prefix}.conv_layer.{i}.1.running_var"] = np.ones(
            embed_dim, dtype=np.float32
        )


def _temporal_self_attn(rng, w, prefix, embed_dim, num_bev_queue=2):
    w[f"{prefix}.sampling_offsets.weight"] = _rand(
        rng, num_bev_queue * NUM_HEADS * TSA_NUM_POINTS * 2, embed_dim * num_bev_queue
    )
    w[f"{prefix}.sampling_offsets.bias"] = _rand(
        rng, num_bev_queue * NUM_HEADS * TSA_NUM_POINTS * 2
    )
    w[f"{prefix}.attention_weights.weight"] = _rand(
        rng, num_bev_queue * NUM_HEADS * TSA_NUM_POINTS, embed_dim * num_bev_queue
    )
    w[f"{prefix}.attention_weights.bias"] = _rand(
        rng, num_bev_queue * NUM_HEADS * TSA_NUM_POINTS
    )
    w[f"{prefix}.value_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.value_proj.bias"] = _rand(rng, embed_dim)
    w[f"{prefix}.output_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.output_proj.bias"] = _rand(rng, embed_dim)


def _spatial_cross_attn(rng, w, prefix, embed_dim, num_levels=4):
    dw = f"{prefix}.deformable_attention"
    w[f"{dw}.sampling_offsets.weight"] = _rand(
        rng, NUM_HEADS * num_levels * SCA_NUM_POINTS * 2, embed_dim
    )
    w[f"{dw}.sampling_offsets.bias"] = _rand(
        rng, NUM_HEADS * num_levels * SCA_NUM_POINTS * 2
    )
    w[f"{dw}.attention_weights.weight"] = _rand(
        rng, NUM_HEADS * num_levels * SCA_NUM_POINTS, embed_dim
    )
    w[f"{dw}.attention_weights.bias"] = _rand(
        rng, NUM_HEADS * num_levels * SCA_NUM_POINTS
    )
    w[f"{dw}.value_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{dw}.value_proj.bias"] = _rand(rng, embed_dim)
    w[f"{prefix}.output_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.output_proj.bias"] = _rand(rng, embed_dim)


def _custom_ms_deform_attn(rng, w, prefix, embed_dim, num_levels=1):
    w[f"{prefix}.sampling_offsets.weight"] = _rand(
        rng, NUM_HEADS * num_levels * DECODER_NUM_POINTS * 2, embed_dim
    )
    w[f"{prefix}.sampling_offsets.bias"] = _rand(
        rng, NUM_HEADS * num_levels * DECODER_NUM_POINTS * 2
    )
    w[f"{prefix}.attention_weights.weight"] = _rand(
        rng, NUM_HEADS * num_levels * DECODER_NUM_POINTS, embed_dim
    )
    w[f"{prefix}.attention_weights.bias"] = _rand(
        rng, NUM_HEADS * num_levels * DECODER_NUM_POINTS
    )
    w[f"{prefix}.value_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.value_proj.bias"] = _rand(rng, embed_dim)
    w[f"{prefix}.output_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.output_proj.bias"] = _rand(rng, embed_dim)


def _multihead_attn(rng, w, prefix, embed_dim):
    w[f"{prefix}.in_proj_weight"] = _rand(rng, 3 * embed_dim, embed_dim)
    w[f"{prefix}.in_proj_bias"] = _rand(rng, 3 * embed_dim)
    w[f"{prefix}.out_proj.weight"] = _rand(rng, embed_dim, embed_dim)
    w[f"{prefix}.out_proj.bias"] = _rand(rng, embed_dim)


def _ffn(rng, w, prefix, embed_dim, ffn_channels):
    w[f"{prefix}.layers.0.0.weight"] = _rand(rng, ffn_channels, embed_dim)
    w[f"{prefix}.layers.0.0.bias"] = _rand(rng, ffn_channels)
    w[f"{prefix}.layers.1.weight"] = _rand(rng, embed_dim, ffn_channels)
    w[f"{prefix}.layers.1.bias"] = _rand(rng, embed_dim)


def _layernorm(rng, w, name, dim):
    w[f"{name}.weight"] = _rand(rng, dim) + 1.0
    w[f"{name}.bias"] = _rand(rng, dim)


def _bev_former_layer(rng, w, prefix, embed_dim, ffn_channels):
    _temporal_self_attn(rng, w, f"{prefix}.attentions.0", embed_dim)
    _spatial_cross_attn(rng, w, f"{prefix}.attentions.1", embed_dim)
    _ffn(rng, w, f"{prefix}.ffns.0", embed_dim, ffn_channels)
    for i in range(3):
        _layernorm(rng, w, f"{prefix}.norms.{i}", embed_dim)


def _detr_decoder_layer(rng, w, prefix, embed_dim, ffn_channels):
    _multihead_attn(rng, w, f"{prefix}.attentions.0.attn", embed_dim)
    _custom_ms_deform_attn(rng, w, f"{prefix}.attentions.1", embed_dim)
    _ffn(rng, w, f"{prefix}.ffns.0", embed_dim, ffn_channels)
    for i in range(3):
        _layernorm(rng, w, f"{prefix}.norms.{i}", embed_dim)


def _reg_branch(rng, w, prefix, embed_dim, code_size, num_reg_fcs=2):
    for i in range(num_reg_fcs):
        w[f"{prefix}.{2 * i}.weight"] = _rand(rng, embed_dim, embed_dim)
        w[f"{prefix}.{2 * i}.bias"] = _rand(rng, embed_dim)
    w[f"{prefix}.{2 * num_reg_fcs}.weight"] = _rand(rng, code_size, embed_dim)
    w[f"{prefix}.{2 * num_reg_fcs}.bias"] = _rand(rng, code_size)


def _cls_branch(rng, w, prefix, embed_dim, num_classes, num_reg_fcs=2):
    for i in range(num_reg_fcs):
        w[f"{prefix}.{3 * i}.weight"] = _rand(rng, embed_dim, embed_dim)
        w[f"{prefix}.{3 * i}.bias"] = _rand(rng, embed_dim)
        _layernorm(rng, w, f"{prefix}.{3 * i + 1}", embed_dim)
    w[f"{prefix}.{3 * num_reg_fcs}.weight"] = _rand(rng, num_classes, embed_dim)
    w[f"{prefix}.{3 * num_reg_fcs}.bias"] = _rand(rng, num_classes)


def _residual_stage_3d(rng, w, prefix, in_channels, out_channels, stride, num_blocks=2):
    for i in range(num_blocks):
        p = f"{prefix}.blocks.{i}"
        block_in = in_channels if i == 0 else out_channels
        block_stride = stride if i == 0 else (1, 1, 1)
        w[f"{p}.conv1.weight"] = _conv3d_w(rng, out_channels, block_in, 3)
        w[f"{p}.norm1.weight"] = _rand(rng, out_channels)
        w[f"{p}.norm1.bias"] = _rand(rng, out_channels)
        w[f"{p}.norm1.running_mean"] = np.zeros(out_channels, dtype=np.float32)
        w[f"{p}.norm1.running_var"] = np.ones(out_channels, dtype=np.float32)
        w[f"{p}.conv2.weight"] = _conv3d_w(rng, out_channels, out_channels, 3)
        w[f"{p}.norm2.weight"] = _rand(rng, out_channels)
        w[f"{p}.norm2.bias"] = _rand(rng, out_channels)
        w[f"{p}.norm2.running_mean"] = np.zeros(out_channels, dtype=np.float32)
        w[f"{p}.norm2.running_var"] = np.ones(out_channels, dtype=np.float32)
        if block_stride != (1, 1, 1) or block_in != out_channels:
            w[f"{p}.downsample.0.weight"] = _conv3d_w(rng, out_channels, block_in, 1)
            w[f"{p}.downsample.1.weight"] = _rand(rng, out_channels)
            w[f"{p}.downsample.1.bias"] = _rand(rng, out_channels)
            w[f"{p}.downsample.1.running_mean"] = np.zeros(
                out_channels, dtype=np.float32
            )
            w[f"{p}.downsample.1.running_var"] = np.ones(out_channels, dtype=np.float32)


def _occ_voxel_unet_refiner(rng, w, prefix, in_channels, out_channels):
    c0 = max(64, out_channels * 2)
    c1, c2, c3 = c0 * 2, c0 * 4, c0 * 6
    _residual_stage_3d(rng, w, f"{prefix}.input_proj", in_channels, c0, (1, 1, 1))
    _residual_stage_3d(rng, w, f"{prefix}.enc1", c0, c1, (1, 2, 2))
    _residual_stage_3d(rng, w, f"{prefix}.enc2", c1, c2, (1, 2, 2))
    _residual_stage_3d(rng, w, f"{prefix}.enc3", c2, c3, (1, 2, 2))
    _residual_stage_3d(rng, w, f"{prefix}.bottleneck", c3, c3, (1, 1, 1))
    _residual_stage_3d(rng, w, f"{prefix}.dec2", c3 + c2, c2, (1, 1, 1))
    _residual_stage_3d(rng, w, f"{prefix}.dec1", c2 + c1, c1, (1, 1, 1))
    _residual_stage_3d(rng, w, f"{prefix}.dec0", c1 + c0, c0, (1, 1, 1))
    _residual_stage_3d(rng, w, f"{prefix}.out_block", c0, c0, (1, 1, 1))
    w[f"{prefix}.out_proj.0.weight"] = _conv3d_w(rng, out_channels, c0, 3)
    w[f"{prefix}.out_proj.1.weight"] = _rand(rng, out_channels)
    w[f"{prefix}.out_proj.1.bias"] = _rand(rng, out_channels)
    w[f"{prefix}.out_proj.1.running_mean"] = np.zeros(out_channels, dtype=np.float32)
    w[f"{prefix}.out_proj.1.running_var"] = np.ones(out_channels, dtype=np.float32)


def _basic_block_2d_gn_resnet18(rng, w, prefix, in_planes, planes):
    if in_planes != planes:
        _basic_block_2d_downsample(
            rng, w, prefix, in_planes, planes, norm_names=("bn1", "bn2")
        )
    else:
        _basic_block_2d(rng, w, prefix, planes, norm_names=("bn1", "bn2"))


def _map_seg_encode(rng, w, prefix, in_c, out_c):
    w[f"{prefix}.conv1.weight"] = _conv2d_w(rng, 64, in_c, 7)
    w[f"{prefix}.bn1.weight"] = _rand(rng, 64)
    w[f"{prefix}.bn1.bias"] = _rand(rng, 64)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer1.0", 64, 64)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer1.1", 64, 64)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer2.0", 64, 128)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer2.1", 128, 128)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer3.0", 128, 256)
    _basic_block_2d_gn_resnet18(rng, w, f"{prefix}.layer3.1", 256, 256)
    w[f"{prefix}.up1.conv.0.weight"] = _conv2d_w(rng, 256, 64 + 256, 3)
    w[f"{prefix}.up1.conv.1.weight"] = _rand(rng, 256)
    w[f"{prefix}.up1.conv.1.bias"] = _rand(rng, 256)
    w[f"{prefix}.up1.conv.3.weight"] = _conv2d_w(rng, 256, 256, 3)
    w[f"{prefix}.up1.conv.4.weight"] = _rand(rng, 256)
    w[f"{prefix}.up1.conv.4.bias"] = _rand(rng, 256)
    w[f"{prefix}.up2.1.weight"] = _conv2d_w(rng, 128, 256, 3)
    w[f"{prefix}.up2.2.weight"] = _rand(rng, 128)
    w[f"{prefix}.up2.2.bias"] = _rand(rng, 128)
    w[f"{prefix}.up2.4.weight"] = _conv2d_w(rng, out_c, 128, 1)
    w[f"{prefix}.up2.4.bias"] = _rand(rng, out_c)


_TINY_CONFIG = dict(
    llm_dim=16,
    vit_dim=16,
    embed_dim=16,
    bev_h=8,
    bev_w=8,
    occ_pillar_h=2,
    occ_dim=4,
    num_query=6,
    code_size=10,
    det_num_classes=3,
    occ_num_classes=4,
    map_num_classes=3,
    num_encoder_layers=2,
    num_decoder_layers=2,
    det_pc_range=[-4.0, -4.0, -1.0, 4.0, 4.0, 1.0],
    det_voxel_size=[1.0, 1.0, 2.0],
    nuscenes_occ_pc_range=[-4.0, -4.0, -1.0, 4.0, 4.0, 1.0],
    nuscenes_occ_voxel_size=[1.0, 1.0, 1.0],
    nuplan_occ_pc_range=[-4.0, -4.0, -1.0, 4.0, 4.0, 1.0],
    nuplan_occ_voxel_size=[1.0, 1.0, 1.0],
    map_xbound=[-16.0, 16.0, 1.0],
    map_ybound=[-8.0, 8.0, 1.0],
    frustum_range=[0.0, 0.0, 1.0, 4.0, 4.0, 2.0],
    frustum_size=[2.0, 2.0, 0.5],
    post_center_range=[-4.0, -4.0, -1.0, 4.0, 4.0, 1.0],
    max_num_boxes=8,
    ffn_channels=32,
)


def _build_tiny_checkpoint(tmp_path, cfg=None, seed=0):
    """A checkpoint directory whose every tensor shape mirrors the real
    ``qwen_drive_perception`` PyTorch modules' own ``__init__`` methods
    (confirmed against source -- see this module's own docstring)."""
    cfg = dict(_TINY_CONFIG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    w = {}
    embed_dim = cfg["embed_dim"]
    ffn_channels = cfg["ffn_channels"]

    _simple_fpn(
        rng, w, "bev_modeling.adaptor", cfg["llm_dim"], embed_dim, (4.0, 2.0, 1.0, 0.5)
    )
    _simple_fpn(rng, w, "bev_modeling.vit_neck", cfg["vit_dim"], embed_dim, (1.0,))

    depth_dim = int(
        (cfg["frustum_range"][5] - cfg["frustum_range"][2]) / cfg["frustum_size"][2]
    )
    _depth_net(rng, w, "bev_modeling.depth_net", embed_dim, depth_dim, 96)
    _view_transform(rng, w, "bev_modeling.view_trans", embed_dim)
    w["bev_modeling.uvtr_query_proj.weight"] = _conv2d_w(
        rng, embed_dim, embed_dim * cfg["occ_pillar_h"], 1
    )
    w["bev_modeling.uvtr_query_proj.bias"] = _rand(rng, embed_dim)

    bev_h, bev_w = cfg["bev_h"], cfg["bev_w"]
    w["bev_modeling.head.bev_embedding.weight"] = _rand(rng, bev_h * bev_w, embed_dim)
    w["bev_modeling.head.positional_encoding.row_embed.weight"] = _rand(
        rng, bev_h, embed_dim // 2
    )
    w["bev_modeling.head.positional_encoding.col_embed.weight"] = _rand(
        rng, bev_w, embed_dim // 2
    )
    w["bev_modeling.head.transformer.level_embeds"] = _rand(rng, 4, embed_dim)

    for i in range(cfg["num_encoder_layers"]):
        _bev_former_layer(
            rng,
            w,
            f"bev_modeling.head.transformer.encoder.layers.{i}",
            embed_dim,
            ffn_channels,
        )

    w["bev_modeling.head.query_embedding.weight"] = _rand(
        rng, cfg["num_query"], 2 * embed_dim
    )
    w["bev_modeling.head.transformer.reference_points.weight"] = _rand(
        rng, 3, embed_dim
    )
    w["bev_modeling.head.transformer.reference_points.bias"] = _rand(rng, 3)
    for i in range(cfg["num_decoder_layers"]):
        _detr_decoder_layer(
            rng,
            w,
            f"bev_modeling.head.transformer.decoder.layers.{i}",
            embed_dim,
            ffn_channels,
        )
        _reg_branch(
            rng,
            w,
            f"bev_modeling.head.transformer.decoder.reg_branches.{i}",
            embed_dim,
            cfg["code_size"],
        )
        _cls_branch(
            rng,
            w,
            f"bev_modeling.head.cls_branches.{i}",
            embed_dim,
            cfg["det_num_classes"],
        )

    middle_dims = embed_dim // cfg["occ_pillar_h"]
    w["bev_modeling.head.transformer.uvtr_occ_proj.weight"] = _conv3d_w(
        rng, middle_dims, embed_dim, 1
    )
    w["bev_modeling.head.transformer.uvtr_occ_proj.bias"] = _rand(rng, middle_dims)
    w["bev_modeling.head.transformer.uvtr_occ_fuse.conv.weight"] = _conv3d_w(
        rng, middle_dims, middle_dims * 2, 1
    )
    w["bev_modeling.head.transformer.uvtr_occ_fuse.bn.weight"] = _rand(rng, middle_dims)
    w["bev_modeling.head.transformer.uvtr_occ_fuse.bn.bias"] = _rand(rng, middle_dims)
    w["bev_modeling.head.transformer.uvtr_occ_fuse.bn.running_mean"] = np.zeros(
        middle_dims, dtype=np.float32
    )
    w["bev_modeling.head.transformer.uvtr_occ_fuse.bn.running_var"] = np.ones(
        middle_dims, dtype=np.float32
    )
    _occ_voxel_unet_refiner(
        rng, w, "bev_modeling.head.transformer.occ_decoder", middle_dims, cfg["occ_dim"]
    )
    w["bev_modeling.head.transformer.occ_pred_head.0.weight"] = _rand(
        rng, cfg["occ_dim"] * 2, cfg["occ_dim"]
    )
    w["bev_modeling.head.transformer.occ_pred_head.0.bias"] = _rand(
        rng, cfg["occ_dim"] * 2
    )
    w["bev_modeling.head.transformer.occ_pred_head.2.weight"] = _rand(
        rng, cfg["occ_num_classes"], cfg["occ_dim"] * 2
    )
    w["bev_modeling.head.transformer.occ_pred_head.2.bias"] = _rand(
        rng, cfg["occ_num_classes"]
    )

    _map_seg_encode(
        rng, w, "bev_modeling.head.seg_decoder", embed_dim, cfg["map_num_classes"]
    )

    hf_dir = tmp_path / "tiny_qwen_drive_perception"
    hf_dir.mkdir()
    _write_safetensors(hf_dir / "model.safetensors", w)
    config_json = dict(cfg)
    config_json["model_type"] = "qwen_drive_perception"
    with open(hf_dir / "config.json", "w") as f:
        json.dump(config_json, f)
    return str(hf_dir), cfg


def _tiny_calibration(num_cams):
    lidar2img = np.stack([np.eye(4, dtype=np.float32) for _ in range(num_cams)])
    for c in range(num_cams):
        # Push points in front of the camera (positive depth after projection).
        lidar2img[c][2, 3] = 5.0
    lidar2ego = np.eye(4, dtype=np.float32)
    return lidar2img, lidar2ego


@pytest.mark.parametrize(
    "num_cams,num_encoder_layers,num_decoder_layers", [(2, 1, 1), (3, 2, 2)]
)
def test_reconstruct_qwen_drive_perception_builds_and_runs(
    tmp_path, num_cams, num_encoder_layers, num_decoder_layers
):
    """Structural test: the *entire* graph (spine + all 3 heads) builds,
    passes ``onnx.checker``, and actually executes via
    ``onnx.reference.ReferenceEvaluator`` with plausible output shapes --
    exercised at two different (camera count, layer count) combinations to
    catch any hardcoded-to-the-first-case indexing bug."""
    cfg = dict(
        _TINY_CONFIG,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )
    hf_dir, cfg = _build_tiny_checkpoint(tmp_path, cfg)
    lidar2img, lidar2ego = _tiny_calibration(num_cams)

    model = reconstruct_qwen_drive_perception(
        hf_dir,
        num_cams=num_cams,
        lidar2img=lidar2img,
        lidar2ego=lidar2ego,
        img_shape=(4, 4),
        llm_feat_hw=(4, 4),
        vit_feat_hw=(2, 2),
    )
    onnx.checker.check_model(model)

    rng = np.random.default_rng(3)
    inputs = {
        "img_llm_feats": rng.standard_normal((num_cams, 4, 4, cfg["llm_dim"])).astype(
            np.float32
        )
        * 0.1,
        "img_vit_feats": rng.standard_normal((num_cams, 2, 2, cfg["vit_dim"])).astype(
            np.float32
        )
        * 0.1,
    }
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, inputs)
    output_names = [o.name for o in model.graph.output]
    out = dict(zip(output_names, outputs))

    max_num = cfg["max_num_boxes"]
    assert out[output_names[0]].shape == (max_num,)  # det_scores
    assert out[output_names[1]].shape == (max_num,)  # det_labels
    assert out[output_names[2]].shape == (max_num, 9)  # det_boxes
    assert out[output_names[3]].shape == (max_num,)  # det_valid
    assert np.isin(out[output_names[3]], [0, 1]).all()
    assert np.isin(out[output_names[1]], np.arange(cfg["det_num_classes"])).all()
    scores = out[output_names[0]]
    assert (scores[:-1] >= scores[1:]).all()  # TopK sorted descending
    assert np.all((scores >= 0) & (scores <= 1))  # sigmoid outputs

    occ_logits = out[output_names[4]]
    assert occ_logits.shape[-1] == cfg["occ_num_classes"]
    map_logits = out[output_names[5]]
    assert map_logits.shape[0] == cfg["map_num_classes"]


def test_unsupported_model_type_raises(tmp_path):
    hf_dir = tmp_path / "not_perception"
    hf_dir.mkdir()
    with open(hf_dir / "config.json", "w") as f:
        json.dump({"model_type": "llama"}, f)
    _write_safetensors(hf_dir / "model.safetensors", {})
    with pytest.raises(UnsupportedArchitectureError):
        reconstruct_qwen_drive_perception(
            str(hf_dir),
            num_cams=1,
            lidar2img=np.eye(4, dtype=np.float32)[None],
            lidar2ego=np.eye(4, dtype=np.float32),
            img_shape=(4, 4),
            llm_feat_hw=(4, 4),
            vit_feat_hw=(2, 2),
        )


# ---------------------------------------------------------------------------
# Independent numpy cross-check of the riskiest translation: multi-scale
# deformable attention sampling via ONNX's native GridSample, standing in
# for the real CUDA kernel (see qwen_drive_perception_reconstruct.py's own
# docstring for why GridSample is exactly equivalent). This reference is a
# fresh, from-scratch port of the real repo's own CPU fallback
# (multi_scale_deformable_attn_pytorch), not a reuse of anything in
# onnxsim's own reconstruction module.


def _ms_deform_attn_reference_np(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """``layers.py::multi_scale_deformable_attn_pytorch``, ported to numpy.

    value: (bs, sum_hw, heads, head_dim)
    sampling_locations: (bs, num_queries, heads, levels, points, 2), in [0, 1]
    attention_weights: (bs, num_queries, heads, levels, points)
    """
    bs, _, heads, head_dim = value.shape
    _, num_queries, _, levels, points, _ = sampling_locations.shape
    sizes = [h * w for h, w in spatial_shapes]
    splits = np.split(value, np.cumsum(sizes)[:-1], axis=1)
    sampling_grids = 2 * sampling_locations - 1

    def bilinear_sample(img, grid):
        # img: (C, H, W). grid: (Nq, Np, 2) in [-1, 1], (x, y) order,
        # align_corners=False, padding_mode="zeros" -- matches GridSample.
        c, h, w = img.shape
        gx, gy = grid[..., 0], grid[..., 1]
        ix = (gx + 1) * w / 2 - 0.5
        iy = (gy + 1) * h / 2 - 0.5
        ix0, iy0 = np.floor(ix).astype(np.int64), np.floor(iy).astype(np.int64)
        ix1, iy1 = ix0 + 1, iy0 + 1
        wx1, wy1 = ix - ix0, iy - iy0
        wx0, wy0 = 1 - wx1, 1 - wy1

        def gather(yy, xx):
            valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            xx_c, yy_c = np.clip(xx, 0, w - 1), np.clip(yy, 0, h - 1)
            out = img[:, yy_c, xx_c]  # (C, Nq, Np)
            return out * valid[None, :, :]

        out = (
            gather(iy0, ix0) * (wy0 * wx0)[None]
            + gather(iy0, ix1) * (wy0 * wx1)[None]
            + gather(iy1, ix0) * (wy1 * wx0)[None]
            + gather(iy1, ix1) * (wy1 * wx1)[None]
        )
        return out  # (C, Nq, Np)

    output = np.zeros((bs, num_queries, heads, head_dim), dtype=np.float64)
    for b_i in range(bs):
        for h_i in range(heads):
            level_samples = []
            for level, (h_l, w_l) in enumerate(spatial_shapes):
                img = (
                    splits[level][b_i, :, h_i, :]
                    .transpose(1, 0)
                    .reshape(head_dim, h_l, w_l)
                )
                grid = sampling_grids[b_i, :, h_i, level]  # (num_queries, points, 2)
                level_samples.append(
                    bilinear_sample(img, grid)
                )  # (head_dim, num_queries, points)
            stacked = np.stack(
                level_samples, axis=-2
            )  # (head_dim, num_queries, levels, points)
            weights = attention_weights[b_i, :, h_i]  # (num_queries, levels, points)
            weighted = stacked * weights[None]
            output[b_i, :, h_i, :] = (
                weighted.reshape(head_dim, num_queries, -1).sum(-1).transpose(1, 0)
            )
    return output.reshape(bs, num_queries, heads * head_dim)


def test_ms_deform_attn_sample_matches_pytorch_reference():
    rng = np.random.default_rng(42)
    batch, num_queries, num_heads, head_dim = 2, 3, 2, 4
    spatial_shapes = [(3, 2), (2, 2)]
    num_levels, num_points = len(spatial_shapes), 3
    sum_hw = sum(h * w for h, w in spatial_shapes)

    value = rng.standard_normal((batch, sum_hw, num_heads, head_dim)).astype(np.float32)
    sampling_locations = rng.uniform(
        0.0, 1.0, (batch, num_queries, num_heads, num_levels, num_points, 2)
    ).astype(np.float32)
    raw_weights = rng.standard_normal(
        (batch, num_queries, num_heads, num_levels, num_points)
    )
    attention_weights = np.exp(raw_weights) / np.exp(raw_weights).sum(
        axis=(-1, -2), keepdims=True
    )
    attention_weights = attention_weights.astype(np.float32)

    expected = _ms_deform_attn_reference_np(
        value, spatial_shapes, sampling_locations, attention_weights
    )

    b = _Builder()
    value_in, loc_in, aw_in = "value", "sampling_locations", "attention_weights"
    out = _ms_deform_attn_sample(
        b,
        value_in,
        spatial_shapes,
        loc_in,
        aw_in,
        batch=batch,
        num_queries=num_queries,
        num_heads=num_heads,
        head_dim=head_dim,
        num_levels=num_levels,
        num_points=num_points,
        prefix="test",
    )
    graph = onnx.helper.make_graph(
        b.nodes,
        "ms_deform_attn_test",
        [
            onnx.helper.make_tensor_value_info(
                value_in, onnx.TensorProto.FLOAT, [batch, sum_hw, num_heads, head_dim]
            ),
            onnx.helper.make_tensor_value_info(
                loc_in,
                onnx.TensorProto.FLOAT,
                [batch, num_queries, num_heads, num_levels, num_points, 2],
            ),
            onnx.helper.make_tensor_value_info(
                aw_in,
                onnx.TensorProto.FLOAT,
                [batch, num_queries, num_heads, num_levels, num_points],
            ),
        ],
        [
            onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT, [batch, num_queries, num_heads * head_dim]
            )
        ],
        initializer=b.initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
    onnx.checker.check_model(model)

    sess = ReferenceEvaluator(model)
    (actual,) = sess.run(
        None, {value_in: value, loc_in: sampling_locations, aw_in: attention_weights}
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Independent check of denormalize_bbox + the atan2 workaround.


def test_denormalize_bbox_matches_reference():
    rng = np.random.default_rng(7)
    boxes = rng.standard_normal((5, 10)).astype(np.float32) * 0.3

    def reference_denormalize(b):
        rot = np.arctan2(b[..., 6:7], b[..., 7:8])
        cx, cy, cz = b[..., 0:1], b[..., 1:2], b[..., 4:5]
        w, length, h = np.exp(b[..., 2:3]), np.exp(b[..., 3:4]), np.exp(b[..., 5:6])
        vx, vy = b[..., 8:9], b[..., 9:10]
        return np.concatenate([cx, cy, cz, w, length, h, rot, vx, vy], axis=-1)

    expected = reference_denormalize(boxes)

    b = _Builder()
    boxes_in = "boxes"
    out = _denormalize_bbox(b, boxes_in, "test")
    graph = onnx.helper.make_graph(
        b.nodes,
        "denorm_test",
        [onnx.helper.make_tensor_value_info(boxes_in, onnx.TensorProto.FLOAT, [5, 10])],
        [onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, [5, 9])],
        initializer=b.initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
    onnx.checker.check_model(model)
    sess = ReferenceEvaluator(model)
    (actual,) = sess.run(None, {boxes_in: boxes})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_atan2_matches_numpy_away_from_singularity():
    rng = np.random.default_rng(9)
    # Avoid the x<=0, y=0 ray the half-angle formula is singular on --
    # documented in _atan2's own docstring as the one excluded case.
    x = rng.uniform(0.2, 2.0, size=50).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=50
    ).astype(np.float32)
    y = rng.uniform(0.2, 2.0, size=50).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=50
    ).astype(np.float32)
    expected = np.arctan2(y, x)

    b = _Builder()
    x_in, y_in = "x", "y"
    out = _atan2(b, y_in, x_in, "test")
    graph = onnx.helper.make_graph(
        b.nodes,
        "atan2_test",
        [
            onnx.helper.make_tensor_value_info(y_in, onnx.TensorProto.FLOAT, [50]),
            onnx.helper.make_tensor_value_info(x_in, onnx.TensorProto.FLOAT, [50]),
        ],
        [onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, [50])],
        initializer=b.initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
    onnx.checker.check_model(model)
    sess = ReferenceEvaluator(model)
    (actual,) = sess.run(None, {y_in: y, x_in: x})
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
