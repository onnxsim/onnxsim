#!/bin/bash
# Install the demo APK and give it its models and test images.
#   ./deploy.sh                    models copied on-device from ../e2e_pipeline's /data/local/tmp/e2e
#   MODELS=<build_models.py --out dir> ./deploy.sh      models pushed from the host instead
#   IMGS="a.jpg b.jpg ..." ./deploy.sh                  JPEGs for the "images" test mode
#   YOLO="<deploy work>/yolo26n/pipe/yolo26n.onnx ..." ./deploy.sh   models for the YOLO mode
#   TG="<export_cl.py bundle dir, named <model>.tg> ..." ./deploy.sh   tinygrad engines (--es opts engine=tinygrad)
#   RFDETR=<rfdetr export.py model, e.g. ~/.cache/onnxsim-rfdetr/work/nano@320.u8.onnx> ./deploy.sh
#                                   the RF-DETR button (pushed as rfdetr_nano.onnx)
#   RTDETR=<rtdetr split.py work dir, e.g. ~/.cache/onnxsim-rtdetr/work/split> ./deploy.sh   RT-DETR
#   SAM=<sam.py work dir, e.g. ~/.cache/onnxsim-sam/efficientvit_sam_l0> ./deploy.sh   the SAM mode
#   MCC=<mcc.py work dir, e.g. ~/.cache/onnxsim-mcc/work> MOGE=<depth.py static dir, e.g. ~/.cache/onnxsim-mcc/moge>
#   MCC_HMX=<../mcc_hmx/ref.py weights dir> ./deploy.sh   the MCC 3D mode (also needs the SAM mode's models: SAM=...);
#                                   MCC_HMX: the DSP decoder's weights (the default decoder, opts dec=hmx)
#   SR=<superres.py models dir, e.g. ~/.cache/superres/models> ./deploy.sh   the super-resolution mode
#   GAME=<game_seq.py --out dir, e.g. ~/.cache/arm-nss/game> ./deploy.sh   the game-upscaling mode (NSS + NFRU)
# Then:  adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity [--es mode images] [--es pipe pipe_e_opt.txt]
#        adb shell am start -n org.onnxsim.maskrcnndemo/.YoloActivity [--es mode images] [--es model yolo11n]
#        adb shell am start -n org.onnxsim.maskrcnndemo/.SamActivity [--es mode images] [--es tap 0.5,0.5]
#        adb shell am start -n org.onnxsim.maskrcnndemo/.MccActivity [--es mode images] [--es tap 0.5,0.5 --ez recon true]
#        adb shell am start -n org.onnxsim.maskrcnndemo/.SrActivity [--es mode images] [--es ref original]
#        adb shell am start -n org.onnxsim.maskrcnndemo/.GameActivity [--ez nfru true]
#
# Files go to the app's *internal* files dir through `run-as` (the APK is debuggable): files adb
# puts under /sdcard/Android/data/<pkg> are owned by the shell user and unreadable by the app.
set -euo pipefail
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
A=(adb -s "$DEVICE_SERIAL")
PKG=org.onnxsim.maskrcnndemo
HERE="$(cd "$(dirname "$0")" && pwd)"
APK="$HERE/app/build/outputs/apk/debug/app-debug.apk"
STAGE="${STAGE:-/data/local/tmp/maskrcnn_demo_stage}"
"${A[@]}" install -r -g "$APK"            # -g grants CAMERA
"${A[@]}" shell am force-stop $PKG
RA() { "${A[@]}" shell "run-as $PKG sh -c '$1'"; }
RA "mkdir -p files/models files/imgs"
FILES="pipe_e_u8ra_ctx.txt pipe_e_u8ra.txt pipe_e_u8_ctx.txt pipe_e_u8.txt pipe_e_opt.txt pipe_e_opt_ctx.txt levels.txt model.txt l0_anchors.bin l1_anchors.bin l2_anchors.bin l3_anchors.bin
l4_anchors.bin backbone_opt.onnx adapt_scores.onnx seg_e_opt_1.onnx seg_e_opt_2.onnx seg_e_opt_3.onnx seg_e_opt_4.onnx
seg_e_opt_5.onnx box_head_1000_nhwc.onnx mask_head_32_nhwc.onnx mask_head_100_nhwc.onnx
box_head_1000_u8.onnx mask_head_32_u8.onnx mask_head_100_u8.onnx"
if [ -n "${MODELS:-}" ]; then
  SRC=$STAGE/models
  "${A[@]}" shell "mkdir -p $SRC"
  for f in $FILES; do "${A[@]}" push -q "$MODELS/$f" "$SRC/"; done
else
  SRC="${DEVICE_MODELS:-/data/local/tmp/e2e}"
fi
for f in $FILES; do RA "cmp -s $SRC/$f files/models/$f || cp $SRC/$f files/models/"; done
# EP-context models compiled earlier on this phone (e2e_run writes them on its first ctx run); if
# absent, the app compiles them itself on its first launch with that pipe (JIT cost, once) and
# reuses them after. NO_CTX=1 skips the copy (to measure that first launch).
if [ -z "${NO_CTX:-}" ]; then
  for f in backbone_opt box_head_1000_nhwc mask_head_32_nhwc mask_head_100_nhwc; do
    RA "[ ! -f $SRC/$f.ctx.onnx ] || cmp -s $SRC/$f.ctx.onnx files/models/$f.ctx.onnx || cp $SRC/$f.ctx.onnx files/models/"
  done
  for f in backbone_opt.ctx0.onnx backbone_opt.ctx0_qnn.bin box_head_1000_u8.ctx0.onnx box_head_1000_u8.ctx0_qnn.bin \
           mask_head_32_u8.ctx0.onnx mask_head_32_u8.ctx0_qnn.bin mask_head_100_u8.ctx0.onnx mask_head_100_u8.ctx0_qnn.bin; do
    RA "[ ! -f $SRC/$f ] || cmp -s $SRC/$f files/models/$f || cp $SRC/$f files/models/"
  done
fi
# YOLO mode: deploy-pipeline YOLO models (../deploy: <work>/<name>/pipe/<name>.onnx, uint8 NHWC in,
# the head's (1, 4+nc, N) out), e.g. YOLO="$HOME/.cache/onnxsim-deploy/yolo26n/pipe/yolo26n.onnx ..."
# The app compiles each one's EP-context model (<name>.ctx0.onnx) on its first launch.
if [ -n "${YOLO:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/yolo"
  for f in $YOLO; do
    "${A[@]}" push -q "$f" "$STAGE/yolo/"
    b=$(basename "$f")
    RA "cmp -s $STAGE/yolo/$b files/models/$b || { cp $STAGE/yolo/$b files/models/ && rm -f files/models/${b%.onnx}.ctx0*; }"
  done
fi
# tinygrad AOT OpenCL bundles (../tinygrad_aot/export_cl.py output dirs, named <model>.tg, e.g.
# TG="yolo11n.tg"): what engine=tinygrad loads instead of the HTP session (--es opts engine=tinygrad)
if [ -n "${TG:-}" ]; then
  for d in $TG; do
    b=$(basename "$d")
    "${A[@]}" shell "mkdir -p $STAGE/tg/$b"
    for f in kernels.cl plan.txt consts.bin meta.txt; do "${A[@]}" push -q "$d/$f" "$STAGE/tg/$b/"; done
    RA "mkdir -p files/models/$b && for f in kernels.cl plan.txt consts.bin meta.txt; do cmp -s $STAGE/tg/$b/\$f files/models/$b/\$f || cp $STAGE/tg/$b/\$f files/models/$b/; done"
  done
fi
# RF-DETR (the YOLO activity's post=detr): one strict-HTP model from ../vision_models/rfdetr (uint8
# NHWC SxS in, logits + boxes out), stored as rfdetr_nano.onnx.
if [ -n "${RFDETR:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/yolo"
  "${A[@]}" push -q "$RFDETR" "$STAGE/yolo/rfdetr_nano.onnx"
  RA "cmp -s $STAGE/yolo/rfdetr_nano.onnx files/models/rfdetr_nano.onnx || { cp $STAGE/yolo/rfdetr_nano.onnx files/models/ && rm -f files/models/rfdetr_nano.ctx0*; }"
fi
# SAM mode: EfficientViT-SAM-L0 from ../vision_models/sam (sam.py export + quantize; its work dir,
# e.g. SAM=$HOME/.cache/onnxsim-sam/efficientvit_sam_l0): enc.fp16.onnx -> sam_l0_enc.onnx,
# dec.sim.onnx -> sam_l0_dec.onnx. EP-context models are compiled on the app's first SAM launch.
if [ -n "${SAM:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/sam"
  "${A[@]}" push -q "$SAM/enc.fp16.onnx" "$STAGE/sam/sam_l0_enc.onnx"
  "${A[@]}" push -q "$SAM/dec.sim.onnx" "$STAGE/sam/sam_l0_dec.onnx"
  for b in sam_l0_enc.onnx sam_l0_dec.onnx; do
    RA "cmp -s $STAGE/sam/$b files/models/$b || { cp $STAGE/sam/$b files/models/ && rm -f files/models/${b%.onnx}.ctx0*; }"
  done
fi
# MCC 3D mode: MCC's pieces from ../vision_models/mcc (mcc.py export --chunks 1024, then dec_opt.py prep +
# quant --policy a16c; MCC=its work dir): enc.onnx -> mcc_enc.onnx, the w8a16 decoder dec_q1024.a16c.onnx
# (MCC_DEC=dec_q1024.onnx for the fp16 one) -> mcc_dec_q1024.onnx; and MoGe-2 ViT-S in both orientations
# (depth.py static --h 640 --w 480 and --h 480 --w 640; MOGE=their dir) -> moge_640x480.onnx,
# moge_480x640.onnx. The segmentation is the SAM mode's sam_l0_enc/dec. EP-context models are compiled on
# the app's first MCC launch (MoGe-2 on its first use per orientation).
if [ -n "${MCC:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/mcc"
  "${A[@]}" push -q "$MCC/enc.onnx" "$STAGE/mcc/mcc_enc.onnx"
  "${A[@]}" push -q "$MCC/${MCC_DEC:-dec_q1024.a16c.onnx}" "$STAGE/mcc/mcc_dec_q1024.onnx"
  for hw in 640x480 480x640; do "${A[@]}" push -q "$MOGE/model.$hw.t1200.onnx" "$STAGE/mcc/moge_$hw.onnx"; done
  for b in mcc_enc.onnx mcc_dec_q1024.onnx moge_640x480.onnx moge_480x640.onnx; do
    RA "cmp -s $STAGE/mcc/$b files/models/$b || { cp $STAGE/mcc/$b files/models/ && rm -f files/models/${b%.onnx}.ctx0*; }"
  done
fi
# MCC 3D mode's DSP decoder (../mcc_hmx, ref.py weights --out <dir>): blk0..7.bin, head.bin -> mcc_hmx_*.bin
if [ -n "${MCC_HMX:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/mcc"
  for b in blk0 blk1 blk2 blk3 blk4 blk5 blk6 blk7 head; do
    "${A[@]}" push -q "$MCC_HMX/$b.bin" "$STAGE/mcc/mcc_hmx_$b.bin"
    RA "cmp -s $STAGE/mcc/mcc_hmx_$b.bin files/models/mcc_hmx_$b.bin || cp $STAGE/mcc/mcc_hmx_$b.bin files/models/"
  done
fi
# RT-DETR mode: the pieces from ../vision_models/rtdetr/msda_hvx/split.py (export + quant --policy
# bb8enc16 --u8-values), e.g. RTDETR=$HOME/.cache/onnxsim-rtdetr/work/split: pre.bb8enc16.v8.onnx,
# mid0/mid1/post.sim.onnx -> rtdetr_{pre,mid0,mid1,post}.onnx
if [ -n "${RTDETR:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/rtdetr"
  "${A[@]}" push -q "$RTDETR/pre.bb8enc16.v8.onnx" "$STAGE/rtdetr/rtdetr_pre.onnx"
  for n in mid0 mid1 post; do "${A[@]}" push -q "$RTDETR/$n.sim.onnx" "$STAGE/rtdetr/rtdetr_$n.onnx"; done
  for n in pre mid0 mid1 post; do
    RA "cmp -s $STAGE/rtdetr/rtdetr_$n.onnx files/models/rtdetr_$n.onnx || { cp $STAGE/rtdetr/rtdetr_$n.onnx files/models/ && rm -f files/models/rtdetr_$n.ctx0*; }"
  done
fi
# Super-resolution mode: x4 models from ../vision_models/superres (superres.py build <model> 270 480
# and 480 270), e.g. SR=$HOME/.cache/superres/models: <model>_<HxW>/{int8,fp16}.onnx ->
# sr_<name>_<prec>_<HxW>.onnx, both orientations. EP-context models are compiled on first use.
if [ -n "${SR:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/sr"
  for spec in xlsr:xlsr:int8 quicksrnetmedium:qsrm:int8 xlsr:xlsr:fp16 realesr-animevideov3:esrgan:int8; do
    IFS=: read -r m n p <<<"$spec"
    for hw in 270x480 480x270; do
      b="sr_${n}_${p}_$hw.onnx"
      "${A[@]}" push -q "$SR/${m}_$hw/$p.onnx" "$STAGE/sr/$b"
      RA "cmp -s $STAGE/sr/$b files/models/$b || { cp $STAGE/sr/$b files/models/ && rm -f files/models/${b%.onnx}.ctx0*; }"
    done
  done
fi
# Game-upscaling mode: the replay sequence from game_seq.py (GAME=its --out dir, e.g. ~/.cache/arm-nss/game:
# game_seq.bin + Arm's license) and the two int8 networks, NSS's CNN (NSS_ONNX, default
# ~/.cache/arm-nss/onnx/cnn_int8_qat.onnx) and NFRU's (NFRU_ONNX, default ~/.cache/arm-nfru/onnx/net_int8_qat.onnx)
if [ -n "${GAME:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/game"
  "${A[@]}" push -q "$GAME/game_seq.bin" "$GAME/LICENSE_Arm_AI_Model_Community.pdf" "$STAGE/game/"
  "${A[@]}" push -q "${NSS_ONNX:-$HOME/.cache/arm-nss/onnx/cnn_int8_qat.onnx}" "$STAGE/game/game_nss_cnn.onnx"
  "${A[@]}" push -q "${NFRU_ONNX:-$HOME/.cache/arm-nfru/onnx/net_int8_qat.onnx}" "$STAGE/game/game_nfru_net.onnx"
  for b in game_seq.bin LICENSE_Arm_AI_Model_Community.pdf; do
    RA "cmp -s $STAGE/game/$b files/models/$b || cp $STAGE/game/$b files/models/"
  done
  for b in game_nss_cnn.onnx game_nfru_net.onnx; do
    RA "cmp -s $STAGE/game/$b files/models/$b || { cp $STAGE/game/$b files/models/ && rm -f files/models/${b%.onnx}.ctx0*; }"
  done
fi
if [ -n "${IMGS:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/imgs"
  for i in $IMGS; do "${A[@]}" push -q "$i" "$STAGE/imgs/"; done
  RA "cp $STAGE/imgs/*.jpg files/imgs/"
fi
RA "ls -la files/models files/imgs"
