#!/bin/bash
# Materialize the hand-written Hexagon kernel headers from the onnxsim/tinygrad fork.
#
# These headers (HMX block/gemm/qconv, the ResNet runner's model+exec+load, MSDA, the RPN family's
# NMS/proposal-decode/TopK, RoiAlign fp32+u8, the FPN layout) live in exactly one place now:
# onnxsim/tinygrad test/external/dsp/hand/, where they are the oracle that tinygrad's generated
# kernels are checked against. Keeping a second copy under scripts/android/ meant a kernel fix could
# land in one tree and not the other, which is the failure mode oracles exist to prevent.
#
# So they are not committed here. This script fetches the pinned revision into a cache directory and
# symlinks each header into place, so the build scripts' `cp "$SRC"/<header>` and the sources'
# `#include "<header>"` keep working unchanged.
#
#   ./fetch_hand_kernels.sh              # populate from $TINYGRAD_SHA (or the pin below)
#   TINYGRAD_SHA=<sha> ./fetch_hand_kernels.sh
#   TINYGRAD_PATH=<checkout> ./fetch_hand_kernels.sh   # use a local checkout, no network
#
# Idempotent: an already-correct link is left alone. Needs git and network on first use.
set -euo pipefail
# this script sits in scripts/android/, and the links go next to it
ANDROID="$(cd "$(dirname "$0")" && pwd)"

TINYGRAD_REPO="${TINYGRAD_REPO:-https://github.com/onnxsim/tinygrad.git}"
# Keep in step with .github/workflows/hexagon-tinygrad.yml (TINYGRAD_SHA). dsp-consolidated:
# the whole Hexagon DSP stack, including test/external/dsp/hand/.
TINYGRAD_SHA="${TINYGRAD_SHA:-c817073e5558918ee08e7f7c0cfc638d8af41478}"
HAND="test/external/dsp/hand"

# header under scripts/android -> path under the fork's hand/
LINKS="
hmx_gemm/hmx_block.h:hmx/hmx_block.h
hmx_gemm/hmx_gemm.h:hmx/hmx_gemm.h
hmx_gemm/hmx_gemm_u8.h:hmx/hmx_gemm_u8.h
hmx_gemm/hmx_runtime.h:hmx/hmx_runtime.h
hmx_gemm/hmx_qconv.h:hmx/hmx_qconv.h
hmx_gemm/hmx_qconv3.h:hmx/hmx_qconv3.h
hmx_gemm/qc_case.h:hmx/qc_case.h
hmx_gemm/runner/rn_exec.h:hmx/runner/rn_exec.h
hmx_gemm/runner/rn_load.h:hmx/runner/rn_load.h
hmx_gemm/runner/rn_model.h:hmx/runner/rn_model.h
msda_hvx/msda_kernel.h:msda/msda_kernel.h
msda_hvx/msda_shape.h:msda/msda_shape.h
tinygrad_hexagon_bridge/fpn_channels_last/layout_kernels.h:layout/layout_kernels.h
tinygrad_hexagon_bridge/nms/nms_kernel.h:rpn/nms_kernel.h
tinygrad_hexagon_bridge/proposal_decode/pd_kernel.h:rpn/pd_kernel.h
tinygrad_hexagon_bridge/roialign_fast/roialign_kernel.h:roialign/roialign_kernel.h
tinygrad_hexagon_bridge/roialign_fast/roialign_u8_kernel.h:roialign/roialign_u8_kernel.h
tinygrad_hexagon_bridge/topk/topk_kernel.h:rpn/topk_kernel.h
"

# A local checkout wins: it is what a developer iterating on the fork has, and it needs no network.
root=""
if [ -n "${TINYGRAD_PATH:-}" ] && [ -d "$TINYGRAD_PATH/$HAND" ]; then
  root="$TINYGRAD_PATH"
  echo "using TINYGRAD_PATH=$root"
else
  root="${TINYGRAD_CACHE:-$HOME/.cache/tinygrad-dsp-hand-$TINYGRAD_SHA}"
  if [ ! -d "$root/$HAND" ]; then
    echo "fetching the hand kernels from $TINYGRAD_REPO @ $TINYGRAD_SHA"
    rm -rf "$root"
    git init -q "$root"
    git -C "$root" fetch -q --depth 1 "$TINYGRAD_REPO" "$TINYGRAD_SHA"
    git -C "$root" checkout -q FETCH_HEAD
  fi
  echo "using $root"
fi

[ -d "$root/$HAND" ] || { echo "no hand kernels at $root/$HAND" >&2; exit 1; }
n=0
while IFS=: read -r rel hand_rel; do
  [ -n "$rel" ] || continue
  src="$root/$HAND/$hand_rel"
  [ -f "$src" ] || { echo "missing in the fork: $HAND/$hand_rel" >&2; exit 1; }
  dst="$ANDROID/$rel"
  mkdir -p "$(dirname "$dst")"
  # rm first: `ln -sfn` will not replace an existing regular file, and these paths used to hold a
  # real copy of the header before it moved to the fork.
  rm -f "$dst"
  # A relative link, so a checkout that has run this stays valid wherever it is moved to. The target
  # is inside the cache dir (or a developer's TINYGRAD_PATH), which is reached from here by relpath.
  ln -s "$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$src" "$(dirname "$dst")")" "$dst"
  n=$((n + 1))
done <<< "$LINKS"
echo "linked $n hand-kernel headers from the tinygrad fork"
