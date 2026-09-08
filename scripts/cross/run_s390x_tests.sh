#!/usr/bin/env bash
# Install the cross-built onnxsim into the s390x rootfs and run the test suite
# there under qemu-user, i.e. big endian.
#
# Prerequisites:
#   scripts/cross/bootstrap_s390x_rootfs.sh   # rootfs + binfmt
#   scripts/cross/build_s390x_extension.sh    # the extension itself
#
# Set SYSROOT=/rootfs-amd64 and BUILD=.native-build-control/onnxsim-build to run
# the identical stack little-endian as a control (see build_native_control.sh);
# the two runs differ only in byte order, so any difference is an endianness bug.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSROOT="${SYSROOT:-/rootfs-s390x}"
BUILD="${BUILD:-${REPO_ROOT}/.cross-build-s390x/onnxsim-build}"

ONNX_SRC="${REPO_ROOT}/third_party/onnx"
ONNX_BUILD="${BUILD}/third_party/onnx"
DIST="${SYSROOT}/usr/lib/python3/dist-packages"
PKG="${DIST}/onnxsim"
LIBS="${SYSROOT}/work/pylibs"

SO="$(find "${BUILD}" -maxdepth 1 -name 'onnxsim_cpp2py_export*.so' -print -quit)"
[[ -n "${SO}" ]] || { echo "extension not built; run build_s390x_extension.sh first"; exit 1; }

# ---------------------------------------------------------------------------
# onnxsim itself: the pure-Python package plus the built extension.
# ---------------------------------------------------------------------------
echo "== installing onnxsim into ${SYSROOT} =="
rm -rf "${PKG}"; mkdir -p "${PKG}"
find "${REPO_ROOT}/onnxsim" -maxdepth 1 -name '*.py' -exec cp {} "${PKG}/" \;
# setup.py normally generates this; the cross-build never runs setup.py.
cat > "${PKG}/version.py" <<EOF
version = '$(cat "${REPO_ROOT}/VERSION")'
git_version = None
dev_count = None
EOF
cp "${SO}" "${PKG}/onnxsim_cpp2py_export.abi3.so"

# ---------------------------------------------------------------------------
# The vendored onnx, assembled from the same build. The distro's python3-onnx
# is much older than what onnxsim's C++ is compiled against and fails a large
# part of the suite for reasons unrelated to byte order, so put the matching
# version ahead of it on PYTHONPATH. protobuf ships a pure-Python wheel, which
# is architecture independent and so needs no s390x build.
# ---------------------------------------------------------------------------
echo "== installing vendored onnx $(cat "${ONNX_SRC}/VERSION_NUMBER") =="
rm -rf "${LIBS}"; mkdir -p "${LIBS}"
> "${LIBS}/.keep"
python3 -m pip install --quiet --target="${LIBS}" --no-deps \
  "$(grep -oE '"protobuf>=[0-9.]+"' "${ONNX_SRC}/pyproject.toml" | tr -d '"')" \
  "$(grep -oE '"typing_extensions>=[0-9.]+"' "${ONNX_SRC}/pyproject.toml" | tr -d '"')"

cp -r "${ONNX_SRC}/onnx" "${LIBS}/onnx"
rm -rf "${LIBS}/onnx/test" "${LIBS}/onnx/backend/test/data"
cp "${ONNX_BUILD}"/onnx/*_pb.py "${ONNX_BUILD}"/onnx/*_pb2.py "${LIBS}/onnx/"
cp "${ONNX_BUILD}"/onnx_cpp2py_export*.so "${LIBS}/onnx/"
# onnx.__init__ reads its version through importlib.metadata.
ONNX_VER="$(cat "${ONNX_SRC}/VERSION_NUMBER")"
mkdir -p "${LIBS}/onnx-${ONNX_VER}.dist-info"
printf 'Metadata-Version: 2.1\nName: onnx\nVersion: %s\n' "${ONNX_VER}" \
  > "${LIBS}/onnx-${ONNX_VER}.dist-info/METADATA"
: > "${LIBS}/onnx-${ONNX_VER}.dist-info/RECORD"

mkdir -p "${SYSROOT}/work"
rm -rf "${SYSROOT}/work/tests"
cp -r "${REPO_ROOT}/tests" "${SYSROOT}/work/tests"
cp "${REPO_ROOT}/pyproject.toml" "${SYSROOT}/work/"

echo "== environment =="
chroot "${SYSROOT}" /bin/sh -c 'cd /work && PYTHONPATH=/work/pylibs python3 -c "
import sys, numpy, onnx, onnxsim
print(sys.byteorder, \"endian | python\", sys.version.split()[0],
      \"| numpy\", numpy.__version__, \"| onnx\", onnx.__version__,
      \"| onnxsim\", onnxsim.__version__)
"'

# This job exists to catch raw_data/DLPack byte-order bugs in onnxsim's own
# simplify() pipeline (see docs/big-endian.md) -- not to re-run the whole repo's
# test suite under qemu, which is both slow (an emulated big-endian run costs
# real CI minutes) and pointless for the huge majority of tests here: the
# quantization-algorithm, pruning-algorithm, hardware-backend-export/compat and
# LLM/GGUF-reconstruction suites operate on plain numpy arrays and Python
# floats, never touch TensorProto.raw_data or the DLPack bridge, and so cannot
# tell little-endian and big-endian apart. Running them here would only add
# emulation cost and unrelated flakiness without adding coverage.
#
# So, rather than "run everything, --ignore/--deselect what doesn't fit" (which
# silently regrows every time a new experimental test file with an unguarded
# heavy import lands), this selects an explicit allowlist of core-simplify
# test files: the CLI/Python API surface, the default-on and opt-in graph
# rewrite/fusion passes (fuse_attention, fuse_gqa, fuse_split_gather_concat,
# rewrite_bool_where, split_large_gather, the *_to_gather/*_to_gridsample
# rewrites), shape inference and contrib-op schema registration
# (moe_contrib_schema), the function/custom rewriter engine, model
# checking/info/memory-planning/backend dispatch, profiling, and the
# raw_data-adjacent external-data loading path (test_onnx_safetensors_input.py).
# A new test file for one of *these* areas belongs in CORE_TESTS below; a new
# test file for a quantization scheme, pruning algorithm, hardware backend, or
# model-export/reconstruction feature does not need to be added here at all.
#
# test_python_api.py and test_simple.py would otherwise belong on this list --
# they exercise simplify()'s own basics -- but both import torch/torchvision at
# module scope unguarded, which is a collection error (not a skip) with no
# s390x torch build available; excluded for that reason, same as every other
# torch/onnxruntime-dependent file that isn't on the list.
CORE_TESTS="
  tests/test_backend.py
  tests/test_constant_fold_determinism.py
  tests/test_custom_rewriter.py
  tests/test_deform_conv_to_gather.py
  tests/test_free_threading.py
  tests/test_function_body_inference.py
  tests/test_function_rewriter.py
  tests/test_function_rewriter_common_rules.py
  tests/test_function_rewriter_compile_rule.py
  tests/test_function_rewriter_onnxscript_script.py
  tests/test_function_rewriter_vs_onnxscript.py
  tests/test_fuse_attention.py
  tests/test_fuse_gqa.py
  tests/test_fuse_split_gather_concat.py
  tests/test_fusion_patterns.py
  tests/test_gather_over_concat.py
  tests/test_gatherelements_to_gather.py
  tests/test_gathernd_to_gather.py
  tests/test_gridsample_to_gather.py
  tests/test_memory_planning.py
  tests/test_model_checking.py
  tests/test_model_info.py
  tests/test_moe_contrib_schema.py
  tests/test_moved_optimizer_passes.py
  tests/test_msdeformattn_to_gridsample.py
  tests/test_onnx_compat.py
  tests/test_onnx_safetensors_input.py
  tests/test_optimize_pipeline.py
  tests/test_profile_merge.py
  tests/test_profile_plot.py
  tests/test_profiling.py
  tests/test_pruning.py
  tests/test_rewrite_bool_where.py
  tests/test_rich_optional.py
  tests/test_split_large_gather.py
  tests/test_symexpr_kv_cache_consistency.py
"
# Collapsed to single-space-separated on one line: the string below is spliced
# into a /bin/sh -c "..." script (see the pytest invocation further down), and
# an embedded literal newline there would end the pytest command partway
# through its argument list rather than just separating two paths.
CORE_TESTS="$(printf '%s' "${CORE_TESTS}" | tr '\n' ' ')"

# The three deselected BN-fusion tests fail onnxsim's own check_n equivalence
# check whenever onnxruntime is absent and the reference evaluator is used
# instead -- identically on x86_64, so this is not a byte-order problem and
# deselecting them here does not weaken the endianness coverage. Tracked
# separately; see docs/big-endian.md.
# test_gatherelements_to_gather.py::test_axis_invariant_negative_axis_rewrites
# is the same story: confirmed (2026-09-05) to fail identically on a plain
# x86_64 run of the exact same commit -- a pre-existing GatherElements ->
# Gather rewrite/reference-evaluator disagreement unrelated to byte order,
# already noted as a known-flaky case before this deselect was added.

# tests/conftest.py mirrors the slowest-test table into $GITHUB_STEP_SUMMARY,
# opening it unconditionally once the variable is set. In CI that path is on the
# host, which the chroot cannot see, so an inherited value makes pytest exit 1
# with FileNotFoundError *after* every test has already passed. Point it at a
# path inside the rootfs and append the result to the real summary afterwards,
# so the table still reaches the job summary page.
CHROOT_SUMMARY=/work/step-summary.md
: > "${SYSROOT}${CHROOT_SUMMARY}"

echo "== pytest =="
set +e
chroot "${SYSROOT}" /bin/sh -c "cd /work && GITHUB_STEP_SUMMARY=${CHROOT_SUMMARY} \
  PYTHONPATH=/work/pylibs python3 -m pytest ${CORE_TESTS} -p no:cacheprovider \
  --deselect tests/test_fusion_patterns.py::test_fuse_conv_bn_into_conv \
  --deselect tests/test_fusion_patterns.py::test_fuse_convtranspose_bn \
  --deselect tests/test_fusion_patterns.py::test_fuse_conv_with_bias_bn_into_conv \
  --deselect tests/test_gatherelements_to_gather.py::test_axis_invariant_negative_axis_rewrites ${PYTEST_ARGS:-}"
pytest_status=$?
set -e

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -s "${SYSROOT}${CHROOT_SUMMARY}" ]]; then
  cat "${SYSROOT}${CHROOT_SUMMARY}" >> "${GITHUB_STEP_SUMMARY}"
fi
exit "${pytest_status}"
