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

# tests/test_check_decode_parity.py imports check_decode_parity from
# scripts/apple/ at module scope (see that test file's sys.path.insert).
# Copy just that one module in, not the rest of scripts/apple/ -- its
# siblings (export_llm_to_coreml.py, coreml_backend.py, ...) pull in
# coremltools/torch/transformers, which have no s390x build.
# check_decode_parity.py itself only imports argparse/sys/pathlib, so it's
# safe to run standalone here.
rm -rf "${SYSROOT}/work/scripts"
mkdir -p "${SYSROOT}/work/scripts/apple"
cp "${REPO_ROOT}/scripts/apple/check_decode_parity.py" "${SYSROOT}/work/scripts/apple/"

echo "== environment =="
chroot "${SYSROOT}" /bin/sh -c 'cd /work && PYTHONPATH=/work/pylibs python3 -c "
import sys, numpy, onnx, onnxsim
print(sys.byteorder, \"endian | python\", sys.version.split()[0],
      \"| numpy\", numpy.__version__, \"| onnx\", onnx.__version__,
      \"| onnxsim\", onnxsim.__version__)
"'

# torch, timm and onnxruntime have no s390x builds, so the test modules that
# import them at module scope cannot be collected here. test_qnn_compat,
# test_coreml_compat, test_migraphx_compat and test_openvino_compat each
# import their vendor backend module (e.g. ``coreml_backend``) and a ``models``
# fixture module from a sibling scripts/<vendor>/ directory at module scope --
# above, only tests/ and pyproject.toml are copied into the chroot, so none of
# those imports resolve there regardless of what's pip-installed. Same story
# for test_check_decode_parity.py, which imports scripts/apple/check_decode_parity
# at module scope, and test_compute_retention.py, test_run_quality_eval.py and
# test_aggregate_quality_trend.py, which import scripts/apple/compute_retention,
# scripts/apple/run_quality_eval and scripts/apple/aggregate_quality_trend the
# same way. test_rtdetrv4.py imports onnxruntime directly at module scope for
# the same "no s390x build" reason as test_qnn_compat.py and friends above.
# test_pulsar2_compat.py and test_pulsar2_simulator.py import pulsar2_backend/
# pulsar2_simulator from scripts/axera/ at module scope, the identical "vendor
# module not copied into the chroot" reason as the four vendor-compat tests
# above.
#
# The three deselected BN-fusion tests fail onnxsim's own check_n equivalence
# check whenever onnxruntime is absent and the reference evaluator is used
# instead -- identically on x86_64, so this is not a byte-order problem and
# deselecting them here does not weaken the endianness coverage. Tracked
# separately; see docs/big-endian.md. Everything else must pass, so a genuine
# big-endian regression still turns this red.

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
  PYTHONPATH=/work/pylibs python3 -m pytest tests -p no:cacheprovider \
  --ignore=tests/test_mnn_llm_export.py --ignore=tests/test_python_api.py --ignore=tests/test_rfdetr.py \
  --ignore=tests/test_simple.py --ignore=tests/test_timm.py \
  --ignore=tests/test_torch_export_integration.py --ignore=tests/test_yolo.py \
  --ignore=tests/test_gan.py \
  --ignore=tests/test_qnn_compat.py --ignore=tests/test_modelopt_integration.py \
  --ignore=tests/test_mmdeploy_integration.py \
  --ignore=tests/test_coreml_compat.py --ignore=tests/test_migraphx_compat.py \
  --ignore=tests/test_openvino_compat.py --ignore=tests/test_check_decode_parity.py \
  --ignore=tests/test_compute_retention.py --ignore=tests/test_rtdetrv4.py \
  --ignore=tests/test_run_quality_eval.py --ignore=tests/test_aggregate_quality_trend.py \
  --ignore=tests/test_pulsar2_compat.py --ignore=tests/test_pulsar2_simulator.py \
  --deselect tests/test_fusion_patterns.py::test_fuse_conv_bn_into_conv \
  --deselect tests/test_fusion_patterns.py::test_fuse_convtranspose_bn \
  --deselect tests/test_fusion_patterns.py::test_fuse_conv_with_bias_bn_into_conv ${PYTEST_ARGS:-}"
pytest_status=$?
set -e

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -s "${SYSROOT}${CHROOT_SUMMARY}" ]]; then
  cat "${SYSROOT}${CHROOT_SUMMARY}" >> "${GITHUB_STEP_SUMMARY}"
fi
exit "${pytest_status}"
