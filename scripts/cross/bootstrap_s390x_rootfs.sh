#!/usr/bin/env bash
# Build an Ubuntu s390x (big endian) rootfs and register qemu-user for it, so
# onnxsim can be run and tested big-endian on an x86_64 host.
#
# s390x is the only mainstream big-endian Linux target a major distro still
# ships, which makes it the practical way to check onnxsim's byte-order
# assumptions. Run as root. See README.md for the full procedure.
set -euo pipefail

# ARCH=amd64 builds the little-endian control rootfs instead (same distro, same
# package versions), so a big-endian run can be diffed against one that differs
# only in byte order. Ubuntu keeps s390x on ports.ubuntu.com and amd64 on the
# main archive.
ARCH="${ARCH:-s390x}"
SUITE="${SUITE:-noble}"
if [[ "${ARCH}" == "amd64" ]]; then
  SYSROOT="${SYSROOT:-/rootfs-amd64}"
  MIRROR="${MIRROR:-http://archive.ubuntu.com/ubuntu/}"
else
  SYSROOT="${SYSROOT:-/rootfs-s390x}"
  MIRROR="${MIRROR:-http://ports.ubuntu.com/ubuntu-ports/}"
fi

command -v debootstrap >/dev/null || {
  echo "installing debootstrap + qemu-user-static + the s390x cross toolchain"
  apt-get update -q
  apt-get install -y -q debootstrap qemu-user-static g++-s390x-linux-gnu
}

# ---------------------------------------------------------------------------
# binfmt_misc so s390x binaries (including everything dpkg runs inside the
# chroot) transparently execute under qemu.
# ---------------------------------------------------------------------------
if [[ "${ARCH}" == "s390x" ]]; then
if [[ ! -e /proc/sys/fs/binfmt_misc/register ]]; then
  mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
fi
if [[ -e /proc/sys/fs/binfmt_misc/qemu-s390x ]]; then
  echo -1 > /proc/sys/fs/binfmt_misc/qemu-s390x
fi
# Bytes 0-6 are the ELF ident (64-bit, MSB, v1); 16-17 e_type; 18-19 e_machine
# (0x0016 = EM_S390). The mask zeroes EI_OSABI/EI_ABIVERSION -- glibc's own
# binaries (e.g. ldconfig.real) set EI_OSABI=GNU, and a mask that pins them to
# SYSV makes those, and only those, fail with "Exec format error" mid-install.
# The 0xfe on byte 17 accepts both ET_EXEC and ET_DYN (static-pie binaries).
printf ':qemu-s390x:M::\x7fELF\x02\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x16:\xff\xff\xff\xff\xff\xff\xff\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff:/usr/bin/qemu-s390x-static:F' \
  > /proc/sys/fs/binfmt_misc/register
fi

# ---------------------------------------------------------------------------
# The rootfs. python3-onnx/python3-numpy/python3-pytest exist for s390x in
# Ubuntu, which is what makes this practical -- PyPI has no s390x wheels.
# The distro onnx is older than the one onnxsim vendors; run_s390x_tests.sh
# installs the vendored version over it.
# ---------------------------------------------------------------------------
if [[ ! -e "${SYSROOT}/etc/os-release" ]]; then
  # debootstrap's second stage configures every package under emulation, and
  # that dominates this script: stage 1 (download + native unpack) is ~40s of a
  # ~280s run. So the list is worth keeping tight -- but only where a package is
  # genuinely unused.
  #
  # Dropped, measured at 289s/888 MB -> 281s/694 MB:
  #   python3-onnx, python3-protobuf -- run_s390x_tests.sh installs the vendored
  #     onnx and a pure-Python protobuf ahead of them on PYTHONPATH, so the
  #     distro copies were only ever shadowed (verified: the suite passes in
  #     full with both removed).
  #   libprotobuf-dev, protobuf-compiler -- the cross-build builds its own
  #     protobuf at the version onnx's SBOM pins and points CMAKE_PREFIX_PATH at
  #     it; the rootfs copy was unused, and a different version sitting in the
  #     sysroot is a hazard rather than a help.
  #   cmake, git -- nothing in the rootfs builds with them (see the ml_dtypes
  #     version pin below for why that stays true despite ml_dtypes 0.6.0
  #     switching its own build backend to a CMake-based one).
  #
  # build-essential and python3-pip stay, even though only the ml_dtypes build
  # below needs them and that is normally served from cache. Dropping them does
  # cut the bootstrap to ~215s/384 MB, but then a cache miss has to apt-get them
  # *inside* the chroot, where both unpack and configure run emulated -- far
  # slower than debootstrap's native stage-1 unpack, and enough to swamp the 65s
  # saved. This job runs weekly and GitHub evicts caches after 7 days idle, so
  # misses sit near the norm rather than the exception; paying 65s every run
  # beats paying ten minutes on the ones that miss.
  debootstrap --arch="${ARCH}" --variant=minbase --components=main,universe \
    --include=python3,python3-dev,libpython3-dev,python3-numpy,python3-pytest,ca-certificates,python3-pip,build-essential \
    "${SUITE}" "${SYSROOT}" "${MIRROR}"
fi

cat > "${SYSROOT}/etc/apt/sources.list" <<EOF
deb ${MIRROR} ${SUITE} main universe
deb ${MIRROR} ${SUITE}-updates main universe
EOF
cp /etc/resolv.conf "${SYSROOT}/etc/resolv.conf"
# Proxied environments hand pip a CA bundle by absolute path; make it resolve
# inside the chroot too.
if [[ -f /root/.ccr/ca-bundle.crt ]]; then
  mkdir -p "${SYSROOT}/root/.ccr"
  cp /root/.ccr/ca-bundle.crt "${SYSROOT}/root/.ccr/"
fi

# rich is an optional dependency of onnxsim (without it the reports fall back to
# plain-text tables) and pure Python, so the host's pip can drop it straight in
# and the rootfs exercises the rich path.
python3 -m pip install --quiet --target="${SYSROOT}/usr/lib/python3/dist-packages" rich

# ml_dtypes is a C extension the vendored onnx needs and has no s390x wheel, so
# it has to be compiled inside the rootfs under emulation. That is by far the
# slowest step here, so when WHEELHOUSE points at a directory the built wheel is
# kept there and reused on the next run (CI caches it).
if ! chroot "${SYSROOT}" /usr/bin/python3 -c "import ml_dtypes" 2>/dev/null; then
  mkdir -p "${SYSROOT}/wheelhouse"
  if [[ -n "${WHEELHOUSE:-}" ]]; then
    mkdir -p "${WHEELHOUSE}"
    cp "${WHEELHOUSE}"/*.whl "${SYSROOT}/wheelhouse/" 2>/dev/null || true
  fi
  chroot "${SYSROOT}" /bin/sh -c '
    set -e
    export PIP_BREAK_SYSTEM_PACKAGES=1
    if ! ls /wheelhouse/ml_dtypes-*.whl >/dev/null 2>&1; then
      # noble ships setuptools 68, which rejects ml_dtypes 0.5.x'"'"'s SPDX
      # `project.license`; --no-build-isolation then needs pybind11 present.
      #
      # Pinned below 0.6.0 (onnx itself only requires >=0.5.4, so any 0.5.x
      # satisfies it): 0.6.0 switched its build backend to scikit-build-core,
      # which needs a real `cmake` binary (installable, but see below), *and*
      # its C++ sources now use NumPy 2.x-only C-API symbols
      # (`PyArray_DescrProto`, `PyDataType_GetArrFuncs`) that noble'"'"'s
      # python3-numpy (still 1.x -- there is no s390x NumPy 2 wheel and
      # building one from source here is far too heavy for this bootstrap) does
      # not declare, so the extension fails to compile regardless of the build
      # backend. 0.5.x builds with plain setuptools against the 1.x API and
      # needs neither.
      pip3 install -U --ignore-installed setuptools wheel pybind11
      pip3 wheel --no-build-isolation --no-deps -w /wheelhouse "ml_dtypes>=0.5.4,<0.6.0"
    fi
    pip3 install --no-deps /wheelhouse/ml_dtypes-*.whl
  '
  if [[ -n "${WHEELHOUSE:-}" ]]; then
    cp "${SYSROOT}/wheelhouse"/*.whl "${WHEELHOUSE}/" 2>/dev/null || true
  fi
fi

chroot "${SYSROOT}" /usr/bin/python3 -c "
import sys, numpy
print('rootfs ready:', sys.byteorder, 'endian | python', sys.version.split()[0],
      '| numpy', numpy.__version__)
"
