#!/usr/bin/env bash
# Set up an aarch64 Ubuntu host (tested on a Radxa Dragon Q8B) with an Axelera
# Metis AIPU so onnxsim's real-device Voyager tests can run:
#
#   1. Metis kernel driver (metis-dkms) from Axelera's apt repository.
#   2. Voyager SDK (quick-start.sh) in ./venv_<version> under $VOYAGER_DIR.
#   3. onnxsim wheel built from this checkout with Ninja + ccache, installed
#      into the SDK venv so axelera.compiler and onnxsim share one environment.
#
# Every step is idempotent: re-running skips work that is already done.
# Known SDK problems on Ubuntu 26.04 (operator build flags, PyGObject
# Gst.init) are NOT patched here; see docs/axelera-q8b-device-testing.md.
#
# Usage:
#   scripts/axelera/q8b_device_setup.sh
#
# Environment overrides:
#   VOYAGER_DIR   Voyager SDK checkout           (default: ~/voyager-sdk)
#   WHEEL_DIR     where the onnxsim wheel goes   (default: ~/wheels)
#   CCACHE_DIR    ccache directory               (default: ~/.cache/ccache)
#   APT_SUITE     Axelera apt suite              (default: ubuntu24)
#   PYTHON_BIN    Python 3.10-3.13 for the SDK   (default: first match on PATH)
#   SKIP_DRIVER=1 skip step 1
#   SKIP_SDK=1    skip step 2
#   SKIP_ONNXSIM=1 skip step 3

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VOYAGER_DIR="${VOYAGER_DIR:-$HOME/voyager-sdk}"
WHEEL_DIR="${WHEEL_DIR:-$HOME/wheels}"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
APT_SUITE="${APT_SUITE:-ubuntu24}"
SKIP_DRIVER="${SKIP_DRIVER:-0}"
SKIP_SDK="${SKIP_SDK:-0}"
SKIP_ONNXSIM="${SKIP_ONNXSIM:-0}"

say() { printf '\033[0;32m==> %s\033[0m\n' "$*"; }
die() { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Base packages
# ---------------------------------------------------------------------------
say "Installing base build tools (ccache, cmake, ninja, python3-venv)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ccache cmake ninja-build git python3-venv build-essential

# ---------------------------------------------------------------------------
# 1. Metis kernel driver
# ---------------------------------------------------------------------------
if [[ "$SKIP_DRIVER" != 1 ]]; then
    say "Configuring Axelera apt repository (suite: $APT_SUITE)"
    sudo install -d -m 0755 /etc/apt/keyrings
    if [[ ! -s /etc/apt/keyrings/axelera.gpg ]]; then
        curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public \
            | sudo gpg --dearmor -o /etc/apt/keyrings/axelera.gpg
    fi
    echo "deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source $APT_SUITE main" \
        | sudo tee /etc/apt/sources.list.d/axelera.list >/dev/null
    sudo apt-get update

    say "Installing metis-dkms (builds the module for every installed kernel)"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y metis-dkms
    sudo modprobe metis

    if ! lsmod | grep -q '^metis '; then
        die "metis module did not load; check 'sudo dmesg | grep -i metis'"
    fi
    say "Metis driver $(cat /sys/class/metis/version 2>/dev/null || echo '?') loaded"

    # The apt udev rule (72-axelera.rules) only applies on device add events.
    # Re-trigger so existing nodes get the axelera group, then add this user.
    sudo udevadm trigger --subsystem-match=metis --action=add || true
    sudo usermod -aG axelera "${USER:-$(id -un)}" || true
    if ! ls -l /dev/metis-* 2>/dev/null | grep -q axelera; then
        echo "WARNING: /dev/metis-* is not group 'axelera' yet. Reboot, or run:" >&2
        echo "  sudo chgrp axelera /dev/metis-*-*-* && sudo chmod 660 /dev/metis-*-*-*" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 2. Voyager SDK
# ---------------------------------------------------------------------------
if [[ "$SKIP_SDK" != 1 ]]; then
    if [[ ! -d "$VOYAGER_DIR/.git" && ! -f "$VOYAGER_DIR/quick-start.sh" ]]; then
        say "Cloning Voyager SDK into $VOYAGER_DIR"
        git clone --depth 1 https://github.com/axelera-ai-hub/voyager-sdk.git "$VOYAGER_DIR"
    fi
    SDK_VERSION="$(< "$VOYAGER_DIR/VERSION.TXT")"
    SDK_VENV="$VOYAGER_DIR/venv_$SDK_VERSION"

    if [[ ! -d "$SDK_VENV" ]]; then
        # quick-start.sh calls plain `python3`; the SDK supports 3.10-3.13 only.
        if [[ -z "${PYTHON_BIN:-}" ]]; then
            for cand in python3.13 python3.12 python3.11 python3.10; do
                if command -v "$cand" >/dev/null; then PYTHON_BIN="$(command -v "$cand")"; break; fi
            done
        fi
        [[ -n "${PYTHON_BIN:-}" ]] || die "no Python 3.10-3.13 found; set PYTHON_BIN (e.g. from 'uv python install 3.12')"
        shim="$(mktemp -d)"
        ln -s "$PYTHON_BIN" "$shim/python3"
        say "Running quick-start.sh with $("$PYTHON_BIN" --version)"
        (cd "$VOYAGER_DIR" && PATH="$shim:$PATH" ./quick-start.sh) \
            || die "quick-start.sh failed; see docs/axelera-q8b-device-testing.md (known SDK build issues)"
    else
        say "SDK venv $SDK_VENV exists, skipping quick-start.sh"
    fi
fi

# ---------------------------------------------------------------------------
# 3. onnxsim wheel (ccache + Ninja), installed into the SDK venv
# ---------------------------------------------------------------------------
if [[ "$SKIP_ONNXSIM" != 1 ]]; then
    SDK_VERSION="$(< "$VOYAGER_DIR/VERSION.TXT")"
    SDK_PY="$VOYAGER_DIR/venv_$SDK_VERSION/bin/python3"
    [[ -x "$SDK_PY" ]] || die "SDK venv python not found at $SDK_PY (run step 2 first)"

    say "Initializing submodules (recursive: onnx-optimizer carries its own onnx/blake3)"
    git -C "$REPO_DIR" submodule update --init --recursive

    mkdir -p "$WHEEL_DIR" "$CCACHE_DIR"
    ccache -z >/dev/null

    # setup.py resolves cmake with shutil.which(). The SDK venv ships a cmake
    # script that needs its own Python module, which fails inside pip's build
    # isolation, so keep the venv off PATH for the build.
    say "Building onnxsim wheel (Ninja + ccache)"
    (
        cd "$REPO_DIR"
        export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
        export CMAKE_GENERATOR=Ninja
        export CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        "$SDK_PY" -m pip wheel . --no-deps -w "$WHEEL_DIR"
    )
    WHEEL="$(ls -t "$WHEEL_DIR"/onnxsim-*.whl | head -1)"
    say "Installing $WHEEL into the SDK venv"
    "$SDK_PY" -m pip install --force-reinstall --no-deps "$WHEEL"
    "$SDK_PY" -c "import axelera.compiler, onnxsim; print('axelera.compiler and onnxsim import OK')"
    ccache -s | sed -n '1,4p'
fi

say "Done. Run the device tests with the SDK venv active:"
echo "  source $VOYAGER_DIR/venv_\$(cat $VOYAGER_DIR/VERSION.TXT)/bin/activate"
echo "  cd $REPO_DIR && python -m pytest tests/test_axelera_voyager_real_compiler.py -v"
