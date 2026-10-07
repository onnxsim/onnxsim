#!/usr/bin/env python3
"""Build and execute a dense FP32 graph through an installed Android NN runtime."""

import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def run(command):
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="ADB device serial")
    parser.add_argument("--ndk", required=True, type=Path, help="Android NDK root")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--stub", default="/vendor/lib64/libhexagon_nn_stub.so")
    parser.add_argument(
        "--dsp-library-path",
        default="/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp",
    )
    parser.add_argument(
        "--uri",
        default="file:///libhexagon_nn_skel.so?hexagon_nn_domains_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    )
    args = parser.parse_args()
    compiler = (
        args.ndk
        / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
    )
    if not compiler.is_file():
        parser.error(f"NDK compiler not found: {compiler}")
    adb = [args.adb, "-s", args.serial]
    with tempfile.TemporaryDirectory(prefix="hexagon-nn-") as temporary:
        binary = Path(temporary) / "dense_inference"
        run(
            [
                compiler,
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                Path(__file__).with_name("dense_inference.c"),
                "-ldl",
                "-lm",
                "-o",
                binary,
            ]
        )
        # Device mktemp avoids overwriting another run's staging files.
        stage = subprocess.check_output(
            adb + ["shell", "mktemp -d /data/local/tmp/hexagon-nn.XXXXXX"], text=True
        ).strip()
        if not re.fullmatch(r"/data/local/tmp/hexagon-nn\.[A-Za-z0-9]+", stage):
            raise RuntimeError(f"Unexpected staging path: {stage!r}")
        try:
            run(adb + ["push", binary, stage + "/dense_inference"])
            command = " ".join(
                [
                    "env",
                    shlex.quote("LD_LIBRARY_PATH=/vendor/lib64"),
                    shlex.quote("ADSP_LIBRARY_PATH=" + args.dsp_library_path),
                    shlex.quote(stage + "/dense_inference"),
                    shlex.quote(args.stub),
                    shlex.quote(args.uri),
                ]
            )
            run(adb + ["shell", command])
        finally:
            run(adb + ["shell", "rm -rf " + shlex.quote(stage)])


if __name__ == "__main__":
    main()
