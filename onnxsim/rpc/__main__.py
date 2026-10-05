"""``python -m onnxsim.rpc server|tracker``: start an RPC server or a tracker."""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m onnxsim.rpc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser(
        "server", help="run models on this machine for remote clients"
    )
    server.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: loopback only)"
    )
    server.add_argument("--port", type=int, default=9090)
    server.add_argument(
        "--key",
        default="",
        help="device key clients must present (also the tracker key)",
    )
    server.add_argument(
        "--workspace", default=None, help="directory for uploaded files"
    )
    server.add_argument(
        "--tracker", default=None, metavar="HOST:PORT", help="register with a tracker"
    )
    server.add_argument(
        "--advertise", default=None, help="address to advertise to the tracker"
    )
    server.add_argument("--verbose", action="store_true")
    server.add_argument(
        "--xdna-python",
        default=sys.executable,
        help="server-side Python with IRON/XRT installed (default: this server's Python)",
    )
    server.add_argument(
        "--vitis-python",
        default=None,
        help="server-side Python with Vitis AI Execution Provider (default: --xdna-python)",
    )
    server.add_argument(
        "--tpu-ssh-host",
        default=None,
        help="enable TPU-MLIR execution and run compiled models on this SG2002 over SSH",
    )
    server.add_argument("--tpu-ssh-user", default="root")
    server.add_argument("--tpu-ssh-port", type=int, default=22)
    server.add_argument(
        "--tpu-password-env",
        default="TPU_MLIR_SSH_PASSWORD",
        help="environment variable containing the board SSH password",
    )
    server.add_argument("--tpu-chip", default="cv181x")
    server.add_argument("--tpu-quantize", default="BF16")
    server.add_argument(
        "--tpu-calibration-table",
        default=None,
        help="default activation calibration table, required when compiling INT8",
    )
    server.add_argument(
        "--tpu-python",
        default=sys.executable,
        help="Python environment containing the TPU-MLIR tools",
    )
    server.add_argument("--tpu-model-transform", default="model_transform.py")
    server.add_argument("--tpu-model-deploy", default="model_deploy.py")
    server.add_argument("--tpu-runner", default="/usr/bin/model_runner")
    server.add_argument("--tpu-library-dir", default="/usr/bin/lib")
    server.add_argument("--tpu-remote-dir", default="/data/onnxsim-tpu-rpc")
    tracker = sub.add_parser("tracker", help="run a device-key tracker")
    tracker.add_argument("--host", default="127.0.0.1")
    tracker.add_argument("--port", type=int, default=9190)
    args = parser.parse_args()

    if args.command == "server":
        from .server import RPCServer

        rpc_server = RPCServer(
            args.host,
            args.port,
            args.key,
            args.workspace,
            verbose=args.verbose,
            xdna_python=args.xdna_python,
            vitis_python=args.vitis_python,
            tpu_mlir_options={
                "ssh_host": args.tpu_ssh_host,
                "ssh_user": args.tpu_ssh_user,
                "ssh_port": args.tpu_ssh_port,
                "password_env": args.tpu_password_env,
                "chip": args.tpu_chip,
                "quantize": args.tpu_quantize,
                "calibration_table": args.tpu_calibration_table,
                "python": args.tpu_python,
                "model_transform": args.tpu_model_transform,
                "model_deploy": args.tpu_model_deploy,
                "runner": args.tpu_runner,
                "library_dir": args.tpu_library_dir,
                "remote_dir": args.tpu_remote_dir,
            },
        )
        if args.tracker:
            host, _, port = args.tracker.rpartition(":")
            rpc_server.register_with_tracker((host, int(port)), args.advertise)
        print(
            f"onnxsim RPC server on {rpc_server.address[0]}:{rpc_server.address[1]} key={args.key!r}",
            flush=True,
        )
        rpc_server.serve_forever()
    else:
        from .tracker import Tracker

        rpc_tracker = Tracker(args.host, args.port).start()
        print(
            f"onnxsim RPC tracker on {rpc_tracker.address[0]}:{rpc_tracker.address[1]}",
            flush=True,
        )
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
