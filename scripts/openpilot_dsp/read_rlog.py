"""Pull calibration and on-device model timings out of openpilot rlogs (.zst).

  python read_rlog.py --cereal <openpilot>/openpilot/cereal --car-capnp-dir <dir with opendbc's car.capnp
      and an include/ dir holding pycapnp's c++.capnp> seg3_rlog.zst ...

Prints initData version/device, the mean `extrinsicsCalibration.rpyCalib` (what modeld feeds
get_warp_matrix), and modelV2 / driverStateV2 execution times as measured on the device that drove.
"""

import argparse
import os

import capnp
import numpy as np
import zstandard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cereal", required=True)
    ap.add_argument("--car-capnp-dir", required=True)
    ap.add_argument("rlogs", nargs="+")
    args = ap.parse_args()
    capnp.remove_import_hook()
    log = capnp.load(
        os.path.join(args.cereal, "log.capnp"), imports=[args.car_capnp_dir]
    )
    for path in args.rlogs:
        raw = (
            zstandard.ZstdDecompressor()
            .decompressobj()
            .decompress(open(path, "rb").read())
        )
        cal, mt, dt, dg, ver = [], [], [], [], None
        for ev in log.Event.read_multiple_bytes(raw):
            w = ev.which()
            if w == "initData":
                ver = (
                    ev.initData.version,
                    ev.initData.gitCommit[:10],
                    str(ev.initData.deviceType),
                )
            elif w == "extrinsicsCalibration":
                cal.append(list(ev.extrinsicsCalibration.rpyCalib))
            elif w == "modelV2":
                mt.append(ev.modelV2.modelExecutionTime)
            elif w == "driverStateV2":
                dt.append(ev.driverStateV2.modelExecutionTime)
                dg.append(ev.driverStateV2.gpuExecutionTime)

        def q(a):
            return (
                (
                    f"n={len(a)} mean={np.mean(a) * 1e3:.2f} p50={np.median(a) * 1e3:.2f} p95={np.percentile(a, 95) * 1e3:.2f} "
                    f"max={np.max(a) * 1e3:.2f} ms"
                )
                if a
                else None
            )

        print(path, ver)
        print("  rpyCalib mean", np.round(np.mean(cal, 0), 5).tolist() if cal else None)
        print("  modelV2.modelExecutionTime", q(mt))
        print("  driverStateV2.modelExecutionTime", q(dt), "| gpuExecutionTime", q(dg))


if __name__ == "__main__":
    main()
