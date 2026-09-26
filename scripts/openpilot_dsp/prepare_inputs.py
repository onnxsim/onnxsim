"""Turn a public openpilot route's camera files into model inputs, the way modeld does.

Decodes `fcamera.hevc` (narrow road), `ecamera.hevc` (wide road) and `dcamera.hevc` (driver) with
ffmpeg and applies modeld's warps in numpy:

- driving: `get_warp_matrix(rpy_calib, intrinsics, bigmodel_frame)` (camera_from_model, used as
  `M_inv` by tinygrad's `compile_warp.py`), nearest-neighbour with clamped coordinates, then
  `frames_to_tensor`'s 6-channel YUV420 layout -> `new_img[2, 6, 128, 256]` uint8 per frame
  (index 0 = narrow road `img`, 1 = wide road `big_img`).
- driver monitoring: `cam.intrinsics @ inv(dmonitoringmodel_intrinsics)`, luma only, border fill 16
  -> `input_img[1, 1440*960]` uint8.

Needs the openpilot source tree on PYTHONPATH (for `openpilot.common.transformations`); only numpy
is imported from it. Writes `<out>.npz` with `road` [N,2,6,128,256] and `driver` [N,1,1382400].
"""

import argparse
import subprocess

import numpy as np
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import (
    DM_INPUT_SIZE,
    dmonitoringmodel_intrinsics,
    get_warp_matrix,
)


def decode_yuv420(path, w, h, n, start=0):
    """Returns [n, h*3/2, w] uint8 planar I420 frames (Y plane, then U, then V)."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"select=gte(n\\,{start})",
        "-vsync",
        "0",
        "-frames:v",
        str(n),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    fs = w * h * 3 // 2
    got = len(raw) // fs
    return np.frombuffer(raw[: got * fs], np.uint8).reshape(got, h * 3 // 2, w)


def warp_nn(src, m, dst_w, dst_h, border=None):
    """compile_warp.warp_perspective_tinygrad in numpy: dst(x,y) = src(round(M @ [x,y,1]))."""
    h, w = src.shape
    x, y = np.meshgrid(
        np.arange(dst_w, dtype=np.float32), np.arange(dst_h, dtype=np.float32)
    )
    m = m.astype(np.float32)
    sw = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    sx = np.round((m[0, 0] * x + m[0, 1] * y + m[0, 2]) / sw)
    sy = np.round((m[1, 0] * x + m[1, 1] * y + m[1, 2]) / sw)
    out = src[
        np.clip(sy, 0, h - 1).astype(np.int64), np.clip(sx, 0, w - 1).astype(np.int64)
    ]
    if border is not None:
        inb = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
        out = np.where(inb, out, np.uint8(border))
    return out


def road_frame_to_tensor(frame, m, cam_w, cam_h, model_w=512, model_h=256):
    y = frame[:cam_h]
    u = frame[cam_h : cam_h + cam_h // 4].reshape(cam_h // 2, cam_w // 2)
    v = frame[cam_h + cam_h // 4 :].reshape(cam_h // 2, cam_w // 2)
    m_uv = m * np.array([[1, 1, 0.5], [1, 1, 0.5], [2, 2, 1]], np.float32)
    yw = warp_nn(y, m, model_w, model_h)
    uw = warp_nn(u, m_uv, model_w // 2, model_h // 2)
    vw = warp_nn(v, m_uv, model_w // 2, model_h // 2)
    return np.stack(
        [yw[0::2, 0::2], yw[1::2, 0::2], yw[0::2, 1::2], yw[1::2, 1::2], uw, vw]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--road", required=True, help="fcamera.hevc")
    ap.add_argument("--wide", required=True, help="ecamera.hevc")
    ap.add_argument("--driver", required=True, help="dcamera.hevc")
    ap.add_argument("--device", default="mici", help="DEVICE_CAMERAS device type")
    ap.add_argument("--sensor", default="os04c10")
    ap.add_argument(
        "--rpy-calib",
        default="0,0,0",
        help="device_from_calib euler; the route's liveCalibration "
        "is not read, so the nominal 0,0,0 is used by default",
    )
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dc = DEVICE_CAMERAS[(args.device, args.sensor)]
    rpy = np.array([float(v) for v in args.rpy_calib.split(",")], np.float32)
    m_main = get_warp_matrix(rpy, dc.narrow_road.intrinsics, False)
    m_wide = get_warp_matrix(rpy, dc.wide_road.intrinsics, True)
    m_dm = dc.cabin.intrinsics @ np.linalg.inv(dmonitoringmodel_intrinsics)

    w, h = dc.narrow_road.width, dc.narrow_road.height
    road = decode_yuv420(args.road, w, h, args.frames, args.start)
    wide = decode_yuv420(args.wide, w, h, args.frames, args.start)
    drv = decode_yuv420(
        args.driver, dc.cabin.width, dc.cabin.height, args.frames, args.start
    )
    n = min(len(road), len(wide), len(drv))
    road_t = np.stack(
        [
            np.stack(
                [
                    road_frame_to_tensor(road[i], m_main, w, h),
                    road_frame_to_tensor(wide[i], m_wide, w, h),
                ]
            )
            for i in range(n)
        ]
    )
    dm_w, dm_h = DM_INPUT_SIZE
    drv_t = np.stack(
        [
            warp_nn(drv[i][: dc.cabin.height], m_dm, dm_w, dm_h, border=16).reshape(
                1, -1
            )
            for i in range(n)
        ]
    )
    np.savez_compressed(args.out, road=road_t, driver=drv_t)
    print(
        f"{args.out}: road {road_t.shape}, driver {drv_t.shape}, road mean {road_t.mean():.1f}, driver mean {drv_t.mean():.1f}"
    )


if __name__ == "__main__":
    main()
