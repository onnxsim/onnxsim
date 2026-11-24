"""DMA size verification (compile only). For each config, compare the descriptor size fields with their formulas:
  input bytes   = cin * hin * hin * 2          (task.dma.0.0 word 286)
  output bytes  = cout * hout * hout * 2       (task.dma.0.1 word 6)
  weight bytes  = 2 * cout * R + 256 * ceil(cout / 16), R = roundup(cin * k * k, 16)   (task.dma.1.0 word 6)
Configs: 1x1 and 3x3 (pad 1), cin in {8, 16, 32, 64}, cout in {8, 16, 32, 64, 96}, hin in {8, 16}. Run on H8 after reloc.py."""
import os
import sys
import numpy as np
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import reloc as R


def word(data, secs, name, j):
    s = secs[name]
    return int(np.frombuffer(data[s["off"] + 4 * j: s["off"] + 4 * j + 4], dtype=np.uint32)[0])


ok = bad = 0
for k, pad in ((1, 0), (3, 1)):
    for cin in (8, 16, 32, 64):
        for cout in (8, 16, 32, 64, 96):
            for hin in (8, 16):
                hout = hin + 2 * pad - k + 1
                inp = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, cin, hin, hin])
                out = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, cout, hout, hout])
                W = numpy_helper.from_array((np.ones((cout, cin, k, k))).astype(np.float16), "W")
                node = helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k, k], pads=[pad] * 4)
                path = R.compile_model("dsz_%d_%d_%d_%d_%d" % (k, cin, cout, hin, pad), [node], [W], inp, out)
                data = open(path, "rb").read()
                secs = {s["name"]: s for s in R.elf_sections(data)}
                got_in = word(data, secs, "task.dma.0.0", 286)
                got_out = word(data, secs, "task.dma.0.1", 6)
                got_w = word(data, secs, "task.dma.1.0", 6)
                R_ = -(-(cin * k * k) // 16) * 16
                pred_in = cin * hin * hin * 2
                pred_out = cout * hout * hout * 2
                pred_w = 2 * cout * R_ + 256 * -(-cout // 16)
                good = (got_in, got_out, got_w) == (pred_in, pred_out, pred_w)
                ok += good
                bad += (not good)
                if not good:
                    print("MISMATCH k%d cin%d cout%d hin%d: in %d/%d out %d/%d w %d/%d (got/pred)" % (
                        k, cin, cout, hin, got_in, pred_in, got_out, pred_out, got_w, pred_w), flush=True)
print("configs matching all three formulas: %d, mismatched: %d" % (ok, bad), flush=True)
