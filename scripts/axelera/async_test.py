import sys, json
from pathlib import Path
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
from axelera import compiler
from axelera.runtime import Context
import layer_profile as lp
M, K, N, flag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
rng = np.random.default_rng(1)
W = (rng.standard_normal((N, K, 1, 1)) * 0.05).astype(np.float32)
g = helper.make_graph([helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, K, M, 1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N, M, 1])],
    initializer=[numpy_helper.from_array(W, "W")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
out = Path(f"async-{M}-{K}-{N}-{int(flag)}"); out.mkdir(exist_ok=True)
onnx.save(m, out / "g.onnx")
calib = (rng.uniform(-1, 1, (1, K, M, 1)).astype(np.float32) for _ in range(8))
grp = len(sys.argv) > 5 and sys.argv[5] == "1"
cfg = compiler.CompilerConfig(inter_operator_async=flag, group_ifdw_tasks=grp)
q = compiler.quantize(model=str(out / "g.onnx"), calibration_dataset=calib, config=cfg)
compiler.compile(model=q, config=cfg, output_dir=out / "compiled")
st = lp.measure(Context(), out / "compiled" / "model.json", 200)
ops = 2 * M * K * N
print(json.dumps({"flag": flag, "kernel_us": st["kernel_us_median"], "tops": ops / (st["kernel_us_median"] * 1e-6) / 1e12}), flush=True)
