import sys, json
from pathlib import Path
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_profile as lp
from axelera.runtime import Context
M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(1)
W = (rng.standard_normal((N, K, 1, 1)) * 0.05).astype(np.float32)
g = helper.make_graph([helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[1, 1])], "gemm_as_conv",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, K, M, 1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N, M, 1])],
    initializer=[numpy_helper.from_array(W, "W")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
out = Path(f"gemmconv-{M}-{K}-{N}"); out.mkdir(exist_ok=True)
mj = lp.compile_layer(m, out, 8)
st = lp.measure(Context(), mj, 200)
ops = 2 * M * K * N
print(json.dumps({"M": M, "K": K, "N": N, "kernel_us": st["kernel_us_median"], "tops_kernel": ops / (st["kernel_us_median"] * 1e-6) / 1e12}), flush=True)
