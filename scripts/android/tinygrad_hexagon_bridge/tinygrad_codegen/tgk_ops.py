"""Three ops written as plain tinygrad Tensor code (no custom_kernel, no hand-written C), at the real Mask R-CNN shapes
the hand kernels in .. were built for, plus the matching hand-kernel calls and numpy references.

Plain versions are what tinygrad's own DSP codegen produces once the onnxsim/tinygrad `hvx-codegen` changes are in
(re-vectorized ALU, HVX-width upcast, store-width-capped load coalescing, per-line prefetch)."""
import os, sys
import numpy as np
from tinygrad import Tensor, dtypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# a representative Q31 requantize: (multiplier, shift) as TVM's GetFixedPointMultiplierShift produces them
MULT, SHIFT, IN_ZP, OUT_ZP = 1518500250, -8, 0, 3

SHAPES = {  # small: fast qemu iteration / HEXSIM; real: the backbone shapes the hand kernels target
  "add":     {"small": (1, 256, 25, 34), "real": (1, 256, 200, 272)},
  "bias_add": {"small": (25 * 34, 64), "real": (400 * 544, 64)},
  "relu":     {"small": (1 << 16,), "real": (2_000_000,)},
  "maxpool": {"small": (2, 50, 68),      "real": (2, 400, 544)},     # (ic_chunks, H, W), packed NCHWc, 32-ch blocks
  "requant": {"small": (1 << 16,),       "real": (2_000_000,)},
}

def inputs(op:str, size:str) -> list[np.ndarray]:
  shp = SHAPES[op][size]
  if op == "add":
    rng = np.random.default_rng(0)
    return [rng.integers(-2**20, 2**20, shp).astype(np.int32) for _ in range(2)]
  if op == "bias_add":
    rng = np.random.default_rng(3)
    pos, cout = shp
    return [rng.integers(-2**20, 2**20, (pos, cout)).astype(np.int32),
            rng.integers(-2**12, 2**12, (cout,)).astype(np.int32)]
  if op == "relu":
    return [np.random.default_rng(4).integers(-(1 << 20), 1 << 20, shp).astype(np.int32)]
  if op == "maxpool":
    ic, H, W = shp
    # pre-padded by 1 with 0 (uint8's minimum, so padding never wins a max)
    x = np.random.default_rng(1).integers(0, 256, (1, ic, H + 2, W + 2, 32)).astype(np.uint8)
    x[:, :, 0], x[:, :, -1], x[:, :, :, 0], x[:, :, :, -1] = 0, 0, 0, 0
    return [x]
  if op == "requant":
    return [np.random.default_rng(2).integers(-2**24, 2**24, shp).astype(np.int32)]
  raise ValueError(op)

def _pool_views(x, OH, OW, maximum):
  out = None
  for kh in range(3):
    for kw in range(3):
      v = x[:, :, kh:kh + 2*OH:2, kw:kw + 2*OW:2, :]
      out = v if out is None else maximum(out, v)
  return out

def reference(op:str, ins:list[np.ndarray]) -> np.ndarray:
  if op == "add": return ins[0] + ins[1]
  if op == "bias_add": return ins[0] + ins[1][None, :]
  if op == "relu": return np.maximum(ins[0], 0)
  if op == "maxpool":
    x = ins[0]; return _pool_views(x, (x.shape[2]-2)//2, (x.shape[3]-2)//2, np.maximum)
  if op == "requant":
    ls, rs = max(SHIFT, 0), max(-SHIFT, 0); ts = rs + 31
    t = (ins[0].astype(np.int64) - IN_ZP) << ls
    return np.clip(((t * MULT + (np.int64(1) << (ts - 1))) >> ts) + OUT_ZP, 0, 255).astype(np.uint8)
  raise ValueError(op)

def plain(op:str, ins:list[np.ndarray]) -> Tensor:
  """The op as ordinary tinygrad Tensor code -- what this directory is about."""
  if op == "add": return Tensor(ins[0]) + Tensor(ins[1])
  if op == "bias_add": return Tensor(ins[0]) + Tensor(ins[1]).reshape(1, -1)
  if op == "relu": return Tensor(ins[0]).maximum(0)
  if op == "maxpool":
    x = ins[0]; return _pool_views(Tensor(x), (x.shape[2]-2)//2, (x.shape[3]-2)//2, lambda a, b: a.maximum(b))
  if op == "requant":
    ls, rs = max(SHIFT, 0), max(-SHIFT, 0); ts = rs + 31
    t = (Tensor(ins[0]).cast(dtypes.int64) - IN_ZP) << ls
    return (((t * MULT + (1 << (ts - 1))) >> ts) + OUT_ZP).clip(0, 255).cast(dtypes.uint8)
  raise ValueError(op)

def hand(op:str, ins:list[np.ndarray]) -> Tensor:
  """The existing hand-written custom_kernel for the same op (hex_*_kernel.py), for comparison."""
  if op == "add":
    import hex_add_kernel as K
    return K.build_vector_kernel(ins[0].size, Tensor(ins[0].reshape(-1)), Tensor(ins[1].reshape(-1)))
  if op == "bias_add":
    import hex_bias_add_kernel as K
    pos, cout = ins[0].shape
    return K.build_kernel(pos, cout, Tensor(ins[0].reshape(-1)), Tensor(ins[1]))
  if op == "relu":
    import hex_relu_kernel as K
    return K.build_kernel(ins[0].size, Tensor(ins[0]))
  if op == "maxpool":
    import hex_maxpool_kernel as K
    _, ic, Hp, Wp, _ = ins[0].shape
    return K.build_kernel(ic, Hp - 2, Wp - 2, 3, 2, 1, Tensor(ins[0].reshape(-1)))
  if op == "requant":
    import hex_requantize_kernel as K
    return K.build_kernel(ins[0].size, MULT, SHIFT, IN_ZP, OUT_ZP, Tensor(ins[0]))
  raise ValueError(op)
