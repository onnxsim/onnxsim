"""Render the tinygrad fp16 matmul (HMX TensorCore, tinygrad fork branch hvx-hmx) for one shape into tg_kernel.c.

  HMX=1 DEV=DSP MOCKDSP=1 TC=1 CC=clang-19 PYTHONPATH=<tinygrad hvx-hmx> python gen_kernel.py M K N out_dir [--i8]

--i8: a uint8 x int8 -> int32 matmul (the hexagon_hmx_i8 ":cm" TensorCore) instead of fp16.
--conv H W C N S: a QDQ 3x3 conv (stride S) in hmxsim_conv.py's grid form, requantization fused (--relu); M K N are ignored.
      Also writes a phone case (a.bin, b.bin = weights | bias | scale, ref.bin: ORT's formula) to out_dir/case.
--qadd N: ORT's QLinearAdd on N uint8 (hmxsim_add.py's constants; --ties for power-of-two ratios), plus a phone case dir.
--rq: the int8 matmul with ORT's requantization fused (hmxsim_rq.py's layer, zy 131, lo 0 or --relu): uint8 out; the skel
      passes bias (int32 [N]) and scale (fp32 [N]) from the B buffer, after the weights.

Writes out_dir/tg_kernel.c (the kernel function plus its HMX tile op; everything the renderer puts after
"/* DSP boilerplate */" is dropped) and out_dir/tg_kernel.h with the entry name and shape for tg_hmx_impl.c.
"""
import os, sys, re, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hmxsim  # noqa: E402  (installs the MOCKDSP capture hooks)
from tinygrad import Tensor, dtypes  # noqa: E402

M, K, N = (int(x) for x in sys.argv[1:4]); out = pathlib.Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
conv = "--conv" in sys.argv
qadd = "--qadd" in sys.argv
rq = "--rq" in sys.argv or conv; i8 = "--i8" in sys.argv or rq or qadd
if qadd:
  import hmxsim_add
  from tinygrad.runtime.ops_dsp import hmx_qlinear_add
  n = int(sys.argv[sys.argv.index("--qadd") + 1])
  rng = np.random.default_rng(4)
  A = rng.integers(0, 256, n, dtype=np.uint8); B = rng.integers(0, 256, n, dtype=np.uint8)
  ra, rb, fixed = hmxsim_add.consts(0.5, 0.25, 1.0, 3, 0, 1) if "--ties" in sys.argv else hmxsim_add.consts(0.0371, 0.0529, 0.0613, 118, 131, 7)
  (out / "case").mkdir(exist_ok=True)
  A.tofile(out / "case" / "a.bin"); B.tofile(out / "case" / "b.bin"); hmxsim_add.reference(A, B, ra, rb, fixed).tofile(out / "case" / "ref.bin")
  hmx_qlinear_add(Tensor(A), Tensor(B), float(ra), float(rb), float(fixed)).realize()
  M, K, N = 1, n, 1
elif conv:
  import hmxsim_conv
  from hmxsim_rq import reference
  H, W, C_, N_, st = (int(a) for a in sys.argv[sys.argv.index("--conv") + 1:][:5])
  zy = 131; lo = zy if "--relu" in sys.argv else 0
  Wp, P64, L = hmxsim_conv.grid(H, W, st)
  rng = np.random.default_rng(3)
  X = rng.integers(0, 256, (L, C_), dtype=np.uint8); Wt = rng.integers(-128, 128, (3, 3, C_, N_), dtype=np.int8)
  bias = rng.integers(-20000, 20000, N_, dtype=np.int32)
  Am = np.stack([X[dy * Wp + dx: dy * Wp + dx + st * (P64 - 1) + 1: st] for dy in range(3) for dx in range(3)], 1).reshape(P64, 9 * C_)
  acc0 = Am.astype(np.int64) @ Wt.reshape(9 * C_, N_).astype(np.int64) + bias
  mv = (rng.uniform(0.5, 2.0, N_) * 50 / (np.abs(acc0).max(axis=0) + 1)).astype(np.float32); mv[::8] = np.float32(2.0 ** -8)
  (out / "case").mkdir(exist_ok=True)
  X.tofile(out / "case" / "a.bin"); 
  (out / "case" / "b.bin").write_bytes(Wt.tobytes() + bias.tobytes() + mv.tobytes())
  reference(Am, Wt.reshape(9 * C_, N_), bias, mv, zy, lo).tofile(out / "case" / "ref.bin")
  hmxsim_conv.layer(np.zeros((L, C_), np.uint8), np.zeros((3, 3, C_, N_), np.int8), np.zeros(N_, np.int32), np.ones(N_, np.float32),
                    zy, lo, Wp, P64, st).realize()
  M, K, N = P64, 9 * C_, N_
elif rq:
  import hmxsim_rq
  zy = 131; lo = zy if "--relu" in sys.argv else 0
  hmxsim_rq.layer(np.zeros((M, K), np.uint8), np.zeros((K, N), np.int8), np.zeros(N, np.int32), np.ones(N, np.float32), zy, lo).realize()
elif i8: Tensor(np.zeros((M, K), np.uint8)).matmul(Tensor(np.zeros((K, N), np.int8)), dtype=dtypes.int32).realize()
else: Tensor(np.zeros((M, K), np.float16)).matmul(Tensor(np.zeros((K, N), np.float16)), dtype=dtypes.half).realize()
assert len(hmxsim._calls) == 1, f"expected one WMMA kernel, got {len(hmxsim._calls)}"
body = hmxsim._calls[0][0].split("/* DSP boilerplate */")[0]
name = re.search(r"void\s+(qadd_\d+)\(" if qadd else r"noinline\)\) void\s+(\w+)\(", body).group(1)
# tinygrad's int vector typedefs (int32 = 32 lanes of int, ...) collide with the Hexagon SDK's scalar int32 etc. in the skel
body = re.sub(r"\bint(\d+)\b", r"tgint\1", body)
(out / "tg_kernel.c").write_text(body)
ta, tb, tc_ = ("unsigned char", "signed char", "unsigned char" if rq else "int") if i8 else ("__fp16", "__fp16", "__fp16")
if qadd: ta, tb, tc_ = "unsigned char", "unsigned char", "unsigned char"
hdr = (f"#define TG_VTCM_KB {int(os.environ.get('HMX_VTCM_KB', 256))}\n#define TG_M {M}\n#define TG_K {K}\n#define TG_N {N}\n#define TG_I8 {int(i8)}\n#define TG_RQ {int(rq)}\n"
       f"typedef {ta} tg_a_t;\ntypedef {tb} tg_b_t;\ntypedef {tc_} tg_c_t;\n")
hdr += f"#define TG_SA {X.nbytes if conv else M * K}\n" if i8 else ""
if qadd: hdr += f"#define TG_SC_BYTES {n}\n"
if rq:  # B buffer = weights [K, N] | bias int32 [N] | scale fp32 [N]
  hdr += (f"#define TG_ZY {zy}\n#define TG_LO {lo}\n#define TG_B_EXTRA (8 * TG_N)\n"
          f"void {name}(tg_c_t* out, tg_a_t* a, tg_b_t* b, int* bias, float* m);\n"
          f"#define TG_CALL(c, a, b) {name}(c, a, b, (int*)((b) + TG_K * TG_N), (float*)((b) + TG_K * TG_N + 4 * TG_N))\n")
else:
  hdr += f"#define TG_B_EXTRA 0\nvoid {name}(tg_c_t* out, tg_a_t* a, tg_b_t* b);\n#define TG_CALL(c, a, b) {name}(c, a, b)\n"
(out / "tg_kernel.h").write_text(hdr)
print(f"{name}: {len(body)} bytes -> {out}/tg_kernel.c")
