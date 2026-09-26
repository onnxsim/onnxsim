"""Compile an ONNX model with tinygrad ahead of time into a plain OpenCL bundle the app runs without Python.

  DEV=CL JIT_BATCH_SIZE=0 PYTHONPATH=<tinygrad> python export_cl.py model.onnx out_dir [--u8-nhwc] [--check img.npy]

tinygrad lowers the ONNX graph through OnnxRunner + TinyJit on the host's OpenCL device (any OpenCL 1.2+
device works; the kernels are plain OpenCL C, the launch dimensions are fixed at capture time). The captured
JIT -- the ordered kernel calls, the buffers they touch and their memory plan -- is written out as:

  out_dir/kernels.cl    every kernel's OpenCL C source, concatenated (one clBuildProgram on the phone)
  out_dir/plan.txt      buffers and kernel calls, one per line (format below), read by tg_cl_runner.h
  out_dir/consts.bin    initial contents of every buffer the kernels read before writing (weights)

plan.txt lines (whitespace separated):
  arena <nbytes>                  (first line) one scratch allocation shared by every arena buffer
  buf <id> <nbytes> in <k> | out <k> | const <offset into consts.bin> | persist | arena <offset> | unused
  kern|init <name> <g0> <g1> <g2> <l0> <l1> <l2> <nargs> <arg>...
    arg: b<id> (buffer), i<id>,<h>,<w>,<itemsize> (image2d_t over buffer <id>, RGBA, h x w pixels,
         itemsize 2 = CL_HALF_FLOAT / 4 = CL_FLOAT), v<int> (int scalar)
  global sizes are in work groups, as tinygrad records them: the NDRange is g*l. "init" calls run once after loading
  (they only derive constants from constants, e.g. IMAGE=1's fp16 weight images, into "persist" buffers).

--u8-nhwc wraps the model so its input is the uint8 NHWC image (the QNN engines' input): x/255, NHWC->NCHW.
"""
import argparse, math, re
from dataclasses import replace
from pathlib import Path
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.device import Compiler, CompileError
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.helpers import is_image_shape
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.helpers import FLOAT16

class _TryCompile(Compiler):
  """Build every kernel once with the vendor compiler in a separate process (TG_CLC: the clc tool from this directory)
  before tinygrad uses it. Qualcomm's compiler can crash on a kernel ("Custom lowering code for this instruction is not
  implemented yet: 150") and every later build in the same process then fails ("Program not built!"), so a crash must
  not happen in the exporting process. A crash raises CompileError here, in to_program, where the exporter retries the
  kernel unoptimized. The source itself is what tinygrad keeps: the CL runtime builds from source."""
  def compile(self, src:str) -> bytes:
    import os, subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".cl", delete=False) as f: f.write(src)
    try: res = subprocess.run([os.environ["TG_CLC"], f.name], capture_output=True, text=True)
    finally: os.unlink(f.name)
    if res.returncode != 0 or " OK" not in res.stdout: raise CompileError(f"vendor compiler rejected the kernel: {res.stdout[-300:]}")
    return src.encode()

class AdrenoCLRenderer(OpenCLRenderer):
  """OpenCL C for Qualcomm's OpenCL compiler: QCOMCLRenderer's workarounds (no fp8/bf16/double, bool buffers as uchar),
  but plain source for the vendor OpenCL runtime instead of QCOMCLRenderer's kgsl-only binaries."""
  # native half whenever FLOAT16 is on (QCOMCLRenderer only allows it with IMAGE too): a kernel that falls back to plain
  # buffers (below) keeps its fp16 buffers native instead of emulating them as ushort (2 s instead of ~10 ms on the
  # -seg models' prototype upsampling conv)
  def supported_dtypes(self):
    return {d for d in Renderer.supported_dtypes(self)
            if (d != dtypes.float16 or bool(FLOAT16)) and d not in dtypes.fp8s+(dtypes.bfloat16, dtypes.double)}
  def __init__(self, target):
    super().__init__(target)
    self.compiler = _TryCompile()
  def _render_dtype(self, dtype, sz=1, addrspace=AddrSpace.ALU, mutable=True, override_ptr=False, shape=None):
    if dtype == dtypes.bool and addrspace == AddrSpace.GLOBAL: dtype = dtypes.uint8
    return super()._render_dtype(dtype, sz, addrspace, mutable, override_ptr, shape)

def GridSample(X:Tensor, grid:Tensor, align_corners:int=0, mode:str="linear", padding_mode:str="zeros"):
  """ONNX GridSample, 2D, bilinear, zero padding (what RF-DETR's deformable attention uses; tinygrad's OnnxRunner has
  no GridSample): four gathers from the flattened feature map, masked where a corner falls outside"""
  if mode not in ("linear", "bilinear") or padding_mode != "zeros" or X.ndim != 4:
    raise NotImplementedError(f"GridSample {mode=} {padding_mode=} ndim={X.ndim}")
  N, C, H, W = X.shape
  _, Ho, Wo, _ = grid.shape
  gx, gy = grid[..., 0], grid[..., 1]
  if align_corners: x, y = (gx + 1) * (W - 1) / 2, (gy + 1) * (H - 1) / 2
  else: x, y = ((gx + 1) * W - 1) / 2, ((gy + 1) * H - 1) / 2
  x0, y0 = x.floor(), y.floor()
  flat = X.reshape(N, C, H * W)
  out = None
  for dy in (0, 1):
    for dx in (0, 1):
      xi, yi = x0 + dx, y0 + dy
      w = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
      valid = (xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1)
      idx = (yi.clip(0, H - 1) * W + xi.clip(0, W - 1)).cast(dtypes.int32).reshape(N, 1, Ho * Wo).expand(N, C, Ho * Wo)
      v = flat.gather(2, idx).reshape(N, C, Ho, Wo) * (w * valid).reshape(N, 1, Ho, Wo)
      out = v if out is None else out + v
  return out

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx"); ap.add_argument("out")
  ap.add_argument("--u8-nhwc", action="store_true", help="input is uint8 NHWC pixels (scaled by 1/255 inside the graph)")
  ap.add_argument("--adreno", action="store_true", help="render for Qualcomm's OpenCL compiler (run this on the phone, DEV=CL)")
  ap.add_argument("--fallback-beam", type=int, default=4, help="BEAM width for kernels the vendor compiler rejects")
  ap.add_argument("--check", help="optional raw input file (input dtype/shape) to run through the captured JIT; saves ref_<output>.bin")
  args = ap.parse_args()
  out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
  fallbacks: list[str] = []
  if args.adreno:
    Device[Device.DEFAULT].renderers = [AdrenoCLRenderer]
    from tinygrad.runtime.ops_cl import CLCompiler
    orig = CLCompiler.compile
    def compile_dump(self, src):
      try: return orig(self, src)
      except Exception:
        (out / "failed_kernel.cl").write_text(src)
        raise
    CLCompiler.compile = compile_dump
    # Qualcomm's compiler crashes on a few of tinygrad's optimized kernels ("Custom lowering code for this instruction is
    # not implemented yet: 150", seen on an unrolled conv loop with a masked image-row index). Retry those unoptimized.
    import tinygrad.codegen as cg, tinygrad.engine.realize as rz
    from tinygrad.helpers import Context
    orig_to_program = cg.to_program
    def to_program_fallback(ast, renderer):
      try: return orig_to_program(ast, renderer)
      except KeyError:
        # tinygrad's image rewrite (codegen/late/coalesce.py _drop_valid_stmts) can fail on a kernel (KeyError on a
        # FLOORMOD, the -seg models' mask-prototype kernels): render that kernel on plain buffers. image2d_t args are
        # created over the same buffers with no pitch padding, so kernels may mix the two views of one buffer.
        # tinygrad's heuristics for this kernel shape are poor without images (27 ms): BEAM it
        with Context(IMAGE=0):
          b = ast.replace(arg=replace(ast.arg, beam=args.fallback_beam)) if ast.op is Ops.SINK and not ast.arg.beam else ast
          prg = orig_to_program(b, renderer)
        fallbacks.append(f"{prg.src[0].arg.function_name}(no image, beam)")
        return prg
      except CompileError:
        # BEAM skips the candidates the vendor compiler rejects (the screening compile raises for them); NOOPT is the
        # last resort (slow: it launches one work item per output)
        if ast.op is Ops.SINK and not ast.arg.beam:
          try:
            prg = orig_to_program(ast.replace(arg=replace(ast.arg, beam=args.fallback_beam)), renderer)
            fallbacks.append(f"{prg.src[0].arg.function_name}(beam)")
            return prg
          except CompileError: pass
        with Context(NOOPT=1): prg = orig_to_program(ast, renderer)
        fallbacks.append(f"{prg.src[0].arg.function_name}(noopt)")
        return prg
    cg.to_program = rz.to_program = to_program_fallback

  # OnnxRunner reads shape-like inputs (Gather indices, Reshape shapes, ...) back as Python values; for a scalar
  # constant that means realizing a CONST with a kernel on tinygrad's CPU device, which needs a C compiler the
  # Android CPython doesn't have. Read constants straight off the graph instead.
  import tinygrad.nn.onnx as onnx_mod
  orig_const = onnx_mod._to_python_const
  def to_python_const(t:Tensor):
    b = t.uop.base
    if b.op is Ops.CAST and b.src[0].op is Ops.CONST: b = b.src[0]  # a scalar initializer: CAST(CONST)
    if b.op is Ops.CONST and t.dtype != dtypes.uint8 and all(isinstance(d, int) for d in t.shape):
      v = (int if dtypes.is_int(t.dtype) else bool if t.dtype == dtypes.bool else float)(b.arg)
      def nest(shape): return v if not shape else [nest(shape[1:]) for _ in range(shape[0])]
      return nest(tuple(t.shape))
    return orig_const(t)
  onnx_mod._to_python_const = to_python_const
  runner = OnnxRunner(Path(args.onnx))
  runner.onnx_ops = {**runner.onnx_ops, "GridSample": GridSample}
  (iname, itensor), = runner.graph_inputs.items()
  ishape = tuple(itensor.shape)
  if args.u8_nhwc:
    n, c, h, w = ishape
    in_shape, in_dtype = (n, h, w, c), dtypes.uint8
  else: in_shape, in_dtype = ishape, itensor.dtype

  def model(x:Tensor) -> dict[str, Tensor]:
    if args.u8_nhwc: x = x.cast(dtypes.float).permute(0, 3, 1, 2) / 255.0
    return runner({iname: x})

  out_specs = {k: (v.shape, v.dtype) for k, v in model(Tensor.zeros(*in_shape, dtype=in_dtype)).items()}
  onames = list(out_specs)

  @TinyJit
  def run(x:Tensor, *obufs:Tensor):
    outs = model(x)
    Tensor.realize(*(ob.assign(outs[k].cast(ob.dtype)) for ob, k in zip(obufs, onames)))

  obufs = [Tensor.empty(*s, dtype=dt).realize() for s, dt in out_specs.values()]
  def rand_in():
    return (Tensor.randint(*in_shape, low=0, high=256) if in_dtype == dtypes.uint8 else Tensor.randn(*in_shape)).cast(in_dtype).realize()
  for _ in range(3): run(rand_in(), *obufs)
  cap = run.captured

  # the JIT's inputs are (x, *obufs), in call order
  buf_ids: dict[int, int] = {}   # id(Buffer) -> plan id
  bufs: list = []                # plan id -> (Buffer|None, kind)
  def param_id(slot:int) -> int:
    key = -1 - slot
    if key not in buf_ids:
      buf_ids[key] = len(bufs)
      if slot == 0: bufs.append((None, ("in", 0), math.prod(in_shape) * in_dtype.itemsize))
      else:
        s, dt = out_specs[onames[slot-1]]
        bufs.append((None, ("out", slot-1), math.prod(s) * dt.itemsize))
    return buf_ids[key]
  def buffer_id(b) -> int:
    if id(b) not in buf_ids:
      buf_ids[id(b)] = len(bufs)
      bufs.append((b, None, b.nbytes))
    return buf_ids[id(b)]

  srcs: dict[str, str] = {}
  by_src: dict[str, str] = {}
  kerns, written, read_first = [], set(), set()
  kio: list[tuple[set[int], set[int]]] = []
  calls = cap._linear.src
  for call in calls:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM:
      raise SystemExit(f"unsupported JIT item {call.op} {call.src[0].op if call.src else ''} (only kernel calls are exported)")
    ast = call.src[0]
    elf = ast.to_elf()
    # tinygrad names kernels by shape, so two different kernels can share a name: rename per distinct source
    src = elf.lib.decode()
    if src not in by_src:
      kname = elf.name if elf.name not in srcs else f"{elf.name}_v{len(srcs)}"
      srcs[kname] = re.sub(rf"\b{re.escape(elf.name)}\(", f"{kname}(", src, count=1)
      by_src[src] = kname
    kname = by_src[src]
    cargs = get_call_arg_uops(call)
    ids = []
    for a in cargs:
      if a.op is Ops.PARAM: ids.append(param_id(a.arg.slot))
      elif a.op is Ops.BUFFER: ids.append(buffer_id(a.buffer))
      else: raise SystemExit(f"unsupported call arg {a.op}")
    kbufs = [ids[i] for i in ast.arg.globals]
    g, l = ast.arg.launch_dims({})
    l = tuple(l) if l is not None else (1, 1, 1)
    vals = ast.arg.vals({})
    kargs = []
    for name, slot, dt, shape in elf.signature:
      if slot < len(kbufs):
        bid = kbufs[slot]
        kargs.append(f"i{bid},{shape[0]},{shape[1]},{dt.itemsize}" if is_image_shape(shape) else f"b{bid}")
      else: kargs.append(f"v{vals[slot-len(kbufs)]}")
    # const detection: a buffer read before any kernel writes it holds data the model needs (weights)
    outs_ids = {ids[i] for i in ast.arg.outs}
    for i in ast.arg.ins:
      if ids[i] not in written: read_first.add(ids[i])
    written |= outs_ids
    kerns.append((kname, tuple(g) + (1,) * (3 - len(g)), l + (1,) * (3 - len(l)), kargs))
    kio.append(({ids[i] for i in ast.arg.ins}, outs_ids))

  # kernels that only turn weights into other weights (e.g. IMAGE=1's fp16 image repacking) run once at load, not per
  # frame: a kernel is "init" if everything it reads is constant and nothing else ever writes its outputs
  writers: dict[int, list[int]] = {}
  for ci, (_, outs_) in enumerate(kio):
    for o in outs_: writers.setdefault(o, []).append(ci)
  params = {p for p, (b, k, n) in enumerate(bufs) if k is not None}
  stable, init = read_first - params, set()
  for ci, (ins_, outs_) in enumerate(kio):
    if ins_ <= stable and all(writers[o] == [ci] and o not in params for o in outs_):
      init.add(ci); stable |= outs_
  persist = stable - read_first

  # scratch buffers share one arena: first-fit by lifetime (first..last kernel call touching them), 4 KB aligned
  # so every slice is a valid clCreateSubBuffer origin and image2d base on any device
  life: dict[int, list[int]] = {}
  for ci, (_, _, _, kargs) in enumerate(kerns):
    for a in kargs:
      if a[0] in "bi":
        bid = int(a[1:].split(",")[0])
        life.setdefault(bid, [ci, ci])[1] = ci
  ALIGN, placed, arena = 4096, [], 0
  offsets: dict[int, int] = {}
  for pid in sorted((p for p, (b, k, n) in enumerate(bufs) if k is None and p not in stable and p in life),
                    key=lambda p: -bufs[p][2]):
    lo, hi, size = life[pid][0], life[pid][1], (bufs[pid][2] + ALIGN - 1) // ALIGN * ALIGN
    busy = sorted((o, o + sz) for (o, sz, a, b2) in placed if not (b2 < lo or a > hi))
    off = 0
    for s0, s1 in busy:
      if off + size <= s0: break
      off = max(off, s1)
    placed.append((off, size, lo, hi)); offsets[pid] = off; arena = max(arena, off + size)

  consts = bytearray()
  lines = [f"arena {arena}"]
  for pid, (b, kind, nbytes) in enumerate(bufs):
    if kind is not None: lines.append(f"buf {pid} {nbytes} {kind[0]} {kind[1]}")
    elif pid in read_first:
      data = bytes(b.as_memoryview())
      off = len(consts); consts += data + bytes((-len(data)) % ALIGN)
      lines.append(f"buf {pid} {nbytes} const {off}")
    elif pid in persist: lines.append(f"buf {pid} {nbytes} persist")
    elif pid in offsets: lines.append(f"buf {pid} {nbytes} arena {offsets[pid]}")
    else: lines.append(f"buf {pid} {nbytes} unused")
  for ci, (name, g, l, kargs) in enumerate(kerns):
    lines.append(f"{'init' if ci in init else 'kern'} {name} {' '.join(map(str, g))} {' '.join(map(str, l))} {len(kargs)} {' '.join(kargs)}")
  (out / "plan.txt").write_text("\n".join(lines) + "\n")
  (out / "consts.bin").write_bytes(bytes(consts))
  (out / "kernels.cl").write_text("\n".join(srcs.values()))
  dtname = lambda dt: dt.name.replace("unsigned ", "u").replace(" ", "_")  # one token per field ("unsigned char" -> uchar)
  meta = [f"input {iname} {'x'.join(map(str, in_shape))} {dtname(in_dtype)}"] + \
         [f"output {k} {'x'.join(map(str, s))} {dtname(dt)}" for k, (s, dt) in out_specs.items()]
  (out / "meta.txt").write_text("\n".join(meta) + "\n")
  if fallbacks: print(f"{len(fallbacks)} kernels compiled unoptimized after a compiler error: {' '.join(fallbacks)}")
  scratch = sum(n for pid, (b, k, n) in enumerate(bufs) if k is None and pid not in read_first)
  print(f"{len(kerns)-len(init)} kernel calls per run (+{len(init)} once at load), {len(srcs)} kernels, {len(bufs)} buffers, consts {len(consts)/1e6:.2f} MB, "
        f"scratch {scratch/1e6:.2f} MB -> arena {arena/1e6:.2f} MB")

  if args.check:
    raw = Path(args.check).read_bytes()
    assert len(raw) == math.prod(in_shape) * in_dtype.itemsize, (len(raw), in_shape)
    xin = Tensor(raw) if in_dtype == dtypes.uint8 else Tensor(raw).bitcast(in_dtype)
    run(xin.reshape(in_shape).contiguous().realize(), *obufs)
    for k, ob in zip(onames, obufs): (out / f"ref_{k}.bin").write_bytes(bytes(ob.data()))
    print("saved reference outputs (tinygrad on this device)")

if __name__ == "__main__": main()
