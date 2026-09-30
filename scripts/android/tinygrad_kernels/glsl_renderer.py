"""A Vulkan GLSL compute renderer for tinygrad (0.14.x), built on the same CStyleLanguage base as WGSLRenderer.

tinygrad ships a WGSL renderer (WebGPU) but no SPIR-V/Vulkan one. This renders the same lowered UOp kernels as
GLSL 4.50 compute shaders (`#version 450`, one std430 storage buffer per kernel parameter at binding = slot),
which `glslangValidator -V` turns into SPIR-V that Vulkan consumes directly. Scope: float/int32/uint32/bool
kernels (matmul, conv, elementwise, reductions); no vectors, no sub-32-bit types, no images.

Differences from the WGSL renderer that matter:
- there is no INFINITY uniform: buffers start at binding 0 (WGSL reserves binding 0 for it);
- GLSL `?:` evaluates only the chosen operand, so a gated load is a real predicated load (WGSL `select` reads both);
- `shared` arrays must be declared at module scope, so they are hoisted out of the kernel body.
"""

from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import strip_parens
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat


def _bitcast(ctx, x):
    src, dst = x.src[0].dtype, x.dtype
    v = ctx[x.src[0]]
    if src == dst:
        return v
    if src == dtypes.float and dst == dtypes.int32:
        return f"floatBitsToInt({v})"
    if src == dtypes.float and dst == dtypes.uint32:
        return f"floatBitsToUint({v})"
    if src == dtypes.int32 and dst == dtypes.float:
        return f"intBitsToFloat({v})"
    if src == dtypes.uint32 and dst == dtypes.float:
        return f"uintBitsToFloat({v})"
    if {src, dst} == {dtypes.int32, dtypes.uint32}:
        return f"{ctx.type_map[dst]}({v})"
    raise NotImplementedError(f"GLSL bitcast {src} -> {dst}")


glsl_matcher = PatternMatcher(
    [
        # GLSL has no ordering on bool: compare as ints
        (
            UPat(
                Ops.CMPLT,
                src=(UPat(name="a", dtype=dtypes.bool), UPat.var("b")),
                name="c",
            ),
            lambda a, b, c: a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)),
        ),
    ]
)


class GLSLRenderer(CStyleLanguage):
    global_max = (65535, 65535, 65535)
    local_max = (256, 256, 64)
    code_for_workitem = {
        "g": lambda x: f"int(gl_WorkGroupID.{'xyz'[int(x)]})",
        "l": lambda x: f"int(gl_LocalInvocationID.{'xyz'[int(x)]})",
    }
    supports_float4 = False
    smem_prefix = "shared "
    smem_prefix_for_cast = False
    barrier = "memoryBarrierShared(); barrier();"
    extra_matcher = glsl_matcher
    # GLSL has no bitwise operators on bool: use the logical ones
    code_for_op = {
        **CStyleLanguage.code_for_op,
        Ops.AND: lambda a, b, dtype: f"({a}&&{b})"
        if dtype == dtypes.bool
        else f"({a}&{b})",
        Ops.OR: lambda a, b, dtype: f"({a}||{b})"
        if dtype == dtypes.bool
        else f"({a}|{b})",
        Ops.XOR: lambda a, b, dtype: f"({a}!={b})"
        if dtype == dtypes.bool
        else f"({a}^{b})",
    }
    type_map = {
        dtypes.float: "float",
        dtypes.int32: "int",
        dtypes.uint32: "uint",
        dtypes.bool: "bool",
    }

    string_rewrite = (
        PatternMatcher(
            [
                (
                    UPat.cvar("c").cast(dtypes.bool),
                    lambda c: "true" if c.val else "false",
                ),
                (UPat(Ops.BITCAST, name="x"), lambda ctx, x: _bitcast(ctx, x)),
                (
                    UPat.load(UPat.var("b"), UPat.var("v"), UPat.var("gate")),
                    lambda ctx, b, v, gate: f"({ctx[gate]}?{ctx[b]}:{ctx[v]})",
                ),
                (UPat.load(UPat.var("b")), lambda ctx, b: ctx[b]),
                (
                    UPat.store(UPat.var("b"), UPat.var("v")),
                    lambda ctx, b, v: f"{ctx[b]} = {ctx[v]};",
                ),
                (
                    UPat(Ops.INDEX, src=(UPat.var("b"), UPat.var("idx"))),
                    lambda ctx,
                    b,
                    idx: f"{ctx[b]}[{strip_parens(ctx[idx]) if idx.arg is Ops.ADD else ctx[idx]}]",
                ),
            ]
        )
        + base_rewrite
    )

    def render_cast(self, u: UOp, val: str) -> str:
        return f"{self.type_map[u.dtype]}({val})"

    def render_kernel(
        self,
        function_name: str,
        kernel: list[str],
        bufs: list[tuple[str, tuple[UOp, bool]]],
        uops: list[UOp],
        prefix=None,
    ) -> str:
        local_size = [
            u.src[0].ssimplify()
            for u in sorted(
                [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"],
                key=lambda u: u.arg,
            )
        ] or [1]
        local_size = (list(local_size) + [1, 1, 1])[:3]
        shared = [
            line.strip() for line in kernel if line.lstrip().startswith("shared ")
        ]
        kernel[:] = [line for line in kernel if not line.lstrip().startswith("shared ")]
        prg = "#version 450\n"
        prg += "#define INFINITY uintBitsToFloat(0x7f800000u)\n#define NAN uintBitsToFloat(0x7fc00000u)\n"
        prg += f"// kernel: {function_name}\n"
        prg += f"layout(local_size_x={local_size[0]}, local_size_y={local_size[1]}, local_size_z={local_size[2]}) in;\n"
        for i, (name, (u, _)) in enumerate(bufs):
            assert u.addrspace == AddrSpace.GLOBAL, (
                f"unsupported non-global kernel argument {name}"
            )
            prg += f"layout(std430, set=0, binding={i}) buffer B{i} {{ {self.type_map[u.dtype]} {name}[]; }};\n"
        prg += "\n".join(shared) + "\n" if shared else ""
        return prg + "void main() {\n" + "\n".join(kernel) + "\n}\n"

    def supported_dtypes(self):
        return {dtypes.bool, dtypes.int32, dtypes.uint32, dtypes.float}
