import os
import sys

STAGE = int(os.environ.get("STAGE", "1"))  # 1 = load only, 2 = +imc, 3 = +matmul, 4 = +store
M, K, N = 64, 256, 4096
KB, NB = K // 64, N // 64
TILE_NB = 8
NT = NB // TILE_NB

import axelera.compiler.alto.script as A  # noqa: E402
from axelera.compiler.alto import axol  # noqa: E402
from axelera.compiler.alto.compiler.config.config import AltoConfig  # noqa: E402
from axelera.compiler.alto.compiler.device import BufferScope  # noqa: E402
from axelera.compiler.alto.compiler.driver import compile as alto_compile  # noqa: E402
from axelera.config import HardwareGeneration  # noqa: E402
from tvm import tir  # noqa: E402

LOAD_ID, LOAD_IMC_ID, COMPUTE_ID, STORE_ID = "pmm_load", "pmm_load_imc", "pmm_compute", "pmm_store"


def ftile(base):
    return A.Ramp(A.max(0, A.min(base, NB - TILE_NB)), 1, TILE_NB)


@A.ir_module
class Bisect:
    @A.prim_func
    def main(a: A.Buffer((M, KB), "int8x64", scope=A.DDR),
             b: A.Buffer((K, NB), "int8x64", scope=A.DDR)):
        a_l1 = A.alloc_buffer((M, KB), "int8x64", scope=BufferScope.L1)
        b_l1 = A.alloc_buffer((2, K, NB), "int8x64", scope=BufferScope.L1)
        out = A.alloc_buffer((M, NB), "float32x64", scope=A.DDR)
        axol.atomic.copy(a, output=a_l1, id="pmm_load_a")
        A.sync(waits=["pmm_load_a"])
        for s in A.tiles(NT):
            axol.atomic.copy(b[0:K, 0:TILE_NB], output=b_l1[0, A.Ramp(0, 1, K), A.Ramp(0, 1, TILE_NB)], id=LOAD_ID)
        A.output(out)


axm = alto_compile(Bisect, AltoConfig(output_dir=os.path.abspath("bisect-build"), clean_output_dir=True,
                                      generation=HardwareGeneration.OMEGA, num_cores=1))
print("COMPILED stage", STAGE, axm, flush=True)
