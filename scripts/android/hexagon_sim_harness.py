"""Run a TVM-compiled Hexagon kernel on the Hexagon instruction-set simulator (no phone).

TVM builds the kernel with an `llvm -mtriple=hexagon ...` target into an unlinked object whose
entry point uses TVM's packed calling convention. This module generates a small C harness that
fills `DLTensor`s, provides the three runtime hooks TVM's generated code imports
(`TVMBackendAllocWorkspace`, `TVMBackendFreeWorkspace`, `TVMAPISetLastError`), calls the kernel,
compares the output with a NumPy reference, and links/runs everything with the Hexagon
toolchain's `hexagon-clang` and `hexagon-sim` (standalone OS, HVX 128B). Set `HEXAGON_TOOLS` to
the toolchain's `Tools` directory (the open-access toolchain or `<SDK>/tools/HEXAGON_Tools/*/Tools`).

Kernels must not use `parallel` loops (no thread runtime in the harness).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HEXAGON_ARCH = "v73"
# Same target the phone path uses, but the plain `llvm` kind, which emits an unlinked object.
TARGET = (
    "llvm -mtriple=hexagon -mcpu=hexagon{arch} -mattr=+hvx{arch},+hvx-length128b,"
    "+hvx-qfloat,-hvx-ieee-fp"
)

_HARNESS = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
typedef struct { int32_t device_type, device_id; } DLDevice;
typedef struct { uint8_t code, bits; uint16_t lanes; } DLDataType;
typedef struct {
  void* data; DLDevice device; int32_t ndim; DLDataType dtype;
  int64_t* shape; int64_t* strides; uint64_t byte_offset;
} DLTensor;
typedef union { int64_t v_int64; double v_float64; void* v_handle; const char* v_str; } TVMValue;
extern void* __TVMBackendAllocWorkspace;
extern void* __TVMBackendFreeWorkspace;
extern void* __TVMAPISetLastError;
extern int kernel(TVMValue*, int*, int, TVMValue*, int*, void*);
static unsigned char arena[1 << 20] __attribute__((aligned(2048)));
static unsigned long top;
static void* alloc_ws(int dt, int di, uint64_t n, int c, int b) {
  top = (top + 127) & ~127ul; void* p = arena + top; top += n; return p;
}
static int free_ws(int dt, int di, void* p) { return 0; }
static void set_err(const char* m) { printf("TVM error: %s\n", m); }
static float h2f(uint16_t h) {
  int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
  if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY;
  else v = ldexpf((float)(m + 1024), e - 25);
  return s ? -v : v;
}
#include "data.h"
int main(void) {
  __TVMBackendAllocWorkspace = (void*)alloc_ws;
  __TVMBackendFreeWorkspace = (void*)free_ws;
  __TVMAPISetLastError = (void*)set_err;
  DLTensor t[NUM_ARGS]; TVMValue v[NUM_ARGS]; int codes[NUM_ARGS];
  for (int i = 0; i < NUM_ARGS; i++) {
    t[i] = (DLTensor){arg_data[i], {1, 0}, arg_ndim[i], arg_dtype[i], arg_shape[i], 0, 0};
    v[i].v_handle = &t[i]; codes[i] = 7;
  }
  TVMValue ret; int ret_code;
#ifndef REPEAT
#define REPEAT 1
#endif
  int rc = 0;
  for (int r = 0; r < REPEAT && !rc; r++) rc = kernel(v, codes, NUM_ARGS, &ret, &ret_code, 0);
  if (rc) { printf("kernel returned %d\n", rc); return 2; }
  double worst = 0;
  for (int i = 0; i < OUT_COUNT; i++) {
    double d = fabs((double)OUT_GET(i) - expected[i]);
    if (d > worst || d != d) worst = d;
  }
  printf("max_abs_err=%g tol=%g\n", worst, (double)TOL);
  return worst <= TOL ? 0 : 1;
}
"""


def tools_dir() -> Path | None:
    """The Hexagon toolchain `Tools` directory, or None when it is not configured."""
    root = os.environ.get("HEXAGON_TOOLS") or os.environ.get("HEXAGON_TOOLCHAIN")
    if not root:
        return None
    path = Path(root)
    if (path / "bin" / "hexagon-clang").exists() and (path / "bin" / "hexagon-sim").exists():
        return path
    return None


def kernel_pcycles(output: str) -> int:
    """Cycles of one kernel call, from a `run_kernel(..., measure_cycles=True)` output."""
    import re

    return int(re.search(r"kernel_pcycles=(-?\d+)", output).group(1))


def _total_pcycles(output: str) -> int:
    import re

    return int(re.search(r"Pcycles=(\d+)", output).group(1))


def target(arch: str = HEXAGON_ARCH) -> str:
    return TARGET.format(arch=arch)


def _c_array(name: str, values: np.ndarray, ctype: str, aligned: bool = False) -> str:
    flat = values.ravel()
    body = ",".join(str(int(x)) if ctype != "float" else repr(float(x)) for x in flat)
    attr = " __attribute__((aligned(2048)))" if aligned else ""
    return f"static {ctype} {name}[{max(flat.size, 1)}]{attr} = {{{body}}};\n"


def _dtype_info(dtype: str):
    dtype = np.dtype(dtype).name
    if dtype == "float32":
        return "float", (2, 32, 1), lambda a: a.astype("float32")
    if dtype == "float16":
        return "uint16_t", (2, 16, 1), lambda a: a.astype("float16").view("uint16")
    raise ValueError(f"unsupported dtype {dtype}")


def _ensure_ncurses5(env: dict[str, str], sim: Path, workdir: Path) -> None:
    """hexagon-sim links libncurses.so.5; modern distros only ship .so.6 (same ABI for it)."""
    probe = subprocess.run(["ldd", str(sim)], capture_output=True, text=True, check=False)
    if "not found" not in probe.stdout:
        return
    shim = workdir / "shim"
    shim.mkdir(exist_ok=True)
    for name in ("ncurses", "tinfo"):
        for lib_dir in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            source = Path(lib_dir) / f"lib{name}.so.6"
            if source.exists():
                (shim / f"lib{name}.so.5").symlink_to(source)
                break
    env["LD_LIBRARY_PATH"] = f"{shim}:{env.get('LD_LIBRARY_PATH', '')}"


def run_kernel(
    module,
    inputs: list[np.ndarray],
    out_shape,
    out_dtype: str,
    expected: np.ndarray,
    tol: float,
    arch: str = HEXAGON_ARCH,
    keep_dir: str | None = None,
    timeout: int = 900,
    measure_cycles: bool = False,
) -> str:
    """Link `module` (entry point `kernel`) with a harness, run it on hexagon-sim, and return
    the simulator output. Raises AssertionError if the result exceeds `tol` (max abs error). With
    `measure_cycles`, the output gains a `kernel_pcycles=N` line (cycles of one kernel call)."""
    tools = tools_dir()
    if tools is None:
        raise RuntimeError("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    workdir = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="hexagon_sim_"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        module.save(str(workdir / "kernel.o"))
        lines = [f"#define NUM_ARGS {len(inputs) + 1}\n"]
        shapes, dtypes, ndims, pointers = [], [], [], []
        for index, array in enumerate([*inputs, np.zeros(out_shape, dtype=out_dtype)]):
            ctype, dl_dtype, convert = _dtype_info(array.dtype)
            lines.append(_c_array(f"a{index}", convert(array), ctype, aligned=True))
            lines.append(
                f"static int64_t s{index}[] = {{{','.join(str(int(d)) for d in array.shape)}}};\n"
            )
            pointers.append(f"a{index}")
            shapes.append(f"s{index}")
            ndims.append(str(array.ndim))
            dtypes.append("{%d,%d,%d}" % dl_dtype)
        out_index = len(inputs)
        lines.append(f"static void* arg_data[] = {{{','.join(pointers)}}};\n")
        lines.append(f"static int arg_ndim[] = {{{','.join(ndims)}}};\n")
        lines.append(f"static DLDataType arg_dtype[] = {{{','.join(dtypes)}}};\n")
        lines.append(f"static int64_t* arg_shape[] = {{{','.join(shapes)}}};\n")
        lines.append(_c_array("expected", np.asarray(expected, dtype="float32"), "float"))
        out_ctype = _dtype_info(out_dtype)[0]
        getter = f"h2f(a{out_index}[i])" if out_ctype == "uint16_t" else f"a{out_index}[i]"
        lines.append(f"#define OUT_COUNT {int(np.prod(out_shape))}\n")
        lines.append(f"#define OUT_GET(i) {getter}\n#define TOL {tol!r}\n")
        (workdir / "data.h").write_text("".join(lines))
        (workdir / "harness.c").write_text(_HARNESS)

        clang = tools / "bin" / "hexagon-clang"
        sim_bin = tools / "bin" / "hexagon-sim"
        env = dict(os.environ)
        _ensure_ncurses5(env, sim_bin, workdir)

        def build_and_run(repeat: int) -> str:
            elf = f"test{repeat}.elf"
            compile_cmd = [
                str(clang), f"-m{arch}", "-mhvx", "-mhvx-length=128B", "-O1", f"-DREPEAT={repeat}",
                "-I.", "harness.c", "kernel.o", "-o", elf, "-lm",
            ]  # fmt: skip
            build = subprocess.run(
                compile_cmd, cwd=workdir, capture_output=True, text=True, check=False
            )
            if build.returncode:
                raise RuntimeError(f"hexagon-clang failed:\n{build.stderr}")
            result = subprocess.run(
                [str(sim_bin), f"-m{arch}", "--simulated_returnval", elf],
                cwd=workdir, capture_output=True, text=True, env=env, timeout=timeout, check=False,
            )  # fmt: skip
            output = result.stdout + result.stderr
            if result.returncode != 0:
                raise AssertionError(f"hexagon-sim exit {result.returncode}:\n{output}")
            return output

        output = build_and_run(1)
        if measure_cycles:
            # The simulator is deterministic: one extra kernel call costs exactly the kernel.
            output += f"\nkernel_pcycles={_total_pcycles(build_and_run(2)) - _total_pcycles(output)}\n"
        return output
    finally:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)
