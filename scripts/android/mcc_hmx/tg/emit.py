"""Replay a captured tinygrad kernel sequence (capture.py save) on hexagon-sim with real HMX.

  python emit.py sim <bundle dir> [--repeat-cycles] [--ref <float32 file> --q Q]
  python emit.py skel <bundle dir>      replay_gen.h for the phone skel (replay_impl.c; build.sh)

Each kernel k<i>.c is compiled on its own (hexagon-clang -mv69 -mhmx -mhvx, no HMX_REF: the real HMX
tile op), a driver main() loads every buffer's first-seen contents, points __hmx_vtcm at the simulator's
VTCM, sets SSR bit 26 (HMX enable; hmx_lock does it on the phone), runs the calls in order and writes
the output buffer to <bundle>/sim_out.bin (functional simulation). --repeat-cycles runs in the
cycle-accurate --timing mode, once and twice, and prints the Pcycles difference (the in-sim cycle
counter reads 0 in standalone mode); that mode is slow on scalar-heavy kernels.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

TOOLS = Path(os.environ.get("HEXAGON_TOOLS", os.path.expanduser("~/.cache/hexagon-oa-19/Tools")))
CFLAGS = ["-mv69", "-mhmx", "-mhvx", "-mhvx-length=128B", "-O2"]


def load(bundle: Path):
    calls = [list(map(int, ln.split())) for ln in (bundle / "calls.txt").read_text().split("\n") if ln.strip()]
    bufs = {int(t[0]): tuple(map(int, t[1:])) for t in (ln.split() for ln in (bundle / "bufs.txt").read_text().split("\n") if ln.strip())}
    regions = dict(tuple(map(int, ln.split())) for ln in (bundle / "regions.txt").read_text().split("\n") if ln.strip())
    out = int((bundle / "out.txt").read_text())
    nk = len(list(bundle.glob("k*.c")))
    return calls, bufs, regions, out, nk


def driver_c(calls, bufs, regions, out, nk, repeat, dump=False):
    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#pragma clang diagnostic ignored \"-Wdeprecated-non-prototype\"",
             "unsigned char* __hmx_vtcm;", "unsigned int __hmx_gen = 1;"]
    lines += [f"void tgk{i}();" for i in range(nk)]
    lines += [f"static unsigned char r{i}[{max(s, 1)}] __attribute__((aligned(128)));" for i, s in regions.items()]
    lines += [f"#define b{i} (r{r} + {o})" for i, (r, o, s) in bufs.items()]
    lines += ["static void load(const char* n, unsigned char* p, int sz) { FILE* f = fopen(n, \"rb\"); fread(p, 1, sz, f); fclose(f); }"]
    lines += ["int main(void) {", "  unsigned base; __asm__ volatile(\"%0 = cfgbase\" : \"=r\"(base));",
              "  __hmx_vtcm = (unsigned char*)(*(volatile unsigned*)((base << 16) + 0x38) << 16);",
              "  unsigned r; __asm__ volatile(\"%0 = ssr\" : \"=r\"(r)); r |= 1u << 26; __asm__ volatile(\"ssr = %0; isync\" :: \"r\"(r));"]
    lines += [f"  load(\"r{i}.bin\", r{i}, {s});" for i, s in regions.items()]
    lines += [f"  for (int rep = 0; rep < {repeat}; rep++) {{"]
    for n, c in enumerate(calls):
        lines.append(f"    tgk{c[0]}({', '.join(f'(void*)b{a}' for a in c[1:])});")
        if dump:  # every argument buffer after the call: dump/c<n>_b<id>.bin (verify.py)
            lines += [f"    {{ FILE* f = fopen(\"dump/c{n}_b{a}.bin\", \"wb\"); fwrite(b{a}, 1, {bufs[a][2]}, f); fclose(f); }}" for a in c[1:]]
    lines += ["  }", f"  {{ FILE* f = fopen(\"sim_out.bin\", \"wb\"); fwrite(b{out}, 1, {bufs[out][2]}, f); fclose(f); }}", "  return 0;", "}"]
    return "\n".join(lines) + "\n"


def skel_h(calls, bufs, regions, out, nk, vtcm_kb=256):
    """replay_gen.h: the regions, the output buffer and the call sequence (each call timed in pcycles)"""
    lines = [f"#define NR {len(regions)}", f"#define NCALL {len(calls)}",
             f"static const unsigned int RSZ[NR] = {{{', '.join(str(max(s, 1)) for s in regions.values())}}};",
             "static unsigned char* R[NR];", f"#define VTCM_KB {vtcm_kb}", f"#define OUT_R {bufs[out][0]}", f"#define OUT_O {bufs[out][1]}", f"#define OUT_N {bufs[out][2]}",
             "#pragma clang diagnostic ignored \"-Wdeprecated-non-prototype\""]
    lines += [f"void tgk{i}();" for i in range(nk)]
    lines += ["static void tg_calls(unsigned long long* pc) {", "  unsigned long long t;"]
    for n, c in enumerate(calls):
        args = ", ".join(f"(void*)(R[{bufs[a][0]}] + {bufs[a][1]})" for a in c[1:])
        lines.append(f"  t = qurt_get_core_pcycles(); tgk{c[0]}({args}); pc[{n}] += qurt_get_core_pcycles() - t;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_run(bundle: Path, repeat: int, timing: bool = False, dump: bool = False) -> int:
    calls, bufs, regions, out, nk = load(bundle)
    (bundle / "driver.c").write_text(driver_c(calls, bufs, regions, out, nk, repeat, dump))
    (bundle / "dump").mkdir(exist_ok=True)
    objs = []
    for i in range(nk):
        o = bundle / f"k{i}.o"
        if not o.exists() or o.stat().st_mtime < (bundle / f"k{i}.c").stat().st_mtime:
            subprocess.run([str(TOOLS / "bin/hexagon-clang"), *CFLAGS, "-c", f"k{i}.c", "-o", o.name], cwd=bundle, check=True)
        objs.append(o.name)
    subprocess.run([str(TOOLS / "bin/hexagon-clang"), *CFLAGS, "driver.c", *objs, "-o", "replay.elf", "-lm"], cwd=bundle, check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=os.environ.get("NCSHIM", os.path.expanduser("~/.cache/ncshim")))
    r = subprocess.run([str(TOOLS / "bin/hexagon-sim"), "-mv69", "--mhmx", "1", *(["--timing"] if timing else []), "replay.elf"], cwd=bundle, env=env,
                       capture_output=True, text=True, check=True)
    return int(re.search(r"Pcycles=(\d+)", r.stdout + r.stderr).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sim", "skel"])
    ap.add_argument("bundle")
    ap.add_argument("--repeat-cycles", action="store_true")
    ap.add_argument("--ref", help="float32 reference of the output (e.g. ref.py's ref_out<i>.bin); its first Q rows are compared")
    ap.add_argument("--q", type=int)
    ap.add_argument("--f32", action="store_true", help="the output is float32 (default fp16)")
    ap.add_argument("--dump", action="store_true", help="write every call's argument buffers (for verify.py)")
    a = ap.parse_args()
    b = Path(a.bundle)
    if a.cmd == "skel":
        calls, bufs, regions, out, nk = load(b)
        vk = b / "vtcm_kb.txt"
        (b / "replay_gen.h").write_text(skel_h(calls, bufs, regions, out, nk, int(vk.read_text()) if vk.exists() else 256))
        (b / "out_bytes.txt").write_text(f"{bufs[out][2]}\n")
        print(f"{b}/replay_gen.h: {len(calls)} calls, {len(regions)} regions")
        return
    c1 = build_run(b, 1, a.repeat_cycles, a.dump)
    print(f"replayed {len(load(b)[0])} kernel calls on hexagon-sim -> {b}/sim_out.bin ({c1} Pcycles incl. setup)")
    if a.repeat_cycles:
        (b / "sim_out1.bin").write_bytes((b / "sim_out.bin").read_bytes())
        c2 = build_run(b, 2, True)
        print(f"one pass: {c2 - c1} Pcycles")
        (b / "sim_out.bin").write_bytes((b / "sim_out1.bin").read_bytes())
    if a.ref:
        import numpy as np

        got = np.fromfile(b / "sim_out.bin", np.float32 if a.f32 else np.float16).astype(np.float64).reshape(a.q, -1)
        ref = np.fromfile(a.ref, np.float32).reshape(-1, got.shape[1])[: a.q].astype(np.float64)
        cs = float((got * ref).sum() / np.sqrt((got**2).sum() * (ref**2).sum()))
        print(f"vs {a.ref}: max abs err {np.abs(got - ref).max():.4g} (max |ref| {np.abs(ref).max():.4g}), cos {cs:.7f}")


if __name__ == "__main__":
    main()
