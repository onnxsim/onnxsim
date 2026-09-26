"""Per-kernel differential check of a hexagon-sim replay (emit.py sim --dump) against tinygrad's CPU backend.

  PYTHONPATH=<tinygrad hvx-hmx> python verify.py <bundle dir>

For every call in order: the call's input buffers as the replay had them just before it (the latest dump,
or the first-seen contents), the kernel's AST (capture.py's calls.pkl) compiled for the CPU and run on
them, and every argument buffer compared with the replay's dump after the call. Prints each call's max
difference; the first call beyond tolerance is where the DSP code goes wrong.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from tinygrad import Device
from tinygrad.codegen import to_program
from tinygrad.device import Buffer
from tinygrad.engine.realize import get_runtime


def main():
    b = Path(sys.argv[1])
    calls = [list(map(int, ln.split())) for ln in (b / "calls.txt").read_text().split("\n") if ln.strip()]
    bufs = {int(t[0]): tuple(map(int, t[1:])) for t in (ln.split() for ln in (b / "bufs.txt").read_text().split("\n") if ln.strip())}
    sizes = {i: s for i, (r, o, s) in bufs.items()}
    asts = pickle.loads((b / "calls.pkl").read_bytes())
    # the replay's memory: region images; a buffer's state = its slice of the region
    mem = {int(p.stem[1:]): bytearray(p.read_bytes()) for p in b.glob("r*.bin")}
    for n, (c, (sink, globs)) in enumerate(zip(calls, asts)):
        ids = c[1:]  # the DSP kernel's parameters, in its globals order
        prg = to_program(sink, Device["CPU"].renderer)
        rt = get_runtime("CPU", prg)
        # map: call-buffer index (the value in globals) -> our buffer id
        by_global = dict(zip(globs, ids))
        cpu_ids = [by_global[g] for g in prg.arg.globals]
        cbufs = {i: Buffer("CPU", sizes[i], __import__("tinygrad").dtypes.uint8).ensure_allocated() for i in set(cpu_ids)}
        for i, buf in cbufs.items():
            r, o, sz = bufs[i]
            buf.allocator._copyin(buf._buf, memoryview(bytearray(mem[r][o : o + sz])))
        gs, ls = prg.arg.launch_dims({})
        rt(*[cbufs[i]._buf for i in cpu_ids], global_size=gs, local_size=ls, vals=())
        worst = []
        for i in ids:
            got = (b / "dump" / f"c{n}_b{i}.bin").read_bytes()
            want = bytes(cbufs[i].as_memoryview())
            if got != want:
                g16, w16 = np.frombuffer(got, np.float16).astype(np.float64), np.frombuffer(want, np.float16).astype(np.float64)
                g32, w32 = np.frombuffer(got, np.float32).astype(np.float64), np.frombuffer(want, np.float32).astype(np.float64)
                d16 = np.nanmax(np.abs(g16 - w16)) / max(np.nanmax(np.abs(w16)), 1e-6)
                d32 = np.nanmax(np.abs(g32 - w32)) / max(np.nanmax(np.abs(w32)), 1e-6)
                worst.append(f"b{i}: rel diff fp16-view {d16:.3g} fp32-view {d32:.3g}")
            r, o, sz = bufs[i]
            mem[r][o : o + sz] = got  # continue from what the replay actually had
        name = (b / f"k{c[0]}.c").read_text().split("\n")[0]
        print(f"call {n:2d} k{c[0]:<2d} {name:48s} {'OK (bit-exact)' if not worst else '; '.join(worst)}")


if __name__ == "__main__":
    main()
