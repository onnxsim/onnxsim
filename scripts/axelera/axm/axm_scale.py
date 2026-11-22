#!/usr/bin/env python3
"""Edit and verify the requantization scale in a compiled Metis .axm package.

The unrolled int8 GEMM (scripts/axelera/pipe_unroll.py) stores its requantization scale as
float32 values in the ELF's `.l1_memory` section (one value per lane per tile, 2048 bytes).
This tool finds that section by parsing the ELF section table (stdlib only), rewrites the
values, and can run the package on the card to check the output against a numpy reference.

Device state persists between runs: a result can reflect the previous model. `check` runs a
reset model first, so every check starts from a clean device state.

Usage:
    python axm_scale.py show   PKG.axm
    python axm_scale.py set    IN.axm OUT.axm --den 2048 [--from-den 4096]
    python axm_scale.py check  PKG.axm --den 2048 --tiles 8 --m 1024 --k 512 --n 512
"""

import argparse
import struct
import sys
import zipfile


def elf_section(elf: bytes, name: str):
    """Return (file_offset, size) of a named ELF section, parsed from the section table."""
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    def sh(i):
        return struct.unpack_from("<IIQQQQ", elf, shoff + i * shentsize)
    _, _, _, _, stroff, strsize = sh(shstrndx)
    names = elf[stroff:stroff + strsize]
    for i in range(shnum):
        name_off, _, _, _, off, size = sh(i)
        end = names.index(b"\0", name_off)
        if names[name_off:end].decode() == name:
            return off, size
    raise KeyError(f"section {name} not found")


def read_pkg(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def write_pkg(path, entries, order):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for n in order:
            z.writestr(n, entries[n])


def scale_region(elf: bytes):
    off, size = elf_section(elf, ".l1_memory")
    assert size % 4 == 0
    vals = struct.unpack_from(f"<{size // 4}f", elf, off)
    return off, size, vals


def cmd_show(args):
    entries = read_pkg(args.pkg)
    off, size, vals = scale_region(entries["ai_core_kernel_0.elf"])
    uniq = sorted(set(vals))
    print(f".l1_memory at file offset {off}, {size} bytes, {len(vals)} floats")
    for u in uniq:
        print(f"  value {u!r} (1/{1/u:.0f}) x {vals.count(u)}" if u else f"  value 0.0 x {vals.count(u)}")


def cmd_set(args):
    entries = read_pkg(args.inp)
    order = list(zipfile.ZipFile(args.inp).namelist())
    elf = bytearray(entries["ai_core_kernel_0.elf"])
    off, size, vals = scale_region(elf)
    old = 1.0 / args.from_den
    if any(v != old for v in vals):
        sys.exit(f"refusing: not all L1 scale values equal 1/{args.from_den}")
    new = struct.pack(f"<{len(vals)}f", *([1.0 / args.den] * len(vals)))
    elf[off:off + size] = new
    entries["ai_core_kernel_0.elf"] = bytes(elf)
    write_pkg(args.out, entries, order)
    print(f"wrote {args.out}: {len(vals)} scale values 1/{args.from_den} -> 1/{args.den}")


def cmd_check(args):
    import numpy as np
    from axelera.runtime import Context

    # reset in a separate process: device state persists across models in one process
    code = (
        "from axelera.runtime import Context\n"
        "import numpy as np\n"
        f"ctx = Context(); rst = ctx.load_model({args.reset!r})\n"
        "conn = ctx.device_connect(None, num_sub_devices=1)\n"
        "inst = conn.load_model_instance(rst, num_sub_devices=1, aipu_cores=1, host_profiling=True)\n"
        "ins = [np.zeros(t.shape, t.dtype) for t in rst.inputs()]\n"
        "outs = [np.zeros(t.shape, t.dtype) for t in rst.outputs()]\n"
        "inst.run(ins, outs)\n"
    )
    import subprocess
    subprocess.run([sys.executable, "-c", code], check=True)

    M, K, N, NT = args.m, args.k, args.n, args.tiles
    TILE, KB, NB = M // NT, K // 64, N // 64
    rng = np.random.default_rng(0)                      # same inputs as pipe_unroll.py
    xs = [rng.integers(-128, 128, (TILE, KB, 64), dtype=np.int8) for _ in range(NT)]
    w = rng.integers(-128, 128, (K, NB, 64), dtype=np.int8)

    ctx = Context()
    m = ctx.load_model(args.pkg)
    conn = ctx.device_connect(None, num_sub_devices=1)
    inst = conn.load_model_instance(m, num_sub_devices=1, aipu_cores=1, host_profiling=True)
    outs = [np.zeros((TILE, NB, 64), dtype=np.int8) for _ in range(NT)]
    inst.run(xs + [w], outs)
    dev = np.concatenate([o.reshape(TILE, -1) for o in outs], axis=0).astype(np.int64)

    acc = xs_flat(xs, M, K).astype(np.int64) @ w.reshape(K, N).astype(np.int64)
    ref = np.clip(np.floor(acc / float(args.den) + 0.5), -128, 127)
    exact = float(np.mean(dev == ref))
    print(f"{args.pkg} vs reference(1/{args.den}): exact {exact:.4f}, mean abs err {np.abs(dev - ref).mean():.3f}, "
          f"range {dev.min()}..{dev.max()}")
    if exact < args.min_exact:
        sys.exit(1)


def xs_flat(xs, M, K):
    import numpy as np
    return np.concatenate([x.reshape(x.shape[0], -1) for x in xs], axis=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("pkg"); s.set_defaults(fn=cmd_show)
    t = sub.add_parser("set"); t.add_argument("inp"); t.add_argument("out")
    t.add_argument("--den", type=int, required=True); t.add_argument("--from-den", type=int, default=4096)
    t.set_defaults(fn=cmd_set)
    c = sub.add_parser("check"); c.add_argument("pkg"); c.add_argument("--den", type=int, required=True)
    c.add_argument("--tiles", type=int, default=8); c.add_argument("--m", type=int, default=1024)
    c.add_argument("--k", type=int, default=512); c.add_argument("--n", type=int, default=512)
    c.add_argument("--min-exact", type=float, default=0.9)
    c.add_argument("--reset", required=True, help="a small .axm used to reset device state before the check")
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)
