import struct, sys
sys.path.insert(0, "/home/takecheeze/layerprof")
from decode_packed import load_words, to_bytes


def parse_task(d, p):
    t = d[p]; size = struct.unpack_from("<H", d, p + 2)[0]
    hdr = {"type": t, "size": size}
    if t == 0:    # COMMAND: target u8 @8, words @16
        return hdr | {"target": d[p+8], "words": [struct.unpack_from("<Q", d, p+16+8*i)[0] for i in range((size-16)//8)]}
    if t == 1:    # INSTRUCTION: target @8, offset u16 @10, count u32 @12, payload bytes @16
        return hdr | {"target": d[p+8], "offset": struct.unpack_from("<H", d, p+10)[0],
                      "count": struct.unpack_from("<I", d, p+12)[0], "payload": bytes(d[p+16:p+size])}
    if t == 2:    # TOKEN: mode @8, targets[14] @10
        return hdr | {"mode": d[p+8], "targets": bytes(d[p+10:p+24])}
    if t == 3:    # POLLING: reserved[2] @8, targets[14] @10
        return hdr | {"targets": bytes(d[p+10:p+24])}
    if t == 4:    # START_DMA: ch @8, count u32 @12, transfers (3 u64) @16
        n = struct.unpack_from("<I", d, p+12)[0]
        return hdr | {"ch": d[p+8], "count": n, "transfers": [struct.unpack_from("<QQQ", d, p+16+24*k) for k in range(n)]}
    if t == 5:    # POLL_DMA: ch @8
        return hdr | {"ch": d[p+8]}
    if t == 6:    # C2C_SYNC: mode @8, targets[5] @11
        return hdr | {"mode": d[p+8], "targets": bytes(d[p+11:p+16])}
    if t == 7:    # TIMESTAMP: message @8
        return hdr | {"message": d[p+8]}
    if t == 8:    # FUNCTION: function_id u32 @8, count u32 @12, args u64 @16
        n = struct.unpack_from("<I", d, p+12)[0]
        return hdr | {"fid": struct.unpack_from("<I", d, p+8)[0], "args": [struct.unpack_from("<Q", d, p+16+8*k)[0] for k in range(n)]}
    raise ValueError(f"unknown task type {t} at {p}")


def parse_stream(d):
    chunks, off = [], 0
    while off + 8 <= len(d):
        count, size = struct.unpack_from("<HxxI", d, off)
        cs = off + 8; p = cs; tasks = []
        for _ in range(count):
            tasks.append(parse_task(d, p)); p += tasks[-1]["size"]
        pad = bytes(d[p:cs + size])
        chunks.append({"count": count, "size": size, "tasks": tasks, "pad": pad})
        off = cs + size
    return chunks


def enc_task(tk):
    t, size = tk["type"], tk["size"]
    b = bytearray(size)
    b[0] = t; struct.pack_into("<H", b, 2, size)
    if t == 0:
        b[8] = tk["target"]
        for i, w in enumerate(tk["words"]): struct.pack_into("<Q", b, 16+8*i, w)
    elif t == 1:
        b[8] = tk["target"]; struct.pack_into("<H", b, 10, tk["offset"]); struct.pack_into("<I", b, 12, tk["count"])
        b[16:size] = tk["payload"]
    elif t == 2:
        b[8] = tk["mode"]; b[10:24] = tk["targets"]
    elif t == 3:
        b[10:24] = tk["targets"]
    elif t == 4:
        b[8] = tk["ch"]; struct.pack_into("<I", b, 12, tk["count"])
        for k, (a, c, s) in enumerate(tk["transfers"]): struct.pack_into("<QQQ", b, 16+24*k, a, c, s)
    elif t == 5:
        b[8] = tk["ch"]
    elif t == 6:
        b[8] = tk["mode"]; b[11:16] = tk["targets"]
    elif t == 7:
        b[8] = tk["message"]
    elif t == 8:
        struct.pack_into("<I", b, 8, tk["fid"]); struct.pack_into("<I", b, 12, len(tk["args"]))
        for k, a in enumerate(tk["args"]): struct.pack_into("<Q", b, 16+8*k, a)
    return bytes(b)


def enc_stream(chunks):
    out = bytearray()
    for ch in chunks:
        body = b"".join(enc_task(tk) for tk in ch["tasks"]) + ch["pad"]
        out += struct.pack("<HxxI", ch["count"], ch["size"]) + body
    return bytes(out)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        d = to_bytes(load_words(name + "/ai_core_kernel_0.c"))
        chunks = parse_stream(d)
        re = enc_stream(chunks)
        ok = re == d
        first_diff = next((i for i in range(min(len(re), len(d))) if re[i] != d[i]), None)
        print(f"{name}: bytes {len(d)} re-encoded {len(re)} identical={ok} first_diff={first_diff} chunks={len(chunks)}")
