"""MCC decoder blocks on HMX + HVX: host-side data (packed weights, inputs, float64 references).

  python ref.py export --out <dir> [--q 128] [--blocks 8]
  python ref.py weights --out <dir>          just blk0..7.bin + head.bin (what the demo app loads)

Writes <dir>/blk<i>.bin (every block's weights prepacked into HMX tiles, see tiles / BLOCK_LAYOUT),
<dir>/head.bin (positional embedding, final LayerNorm, prediction layer: HEAD_LAYOUT), <dir>/x0.bin
(the query chunk after the positional embedding, fp16 row-major [Q, 512]), the float64 references
ref_attn<i>.bin / ref_out<i>.bin (fp32, [Q, 512]) of every block run on the fp16-rounded input of block
0 (each block's reference input is the previous reference output), and for the whole decoder:
xyz.bin (fp32 [Q, 3]), kv.bin (fp32 k then v, [8, 16, 197, 32] each: what set_kv takes) and
ref_occ.bin / ref_rgb.bin (fp32 [Q] / [Q, 3], float64 model.QueryDecoder).

Queries: the first Q of the demo app's quest2m level-2 (surface) query set, K/V: its encoder cache
(../vision_models/mcc/dec_opt.py prep) -- real data, not random.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vision_models" / "mcc"))
import model as M  # noqa: E402

WORK = Path.home() / ".cache/onnxsim-mcc/work"
CKPT = Path.home() / ".cache/onnxsim-mcc/co3dv2_all_categories.pth"
D, HEADS, HD, SEEN, SEEN_PAD, HID = 512, 16, 32, 197, 224, 2048
LOG2E = 1.4426950408889634

# IDX(i, j) = 64*(i/2) + 2*j + i%2: halfword of element (i, j) in a 32x32 HMX tile
_I, _J = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
IDX = 64 * (_I // 2) + 2 * _J + _I % 2


def tiles(a, order="rc"):
    """(R, C) -> fp16 tiles, element (i, j) of tile at halfword IDX(i, j). order "rc": [R/32][C/32]
    tiles (activations / outputs: row blocks, then the K or column tiles of a row block); "cr":
    [C/32][R/32] (weights W[K, N]: per output-column block, its K tiles)."""
    r, c = a.shape
    t = a.astype(np.float16).reshape(r // 32, 32, c // 32, 32).transpose(0, 2, 1, 3)  # [rb][cb][i][j]
    if order == "cr":
        t = t.transpose(1, 0, 2, 3)
    out = np.zeros(t.shape[:2] + (1024,), np.float16)
    out[:, :, IDX] = t
    return out


def untiles(t, r, c):
    """[R/32][C/32][1024] fp16 tiles -> (R, C)"""
    return t.reshape(r // 32, c // 32, 1024)[:, :, IDX].transpose(0, 2, 1, 3).reshape(r, c)


def table(bias):
    """column tables: [N/32][64] uint32, word c = fp16 bias of column c in the high half"""
    b = np.asarray(bias, np.float16).view(np.uint16).astype(np.uint32).reshape(-1, 32)
    t = np.zeros((b.shape[0], 64), np.uint32)
    t[:, :32] = b << 16
    return t


# one block's weights, in file order (the C side, mcc_block.h, reads the same order)
BLOCK_LAYOUT = [
    "wqkv",  # [48 cb][16 kb] tiles: qkv W (512 x 1536), column block h = q head h, 16+h = k, 32+h = v
    "tqkv",  # [48][64] u32 tables (bias)
    "wproj",  # [16][16]
    "tproj",
    "wfc1",  # [64][16]
    "tfc1",
    "wfc2",  # [16][64]
    "tfc2",
    "kt",  # [16 heads][7 cb] tiles: K_h^T * scale * log2(e) (32 x 224, tokens >= 197 zero): scores in log2 units
    "vt",  # [16 heads][7 kb] tiles: V_h (224 x 32, tokens >= 197 zero)
    "ln1",  # [2][512] fp16: gamma, beta
    "ln2",
]


def pack_block(blk, k, v):
    """blk: an MCC decoder block (torch), k/v: that block's seen K/V (16, 197, 32)"""
    a = blk.attn
    f = {}
    f["wqkv"] = tiles(a.qkv.weight.detach().numpy().T, "cr")
    f["tqkv"] = table(a.qkv.bias.detach().numpy())
    f["wproj"] = tiles(a.proj.weight.detach().numpy().T, "cr")
    f["tproj"] = table(a.proj.bias.detach().numpy())
    f["wfc1"] = tiles(blk.mlp.fc1.weight.detach().numpy().T, "cr")
    f["tfc1"] = table(blk.mlp.fc1.bias.detach().numpy())
    f["wfc2"] = tiles(blk.mlp.fc2.weight.detach().numpy().T, "cr")
    f["tfc2"] = table(blk.mlp.fc2.bias.detach().numpy())
    kt, vt = [], []
    for h in range(HEADS):
        kp = np.zeros((HD, SEEN_PAD), np.float32)
        kp[:, :SEEN] = k[h].T * (a.scale * LOG2E)
        kt.append(tiles(kp, "cr")[:, 0])  # one K tile per column block
        vp = np.zeros((SEEN_PAD, HD), np.float32)
        vp[:SEEN] = v[h]
        vt.append(tiles(vp, "cr")[0])  # one column block, 7 K tiles
    f["kt"], f["vt"] = np.stack(kt), np.stack(vt)
    for n, ln in (("ln1", blk.norm1), ("ln2", blk.norm2)):
        f[n] = np.stack([ln.weight.detach().numpy(), ln.bias.detach().numpy()]).astype(np.float16)
    return b"".join(np.ascontiguousarray(f[n]).tobytes() for n in BLOCK_LAYOUT)


# the decoder's ends, in file order (mcc_decoder.h reads the same order)
HEAD_LAYOUT = [
    "wpos",  # [16 cb][1 kb] tiles: the 3 -> 512 positional embedding, K padded 3 -> 32
    "tpos",  # [16][64] u32 tables
    "lnf",  # [2][512] fp16: the final LayerNorm (decoder_norm) gamma, beta
    "wpred",  # [25 cb][16 kb] tiles: decoder_pred with its columns reordered (PRED_ORDER), 769 -> 800
    "tpred",  # [25][64]
]
# pred output columns: color channel c's 256 logits at 256 c .. 256 c + 255 (8 whole tiles each), the
# occupancy logit at 768 (upstream order: occupancy at 0, then the 3 x 256 color logits)
PRED_ORDER = list(range(1, 769)) + [0]


def pack_head(m):
    f = {}
    wp = np.zeros((32, D), np.float32)
    wp[:3] = m.decoder_xyz_pos_embed.pos_embed.weight.detach().numpy().T
    f["wpos"] = tiles(wp, "cr")
    f["tpos"] = table(m.decoder_xyz_pos_embed.pos_embed.bias.detach().numpy())
    f["lnf"] = np.stack([m.decoder_norm.weight.detach().numpy(), m.decoder_norm.bias.detach().numpy()]).astype(np.float16)
    w = np.zeros((D, 800), np.float32)
    b = np.zeros(800, np.float32)
    w[:, :769] = m.decoder_pred.weight.detach().numpy().T[:, PRED_ORDER]
    b[:769] = m.decoder_pred.bias.detach().numpy()[PRED_ORDER]
    f["wpred"] = tiles(w, "cr")
    f["tpred"] = table(b)
    return b"".join(np.ascontiguousarray(f[n]).tobytes() for n in HEAD_LAYOUT)


def block_ref(blk, x, k, v):
    """float64 block forward (model.QueryDecoder's per-block math) -> (after attention, after MLP)"""
    a = blk.attn
    q, kk, vv = (t[0] for t in M.split_qkv(a, blk.norm1(x), HEADS))
    s_seen = (q @ k.transpose(-2, -1)) * a.scale
    s_self = (q * kk).sum(-1, keepdim=True) * a.scale
    p = torch.softmax(torch.cat([s_seen, s_self], -1), -1)
    o = p[..., :SEEN] @ v + p[..., SEEN:] * vv
    x = x + a.proj(o.permute(1, 0, 2).reshape(1, -1, D))
    xa = x
    return xa, x + blk.mlp(blk.norm2(x))


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    m = M.load_mcc(str(CKPT))
    kv = np.load(WORK / "kv_quest2m.npz")
    xyz = torch.from_numpy(np.load(WORK / "queries_quest2m.npz")["arr_2"][: a.q][None].astype(np.float32))
    x = m.decoder_xyz_pos_embed(M.shrink(xyz)).half().double()
    x.float().half().numpy().reshape(a.q, D).tofile(out / "x0.bin")
    md = copy.deepcopy(m).double()  # float64 reference; m stays float32 for packing
    for i in range(a.blocks):
        k, v = kv["k"][i], kv["v"][i]
        (out / f"blk{i}.bin").write_bytes(pack_block(m.decoder_blocks[i], k, v))
        xa, x = block_ref(md.decoder_blocks[i], x, torch.from_numpy(k).double(), torch.from_numpy(v).double())
        xa.float().numpy().reshape(a.q, D).tofile(out / f"ref_attn{i}.bin")
        x.float().numpy().reshape(a.q, D).tofile(out / f"ref_out{i}.bin")
    (out / "head.bin").write_bytes(pack_head(m))
    xyz.numpy().reshape(a.q, 3).astype(np.float32).tofile(out / "xyz.bin")
    np.concatenate([kv["k"].astype(np.float32).ravel(), kv["v"].astype(np.float32).ravel()]).tofile(out / "kv.bin")
    occ, rgb = M.QueryDecoder(md).eval()(xyz.double(), torch.from_numpy(kv["k"]).double(), torch.from_numpy(kv["v"]).double())
    occ.float().numpy().ravel().tofile(out / "ref_occ.bin")
    rgb.float().numpy().reshape(a.q, 3).tofile(out / "ref_rgb.bin")
    print(f"{a.blocks} blocks, Q={a.q} -> {out} ({(out / 'blk0.bin').stat().st_size} B per block, head {(out / 'head.bin').stat().st_size} B)")


def cmd_weights(a):
    """blocks + head only; each block's K / V slots hold zeros (the app fills them per image: set_kv)"""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    m = M.load_mcc(str(CKPT))
    z = np.zeros((HEADS, SEEN, HD), np.float32)
    for i in range(8):
        (out / f"blk{i}.bin").write_bytes(pack_block(m.decoder_blocks[i], z, z))
    (out / "head.bin").write_bytes(pack_head(m))
    print(f"-> {out}: blk0..7.bin, head.bin")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--q", type=int, default=128)
    e.add_argument("--blocks", type=int, default=8)
    a = ap.parse_args()
    {"export": cmd_export, "weights": cmd_weights}[a.cmd](a)


if __name__ == "__main__":
    main()
