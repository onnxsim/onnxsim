"""A float checkpoint (``LLM_PF_SNAPSHOT``) from an AWQ int4 checkpoint: each
packed linear dequantized to float16, the rest copied. For a model whose float
weights are not cached (Qwen3-4B: only ``Qwen/Qwen3-4B-AWQ`` is).

The AWQ ``gemm`` layout (group size 128, zero points): ``qweight`` int32
``[in, out / 8]``, ``qzeros`` int32 ``[in / g, out / 8]``, ``scales`` float16
``[in / g, out]``. Each int32 holds eight 4-bit values; nibble ``i`` is output
column ``ORDER[i]`` of its group of eight (``ORDER`` = autoawq's interleave).
``W[i, o] = (q[i, o] - z[i // g, o]) * s[i // g, o]``; the float linear's weight is
``W.T`` ``[out, in]`` (HF's ``nn.Linear``).

``--layers N`` keeps the first N layers (a shorter checkpoint with the same dims:
the card test of layers 0-1 needs only those). The embedding is tied to the LM head.

usage: pf_awq_snapshot.py <awq snapshot dir> <out dir> [--layers N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import numpy as np

#: autoawq's interleave: nibble i of a packed int32 is output column ORDER[i] (of 8)
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
GROUP = 128
COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "generation_config.json")  # fmt: skip


def unpack(packed: np.ndarray, order=ORDER) -> np.ndarray:
    """int32 ``[r, c]`` packed as AWQ does -> int ``[r, c * 8]`` of 4-bit values."""
    p = packed.astype(np.int64) & 0xFFFFFFFF
    nib = np.stack([(p >> (4 * i)) & 0xF for i in range(8)], axis=-1)
    out = np.empty_like(nib)
    out[..., order] = nib  # out[..., order[i]] = nib[..., i]
    return out.reshape(packed.shape[0], -1)


def dequant(qweight, qzeros, scales, order=ORDER) -> np.ndarray:
    """The float32 weight ``[in, out]`` of one AWQ linear."""
    q = unpack(qweight, order).astype(np.float32)
    z = unpack(qzeros, order).astype(np.float32)
    s = scales.astype(np.float32)
    g = np.arange(q.shape[0]) // GROUP  # the group of each input row
    return (q - z[g]) * s[g]


def _open(path):
    import safetensors

    return safetensors.safe_open(path, framework="pt")


def _np(st, k: str) -> np.ndarray:
    """A tensor as numpy (bfloat16 widened: numpy has no bfloat16)."""
    t = st.get_tensor(k)
    if str(t.dtype) == "torch.bfloat16":
        t = t.float()
    return t.numpy()


def convert(src: str, dst: str, layers: int | None = None) -> dict:
    """Write the float checkpoint of ``src`` (an AWQ snapshot) into ``dst``."""
    from safetensors.numpy import save_file

    cfg = json.load(open(os.path.join(src, "config.json")))
    n = cfg["num_hidden_layers"] if layers is None else layers
    cfg.pop("quantization_config", None)
    cfg["num_hidden_layers"] = n
    cfg["torch_dtype"] = "float16"
    cfg["dtype"] = "float16"
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in COPY:
        if os.path.exists(os.path.join(src, f)):
            if os.path.exists(
                os.path.join(dst, f)
            ):  # a read-only copy of an earlier run
                os.remove(os.path.join(dst, f))
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    st = _open(os.path.join(src, "model.safetensors"))
    keep = re.compile(r"model\.(embed_tokens|norm)\.|model\.layers\.(\d+)\.")
    out, names = {}, []
    for k in st.keys():
        m = keep.match(k)
        if not m or (m[2] is not None and int(m[2]) >= n):
            continue
        names.append(k)
    for k in sorted(names):
        base = k.rsplit(".", 1)[0]
        if k.endswith(".qweight"):  # a packed linear: the float weight
            w = dequant(
                _np(st, k), _np(st, base + ".qzeros"), _np(st, base + ".scales")
            )  # fmt: skip
            # C order: the transpose is Fortran-ordered and the file takes the bytes as they lie
            out[base + ".weight"] = np.ascontiguousarray(w.T, dtype=np.float16)
        elif k.endswith((".qzeros", ".scales")):
            continue
        else:
            out[k] = _np(st, k).astype(np.float16)
    # the LM head is the embedding (tie_word_embeddings)
    save_file(out, os.path.join(dst, "model.safetensors"))
    return dict(tensors=len(out), layers=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    print(convert(a.src, a.dst, a.layers))


if __name__ == "__main__":
    main()
