# Exporting nerfstudio's nerfacto to ONNX

Measured on nerfstudio 1.1.5 (`pip install nerfstudio`, CPU, `implementation="torch"`),
with the repo's own `onnxsim.simplify` doing the round-trip check.

## Summary

nerfacto's feed-forward core — everything except the hash-grid interpolation —
exports to ONNX cleanly and survives simplification with identical numerics.
The hash grid does **not** export, and the reason is structural rather than a
missing pass in onnxsim: it uses a bitwise op ONNX has no equivalent for.

| nerfacto component | Export to ONNX | Notes |
|---|---|---|
| `position_encoding` | ✅ 12 nodes | `Mul`/`Add`/`Concat` — the frequency embedding |
| `mlp_base_mlp` (geo-feature MLP) | ✅ 3 nodes | `Gemm`, `Relu` |
| `direction_encoding` (view MLP) | ✅ 603 nodes | heavy but plain `Gemm`/elementwise |
| `mlp_head` (colour head) | ✅ 4 nodes | `Gemm`, `Relu`, `Sigmoid` |
| `mlp_base_grid` (hash encoding) | ❌ **fails** | `aten::bitwise_xor` |
| `mlp_base` (grid + MLP, fused) | ❌ fails | contains the hash grid |

Numeric round-trip through `onnxsim.simplify` on the colour head:

```
onnxsim.simplify -> ok=True, 4 nodes (was 4)
max_abs_diff = 5.96e-08
```

## Why the hash grid can't export

`HashEncoding.hash_fn` (nerfstudio/field_components/encodings.py) is the
Instant-NGP space hash:

```python
in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861])
x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
x = torch.bitwise_xor(x, in_tensor[..., 2])
x %= self.hash_table_size
```

`torch.onnx.export` fails with:

```
UnsupportedOperatorError: Exporting the operator 'aten::bitwise_xor'
to ONNX opset version 17 is not supported
```

ONNX has **no bitwise-xor operator at any opset** (its bitwise set is only
`And`/`Or`/`Xor` on *bools*, plus integer shifts and `BitShift`). So this is not
something onnxsim can simplify away: the operator simply has no target.

Getting past it needs the hash computed *outside* the graph — precompute the
`Gather` indices for a fixed set of sample positions, or replace the encoding
with one that composes from supported ops. Both change the model's structure,
so neither is appropriate to do implicitly.

## What is exportable, concretely

The whole colour path, verified above:

```
positions, directions
  -> direction_encoding(directions)      # 603 nodes, exportable
  -> mlp_head(cat([d, geo_feat, appearance]))
  -> rgb
```

and the geo-feature MLP once the grid indices are precomputed. In other words
roughly two thirds of nerfacto's parameter count exports as-is; the remaining
third is the hash table plus its lookup.

For completeness: `splatfacto` is further off still — `gsplat` is fused CUDA, so
its rasteriser has no ONNX representation at all. That is a property of the
model, not of this exporter work.

## Environment notes

Two things needed working around to run this, neither related to ONNX:

1. **nerfstudio 1.1.5 does not import on Python 3.11+**
   (nerfstudio-project/nerfstudio#3106). `configs/base_config.py` declares
   `local_writer: LocalWriterConfig = LocalWriterConfig(enable=True)` as a
   dataclass default; Python 3.11 widened the mutable-default check to "is its
   type unhashable", so the module raises `ValueError` at import and every
   `Field`/`Model` import fails with it. Rewriting that one declaration to
   `dataclasses.field(default_factory=...)` is the fix Python's own error names.

2. **`onnx` has no Python 3.14 wheel**, so `pip install` builds it from source
   and the bundled protobuf fails on arm64. Use 3.11/3.12.

`onnxsim` itself must be the installed wheel, not the source tree, when the
serving interpreter differs from the one the C++ extension was built for — the
`.abi3.so` resolves against the *running* interpreter.