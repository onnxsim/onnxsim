# Exporting nerfstudio's nerfacto to ONNX

Measured on nerfstudio 1.1.5 (`pip install nerfstudio`, CPU, `implementation="torch"`),
exporting with `torch.onnx.export(dynamo=True)` and round-tripping through the
repo's own `onnxsim.simplify`.

These numbers are re-verified on every run of
`tests/test_nerfstudio_onnx_export.py`, which builds the same field and asserts
the same properties (ORT agreement, the hash graph surviving simplification, and
that the legacy exporter still cannot emit `bitwise_xor`). That test is skipped
unless nerfstudio is installed; the `nerfstudio-onnx-export` workflow installs it
and runs it on PRs that touch the exporter paths, weekly, and on demand.

## Summary

nerfacto exports to ONNX in full, hash grid included. Every component survives
simplification with the numerics unchanged.

| nerfacto component | nodes | key ops |
|---|---|---|
| `position_encoding` | 12 | `Mul`/`Add`/`Concat` -- the frequency embedding |
| `mlp_base_grid` (hash encoding) | 136 | `BitwiseXor` x16, `Gather` x24, `Mul`/`Cast`/`Floor`/`Slice` |
| `mlp_base_mlp` (geo-feature MLP) | 3 | `Gemm`, `Relu` |
| `mlp_base` (grid + MLP, fused) | 139 | the above plus the geo MLP |
| `direction_encoding` (view MLP) | 603 | heavy but plain `Gemm`/elementwise |
| `mlp_head` (colour head) | 4 | `Gemm`, `Relu`, `Sigmoid` |

Verified round-trips through `onnxsim.simplify`:

```
mlp_base_grid:  max_abs_diff = 0.00e+00   (bit-exact against torch)
mlp_base:       ok=True, 139 -> 139 nodes, max_abs_diff = 5.96e-08
mlp_head:       ok=True,    4 ->   4 nodes, max_abs_diff = 5.96e-08
```

`onnxsim` leaves the hash graph alone (139 -> 139 nodes, the 16 `BitwiseXor`
preserved), which is correct: there is nothing redundant to remove in it.

## End to end: the whole feed-forward core runs under ORT

`position_encoding` + hash grid + geo MLP + view MLP + colour head, i.e. the
entirety of nerfacto's density and RGB computation, at the defaults
`geo_feat_dim=15`, `num_levels=8`, `log2_hashmap_size=17`:

```
exported:  221 nodes, 16 x BitwiseXor
ORT direct:            density max_diff = 0.00e+00   rgb max_diff = 5.96e-08
onnxsim.simplify:      ok=True, 221 -> 218 nodes, BitwiseXor preserved
ORT after simplify:    density max_diff = 0.00e+00   rgb max_diff = 5.96e-08
density finite, rgb within [0, 1]
```

With a dynamic batch axis, over 5 random batches at two different row counts:

```
varying rows: density 8.20e-08   rgb 5.96e-08
128 rows:     density 7.45e-08   rgb 5.96e-08
```

So the bar that matters -- onnxruntime executes a genuinely exported nerfstudio
field, and onnxsim's simplification preserves it -- is met for the whole
feed-forward core, hash grid included.

## The one real constraint: the dynamo exporter

`HashEncoding.hash_fn` (nerfstudio/field_components/encodings.py) is the
Instant-NGP space hash:

```python
in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861])
x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
x = torch.bitwise_xor(x, in_tensor[..., 2])
x %= self.hash_table_size
```

That maps to ONNX's
[`BitwiseXor`](https://onnx.ai/onnx/operators/onnx__BitwiseXor.html), which
exists since **opset 18** over exactly the integer types the hash uses
(`int8/16/32/64`, `uint8/16/32/64`). But the **legacy** TorchScript exporter has
no symbolic registered for `aten::bitwise_xor`, so it fails before the opset ever
comes into play:

| exporter | opset 17 | opset 18 |
|---|---|---|
| `dynamo=False` (legacy TorchScript) | FAIL `UnsupportedOperatorError: aten::bitwise_xor` | FAIL `UnsupportedOperatorError: aten::bitwise_xor` |
| `dynamo=True` (dynamo, needs `onnxscript`) | OK -- 136 nodes, 16 x `BitwiseXor` | OK -- 136 nodes, 16 x `BitwiseXor` |

So the hash grid needs `dynamo=True`; the opset floor of 18 matters only for a
backend that wants to run the result. Note that
`onnxsim.test_utils.export_simplify_and_check_by_python_api` deliberately sets
`dynamo=False` (so `ScriptModule` inputs keep working), so a caller exporting a
nerfacto field through that helper has to override it.

## Downstream: what the exporters can take

The exported ops (`Gather`, `Mul`, `Cast`, `Floor`, `Slice`, `Gemm`, `Relu`,
`Sigmoid`, `BitwiseXor`) are all in onnxsim's hand-written exporter tables. Two
caveats to check against the target backend rather than assume:

- **Core ML**: `BitwiseXor` has no MIL op, and onnxsim's Core ML exporter does
  not lower it, so that backend reports it unsupported -- correctly, rather than
  emitting something wrong. Whether Core ML's own compute plan could accept a
  lowering is untested here (see the `coremltools` note below).
- **TFLite**: likewise no `BitwiseXor` builtin; it would need Flex or a
  lowering.

For completeness: `splatfacto` is a different story -- `gsplat` is fused CUDA, so
its rasteriser has no ONNX representation at all. That is a property of the
model, not of this exporter work.

## Environment notes

Three obstacles, none of them about the model:

1. **nerfstudio 1.1.5 does not import on Python 3.11+**
   (nerfstudio-project/nerfstudio#3106). `configs/base_config.py` declares
   `local_writer: LocalWriterConfig = LocalWriterConfig(enable=True)` as a
   dataclass field default. Python 3.11 widened the mutable-default check from
   "is the value a list/set/dict" to "is its *type* unhashable"
   (https://discuss.python.org/t/better-communicate-dataclass-mutable-default-check-change-in-python-3-11/19028).
   `LocalWriterConfig` defines `__hash__`, so the module raises `ValueError` at
   import and every `Field`/`Model` import fails with it. Rewriting that one
   declaration to `dataclasses.field(default_factory=...)` -- the fix Python's
   own error message names -- is enough.

2. **`onnx` has no Python 3.14 wheel**, so `pip install` builds it from source
   and the bundled protobuf fails on arm64. Use 3.11 or 3.12.

3. **`onnxscript` is required** for `dynamo=True`, which is what makes the hash
   grid exportable at all.

Also: `onnxsim` itself must be the installed wheel rather than the source tree
when the serving interpreter differs from the one the C++ extension was built
for -- the `.abi3.so` resolves against the *running* interpreter.