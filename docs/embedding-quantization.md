# Embedding-output quantization (`quantize_embedding_binary` / `quantize_embedding_int8`)

## What this is

`onnxsim.quantize_embedding_binary` and `onnxsim.quantize_embedding_int8`
compress a retrieval encoder model's own output embedding, at inference
time, by rewiring the graph's declared output to a freshly quantized
tensor. Every other quantizer in onnxsim rewrites how a model *computes*
(weights, KV cache, attention); this pair rewrites what the model's own
graph *emits*, for downstream storage in a vector index rather than for
cheaper on-device math.

This follows Mixedbread's ["Asymmetric Quantization: Near-Lossless
Late-Interaction Retrieval with 97% Storage
Reduction"](https://www.mixedbread.com/blog/asymmetric-quant) and its
predecessor, ["Binary and Scalar Embedding Quantization for Significantly
Faster & Cheaper Retrieval"](https://huggingface.co/blog/embedding-quantization)
(Mixedbread + Hugging Face). Retrieval systems that embed both a query and
a very large number of documents naturally want **asymmetric** precision:
keep the single, per-query vector at higher precision for accurate
scoring, and compress the many, stored-forever document vectors as hard as
the accuracy budget allows. Mixedbread's own numbers: an int8 query scored
against binary documents keeps 89.65 NDCG@10 versus 90.26 for full
float32, at roughly 32x less per-document storage.

```
Before:
  embedding = <some node>(...)          -- graph output, float32 [..., D]

quantize_embedding_int8:
  scale: initializer, float32 scalar    -- calibrated once, per-tensor
  embedding = QuantizeLinear(embedding, scale, zero_point=0)   -- int8, same shape

quantize_embedding_binary:
  bits = Cast(Greater(embedding, 0), INT64)              -- 1 if > 0, else 0
  grouped = Reshape(bits, [..., D/8, 8])
  packed = Cast(ReduceSum(grouped * [128,64,32,16,8,4,2,1], axis=-1), UINT8)
  embedding = packed                    -- graph output, uint8 [..., D/8]
```

`quantize_embedding_binary`'s packing is exactly
`numpy.packbits(embedding > 0, axis=-1, bitorder="big")` -- independently
checkable against that reference, and needs no calibration data at all.
`quantize_embedding_int8` calibrates a single scale for the whole tensor
(Mixedbread/Hugging Face's own "scalar quantization"), the same
calibration flow every other onnxsim quantizer uses.

## Usage

```python
import onnx
import onnxsim

encoder = onnx.load("encoder.onnx")

# Query side: keep more precision.
query_model = onnxsim.quantize_embedding_int8(encoder, num_samples=32)
onnx.save(query_model, "encoder.query.onnx")

# Document side: compress hard -- this is the one stored billions of times.
doc_model = onnxsim.quantize_embedding_binary(encoder)
onnx.save(doc_model, "encoder.doc.onnx")
```

Both default to the graph's sole float32 output when `output_name` is
omitted (declining rather than guessing if there's more than one); pass
`output_name` explicitly for a model with multiple outputs.

## Scope

Handled:

- A float32 graph output, resolved either by explicit `output_name` or as
  the graph's one and only float32 output.
- Opsets >= 13 (`QuantizeLinear`'s scale/zero-point convention and
  `ReduceSum`'s axes-as-input both assume opset 13).
- `quantize_embedding_binary` additionally needs the target output's last
  dimension known statically and a multiple of 8.

Left untouched (safe no-op, model returned unchanged):

- An ambiguous target (no `output_name` given, more than one float32
  output, or an `output_name` that doesn't resolve to a float32 output).
- An opset older than 13.
- For `quantize_embedding_binary` only: a target whose last dimension is
  symbolic/unknown or not a multiple of 8.

## Relationship to onnxsim's other quantizers

Every `quantize_weight_only_*`/`quantize_kv_cache`/etc. function compresses
something the model reads or produces *internally*, to make the model's
own math cheaper or its cache smaller. This pair is the odd one out: the
quantized tensor is the model's *own final output*, produced once per
encode call and then stored externally (in a vector index) for the
lifetime of that document -- so the right precision tradeoff is set by
retrieval quality vs. index storage cost, not by inference latency, and
by which side of a query/document pair the exported model serves.
