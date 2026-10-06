# Transformers on the Snapdragon HTP (Xiaomi 12S, Hexagon V69)

Real-hardware counterpart of `scripts/allwinner/transformer_accuracy/`: the same DistilBERT SST-2 study and the same
LayerNorm rewrite (`scripts/allwinner/npu_rewrite.py`), run on the phone's NPU through ONNX Runtime's QNN EP instead of
being emulated.

| File | Purpose |
|---|---|
| `prep_distilbert.py` | export DistilBERT (batch 1, 64 tokens), build the graph variants (`orig`, `naive`, `adaptive`; `--quant` adds `w8a8`, `w8a16` QDQ models) and the 872-sentence validation inputs |
| `run_phone.py` | push model + data, run `qnn_eval` under the phone lock, score accuracy / agreement with host fp32 / logit error / latency |
| `fp16_probe.py` | minimal graphs with squares above the fp16 maximum: does the HTP's fp16 mode overflow? |
| `../htp_exploration/qnn_shell/qnn_eval.cpp` | multi-sample runner (one session, N samples, latency percentiles, outputs saved) |

```
python -I prep_distilbert.py --work DIR [--quant]        # not from the repository root
python -I run_phone.py --work DIR --model distilbert_adaptive --mode htp --fp16 --json results.jsonl
python -I fp16_probe.py --work DIR
```

## Results (DistilBERT, SST-2 validation, 872 sentences, batch 1 x 64 tokens)

| Variant | Where it runs | ms/sentence | Accuracy | Agreement with host fp32 |
|---|---|---|---|---|
| `orig` (native LayerNorm) | CPU, 4 threads | 45.4 | | |
| `orig` | HTP fp16, strict (no CPU fallback) | 2.27 | 91.06% | 100% |
| `naive` decomposed norm | HTP fp16, strict | 3.27 | 91.06% | 100% |
| `adaptive` row-max-scaled norm | HTP fp16, strict | 3.30 | 91.06% | 100% |
| `w8a8` QDQ | HTP, CPU fallback for 12 nodes | 10.2 | 50.9% | 53% |
| `w8a16` QDQ, `Reciprocal -> Mul` kept (as `adaptive`) | HTP, CPU fallback for 12 nodes | 13.7 | 49.1% | 47% |
| `w8a16` QDQ, `Div` form (`prep_distilbert.py --quant`) | HTP, CPU fallback for 12 nodes | 17.1 | 90.37% | 99.3% |

Host ORT CPU on the first 200 sentences: `w8a8` 49.5%, `w8a16` 90.5%.

## Findings

- fp16 on the HTP is about 20x faster than the CPU with no accuracy loss, and the native LayerNorm graph is the fastest.
- The naive decomposed norm happens to survive on DistilBERT's activations. It is **not** safe in general: `fp16_probe.py` with one
  channel at 600 gives a wrong result on the HTP (max 45.7 against an exact 27.7) and no NaN/inf to warn about it, while the
  row-max-scaled form stays within 4e-4 of the range. A squared tensor of 3.6e5 comes back as 1.3e5, so the fp16 mode does not
  produce infinities here, it silently saturates or loses range.
- `w8a8` is a recipe problem (chance accuracy on host CPU as well, outlier activations).
- `w8a16` was accurate on the host but at chance on the HTP, with or without `--fp16`. Bisecting intermediate tensors put the first
  wrong value at the decomposed norm's `Mul(d, Reciprocal(std))` (peak 4.8 against 14.7); leaving the Reciprocal float did not
  help. Rewriting `Mul(a, Reciprocal(b))` as `Div(a, b)` before quantizing (done by `prep_distilbert.py`) gives 90.37% on the HTP.
  The mechanism inside the EP is not identified; treat 16-bit `Reciprocal -> Mul` as unsafe on the HTP.
- Even the fixed w8a16 model is 7.5x slower than fp16 (17.1 against 2.27 ms) because of the CPU-fallback nodes.
- The QDQ models are slower than fp16 because the Softmax and the masked Add are kept float (the mask's -3.4e38 fill would set the
  quantization range) and strict mode refuses them.

## Caveats

- The QDQ calibration sentences come from the validation set (overlap with the scored sentences), so the quantized accuracies are slightly optimistic.
- Only DistilBERT so far; models with larger outliers (SmolLM2) are untested on the HTP.
- Latency is per sentence with batch 1, burst performance mode, EP-context cache (compile time excluded).
