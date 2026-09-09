// Build real sample tensors (an image, tokenized text) for the "Run
// inference" panel's "sample data" fill mode. Dispatches purely on an input's
// declared type/shape/name — no model-specific knowledge beyond the
// filename->tokenizer-family lookup in tokenize.mjs — so it works the same way
// for any model that happens to have an image- or text-shaped input, not just
// the curated Hugging Face set.
//
// Unrecognized/unsupported inputs return null so the caller
// (inference_browser.mjs's makeDummyInputs) falls back to input_fill.mjs's
// synthetic fill, logging why.

import { fetchSampleImageBytes, fetchSampleSentence } from "./hf_datasets.mjs";
import { tokenizeForModel } from "./tokenize.mjs";

// ImageNet per-channel normalization (mean/std over [0,1]-scaled RGB), the
// convention most timm/torchvision-exported models in the curated set use.
// This is a documented approximation, not a per-model guarantee: a handful of
// architectures (e.g. Inception-style) expect [0.5,0.5,0.5]/[0.5,0.5,0.5]
// instead — same "reasonable default, not an exact match" spirit as
// input_fill.mjs's synthetic modes.
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// True for a single 4-D float tensor input shaped like an image batch:
// [N, C, H, W] with C in {1, 3} (grayscale or RGB) and concrete spatial dims.
export function isImageInput(meta, dims) {
  return (
    meta.type === "float32" &&
    Array.isArray(dims) &&
    dims.length === 4 &&
    (dims[1] === 1 || dims[1] === 3) &&
    dims[2] > 0 &&
    dims[3] > 0
  );
}

const TEXT_INPUT_NAMES = /^(input_ids|attention_mask|token_type_ids|decoder_input_ids)$/i;

// True for an int64 rank-2 [N, seqLen] tensor named like a tokenizer output.
export function isTextInput(meta, dims) {
  return (
    meta.type === "int64" &&
    Array.isArray(dims) &&
    dims.length === 2 &&
    dims[1] > 0 &&
    TEXT_INPUT_NAMES.test(meta.name || "")
  );
}

// Convert one decoded image's RGBA pixels (a canvas ImageData.data, exactly
// W*H*4 bytes, already resized to the target W/H) into a normalized NCHW
// Float32Array of length C*H*W (one sample, no batch axis). Pure — no
// DOM/network — so it's unit-testable without a real image decode.
export function normalizePixels(rgba, c, h, w) {
  const out = new Float32Array(c * h * w);
  const mean = c === 3 ? IMAGENET_MEAN : [0.5];
  const std = c === 3 ? IMAGENET_STD : [0.5];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      for (let ch = 0; ch < c; ch++) {
        // Grayscale: average the decoded RGB into the single channel.
        const raw = c === 3 ? rgba[p + ch] : (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
        const v = (raw / 255 - mean[ch]) / std[ch];
        out[ch * h * w + y * w + x] = v;
      }
    }
  }
  return out;
}

// Decode `bytes` (JPEG/PNG) and resize to [w, h] via an offscreen canvas,
// returning the raw RGBA ImageData.data. DOM-coupled — not unit tested at the
// Node level, matching inference_browser.mjs's own untested DOM glue.
async function decodeAndResize(bytes, w, h) {
  const blob = new Blob([bytes]);
  const bitmap = await createImageBitmap(blob);
  const canvas =
    typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

// Per-call cache so a model with several image/text inputs reuses ONE fetched
// image and ONE fetched+tokenized sentence across all of them, instead of a
// fresh (and inconsistent) sample per input. Callers create one context per
// makeDummyInputs() call and pass it to every buildSampleInput() call in that
// loop.
export function createSampleContext() {
  return {};
}

function getSampleImage(ctx, log) {
  if (!ctx.imagePromise) {
    ctx.imagePromise = fetchSampleImageBytes().then((r) => {
      log(`sample image: ${r.label} (uoft-cs/cifar10)`);
      return r;
    });
  }
  return ctx.imagePromise;
}

function getSampleText(ctx, log) {
  if (!ctx.textPromise) {
    ctx.textPromise = fetchSampleSentence().then((r) => {
      log(`sample text: "${r.text}" (${r.label}, stanfordnlp/sst2)`);
      return r;
    });
  }
  return ctx.textPromise;
}

function getTokenized(ctx, modelName, seqLen, log) {
  const key = `tok:${seqLen}`;
  if (!ctx[key]) {
    ctx[key] = getSampleText(ctx, log).then((r) => tokenizeForModel(modelName, r.text, seqLen));
  }
  return ctx[key];
}

// Build one sample-filled typed array for `meta`/`dims` (dims already resolved
// to concrete integers, e.g. via inference_browser.mjs's concreteShape), using
// `Ctor` (the ort.Tensor-compatible typed array constructor for meta.type).
// Returns null when this input isn't image- or recognized-text-shaped, or when
// fetching/decoding/tokenizing the sample fails — the caller falls back to
// synthetic fill either way, logging why via `log`.
export async function buildSampleInput(meta, dims, modelName, Ctor, ctx, log) {
  try {
    if (isImageInput(meta, dims)) {
      const [n, c, h, w] = dims;
      const { bytes } = await getSampleImage(ctx, log);
      const rgba = await decodeAndResize(bytes, w, h);
      const sample = normalizePixels(rgba, c, h, w);
      const out = new Ctor(n * c * h * w);
      for (let i = 0; i < n; i++) out.set(sample, i * sample.length);
      return out;
    }
    if (isTextInput(meta, dims)) {
      const [n, seqLen] = dims;
      const tok = await getTokenized(ctx, modelName, seqLen, log);
      const field = tok[meta.name];
      if (!field) return null; // e.g. decoder_input_ids: not something we produce
      const out = new Ctor(n * seqLen);
      const values = field.map((v) => (Ctor === BigInt64Array || Ctor === BigUint64Array ? BigInt(v) : v));
      for (let i = 0; i < n; i++) out.set(values, i * seqLen);
      return out;
    }
  } catch (e) {
    log(
      `sample data unavailable for input '${meta.name}' (${e && e.message ? e.message : e}); ` +
        "falling back to random fill",
    );
    return null;
  }
  return null;
}
