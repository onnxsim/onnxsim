// Browser glue for the "Quantize a model" panel on the converter page.
//
// Runs one of onnxsim's quantization methods on either the original upload or
// the Convert section's own simplify/optimize output (a "Quantize input"
// radio picks which -- see resolveQuantizeInput below), entirely in the
// browser -- nothing is uploaded:
//   - Dynamic / Ternary / Weight-only / Weight-only INT4 need no calibration
//     data, so it's a single WASM call.
//   - Static (QDQ) / QOperator (QLinearMatMul) are calibration-based: the
//     candidate tensor names come from the WASM module
//     (onnxsim_list_quantizable_activations / _list_qoperator_quantizable_outputs),
//     and their (min, max) ranges are measured by actually running the model
//     over synthetic random inputs through onnxruntime-web
//     (quantize_calibration.mjs), mirroring onnxsim.calibration.calibrate().
//
// The result is kept independent of the Convert section's own output: it is
// published on window.__onnxsimQuantized (not window.__onnxsimConverted) and
// shown in its own "Quantized" Netron pane (window.netronShowQuantized, not
// netronShowAfter) -- so a user can inspect all three pipeline stages side by
// side (original / onnxsim-simplified / quantized) rather than the quantized
// result silently replacing the plain simplify/optimize one.
//
// Like debug_tools.mjs's "parse a text graph" panel, the quantization
// rewrites themselves need no model executor, so they run on this page's own
// WASM runtime directly (no worker round trip, unlike Simplify/Optimize) --
// only the Static/QOperator calibration step needs onnxruntime-web, loaded on
// demand the same way the "Run inference" panel does.

import { downloadBytes } from "./download.mjs";
import { resolveOriginalModelBytes } from "./inference_browser.mjs";
import { calibrateRanges } from "./quantize_calibration.mjs";
import { providersForEp, isWebnnEp } from "./webnn.mjs";
import { computeQuantizationQuality, renderQuantizationQuality } from "./quantize_metrics.mjs";

// Resolves the model to quantize per the "Quantize input" radio: the raw
// original upload, or the Convert section's own simplify/optimize output
// (window.__onnxsimConverted, published by index.html's worker glue when a
// conversion finishes) -- letting quantization run on an already-simplified
// graph, where fused patterns (e.g. Conv+BN, MatMul+Add -> Gemm) match more
// of what onnxsim's quantize passes look for. Throws a user-facing message
// if "converted" is selected but nothing has been converted yet.
export async function resolveQuantizeInput(source, fileInput) {
  if (source === "converted") {
    const converted = window.__onnxsimConverted;
    if (!converted || !converted.bytes) {
      throw new Error(
        "no onnxsim-simplified result yet -- run Simplify/Optimize in the " +
          "Convert section above first, or switch back to 'Original upload'.",
      );
    }
    return { bytes: converted.bytes, name: converted.name };
  }
  return resolveOriginalModelBytes(fileInput);
}

// Resolve this page's WASM runtime (published by index.html's inline loader).
export function getRuntime() {
  return window.__onnxsimRuntimePromise;
}

// Methods that quantize straight from a single WASM call -- the weight is
// quantized from its own static values and the activation either isn't
// touched (weight-only) or is quantized dynamically at inference time
// (dynamic/ternary), so no calibration data is needed. Float16/BFloat16 are
// a different kind of "quantization" (a narrower floating-point format, not
// an integer scheme) but need no calibration data either, so they fit the
// same single-call shape; Float8 needs an extra "e4m3"/"e5m2" format
// argument and is called separately below instead of through this map.
const NO_CALIBRATION_METHODS = {
  dynamic: "onnxsim_quantize_dynamic",
  ternary: "onnxsim_quantize_ternary",
  weight_only: "onnxsim_quantize_weight_only",
  weight_only_int4: "onnxsim_quantize_weight_only_int4",
  fp16: "onnxsim_quantize_fp16",
  bf16: "onnxsim_quantize_bf16",
};

function toArray(vec) {
  const out = new Array(vec.size());
  for (let i = 0; i < out.length; i++) out[i] = vec.get(i);
  return out;
}

function initQuantizePanel() {
  const btn = document.getElementById("quantize-button");
  if (!btn) return;
  const fileInput = document.getElementById("file-input");
  const statusEl = document.getElementById("quantize-status");
  const dlBtn = document.getElementById("quantize-download");
  const samplesEl = document.getElementById("quantize-samples");
  const epEl = document.getElementById("quantize-ep");
  const metricsToggleEl = document.getElementById("quantize-metrics-toggle");
  const metricsEl = document.getElementById("quantize-metrics");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  let lastBytes = null;

  btn.addEventListener("click", async () => {
    const methodEl = document.querySelector('input[name="quantize-method"]:checked');
    const method = methodEl ? methodEl.value : "dynamic";
    const sourceEl = document.querySelector('input[name="quantize-source"]:checked');
    const source = sourceEl ? sourceEl.value : "original";
    btn.disabled = true;
    if (dlBtn) dlBtn.style.display = "none";
    if (metricsEl) {
      metricsEl.style.display = "none";
      metricsEl.innerHTML = "";
    }
    try {
      setStatus("loading model…");
      const { bytes, name } = await resolveQuantizeInput(source, fileInput);
      // A fresh ArrayBuffer copy: the WASM calls below read it (never
      // transferred/detached), and it's reused across several calls when
      // calibrating.
      const modelBuf = bytes.slice().buffer;
      const runtime = await getRuntime();

      let resultBuf;
      if (method === "fp8") {
        const formatEl = document.getElementById("quantize-fp8-format");
        const format = formatEl ? formatEl.value : "e4m3";
        setStatus(`quantizing (fp8, ${format})…`);
        resultBuf = runtime.onnxsim_quantize_fp8(modelBuf, format);
      } else if (method in NO_CALIBRATION_METHODS) {
        setStatus(`quantizing (${method})…`);
        resultBuf = runtime[NO_CALIBRATION_METHODS[method]](modelBuf);
      } else if (method === "static" || method === "qoperator") {
        setStatus("finding quantizable tensors…");
        let names = toArray(runtime.onnxsim_list_quantizable_activations(modelBuf));
        if (method === "qoperator") {
          const outNames = toArray(runtime.onnxsim_list_qoperator_quantizable_outputs(modelBuf));
          names = Array.from(new Set([...names, ...outNames]));
        }
        if (names.length === 0) {
          setStatus(
            "no quantizable MatMul/Gemm" + (method === "static" ? "/Conv" : "") +
              " (with a constant float32 weight) found in this model.",
          );
          return;
        }
        const numSamples = Math.max(1, parseInt((samplesEl && samplesEl.value) || "8", 10) || 8);
        // Calibration's own forward passes are the one part of quantizing that
        // actually runs the model, so they honour this panel's own execution
        // provider picker (WASM by default; every accelerated choice falls
        // back to WASM -- see webnn.mjs / quantize_calibration.mjs).
        const epValue = epEl ? epEl.value : "wasm";
        const { providers, needWebnn } = providersForEp(epValue);
        if (isWebnnEp(epValue)) {
          setStatus("WebNN is experimental; calibration falls back to WASM if the device or an operator is unsupported.");
        }
        setStatus(`calibrating ${names.length} tensor(s) over ${numSamples} random sample(s) on ${epValue}…`);
        const { names: calNames, flat } = await calibrateRanges(runtime, modelBuf, names, numSamples, setStatus, {
          providers,
          needWebnn,
        });
        if (calNames.length === 0) {
          setStatus("calibration produced no ranges (the model may have no runnable inputs).");
          return;
        }
        setStatus(`quantizing (${method})…`);
        const fn = method === "static" ? "onnxsim_quantize_static" : "onnxsim_quantize_qoperator";
        resultBuf = runtime[fn](modelBuf, calNames, flat);
      } else {
        setStatus(`unknown quantize method: ${method}`);
        return;
      }

      if (!resultBuf) {
        setStatus("quantization failed (see console for details).");
        return;
      }
      // Copy out of the wasm heap and grab a base64 copy for Netron/inference
      // before any other wasm call can invalidate the view.
      const outBytes = new Uint8Array(resultBuf).slice();
      const dataUrl = "data:application/octet-stream;base64," + resultBuf.toBase64();
      lastBytes = outBytes;
      const outName = name.replace(/\.onnx$/i, "") + `.quant_${method}.onnx`;
      setStatus(`done (${outBytes.length.toLocaleString()} bytes).`);
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, outName);
      }
      // Show the result in its own "Quantized" Netron pane and publish it as
      // the inference panel's "quantized" source -- deliberately independent
      // of netronShowAfter/window.__onnxsimConverted (the Convert section's
      // own output), so quantizing never clobbers the plain simplify/optimize
      // result and all three stages stay inspectable side by side.
      if (window.netronShowQuantized) window.netronShowQuantized(dataUrl, outName);
      window.__onnxsimQuantized = { bytes: outBytes, name: outName };

      // Best-effort result-quality report: runs float vs. quantized on the
      // same random input through onnxruntime-web. A failure here (e.g. no
      // usable execution provider, or an input shape onnxruntime-web can't
      // auto-fill) shouldn't take away the quantized model itself, which is
      // already valid and downloadable at this point -- so it only appends a
      // note to the status line rather than replacing it or throwing.
      if (!metricsToggleEl || metricsToggleEl.checked) {
        try {
          setStatus(`done (${outBytes.length.toLocaleString()} bytes). measuring result quality…`);
          const quality = await computeQuantizationQuality(bytes, outBytes, setStatus);
          if (metricsEl) {
            metricsEl.innerHTML = renderQuantizationQuality(quality);
            metricsEl.style.display = "";
          }
          setStatus(`done (${outBytes.length.toLocaleString()} bytes).`);
        } catch (metricsErr) {
          setStatus(
            `done (${outBytes.length.toLocaleString()} bytes). ` +
              "result-quality check failed: " +
              (metricsErr && metricsErr.message ? metricsErr.message : metricsErr),
          );
        }
      }
    } catch (err) {
      setStatus("quantize error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

initQuantizePanel();
