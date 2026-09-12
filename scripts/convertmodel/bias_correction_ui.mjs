// Browser glue for the "Correct bias" panel on the converter page.
//
// Recovers accuracy an output-preserving algorithm change cost a model --
// quantization, or (the case this panel is really for) a Resize node's
// mode/coordinate_transformation_mode swapped for one a deployment
// accelerator supports -- entirely in the browser. See
// onnxsim/bias_correction.py's own module docstring for the technique
// (measure a systematic per-channel, or per-position, mean shift on
// calibration data and cancel it with a constant Add) and when it does and
// doesn't help (a mode/coordinate swap's error is mostly *not* a per-channel
// constant -- see "per-position (spatial)" below).
//
// The division of labour, from onnxsim/bias_correction_entry.h:
//   - runtime.onnxsim_list_correctable_outputs (WASM) finds which Conv/Gemm/
//     MatMul/Resize outputs are eligible;
//   - bias_correction_calibration.mjs runs both models on synthetic
//     calibration data through onnxruntime-web and measures each one's
//     correction (this is the only "expensive" step -- ordinary forward
//     passes only, no training loop, no gradients -- see
//     onnxsim.bias_correction's own module docstring for the cost);
//   - runtime.onnxsim_apply_bias_corrections (WASM) splices the
//     already-measured numbers into the modified model.
//
// This file is only the DOM: reading the panel, resolving which two models
// to compare (mirrors qat_ui.mjs's own three-model-slot resolveModel, plus a
// dedicated file input here since a Resize-mode-swapped model, unlike a
// quantized or QAT-trained one, isn't itself something this page produces),
// and reporting -- published on window.__onnxsimBiasCorrected, downloadable,
// and never overwriting any other panel's own output.

import { downloadBytes } from "./download.mjs";
import { resolveOriginalModelBytes } from "./inference_browser.mjs";
import { measureBiasCorrections } from "./bias_correction_calibration.mjs";
import { computeQuantizationQuality, renderQuantizationQuality } from "./quantize_metrics.mjs";
import { providersForEp, isWebnnEp } from "./webnn.mjs";

// Resolves one of the panel's model slots. "uploaded" reads a second,
// panel-local file input (`modifiedFileInput`) rather than the page's global
// one, since the modified model here is typically something produced
// outside this page entirely (an accelerator vendor's own converter).
export async function resolveModel(source, fileInput, modifiedFileInput) {
  if (source === "converted") {
    const converted = window.__onnxsimConverted;
    if (!converted || !converted.bytes) {
      throw new Error(
        "no onnxsim-simplified result yet -- run Simplify/Optimize in the Convert section first.",
      );
    }
    return { bytes: converted.bytes, name: converted.name };
  }
  if (source === "quantized") {
    const quantized = window.__onnxsimQuantized;
    if (!quantized || !quantized.bytes) {
      throw new Error("no quantized model yet -- quantize one in the Quantize panel above first.");
    }
    return { bytes: quantized.bytes, name: quantized.name };
  }
  if (source === "uploaded") {
    const file = modifiedFileInput && modifiedFileInput.files && modifiedFileInput.files[0];
    if (!file) {
      throw new Error("pick a .onnx file for the modified model below first.");
    }
    return { bytes: new Uint8Array(await file.arrayBuffer()), name: file.name };
  }
  return resolveOriginalModelBytes(fileInput);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderCandidates(candidates, corrections) {
  if (candidates.length === 0) {
    return '<p class="tool-note">no eligible Conv/Gemm/MatMul/Resize output found in both models.</p>';
  }
  const corrected = new Set(corrections.map((c) => c.name));
  const rows = candidates.map((c) => {
    const status = corrected.has(c.name) ? "corrected" : "no measurable/verified bias -- left alone";
    return `<tr><td style="padding: 2px 0.8em 2px 0;">${esc(c.name)}</td><td style="color: var(--muted);">${status}</td></tr>`;
  });
  return `<table style="border-collapse: collapse; font-size: 0.85em;">${rows.join("")}</table>`;
}

function initBiasCorrectionPanel() {
  const el = (id) => document.getElementById(id);
  const btn = el("bc-button");
  if (!btn) return;
  const fileInput = el("file-input");
  const modifiedFileInput = el("bc-modified-file");
  const statusEl = el("bc-status");
  const dlBtn = el("bc-download");
  const candidatesEl = el("bc-candidates");
  const metricsEl = el("bc-metrics");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  const modifiedSourceEl = el("bc-modified-source");
  const syncModifiedFileVisibility = () => {
    if (!modifiedFileInput || !modifiedSourceEl) return;
    modifiedFileInput.style.display = modifiedSourceEl.value === "uploaded" ? "" : "none";
  };
  if (modifiedSourceEl) {
    modifiedSourceEl.addEventListener("change", syncModifiedFileVisibility);
    syncModifiedFileVisibility();
  }

  let lastBytes = null;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    if (dlBtn) dlBtn.style.display = "none";
    for (const box of [candidatesEl, metricsEl]) {
      if (box) {
        box.style.display = "none";
        box.innerHTML = "";
      }
    }
    try {
      setStatus("loading models…");
      const floatSource = el("bc-float") ? el("bc-float").value : "original";
      const modifiedSource = modifiedSourceEl ? modifiedSourceEl.value : "quantized";
      const floatModel = await resolveModel(floatSource, fileInput, modifiedFileInput);
      const modifiedModel = await resolveModel(modifiedSource, fileInput, modifiedFileInput);
      const floatBytes = floatModel.bytes.slice();
      const modifiedBytes = modifiedModel.bytes.slice();

      const runtime = await window.__onnxsimRuntimePromise;
      const spatial = el("bc-spatial") ? el("bc-spatial").checked : false;
      const numSamples = Math.max(
        1,
        parseInt((el("bc-samples") || {}).value || (spatial ? "16" : "8"), 10) || (spatial ? 16 : 8),
      );
      const gridSize = Math.max(1, parseInt((el("bc-grid-size") || {}).value || "8", 10) || 8);
      const validationFraction = Math.min(
        0.9,
        Math.max(0.1, parseFloat((el("bc-validation-fraction") || {}).value || "0.3") || 0.3),
      );
      const epValue = el("bc-ep") ? el("bc-ep").value : "wasm";
      const { providers, needWebnn } = providersForEp(epValue);
      if (isWebnnEp(epValue)) {
        setStatus("WebNN is experimental; calibration falls back to WASM if the device is unsupported.");
      }

      setStatus(`measuring ${spatial ? "per-position" : "per-channel"} bias correction…`);
      const { corrections, candidates } = await measureBiasCorrections(runtime, floatBytes, modifiedBytes, {
        spatial,
        numSamples,
        gridSize,
        validationFraction,
        providers,
        log: setStatus,
      });

      if (corrections.length === 0) {
        setStatus(
          candidates.length === 0
            ? "no eligible Conv/Gemm/MatMul/Resize output found in both models -- nothing to correct."
            : "no candidate's correction was measurable" +
                (spatial ? " and verified to help on held-out data" : "") +
                " -- the modified model is returned unchanged.",
        );
        if (candidatesEl) {
          candidatesEl.innerHTML = renderCandidates(candidates, corrections);
          candidatesEl.style.display = "";
        }
        return;
      }

      setStatus(`applying ${corrections.length} correction(s)…`);
      const correctedBuf = runtime.onnxsim_apply_bias_corrections(modifiedBytes, corrections);
      if (!correctedBuf) {
        throw new Error("applying the correction failed (see console for details).");
      }
      const correctedBytes = new Uint8Array(correctedBuf).slice();
      lastBytes = correctedBytes;
      const outName = modifiedModel.name.replace(/\.onnx$/i, "") + ".bias_corrected.onnx";

      if (candidatesEl) {
        candidatesEl.innerHTML = renderCandidates(candidates, corrections);
        candidatesEl.style.display = "";
      }
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, outName);
      }
      window.__onnxsimBiasCorrected = { bytes: correctedBytes, name: outName };

      setStatus(`done: ${corrections.length} of ${candidates.length} candidate(s) corrected.`);

      // Before/after against the float model -- best-effort, same convention
      // as the QAT/Quantize panels: a failure here must not take away the
      // corrected model, which is already valid and downloadable.
      if (metricsEl && (!el("bc-metrics-toggle") || el("bc-metrics-toggle").checked)) {
        try {
          setStatus("measuring result quality (before vs after)…");
          const before = await computeQuantizationQuality(floatBytes, modifiedBytes, setStatus);
          const after = await computeQuantizationQuality(floatBytes, correctedBytes, setStatus);
          metricsEl.innerHTML =
            '<p class="tool-note"><b>Before correction</b></p>' +
            renderQuantizationQuality(before) +
            '<p class="tool-note"><b>After correction</b></p>' +
            renderQuantizationQuality(after);
          metricsEl.style.display = "";
          setStatus(
            `done: ${corrections.length} of ${candidates.length} candidate(s) corrected, ` +
              `relative L2 error ${(before.relL2 * 100).toFixed(2)}% → ${(after.relL2 * 100).toFixed(2)}%.`,
          );
        } catch (metricsErr) {
          setStatus(
            `done: ${corrections.length} of ${candidates.length} candidate(s) corrected. ` +
              "result-quality check failed: " + (metricsErr && metricsErr.message ? metricsErr.message : metricsErr),
          );
        }
      }
    } catch (err) {
      setStatus("bias correction error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

initBiasCorrectionPanel();
