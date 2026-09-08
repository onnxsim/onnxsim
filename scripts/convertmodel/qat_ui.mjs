// Browser glue for the "Fine-tune (QAT)" panel on the converter page.
//
// Block-wise, label-free quantization-aware training, run entirely in the
// browser: the float model is the teacher, the quantized model is the student,
// and each block's weights (optionally its scales) are trained by Adam against
// the teacher's own activations at that block's output. docs/qat.md calls this
// stage 3, and it is the one place the whole feature becomes something a user
// can reach without Python -- client-side QAT with no server.
//
// The division of labour, from onnxsim/qat_entry.h:
//   - the wasm module builds one optimizer step as a plain inference graph
//     (onnxsim_qat_build_step_graph) and folds the finished state back in
//     (onnxsim_qat_write_back);
//   - onnxruntime-web captures the teacher's activations and then runs that
//     step graph in a loop, on whichever execution provider this panel's own
//     picker selects -- WebGPU by default here, since the loop is hundreds of
//     forward+backward passes over one block and is the one thing on this page
//     with enough arithmetic in it to pay for an accelerator;
//   - qat_finetune.mjs holds the loop itself and every pure decision inside it
//     (the scalar schedule, the minibatch stream, the capture binding, the
//     state ping-pong), and qat_blocks.mjs proposes the blocks.
//
// This file is only the DOM: reading the controls, sequencing the blocks, and
// reporting. Like quantize_ui.mjs it runs the wasm calls on the page's own
// runtime rather than through the worker, since none of them needs a model
// executor, and it keeps its result independent of every other panel's --
// published on window.__onnxsimFineTuned, downloadable, and never overwriting
// the Quantize panel's own output.
//
// **Two things the panel does not claim.** Activations are captured from the
// float model for every block, i.e. onnxsim's capture-once walk rather than
// apply_qat_all_blocks' sequential one (which re-captures each block's input
// from the partly-tuned student, and measures better -- docs/qat.md, stage 2).
// And the objective is reconstruction of the teacher: it cannot exceed the
// float model, and on a calibration set this small it is a demonstration of
// the machinery rather than a production tuning run.

import { downloadBytes } from "./download.mjs";
import { resolveOriginalModelBytes, loadOrt } from "./inference_browser.mjs";
import { discoverBlocks } from "./qat_blocks.mjs";
import { buildCalibrationRows, fineTuneBlock, renderLossCurve } from "./qat_finetune.mjs";
import { computeQuantizationQuality, renderQuantizationQuality } from "./quantize_metrics.mjs";
import { providersForEp, isWebnnEp } from "./webnn.mjs";

// Resolve one of the page's three model slots, the way quantize_ui.mjs's
// resolveQuantizeInput resolves its two. "quantized" is the Quantize panel's
// own output (window.__onnxsimQuantized), which is the student this panel
// trains; with fake-quant off the student is instead whatever else changed the
// model (a pruned or simplified graph), which is why "converted" is offered
// for it too.
export async function resolveModel(source, fileInput) {
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
      throw new Error(
        "no quantized model yet -- quantize one in the Quantize panel above first.",
      );
    }
    return { bytes: quantized.bytes, name: quantized.name };
  }
  return resolveOriginalModelBytes(fileInput);
}

// The QatOptions object onnxsim_qat_build_step_graph takes, read off the
// panel. Absent fields keep the binding's own defaults, so only what the user
// actually chose is sent.
function readOptions(el) {
  const checked = (id) => {
    const box = el(id);
    return box ? box.checked : false;
  };
  const number = (id, fallback) => {
    const input = el(id);
    const value = parseInt((input && input.value) || "", 10);
    return Number.isFinite(value) ? value : fallback;
  };
  const fakeQuant = !checked("qat-finetune-only");
  if (!fakeQuant && (checked("qat-learn-scales") || checked("qat-learn-act-scales"))) {
    // Both flags name a parameter of a quantizer, and fake-quant off is the
    // mode with no quantizer in it. BuildQatStepGraph refuses that pairing
    // rather than ignoring it, and the panel says why rather than quietly
    // unticking a box: a user who asked to learn scales and got a model whose
    // scales are untouched has no way to tell that from a run where learning
    // them did not help.
    throw new Error(
      "'fine-tune only' takes the quantizer out of the middle, so there are no weight or " +
        "activation scales left to learn -- untick one of the three.",
    );
  }
  return {
    fakeQuant,
    learnScales: checked("qat-learn-scales"),
    learnActivationScales: checked("qat-learn-act-scales"),
    preserveSparsity: checked("qat-preserve-sparsity"),
    batchSize: number("qat-batch-size", 0),
    batchSeed: number("qat-batch-seed", 0),
    shuffle: true,
  };
}

function initQatPanel() {
  const el = (id) => document.getElementById(id);
  const btn = el("qat-button");
  if (!btn) return;
  const fileInput = el("file-input");
  const statusEl = el("qat-status");
  const dlBtn = el("qat-download");
  const curveEl = el("qat-curve");
  const metricsEl = el("qat-metrics");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  let lastBytes = null;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    if (dlBtn) dlBtn.style.display = "none";
    for (const box of [curveEl, metricsEl]) {
      if (box) {
        box.style.display = "none";
        box.innerHTML = "";
      }
    }
    try {
      setStatus("loading models…");
      const teacherSource = el("qat-teacher") ? el("qat-teacher").value : "original";
      const studentSource = el("qat-student") ? el("qat-student").value : "quantized";
      const teacher = await resolveModel(teacherSource, fileInput);
      const student = await resolveModel(studentSource, fileInput);
      const floatBytes = teacher.bytes.slice();
      let studentBytes = student.bytes.slice();

      const runtime = await window.__onnxsimRuntimePromise;
      const options = readOptions(el);
      const numRows = Math.max(1, parseInt((el("qat-rows") || {}).value || "8", 10) || 8);
      const numSteps = Math.max(1, parseInt((el("qat-steps") || {}).value || "200", 10) || 200);
      const rates = {
        learningRate: parseFloat((el("qat-lr") || {}).value || "1e-4") || 1e-4,
        scaleLearningRate: 1e-5,
        activationLearningRate: 1e-2,
        lrDecay: !el("qat-lr-decay") || el("qat-lr-decay").checked,
      };
      const epValue = el("qat-ep") ? el("qat-ep").value : "webgpu";
      const { providers, needWebnn } = providersForEp(epValue);
      if (isWebnnEp(epValue)) {
        setStatus(
          "WebNN is experimental; the training loop falls back to WASM if the device or an operator is unsupported.",
        );
      }

      // The blocks: either the two tensor names the user typed, or the
      // liveness-cut proposal over the float graph.
      let blocks;
      const blockInput = ((el("qat-block-input") || {}).value || "").trim();
      const blockOutput = ((el("qat-block-output") || {}).value || "").trim();
      if (blockInput && blockOutput) {
        blocks = [{ input: blockInput, output: blockOutput }];
      } else {
        const maxLayers = Math.max(1, parseInt((el("qat-max-layers") || {}).value || "2", 10) || 2);
        blocks = discoverBlocks(floatBytes, { maxLayersPerBlock: maxLayers });
        if (blocks.length === 0) {
          setStatus(
            "no trainable block found in this model -- its activations never narrow to a " +
              "single live tensor, so there is nowhere to cut. Name a block's input and " +
              "output tensor explicitly to train one anyway.",
          );
          return;
        }
        setStatus(`discovered ${blocks.length} block(s).`);
      }

      // The calibration rows, built once and reused by every block: one
      // Hugging Face sample per row under "sample data" (hf_datasets.mjs), or
      // a synthetic fill.
      const fill = el("qat-fill") ? el("qat-fill").value : "sample";
      const ort = await loadOrt(needWebnn ? "all" : "default");
      const metaSession = await ort.InferenceSession.create(floatBytes, {
        executionProviders: ["wasm"],
      });
      setStatus(`building ${numRows} calibration row(s) (${fill})…`);
      const rowFeeds = await buildCalibrationRows(ort, metaSession, numRows, {
        fill,
        modelName: teacher.name,
        log: setStatus,
      });

      const allLosses = [];
      let trained = 0;
      const skipped = [];
      for (let i = 0; i < blocks.length; i++) {
        const block = blocks[i];
        const label = `block ${i + 1}/${blocks.length} (${block.input} → ${block.output})`;
        try {
          const result = await fineTuneBlock(runtime, {
            floatBytes,
            quantBytes: studentBytes,
            blockInput: block.input,
            blockOutput: block.output,
            rowFeeds,
            options,
            numSteps,
            rates,
            providers,
            needWebnn,
            log: setStatus,
            onStep: (t, loss) => {
              if (t % 10 === 0) {
                setStatus(
                  `${label}: step ${t + 1}/${numSteps}` +
                    (loss != null ? `, loss ${loss.toPrecision(4)}` : "") + "…",
                );
              }
            },
          });
          studentBytes = result.bytes;
          allLosses.push(...result.losses);
          trained += 1;
        } catch (blockErr) {
          // A block the builder refuses -- an op with no gradient rule, no
          // layer of the requested scheme, shapes that will not infer -- is a
          // block to skip, not a run to abandon: apply_qat_all_blocks records
          // the reason and moves on, and so does this.
          skipped.push(`${label}: ${blockErr && blockErr.message ? blockErr.message : blockErr}`);
        }
      }

      if (trained === 0) {
        setStatus(`no block could be trained. ${skipped.join(" | ")}`);
        return;
      }

      lastBytes = studentBytes;
      const outName = student.name.replace(/\.onnx$/i, "") + ".qat.onnx";
      const skippedNote = skipped.length ? ` ${skipped.length} block(s) skipped (see below).` : "";
      setStatus(
        `done: ${trained}/${blocks.length} block(s) trained, ` +
          `${studentBytes.length.toLocaleString()} bytes.${skippedNote}`,
      );
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, outName);
      }
      window.__onnxsimFineTuned = { bytes: studentBytes, name: outName };

      if (curveEl && allLosses.length) {
        curveEl.innerHTML =
          renderLossCurve(allLosses) +
          (skipped.length
            ? `<p class="tool-note">skipped: ${skipped.map(escapeHtml).join("<br>")}</p>`
            : "");
        curveEl.style.display = "";
      }

      // Before/after against the float model, the same measurement the
      // Quantize panel reports for its own output -- best-effort, exactly as
      // there: a failure here (no usable execution provider, an input
      // onnxruntime-web cannot auto-fill) must not take away the tuned model,
      // which is already valid and downloadable.
      if (metricsEl && (!el("qat-metrics-toggle") || el("qat-metrics-toggle").checked)) {
        try {
          setStatus("measuring result quality (before vs after)…");
          const before = await computeQuantizationQuality(floatBytes, student.bytes, setStatus);
          const after = await computeQuantizationQuality(floatBytes, studentBytes, setStatus);
          metricsEl.innerHTML =
            "<p class=\"tool-note\"><b>Before fine-tuning</b> (quantized vs float)</p>" +
            renderQuantizationQuality(before) +
            "<p class=\"tool-note\"><b>After fine-tuning</b> (tuned vs float)</p>" +
            renderQuantizationQuality(after);
          metricsEl.style.display = "";
          setStatus(
            `done: ${trained}/${blocks.length} block(s) trained, ` +
              `relative L2 error ${(before.relL2 * 100).toFixed(2)}% → ` +
              `${(after.relL2 * 100).toFixed(2)}%.${skippedNote}`,
          );
        } catch (metricsErr) {
          setStatus(
            `done: ${trained}/${blocks.length} block(s) trained. result-quality check failed: ` +
              (metricsErr && metricsErr.message ? metricsErr.message : metricsErr),
          );
        }
      }
    } catch (err) {
      setStatus("fine-tune error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

// Skipped-block reasons come from exception messages, which can carry a tensor
// name straight out of the model -- untrusted text, unlike every other value
// this panel interpolates.
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

initQatPanel();
