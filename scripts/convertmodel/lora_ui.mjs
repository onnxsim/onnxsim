// Browser glue for the "Fine-tune with a LoRA adapter" panel on the
// converter page.
//
// Client-side LoRA/QLoRA-style fine-tuning: a trainable low-rank
// (X @ A @ B) branch is injected around each eligible
// MatMul/Gemm/Conv layer of a *base* model, and only A/B are trained --
// every base weight stays frozen -- against a *reference* model's own
// output on the same calibration rows. docs/qat.md's "Browser QAT panel"
// subsection is this panel's own direct precedent, both in shape and in the
// two lessons it recorded (ort_executor.mjs being the wrong runner for a
// training loop, and the minibatch row stream not being bit-compatible with
// Python); lora_finetune.mjs's own top comment explains where this panel's
// C++/JS split follows qat_ui.mjs's exactly and where it does not (LoRA's
// own injection step, dual-model capture).
//
// The division of labour, from onnxsim/lora_entry.h:
//   - the wasm module injects the adapter (onnxsim_lora_inject), builds one
//     optimizer step as a plain inference graph
//     (onnxsim_lora_build_step_graph) and folds the finished state back in
//     (onnxsim_lora_write_back);
//   - onnxruntime-web captures the base model's own block-external
//     activations and the reference model's reconstruction target, then runs
//     that step graph in a loop, on whichever execution provider this
//     panel's own picker selects -- WebGPU by default, for the same reason
//     the QAT panel defaults to it (the loop is hundreds of forward+backward
//     passes over the block);
//   - lora_finetune.mjs holds the loop itself and every pure decision inside
//     it, exactly as qat_finetune.mjs does for QAT (and, for the pieces that
//     are not LoRA-specific, literally re-exports qat_finetune.mjs's own
//     functions rather than re-implementing them -- see that file's own top
//     comment).
//
// This file is only the DOM: reading the controls, running the flow, and
// reporting. Like qat_ui.mjs it runs the wasm calls on the page's own
// runtime rather than through the worker, and it publishes its own result on
// window.__onnxsimLoraTrained -- independent of the Quantize and QAT panels'
// own outputs, and never overwriting them.
//
// **Block boundaries**, like the QAT panel's own: a caller-named tensor pair
// trains exactly that one block; leaving both fields blank discovers blocks
// via lora_blocks.mjs's discoverLoraBlocks (a liveness-cut walk over the
// *injected* model, closing a block once it has accumulated
// "adapters per block" of the adapter's own targets -- lora_blocks.mjs's own
// header has the full argument, ported from qat_blocks.mjs's identical one
// for QAT) and trains each discovered block in turn via
// trainLoraAdapterAllBlocks, the same "propose then train every block,
// skipping one build_step_graph refuses rather than aborting the run" shape
// the QAT panel's own button handler already has inline. A block that turns
// out untrainable is skipped and reported, not fatal to the others.
//
// **Two things this panel does not do**, both flagged in the panel's own
// static copy in index.html rather than only here: it has no QLoRA
// (NF4-quantized base) composition wired in -- onnxsim.nf4's quantizer has
// no C++ port, per lora_entry.h's own top comment on why that is
// deliberately out of this header's scope -- and it trains only against a
// reference model's own activations (train_lora's `reference_model=` mode),
// never against caller-supplied `target_data`. Both are follow-ups, not
// oversights; see lora_finetune.mjs's trainLoraAdapter for the fuller
// reasoning on the second.

import { downloadBytes } from "./download.mjs";
import { resolveOriginalModelBytes, loadOrt } from "./inference_browser.mjs";
import {
  buildCalibrationRows,
  renderLossCurve,
  trainLoraAdapter,
  trainLoraAdapterAllBlocks,
} from "./lora_finetune.mjs";
import { computeQuantizationQuality, renderQuantizationQuality } from "./quantize_metrics.mjs";
import { providersForEp, isWebnnEp } from "./webnn.mjs";

// Resolve one of the page's three model slots -- the base to inject/train and
// the reference to reproduce are both drawn from the same three sources
// quantize_ui.mjs's/qat_ui.mjs's own resolvers use, since a LoRA run
// legitimately wants any pairing of them (base = simplified against
// reference = original is the panel's own suggested default; original
// against original is a same-model smoke test with a loss that starts at
// zero and stays there, per lora_entry.h's own numeric-no-op note).
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

// The InjectLoraOptions object onnxsim_lora_inject takes, read off the panel.
// Absent fields keep the binding's own defaults.
function readInjectOptions(el) {
  const checked = (id) => {
    const box = el(id);
    return box ? box.checked : false;
  };
  const number = (id, fallback) => {
    const input = el(id);
    const value = parseInt((input && input.value) || "", 10);
    return Number.isFinite(value) ? value : fallback;
  };
  const targetOpTypes = [
    ["lora-target-matmul", "MatMul"],
    ["lora-target-gemm", "Gemm"],
    ["lora-target-conv", "Conv"],
  ]
    .filter(([id]) => checked(id))
    .map(([, op]) => op);

  const alphaText = ((el("lora-alpha") || {}).value || "").trim();
  const hasAlpha = alphaText.length > 0;
  const alpha = hasAlpha ? parseFloat(alphaText) : 0;
  if (hasAlpha && !Number.isFinite(alpha)) {
    throw new Error(`'${alphaText}' is not a valid alpha`);
  }

  const restrictTargetNames = checked("lora-restrict-names");
  const targetNames = ((el("lora-target-names") || {}).value || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  return {
    rank: number("lora-rank", 8),
    hasAlpha,
    alpha,
    targetOpTypes,
    restrictTargetNames,
    targetNames,
    seed: number("lora-seed", 0),
  };
}

function initLoraPanel() {
  const el = (id) => document.getElementById(id);
  const btn = el("lora-button");
  if (!btn) return;
  const fileInput = el("file-input");
  const statusEl = el("lora-status");
  const dlBtn = el("lora-download");
  const curveEl = el("lora-curve");
  const metricsEl = el("lora-metrics");

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
      const baseSource = el("lora-base") ? el("lora-base").value : "converted";
      const referenceSource = el("lora-reference") ? el("lora-reference").value : "original";
      const base = await resolveModel(baseSource, fileInput);
      const reference = await resolveModel(referenceSource, fileInput);
      const baseBytes = base.bytes.slice();
      const referenceBytes = reference.bytes.slice();

      const runtime = await window.__onnxsimRuntimePromise;
      const injectOptions = readInjectOptions(el);
      const numRows = Math.max(1, parseInt((el("lora-rows") || {}).value || "8", 10) || 8);
      const numSteps = Math.max(1, parseInt((el("lora-steps") || {}).value || "200", 10) || 200);
      const rates = {
        learningRate: parseFloat((el("lora-lr") || {}).value || "1e-3") || 1e-3,
        lrDecay: !el("lora-lr-decay") || el("lora-lr-decay").checked,
      };
      const options = {
        batchSize: Math.max(0, parseInt((el("lora-batch-size") || {}).value || "0", 10) || 0),
        batchSeed: Math.max(0, parseInt((el("lora-batch-seed") || {}).value || "0", 10) || 0),
        shuffle: true,
      };
      const epValue = el("lora-ep") ? el("lora-ep").value : "webgpu";
      const { providers, needWebnn } = providersForEp(epValue);
      if (isWebnnEp(epValue)) {
        setStatus(
          "WebNN is experimental; the training loop falls back to WASM if the device or an operator is unsupported.",
        );
      }

      const blockInput = ((el("lora-block-input") || {}).value || "").trim() || null;
      const blockOutput = ((el("lora-block-output") || {}).value || "").trim() || null;

      // The calibration rows, built once against the base model: one
      // Hugging Face sample per row under "sample data" (hf_datasets.mjs), or
      // a synthetic fill.
      const fill = el("lora-fill") ? el("lora-fill").value : "sample";
      const ort = await loadOrt(needWebnn ? "all" : "default");
      const metaSession = await ort.InferenceSession.create(baseBytes, {
        executionProviders: ["wasm"],
      });
      setStatus(`building ${numRows} calibration row(s) (${fill})…`);
      const rowFeeds = await buildCalibrationRows(ort, metaSession, numRows, {
        fill,
        modelName: base.name,
        log: setStatus,
      });

      // Either the two tensor names the user typed (one block, exactly as
      // before), or lora_blocks.mjs's own liveness-cut proposal over the
      // injected model -- the QAT panel's own default shape, ported: leaving
      // both blank discovers blocks rather than assuming the whole graph is
      // the right unit, since a model whose injected adapters are spread
      // across enough undifferentiable ops has no other way to train more
      // than the first reachable one.
      let result;
      let skipped = [];
      if (blockInput && blockOutput) {
        setStatus("injecting adapter and training…");
        result = await trainLoraAdapter(runtime, {
          baseBytes,
          referenceBytes,
          injectOptions,
          blockInput,
          blockOutput,
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
                `step ${t + 1}/${numSteps}` +
                  (loss != null ? `, loss ${loss.toPrecision(4)}` : "") + "…",
              );
            }
          },
        });
      } else {
        const maxTargetsPerBlock = Math.max(
          1,
          parseInt((el("lora-max-targets") || {}).value || "2", 10) || 2,
        );
        setStatus("injecting adapter and discovering blocks…");
        result = await trainLoraAdapterAllBlocks(runtime, {
          baseBytes,
          referenceBytes,
          injectOptions,
          maxTargetsPerBlock,
          rowFeeds,
          options,
          numSteps,
          rates,
          providers,
          needWebnn,
          log: setStatus,
          onStep: (i, total, t, loss) => {
            if (t % 10 === 0) {
              setStatus(
                `block ${i + 1}/${total}: step ${t + 1}/${numSteps}` +
                  (loss != null ? `, loss ${loss.toPrecision(4)}` : "") + "…",
              );
            }
          },
        });
        if (result.results.length === 0) {
          setStatus(
            "no trainable block found in this model -- its activations never narrow to a " +
              "single live tensor with an injected adapter behind it, so there is nowhere " +
              "to cut. Name a block's input and output tensor explicitly to train one anyway.",
          );
          return;
        }
        // A block onnxsim_lora_build_step_graph refuses (an op with no
        // gradient rule discoverLoraBlocks' own pre-filter missed, say) is a
        // block to skip, not a run to abandon -- apply_qat_all_blocks' own
        // rule, mirrored by trainLoraAdapterAllBlocks itself.
        skipped = result.results
          .filter((r) => !r.trained)
          .map((r) => `${r.block.input} → ${r.block.output}: ${r.skippedReason}`);
        if (skipped.length === result.results.length) {
          setStatus(`no block could be trained. ${skipped.join(" | ")}`);
          return;
        }
      }

      lastBytes = result.bytes;
      const outName = base.name.replace(/\.onnx$/i, "") + ".lora.onnx";
      const skippedNote = skipped.length ? ` ${skipped.length} block(s) skipped (see below).` : "";
      setStatus(
        `done: ${result.adapter.length} adapter(s) trained, ` +
          `${result.bytes.length.toLocaleString()} bytes.${skippedNote}`,
      );
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, outName);
      }
      window.__onnxsimLoraTrained = { bytes: result.bytes, name: outName };

      if (curveEl && result.losses.length) {
        curveEl.innerHTML =
          renderLossCurve(result.losses) +
          (skipped.length
            ? `<p class="tool-note">skipped: ${skipped.map(escapeHtml).join("<br>")}</p>`
            : "");
        curveEl.style.display = "";
      }

      // Before/after against the reference model -- best-effort, exactly as
      // the QAT panel's own metrics check: a failure here (no usable
      // execution provider, an input onnxruntime-web cannot auto-fill) must
      // not take away the trained model, which is already valid and
      // downloadable.
      if (metricsEl && (!el("lora-metrics-toggle") || el("lora-metrics-toggle").checked)) {
        try {
          setStatus("measuring result quality (before vs after)…");
          const before = await computeQuantizationQuality(referenceBytes, base.bytes, setStatus);
          const after = await computeQuantizationQuality(referenceBytes, result.bytes, setStatus);
          metricsEl.innerHTML =
            "<p class=\"tool-note\"><b>Before training</b> (base vs reference)</p>" +
            renderQuantizationQuality(before) +
            "<p class=\"tool-note\"><b>After training</b> (adapted vs reference)</p>" +
            renderQuantizationQuality(after);
          metricsEl.style.display = "";
          setStatus(
            `done: ${result.adapter.length} adapter(s) trained, relative L2 error ` +
              `${(before.relL2 * 100).toFixed(2)}% → ${(after.relL2 * 100).toFixed(2)}%.${skippedNote}`,
          );
        } catch (metricsErr) {
          setStatus(
            `done: ${result.adapter.length} adapter(s) trained. result-quality check failed: ` +
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

// Skipped-block reasons come from exception messages, which can carry a
// tensor name straight out of the model -- untrusted text, unlike every
// other value this panel interpolates. Mirrors qat_ui.mjs's own copy rather
// than importing it: neither module exports it, and it is four lines.
function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

initLoraPanel();
