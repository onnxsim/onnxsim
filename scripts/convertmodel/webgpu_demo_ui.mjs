// DOM glue for the "WebGPU training demo" panel: loads onnxruntime-web and
// one of the two pre-built step-graph fixtures (test/step_qat_hf_demo.onnx
// or test/step_qat_cifar10_pretrain.onnx -- both checked in, deployed
// alongside this page's own static files, and built entirely offline by
// test/make_step_graph_fixtures.py), runs webgpu_hf_demo.mjs's shared
// driving loop against real data fetched live from Hugging Face, and renders
// the loss curve with qat_finetune.mjs's own renderLossCurve -- the same
// widget the QAT/LoRA panels already use for their own training runs.
//
// Unlike those panels, this one operates on no user-supplied model: both
// demos are fixed, tiny, self-contained step graphs whose only per-run input
// is the real photo(s) fetched live, so there is nothing here to upload,
// convert, or pick a block boundary in -- just a mode and a button.

import { loadOrt } from "./inference_browser.mjs";
import { runPhotoDemo, runCifar10PretrainDemo } from "./webgpu_hf_demo.mjs";
import { renderLossCurve } from "./qat_finetune.mjs";

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const DEMOS = {
  photo: { file: "step_qat_hf_demo", run: runPhotoDemo },
  cifar10: { file: "step_qat_cifar10_pretrain", run: runCifar10PretrainDemo },
};

async function fetchJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${path}`);
  return r.json();
}

async function fetchBytes(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${path}`);
  return r.arrayBuffer();
}

if (el("webgpu-demo-button")) {
  el("webgpu-demo-button").addEventListener("click", async () => {
    const button = el("webgpu-demo-button");
    const statusEl = el("webgpu-demo-status");
    const curveEl = el("webgpu-demo-curve");
    const mode = el("webgpu-demo-mode") ? el("webgpu-demo-mode").value : "photo";
    const demo = DEMOS[mode] || DEMOS.photo;

    const setStatus = (msg) => {
      statusEl.textContent = msg;
    };
    button.disabled = true;
    curveEl.style.display = "none";
    curveEl.innerHTML = "";
    try {
      setStatus("loading onnxruntime-web…");
      const ort = await loadOrt(); // "default" bundle already carries WebGPU

      setStatus("loading the step graph…");
      const [manifest, modelBytes] = await Promise.all([
        fetchJson(`./test/${demo.file}.json`),
        fetchBytes(`./test/${demo.file}.onnx`),
      ]);

      setStatus("fetching real data from Hugging Face…");
      const result = await demo.run({
        ort,
        modelBytes,
        manifest,
        onStep: (t, loss) => {
          if (t % 5 === 0 || t === manifest.scalars.length - 1) {
            setStatus(`training on WebGPU — step ${t + 1}/${manifest.scalars.length}, loss ${loss.toPrecision(4)}…`);
          }
        },
      });

      curveEl.style.display = "";
      let extra;
      if (mode === "cifar10") {
        extra =
          `<p class="tool-note">sample: ${escapeHtml(result.labelNames.join(", "))} — ` +
          `${result.correct}/${result.total} correct after training.</p>`;
      } else {
        extra = `<p class="tool-note">sample photo: ${escapeHtml(result.imageLabel)} (frgfm/imagenette).</p>`;
      }
      curveEl.innerHTML = renderLossCurve(result.losses) + extra;
      setStatus("done.");
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      setStatus(
        `failed: ${message}. This demo requires WebGPU (no wasm/CPU fallback) -- try a recent ` +
          "Chrome/Edge; some setups need chrome://flags/#enable-unsafe-webgpu.",
      );
    } finally {
      button.disabled = false;
    }
  });
}
