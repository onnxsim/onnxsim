// Browser glue for the "Load nanochat demo model" button: fetches the
// bundled nanochat_wasm_demo.onnx same-origin (no network dependency beyond
// this page itself) and drives it through the same Simplify/Optimize/
// Netron/Run-inference path as an uploaded file or a Hugging Face load.
// See scripts/nanochat/README.md for what the model is and why it has no
// upstream ONNX export path to load directly.
//
// nanochat_demo.mjs mirrors hf_load.mjs's conversion hand-off
// (window.__onnxsimOriginal / netronShowBefore / dimParamsShowBefore /
// window.__onnxsimStartConversion) but skips all of hf_load.mjs's
// Hugging-Face-specific machinery (repo resolution, Xet, auth token) since
// this is a single fixed, same-origin file.

const MODEL_NAME = "nanochat_wasm_demo.onnx";

const btn = document.getElementById("nanochat-demo-button");
const statusEl = document.getElementById("nanochat-demo-status");

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

async function loadNanochatDemo() {
  if (typeof window.__onnxsimStartConversion !== "function") {
    setStatus("WebAssembly runtime not ready yet — try again in a moment");
    return;
  }
  btn.disabled = true;
  setStatus("loading…");
  try {
    const resp = await fetch(`./${MODEL_NAME}`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());
    setStatus(`loaded ${MODEL_NAME} (${bytes.length.toLocaleString()} bytes) — converting…`);

    // Publish as the "original" model (there is no file-input entry for this
    // load) and drive the "before" Netron / dim-params panels, same as
    // hf_load.mjs does for a Hugging Face download.
    window.__onnxsimOriginal = { bytes, name: MODEL_NAME };
    if (typeof window.netronShowBefore === "function") {
      window.netronShowBefore(bytes, MODEL_NAME);
    }
    if (typeof window.dimParamsShowBefore === "function") {
      window.dimParamsShowBefore(bytes, MODEL_NAME);
    }

    // startConversion transfers (detaches) the buffer it's given, so hand it
    // a copy and keep `bytes` intact for the inference panel.
    window.__onnxsimStartConversion(MODEL_NAME, bytes.slice().buffer);
    // Leave "converting…" showing; the onnxsim:converted listener below
    // clears it once the worker signals the conversion is done.
  } catch (err) {
    setStatus("load failed: " + (err && err.message ? err.message : String(err)));
  } finally {
    btn.disabled = false;
  }
}

if (btn) btn.addEventListener("click", loadNanochatDemo);

// index.html fires this once a conversion finishes; clear the "converting…"
// status so the panel returns to idle, same as hf_load.mjs's panel.
window.addEventListener("onnxsim:converted", () => {
  setStatus("");
});
