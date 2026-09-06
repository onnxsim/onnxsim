importScripts("./onnxsim.js");

// onnxruntime-web CDN location. Only used by the ORT-web build of the module
// (see setupOrtWebIfNeeded below); the default built-in-ORT build never reads
// it. This classic worker can't import the ES-module single source of truth
// (cdn.mjs), so index.html — which does import it — passes the base in via the
// worker URL's `?ortBase=` query param (new Worker("worker.js?ortBase=…")). The
// literal below is only a defensive fallback for a worker started without it;
// the live value always comes from cdn.mjs through the page.
const ORT_BASE =
    new URLSearchParams(self.location.search).get("ortBase") ||
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

// Turn a low-level WASM out-of-memory abort into an actionable explanation.
// When a model needs more heap than the module can address, Emscripten aborts
// with "Cannot enlarge memory, requested N bytes, but the limit is M bytes!"
// (surfaced via printErr) and throws a RuntimeError from the conversion call.
// Both are accurate but say nothing about *why* or *what to do*, so translate
// them into a message the user can act on. The module caps its heap at 4 GiB,
// the hard addressing limit of a wasm32 build (see MAXIMUM_MEMORY in
// CMakeLists.txt); simplification needs several times the model's size on top
// of the model itself, so large models can exhaust it.
function isOutOfMemory(text) {
    return /Cannot enlarge memory|out of memory|Aborted|enlarge|allocat/i.test(
        String(text || ""));
}

function memoryLimitMessage(text) {
    const gib = (n) => (n / (1024 ** 3)).toFixed(2) + " GiB";
    const m = /requested (\d+) bytes, but the limit is (\d+) bytes/.exec(
        String(text || ""));
    const limit = m ? gib(Number(m[2])) : "4 GiB";
    const needed = m ? ` (it needed about ${gib(Number(m[1]))})` : "";
    return [
        `This model is too large to convert in the browser${needed}.`,
        `The WebAssembly build can address at most ${limit} of memory, and`,
        `simplification (shape inference + constant folding) needs several times`,
        `the model's size on top of the model itself.`,
        ``,
        `Things to try:`,
        `  - run onnxsim locally, where it is not limited to ${limit}:`,
        `      pip install onnxsim && onnxsim input.onnx output.onnx`,
        `  - turn off "constant folding" (or lower the tensor-size threshold),`,
        `    then convert again`,
        `  - reduce the model size first (e.g. split it or externalize weights)`,
    ].join("\n");
}

// For the ORT-web build (onnxsim compiled with ONNXSIM_WASM_ORT_WEB): load
// onnxruntime-web and register the runner JsModelExecutor calls. In the default
// built-in-ORT build onnxsim_needs_ort_web() is false and this is a no-op, so
// the worker keeps behaving exactly as before.
async function setupOrtWebIfNeeded(runtime) {
    if (!(typeof runtime.onnxsim_needs_ort_web === "function" &&
          runtime.onnxsim_needs_ort_web())) {
        return;
    }
    const [ortMod, { makeOrtRunner }] = await Promise.all([
        import(/* @vite-ignore */ `${ORT_BASE}ort.min.mjs`),
        import("./ort_executor.mjs"),
    ]);
    const ort = ortMod.default ?? ortMod;
    // Pull the matching wasm binaries from the same CDN directory.
    ort.env.wasm.wasmPaths = ORT_BASE;
    // JsModelExecutor::Run reaches this via val::module_property("onnxsimOrtWebRun").
    runtime.onnxsimOrtWebRun = makeOrtRunner(ort);
}

create_onnxsim({
    preRun: [(runtime) => {
        runtime.ENV.LOG_THRESHOLD = "-1";
    }],
    print: (str) => {
        postMessage(["stdout", str]);
    },
    printErr: (str) => {
        postMessage(["stderr", str]);
        // Augment the raw Emscripten OOM abort line with a human-readable hint.
        if (typeof str === "string" && str.includes("Cannot enlarge memory")) {
            postMessage(["stderr", memoryLimitMessage(str)]);
        }
    },
}).then(async (runtime) => {
    // Wire up onnxruntime-web before announcing readiness, so the first
    // conversion already has a runner registered (only matters in the ORT-web
    // build; a no-op otherwise).
    try {
        await setupOrtWebIfNeeded(runtime);
    } catch (err) {
        postMessage(["stderr", "failed to load onnxruntime-web: " + err]);
        return;
    }
    // Tell the page the WASM runtime is initialized so it can enable the
    // "Choose file" picker. Registering the message listener below only
    // happens now, so any file posted earlier would be dropped.
    postMessage(["ready"]);

    // Decode a standalone safetensors/gguf archive into plain ONNX model
    // bytes via the matching TensorPool import binding. Shared by the
    // conversion pre-step below (which feeds the result into simplify/
    // optimize) and the standalone "import_*" message (which just wants the
    // decoded bytes to show in Netron -- see the "before" pane). Returns the
    // decoded Uint8Array, or null if the archive has no embedded onnxsim
    // model (e.g. a plain weights-only archive with no graph to import).
    function importArchive(format, buf) {
        const fn = format === "gguf" ?
            runtime.onnxsim_import_gguf : runtime.onnxsim_import_safetensors;
        return fn(buf);
    }

    addEventListener("message", async (e) => {
        // Re-export already-converted model bytes into a standalone archive
        // format for the page's download-format selector (safetensors / gguf).
        // Kept separate from the convert switch below: this takes the already-
        // converted bytes (not a raw upload) and needs none of the inline-
        // functions / annotate-original pre-steps a fresh conversion runs.
        if (e.data[0] === "export_safetensors" || e.data[0] === "export_gguf") {
            const format = e.data[0] === "export_gguf" ? "gguf" : "safetensors";
            const exportBuf = e.data[1];
            const filename = e.data[2];
            try {
                const fn = format === "gguf" ?
                    runtime.onnxsim_export_gguf : runtime.onnxsim_export_safetensors;
                const bytes = fn(exportBuf);
                if (!bytes) {
                    postMessage(["export-format-error", format, format + " export failed!"]);
                    return;
                }
                // Copy out of the wasm heap into a standalone ArrayBuffer (the
                // view `bytes` aliases the module's own memory and cannot be
                // transferred) and hand it to the main thread by transfer, not
                // structured-clone. A safetensors/gguf archive embeds the whole
                // model, easily tens of MB for a real model; a base64 data URL
                // would encode it into an even bigger string, clone that whole
                // string across the worker boundary, and then have the browser
                // parse it back out of the anchor's href on click -- all of
                // which a plain ArrayBuffer transfer + Blob URL skips.
                const copy = bytes.slice();
                postMessage(["export-format-done", format, copy.buffer, filename], [copy.buffer]);
            } catch (err) {
                postMessage(["export-format-error", format, String((err && err.message) || err)]);
            }
            return;
        }
        // Decode a safetensors/gguf upload into plain ONNX bytes without
        // running it through simplify/optimize -- used by the "before" Netron
        // pane (netron_view.mjs, via window.__onnxsimImportArchive) so it
        // renders the actual model graph instead of the raw archive bytes.
        if (e.data[0] === "import_safetensors" || e.data[0] === "import_gguf") {
            const format = e.data[0] === "import_gguf" ? "gguf" : "safetensors";
            const importBuf = e.data[1];
            try {
                const bytes = importArchive(format, importBuf);
                if (!bytes) {
                    postMessage(["import-format-error", format,
                        `no embedded onnxsim model found in this ${format} file`]);
                    return;
                }
                // Transfer, not base64 -- same reasoning as the export path
                // above; the decoded model can be tens of MB.
                const copy = bytes.slice();
                postMessage(["import-format-done", format, copy.buffer], [copy.buffer]);
            } catch (err) {
                postMessage(["import-format-error", format, String((err && err.message) || err)]);
            }
            return;
        }
        let buf = e.data[1];
        // The true uploaded bytes, kept aside so the "annotate the original for
        // the inference-compare panel" step below always sees the model the user
        // gave us, even when the inline pre-step replaces `buf` with the inlined
        // model before the main transform runs. Reassigned below (to the
        // decoded ONNX bytes) when the upload was a safetensors/gguf archive,
        // since the raw archive bytes are not a valid ONNX model on their own.
        let originalBuf = e.data[1];
        // `model` is the converted model bytes (a Uint8Array view); `trace` is
        // the onnxsim profiling trace JSON for "simplify" when profiling was
        // requested, otherwise an empty string.
        let model = null;
        let trace = "";
        // A model that outgrows the wasm heap makes Emscripten abort mid-run and
        // throw a RuntimeError out of the conversion call. Catch it here so the
        // user gets an explanation (see memoryLimitMessage) instead of the worker
        // dying with an opaque, unhandled error.
        try {
        // Optional pre-step: the upload was a standalone safetensors/gguf
        // archive (see the page's file picker), not a raw .onnx file -- decode
        // it into an ordinary ONNX model first via the matching TensorPool
        // import binding, so every transform below (and the "original" used
        // for annotate/inference-compare) sees a plain ModelProto exactly like
        // an .onnx upload would produce. No model executor needed, so this is
        // synchronous in both module variants.
        const source_format = e.data[11] || "onnx";
        if (source_format === "safetensors" || source_format === "gguf") {
            const imported = importArchive(source_format, buf);
            if (!imported) {
                postMessage(["stderr",
                    `failed to import ${source_format}: no embedded onnxsim model ` +
                    "found (a plain weights-only archive has no graph to import)"]);
                return;
            }
            buf = new Uint8Array(imported).buffer;
            originalBuf = buf;
        }
        // Optional pre-step: inline the model's local functions before the main
        // transform, so onnx-optimizer / Simplify / constant folding see through
        // them into a plain op graph. Controlled by a checkbox (e.data[9]) and
        // only meaningful for the whole-model transforms; the single-pass debug
        // modes and the standalone "inline" mode below never run it. Inlining
        // needs no model executor, so it is synchronous in both module variants.
        if (e.data[9] &&
            (e.data[0] === "simplify" || e.data[0] === "optimize" ||
             e.data[0] === "optimize_fixed")) {
            const inlined = runtime.onnxsim_inline_functions(buf, false);
            if (!inlined) {
                postMessage(["stderr", "inline functions failed!"]);
                return;
            }
            // Copy out of the wasm heap (the view is invalidated by later wasm
            // calls) and feed the inlined bytes to the selected transform.
            buf = new Uint8Array(inlined).buffer;
        }
        switch (e.data[0]) {
            case "simplify": {
                // Simplify returns { model, trace } so the profiling trace can
                // ride back alongside the converted model. In the ORT-web build
                // onnxsimplify_export is Asyncified and returns a Promise, so
                // await when needed; in the built-in-ORT build it returns the
                // object synchronously and the await is a harmless pass-through.
                let result = runtime.onnxsimplify_export(
                    buf,
                    e.data[2], // skip optimizers
                    e.data[3], // constant folding
                    e.data[4], // shape inference
                    e.data[5], // tensor size threshold
                    e.data[6], // target opset version (<= 0 means keep)
                    e.data[7], // profile (emit a Chrome trace)
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                    e.data[10], // graph diff (node/value-level before/after report)
                );
                if (result && typeof result.then === "function") {
                    result = await result;
                }
                if (result) {
                    model = result.model;
                    trace = result.trace || "";
                }
                break;
            }
            case "optimize":
                model = runtime.onnxoptimizer_optimize(
                    buf,
                    e.data[2], // target optimizers
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                break;
            case "optimize_fixed":
                model = runtime.onnxoptimizer_optimize_fixed(
                    buf,
                    e.data[2], // target optimizers
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                break;
            // Standalone transform: inline the model's local functions into its
            // main graph and return the flattened model. Unlike the pre-step
            // above (which feeds a following simplify/optimize), this is the
            // whole conversion, so it honors the "annotate model info" toggle.
            case "inline":
                model = runtime.onnxsim_inline_functions(
                    buf,
                    e.data[8], // annotate model info (MACs/FLOPs) into metadata_props
                );
                break;
            // Single-pass debugging modes: run exactly one of Simplify's
            // fixed-point building blocks once. Shape inference and data
            // propagation need no executor; constant folding does (Asyncified in
            // the ORT-web build, so await when it returns a Promise). They all
            // return the model bytes through the same convert-done path below.
            case "infer_shapes":
                model = runtime.onnxsim_infer_shapes(buf);
                break;
            case "data_propagation":
                model = runtime.onnxsim_data_propagation(buf);
                break;
            case "fold_constant": {
                let r = runtime.onnxsim_fold_constant(buf, e.data[5]); // tensor size threshold
                if (r && typeof r.then === "function") r = await r;
                model = r;
                break;
            }
            default:
                postMessage(["stderr", "unknown conversion type: " + e.data[0]]);
                return;
        }
        } catch (err) {
            const detail = String((err && err.message) || err);
            if (isOutOfMemory(detail)) {
                // The raw "Cannot enlarge memory" line (if any) already went out
                // via printErr; this adds the actionable explanation.
                postMessage(["stderr", memoryLimitMessage(detail)]);
            } else {
                postMessage(["stderr", e.data[0] + " failed: " + detail]);
            }
            return;
        }
        if (!model) {
            postMessage(["stderr", e.data[0] + " failed!"]);
            return;
        }
        const data_url = "data:application/octet-stream;base64," + model.toBase64();
        // When "annotate model info" is on, also bake the MAC/FLOP metrics into
        // the *original* uploaded model so the "Run inference" panel can report
        // its throughput too — letting the user compare original vs converted
        // inference speed. Annotation only adds metadata_props, so the bytes run
        // identically. Best-effort: a failure here just leaves the original
        // un-annotated (the panel falls back to the raw upload).
        let original_data_url = "";
        if (e.data[8]) {
            try {
                const annotated = runtime.onnxsim_annotate_model_info(originalBuf);
                if (annotated) {
                    original_data_url =
                        "data:application/octet-stream;base64," + annotated.toBase64();
                }
            } catch (err) {
                postMessage(["stderr", "annotate original model failed: " + err]);
            }
        }
        postMessage(["convert-done", data_url, trace, original_data_url]);
    });
});
