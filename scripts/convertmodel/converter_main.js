// Main converter glue for index.html: gates the file picker on both
// WASM runtimes being ready, wires the worker, and drives conversion.
// Extracted verbatim from the page's inline <script> so it can be
// version-controlled and diffed on its own; it stays a classic script
// (loaded right after onnxsim.js, before the module panels) so its load
// order and behavior are unchanged.

            // The "Choose file" picker must stay disabled until BOTH WASM
            // runtimes are initialized: the one on this page (used to populate
            // the pass list) and the one inside the conversion worker (which
            // actually receives the model). Enabling it earlier lets a user
            // pick a file before the worker is listening, so the conversion
            // message is dropped and the runtime raises uninitialized errors.
            let runtimeReady = false; // this page's WASM (pass list)
            let workerReady = false;  // the worker's WASM (conversion)
            const maybeEnableInput = () => {
                if (!runtimeReady || !workerReady) return;
                const input = document.getElementById("file-input");
                if (input) {
                    input.disabled = false;
                    input.removeAttribute("title");
                }
                // The "Load from Hugging Face" controls and the nanochat demo
                // button share the same runtimes as the file picker, so enable
                // them together.
                for (const id of ["hf-model-select", "hf-model-input", "hf-load-button", "nanochat-demo-button"]) {
                    const el = document.getElementById(id);
                    if (el) {
                        el.disabled = false;
                        el.removeAttribute("title");
                    }
                }
                const status = document.getElementById("loading-status");
                if (status) status.style.display = "none";
                // Signal the module panels (the single-feature debug passes)
                // that BOTH runtimes are up, so a pass posted to the worker is
                // no longer at risk of being dropped before it starts listening.
                if (!window.__onnxsimReady) {
                    window.__onnxsimReady = true;
                    window.dispatchEvent(new Event("onnxsim:ready"));
                }
            };

            // Publish the runtime (and a promise for it) so the ES-module panels
            // — versions, "parse a text graph", and the single-feature debug
            // passes — can call the module directly. None of them need
            // onnxruntime-web, so this page's runtime is enough for them.
            window.__onnxsimRuntimePromise = create_onnxsim().then((runtime) => {
                window.__onnxsimRuntime = runtime;
                to_ary = (vec) => {
                    return new Array(vec.size()).fill(0).map((_, id) => vec.get(id))
                };
                const passes = document.getElementById("pass-list");
                const checkboxes = {};
                for (const p of to_ary(runtime.onnxoptimizer_passes()).sort()) {
                    const c = document.createElement("input");
                    c.type = "checkbox";
                    c.setAttribute("class", "pass");
                    c.name = p;
                    c.id = "id_" + p;
                    checkboxes[p] = c;
                    passes.appendChild(c);
                    const l = document.createElement("label");
                    l.textContent = p;
                    l.setAttribute("for", c.id);
                    passes.appendChild(l);
                    passes.appendChild(document.createElement("br"));
                }
                for (const p of to_ary(runtime.onnxoptimizer_fuse_elimination_passes()).sort()) {
                    checkboxes[p].checked = true;
                }
                runtimeReady = true;
                maybeEnableInput();
                return runtime;
            });

            // The most recent conversion parameters, captured so that the
            // "Report an issue" button can include them in the pre-filled report.
            // Published on `window` so the report-issue module (a separate inline
            // module script) can read it — modules don't share this classic
            // script's scope.
            let last_run_info = null;

            window.onload = () => {
                const worker = new Worker(window.__onnxsimWorkerUrl || "worker.js");
                const log_output = document.getElementById("log-output");
                const input = document.getElementById("file-input");
                const dl_btn = document.getElementById("download-button");
                const format_select = document.getElementById("download-format");
                const format_status = document.getElementById("download-format-status");
                const trace_container = document.getElementById("simplify-trace");
                const node_reduction_container = document.getElementById("simplify-node-reduction");
                let result_name = "";
                let original_name = "";
                // Decode a "data:...;base64,<b64>" URL back into a Uint8Array.
                const dataUrlToBytes = (data_url) => {
                    const b64 = data_url.slice(data_url.indexOf(",") + 1);
                    const bin = atob(b64);
                    const bytes = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                    return bytes;
                };
                // The in-flight window.__onnxsimImportArchive() request (see
                // below), resolved/rejected by the "import-format-done"/
                // "import-format-error" cases below. Only one archive decode
                // is ever in flight at a time -- a later request implicitly
                // supersedes an earlier one, which matches the file picker it
                // serves (picking a new file makes any previous one moot).
                let pendingImport = null;

                worker.onmessage = (e) => {
                    switch (e.data[0]) {
                        case "import-format-done":
                            // e.data[2] is the decoded model's raw bytes (an
                            // ArrayBuffer, transferred -- see worker.js).
                            if (pendingImport) {
                                pendingImport.resolve(new Uint8Array(e.data[2]));
                                pendingImport = null;
                            }
                            break;
                        case "import-format-error":
                            if (pendingImport) {
                                pendingImport.reject(new Error(e.data[2]));
                                pendingImport = null;
                            }
                            break;
                        case "ready":
                            // The worker's WASM runtime has finished loading and
                            // is now listening for conversion requests.
                            workerReady = true;
                            maybeEnableInput();
                            break;
                        case "stdout":
                            log_output.value += e.data[1] + "\n";
                            log_output.scrollTop = log_output.scrollHeight;
                            break;
                        case "stderr":
                            log_output.value += e.data[1] + "\n";
                            log_output.scrollTop = log_output.scrollHeight;
                            break;
                        // The download-format selector asked for the already-
                        // converted model re-exported as a standalone
                        // safetensors/gguf archive (see dl_btn.onclick below);
                        // this is that export's result, so trigger the download
                        // now and re-enable the controls it disabled meanwhile.
                        case "export-format-done": {
                            dl_btn.disabled = false;
                            format_select.disabled = false;
                            format_status.textContent = "";
                            // e.data[2] is the raw archive bytes (an
                            // ArrayBuffer, transferred rather than a base64
                            // data URL -- see worker.js). downloadBytes wraps
                            // it in a Blob URL, which is O(1) to create/click
                            // regardless of size, unlike a multi-ten-MB data:
                            // URL.
                            import("./download.mjs").then(({ downloadBytes }) => {
                                downloadBytes(e.data[2], e.data[3]);
                            });
                            break;
                        }
                        case "export-format-error":
                            dl_btn.disabled = false;
                            format_select.disabled = false;
                            format_status.textContent = `${e.data[1]} export failed: ${e.data[2]}`;
                            break;
                        case "convert-done":
                            input.disabled = false;
                            dl_btn.disabled = false;
                            format_status.textContent = "";
                            const data_url = e.data[1];
                            const trace_json = e.data[2];
                            const original_data_url = e.data[3];
                            dl_btn.onclick = () => {
                                const format = format_select.value;
                                if (format === "onnx") {
                                    const a = document.createElement("a");
                                    a.href = data_url;
                                    a.download = result_name;
                                    a.click();
                                    return;
                                }
                                // Safetensors / GGUF: ask the worker to re-export
                                // the already-converted bytes into that standalone
                                // archive format (it owns the WASM runtime that
                                // does the export). Async, so disable the controls
                                // until "export-format-done"/"-error" above fires.
                                const converted = window.__onnxsimConverted;
                                if (!converted || !converted.bytes) return;
                                const ext = format === "gguf" ? ".gguf" : ".safetensors";
                                const bytes = converted.bytes;
                                const export_buf = bytes.buffer.slice(
                                    bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
                                dl_btn.disabled = true;
                                format_select.disabled = true;
                                format_status.textContent = `exporting ${format}…`;
                                worker.postMessage(
                                    ["export_" + format, export_buf, result_name + ext],
                                    [export_buf]);
                            };
                            // Visualize the simplified/optimized model with Netron.
                            if (window.netronShowAfter) {
                                window.netronShowAfter(data_url, result_name);
                            }
                            // List the converted model's dim_params ("after").
                            if (window.dimParamsShowAfter) {
                                window.dimParamsShowAfter(data_url, result_name);
                            }
                            // Keep the converted model bytes around so the
                            // "Run inference" panel can run them (not just the
                            // original upload). Decode the base64 data URL back
                            // into a Uint8Array and publish it for the panel.
                            try {
                                const bytes = dataUrlToBytes(data_url);
                                window.__onnxsimConverted = { bytes, name: result_name };
                                window.dispatchEvent(new CustomEvent("onnxsim:converted", {
                                    detail: { name: result_name },
                                }));
                            } catch (err) {
                                log_output.value += "failed to keep converted model for inference: " + err + "\n";
                                log_output.scrollTop = log_output.scrollHeight;
                            }
                            // Keep the MAC-annotated *original* bytes (when the
                            // worker produced them) so the "Run inference" panel's
                            // "original" source can report MACs/throughput and be
                            // compared against the converted result.
                            if (original_data_url) {
                                try {
                                    const bytes = dataUrlToBytes(original_data_url);
                                    window.__onnxsimOriginalAnnotated = { bytes, name: original_name };
                                    window.dispatchEvent(new CustomEvent("onnxsim:original-annotated", {
                                        detail: { name: original_name },
                                    }));
                                } catch (err) {
                                    log_output.value += "failed to keep annotated original for inference: " + err + "\n";
                                    log_output.scrollTop = log_output.scrollHeight;
                                }
                            }
                            // Render the onnxsim simplification profile, if one
                            // came back, as an inline flame graph.
                            if (trace_container) trace_container.innerHTML = "";
                            if (node_reduction_container) node_reduction_container.innerHTML = "";
                            if (trace_json && trace_container) {
                                try {
                                    const trace = JSON.parse(trace_json);
                                    import("./trace_viewer.mjs").then(({ renderTrace }) => {
                                        renderTrace(trace_container, trace, {
                                            title: "onnxsim simplify",
                                            filename: "onnxsim.simplify.trace.json",
                                        });
                                    });
                                    if (node_reduction_container) {
                                        import("./node_reduction_view.mjs").then(({ renderNodeReduction }) => {
                                            renderNodeReduction(node_reduction_container, trace);
                                        });
                                    }
                                } catch (err) {
                                    log_output.value += "failed to parse profiling trace: " + err + "\n";
                                    log_output.scrollTop = log_output.scrollHeight;
                                }
                            }
                            break;
                    }
                };

                // Run one conversion. `modelName` names the source (for the
                // download filename and issue report); `buf` is an ArrayBuffer
                // that is *transferred* to the worker. Shared by the file picker
                // and the Hugging Face loader (which calls it via the window
                // handle published below).
                const startConversion = (modelName, buf) => {
                    const optimizer = document.querySelector('input[name="optimizer"]:checked').value;
                    // A ".onnx.safetensors"/".onnx.gguf" upload (the same standalone
                    // archive format the download-format selector produces) needs
                    // decoding back to ONNX before conversion -- see worker.js's
                    // import pre-step. Strip whichever archive suffix is present
                    // first, then the ".onnx" underneath it, so the result name
                    // matches a plain ".onnx" upload's either way.
                    let base = modelName;
                    let source_format = "onnx";
                    if (/\.safetensors$/i.test(base)) {
                        source_format = "safetensors";
                        base = base.slice(0, -".safetensors".length);
                    } else if (/\.gguf$/i.test(base)) {
                        source_format = "gguf";
                        base = base.slice(0, -".gguf".length);
                    }
                    if (/\.onnx$/i.test(base)) {
                        base = base.slice(0, -".onnx".length);
                    }
                    result_name = base + "." + optimizer + ".onnx";
                    original_name = modelName;
                    // Drop any previous run's cached models so the inference panel
                    // never serves a stale converted/annotated result for the new
                    // model while this conversion is still in flight.
                    window.__onnxsimConverted = null;
                    window.__onnxsimOriginalAnnotated = null;

                    input.disabled = true;
                    dl_btn.disabled = true;
                    format_status.textContent = "";
                    let passes = null;
                    if (optimizer == "simplify") {
                        passes = Array.from(document.querySelectorAll('input[class="pass"]:not(:checked)')).map((v) => v.name);
                    } else {
                        passes = Array.from(document.querySelectorAll('input[class="pass"]:checked')).map((v) => v.name);
                    }
                    const constant_fold = document.getElementById("id_simplify_constant_fold").checked;
                    const shape_inference = document.getElementById("id_simplify_shape_inference").checked;
                    const tensor_size_threshold = parseInt(document.getElementById("id_simplify_tensor_size_threshold").value);
                    // An empty / non-positive value means "keep the current opset version".
                    const target_opset_version = parseInt(document.getElementById("id_simplify_target_opset").value) || 0;
                    last_run_info = {
                        optimizer, passes, constant_fold, shape_inference,
                        tensor_size_threshold, target_opset_version,
                        file_name: modelName,
                    };
                    // Profiling only applies to "simplify"; the onnx-optimizer
                    // paths do not run the fixed-point profiler.
                    const profile = optimizer == "simplify" && document.getElementById("id_simplify_profile").checked;
                    // Annotate the converted model with MACs/FLOPs (metadata_props)
                    // by default; applies to every optimizer mode.
                    const annotate_model_info = document.getElementById("id_annotate_model_info").checked;
                    last_run_info.annotate_model_info = annotate_model_info;
                    // Inline the model's local functions before simplify / optimize /
                    // fixed optimize (a no-op for the single-pass debug modes and the
                    // standalone "inline" mode). Off by default so behaviour is
                    // unchanged unless the user opts in.
                    const inline_functions = document.getElementById("id_inline_functions").checked;
                    last_run_info.inline_functions = inline_functions;
                    // Print a node/value-level diff (which nodes/values were removed,
                    // added, or changed) to the log alongside the op-count summary.
                    // Only meaningful for "simplify", where onnxsim actually changes
                    // the graph. Off by default: it can be long for a big model.
                    const graph_diff = optimizer == "simplify" && document.getElementById("id_graph_diff").checked;
                    last_run_info.graph_diff = graph_diff;
                    // Expose the latest run parameters to the report-issue module.
                    window.__onnxsimLastRunInfo = last_run_info;
                    worker.postMessage([optimizer, buf, passes, constant_fold, shape_inference, tensor_size_threshold, target_opset_version, profile, annotate_model_info, inline_functions, graph_diff, source_format], [buf]);
                };
                // Expose the conversion entry point for the Hugging Face loader
                // module (hf_load.mjs), which downloads model bytes and drives
                // the same path as an uploaded file.
                window.__onnxsimStartConversion = startConversion;

                // Decode a standalone safetensors/gguf archive (ArrayBuffer) into
                // plain ONNX model bytes, without running a conversion. Used by
                // netron_view.mjs's "before" pane so a safetensors/gguf upload
                // renders the actual model graph instead of the raw archive
                // bytes (which Netron can only show as an opaque weights list).
                // `format` is "safetensors" or "gguf"; `buf` is transferred to
                // the worker, so the caller must not reuse it afterwards.
                window.__onnxsimImportArchive = (buf, format) => {
                    return new Promise((resolve, reject) => {
                        pendingImport = { resolve, reject };
                        worker.postMessage(["import_" + format, buf], [buf]);
                    });
                };

                input.addEventListener("change", async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    // An uploaded file is readable as "original" from the file
                    // input itself, so clear any model left over from a previous
                    // Hugging Face load to avoid running stale bytes.
                    window.__onnxsimOriginal = null;
                    const buf = await file.arrayBuffer();
                    startConversion(file.name, buf);
                });
            };
