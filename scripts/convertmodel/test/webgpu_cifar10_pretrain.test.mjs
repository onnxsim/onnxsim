// A small, real CIFAR-10 pretraining pipeline: fetch a batch of real, labeled
// photos from Hugging Face's uoft-cs/cifar10, train onnxsim's own
// graph_grad/qat_graph step-graph machinery to fit that fixed batch across
// many steps, and require every step to actually run on WebGPU. Same shape
// of check as webgpu_hf_demo.test.mjs (see that file's own top comment for
// why a real browser via Playwright is what makes WebGPU reachable at all
// here, and why step_graph_ep.test.mjs's own webgpu row is "attempted and
// reported" rather than required) -- the difference is what's being trained:
// a real, ten-class *classification* task, not an arbitrary reconstruction
// target.
//
// What this proves, and what it does not:
//   * step_qat_cifar10_pretrain.onnx (test/make_step_graph_fixtures.py's
//     build_qat_cifar10_pretrain) is a real batched onnxsim step graph --
//     forward, backward, and Adam update -- built entirely offline with no
//     network access, exactly like the other fixtures in this directory. Its
//     "x"/"teacher" constants are declared with no baked values (see that
//     function's own docstring for why), which is what lets this file feed
//     them eight real, labeled CIFAR-10 photos instead.
//   * The batch comes from hf_datasets.mjs's fetchCifar10Batch(), fetched
//     once and then trained on repeatedly (unlike webgpu_hf_demo.test.mjs's
//     own fetchSampleImageBytes(), which feeds a fresh photo every run) --
//     "pretraining on a sample" means a small, fixed, real, labeled sample,
//     not a different one each step.
//   * This is real classification: "teacher" is a one-hot row per photo, and
//     this file checks accuracy (argmax(y) against the true label) at the
//     end of the run, not just that the loss number went down.
//   * fetchCifar10Batch's exact response shape (which of `img`/`image` the
//     dataset-server names its column, whether plain_text/train is right)
//     has not been checked against a live response -- this dev sandbox's
//     outbound network is proxy-blocked to huggingface.co (confirmed via
//     curl and a live Playwright page). hf_datasets.mjs checks both column
//     names defensively, but the first real confirmation either way is this
//     file's own CI run.
//
// Requires the "playwright" package and a Chromium binary -- see
// package.json's test:webgpu-cifar10 comment and
// .github/workflows/convertmodel-webgpu-demo.yml, the only place this runs.
// Not part of test:all, for the same reasons webgpu_hf_demo.test.mjs isn't.
//
// Usage:
//   npx playwright install chromium   # once
//   node test/webgpu_cifar10_pretrain.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "step_qat_cifar10_pretrain.json"), "utf8"));
const NUM_STEPS = MANIFEST.scalars.length;

// Same reasoning as webgpu_hf_demo.test.mjs's own TRAINABLE_OPS.
const TRAINABLE_OPS = new Set([
  "MatMul", "Add", "Sub", "Mul", "Div", "Sqrt",
  "Sigmoid", "Reshape", "Transpose", "ReduceSum", "ReduceMean",
]);

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Identical to webgpu_hf_demo.test.mjs's own server -- duplicated rather
// than shared, matching this repo's established preference for independent
// test files over a shared test-infra module (see e.g. lora_entry.cpp's own
// top comment on why SliceBlock is copied rather than shared with
// qat_entry.cpp).
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu cifar10 pretrain</title>");
      return;
    }
    const filePath = join(ROOT, reqPath);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".json": "application/json", ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
      }[path.extname(filePath)] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, () => resolve(server));
  });
}

// Runs inside the real browser page -- see page.evaluate.
async function runInPage(port) {
  const base = `http://localhost:${port}`;
  const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

  const { fetchCifar10Batch } = await import(`${base}/hf_datasets.mjs`);
  const { normalizePixels } = await import(`${base}/sample_inputs.mjs`);
  const manifest = await fetch(`${base}/test/step_qat_cifar10_pretrain.json`).then((r) => r.json());
  const modelBytes = await fetch(`${base}/test/step_qat_cifar10_pretrain.onnx`).then((r) => r.arrayBuffer());

  // A real, fixed batch of labeled photos -- fetched once, trained on for
  // the whole run (see this file's own top comment on why that's the point).
  const samples = await fetchCifar10Batch(manifest.numSamples);
  const side = Math.round(Math.sqrt(manifest.inputDim));
  const x = new Float32Array(manifest.numSamples * manifest.inputDim);
  const teacher = new Float32Array(manifest.numSamples * manifest.numClasses);
  const labels = [];
  for (let i = 0; i < samples.length; i++) {
    const blob = new Blob([samples[i].bytes]);
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(side, side);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, side, side);
    const rgba = ctx.getImageData(0, 0, side, side).data;
    const flat = normalizePixels(rgba, 1, side, side);
    x.set(flat, i * manifest.inputDim);
    teacher[i * manifest.numClasses + samples[i].label] = 1.0;
    labels.push(samples[i].label);
  }

  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "disabled",
    logSeverityLevel: 0,
    logVerbosityLevel: 4,
  });

  const constants = {
    x: new ort.Tensor("float32", x, [manifest.numSamples, manifest.inputDim]),
    teacher: new ort.Tensor("float32", teacher, [manifest.numSamples, manifest.numClasses]),
  };
  let state = {};
  for (const [name, spec] of Object.entries(manifest.state)) {
    state[name] = new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);
  }

  const t0 = performance.now();
  const losses = [];
  let lastY = null;
  for (let t = 0; t < manifest.scalars.length; t++) {
    const feeds = { ...constants, ...state };
    for (const [name, value] of Object.entries(manifest.scalars[t])) {
      feeds[name] = new ort.Tensor("float32", Float32Array.from([value]), []);
    }
    const out = await session.run(feeds);
    const next = {};
    for (const [name, spec] of Object.entries(manifest.state)) {
      next[name] = out[spec.output];
    }
    state = next;
    losses.push(Number(out[manifest.loss].data[0]));
    lastY = out[manifest.outputName].data; // overwritten each step; final value used below
  }
  const trainMs = performance.now() - t0;
  await session.release?.();

  // Predicted class per sample from the final step's own output -- argmax
  // over each row of [numSamples, numClasses].
  const predictions = [];
  for (let i = 0; i < manifest.numSamples; i++) {
    let best = 0;
    for (let c = 1; c < manifest.numClasses; c++) {
      if (lastY[i * manifest.numClasses + c] > lastY[i * manifest.numClasses + best]) best = c;
    }
    predictions.push(best);
  }
  const correct = predictions.filter((p, i) => p === labels[i]).length;

  return {
    losses, trainMs, correct,
    total: manifest.numSamples,
    labelNames: samples.map((s) => s.labelName),
  };
}

// Same as webgpu_hf_demo.test.mjs's own parsePlacements.
function parsePlacements(lines) {
  const placements = {};
  let current = null;
  for (const raw of lines) {
    const line = raw.replace(/^.*VerifyEachNodeIsAssignedToAnEp\]\s?/, "");
    const header = line.match(/(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/);
    if (header) {
      const all = header[1] === "All nodes";
      current = placements[header[2]] || { count: 0, ops: all ? null : new Set() };
      current.count += Number(header[3]);
      if (all) current.ops = null;
      placements[header[2]] = current;
      continue;
    }
    const node = current && current.ops && line.match(/^\s*([A-Za-z][\w.]*)\s*\(/);
    if (node) current.ops.add(node[1]);
  }
  return placements;
}

async function main() {
  console.log(
    `onnxsim WebGPU + CIFAR-10 pretraining demo -- ${NUM_STEPS} steps, ` +
      `${MANIFEST.numSamples} photos, input dim ${MANIFEST.inputDim}, ` +
      `${MANIFEST.numClasses} classes\n`,
  );

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-unsafe-webgpu"], // see webgpu_hf_demo.test.mjs's own comment
  });
  const logLines = [];
  let result;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (/onnxruntime/i.test(text)) logLines.push(text);
    });
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, port);
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`  sample: ${result.labelNames.join(", ")}`);
  const trace = `loss ${result.losses[0].toExponential(3)} -> ` +
    `${result.losses[result.losses.length - 1].toExponential(3)}`;
  console.log(
    `  ${NUM_STEPS} steps in ${result.trainMs.toFixed(0)}ms, ${trace}, ` +
      `${result.correct}/${result.total} correct at the end`,
  );

  await check("every loss is finite", () => {
    for (const loss of result.losses) {
      assert.ok(Number.isFinite(loss), `loss ${loss} is not finite`);
    }
  });
  await check("loss decreased meaningfully over the run", () => {
    const first = result.losses[0];
    const last = result.losses[result.losses.length - 1];
    assert.ok(
      last < 0.4 * first,
      `loss did not meaningfully decrease: ${first} -> ${last}`,
    );
  });
  await check("the network learned to classify most of the pretraining sample", () => {
    assert.ok(
      result.correct >= Math.ceil(0.75 * result.total),
      `only ${result.correct}/${result.total} correct after ${NUM_STEPS} steps ` +
        `on a fixed batch this small`,
    );
  });

  const placements = parsePlacements(logLines);
  await check("every trainable op ran on WebGPU (JsExecutionProvider)", () => {
    const gpu = placements.JsExecutionProvider;
    assert.ok(gpu, `no JsExecutionProvider placement reported at all -- captured ${logLines.length} log lines`);
    const gpuOps = gpu.ops === null ? new Set(TRAINABLE_OPS) : gpu.ops;
    const missing = [...TRAINABLE_OPS].filter((op) => !gpuOps.has(op));
    assert.deepEqual(
      missing, [],
      `these ops never showed up under JsExecutionProvider: ${missing.join(", ")} -- ` +
        `full placement report: ${JSON.stringify(placements, (k, v) => (v instanceof Set ? [...v] : v))}`,
    );
    for (const [provider, info] of Object.entries(placements)) {
      if (provider === "JsExecutionProvider") continue;
      const strayed = info.ops && [...info.ops].filter((op) => TRAINABLE_OPS.has(op));
      assert.ok(
        !strayed || strayed.length === 0,
        `${strayed && strayed.join(", ")} ran on ${provider} instead of WebGPU`,
      );
    }
  });

  console.log(`\nwebgpu CIFAR-10 pretraining demo: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
