// A small, real, end-to-end training pipeline: fetch one real photo from a
// Hugging Face dataset, train onnxsim/qat_graph's own step-graph machinery on
// it for a couple dozen steps, and require every step to actually run on
// WebGPU -- not wasm, not CPU, and not "attempted and reported" the way
// step_graph_ep.test.mjs treats webgpu (that test runs under plain Node,
// which has no navigator.gpu at all; this one drives a real browser via
// Playwright specifically so WebGPU is reachable).
//
// What this proves, and what it does not:
//   * step_qat_hf_demo.onnx (test/make_step_graph_fixtures.py's
//     build_qat_hf_demo) is a real onnxsim.graph_grad/qat_graph step graph --
//     forward, backward, and Adam update -- built entirely offline, with no
//     network access, exactly like the other four fixtures in this
//     directory. It declares "x"/"teacher" as ordinary graph inputs with no
//     baked values (see that function's own docstring for why), which is
//     what lets this file feed them something the generator script could
//     never reach: a live Hugging Face photo.
//   * The photo comes from hf_datasets.mjs's fetchSampleImageBytes(), the
//     same already-tested function the "Run inference" panel's sample-data
//     fill mode uses -- nothing new is invented for the network/decode path.
//   * "Training" here means fitting one real (downsized, grayscale) photo
//     toward a fixed zero target -- an arbitrary objective chosen because it
//     needs no labels, only a real x (see build_qat_hf_demo's docstring).
//     The point is not what it learns; it's that a real photo, a real
//     gradient/Adam step graph, and WebGPU all work together, end to end,
//     inside the time this CI job's own timeout gives it.
//
// Requires the "playwright" package and a Chromium binary (not installed by
// default here -- see package.json's test:webgpu-hf-demo comment and
// .github/workflows/convertmodel-webgpu-demo.yml, the only place this is
// run). Not part of test:all: every other test here runs in plain Node with
// no browser and no live network, and this one needs both plus a machine
// WebGPU can actually reach.
//
// Usage:
//   npx playwright install chromium   # once
//   node test/webgpu_hf_demo.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "step_qat_hf_demo.json"), "utf8"));

// How many of the fixture's baked steps to actually run. The manifest has
// 20 (see make_step_graph_fixtures.py's build_qat_hf_demo); running all of
// them keeps this close to what the offline sanity check itself measured.
const NUM_STEPS = MANIFEST.scalars.length;

// Ops the forward/backward/Adam-update graph is built from (see
// build_qat_hf_demo). A node landing anywhere other than JsExecutionProvider
// (WebGPU) for one of these is the finding this file exists to catch --
// Identity/MemcpyFromHost/MemcpyToHost are graph-boundary bookkeeping ORT is
// free to keep on CPU regardless of EP, the same "not a coverage gap"
// reasoning onnxsim/qat_graph.py already gives Identity elsewhere.
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

// A minimal static file server over scripts/convertmodel/ -- just enough for
// the page's dynamic import()/fetch() calls to resolve relative URLs. `/`
// itself returns a blank 200 (not a 404) purely to keep the console quiet;
// nothing reads its body.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu hf demo</title>");
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

// Everything below runs inside the real browser page, via page.evaluate --
// it has no access to this file's own scope, only what it's passed and what
// it can import()/fetch() from the server above. The actual "fetch a real
// photo, run the step loop on WebGPU" logic lives in webgpu_hf_demo.mjs,
// shared with the interactive panel (webgpu_demo_ui.mjs) -- this function is
// just that module's Node/Playwright-side caller: load the local
// node_modules onnxruntime-web bundle (the interactive page instead loads
// from a CDN; see webgpu_hf_demo.mjs's own top comment for why the shared
// module takes an already-loaded `ort` rather than caring), fetch the fixture
// bytes from this test's own throwaway static server, and time the run.
async function runInPage(port) {
  const base = `http://localhost:${port}`;
  const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

  const { runPhotoDemo } = await import(`${base}/webgpu_hf_demo.mjs`);
  const manifest = await fetch(`${base}/test/step_qat_hf_demo.json`).then((r) => r.json());
  const modelBytes = await fetch(`${base}/test/step_qat_hf_demo.onnx`).then((r) => r.arrayBuffer());

  const t0 = performance.now();
  const { losses, imageLabel } = await runPhotoDemo({ ort, modelBytes, manifest });
  const trainMs = performance.now() - t0;

  return { losses, trainMs, imageLabel };
}

// The same "Node(s) placed on [Provider]" parser step_graph_ep.test.mjs uses,
// duplicated rather than imported: that file's copy is scoped to Node's
// fs.writeSync log-interception trick, which has no counterpart (and no
// need) in a real browser, where these lines simply arrive as ordinary
// console messages Playwright's page.on("console") already sees.
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
    `onnxsim WebGPU + Hugging Face training demo -- ${NUM_STEPS} steps, ` +
      `input dim ${MANIFEST.inputDim}\n`,
  );

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // The one flag WebGPU needs while it is still an experimental Chromium
    // feature; verified (in this repo's own dev sandbox, via SwiftShader's
    // software Vulkan path) to be sufficient on its own -- no --use-gl,
    // --use-angle, or --enable-features flags needed on top of it. A real
    // GPU (as the macOS CI runner this is meant for has) needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });
  const logLines = [];
  let result;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (/onnxruntime/i.test(text)) logLines.push(text);
    });
    // A real page, not about:blank -- WebGPU's adapter is unreachable from
    // the latter in this browser (checked directly; unclear why, and not
    // worth chasing further since every real use, including the interactive
    // converter page itself, is already served over http/https).
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, port);
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`  sample photo: ${result.imageLabel} (uoft-cs/cifar10)`);
  const trace = `loss ${result.losses[0].toExponential(3)} -> ` +
    `${result.losses[result.losses.length - 1].toExponential(3)}`;
  console.log(`  ${NUM_STEPS} steps in ${result.trainMs.toFixed(0)}ms, ${trace}`);

  await check("every loss is finite", () => {
    for (const loss of result.losses) {
      assert.ok(Number.isFinite(loss), `loss ${loss} is not finite`);
    }
  });
  await check("the loop moved: step 1 differs from step 0", () => {
    const [first, second] = result.losses;
    assert.ok(
      Math.abs(second - first) > 1e-6 * Math.abs(first || 1),
      `the second step did not move the loss (${first} -> ${second})`,
    );
  });
  await check("loss decreased meaningfully over the run", () => {
    const first = result.losses[0];
    const last = result.losses[result.losses.length - 1];
    assert.ok(
      last < 0.8 * first,
      `loss did not meaningfully decrease: ${first} -> ${last}`,
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

  console.log(`\nwebgpu HF demo: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
