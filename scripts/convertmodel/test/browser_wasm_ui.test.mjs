#!/usr/bin/env node

import assert from "node:assert/strict";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const testDir = resolve(fileURLToPath(new URL(".", import.meta.url)));
const appRoot = resolve(process.argv[2] || join(testDir, ".."));
const modelPath = resolve(process.argv[3] || join(testDir, "model.onnx"));
const timeout = 180_000;

assert(existsSync(join(appRoot, "index.html")), `converter page not found: ${appRoot}`);
assert(existsSync(modelPath), `test model not found: ${modelPath}`);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
};

const server = createServer((req, res) => {
  const requestPath = decodeURIComponent((req.url || "/").split("?", 1)[0]);
  const relativePath = requestPath === "/" ? "index.html" : requestPath.replace(/^\/+/, "");
  const filePath = resolve(join(appRoot, normalize(relativePath)));
  if (!filePath.startsWith(`${appRoot}/`) || !existsSync(filePath) || !statSync(filePath).isFile()) {
    res.writeHead(404);
    res.end("not found");
    return;
  }
  res.writeHead(200, {
    "Content-Type": contentTypes[extname(filePath)] || "application/octet-stream",
    "Cache-Control": "no-store",
  });
  createReadStream(filePath).pipe(res);
});

const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
  const address = server.address();
  browser = await chromium.launch({
    headless: true,
    args: ["--enable-features=WebAssemblyUnlimitedSyncCompilation"],
  });
  page = await browser.newPage();
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.stack || error}`));
  page.on("console", (message) => {
    // The converter intentionally has optional same-origin resources (for
    // example models.json on a PR validation build), and browsers report a
    // missing optional resource as a generic console error. JavaScript
    // exceptions are still captured by pageerror below; ignore only this
    // unhelpful network diagnostic so the test checks the actual UI flow.
    if (message.type() === "error" && !message.text().includes("Failed to load resource")) {
      browserErrors.push(`console: ${message.text()}`);
    }
  });

  await page.goto(`http://127.0.0.1:${address.port}/`, {
    waitUntil: "domcontentloaded",
    timeout,
  });
  await page.waitForFunction(
    () => !document.querySelector("#file-input")?.disabled,
    undefined,
    { timeout },
  );

  await page.locator("#file-input").setInputFiles(modelPath);
  await page.waitForFunction(
    () => Boolean(window.__onnxsimConverted?.bytes),
    undefined,
    { timeout },
  );
  await page.waitForFunction(
    () => !document.querySelector("#download-button")?.disabled,
    undefined,
    { timeout },
  );

  await page.locator("#inference-ep").selectOption("wasm");
  await page.locator("#inference-iters").fill("2");
  await page.locator("#inference-warmup").fill("0");
  await page.locator("#inference-profile").uncheck();
  await page.locator("#run-inference").click();
  await page.waitForFunction(
    () => document.querySelector("#inference-output")?.value.includes("PASS: ran both models"),
    undefined,
    { timeout },
  );

  const conversionLog = await page.locator("#log-output").inputValue();
  const inferenceLog = await page.locator("#inference-output").inputValue();
  assert.match(conversionLog, /simplif|convert/i, "conversion log did not report a conversion");
  assert.doesNotMatch(inferenceLog, /FAIL:/, "inference reported failure");
  assert.equal(browserErrors.length, 0, browserErrors.join("\n"));
  console.log("PASS: WASM converter UI converted a model and ran WASM inference");
} catch (error) {
  if (page) {
    const screenshot = process.env.PLAYWRIGHT_SCREENSHOT || "/tmp/wasm-ui-failure.png";
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
    console.error(`screenshot: ${screenshot}`);
  }
  throw error;
} finally {
  await browser?.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}
