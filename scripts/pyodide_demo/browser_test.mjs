#!/usr/bin/env node

import assert from "node:assert/strict";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const demoRoot = resolve(process.argv[2] || fileURLToPath(new URL(".", import.meta.url)));
const modelPath = resolve(
  process.argv[3] || join(demoRoot, "model.onnx"),
);
const port = Number(process.env.PORT || 0);
const timeout = 300_000;

assert(existsSync(join(demoRoot, "index.html")), `demo root has no index.html: ${demoRoot}`);
assert(existsSync(modelPath), `test model not found: ${modelPath}`);

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
  ".so": "application/octet-stream",
  ".onnx": "application/octet-stream",
};

const server = createServer((req, res) => {
  const requestPath = decodeURIComponent((req.url || "/").split("?", 1)[0]);
  const relativePath = requestPath === "/" ? "index.html" : requestPath.replace(/^\/+/, "");
  const filePath = resolve(join(demoRoot, normalize(relativePath)));
  if (!filePath.startsWith(`${demoRoot}/`) || !existsSync(filePath) || !statSync(filePath).isFile()) {
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
  await new Promise((resolveListen) => server.listen(port, "127.0.0.1", resolveListen));
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/`;
  // Pyodide's side module is larger than Chromium's default 8 MiB limit for
  // synchronous WebAssembly compilation. The page's import path is
  // intentionally synchronous from Python, so enable the Chromium feature
  // that Pyodide itself recommends for large dynamic modules.
  browser = await chromium.launch({
    headless: true,
    args: ["--enable-features=WebAssemblyUnlimitedSyncCompilation"],
  });
  page = await browser.newPage();
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.stack || error}`));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });

  await page.goto(url, { waitUntil: "domcontentloaded", timeout });
  await page.locator("#load-btn").click();
  await page.locator("#load-badge .badge").waitFor({ state: "visible", timeout });
  await page.locator("#run-panel").waitFor({ state: "visible", timeout });
  await page.locator("#optimizers-panel").waitFor({ state: "visible", timeout });
  assert(
    Number(await page.locator("#opt-count").textContent()) > 0,
    "the loaded extension registered no optimizer passes",
  );

  await page.locator("#file-input").setInputFiles(modelPath);
  await page.waitForFunction(
    () => !document.querySelector("#run-btn")?.disabled,
    undefined,
    { timeout },
  );
  await page.locator("#run-btn").click();
  await page.locator("#run-output a.download-link").waitFor({ state: "visible", timeout });

  const output = await page.locator("#run-output").textContent();
  assert.match(output, /onnxsim\.simplify\(\)/, "full simplify pass did not render a result");
  assert.equal(browserErrors.length, 0, browserErrors.join("\n"));
  console.log("PASS: Pyodide demo loaded, imported onnxsim, uploaded a model, and ran simplify()");
} catch (error) {
  if (page) {
    const screenshot = process.env.PLAYWRIGHT_SCREENSHOT || "/tmp/pyodide-demo-failure.png";
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
    console.error(`screenshot: ${screenshot}`);
  }
  throw error;
} finally {
  await browser?.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}
