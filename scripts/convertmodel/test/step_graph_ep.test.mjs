// Does a QAT step graph actually run on the execution providers its operator
// allowlist claims coverage for?
//
// onnxsim/qat_graph.py pins the operators a step graph may contain
// (EP_FRIENDLY_OPS) and justifies the set with a claim about backends:
// "check the operator actually has coverage on the WebGPU and WebNN backends
// first, not just on ORT's CPU kernels". Three python test files assert that
// the builders stay inside the set. Nothing asserted the set is *right* --
// docs/qat.md's own risk list says op coverage "needs measuring per EP before
// any claim is made in the README", and the README now makes claims.
//
// This test measures it, on the real graphs the library emits (fixtures built
// by make_step_graph_fixtures.py from the builders themselves). For each
// execution provider onnxruntime-web can reach here it creates a session, runs
// the step loop -- feeding each step's state outputs back in as the next
// step's state inputs, which is what qat_graph.run_step_graph does -- and
// checks the loss is finite, moves, and tracks the trajectory onnxruntime's
// CPU provider produced for the same feeds in Python.
//
// What it can and cannot conclude, stated up front because the difference is
// the whole value of the test:
//
//   * `wasm` is REQUIRED. If a step graph cannot be created or run there, this
//     test fails.
//   * `webgpu` and the WebNN device types are ATTEMPTED and reported. Headless
//     Node has no `navigator.gpu` and no `navigator.ml`, so onnxruntime-web
//     refuses those backends outright; the run is skipped with the reason it
//     gave, never counted as a pass. Verifying them needs a browser (the
//     converter page's own inference panel is the place), or a machine where
//     those APIs exist, where `ORT_REQUIRE` below turns the skip into a
//     hard requirement.
//
// Usage:
//   node test/step_graph_ep.test.mjs
//   ORT_REQUIRE=wasm,webgpu node test/step_graph_ep.test.mjs   # fail if webgpu
//                                                             # is unreachable
//   ORT_VERBOSE=1 node test/step_graph_ep.test.mjs             # dump the raw
//                                                             # ORT log lines

import fs, { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";
import { captureOrtDiagnostics, isOrtDiagnostic } from "../ort_log_capture.mjs";
import {
  WEBNN_DEVICE_TYPES,
  webnnProvider,
  providerLabel,
  detectWebnn,
  formatWebnnStatus,
} from "../webnn.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";

const MANIFEST = JSON.parse(readFileSync(join(HERE, "step_graphs.json"), "utf8"));
// Which EPs must work for this test to pass. Everything else is attempted and
// reported. wasm is the only one a headless container can honestly promise.
const REQUIRED = new Set((process.env.ORT_REQUIRE || "wasm").split(",").filter(Boolean));
const VERBOSE = !!process.env.ORT_VERBOSE;

// onnxruntime-web's CPU kernels and onnxruntime's are the same C++ code, so the
// two trajectories agree to float32 round-off rather than merely being close:
// the largest relative gap measured across the four fixtures is 1.1e-5 (in
// adaquant, whose composed round-half-away-from-zero can pick a different
// integer for a value within an ulp of a .5 boundary). The tolerance leaves
// room for a differently ordered SIMD reduction on another host while staying
// far below what a wrong kernel costs -- an operator computing the wrong thing
// moves a loss by O(1) relative, not by 1e-3.
const RTOL = 1e-3;
const ATOL = 1e-9;

let passed = 0;
let failures = 0;

// Same shape as the other test files here: run fn, log a tick, count it.
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

function closeEnough(got, want) {
  return Math.abs(got - want) <= ATOL + RTOL * Math.abs(want);
}

// ---------------------------------------------------------------------------
// Capturing onnxruntime's own node-assignment report.
//
// ort_log_capture.mjs mirrors onnxruntime-web's diagnostics out of the places
// they hide (console.warn/error, unhandled rejections) so the browser panel can
// show them; it is reused as-is for that half here. It is not enough on its
// own under Node: the ORT logger's output crosses out of the wasm module
// through emscripten, whose Node path is `fs.writeSync(1|2, ...)` directly --
// not console, and not process.stdout.write either, so neither the panel's
// hooks nor a stream wrapper sees a single line of it. So the same idea is
// applied one layer lower, to fs.writeSync, for the duration of a session.
//
// The onnxruntime-tagged lines are swallowed rather than echoed (the opposite
// of the panel's choice, which preserves devtools output): at
// logSeverityLevel 0 there are hundreds per session, and this test's own
// transcript is what a CI reader needs to see. ORT_VERBOSE prints them back.
function captureOrtLogs() {
  const lines = [];
  const keep = (text) => {
    for (const line of text.split("\n")) {
      if (line && isOrtDiagnostic(line)) lines.push(line);
    }
  };
  const originalWriteSync = fs.writeSync;
  fs.writeSync = function (fd, data, ...rest) {
    if ((fd === 1 || fd === 2) && typeof data === "string" && isOrtDiagnostic(data)) {
      keep(data);
      return Buffer.byteLength(data);
    }
    return originalWriteSync.call(this, fd, data, ...rest);
  };
  const restoreConsole = captureOrtDiagnostics({
    log: (m) => lines.push(m),
    target: null,
  });
  return {
    lines,
    restore() {
      fs.writeSync = originalWriteSync;
      restoreConsole();
    },
  };
}

// Pull the "Node placements" section out of captured verbose logs.
//
// onnxruntime prints one of two shapes once it has partitioned the graph:
//
//   All nodes placed on [CPUExecutionProvider]. Number of nodes: 23
//   Node(s) placed on [JsExecutionProvider]. Number of nodes: 5
//    MatMul (matmul_7)
//
// The first means every node went to one provider; the second is emitted once
// per provider, followed by the individual nodes. Returns
// { provider: { count, ops: Set|null } }, with ops === null meaning "the whole
// graph", which is what the "All nodes" line asserts without listing them.
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
    // A node line under the most recent header: "  <OpType> (<node name>)".
    const node = current && current.ops && line.match(/^\s*([A-Za-z][\w.]*)\s*\(/);
    if (node) current.ops.add(node[1]);
  }
  return placements;
}

// ---------------------------------------------------------------------------
// Feeding a step graph.

function tensor(ort, spec) {
  return new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);
}

function indexTensor(ort, spec) {
  // The minibatch row index is int64 -- the one non-float input any step graph
  // has, and the one an accelerator backend is most likely to reject.
  assert.equal(spec.dtype, "int64", `unexpected per-step dtype ${spec.dtype}`);
  return new ort.Tensor("int64", BigInt64Array.from(spec.data.map(BigInt)), spec.dims);
}

// Run one step graph for the manifest's recorded steps, threading state through
// exactly as qat_graph.run_step_graph's feed-per-step path does. Returns the
// per-step losses.
async function runStepLoop(ort, session, graph) {
  const constants = {};
  for (const [name, spec] of Object.entries(graph.constants)) {
    constants[name] = tensor(ort, spec);
  }
  let state = {};
  for (const [name, spec] of Object.entries(graph.state)) {
    state[name] = tensor(ort, spec);
  }

  const losses = [];
  for (let t = 0; t < graph.scalars.length; t++) {
    const feeds = { ...constants, ...state };
    for (const [name, value] of Object.entries(graph.scalars[t])) {
      feeds[name] = new ort.Tensor("float32", Float32Array.from([value]), []);
    }
    for (const [name, spec] of Object.entries((graph.perStep || [])[t] || {})) {
      feeds[name] = indexTensor(ort, spec);
    }
    const out = await session.run(feeds);
    const next = {};
    for (const [name, spec] of Object.entries(graph.state)) {
      next[name] = out[spec.output];
    }
    state = next;
    losses.push(Number(out[graph.loss].data[0]));
  }
  return losses;
}

// ---------------------------------------------------------------------------
// The execution providers to try.
//
// Two bundles, mirroring inference_browser.mjs's loadOrt() variant split.
//
// wasm comes from the plain "onnxruntime-web" specifier, as inference.test.mjs
// and compare.test.mjs use it. Under Node that resolves through the package's
// "node" export condition to ort.node.min.mjs, which registers only the cpu and
// wasm backends -- ask it for webgpu and it answers "[webgpu] backend not
// found", which is a fact about the bundle, not about WebGPU's operator
// coverage. Reporting that as a coverage result would be exactly the dishonest
// verdict this test exists to avoid, so the accelerated EPs are attempted
// against "onnxruntime-web/all" instead: its export map has no node condition,
// so Node gets the browser bundle, which does register webgpu and webnn and
// therefore fails (or not) for the real reason.
const bundles = {};
async function loadBundle(variant) {
  if (!bundles[variant]) {
    bundles[variant] = import(
      variant === "all" ? "onnxruntime-web/all" : "onnxruntime-web"
    ).then((m) => {
      const ort = m.default ?? m;
      // Local wasm artifacts, single-threaded, no worker proxy: the same
      // offline configuration the other onnxruntime-web tests here use.
      ort.env.wasm.wasmPaths = DIST;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      return ort;
    });
  }
  return bundles[variant];
}

const TARGETS = [
  { id: "wasm", variant: "default", provider: "wasm" },
  { id: "webgpu", variant: "all", provider: "webgpu" },
  ...WEBNN_DEVICE_TYPES.map((device) => ({
    id: providerLabel(webnnProvider(device)),
    variant: "all",
    provider: webnnProvider(device),
  })),
];

// An EP that this environment simply does not have. onnxruntime-web reports it
// as a missing backend or a missing browser API, and the distinction from a
// real breakage is the whole point: absent is a skip, present-but-broken is a
// failure, and conflating them is how a test ends up claiming coverage it never
// measured.
function isUnavailable(message) {
  return /no available backend found|not supported in current environment|backend not found/i.test(
    message,
  );
}

// Run every fixture on one EP. Returns a report row and never throws: an EP
// that is not here at all is "unavailable" (skipped), an EP that is here and
// cannot run a step graph is "failed" (always fatal, whatever ORT_REQUIRE says
// -- a backend rejecting an allowlisted operator is precisely the finding this
// test exists to surface).
async function measure(target) {
  const row = { id: target.id, status: "unavailable", reason: "", graphs: [], ops: {} };
  let ort;
  try {
    ort = await loadBundle(target.variant);
  } catch (err) {
    row.reason = `bundle '${target.variant}' would not load: ${err.message}`;
    return row;
  }

  for (const graph of MANIFEST.graphs) {
    const model = new Uint8Array(readFileSync(join(HERE, graph.file)));
    const capture = captureOrtLogs();
    let session;
    try {
      session = await ort.InferenceSession.create(model, {
        executionProviders: [target.provider],
        // Optimization off, so what gets partitioned is the operator set the
        // builders actually emitted rather than whatever ORT fused it into.
        // That is the set EP_FRIENDLY_OPS makes a claim about.
        graphOptimizationLevel: "disabled",
        logSeverityLevel: 0,
        logVerbosityLevel: 4,
      });
    } catch (err) {
      capture.restore();
      const gone = isUnavailable(err.message);
      row.status = gone ? "unavailable" : "failed";
      // Name the graph only when one specifically broke; an absent backend
      // fails on whichever fixture happened to be first, which says nothing.
      row.reason = gone ? err.message : `${graph.name}: ${err.message}`;
      return row;
    }
    let losses;
    let error = null;
    try {
      losses = await runStepLoop(ort, session, graph);
    } catch (err) {
      error = err;
    }
    const placements = parsePlacements(capture.lines);
    if (VERBOSE) {
      for (const line of capture.lines) console.log("    | " + line);
    }
    capture.restore();
    await session.release?.();

    if (error) {
      row.status = "failed";
      row.reason = `${graph.name}: ${error.message}`;
      return row;
    }
    row.status = "ok";
    row.graphs.push({ name: graph.name, losses, placements });
    // Credit an op to a provider only where the placement report says so. The
    // "All nodes" shape names no ops, but then the graph's whole histogram
    // went to that provider.
    for (const [provider, info] of Object.entries(placements)) {
      const ops = info.ops === null ? Object.keys(graph.ops) : [...info.ops];
      for (const op of ops) (row.ops[op] = row.ops[op] || new Set()).add(provider);
    }
  }
  return row;
}


// ---------------------------------------------------------------------------
// EP_FRIENDLY_OPS as onnxsim/qat_graph.py spells it *right now*, read out of
// the python source by regex.
//
// The manifest's copy was baked when the fixtures were generated, so on its own
// it cannot notice the set growing a member afterwards -- and a member with no
// fixture behind it is exactly the unverified claim this test exists to
// prevent. Reading the source closes that loop from Node, with no python. It is
// a text match on a literal frozenset of string literals, which is all that
// declaration has ever been; anything it cannot parse returns null and the
// check says so rather than inventing a verdict.
function epFriendlyOpsFromSource() {
  let text;
  try {
    text = readFileSync(join(HERE, "..", "..", "..", "onnxsim", "qat_graph.py"), "utf8");
  } catch {
    return null; // running from a copy without the python package next to it
  }
  const block = text.match(/EP_FRIENDLY_OPS\s*=\s*frozenset\(\s*\{([^}]*)\}/);
  if (!block) return null;
  const ops = [...block[1].matchAll(/"([A-Za-z][\w.]*)"/g)].map((m) => m[1]);
  return ops.length ? ops.sort() : null;
}

// The fixture and the allowlist have to agree, or everything below measures
// something other than what qat_graph.py claims. Both directions matter: an op
// in a fixture but not in the set would be a builder escaping its own
// allowlist, and an op in the set with no fixture behind it is a claim this
// test cannot reach.
async function checkFixtureIntegrity() {
  const live = epFriendlyOpsFromSource();
  if (live === null) {
    console.log("  note - onnxsim/qat_graph.py not readable; manifest taken as given");
  } else {
    await check("manifest matches onnxsim/qat_graph.py's EP_FRIENDLY_OPS", () => {
      assert.deepEqual(
        MANIFEST.epFriendlyOps,
        live,
        "the allowlist changed since the fixtures were generated -- re-run " +
          "test/make_step_graph_fixtures.py and commit the .onnx files",
      );
    });
  }
  await check("fixtures cover exactly EP_FRIENDLY_OPS", () => {
    const covered = new Set();
    for (const graph of MANIFEST.graphs) {
      for (const op of Object.keys(graph.ops)) {
        assert.ok(
          MANIFEST.epFriendlyOps.includes(op),
          `${graph.file} contains '${op}', which is not in EP_FRIENDLY_OPS`,
        );
        covered.add(op);
      }
    }
    const missing = MANIFEST.epFriendlyOps.filter((op) => !covered.has(op));
    assert.deepEqual(
      missing,
      [],
      `no fixture exercises ${missing.join(", ")}; re-run make_step_graph_fixtures.py`,
    );
  });
}

// Assert what one EP's run actually showed: finite losses, a loop that moves,
// and agreement with the trajectory onnxruntime's CPU provider recorded.
async function checkRun(row) {
  for (const graph of row.graphs) {
    const reference = MANIFEST.graphs.find((g) => g.name === graph.name);
    const first = graph.losses[0];
    const last = graph.losses[graph.losses.length - 1];
    const trace = `loss ${first.toExponential(3)} -> ${last.toExponential(3)}`;
    await check(`${row.id}: ${graph.name} ran ${graph.losses.length} steps, ${trace}`, () => {
      const second = graph.losses[1];
      for (const loss of graph.losses) {
        assert.ok(Number.isFinite(loss), `loss ${loss} is not finite on ${row.id}`);
      }
      // A step that updated nothing would produce a flat trajectory and still
      // pass every "is it finite" check, so the loop moving is the real claim:
      // the state that came out of step 0 is what step 1 was fed.
      assert.ok(
        Math.abs(second - first) > 1e-3 * Math.abs(first),
        `${row.id}/${graph.name}: the second step did not move the loss ` +
          `(${first} -> ${second})`,
      );
      graph.losses.forEach((loss, i) => {
        const want = reference.referenceLosses[i];
        assert.ok(
          closeEnough(loss, want),
          `${row.id}/${graph.name} step ${i}: loss ${loss} vs onnxruntime CPU ${want}`,
        );
      });
    });
  }
}

// The verdict tables. Printed whatever happened, since "which EP was not
// reached" is as much of the result as "which one worked".
function report(rows) {
  console.log("\nexecution providers");
  for (const row of rows) {
    if (row.status === "ok") {
      const ops = Object.keys(row.ops).length;
      console.log(
        `  ${row.id.padEnd(11)} VERIFIED  ${row.graphs.length}/${MANIFEST.graphs.length} ` +
          `step graphs ran; ${ops}/${MANIFEST.epFriendlyOps.length} allowlisted ops executed`,
      );
    } else {
      const verdict = row.status === "failed" ? "BROKEN   " : "UNREACHED";
      console.log(`  ${row.id.padEnd(11)} ${verdict} ${row.reason || "unavailable"}`);
    }
  }

  // "wasm -> CPUExecutionProvider" is not a mislabel: onnxruntime-web's wasm
  // EP *is* ORT's CPU EP compiled to WebAssembly, and that is the name its own
  // partitioner reports. An accelerated EP would show its own name here
  // (JsExecutionProvider for WebGPU, WebNNExecutionProvider for WebNN) for the
  // ops it took, and CPUExecutionProvider for the ops it handed back -- which
  // is exactly the measurement EP_FRIENDLY_OPS needs and cannot get here.
  console.log("\noperator placement (EP_FRIENDLY_OPS x provider actually assigned)");
  for (const op of MANIFEST.epFriendlyOps) {
    const where = [];
    // Only a row that ran every fixture to completion gets to claim an
    // operator: a half-finished run's placements say where ORT *intended* to
    // put nodes, not that they computed the right thing.
    for (const row of rows.filter((r) => r.status === "ok")) {
      if (row.ops[op]) where.push(`${row.id} -> ${[...row.ops[op]].join("+")}`);
    }
    console.log(
      `  ${op.padEnd(11)} ${where.length ? where.join(", ") : "NOT EXECUTED ANYWHERE"}`,
    );
  }

  const unreached = rows.filter((r) => r.status !== "ok").map((r) => r.id);
  if (unreached.length) {
    console.log(
      `\nNOT VERIFIED by this run: ${unreached.join(", ")}. onnxruntime-web refuses ` +
        "these backends in headless Node (no navigator.gpu, no navigator.ml), so " +
        "nothing here supports or contradicts EP_FRIENDLY_OPS' claim about them. " +
        "Measuring them needs a browser -- the converter page's own inference " +
        "panel -- or a host where those APIs exist, where ORT_REQUIRE turns the " +
        "skip into a hard requirement.",
    );
  }
}

async function main() {
  console.log(
    `onnxsim step-graph EP coverage -- ${MANIFEST.graphs.length} fixtures, ` +
      `opset ${MANIFEST.opset}, ${MANIFEST.numSteps} steps each\n`,
  );
  await checkFixtureIntegrity();

  // webnn.mjs's own probe, on whatever navigator this runtime has, so the
  // reason the WebNN rows below are skipped is stated by the same code the
  // converter page states it with rather than inferred from an ORT error.
  console.log("  " + formatWebnnStatus(await detectWebnn(globalThis.navigator)));

  const rows = [];
  for (const target of TARGETS) {
    const row = await measure(target);
    rows.push(row);
    if (row.status === "ok") {
      await checkRun(row);
      continue;
    }
    // "failed" means the backend is here and could not run an allowlisted
    // graph -- never skippable. "unavailable" means it is not here at all,
    // which is only a failure where the caller said it must be.
    const fatal = row.status === "failed" || REQUIRED.has(row.id);
    console.log(
      `  ${fatal ? "FAIL" : "skip"} - ${row.id}: ${row.reason || "unavailable"}`,
    );
    if (fatal) failures += 1;
  }

  report(rows);

  if (failures) {
    console.error(
      `\nFAIL: ${failures} execution provider(s) did not run the step graphs ` +
        "(present but broken, or named in ORT_REQUIRE and unreachable)",
    );
    process.exit(1);
  }
  console.log(`\nstep-graph EP: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
