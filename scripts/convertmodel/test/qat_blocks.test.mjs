// Unit test for the QAT panel's block discovery (qat_blocks.mjs). No DOM, no
// onnxruntime-web and no wasm: discovery reads the model bytes with the page's
// own protobuf wire reader, so the whole thing -- the liveness argument, the
// primary-input heuristic, the span walk -- is exercisable under Node against
// models assembled here, byte by byte.
//
// The models are encoded rather than committed for the reason
// qat_step_graph.test.mjs builds its fixture in-process: what is under test is
// a handful of *topologies* (a chain, a residual, a mask input, an
// undifferentiable op in the middle), and a topology is clearer as six lines of
// makeNode than as a checked-in .onnx nobody can read.
//
// Usage:
//   node test/qat_blocks.test.mjs

import assert from "node:assert/strict";
import {
  DIFFERENTIABLE_OPS,
  discoverBlocks,
  livenessCuts,
  primaryGraphInput,
  readGraph,
} from "../qat_blocks.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

// ---------------------------------------------------------------------------
// A protobuf encoder just large enough for the fields readGraph reads. The
// mirror image of macs.mjs's `fields` reader, and no more general than it
// needs to be: length-delimited fields and nothing else.
function varint(n) {
  const out = [];
  let v = n;
  do {
    let byte = v & 0x7f;
    v >>>= 7;
    if (v) byte |= 0x80;
    out.push(byte);
  } while (v);
  return out;
}

function bytesField(field, bytes) {
  return [...varint(field * 8 + 2), ...varint(bytes.length), ...bytes];
}

const utf8 = new TextEncoder();
const strField = (field, s) => bytesField(field, [...utf8.encode(s)]);

// NodeProto: input = 1, output = 2, name = 3, op_type = 4
function makeNode(opType, inputs, outputs, name = "") {
  return [
    ...inputs.flatMap((n) => strField(1, n)),
    ...outputs.flatMap((n) => strField(2, n)),
    ...strField(3, name || outputs[0] || opType),
    ...strField(4, opType),
  ];
}

// ModelProto { graph = 7 } wrapping GraphProto { node = 1, initializer = 5,
// input = 11, output = 12 }; initializers are TensorProto { name = 8 } and
// graph inputs/outputs ValueInfoProto { name = 1 }.
function makeModel({ nodes = [], initializers = [], inputs = [], outputs = [] }) {
  const graph = [
    ...nodes.flatMap((n) => bytesField(1, n)),
    ...initializers.flatMap((n) => bytesField(5, strField(8, n))),
    ...inputs.flatMap((n) => bytesField(11, strField(1, n))),
    ...outputs.flatMap((n) => bytesField(12, strField(1, n))),
  ];
  return new Uint8Array(bytesField(7, graph));
}

// X -> MatMul -> h -> MatMul -> Y, the shape every "a plain MLP has a cut
// between every pair of layers" claim is about.
const CHAIN = makeModel({
  nodes: [makeNode("MatMul", ["X", "W1"], ["h"]), makeNode("MatMul", ["h", "W2"], ["Y"])],
  initializers: ["W1", "W2"],
  inputs: ["X"],
  outputs: ["Y"],
});

// ---------------------------------------------------------------------------

check("readGraph reads nodes, initializers and the graph interface", () => {
  const graph = readGraph(CHAIN);
  assert.deepEqual(
    graph.nodes.map((n) => [n.opType, n.input, n.output]),
    [
      ["MatMul", ["X", "W1"], ["h"]],
      ["MatMul", ["h", "W2"], ["Y"]],
    ],
  );
  assert.deepEqual(graph.initializers, ["W1", "W2"]);
  assert.deepEqual(graph.inputs, ["X"]);
  assert.deepEqual(graph.outputs, ["Y"]);
});

check("the differentiable set tracks graph_grad's rule table", () => {
  // Not an exact-membership assertion: graph_grad's table grows, this list is
  // only a pre-filter for it (see qat_blocks.mjs), and pinning a count here
  // would fail on a rule being *added* -- which is not a bug. What must hold is
  // that the ops a block-wise reconstruction is made of are in it and that an
  // arbitrary op is not.
  for (const op of ["MatMul", "Gemm", "LayerNormalization", "Softmax", "Transpose", "Add"]) {
    assert.ok(DIFFERENTIABLE_OPS.has(op), `${op} should be differentiable`);
  }
  assert.ok(!DIFFERENTIABLE_OPS.has("Sin"));
  assert.ok(DIFFERENTIABLE_OPS.size >= 21, "graph_grad had 21 rules when this landed");
});

check("a chain cuts at every gap", () => {
  const graph = readGraph(CHAIN);
  assert.deepEqual(livenessCuts(graph, primaryGraphInput(graph)), [
    { index: -1, name: "X" },
    { index: 0, name: "h" },
    { index: 1, name: "Y" },
  ]);
});

// The point of the whole liveness formulation: inside `y = f(x) + x` the skip
// tensor is live alongside every intermediate, so there is no cut in the middle
// of the residual and the next one is the Add's own output -- where a person
// would have drawn the boundary, derived rather than special-cased.
check("a residual places the boundary at the Add, not inside it", () => {
  const residual = makeModel({
    nodes: [makeNode("MatMul", ["X", "W"], ["h"]), makeNode("Add", ["h", "X"], ["Y"])],
    initializers: ["W"],
    inputs: ["X"],
    outputs: ["Y"],
  });
  const graph = readGraph(residual);
  assert.deepEqual(livenessCuts(graph, primaryGraphInput(graph)), [
    { index: -1, name: "X" },
    { index: 1, name: "Y" },
  ]);
});

// A mask input is byte-identical in teacher and student, so teacher-forcing it
// is exact; letting it stay live would suppress every cut in the model.
check("a secondary graph input does not suppress cuts", () => {
  const masked = makeModel({
    nodes: [makeNode("MatMul", ["X", "W"], ["h"]), makeNode("Mul", ["h", "mask"], ["Y"])],
    initializers: ["W"],
    inputs: ["X", "mask"],
    outputs: ["Y"],
  });
  const graph = readGraph(masked);
  assert.equal(primaryGraphInput(graph), "X", "X reaches both nodes, mask only one");
  assert.deepEqual(
    livenessCuts(graph, "X").map((c) => c.name),
    ["X", "h", "Y"],
  );
  // Choosing the other input costs granularity (X is then the live one that
  // spans the model), never correctness -- qat.py's own note on the heuristic.
  assert.deepEqual(
    livenessCuts(graph, "mask").map((c) => c.name),
    ["mask", "Y"],
  );
});

check("maxLayersPerBlock decides how many layers a block merges", () => {
  assert.deepEqual(discoverBlocks(CHAIN, { maxLayersPerBlock: 1 }), [
    { input: "X", output: "h", layers: 1, nodes: 1 },
    { input: "h", output: "Y", layers: 1, nodes: 1 },
  ]);
  assert.deepEqual(discoverBlocks(CHAIN, { maxLayersPerBlock: 2 }), [
    { input: "X", output: "Y", layers: 2, nodes: 2 },
  ]);
  assert.throws(() => discoverBlocks(CHAIN, { maxLayersPerBlock: 0 }), /at least 1/);
});

// A MatMul against another activation is not a layer -- there is no weight to
// train -- so it never closes a block on its own.
check("only a MatMul with a constant weight counts as a layer", () => {
  const attentionish = makeModel({
    nodes: [makeNode("MatMul", ["X", "X2"], ["h"]), makeNode("MatMul", ["h", "W"], ["Y"])],
    initializers: ["W"],
    inputs: ["X", "X2"],
    outputs: ["Y"],
  });
  assert.deepEqual(discoverBlocks(attentionish, { maxLayersPerBlock: 1 }), [
    { input: "X", output: "Y", layers: 1, nodes: 2 },
  ]);
});

// The behaviour that makes whole-model discovery usable: one op with no
// gradient rule becomes a gap between blocks instead of refusing the model.
check("an undifferentiable op becomes a gap, not a failure", () => {
  const withSin = makeModel({
    nodes: [
      makeNode("MatMul", ["X", "W1"], ["h"]),
      makeNode("Sin", ["h"], ["g"]),
      makeNode("MatMul", ["g", "W2"], ["Y"]),
    ],
    initializers: ["W1", "W2"],
    inputs: ["X"],
    outputs: ["Y"],
  });
  assert.deepEqual(
    discoverBlocks(withSin, { maxLayersPerBlock: 1 }).map((b) => [b.input, b.output]),
    [
      ["X", "h"],
      ["g", "Y"],
    ],
  );
  // A gap also *closes* whatever block was still accumulating, even when it
  // has not reached maxLayersPerBlock yet -- a pending block ends at the last
  // cut still on the trainable side rather than swallowing the gap.
  assert.deepEqual(
    discoverBlocks(withSin, { maxLayersPerBlock: 2 }).map((b) => [b.input, b.output]),
    [
      ["X", "h"],
      ["g", "Y"],
    ],
  );
});

// An empty plan is a legitimate answer, not an error: a graph whose
// activations never narrow to one live tensor has nowhere to cut.
check("a never-reconverging branch yields no blocks", () => {
  const forked = makeModel({
    nodes: [makeNode("MatMul", ["X", "W1"], ["a"]), makeNode("MatMul", ["X", "W2"], ["b"])],
    initializers: ["W1", "W2"],
    inputs: ["X"],
    outputs: ["a", "b"],
  });
  assert.deepEqual(discoverBlocks(forked), []);
});

check("a model with no graph input at all is handled", () => {
  const empty = makeModel({});
  assert.equal(primaryGraphInput(readGraph(empty)), null);
  assert.deepEqual(discoverBlocks(empty), []);
});

console.log(`\nqat_blocks: ${passed} checks passed`);
