// Unit test for the LoRA panel's block discovery (lora_blocks.mjs). No DOM,
// no onnxruntime-web and no wasm, the same way qat_blocks.test.mjs covers
// its own module: discovery reads model bytes with the page's own protobuf
// wire reader, so the whole walk is exercisable under Node against models
// assembled here.
//
// livenessCuts/primaryGraphInput/readGraph are qat_blocks.mjs's own, already
// covered by qat_blocks.test.mjs -- this file only exercises what
// lora_blocks.mjs adds: closing a block on injected adapter targets instead
// of quantized layers.
//
// Usage:
//   node test/lora_blocks.test.mjs

import assert from "node:assert/strict";
import { discoverLoraBlocks } from "../lora_blocks.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

// The same dependency-free protobuf encoder qat_blocks.test.mjs uses.
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

function makeNode(opType, inputs, outputs, name = "") {
  return [
    ...inputs.flatMap((n) => strField(1, n)),
    ...outputs.flatMap((n) => strField(2, n)),
    ...strField(3, name || outputs[0] || opType),
    ...strField(4, opType),
  ];
}

function makeModel({ nodes = [], initializers = [], inputs = [], outputs = [] }) {
  const graph = [
    ...nodes.flatMap((n) => bytesField(1, n)),
    ...initializers.flatMap((n) => bytesField(5, strField(8, n))),
    ...inputs.flatMap((n) => bytesField(11, strField(1, n))),
    ...outputs.flatMap((n) => bytesField(12, strField(1, n))),
  ];
  return new Uint8Array(bytesField(7, graph));
}

// X -> [MatMul, branch] -> h (adapter #1's own closing Add) -> [MatMul,
// branch] -> Y (adapter #2's). Every node between two adjacent cuts is
// "the branch", standing in for the extra MatMul/MatMul/Add InjectLora
// actually splices in -- lora_blocks.mjs only ever looks at node outputs, so
// a single Add per target is enough to exercise the walk.
const INJECTED_CHAIN = makeModel({
  nodes: [makeNode("MatMul", ["X", "W1"], ["h"]), makeNode("MatMul", ["h", "W2"], ["Y"])],
  initializers: ["W1", "W2"],
  inputs: ["X"],
  outputs: ["Y"],
});
const TWO_TARGETS = [{ nodeOutput: "h" }, { nodeOutput: "Y" }];

check("maxTargetsPerBlock decides how many adapters a block merges", () => {
  assert.deepEqual(discoverLoraBlocks(INJECTED_CHAIN, TWO_TARGETS, { maxTargetsPerBlock: 1 }), [
    { input: "X", output: "h", targetOutputs: ["h"], nodes: 1 },
    { input: "h", output: "Y", targetOutputs: ["Y"], nodes: 1 },
  ]);
  assert.deepEqual(discoverLoraBlocks(INJECTED_CHAIN, TWO_TARGETS, { maxTargetsPerBlock: 2 }), [
    { input: "X", output: "Y", targetOutputs: ["h", "Y"], nodes: 2 },
  ]);
  assert.throws(
    () => discoverLoraBlocks(INJECTED_CHAIN, TWO_TARGETS, { maxTargetsPerBlock: 0 }),
    /at least 1/,
  );
});

// A cut whose surviving tensor is not one of the adapter's own outputs never
// closes a block on its own -- only injected branches count, not every node
// output the liveness walk happens to land on.
check("only a cut landing on an adapter's own output closes a block", () => {
  const untouched = makeModel({
    nodes: [
      makeNode("MatMul", ["X", "W1"], ["h"]),
      makeNode("Relu", ["h"], ["r"]),
      makeNode("MatMul", ["r", "W2"], ["Y"]),
    ],
    initializers: ["W1", "W2"],
    inputs: ["X"],
    outputs: ["Y"],
  });
  // "h" and "r" both cut the graph (a chain cuts at every gap), but only "Y"
  // is an adapter's own output, so the walk keeps accumulating past both
  // uninjected cuts until it reaches the one that is.
  assert.deepEqual(discoverLoraBlocks(untouched, [{ nodeOutput: "Y" }], { maxTargetsPerBlock: 1 }), [
    { input: "X", output: "Y", targetOutputs: ["Y"], nodes: 3 },
  ]);
});

// No adapter target falls inside any span: nothing to train, so no block --
// discover_lora_blocks' own rule ("a slice with nothing to train is not a
// block"), not an error.
check("a block with no adapter target inside it is never proposed", () => {
  assert.deepEqual(discoverLoraBlocks(INJECTED_CHAIN, [], { maxTargetsPerBlock: 1 }), []);
  assert.deepEqual(
    discoverLoraBlocks(INJECTED_CHAIN, [{ nodeOutput: "nowhere" }], { maxTargetsPerBlock: 1 }),
    [],
  );
});

// The behaviour that makes whole-model discovery usable at all: one op with
// no gradient rule becomes a gap between blocks instead of refusing the
// model -- identical to discoverBlocks' own gap handling, since it is the
// same liveness walk underneath.
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
    discoverLoraBlocks(withSin, [{ nodeOutput: "h" }, { nodeOutput: "Y" }], {
      maxTargetsPerBlock: 1,
    }).map((b) => [b.input, b.output]),
    [
      ["X", "h"],
      ["g", "Y"],
    ],
  );
  // A gap also closes whatever block was still accumulating, even short of
  // maxTargetsPerBlock -- the pending block ends at the last cut still on
  // the trainable side rather than swallowing the gap.
  assert.deepEqual(
    discoverLoraBlocks(withSin, [{ nodeOutput: "h" }, { nodeOutput: "Y" }], {
      maxTargetsPerBlock: 2,
    }).map((b) => [b.input, b.output]),
    [
      ["X", "h"],
      ["g", "Y"],
    ],
  );
});

// A never-reconverging branch has no cut to close a block at, adapters
// notwithstanding -- an empty result is a legitimate answer.
check("a never-reconverging branch yields no blocks", () => {
  const forked = makeModel({
    nodes: [makeNode("MatMul", ["X", "W1"], ["a"]), makeNode("MatMul", ["X", "W2"], ["b"])],
    initializers: ["W1", "W2"],
    inputs: ["X"],
    outputs: ["a", "b"],
  });
  assert.deepEqual(discoverLoraBlocks(forked, [{ nodeOutput: "a" }, { nodeOutput: "b" }]), []);
});

console.log(`\nlora_blocks: ${passed} checks passed`);
