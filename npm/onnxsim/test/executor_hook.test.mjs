// Contract test for the hook used by the WASM ModelExecutor bridge.
// This intentionally uses a fake ORT module: it validates the binary tensor
// packing/unpacking without requiring a WASM runtime, browser, or accelerator.

import assert from "node:assert/strict";
import { makeOrtRunner } from "../../../scripts/convertmodel/ort_executor.mjs";

let seenModel;
let seenFeed;
class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

const ort = {
  InferenceSession: {
    async create(model, options) {
      seenModel = { model, options };
      return {
        inputNames: ["input"],
        outputNames: ["output"],
        async run(feeds) {
          seenFeed = feeds.input;
          return {
            output: {
              type: "float32",
              dims: [2],
              data: new Float32Array([3.5, -2.25]),
            },
          };
        },
        async release() {},
      };
    },
  },
  Tensor: FakeTensor,
};

const runner = makeOrtRunner(ort);
const input = new Float32Array([1.25, -4.0]);
const result = await runner(
  new Uint8Array([0x08, 0x09]),
  new Uint8Array(input.buffer),
  new Float64Array([1, 1, 2]),
);

assert.deepEqual(Array.from(seenModel.model), [0x08, 0x09]);
assert.equal(seenModel.options.graphOptimizationLevel, "disabled");
assert.deepEqual(seenModel.options.executionProviders, ["wasm"]);
assert.equal(seenFeed.type, "float32");
assert.deepEqual(seenFeed.dims, [2]);
assert.deepEqual(Array.from(seenFeed.data), Array.from(input));
assert.equal(result.data.byteLength, 8);
const output = new Float32Array(result.data.buffer, result.data.byteOffset, 2);
assert.deepEqual(Array.from(output), [3.5, -2.25]);
assert.deepEqual(Array.from(result.meta), [1, 1, 2]);
console.log("PASS: executor hook contract");
