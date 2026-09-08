// Block discovery for the converter page's "Fine-tune (QAT)" panel.
//
// onnxsim_qat_build_step_graph names a block the way onnxsim.apply_qat does:
// by two tensor names, its input and its output. A user can type them, but on
// a real model nobody knows them offhand -- so this module proposes them,
// porting onnxsim/qat.py's discover_qat_blocks (_liveness_cuts /
// _primary_graph_input and the span walk over them) to JavaScript, reading the
// model bytes directly with the same dependency-free protobuf wire reader
// macs.mjs and shapes.mjs already share.
//
// It is a liveness argument rather than a pattern match, and that is the whole
// substance: walk the nodes in graph order tracking which activations are
// live at each gap, and cut wherever exactly one is. A slice bounded that way
// is self-contained, so it can be lifted out and differentiated on its own --
// which means residual connections *place* the boundaries instead of defeating
// them (inside `y = f(x) + x` the skip tensor is live alongside every
// intermediate, so the first cut after `x` is the residual Add's own output).
// qat.py's own docstrings carry the full argument; this is the port, not a
// second derivation of it.
//
// **Where this port is deliberately weaker than the Python.** Python closes a
// block after `max_layers_per_block` *quantized* layers, which it knows from
// qat.py's _find_layers matching the float model's layers against the
// quantized model's. There is no binding for that -- the wasm module exposes
// BuildQatStepGraph and WriteBackQatState and nothing else of qat.py -- so a
// "layer" here is a float-side weight-bearing op instead: a MatMul/Gemm/Conv
// with a constant (initializer) weight, which is the population every
// quantizer in onnxsim draws its layers from. The two agree except at the
// margins each scheme adds (quantize_weight_only_int4 skips a MatMul whose
// reduction axis is not a multiple of 32, say), so a proposal can occasionally
// merge one span too few or too many. That costs block granularity, not
// correctness: BuildQatStepGraph refuses a block with no layer of the
// requested scheme in it, loudly, and the panel reports the refusal and moves
// to the next block.
//
// Field numbers below are from onnx.proto:
//   ModelProto     graph = 7
//   GraphProto     node = 1, initializer = 5, input = 11, output = 12
//   NodeProto      input = 1, output = 2, name = 3, op_type = 4
//   TensorProto    name = 8
//   ValueInfoProto name = 1

import { fields, decode } from "./macs.mjs";

// onnxsim.graph_grad.SUPPORTED_OPS -- the op types build_backward has a
// gradient rule for, and therefore the only ops a block may contain. It tracks
// graph_grad.py's _RULES table and graph_grad.cpp's Rules(), which are the same
// table on both sides. A span containing anything else is a gap between blocks
// rather than a failed model, which is what makes whole-model discovery usable
// at all.
//
// This is a pre-filter, not the authority: BuildQatStepGraph checks the real
// table and refuses a block whose ops it cannot differentiate, naming them. So
// a list that lags a newly added rule costs block granularity (that op becomes
// a gap), and a list that runs ahead of one costs a proposed block that is then
// refused and skipped with its reason -- neither trains anything wrongly.
export const DIFFERENTIABLE_OPS = new Set([
  "Add",
  "Clip",
  "Conv",
  "Div",
  "Erf",
  "Exp",
  "Gemm",
  "Identity",
  "LayerNormalization",
  "MatMul",
  "Mul",
  "Neg",
  "ReduceMean",
  "ReduceSum",
  "Relu",
  "Reshape",
  "Sigmoid",
  "Softmax",
  "Sqrt",
  "Sub",
  "Tanh",
  "Transpose",
]);

// The ops onnxsim's quantizers turn into quantized layers -- see the header
// note on why this stands in for qat.py's _find_layers here.
const WEIGHTED_OPS = new Set(["MatMul", "Gemm", "Conv"]);

function parseNode(buf) {
  const input = [];
  const output = [];
  let name = "";
  let opType = "";
  for (const f of fields(buf)) {
    if (f.field === 1 && f.wire === 2) input.push(decode(f.bytes));
    else if (f.field === 2 && f.wire === 2) output.push(decode(f.bytes));
    else if (f.field === 3 && f.wire === 2) name = decode(f.bytes);
    else if (f.field === 4 && f.wire === 2) opType = decode(f.bytes);
  }
  return { name, opType, input, output };
}

// The `name` field of a ValueInfoProto (field 1) or a TensorProto (field 8) --
// the only field either is read for here.
function parseName(buf, field) {
  for (const f of fields(buf)) {
    if (f.field === field && f.wire === 2) return decode(f.bytes);
  }
  return "";
}

function parseGraph(buf) {
  const nodes = [];
  const initializers = [];
  const inputs = [];
  const outputs = [];
  for (const f of fields(buf)) {
    if (f.wire !== 2) continue;
    if (f.field === 1) nodes.push(parseNode(f.bytes));
    else if (f.field === 5) initializers.push(parseName(f.bytes, 8));
    else if (f.field === 11) inputs.push(parseName(f.bytes, 1));
    else if (f.field === 12) outputs.push(parseName(f.bytes, 1));
  }
  return { nodes, initializers, inputs, outputs };
}

// Read the pieces of a model's graph block discovery needs: its nodes in graph
// (topological) order with their input/output tensor names, and the three name
// sets that decide what counts as an activation. Returns
// { nodes: [{ name, opType, input, output }], initializers, inputs, outputs }.
export function readGraph(modelBytes) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  let graph = { nodes: [], initializers: [], inputs: [], outputs: [] };
  for (const f of fields(buf)) {
    if (f.field === 7 && f.wire === 2) graph = parseGraph(f.bytes);
  }
  return graph;
}

// The graph input the most nodes depend on -- the main activation path, and a
// heuristic named as one (qat.py's _primary_graph_input). Models with several
// inputs almost always have one carrying activations and the others carrying
// masks or ids, and "reaches the most nodes" separates those without depending
// on naming conventions. Ties go to the earlier graph input. Getting it wrong
// costs block granularity, not correctness: every non-chosen input is
// teacher-forced exactly.
export function primaryGraphInput(graph) {
  const initializers = new Set(graph.initializers);
  const candidates = graph.inputs.filter((name) => !initializers.has(name));
  if (candidates.length === 0) return null;

  let best = candidates[0];
  let bestReach = -1;
  for (const name of candidates) {
    const reached = new Set([name]);
    let count = 0;
    for (const node of graph.nodes) {
      if (node.input.some((inp) => inp && reached.has(inp))) {
        count += 1;
        for (const out of node.output) if (out) reached.add(out);
      }
    }
    if (count > bestReach) {
      best = name;
      bestReach = count;
    }
  }
  return best;
}

// Every index at which the graph narrows to a single live activation, with the
// tensor that survives it: [{ index, name }], where `index` is the node the
// gap follows (-1 for the gap before the first node). qat.py's _liveness_cuts.
//
// Initializers are excluded (they are not activations -- every block gets its
// own copy in its step graph), and so is every graph input other than
// `primaryInput`: a mask or a position-id tensor is byte-identical in teacher
// and student, so teacher-forcing it into a block is exact rather than an
// approximation, and letting it span the whole graph would otherwise suppress
// every cut in a model that has one.
export function livenessCuts(graph, primaryInput) {
  const initializers = new Set(graph.initializers);
  const graphInputs = graph.inputs.filter((name) => !initializers.has(name));
  const ignored = new Set(graphInputs.filter((name) => name !== primaryInput));

  // A tensor is live until its last consumer; a graph output is live past the
  // end of the graph, so it is never dropped before the final gap.
  const lastUse = new Map();
  graph.nodes.forEach((node, index) => {
    for (const name of node.input) {
      if (name && !initializers.has(name) && !ignored.has(name)) lastUse.set(name, index);
    }
  });
  for (const name of graph.outputs) {
    if (name && !initializers.has(name) && !ignored.has(name)) lastUse.set(name, graph.nodes.length);
  }

  const cuts = [];
  const live = new Set(
    graphInputs.filter((name) => !ignored.has(name) && (lastUse.get(name) ?? -1) > -1),
  );
  if (live.size === 1) cuts.push({ index: -1, name: [...live][0] });
  graph.nodes.forEach((node, index) => {
    for (const name of node.output) {
      if (name && !ignored.has(name) && (lastUse.get(name) ?? -1) > index) live.add(name);
    }
    for (const name of [...live]) {
      if ((lastUse.get(name) ?? -1) <= index) live.delete(name);
    }
    if (live.size === 1) cuts.push({ index, name: [...live][0] });
  });
  return cuts;
}

// True for a node onnxsim's quantizers would turn into a quantized layer: a
// MatMul/Gemm/Conv whose weight is a constant. See the header note.
function isWeightedLayer(node, constants) {
  return node.opType && WEIGHTED_OPS.has(node.opType) && node.input.some((n) => n && constants.has(n));
}

// Partition a model into blocks the step-graph builder can be asked to train,
// as [{ input, output, layers, nodes }] in graph order -- `input`/`output` are
// the two tensor names to pass to onnxsim_qat_build_step_graph, `layers` is how
// many weight-bearing ops the block contains and `nodes` how many nodes.
//
// Consecutive blocks need not be adjacent: a span containing an op with no
// gradient rule becomes a gap between blocks rather than a refusal of the whole
// model. An empty result is a legitimate answer (a graph whose activations
// never narrow to one live tensor has no cut to make), not an error.
//
// `maxLayersPerBlock` is qat.py's own knob and its default of 2 is the
// paired-projection shape BRECQ targets (a ResNet BasicBlock's two convolutions,
// a transformer FFN's up/down pair): 1 gives per-layer blocks, a large value
// gives one block per gap between undifferentiable ops -- on a model with no
// such gap, one block spanning the whole graph, which is the end-to-end
// objective rather than a surrogate for it, and which docs/qat.md measures as
// fitting the calibration set better while generalizing worse.
export function discoverBlocks(modelBytes, { maxLayersPerBlock = 2 } = {}) {
  if (maxLayersPerBlock < 1) throw new Error("maxLayersPerBlock must be at least 1");
  const graph = readGraph(modelBytes);
  const cuts = livenessCuts(graph, primaryGraphInput(graph));
  const constants = new Set(graph.initializers);

  const blocks = [];
  let start = cuts.length ? cuts[0] : null;
  let layers = 0;
  let nodes = 0;
  const close = (end) => {
    blocks.push({ input: start.name, output: end.name, layers, nodes });
    start = end;
    layers = 0;
    nodes = 0;
  };
  for (let i = 1; i < cuts.length; i++) {
    const previous = cuts[i - 1];
    const current = cuts[i];
    const span = graph.nodes.slice(previous.index + 1, current.index + 1);
    if (span.some((node) => !DIFFERENTIABLE_OPS.has(node.opType))) {
      // A gap. Close whatever was pending *before* it (the pending block ends
      // at the last cut still on the trainable side) and reopen after it.
      if (start !== null && layers && start.index < previous.index) close(previous);
      start = current;
      layers = 0;
      nodes = 0;
      continue;
    }
    if (start === null) start = previous;
    layers += span.filter((node) => isWeightedLayer(node, constants)).length;
    nodes += span.length;
    if (layers >= maxLayersPerBlock) close(current);
  }
  const last = cuts.length ? cuts[cuts.length - 1] : null;
  if (start !== null && layers && last && start.index < last.index) close(last);
  return blocks;
}
