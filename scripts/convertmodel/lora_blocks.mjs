// Block discovery for the converter page's "Fine-tune (LoRA)" panel --
// porting onnxsim/lora.py's discover_lora_blocks to JavaScript, the same way
// qat_blocks.mjs ports discover_qat_blocks.
//
// onnxsim_lora_build_step_graph names a block the way onnxsim.train_lora does:
// by two tensor names, its input and its output. lora_finetune.mjs's own
// wholeGraphBlock already covers the common case (InjectLora applies
// globally in one pass, so "the whole model" is always a legal, correct
// block) -- this module is for the other case, a model whose injected
// adapters are spread across enough of the graph, or through enough
// undifferentiable ops, that training it as one block is either too coarse
// or outright refused. Rather than make the user hunt for tensor names by
// hand, this proposes a sequence of smaller blocks, mirroring how
// discoverBlocks already does it for QAT.
//
// **Reused, not reimplemented.** livenessCuts/primaryGraphInput/readGraph
// and DIFFERENTIABLE_OPS come straight from qat_blocks.mjs -- none of the
// four is QAT-specific (see that module's own header for the liveness
// argument itself, which applies to any block-wise reconstruction, LoRA
// included). What differs from discoverBlocks is only the question a span
// answers before it closes a block: "how many quantized layers does it
// contain" becomes "how many of the adapter's own injected targets does it
// contain" -- discover_lora_blocks's own liveness_cuts/_slice_block reuse,
// ported.
//
// **Where this port simplifies, the same way discoverBlocks already does.**
// Python's discover_lora_blocks re-slices each proposed (input, output) pair
// through _slice_block (the forward/backward reachability intersection)
// before reading off which targets actually fall inside it. This port skips
// that second pass, exactly as discoverBlocks does for QAT's own layer
// count: a liveness cut is, by construction, a point where the graph
// narrows to one live tensor, so the span between two consecutive cuts *is*
// the slice a caller would get by naming them -- there is no reachable node
// outside it to disagree about. Safe for the same reason discoverBlocks'
// own simplification is safe: onnxsim_lora_build_step_graph is the
// authority and refuses a block it cannot differentiate, so a proposal this
// module gets wrong costs granularity, never correctness.

import { DIFFERENTIABLE_OPS, livenessCuts, primaryGraphInput, readGraph } from "./qat_blocks.mjs";

export { DIFFERENTIABLE_OPS, livenessCuts, primaryGraphInput, readGraph };

// Partition the *injected* model into blocks onnxsim_lora_build_step_graph
// can be asked to train, as [{ input, output, targetOutputs, nodes }] in
// graph order -- `input`/`output` are the two tensor names to pass to
// onnxsim_lora_build_step_graph, `targetOutputs` is which of `adapterTargets`
// (by their own `nodeOutput`) fall inside the block -- filter the adapter
// array on membership in it to get the LoraAdapter subset that block trains
// -- and `nodes` is how many nodes the block contains.
//
// `adapterTargets` is the array onnxsim_lora_inject's own `adapter` field
// returns (or already a caller-narrowed subset of it): every entry needs
// only its own `nodeOutput`, the tensor name InjectLora restored to the
// original node's output, matching lora.py's LoraTarget.node_output exactly
// (this is the closing Add's own output, not the weight name -- a target the
// discovery walk finds is one whose *branch resolves* inside the block, not
// merely one whose base weight happens to sit there).
//
// Consecutive blocks need not be adjacent: a span containing an op with no
// gradient rule becomes a gap between blocks rather than a refusal of the
// whole model. A block with no adapter target inside it is dropped -- a
// slice with nothing to train is not a block, discover_lora_blocks's own
// rule -- so an empty result is a legitimate answer (no cut a target
// actually falls behind) rather than an error.
//
// `maxTargetsPerBlock` is discover_lora_blocks's own knob: 1 gives
// per-adapter blocks (one train_lora call per injected branch), a large
// value gives one block per gap between undifferentiable ops -- on a model
// with no such gap, the whole graph as a single block, which is what
// wholeGraphBlock already returns directly without walking cuts at all.
export function discoverLoraBlocks(injectedBytes, adapterTargets, { maxTargetsPerBlock = 2 } = {}) {
  if (maxTargetsPerBlock < 1) throw new Error("maxTargetsPerBlock must be at least 1");
  const graph = readGraph(injectedBytes);
  const cuts = livenessCuts(graph, primaryGraphInput(graph));
  const targetOutputs = new Set(adapterTargets.map((t) => t.nodeOutput));

  const blocks = [];
  let start = cuts.length ? cuts[0] : null;
  let matched = [];
  let nodeCount = 0;
  const close = (end) => {
    blocks.push({ input: start.name, output: end.name, targetOutputs: matched, nodes: nodeCount });
    start = end;
    matched = [];
    nodeCount = 0;
  };
  for (let i = 1; i < cuts.length; i++) {
    const previous = cuts[i - 1];
    const current = cuts[i];
    const span = graph.nodes.slice(previous.index + 1, current.index + 1);
    if (span.some((node) => !DIFFERENTIABLE_OPS.has(node.opType))) {
      // A gap. Close whatever was pending *before* it (the pending block
      // ends at the last cut still on the trainable side) and reopen after
      // it -- identical to discoverBlocks' own gap handling.
      if (start !== null && matched.length && start.index < previous.index) close(previous);
      start = current;
      matched = [];
      nodeCount = 0;
      continue;
    }
    if (start === null) start = previous;
    for (const node of span) {
      for (const out of node.output) {
        if (out && targetOutputs.has(out)) matched.push(out);
      }
    }
    nodeCount += span.length;
    if (matched.length >= maxTargetsPerBlock) close(current);
  }
  const last = cuts.length ? cuts[cuts.length - 1] : null;
  if (start !== null && matched.length && last && start.index < last.index) close(last);
  return blocks;
}
