// Runs a knowledge-distillation training step graph from
// ../../scripts/generate_distillation_step_graph.py -- onnxsim's own
// graph_grad/qat_graph autodiff, not onnxruntime.training -- entirely in the
// browser, via the *official* onnxruntime-web package's plain
// ort.InferenceSession.
//
// This deliberately replaces ../src/onnx_finetune_wasm.cpp's whole approach
// for distillation specifically: that file Embind-wraps a custom Emscripten
// build of onnxruntime with --enable_training_apis, needed because it runs
// Ort::TrainingSession/Ort::CheckpointState. A step graph already has the
// forward pass, the KD loss, the backward pass and an Adam update baked in
// as ordinary ONNX nodes (see generate_distillation_step_graph.py's module
// docstring), so running it needs nothing but a plain inference session --
// exactly what onnxruntime-web's stock WASM build already is, with no custom
// compilation, no emsdk, and no --enable_training_apis build at all. It also
// sidesteps wasm/README.md's own "memory access out of bounds" finding: that
// bug reproduced in a training-op kernel this path never touches.
//
// Everything here is plain JS/ort.Tensor construction, one-for-one with
// ../../scripts/generate_distillation_step_graph.py's own
// write_manifest_and_initial_state (the manifest format) and
// src/distill_step_graph_main.cpp (the same loop, in C++) -- the three are
// kept in lockstep by convention, not by shared code, the same relationship
// graph_grad.py/.cpp document for themselves.

// The one shape token generate_distillation_step_graph.py ever writes that
// isn't a decimal integer -- see that script's write_manifest_and_initial_state
// docstring. Only ever the leading (batch) entry of inputShape/
// teacherLogitsShape; state/weight shapes are always fully static.
const DYNAMIC_BATCH = "batch";

function dimToken(tok) {
  return tok === DYNAMIC_BATCH ? tok : Number(tok);
}

/** `tokens` (from `inputShape`/`teacherLogitsShape`) with the `"batch"`
 * sentinel substituted by the batch size this particular step is actually
 * using -- static entries pass through unchanged. */
export function resolveShape(tokens, batchSize) {
  return tokens.map((t) => (t === DYNAMIC_BATCH ? batchSize : t));
}

/** Parses a `<step_graph>.manifest.txt` (see generate_distillation_step_graph.py).
 * `inputShape`/`teacherLogitsShape` entries are numbers, except the batch
 * dimension, which comes back as the literal string `"batch"` -- resolve it
 * per step with `resolveShape`, not by assuming a fixed size. */
export function parseManifest(text) {
  const manifest = { state: [], weights: [] };
  for (const rawLine of text.split("\n")) {
    const parts = rawLine.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) continue;
    const [tag, ...rest] = parts;
    if (tag === "input_name") manifest.inputName = rest[0];
    else if (tag === "input_shape") manifest.inputShape = rest.map(dimToken);
    else if (tag === "teacher_logits_name") manifest.teacherLogitsName = rest[0];
    else if (tag === "teacher_logits_shape") manifest.teacherLogitsShape = rest.map(dimToken);
    else if (tag === "labels_onehot_name") manifest.labelsOnehotName = rest[0];
    else if (tag === "num_classes") manifest.numClasses = Number(rest[0]);
    else if (tag === "loss_name") manifest.lossName = rest[0];
    else if (tag === "state") {
      manifest.state.push({ input: rest[0], output: rest[1], shape: rest.slice(2).map(Number) });
    } else if (tag === "weight") {
      manifest.weights.push({ name: rest[0], shape: rest.slice(1).map(Number) });
    }
  }
  return manifest;
}

/** A float32 one-hot matrix from integer class indices -- built here on the
 * host, not in the graph itself, for the same reason
 * generate_distillation_step_graph.py's own labels_to_onehot is: it keeps
 * Cast/Greater/Less, which have no graph_grad VJP rule, out of the
 * differentiated slice entirely. */
export function labelsToOnehot(labels, numClasses) {
  const rows = labels.length;
  const out = new Float32Array(rows * numClasses);
  for (let i = 0; i < rows; ++i) out[i * numClasses + labels[i]] = 1;
  return out;
}

function prod(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

/** The initial `{state input name: Float32Array}` map -- every weight from
 * `<step_graph>.initial_state.bin`, every Adam __m/__v moment at zero. */
export function loadInitialState(manifest, initialStateBuffer) {
  const state = {};
  const view = new Float32Array(initialStateBuffer);
  let offset = 0;
  for (const { name, shape } of manifest.weights) {
    const count = prod(shape);
    state[name] = view.slice(offset, offset + count);
    offset += count;
  }
  for (const { input, shape } of manifest.state) {
    if (!(input in state)) state[input] = new Float32Array(prod(shape));
  }
  return state;
}

/** Adam's two bias-correction factors at (0-based) step `t`, matching
 * onnxsim.qat_graph.adam_bias_corrections exactly. */
export function adamBiasCorrections(t) {
  return {
    mCorrection: 1 / (1 - Math.pow(0.9, t + 1)),
    vCorrection: 1 / (1 - Math.pow(0.999, t + 1)),
  };
}

export class StepGraphSession {
  static async create(ort, stepGraphBytes) {
    const session = await ort.InferenceSession.create(stepGraphBytes, {
      executionProviders: ["wasm"],
    });
    return new StepGraphSession(ort, session);
  }

  constructor(ort, session) {
    this.ort = ort;
    this.session = session;
  }

  /** Runs one step: `batchInput`/`teacherLogits`/`labels` are this step's
   * batch (Float32Array/Float32Array/Int32Array-or-Array-of-numbers). The
   * batch size is however many `labels` this call passes -- freely
   * choosable step to step, since the step graph's batch dimension is a
   * `dim_param`, decided at `run()` time rather than fixed when the graph
   * was built (see generate_distillation_step_graph.py's module docstring).
   * Returns `{ loss, state }`, `state` ready to feed as next step's own
   * state inputs. */
  async step(manifest, state, batchInput, teacherLogits, labels, lr, t) {
    const { mCorrection, vCorrection } = adamBiasCorrections(t);
    const batchSize = labels.length;
    const onehot = labelsToOnehot(labels, manifest.numClasses);
    const inputShape = resolveShape(manifest.inputShape, batchSize);
    const teacherLogitsShape = resolveShape(manifest.teacherLogitsShape, batchSize);
    const T = this.ort.Tensor;
    const feeds = {
      [manifest.inputName]: new T("float32", batchInput, inputShape),
      [manifest.teacherLogitsName]: new T("float32", teacherLogits, teacherLogitsShape),
      [manifest.labelsOnehotName]: new T("float32", onehot, [batchSize, manifest.numClasses]),
      lr: new T("float32", new Float32Array([lr]), []),
      m_correction: new T("float32", new Float32Array([mCorrection]), []),
      v_correction: new T("float32", new Float32Array([vCorrection]), []),
      batch_size: new T("float32", new Float32Array([batchSize]), []),
    };
    for (const { input, shape } of manifest.state) {
      feeds[input] = new T("float32", state[input], shape);
    }

    const results = await this.session.run(feeds);

    const nextState = {};
    for (const { input, output } of manifest.state) {
      nextState[input] = results[output].data;
    }
    return { loss: results[manifest.lossName].data[0], state: nextState };
  }
}
