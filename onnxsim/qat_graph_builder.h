#pragma once

// The step-graph emitter, in C++.
//
// A "step graph" is one optimizer step expressed as a pure ONNX *inference*
// graph -- (constants, state, per-step scalars) -> (next state, loss) -- so a
// training loop is a sequence of ordinary inference calls and runs wherever
// inference runs. onnxsim/qat_graph.py is the original; this is the same
// emitter, reachable from the WASM build so the browser converter can build a
// step graph for a model without a Python round trip.
//
// **Only the emitter is ported.** qat_graph.py's runtime half (run_step_graph,
// the IOBinding loop, minibatch_indices) has no counterpart here on purpose:
// the caller already has an inference runtime -- onnxruntime-web in the
// browser -- and running the emitted graph in a loop is that runtime's job,
// not this file's. What cannot be duplicated is the *graph*, which is why this
// is the half that had to move.
//
// **Drift is the standing risk, and it is a test's job, not a comment's.** Two
// implementations of one emitter that disagree would produce a browser that
// trains differently from the Python for the same model, silently. So the
// contract is that this file and qat_graph.py emit *structurally identical*
// graphs for the same inputs, and qat_graph_parity_test checks exactly that
// against committed fixtures rather than against a re-description of the
// rules. Any change here that a fixture does not cover is a change that ships
// unverified; add the fixture.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

// The operator set a step graph restricts itself to.
//
// This is qat_graph.py's EP_FRIENDLY_OPS, and it must stay byte-identical to
// it -- it is the same claim about the same backends, and a member present on
// one side only would mean one of the two emitters can produce a graph the
// other's tests reject. The parity test compares the two sets directly.
//
// The reasoning behind the membership (control flow, boolean logic, Where,
// Expand and Round deliberately absent; Round because WebNN has no rounding
// operator at all, which is why RoundToNearest composes one; Gather admitted
// on the same coverage criterion rather than for convenience, because it is
// what lets a step read a minibatch out of a resident calibration set; Conv
// and ConvTranspose considered for GradConv and refused, because the coverage
// they have on both browser backends is 2-D only) lives in qat_graph.py next
// to the Python set. It is not repeated here, so that
// there is one place to update when the reasoning changes.
const std::set<std::string>& EpFriendlyOps();

// Adam's standard hyper-parameters, matching qat_graph.py's module constants
// and, through them, the hand-rolled loops in adaround.py/adaquant.py/brecq.py.
inline constexpr float kAdamBeta1 = 0.9f;
inline constexpr float kAdamBeta2 = 0.999f;
inline constexpr float kAdamEps = 1e-8f;

// The opset and IR version every emitted step graph declares. Pinned rather
// than inherited from the model being trained: the emitter's own operators are
// written against these, and a step graph is a new graph rather than an edit
// of an existing one.
inline constexpr int64_t kStepGraphOpset = 17;
inline constexpr int64_t kStepGraphIrVersion = 8;

// Accumulates nodes and initializers with unique names.
//
// Exists for the reason the Python one does: a hand-derived gradient should
// read like the expression it is -- Mul(Sub(y_hat, y), two_over_n) -- rather
// than like a pile of node construction with hand-managed intermediate names.
//
// Names are `<prefix><hint>_<n>` with `n` a per-builder counter, matching
// GraphBuilder.name exactly. That is load-bearing rather than cosmetic: the
// parity test compares emitted graphs including their tensor names, so the
// two emitters must number identically, which means emitting the same
// operations in the same order.
class GraphBuilder {
 public:
  explicit GraphBuilder(std::string prefix = "") : prefix_(std::move(prefix)) {}

  const std::vector<onnx::NodeProto>& nodes() const { return nodes_; }
  std::vector<onnx::NodeProto>& nodes() { return nodes_; }
  const std::vector<onnx::TensorProto>& initializer() const {
    return initializer_;
  }
  std::vector<onnx::TensorProto>& initializer() { return initializer_; }

  // A fresh unique name. `hint` becomes part of it, as in the Python.
  std::string Name(const std::string& hint = "t");

  // A float32 initializer holding a scalar, or an arbitrarily-shaped array.
  std::string Const(float value, const std::string& hint = "c");
  std::string Const(const std::vector<float>& values,
                    const std::vector<int64_t>& dims,
                    const std::string& hint = "c");
  // An int64 initializer -- the shape/axes operand form Reshape and the
  // opset-13 Reduce ops take as a tensor input rather than an attribute.
  std::string ConstInt64(const std::vector<int64_t>& values,
                         const std::string& hint = "i");

  // Appends `op_type(inputs) -> <fresh name>` and returns the output name.
  // Attributes are set by the AddAttr* overloads on the returned node via
  // the Op overload taking a pre-built attribute list.
  std::string Op(const std::string& op_type,
                 const std::vector<std::string>& inputs,
                 const std::string& hint = "");
  std::string Op(const std::string& op_type,
                 const std::vector<std::string>& inputs,
                 const std::vector<onnx::AttributeProto>& attrs,
                 const std::string& hint = "");
  // As above, but writing to a caller-chosen output name -- for the caller
  // whose result must carry a name other nodes already read.
  void OpInto(const std::string& op_type,
              const std::vector<std::string>& inputs, const std::string& output,
              const std::vector<onnx::AttributeProto>& attrs = {});

  // The handful of operators the hand-derived gradients actually use. Kept to
  // ops with broad execution-provider coverage: no boolean logic ops (a mask
  // is a float 0/1 from Cast(Greater), multiplied in) and no Where.
  std::string Add(const std::string& a, const std::string& b);
  std::string Sub(const std::string& a, const std::string& b);
  std::string Mul(const std::string& a, const std::string& b);
  std::string Div(const std::string& a, const std::string& b);
  std::string MatMul(const std::string& a, const std::string& b);
  std::string Transpose(const std::string& a);
  std::string Transpose(const std::string& a, const std::vector<int64_t>& perm);
  std::string Sqrt(const std::string& a);
  std::string Sigmoid(const std::string& a);
  std::string Clip(const std::string& a, float low, float high);

  // `(a > threshold)` / `(a < threshold)` as a float32 0/1 tensor.
  std::string GreaterMask(const std::string& a, float threshold);
  std::string LessMask(const std::string& a, float threshold);

  // round(a), composed rather than emitted as Round.
  //
  // Round is deliberately absent from EpFriendlyOps -- originally because
  // WebNN had no rounding operator at all, which has since stopped being
  // true (see qat_graph.py's Conv note); the composition is kept because it
  // is verified, not because that argument still holds -- and a fake-quant
  // forward, which is what every caller wants this for, is exactly the code
  // that must run on those backends. A float-to-int32 Cast truncates toward
  // zero, so truncating |a| + 0.5 and re-applying the sign is
  // round-half-away-from-zero. That differs from Round's (and numpy's)
  // round-half-to-even on *exact* ties only.
  std::string RoundToNearest(const std::string& a);

  // mean(a * a) as a scalar, for a reported loss.
  std::string MeanSquare(const std::string& a);

  // Rows `index` of `table` along axis 0 -- the minibatching primitive. The
  // table is bound once as a step-graph constant and so stays resident on the
  // device; the index is a rank-1 int64 per-step input. What crosses the bus
  // per step is batch_size 8-byte integers rather than batch_size x width
  // floats.
  std::string GatherRows(const std::string& table, const std::string& index);
  void GatherRowsInto(const std::string& table, const std::string& index,
                      const std::string& output);

 private:
  std::vector<onnx::NodeProto> nodes_;
  std::vector<onnx::TensorProto> initializer_;
  std::string prefix_;
  int64_t counter_ = 0;
};

// Appends one Adam step to `b` and returns (param', m', v').
//
// `m_correction`/`v_correction` are the bias-correction *factors*
// 1 / (1 - beta^t), passed in as scalars rather than derived from a step
// counter inside the graph: they are two host-side floats per step, so
// computing them outside costs nothing and keeps the graph free of the state a
// Pow over a step counter would need.
struct AdamOutputs {
  std::string param_next;
  std::string m_next;
  std::string v_next;
};
AdamOutputs AdamUpdate(GraphBuilder& b, const std::string& param,
                       const std::string& grad, const std::string& m,
                       const std::string& v, const std::string& lr,
                       const std::string& m_correction,
                       const std::string& v_correction, float eps = kAdamEps);

// The two bias-correction factors for step `t` (0-based), the host-side half
// of AdamUpdate's contract.
std::pair<float, float> AdamBiasCorrections(int64_t t);

// A pure function performing one optimizer step.
//
// `state` pairs each state input with the output that carries its next value,
// which is what makes the loop a ping-pong of two buffers rather than a
// rebuild per step. `per_step` names the inputs whose value changes every step
// (the scalars, and a minibatch index vector when there is one).
struct StepGraphSpec {
  // Float32 graph inputs whose value does not change across steps
  // (calibration activations, the reconstruction target, a frozen scale).
  // They are *inputs* rather than initializers because the runner binds them
  // once and keeps them resident; the emitter only declares their shape.
  struct NamedShape {
    std::string name;
    std::vector<int64_t> dims;
  };
  std::vector<NamedShape> constants;
  // (input name, dims, output name carrying the next value).
  struct StateEntry {
    std::string input;
    std::vector<int64_t> dims;
    std::string next_output;
  };
  std::vector<StateEntry> state;
  // Scalar float inputs supplied fresh each step (learning rates, the Adam
  // bias corrections).
  std::vector<std::string> scalars;
  // Non-scalar per-step inputs, with dims and element type -- today only a
  // minibatch row index (rank-1 int64).
  struct PerStepEntry {
    std::string name;
    std::vector<int64_t> dims;
    int32_t elem_type;
  };
  std::vector<PerStepEntry> per_step;
  // The scalar the loop reports as this step's loss, if any. Empty for none.
  // Nothing in the loop needs it; it is for diagnostics.
  std::string loss_output;
  // The emitted graph's name, matching make_step_graph's default.
  std::string graph_name = "onnxsim_step";
};

// The emitted graph together with the loop-closing map the runner needs:
// which output carries the next value of which state input.
struct StepGraph {
  onnx::ModelProto model;
  std::vector<std::pair<std::string, std::string>> state;  // input -> next
  std::string loss_name;
};

// Closes `b` into a runnable model: every state input declared, every `next`
// output declared, inputs typed and shaped.
StepGraph MakeStepGraph(const GraphBuilder& b, const StepGraphSpec& spec);
