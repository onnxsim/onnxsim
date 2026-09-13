/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises qat_graph_builder.{h,cpp} -- the C++ port of the emitter half of
 * onnxsim/qat_graph.py. Every assertion here is about *parity*: the names, the
 * node order, the operator set and the input/output order this emitter must
 * share with the Python one, because a browser that trains through this file
 * and a host that trains through qat_graph.py must be running the same graph.
 * Numerical behaviour is not tested here; a graph that runs but is a
 * different graph is exactly the failure this file exists to catch.
 *
 * Needs onnx configured (ModelProto, onnx::checker), so this links the
 * fully-configured `onnxsim` CMake target rather than building standalone --
 * mirroring precision_estimator_test.cpp.
 */
#include "qat_graph_builder.h"

#include <onnx/onnx_pb.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "onnx/checker.h"

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckEqual(const std::string& got, const std::string& want,
                const std::string& what) {
  Check(got == want, what + " (got \"" + got + "\", want \"" + want + "\")");
}

void CheckEqual(size_t got, size_t want, const std::string& what) {
  Check(got == want, what + " (got " + std::to_string(got) + ", want " +
                         std::to_string(want) + ")");
}

std::vector<std::string> OpTypes(const GraphBuilder& b) {
  std::vector<std::string> types;
  for (const onnx::NodeProto& node : b.nodes()) types.push_back(node.op_type());
  return types;
}

std::string Joined(const std::vector<std::string>& parts) {
  std::string out;
  for (const std::string& p : parts) {
    if (!out.empty()) out += ",";
    out += p;
  }
  return out;
}

// The four raw_data bytes of a float32, as numpy_helper.from_array writes them.
std::string LittleEndianBytes(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::string out;
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
  return out;
}

// If this fails, the two emitters number their tensors differently and every
// name in every emitted graph diverges -- the parity fixtures would all fail
// at once, and a browser-built step graph would share no tensor name with the
// Python-built one.
void NamesAreThePrefixHintAndAPreIncrementedCounter() {
  GraphBuilder b("g_");
  CheckEqual(b.Name(), "g_t_1",
             "first name uses the default hint and counter 1");
  CheckEqual(b.Name("grad"), "g_grad_2",
             "the hint is spliced between prefix and counter");
  CheckEqual(b.Name(), "g_t_3", "the counter is per-builder, not per-hint");

  GraphBuilder unprefixed;
  CheckEqual(unprefixed.Name("c"), "c_1",
             "an empty prefix contributes nothing and the counter restarts");
}

// If this fails, a wrapper that relies on op()'s default hint (which is all of
// them) names its output after the wrong thing, and every downstream counter
// shifts with it.
void OpDefaultsItsHintToTheLowercasedOpType() {
  GraphBuilder b;
  CheckEqual(b.Op("MatMul", {"a", "b"}), "matmul_1",
             "MatMul's default hint is its lowercased op type");
  CheckEqual(b.Op("ReduceMean", {"x"}), "reducemean_2",
             "a multi-word op type is lowercased whole, not word-wise");
  CheckEqual(b.Op("Gather", {"t", "i"}, "rows"), "rows_3",
             "an explicit hint displaces the op type entirely");
}

// If this fails, Const has stopped matching onnx.numpy_helper.from_array and a
// byte-comparing parity test would reject an initializer that is numerically
// identical -- from_array writes raw_data, never float_data, and gives a numpy
// scalar an empty dims list rather than dims=[1].
void ConstMatchesNumpyHelperFromArrayForFloat32() {
  GraphBuilder b;
  const std::string name = b.Const(0.5f);
  CheckEqual(name, "c_1", "Const's default hint is \"c\"");
  CheckEqual(b.initializer().size(), size_t{1},
             "Const appends one initializer");
  const onnx::TensorProto& scalar = b.initializer()[0];
  CheckEqual(scalar.name(), "c_1",
             "the initializer carries the generated name");
  Check(scalar.data_type() == onnx::TensorProto::FLOAT, "Const is float32");
  CheckEqual(static_cast<size_t>(scalar.dims_size()), size_t{0},
             "a scalar Const has no dims, matching numpy shape ()");
  CheckEqual(scalar.raw_data(), LittleEndianBytes(0.5f),
             "the value lives in little-endian raw_data");
  CheckEqual(static_cast<size_t>(scalar.float_data_size()), size_t{0},
             "from_array never populates float_data, so neither may Const");

  const std::string arr = b.Const({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, "w");
  CheckEqual(arr, "w_2", "the array overload consumes exactly one name");
  const onnx::TensorProto& matrix = b.initializer()[1];
  CheckEqual(static_cast<size_t>(matrix.dims_size()), size_t{2},
             "dims are copied through");
  CheckEqual(matrix.raw_data().size(), size_t{16},
             "four float32 values are sixteen raw bytes");

  const std::string idx = b.ConstInt64({7, 8}, "i");
  CheckEqual(idx, "i_3", "ConstInt64 shares the one counter");
  const onnx::TensorProto& ints = b.initializer()[2];
  Check(ints.data_type() == onnx::TensorProto::INT64, "ConstInt64 is int64");
  CheckEqual(ints.raw_data().size(), size_t{16},
             "two int64 values are sixteen bytes");
}

// If this fails, Clip's two bound constants and the Clip node itself are named
// in the wrong order -- the hazard C++'s unspecified argument evaluation order
// creates where Python's left-to-right order does not.
void ClipNamesBothBoundsBeforeTheClipNode() {
  GraphBuilder b;
  CheckEqual(b.Clip("x", -1.0f, 1.0f), "clip_3",
             "the low bound, the high bound, then the node");
  CheckEqual(Joined(OpTypes(b)), "Clip", "Clip emits exactly one node");
  CheckEqual(b.initializer()[0].raw_data(), LittleEndianBytes(-1.0f),
             "the first initializer is the low bound");
  CheckEqual(b.initializer()[1].raw_data(), LittleEndianBytes(1.0f),
             "the second initializer is the high bound");
  CheckEqual(b.nodes()[0].input(1), "c_1", "Clip's min input is the low bound");
  CheckEqual(b.nodes()[0].input(2), "c_2",
             "Clip's max input is the high bound");
}

// If this fails, a mask has stopped being "const, compare, cast to float" --
// either it names them out of order, or it has started returning a boolean
// tensor, which is the thing EpFriendlyOps exists to keep out of the graph.
void MasksEmitConstThenCompareThenCastToFloat() {
  GraphBuilder greater;
  CheckEqual(greater.GreaterMask("x", 0.0f), "cast_3",
             "threshold constant, Greater, Cast");
  CheckEqual(Joined(OpTypes(greater)), "Greater,Cast",
             "no boolean logic op appears");
  Check(greater.nodes()[1].attribute(0).name() == "to" &&
            greater.nodes()[1].attribute(0).i() == onnx::TensorProto::FLOAT,
        "the mask is cast to float32, not left as bool");

  GraphBuilder less;
  CheckEqual(less.LessMask("x", 0.0f), "cast_3",
             "LessMask numbers identically");
  CheckEqual(Joined(OpTypes(less)), "Less,Cast",
             "LessMask differs only in the comparison");
}

// If this fails, either the composed rounding has drifted from qat_graph.py's
// six-node sequence (shifting every subsequent name), or -- worse -- someone
// has "simplified" it to a Round node, which WebNN cannot run at all and which
// is precisely why this composition exists.
void RoundToNearestComposesRoundingAndNeverEmitsARoundNode() {
  GraphBuilder b;
  CheckEqual(b.RoundToNearest("x"), "mul_7",
             "Abs, the 0.5 constant, Add, Cast, Sign, Cast, Mul");
  CheckEqual(Joined(OpTypes(b)), "Abs,Add,Cast,Sign,Cast,Mul",
             "the composed sequence, in order");
  for (const onnx::NodeProto& node : b.nodes()) {
    Check(node.op_type() != "Round", "no Round node is emitted");
  }
  CheckEqual(b.initializer()[0].raw_data(), LittleEndianBytes(0.5f),
             "the only constant is the half added before truncation");
  Check(b.nodes()[2].attribute(0).i() == onnx::TensorProto::INT32,
        "the truncating cast targets int32, which rounds toward zero");
  Check(b.nodes()[4].attribute(0).i() == onnx::TensorProto::FLOAT,
        "the magnitude is cast back to float before the sign is reapplied");
}

// If this fails, the loss is no longer a rank-0 scalar (keepdims defaults to
// 1), and every caller that reads one float per step reads a tensor instead.
void MeanSquareReducesToARankZeroScalar() {
  GraphBuilder b;
  CheckEqual(b.MeanSquare("d"), "reducemean_2", "Mul then ReduceMean");
  CheckEqual(Joined(OpTypes(b)), "Mul,ReduceMean", "mean(a*a), two nodes");
  Check(b.nodes()[1].attribute(0).name() == "keepdims" &&
            b.nodes()[1].attribute(0).i() == 0,
        "keepdims=0, so the result is a scalar");
  CheckEqual(static_cast<size_t>(b.nodes()[1].attribute_size()), size_t{1},
             "no axes attribute: opset 17's ReduceMean reduces every axis when "
             "omitted");
}

// If this fails, the minibatching primitive has stopped selecting along rows,
// or the explicit-output form has started consuming a name -- which would
// shift every later counter in a graph that uses it, and qat.py's block
// splice uses exactly that form.
void GatherRowsSelectsAlongAxisZeroAndTheIntoFormConsumesNoName() {
  GraphBuilder b;
  CheckEqual(b.GatherRows("table", "idx"), "rows_1",
             "the default hint is \"rows\"");
  Check(b.nodes()[0].attribute(0).name() == "axis" &&
            b.nodes()[0].attribute(0).i() == 0,
        "rows are selected along axis 0");

  b.GatherRowsInto("table", "idx", "block_input");
  CheckEqual(b.nodes()[1].output(0), "block_input",
             "the caller-chosen output name is used verbatim");
  CheckEqual(b.Name(), "t_2",
             "the explicit-output form consumed no counter value");
}

// If this fails, the emitter can produce a graph that the operator set this
// module is built around does not cover -- i.e. a step graph that passes every
// numerical test and still cannot run on WebNN or an NPU EP, which is the one
// thing the whole design is for.
void EveryOpAnyBuilderMethodEmitsIsEpFriendly() {
  GraphBuilder b;
  b.Add("a", "b");
  b.Sub("a", "b");
  b.Mul("a", "b");
  b.Div("a", "b");
  b.MatMul("a", "b");
  b.Transpose("a");
  b.Transpose("a", {1, 0});
  b.Sqrt("a");
  b.Sigmoid("a");
  b.Clip("a", 0.0f, 1.0f);
  b.GreaterMask("a", 0.0f);
  b.LessMask("a", 0.0f);
  b.RoundToNearest("a");
  b.MeanSquare("a");
  b.GatherRows("table", "idx");
  b.GatherRowsInto("table", "idx", "explicit");
  AdamUpdate(b, "p", "g", "m", "v", "lr", "mc", "vc");
  Check(!b.nodes().empty(), "the sweep actually emitted nodes");
  for (const onnx::NodeProto& node : b.nodes()) {
    Check(EpFriendlyOps().count(node.op_type()) == 1,
          node.op_type() +
              " is emitted by a builder method but is not in EpFriendlyOps()");
  }
}

// If this fails, the two sets have drifted and one emitter can build a graph
// the other's tests reject. The membership itself is qat_graph.py's claim; this
// only pins that the C++ copy says the same thing.
void EpFriendlyOpsHasExactlyThePythonSetsMembers() {
  const std::set<std::string> expected = {
      "Abs",    "Add",     "Cast",     "Clip",       "Div",       "Exp",
      "Gather", "Greater", "Identity", "Less",       "MatMul",    "Mul",
      "Neg",    "Pow",     "Reshape",  "ReduceMean", "ReduceSum", "Sigmoid",
      "Sign",   "Sqrt",    "Sub",      "Transpose"};
  CheckEqual(expected.size(), size_t{22}, "the Python set has 22 members");
  Check(EpFriendlyOps() == expected, "EpFriendlyOps() equals EP_FRIENDLY_OPS");
  Check(EpFriendlyOps().count("Round") == 0,
        "Round stays out of EpFriendlyOps");
  Check(EpFriendlyOps().count("Where") == 0, "Where stays out");
}

// If this fails, an Adam step has changed shape -- either its arithmetic or the
// order it is emitted in -- and a ported loop silently trains differently from
// the numpy loops in adaround.py/adaquant.py/brecq.py that it must reproduce.
void AdamUpdateEmitsTheDocumentedArithmeticInTheDocumentedOrder() {
  GraphBuilder b;
  const AdamOutputs out = AdamUpdate(b, "p", "g", "m", "v", "lr", "mc", "vc");

  CheckEqual(Joined(OpTypes(b)),
             "Mul,Mul,Add,Mul,Mul,Mul,Add,Mul,Mul,Mul,Sqrt,Add,Div,Sub",
             "the fourteen nodes of an Adam step, in order");
  CheckEqual(b.initializer().size(), size_t{5},
             "beta1, beta2, 1-beta1, 1-beta2, eps");

  // The four hyper-parameters are named before any node, so the first node is
  // number 5.
  CheckEqual(out.m_next, "add_7", "m' = beta1*m + (1-beta1)*g");
  CheckEqual(out.v_next, "add_11", "v' = beta2*v + (1-beta2)*g*g");
  CheckEqual(out.param_next, "sub_19",
             "p' = p - lr*m_hat / (sqrt(v_hat) + eps)");

  CheckEqual(b.nodes()[0].input(0), "beta1_1",
             "beta1 scales the old first moment");
  CheckEqual(b.nodes()[0].input(1), "m", "...against the incoming m");
  CheckEqual(b.nodes()[4].input(0), "g",
             "the squared gradient is g*g, not Pow");
  CheckEqual(b.nodes()[4].input(1), "g", "...both operands the same tensor");
  CheckEqual(
      b.nodes()[7].input(1), "mc",
      "the first moment is bias-corrected by a fed scalar, not a Pow chain");
  CheckEqual(b.nodes()[8].input(1), "vc", "the second moment likewise");
  CheckEqual(b.nodes()[13].input(0), "p",
             "the update is subtracted from the parameter");

  // The bias-correction *factors* are the host's job, so no step counter, Pow
  // over one, or extra state tensor may appear inside the graph.
  for (const onnx::NodeProto& node : b.nodes()) {
    Check(node.op_type() != "Pow", "no Pow: the bias corrections are fed in");
  }

  // 1 - beta must be the double subtraction narrowed once, as numpy does it;
  // computing it in float32 gives 0x3DCCCCD0 rather than 0x3DCCCCCD.
  CheckEqual(b.initializer()[2].raw_data(),
             LittleEndianBytes(static_cast<float>(1.0 - 0.9)),
             "1-beta1 is float32(0.1), not float32(1.0f-0.9f)");
  CheckEqual(b.initializer()[3].raw_data(),
             LittleEndianBytes(static_cast<float>(1.0 - 0.999)),
             "1-beta2 is float32(0.001), not float32(1.0f-0.999f)");
  CheckEqual(b.initializer()[4].raw_data(), LittleEndianBytes(kAdamEps),
             "the epsilon initializer is the eps argument");
}

// If this fails, an SGD-momentum step has changed shape -- either its
// arithmetic or the order it is emitted in -- and a ported loop silently
// trains differently from qat_graph.py's sgd_momentum_update.
void SgdMomentumUpdateEmitsTheDocumentedArithmeticInTheDocumentedOrder() {
  GraphBuilder b;
  const SgdMomentumOutputs out = SgdMomentumUpdate(b, "p", "g", "mom", "lr");

  CheckEqual(Joined(OpTypes(b)), "Mul,Add,Mul,Sub",
             "the four nodes of an SGD-momentum step, in order");
  CheckEqual(b.initializer().size(), size_t{1}, "just the momentum constant");

  // The momentum constant is named before any node, so the first node is
  // number 2.
  CheckEqual(out.mom_next, "add_3", "mom' = momentum*mom + g");
  CheckEqual(out.param_next, "sub_5", "p' = p - lr*mom'");

  CheckEqual(b.nodes()[0].input(0), "momentum_1",
             "momentum scales the old momentum buffer");
  CheckEqual(b.nodes()[0].input(1), "mom", "...against the incoming mom");
  CheckEqual(b.nodes()[1].input(1), "g",
             "the decayed buffer is added to the raw gradient, unscaled");
  CheckEqual(b.nodes()[3].input(0), "p",
             "the update is subtracted from the parameter");

  // No second moment, no epsilon-guarded denominator: none of Adam's Div/Sqrt
  // appear, and there is exactly one initializer (momentum) rather than
  // Adam's five.
  for (const onnx::NodeProto& node : b.nodes()) {
    Check(node.op_type() != "Div" && node.op_type() != "Sqrt",
          "no Div/Sqrt: SGD-momentum has no second moment to normalize by");
  }

  CheckEqual(b.initializer()[0].raw_data(), LittleEndianBytes(kSgdMomentum),
             "the momentum initializer is the momentum argument");
}

// If this fails, the host half of AdamUpdate's contract disagrees with the
// graph half, and the moments are un-corrected (or over-corrected) at every
// step -- a bias that is largest in exactly the early steps that matter.
void AdamBiasCorrectionsAreTheClosedFormAtStepT() {
  for (int64_t t : {int64_t{0}, int64_t{1}, int64_t{9}, int64_t{99}}) {
    const std::pair<float, float> got = AdamBiasCorrections(t);
    const float want_m = static_cast<float>(
        1.0 / (1.0 - std::pow(0.9, static_cast<double>(t) + 1.0)));
    const float want_v = static_cast<float>(
        1.0 / (1.0 - std::pow(0.999, static_cast<double>(t) + 1.0)));
    // Tolerance rather than equality only because a compiler is free to fold
    // this file's std::pow while the library's stays a call; the claim under
    // test is the formula, not the last ulp.
    Check(std::fabs(got.first - want_m) <= 1e-6f * want_m,
          "m_correction = 1/(1-beta1^(t+1)) at t=" + std::to_string(t));
    Check(std::fabs(got.second - want_v) <= 1e-6f * want_v,
          "v_correction = 1/(1-beta2^(t+1)) at t=" + std::to_string(t));
  }
  // t is 0-based, so the very first step's correction is 1/(1-beta), not 1.
  Check(std::fabs(AdamBiasCorrections(0).first - 10.0f) < 1e-4f,
        "step 0 corrects by 1/(1-0.9) = 10");
}

// If this fails, either the emitted model is not a valid ONNX model at all, or
// its inputs/outputs are declared in an order a positional binding would get
// wrong -- and the runner binds them positionally.
void MakeStepGraphDeclaresInputsAndOutputsInOrderAndPassesTheChecker() {
  GraphBuilder b;
  // A small but complete step: read this step's rows out of a resident table,
  // reduce them to a target, and take one Adam step towards it.
  const std::string rows = b.GatherRows("table", "idx");
  const std::string centre = b.MeanSquare(rows);
  const std::string diff = b.Sub("p", centre);
  const std::string grad = b.Mul(diff, b.Const(2.0f));
  const std::string loss = b.MeanSquare(diff);
  const AdamOutputs adam =
      AdamUpdate(b, "p", grad, "m", "v", "lr", "m_correction", "v_correction");

  StepGraphSpec spec;
  spec.constants = {{"table", {5, 3}}};
  spec.state = {{"p", {3}, adam.param_next},
                {"m", {3}, adam.m_next},
                {"v", {3}, adam.v_next}};
  spec.scalars = {"lr", "m_correction", "v_correction"};
  spec.per_step = {{"idx", {2}, onnx::TensorProto::INT64}};
  spec.loss_output = loss;

  const StepGraph step = MakeStepGraph(b, spec);
  const onnx::GraphProto& graph = step.model.graph();

  CheckEqual(graph.name(), "onnxsim_step", "the default graph name");
  Check(step.model.ir_version() == kStepGraphIrVersion, "ir_version is 8");
  CheckEqual(static_cast<size_t>(step.model.opset_import_size()), size_t{1},
             "exactly one opset import");
  CheckEqual(step.model.opset_import(0).domain(), "", "the default domain");
  Check(step.model.opset_import(0).version() == kStepGraphOpset, "opset 17");
  Check(step.model.producer_name().empty(),
        "no producer_name, matching onnx.helper.make_model's default");

  std::vector<std::string> inputs;
  for (const onnx::ValueInfoProto& vi : graph.input())
    inputs.push_back(vi.name());
  CheckEqual(Joined(inputs), "table,p,m,v,lr,m_correction,v_correction,idx",
             "constants, then state, then scalars, then per-step inputs");

  std::vector<std::string> outputs;
  for (const onnx::ValueInfoProto& vi : graph.output())
    outputs.push_back(vi.name());
  CheckEqual(
      Joined(outputs),
      adam.param_next + "," + adam.m_next + "," + adam.v_next + "," + loss,
      "the state's next values in state order, then the loss");

  // Types and shapes: the scalars are rank 0 with a *present* shape (which is
  // what says "scalar" rather than "rank unknown"), and the index vector keeps
  // the int64 type the Gather needs rather than being cast to float like the
  // scalars are.
  const onnx::TypeProto::Tensor& lr = graph.input(4).type().tensor_type();
  Check(lr.elem_type() == onnx::TensorProto::FLOAT,
        "a scalar input is float32");
  Check(lr.has_shape() && lr.shape().dim_size() == 0,
        "a scalar input is rank 0");
  const onnx::TypeProto::Tensor& idx = graph.input(7).type().tensor_type();
  Check(idx.elem_type() == onnx::TensorProto::INT64,
        "the row index stays int64");
  CheckEqual(static_cast<size_t>(idx.shape().dim_size()), size_t{1},
             "the row index is rank 1");
  const onnx::TypeProto::Tensor& table = graph.input(0).type().tensor_type();
  Check(table.shape().dim(0).dim_value() == 5 &&
            table.shape().dim(1).dim_value() == 3,
        "a constant's declared shape is copied through");

  CheckEqual(static_cast<size_t>(graph.node_size()), b.nodes().size(),
             "every node is carried over");
  CheckEqual(static_cast<size_t>(graph.initializer_size()),
             b.initializer().size(), "every initializer is carried over");

  // The loop-closing map: which output carries the next value of which input.
  CheckEqual(step.state.size(), size_t{3}, "three state tensors");
  CheckEqual(step.state[0].first, "p", "state pairs are in spec order");
  CheckEqual(step.state[0].second, adam.param_next,
             "...mapped to their next value");
  CheckEqual(step.loss_name, loss, "the loss name is reported back");

  try {
    onnx::checker::check_model(step.model);
  } catch (const std::exception& e) {
    Check(false,
          std::string("onnx::checker rejected the step graph: ") + e.what());
  }
}

// If this fails, a caller that asked for no loss gets a graph with a dangling
// output declaration, which the checker rejects outright.
void AStepGraphWithoutALossDeclaresOnlyItsStateOutputs() {
  GraphBuilder b;
  const std::string next = b.Add("p", "p");
  StepGraphSpec spec;
  spec.state = {{"p", {2}, next}};
  spec.graph_name = "no_loss";
  const StepGraph step = MakeStepGraph(b, spec);
  CheckEqual(static_cast<size_t>(step.model.graph().output_size()), size_t{1},
             "only the state's next value is an output");
  Check(step.loss_name.empty(), "no loss is reported");
  CheckEqual(step.model.graph().name(), "no_loss",
             "the graph name is overridable");
  try {
    onnx::checker::check_model(step.model);
  } catch (const std::exception& e) {
    Check(false, std::string("onnx::checker rejected the loss-free graph: ") +
                     e.what());
  }
}

}  // namespace

int main() {
  NamesAreThePrefixHintAndAPreIncrementedCounter();
  OpDefaultsItsHintToTheLowercasedOpType();
  ConstMatchesNumpyHelperFromArrayForFloat32();
  ClipNamesBothBoundsBeforeTheClipNode();
  MasksEmitConstThenCompareThenCastToFloat();
  RoundToNearestComposesRoundingAndNeverEmitsARoundNode();
  MeanSquareReducesToARankZeroScalar();
  GatherRowsSelectsAlongAxisZeroAndTheIntoFormConsumesNoName();
  EveryOpAnyBuilderMethodEmitsIsEpFriendly();
  EpFriendlyOpsHasExactlyThePythonSetsMembers();
  AdamUpdateEmitsTheDocumentedArithmeticInTheDocumentedOrder();
  AdamBiasCorrectionsAreTheClosedFormAtStepT();
  SgdMomentumUpdateEmitsTheDocumentedArithmeticInTheDocumentedOrder();
  MakeStepGraphDeclaresInputsAndOutputsInOrderAndPassesTheChecker();
  AStepGraphWithoutALossDeclaresOnlyItsStateOutputs();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d qat_graph_builder check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all qat_graph_builder tests passed\n");
  return 0;
}
