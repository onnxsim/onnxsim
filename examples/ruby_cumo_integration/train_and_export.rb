# frozen_string_literal: true

# Trains a tiny linear regression (y = w*x + b) by hand-rolled gradient
# descent, then bakes the learned weights into a real ONNX model and
# simplifies it -- cumo (https://github.com/sonots/cumo) doing the training
# math, onnxsim turning the result into a deployable artifact.
#
#   1. Run gradient descent on Cumo::NArray: forward pass, MSE loss gradient,
#      weight update, all as a handful of elementwise ops -- exactly cumo's
#      job (falls back to Numo::NArray with no GPU; see cumo_compat.rb).
#   2. Interpolate the learned w/b into ONNX's textual IR syntax and parse it
#      via onnxsim's C API (onnxsim_parse_model_text -- see
#      build_sample_model.rb for why that's preferable to hand-encoding
#      protobuf bytes) into a ModelProto.
#   3. Simplify it (onnxsim_simplify_path) and print the before/after report
#      (onnxsim_model_info_diff).
#   4. Run the simplified, trained model through a real ONNX Runtime session
#      (the `onnxruntime` gem) and check it against cumo's own forward pass
#      with the same learned weights -- the same cross-engine check
#      simplify_and_run.rb makes, now on a model this script trained itself
#      rather than one built by hand.
#
# This is NOT training an existing ONNX graph in place (onnxsim has no
# autodiff/training-graph feature of its own to drive from Ruby) -- the
# training loop here is plain cumo code; onnxsim only enters once there is a
# model to simplify and export.
#
# Usage: ruby train_and_export.rb [out_path]  (default: trained_linear.onnx)

require 'tmpdir'

require_relative 'onnxsim_capi'
require_relative 'cumo_compat'

require 'onnxruntime'

# cumo reductions (.mean, .sum, ...) return a 0-D Cumo::NArray rather than a
# plain Ruby Float (unlike Numo's, which already return one) -- normalize
# either to a Float so the training loop works the same under both.
def scalar(value)
  return value.to_f if value.is_a?(Numeric)

  value.to_a.flatten.first.to_f
end

TRUE_W = 3.0
TRUE_B = 2.0
X_VALUES = (1..20).map(&:to_f).freeze
LEARNING_RATE = 0.0005
EPOCHS = 40_000
CONVERGENCE_TOLERANCE = 0.01

x = Cumo::SFloat.from_binary(X_VALUES.pack('e*'), [X_VALUES.size])
y_true = (x * TRUE_W) + TRUE_B # synthetic, noise-free training data

w = 0.0
b = 0.0

EPOCHS.times do |epoch|
  y_pred = (x * w) + b
  error = y_pred - y_true
  dw = scalar((error * x * 2).mean)
  db = scalar((error * 2).mean)
  w -= LEARNING_RATE * dw
  b -= LEARNING_RATE * db

  next unless (epoch % 10_000).zero?

  loss = scalar((error**2).mean)
  puts "epoch #{epoch}: loss=#{loss.round(6)} w=#{w.round(4)} b=#{b.round(4)}"
end

puts "trained on cumo: w=#{w.round(6)} (true #{TRUE_W}), b=#{b.round(6)} (true #{TRUE_B})"
if (w - TRUE_W).abs > CONVERGENCE_TOLERANCE || (b - TRUE_B).abs > CONVERGENCE_TOLERANCE
  raise "training did not converge within #{CONVERGENCE_TOLERANCE} of the true w/b " \
        "(got w=#{w}, b=#{b}) -- try more EPOCHS or a different LEARNING_RATE"
end

model_text = <<~ONNX
  <
    ir_version: 8,
    opset_import: ["" : 13]
  >
  trained_linear (float[#{X_VALUES.size}] x) => (float[#{X_VALUES.size}] y)
  <float[1] w = {#{w}}, float[1] b = {#{b}}>
  {
    wx = Mul(x, w)
    y = Add(wx, b)
  }
ONNX

out_path = ARGV[0] || File.join(__dir__, 'trained_linear.onnx')

Dir.mktmpdir('onnxsim_ruby_train') do |tmp|
  in_path = File.join(tmp, 'trained_linear.onnx')
  File.binwrite(in_path, OnnxsimCapi.parse_model_text(model_text))

  OnnxsimCapi.simplify_path(in_path, out_path)

  puts
  puts OnnxsimCapi.model_info_diff(File.binread(in_path), File.binread(out_path))

  ort_pred = OnnxRuntime::Model.new(out_path).predict({ x: X_VALUES })['y']
  cumo_pred = ((x * w) + b).to_a

  puts "onnxruntime prediction (simplified, trained model): #{ort_pred.map { |v| v.round(4) }.inspect}"
  puts "cumo prediction:                                     #{cumo_pred.map { |v| v.round(4) }.inspect}"

  max_diff = ort_pred.zip(cumo_pred).map { |a, c| (a - c).abs }.max
  # A learned (not exactly-representable) weight, unlike simplify_and_run.rb's
  # fixed constants -- ORT's and cumo's float32 Mul+Add can differ by a ULP
  # or two, so this needs a tolerance rather than exact equality.
  raise "onnxruntime/cumo predictions diverge by #{max_diff}" if max_diff > 1e-3

  puts "OK: onnxruntime and cumo agree on the trained model (max diff #{max_diff})"
end

puts "wrote #{out_path} (#{File.size(out_path)} bytes)"
