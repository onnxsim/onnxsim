# frozen_string_literal: true

# End-to-end onnxsim + Ruby + cumo (https://github.com/sonots/cumo) sample.
#
#   1. Build (or reuse) a tiny ONNX model with one foldable Add and one that
#      isn't (see build_sample_model.rb).
#   2. Run the *unsimplified* model through a real ONNX Runtime session (the
#      `onnxruntime` gem -- https://github.com/ankane/onnxruntime-ruby) to get
#      a ground-truth reference output. This ORT is independent of onnxsim's
#      own -- the gem vendors its own prebuilt ONNX Runtime binary, so this
#      step needs no native build at all.
#   3. Simplify the model via onnxsim's C ABI (onnxsim_simplify_path) --
#      constant folding collapses the foldable Add into a new initializer and
#      drops the node.
#   4. Print onnxsim's own before/after op-count report (onnxsim_model_info_diff).
#   5. Export the simplified model to a standalone .safetensors archive
#      (onnxsim_export_safetensors) and read it back with a plain-Ruby
#      reader -- no protobuf parsing needed for the tensor data.
#   6. Load the folded constant into a Cumo::NArray and run the simplified
#      graph's one remaining node (`y = x + folded_c`) on the GPU via cumo.
#   7. Also run the *simplified* model through ONNX Runtime again, and check
#      both that result and cumo's against step 2's reference -- the
#      "embeddability claim" this repo's other backend integrations make
#      (see docs/dlpack-executor.md), now for Ruby + a real ORT session.
#
# Usage: ruby simplify_and_run.rb [model.onnx]
#
# Needs the onnxsim_c shared library built with -DONNXSIM_C_API=ON (see this
# directory's README) discoverable via ONNXSIM_LIB_PATH/ONNXSIM_LIB_DIR, plus
# the `ffi`, `onnxruntime` and `cumo` gems.

require 'tmpdir'

require_relative 'onnxsim_capi'
require_relative 'safetensors_reader'
require_relative 'build_sample_model'
require_relative 'cumo_compat'

require 'onnxruntime'

SAFETENSORS_DTYPE_TO_CUMO = {
  'F32' => Cumo::SFloat,
  'F64' => Cumo::DFloat,
  'I64' => Cumo::Int64,
  'I32' => Cumo::Int32,
  'I16' => Cumo::Int16,
  'I8' => Cumo::Int8,
  'U64' => Cumo::UInt64,
  'U32' => Cumo::UInt32,
  'U16' => Cumo::UInt16,
  'U8' => Cumo::UInt8
}.freeze

def to_cumo_narray(tensor)
  klass = SAFETENSORS_DTYPE_TO_CUMO[tensor.dtype]
  raise "no Cumo::NArray class for safetensors dtype #{tensor.dtype.inspect} " \
        "(tensor #{tensor.name.inspect})" unless klass

  shape = tensor.shape.empty? ? [1] : tensor.shape
  klass.from_binary(tensor.bytes, shape)
end

X_VALUES = [100.0, 200.0, 300.0, 400.0].freeze

Dir.mktmpdir('onnxsim_ruby_cumo') do |tmp|
  in_path = ARGV[0] || File.join(tmp, 'sample_model.onnx')
  File.binwrite(in_path, build_model) unless ARGV[0]

  reference = OnnxRuntime::Model.new(in_path).predict({ x: X_VALUES })['y']
  puts "onnxruntime reference (unsimplified model): #{reference.inspect}"

  out_path = File.join(tmp, 'sample_model.simplified.onnx')
  safetensors_path = File.join(tmp, 'sample_model.simplified.onnx.safetensors')

  puts "\nsimplifying #{in_path} -> #{out_path}"
  OnnxsimCapi.simplify_path(in_path, out_path)

  puts
  puts OnnxsimCapi.model_info_diff(File.binread(in_path), File.binread(out_path))

  ort_simplified = OnnxRuntime::Model.new(out_path).predict({ x: X_VALUES })['y']
  puts "onnxruntime result (simplified model): #{ort_simplified.inspect}"
  raise "onnxruntime mismatch: expected #{reference.inspect}, got #{ort_simplified.inspect}" if ort_simplified != reference

  OnnxsimCapi.export_safetensors(File.binread(out_path), safetensors_path)
  tensors = SafetensorsReader.read(safetensors_path)

  folded = tensors['folded_c']
  unless folded
    raise 'expected the simplified model to carry a folded "folded_c" initializer, ' \
          "found: #{tensors.keys.inspect}"
  end

  puts "folded initializer #{folded.name.inspect}: dtype=#{folded.dtype} shape=#{folded.shape.inspect}"

  folded_c = to_cumo_narray(folded)
  x = Cumo::SFloat.from_binary(X_VALUES.pack('e*'), folded_c.shape)

  y = x + folded_c # the simplified graph's one remaining node, run via cumo
  puts "cumo result (x + folded_c): #{y.to_a.inspect}"
  raise "cumo mismatch: expected #{reference.inspect}, got #{y.to_a.inspect}" if y.to_a != reference

  puts 'OK: onnxsim_c, onnxruntime and cumo all agree'
end
