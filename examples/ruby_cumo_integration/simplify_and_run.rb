# frozen_string_literal: true

# End-to-end onnxsim + Ruby + cumo (https://github.com/sonots/cumo) sample.
#
#   1. Build (or reuse) a tiny ONNX model with one foldable Add and one that
#      isn't (see build_sample_model.rb).
#   2. Simplify it via onnxsim's C ABI (onnxsim_simplify_path) -- constant
#      folding collapses the foldable Add into a new initializer and drops
#      the node.
#   3. Print onnxsim's own before/after op-count report (onnxsim_model_info_diff).
#   4. Export the simplified model to a standalone .safetensors archive
#      (onnxsim_export_safetensors) and read it back with a plain-Ruby
#      reader -- no protobuf parsing needed for the tensor data.
#   5. Load the folded constant into a Cumo::NArray and run the remaining
#      graph (`y = x + folded_c`) on the GPU via cumo, checking the result
#      against the value the *unsimplified* graph would have produced.
#
# Usage: ruby simplify_and_run.rb [model.onnx]
#
# Needs the onnxsim_c shared library built with -DONNXSIM_C_API=ON (see this
# directory's README) discoverable via ONNXSIM_LIB_PATH/ONNXSIM_LIB_DIR, plus
# the `ffi` and `cumo` gems.

require 'tmpdir'

require_relative 'onnx_pb_writer'
require_relative 'onnxsim_capi'
require_relative 'safetensors_reader'
require_relative 'build_sample_model'

begin
  require 'cumo/narray'
rescue LoadError, RuntimeError => e
  # LoadError: the gem isn't installed. RuntimeError (or a subclass): the gem
  # is installed but its native extension couldn't find a CUDA-capable GPU at
  # require time (e.g. "CUDA driver version is insufficient").
  warn "cumo is not available (#{e.message}); install it on a CUDA-capable " \
       'machine -- see this directory\'s README. Falling back to Numo::NArray ' \
       '(cumo\'s CPU-only, API-compatible counterpart) so the rest of the ' \
       'pipeline can still be exercised.'
  require 'numo/narray'
  Cumo = Numo unless defined?(Cumo)
end

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

Dir.mktmpdir('onnxsim_ruby_cumo') do |tmp|
  in_path = ARGV[0] || File.join(tmp, 'sample_model.onnx')
  File.binwrite(in_path, build_model) unless ARGV[0]

  out_path = File.join(tmp, 'sample_model.simplified.onnx')
  safetensors_path = File.join(tmp, 'sample_model.simplified.onnx.safetensors')

  puts "simplifying #{in_path} -> #{out_path}"
  OnnxsimCapi.simplify_path(in_path, out_path)

  puts
  puts OnnxsimCapi.model_info_diff(File.binread(in_path), File.binread(out_path))

  OnnxsimCapi.export_safetensors(File.binread(out_path), safetensors_path)
  tensors = SafetensorsReader.read(safetensors_path)

  folded = tensors['folded_c']
  unless folded
    raise 'expected the simplified model to carry a folded "folded_c" initializer, ' \
          "found: #{tensors.keys.inspect}"
  end

  puts "folded initializer #{folded.name.inspect}: dtype=#{folded.dtype} shape=#{folded.shape.inspect}"

  folded_c = to_cumo_narray(folded)
  x = Cumo::SFloat.from_binary([100.0, 200.0, 300.0, 400.0].pack('e*'), folded_c.shape)

  y = x + folded_c # the simplified graph's one remaining node, run via cumo
  puts "cumo result (x + folded_c): #{y.to_a.inspect}"

  expected = [111.0, 222.0, 333.0, 434.0]
  if y.to_a == expected
    puts 'OK: matches the unsimplified graph\'s reference output'
  else
    raise "mismatch: expected #{expected.inspect}, got #{y.to_a.inspect}"
  end
end
