# frozen_string_literal: true

# Builds a tiny, deliberately unsimplified ONNX model:
#
#   const_a, const_b  --Add-->  folded_c
#   x, folded_c       --Add-->  y
#
# `folded_c` is entirely computable from constants, so onnxsim's constant
# folder should collapse the first Add into a new initializer and drop the
# node, leaving just `y = x + folded_c`. That's the change this sample's
# simplify_and_run.rb demonstrates end to end.
#
# Usage: ruby build_sample_model.rb [out_path]  (default: sample_model.onnx)

require_relative 'onnx_pb_writer'

CONST_A = [1.0, 2.0, 3.0, 4.0].freeze
CONST_B = [10.0, 20.0, 30.0, 30.0].freeze

def build_model
  const_a = OnnxPbWriter.tensor_proto('const_a', CONST_A)
  const_b = OnnxPbWriter.tensor_proto('const_b', CONST_B)

  fold_node = OnnxPbWriter.node_proto(
    op_type: 'Add', inputs: %w[const_a const_b], outputs: ['folded_c'], name: 'add_constants'
  )
  keep_node = OnnxPbWriter.node_proto(
    op_type: 'Add', inputs: %w[x folded_c], outputs: ['y'], name: 'add_x'
  )

  x_input = OnnxPbWriter.value_info_proto('x', [CONST_A.size])
  y_output = OnnxPbWriter.value_info_proto('y', [CONST_A.size])

  graph = OnnxPbWriter.graph_proto(
    name: 'ruby_cumo_sample',
    nodes: [fold_node, keep_node],
    initializers: [const_a, const_b],
    inputs: [x_input],
    outputs: [y_output]
  )

  OnnxPbWriter.model_proto(ir_version: 8, opset_version: 13, graph: graph)
end

if $PROGRAM_NAME == __FILE__
  out_path = ARGV[0] || File.join(__dir__, 'sample_model.onnx')
  File.binwrite(out_path, build_model)
  puts "wrote #{out_path} (#{File.size(out_path)} bytes)"
end
