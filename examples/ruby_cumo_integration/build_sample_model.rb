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
# Written in ONNX's textual IR syntax (https://onnx.ai/onnx/repo-docs/Syntax.html
# -- the same format onnx.parser.parse_model reads in Python, and the format
# this repo's own tests prefer over onnx.helper.make_node/make_graph/make_model
# chains, per the top-level CLAUDE.md) and parsed via onnxsim's own C API
# (onnxsim_parse_model_text) -- no hand-rolled protobuf encoding, no Ruby
# protobuf gem, no Python.
#
# Usage: ruby build_sample_model.rb [out_path]  (default: sample_model.onnx)

require_relative 'onnxsim_capi'

MODEL_TEXT = <<~ONNX
  <
    ir_version: 8,
    opset_import: ["" : 13]
  >
  ruby_cumo_sample (float[4] x) => (float[4] y)
  <float[4] const_a = {1.0, 2.0, 3.0, 4.0}, float[4] const_b = {10.0, 20.0, 30.0, 30.0}>
  {
    folded_c = Add(const_a, const_b)
    y = Add(x, folded_c)
  }
ONNX

def build_model
  OnnxsimCapi.parse_model_text(MODEL_TEXT)
end

if $PROGRAM_NAME == __FILE__
  out_path = ARGV[0] || File.join(__dir__, 'sample_model.onnx')
  File.binwrite(out_path, build_model)
  puts "wrote #{out_path} (#{File.size(out_path)} bytes)"
end
