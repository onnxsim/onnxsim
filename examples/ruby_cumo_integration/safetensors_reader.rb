# frozen_string_literal: true

require 'json'

# Pure-Ruby reader for the standalone `.safetensors` archives
# `onnxsim_export_safetensors` (onnxsim/tensor_pool_bridge.h's
# SaveModelAsSafetensorsStandalone) produces: an 8-byte little-endian header
# length, a JSON header `{name => {dtype, shape, data_offsets: [begin, end]}}`
# (optionally space-padded for 8-byte alignment), then the raw tensor bytes
# back to back. This is the plain, ecosystem-standard safetensors format
# (https://github.com/huggingface/safetensors#format) -- no onnxsim-specific
# framing -- so no protobuf parsing is needed to read a model's weights back
# out of it; only its embedded "model.onnx" entry (see
# onnxsim/tensor_pool_bridge.h's kEmbeddedModelKey) happens to hold a
# serialized ModelProto, which this reader hands back as opaque bytes rather
# than parsing.
module SafetensorsReader
  Tensor = Struct.new(:name, :dtype, :shape, :bytes)

  module_function

  # Returns a Hash of name => Tensor, read from `path`.
  def read(path)
    File.open(path, 'rb') do |f|
      header_len = f.read(8).unpack1('Q<')
      header = JSON.parse(f.read(header_len))
      data_start = 8 + header_len

      header.each_with_object({}) do |(name, meta), tensors|
        begin_off, end_off = meta.fetch('data_offsets')
        f.seek(data_start + begin_off)
        bytes = f.read(end_off - begin_off) || +''
        tensors[name] = Tensor.new(name, meta.fetch('dtype'), meta.fetch('shape'), bytes)
      end
    end
  end
end
