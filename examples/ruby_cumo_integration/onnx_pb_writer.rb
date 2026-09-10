# frozen_string_literal: true

# A minimal, dependency-free writer for the handful of ONNX protobuf messages
# this sample needs (ModelProto / GraphProto / NodeProto / TensorProto /
# ValueInfoProto). It exists so the sample can build its own tiny .onnx test
# model without a Ruby protobuf gem or a Python helper -- "only Ruby" all the
# way down to the bytes onnxsim_simplify_path reads.
#
# Protobuf's wire format is simple enough to hand-roll for a fixed, known set
# of fields: a tag (field_number << 3 | wire_type) as a varint, followed by
# the value (another varint, a fixed-width int, or a length-delimited blob).
# Field numbers below are onnx.proto3's stable wire numbers for the messages
# this sample touches -- see https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3.
module OnnxPbWriter
  WIRE_VARINT = 0
  WIRE_LEN = 2

  module_function

  def varint(value)
    bytes = +''
    v = value
    loop do
      byte = v & 0x7F
      v >>= 7
      if v.zero?
        bytes << byte.chr
        break
      else
        bytes << (byte | 0x80).chr
      end
    end
    bytes
  end

  def tag(field_number, wire_type)
    varint((field_number << 3) | wire_type)
  end

  def varint_field(field_number, value)
    tag(field_number, WIRE_VARINT) + varint(value)
  end

  # Covers both TYPE_STRING and TYPE_BYTES/TYPE_MESSAGE fields -- all three
  # are length-delimited on the wire.
  def len_field(field_number, bytes)
    tag(field_number, WIRE_LEN) + varint(bytes.bytesize) + bytes
  end

  def string_field(field_number, str)
    len_field(field_number, str.to_s.dup.force_encoding(Encoding::BINARY))
  end

  # ONNX TensorProto::DataType::FLOAT
  ELEM_TYPE_FLOAT = 1

  # Builds a TensorProto (fields: dims=1, data_type=2, name=8, raw_data=9)
  # holding `values` (an Array of Float) as a 1-D or reshaped float32 tensor.
  def tensor_proto(name, values, shape: [values.size])
    bytes = +''
    shape.each { |d| bytes << varint_field(1, d) }
    bytes << varint_field(2, ELEM_TYPE_FLOAT)
    bytes << string_field(8, name)
    bytes << len_field(9, values.pack('e*')) # float32 little-endian, matches raw_data's layout
    bytes
  end

  # TensorShapeProto::Dimension (field: dim_value=1) wrapped in
  # TensorShapeProto (field: dim=1, repeated).
  def tensor_shape_proto(shape)
    bytes = +''
    shape.each { |d| bytes << len_field(1, varint_field(1, d)) }
    bytes
  end

  # ValueInfoProto (fields: name=1, type=2), where `type` is a TypeProto
  # (field: tensor_type=1) wrapping a TypeProto.Tensor (fields: elem_type=1,
  # shape=2).
  def value_info_proto(name, shape, elem_type: ELEM_TYPE_FLOAT)
    tensor_type = varint_field(1, elem_type) + len_field(2, tensor_shape_proto(shape))
    type_proto = len_field(1, tensor_type)
    string_field(1, name) + len_field(2, type_proto)
  end

  # NodeProto (fields: input=1 repeated, output=2 repeated, name=3,
  # op_type=4).
  def node_proto(op_type:, inputs:, outputs:, name: nil)
    bytes = +''
    inputs.each { |i| bytes << string_field(1, i) }
    outputs.each { |o| bytes << string_field(2, o) }
    bytes << string_field(3, name) if name
    bytes << string_field(4, op_type)
    bytes
  end

  # GraphProto (fields: node=1 repeated, name=2, initializer=5 repeated,
  # input=11 repeated, output=12 repeated).
  def graph_proto(name:, nodes:, initializers:, inputs:, outputs:)
    bytes = +''
    nodes.each { |n| bytes << len_field(1, n) }
    bytes << string_field(2, name)
    initializers.each { |t| bytes << len_field(5, t) }
    inputs.each { |v| bytes << len_field(11, v) }
    outputs.each { |v| bytes << len_field(12, v) }
    bytes
  end

  # OperatorSetIdProto (field: version=2; domain=1 omitted for the default "" domain).
  def opset_id_proto(version)
    varint_field(2, version)
  end

  # ModelProto (fields: ir_version=1, opset_import=8 repeated, graph=7).
  def model_proto(ir_version:, opset_version:, graph:)
    bytes = +''
    bytes << varint_field(1, ir_version)
    bytes << len_field(8, opset_id_proto(opset_version))
    bytes << len_field(7, graph)
    bytes
  end
end
