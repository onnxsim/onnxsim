# frozen_string_literal: true

require 'ffi'

# Thin FFI binding to onnxsim's C ABI (onnxsim/capi/onnxsim_c_api.h) -- the
# same seam the Rust crates (rust/onnxsim-sys) bind, just from Ruby. Only the
# entry points this sample actually calls are declared; see the header for
# the full API (custom rewriters, GGUF, the executor-callback seam, ...).
module OnnxsimCapi
  extend FFI::Library

  # The shared library is not on a Ruby-known search path by default -- point
  # ONNXSIM_LIB_PATH at it (a full path to libonnxsim_c.<so|dylib|dll>) or
  # ONNXSIM_LIB_DIR at the directory holding it (same variable name the Rust
  # bindings use for a pre-built library; see rust/README.md "Building the
  # native library"). Falls back to the bare library name, which works if the
  # dynamic loader can already find it (e.g. via LD_LIBRARY_PATH).
  def self.locate_library
    return ENV['ONNXSIM_LIB_PATH'] if ENV['ONNXSIM_LIB_PATH']

    if (dir = ENV['ONNXSIM_LIB_DIR'])
      dir.split(File::PATH_SEPARATOR).each do |d|
        %w[libonnxsim_c.so libonnxsim_c.dylib onnxsim_c.dll].each do |basename|
          candidate = File.join(d, basename)
          return candidate if File.exist?(candidate)
        end
      end
    end

    'onnxsim_c'
  end

  ffi_lib locate_library

  # OnnxsimStatus
  ONNXSIM_OK = 0
  ONNXSIM_ERROR = 1

  attach_function :onnxsim_simplify_path, [
    :string, :string,               # in_path, out_path
    :pointer, :size_t,               # skip_optimizers, num_skip_optimizers
    :int, :int, :int,                # skip_optimizers_is_null, constant_folding, shape_inference
    :size_t, :int,                   # tensor_size_threshold, target_opset_version
    :pointer, :size_t,               # extra_optimizers, num_extra_optimizers
    :pointer, :pointer, :pointer,    # rewrite_fn, rewrite_free_fn, rewrite_user_data
    :pointer                         # out_error
  ], :int

  attach_function :onnxsim_export_safetensors, [
    :pointer, :size_t, :string, :pointer
  ], :int

  attach_function :onnxsim_model_info_diff, [
    :pointer, :size_t, :pointer, :size_t, :pointer, :pointer
  ], :int

  attach_function :onnxsim_parse_model_text, [
    :string, :pointer, :pointer, :pointer
  ], :int

  attach_function :onnxsim_free_buffer, [:pointer], :void
  attach_function :onnxsim_free_string, [:pointer], :void

  # 1.5GB, onnxsim's own default (DEFAULT_TENSOR_SIZE_THRESHOLDHOLD in
  # onnxsim/onnx_simplifier.py) -- the byte-size ceiling for tensors produced
  # by constant folding that are kept as initializers.
  DEFAULT_TENSOR_SIZE_THRESHOLD = 1_610_612_736

  # Reads and frees a `char**` out_error/out_text pointer; nil if it's NULL.
  def self.take_string(ptr_ptr)
    ptr = ptr_ptr.read_pointer
    return nil if ptr.null?

    str = ptr.read_string
    onnxsim_free_string(ptr)
    str
  end

  # Reads and frees a `void** out_data` / `size_t* out_size` pair as a
  # binary Ruby String; nil if out_data is NULL.
  def self.take_buffer(out_data_ptr, out_size_ptr)
    ptr = out_data_ptr.read_pointer
    return nil if ptr.null?

    bytes = ptr.read_bytes(out_size_ptr.read_ulong)
    onnxsim_free_buffer(ptr)
    bytes
  end

  # Parses ONNX's textual IR syntax (the same format onnx.parser.parse_model
  # reads in Python -- see onnxsim_parse_model_text's doc comment in
  # onnxsim_c_api.h) into a serialized ModelProto, as a binary Ruby String.
  # Raises RuntimeError with the parser's error message on a syntax error.
  def self.parse_model_text(text)
    out_data = FFI::MemoryPointer.new(:pointer)
    out_size = FFI::MemoryPointer.new(:size_t)
    out_error = FFI::MemoryPointer.new(:pointer)

    status = onnxsim_parse_model_text(text, out_data, out_size, out_error)
    error = take_string(out_error)
    raise "onnxsim_parse_model_text failed: #{error}" if status != ONNXSIM_OK

    take_buffer(out_data, out_size)
  end

  # Simplifies `in_path` into `out_path` with onnxsim's defaults (constant
  # folding + shape inference on, every optimizer pass, no target opset
  # conversion). Raises RuntimeError with the C error message on failure.
  def self.simplify_path(in_path, out_path, tensor_size_threshold: DEFAULT_TENSOR_SIZE_THRESHOLD)
    out_error = FFI::MemoryPointer.new(:pointer)
    status = onnxsim_simplify_path(
      in_path, out_path,
      nil, 0,                 # skip_optimizers / num_skip_optimizers
      0, 1, 1,                 # skip_optimizers_is_null=false, constant_folding, shape_inference
      tensor_size_threshold, 0, # tensor_size_threshold, target_opset_version (0 = unchanged)
      nil, 0,                  # extra_optimizers / num_extra_optimizers
      nil, nil, nil,            # rewrite_fn / rewrite_free_fn / rewrite_user_data
      out_error
    )
    error = take_string(out_error)
    raise "onnxsim_simplify_path failed: #{error}" if status != ONNXSIM_OK

    out_path
  end

  # Exports `model_bytes` (a serialized ModelProto) to a standalone
  # .safetensors archive at `out_path` (see onnxsim_export_safetensors's
  # doc comment in onnxsim_c_api.h). Raises RuntimeError on failure.
  def self.export_safetensors(model_bytes, out_path)
    model_ptr = FFI::MemoryPointer.new(:uint8, model_bytes.bytesize)
    model_ptr.put_bytes(0, model_bytes)
    out_error = FFI::MemoryPointer.new(:pointer)

    status = onnxsim_export_safetensors(model_ptr, model_bytes.bytesize, out_path, out_error)
    error = take_string(out_error)
    raise "onnxsim_export_safetensors failed: #{error}" if status != ONNXSIM_OK

    out_path
  end

  # Renders onnxsim's op-count/size "the difference" ASCII table between two
  # serialized ModelProtos -- the same report the Python CLI prints after
  # simplifying. Raises RuntimeError on failure.
  def self.model_info_diff(original_bytes, simplified_bytes)
    original_ptr = FFI::MemoryPointer.new(:uint8, original_bytes.bytesize)
    original_ptr.put_bytes(0, original_bytes)
    simplified_ptr = FFI::MemoryPointer.new(:uint8, simplified_bytes.bytesize)
    simplified_ptr.put_bytes(0, simplified_bytes)

    out_text = FFI::MemoryPointer.new(:pointer)
    out_error = FFI::MemoryPointer.new(:pointer)
    status = onnxsim_model_info_diff(
      original_ptr, original_bytes.bytesize,
      simplified_ptr, simplified_bytes.bytesize,
      out_text, out_error
    )
    if status != ONNXSIM_OK
      error = take_string(out_error)
      raise "onnxsim_model_info_diff failed: #{error}"
    end
    take_string(out_text)
  end
end
