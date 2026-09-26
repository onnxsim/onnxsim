# Remote compiler manifest

The binary transport treats `Response.manifest` as opaque so constrained
runners do not need a JSON dependency. Compiler services should nevertheless
emit a UTF-8 JSON document with the following stable top-level contract:

```json
{
  "schema_version": 1,
  "compiler": {"name": "qairt", "version": "2.31.0", "id": "..."},
  "target": {"backend": "qnn", "device": "htp", "chip": "..."},
  "artifact": {"format": "axmodel", "abi": "..."},
  "io": {"dtype": "float32", "layout": "...", "dynamic_shapes": false},
  "capabilities": {"ops": ["Conv", "Relu"], "dtypes": ["float32"]},
  "legalization": {"profile": "qnn-htp", "version": 1}
}
```

For TensorRT, the reference adapter uses `artifact.format` equal to
`tensorrt-engine` and includes a `profiling` object with engine build time,
layer count, selected tactics, and TensorRT's detailed layer metadata. This is
compile-time profiling; runtime latency remains a runner-side measurement.

The compiler owns the meaning of `artifact.format` and `artifact.abi`. A
runner must reject an artifact when its compiler ID, target backend/device,
runtime ABI, or required I/O contract does not match. `capabilities.ops` and
`capabilities.dtypes` are advisory until onnxsim is given an explicit
legalization callback. Native remote-executor callers can use that callback to
rewrite a fold subgraph before `COMPILE`, then use `supported_ops` and a
manifest validator to reject anything outside the target contract.

Python-side integrations can use `onnxsim.remote_manifest.preflight_model()`
and `legalize_for_manifest()` for the same policy without depending on a
vendor SDK. Empty capability lists mean "unknown" and therefore do not reject
operators; non-empty lists are enforced. `onnxsim.profile_merge` also exposes
`summarize_remote_events()` for compact compile/load/execute timing reports.

The manifest is deliberately versioned independently from the transport. A
new manifest field must not require a wire-protocol version bump, while a
change to tensor or artifact framing does.

`scripts/qualcomm/qnn_compile.py` is the reference QNN adapter. It generates an
embedded ONNX Runtime EP-context artifact and records the QNN runtime version,
backend target, compiler host, and legalization profile in this manifest. It
does not claim operator capabilities until the QNN graph has been compiled;
the empty capability lists are intentional rather than an assertion that QNN
supports no operators.
