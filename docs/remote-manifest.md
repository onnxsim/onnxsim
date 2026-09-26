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

The compiler owns the meaning of `artifact.format` and `artifact.abi`. A
runner must reject an artifact when its compiler ID, target backend/device,
runtime ABI, or required I/O contract does not match. `capabilities.ops` and
`capabilities.dtypes` are advisory until onnxsim is given an explicit
legalization callback; they are intended for a future preflight pass that can
partition or legalize a fold subgraph before `COMPILE`.

The manifest is deliberately versioned independently from the transport. A
new manifest field must not require a wire-protocol version bump, while a
change to tensor or artifact framing does.

`scripts/qualcomm/qnn_compile.py` is the reference QNN adapter. It generates an
embedded ONNX Runtime EP-context artifact and records the QNN runtime version,
backend target, compiler host, and legalization profile in this manifest. It
does not claim operator capabilities until the QNN graph has been compiled;
the empty capability lists are intentional rather than an assertion that QNN
supports no operators.
