"""Structural compatibility with Axelera's `Voyager SDK
<https://github.com/axelera-ai-hub/voyager-sdk>`_, which deploys ONNX models
onto Metis AI accelerators via its `deploy.py` tool.

Voyager SDK's own documentation already recommends running `onnxsim.simplify`
as a manual pre-processing step before deployment (see e.g. its
`ax_models/reference/apps/fastsam/deploy_FastSAM.md` CLIP export tutorial,
which calls `onnxsim.simplify` directly). Before deployment, Voyager SDK also
runs its own ONNX graph rewriter (`ax_models/onnx_optimizations.py`) that
recognizes two structural patterns and rewrites them into forms its compiler
tiles better on the AIPU:

- A "Focus" / space-to-depth block: four `Slice` chains that each pick one
  parity quadrant of the spatial dims, concatenated on the channel axis,
  feeding a `Conv` (`detect_focus_conv_pattern`). Common in YOLOX-family
  exports.
- A flattened fully-connected head: `Reshape` from `[N, C, H, W]` to
  `[N, C*H*W]` feeding a `Gemm`/`MatMul` (`detect_gemm_to_conv_pattern`).
  Common in face-recognition heads (ArcFace and similar).

Both detectors work purely structurally: op types, `Slice` start/axis
constants, `Concat` axis, and (for the FC case) recovering the pre-flatten
spatial shape via `onnx.shape_inference` or the FC weight's own shape. This
suite independently re-implements the same structural checks (original code,
not copied from Voyager SDK) and runs them against `onnxsim.simplify`'s
actual output, to confirm onnxsim's fusions do not inadvertently destroy the
shape these detectors look for -- i.e. that `onnxsim.simplify()` can safely
run *before* Voyager SDK's own optimizer.

It also checks that a custom operator living in a private domain (as a
custom decode/post-processing node in a Voyager SDK pipeline would use)
survives `simplify()` unchanged, per the general guarantee documented in the
README's "Custom operators" section.

No part of Voyager SDK is imported or required to run these tests -- the
structural checks below are self-contained so this suite has no dependency
on that (much larger, GStreamer/torch-based) project.
"""

from typing import Dict, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, parser, shape_inference

import onnxsim


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _output_producers(graph) -> Dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output}


def _constant_value(graph, name: str) -> Optional[np.ndarray]:
    """Value of an initializer or `Constant` node output named `name`."""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def _slice_chain_parity_and_root(
    start_node: onnx.NodeProto, graph, producers: Dict[str, onnx.NodeProto]
) -> Optional[Tuple[Tuple[int, int], str]]:
    """Walk a chain of `Slice` nodes back to its root, accumulating the H
    (axis 2) / W (axis 3) start offsets' parity -- the two bits a "Focus"
    space-to-depth quadrant is identified by. Returns
    `((h_parity, w_parity), root_tensor_name)`, or None if the chain doesn't
    statically resolve.
    """
    h_start, w_start = 0, 0
    node = start_node
    while True:
        if node.op_type != "Slice" or len(node.input) < 4:
            return None
        starts = _constant_value(graph, node.input[1])
        axes = _constant_value(graph, node.input[3])
        if starts is None or axes is None:
            return None
        for i, axis in enumerate(axes):
            if axis == 2:
                h_start += int(starts[i])
            elif axis == 3:
                w_start += int(starts[i])
        data_input = node.input[0]
        producer = producers.get(data_input)
        if producer is None or producer.op_type != "Slice":
            return (h_start % 2, w_start % 2), data_input
        node = producer


def focus_conv_pattern_present(model: onnx.ModelProto) -> bool:
    """Whether `model` contains a Focus-style space-to-depth block (four
    parity-quadrant `Slice` chains rooted at one tensor, concatenated, feeding
    a `Conv`) -- the structural shape
    `ax_models.onnx_optimizations.detect_focus_conv_pattern` looks for.
    """
    graph = model.graph
    producers = _output_producers(graph)
    for node in graph.node:
        if node.op_type != "Concat" or len(node.input) != 4:
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), None)
        if axis not in (1, 3):
            continue

        parities: Set[Tuple[int, int]] = set()
        roots: Set[str] = set()
        for inp in node.input:
            producer = producers.get(inp)
            if producer is None or producer.op_type != "Slice":
                parities.clear()
                break
            result = _slice_chain_parity_and_root(producer, graph, producers)
            if result is None:
                parities.clear()
                break
            parity, root = result
            parities.add(parity)
            roots.add(root)

        if len(roots) != 1 or parities != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue

        consumers = [n for n in graph.node if node.output[0] in n.input]
        if any(c.op_type == "Conv" for c in consumers):
            return True
    return False


def reshape_to_fc_pattern_present(model: onnx.ModelProto) -> bool:
    """Whether `model` contains a `Reshape` feeding a `Gemm`/`MatMul` -- the
    structural shape `ax_models.onnx_optimizations.detect_gemm_to_conv_pattern`
    looks for.
    """
    producers = _output_producers(model.graph)
    for node in model.graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        producer = producers.get(node.input[0])
        if producer is not None and producer.op_type == "Reshape":
            return True
    return False


def spatial_dims_via_shape_inference(
    model: onnx.ModelProto, tensor_name: str
) -> Optional[Tuple[int, int, int]]:
    """(C, H, W) for a statically-4D tensor, recovered via `onnx.shape_
    inference` the same way `_infer_spatial_dims_from_shape_inference` does.
    """
    inferred = shape_inference.infer_shapes(model)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        if vi.name == tensor_name:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(dims) == 4 and all(d > 0 for d in dims[1:]):
                return (dims[1], dims[2], dims[3])
    return None


def _focus_conv_model() -> onnx.ModelProto:
    """A minimal YOLOX-style Focus block: four `Slice`-chain quadrants of a
    NCHW input, concatenated on the channel axis, feeding a `Conv`.
    """
    model = _model(
        """
        agraph (float[1,3,8,8] x) => (float[1,4,4,4] y)
        <
          int64[1] s0 = {0},
          int64[1] s1 = {1},
          int64[1] e = {10000},
          int64[1] ax2 = {2},
          int64[1] ax3 = {3},
          int64[1] step = {2}
        >
        {
          h_tl = Slice(x, s0, e, ax2, step)
          tl = Slice(h_tl, s0, e, ax3, step)
          h_bl = Slice(x, s1, e, ax2, step)
          bl = Slice(h_bl, s0, e, ax3, step)
          h_tr = Slice(x, s0, e, ax2, step)
          tr = Slice(h_tr, s1, e, ax3, step)
          h_br = Slice(x, s1, e, ax2, step)
          br = Slice(h_br, s1, e, ax3, step)
          cat = Concat <axis = 1>(tl, bl, tr, br)
          y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(cat, w)
        }
        """
    )
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 12, 3, 3)).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, name="w"))
    return model


def test_focus_conv_pattern_detected_before_and_after_simplify():
    model = _focus_conv_model()
    assert focus_conv_pattern_present(model)
    assert [n.op_type for n in model.graph.node].count("Slice") == 8

    model_simp, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok, "simplified Focus+Conv model failed the equivalence check"

    # onnxsim fuses each two-node H-then-W Slice chain into a single
    # multi-axis Slice (8 Slice nodes -> 4), but the fused quadrant chains
    # still resolve to the same four parities and common root, so Voyager
    # SDK's structural detector still finds the pattern.
    assert [n.op_type for n in model_simp.graph.node].count("Slice") == 4
    assert focus_conv_pattern_present(model_simp)


def _conv_reshape_gemm_model() -> onnx.ModelProto:
    """`Conv -> Reshape([N, C*H*W]) -> Gemm`, the flattened-FC-head shape."""
    model = _model(
        """
        agraph (float[1,3,8,8] x) => (float[1,10] y)
        <
          int64[2] shape = {1, 256}
        >
        {
          conv_out = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w)
          flat = Reshape(conv_out, shape)
          y = Gemm<transB = 1>(flat, fc_w, fc_b)
        }
        """
    )
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    fc_w = rng.standard_normal((10, 256)).astype(np.float32)
    fc_b = rng.standard_normal((10,)).astype(np.float32)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(w, name="w"),
            numpy_helper.from_array(fc_w, name="fc_w"),
            numpy_helper.from_array(fc_b, name="fc_b"),
        ]
    )
    return model


def test_reshape_to_fc_pattern_and_shape_inference_survive_simplify():
    model = _conv_reshape_gemm_model()
    assert reshape_to_fc_pattern_present(model)

    reshape_input = next(n for n in model.graph.node if n.op_type == "Reshape").input[0]
    assert spatial_dims_via_shape_inference(model, reshape_input) == (4, 8, 8)

    model_simp, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok, "simplified Reshape+Gemm model failed the equivalence check"

    assert reshape_to_fc_pattern_present(model_simp)
    reshape_input_simp = next(
        n for n in model_simp.graph.node if n.op_type == "Reshape"
    ).input[0]
    # Voyager SDK's Gemm->Conv detector re-derives the pre-flatten spatial
    # shape via onnx.shape_inference on whatever graph it's handed; onnxsim's
    # own shape inference must leave enough value_info behind for that to
    # still work on the simplified graph.
    assert spatial_dims_via_shape_inference(model_simp, reshape_input_simp) == (
        4,
        8,
        8,
    )


def _model_with_custom_domain_op() -> onnx.ModelProto:
    model = _model(
        """
        agraph (float[1,3,8,8] x) => (float[1,4,6,6] y)
        {
          conv_out = Conv<kernel_shape = [3, 3]>(x, w)
          relu_out = Relu(conv_out)
          y = com.axelera.AxPostProcess(relu_out)
        }
        """,
        opset='13, "com.axelera": 1',
    )
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(w, name="w"))
    return model


def _custom_op_present(model: onnx.ModelProto) -> bool:
    return any(
        n.domain == "com.axelera" and n.op_type == "AxPostProcess"
        for n in model.graph.node
    )


def test_custom_domain_op_survives_simplify_without_registered_schema():
    model = _model_with_custom_domain_op()
    onnx.checker.check_model(model)
    assert _custom_op_present(model)

    model_simp, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok  # vacuously true: check_n=0 skips the numeric check
    assert _custom_op_present(model_simp)
    # The rest of the graph (Conv, Relu) is still simplifiable around it.
    assert [n.op_type for n in model_simp.graph.node] == [
        "Conv",
        "Relu",
        "AxPostProcess",
    ]


def test_custom_domain_op_survives_simplify_with_registered_schema():
    schema = onnx.defs.OpSchema(
        "AxPostProcess",
        "com.axelera",
        1,
        "Example Axelera custom post-processing op.",
        inputs=[onnx.defs.OpSchema.FormalParameter("X", "T")],
        outputs=[onnx.defs.OpSchema.FormalParameter("Y", "T")],
        type_constraints=[("T", ["tensor(float)"], "any float tensor")],
    )
    if not onnx.defs.has("AxPostProcess", domain="com.axelera"):
        onnx.defs.register_schema(schema)

    model = _model_with_custom_domain_op()
    onnx.checker.check_model(model)

    model_simp, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    assert _custom_op_present(model_simp)
