"""Generate the ONNX artifacts ``index.html`` in this directory trains in the
browser via ONNX Runtime Web's on-device training API.

This is an offline preprocessing step, run once with Python (needs the heavy
``torch``/``transformers``/``onnxruntime-training`` packages -- see below) to
produce small, self-contained files that ``index.html`` then loads and trains
entirely client-side, with no server and no Python at request time:

- ``assets/teacher_model.onnx`` -- a plain inference-only export of the
  (frozen) teacher, run via a normal ``ort.InferenceSession`` in the browser.
- ``assets/checkpoint``, ``assets/training_model.onnx``,
  ``assets/optimizer_model.onnx`` -- the student's ONNX Runtime *training*
  artifacts (forward+loss graph, AdamW optimizer graph, and initial
  parameter checkpoint), produced by
  ``onnxruntime.training.artifacts.generate_artifacts()`` and driven in the
  browser via ``ort.TrainingSession``.

Both models use the same tiny architecture as ``examples/llm_distillation/
distill.py``'s ``TINY_SPEC`` (a real, if microscopic, Llama-shaped causal LM:
attention, RoPE, RMSNorm, SwiGLU MLP) -- kept separate from that file's
constants rather than importing them, since this script needs its own
export/training-specific wiring (``use_cache=False``, an explicit
all-ones ``attention_mask`` to dodge a masking-utils code path that isn't
exportable -- see ``_export_to_onnx`` below -- and the custom distillation
loss graph ``DistillationLoss`` adds). ~22.5K parameters at these
dimensions -- not the ~1B/~162M scale the main (PyTorch, native-Python) demo
trains at, which is many times over what fits in WASM32's linear memory,
per this repo's ``tools/onnx-finetune/wasm/README.md``.

One further simplification versus ``distill.py``'s real next-token loss:
the hard-label loss here is **position-aligned** (student's prediction at
position *i* is scored against ``labels[i]``, not a shifted
``input_ids[i+1]``) rather than a true autoregressive next-token loss --
avoids a dynamic Slice in the loss graph for a case that's purely about
exercising the training *mechanics* in-browser, not about training a good
language model. Document any reuse of this loss graph elsewhere
accordingly.

Needs (not part of onnxsim's own dependencies -- this whole example is
standalone, see its own README):

    pip install torch transformers onnxruntime-training

Regenerate with::

    python examples/llm_distillation/wasm_demo/generate_web_artifacts.py
"""

import copy
import os

import onnx
from onnx import TensorProto, helper

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

VOCAB_SIZE = 64
MAX_POSITION_EMBEDDINGS = 32
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1

# onnxruntime-training's C++ runtime (as bundled by the ONNX Runtime Web
# training build these artifacts target) rejects any IR version newer than
# this -- see this script's own docstring note below on why every artifact
# gets its ir_version clamped after generation.
MAX_SUPPORTED_IR_VERSION = 10


def _build_causal_lm():
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        use_cache=False,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _export_to_onnx(model, path):
    """Export ``model`` (a ``LlamaForCausalLM``) to a plain ONNX graph made
    entirely of primitive ops -- no fused ``com.microsoft`` kernels, no
    ``DynamicCache`` in the traced output -- so both plain ``ort.
    InferenceSession`` (teacher) and ONNX Runtime's training gradient
    builder (student) can consume it.

    Two real transformers/torch.onnx quirks this works around, found by
    trial and error against transformers' current ``masking_utils``/
    ``torch.export`` behavior (see this repo's PR discussion for the
    original repro):

    - The new (``torch.export``-based) exporter chokes on
      ``LlamaForCausalLM``'s output containing a ``DynamicCache`` object
      even with ``use_cache=False`` requested at call time, so this uses
      the legacy TorchScript-based exporter (``dynamo=False``) instead.
    - That legacy exporter can't trace ``aten::diff``, which transformers'
      ``masking_utils.find_packed_sequence_indices`` uses to detect packed
      sequences -- but only on the path taken when ``attention_mask is
      None``. Passing an explicit all-ones ``attention_mask`` (as any real
      training loop would anyway) skips that path entirely.
    """
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids, attention_mask=attention_mask, use_cache=False).logits

    wrapped = Wrapper(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8), dtype=torch.int64)
    attention_mask = torch.ones((1, 8), dtype=torch.int64)

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )


def _make_distillation_loss_class():
    """Returns a ``DistillationLoss`` class subclassing ``onnxblock.blocks.
    Block`` -- built lazily inside a function (rather than at module import
    time) so importing this module doesn't require ``onnxruntime-training``
    to be installed just to read/test the export helpers above.
    """
    from onnxruntime.training.onnxblock import blocks

    class DistillationLoss(blocks.Block):
        """Custom ``onnxblock`` loss: ``alpha`` * soft-target cross-entropy
        (temperature-scaled, per Hinton et al.) + ``(1 - alpha)`` *
        hard-label cross-entropy, both position-aligned (see module
        docstring).

        Built from raw ONNX nodes rather than composing the built-in
        ``onnxblock.loss.CrossEntropyLoss`` convenience block: that block's
        labels-input creation requires its score input to already be a
        registered graph *output*, which the soft-loss branch's
        intermediate tensors are not.
        """

        def __init__(self, temperature=2.0, alpha=0.5):
            super().__init__()
            self._t = temperature
            self._alpha = alpha
            self._counter = 0

        def _name(self, suffix):
            self._counter += 1
            return f"kd_loss/{suffix}_{self._counter}"

        def _node(self, op_type, inputs, **attrs):
            out = self._name(f"{op_type.lower()}_out")
            self.base.graph.node.append(
                helper.make_node(op_type, inputs, [out], name=self._name(op_type), **attrs)
            )
            return out

        def _const_scalar(self, value, dtype=TensorProto.FLOAT):
            name = self._name("const")
            self.base.graph.initializer.append(helper.make_tensor(name, dtype, [], [value]))
            return name

        def build(self, logits_name, teacher_logits_name="teacher_logits", labels_name="labels"):
            import onnxruntime.training.onnxblock._graph_utils as _graph_utils

            g = self.base.graph
            logits_vi = _graph_utils.get_output_from_output_name(self.base, logits_name)

            if not _graph_utils.node_arg_exists(self.base, teacher_logits_name):
                teacher_vi = copy.deepcopy(logits_vi)
                teacher_vi.name = teacher_logits_name
                g.input.append(teacher_vi)

            if not _graph_utils.node_arg_exists(self.base, labels_name):
                labels_vi = copy.deepcopy(logits_vi)
                labels_vi.name = labels_name
                labels_vi.type.tensor_type.elem_type = TensorProto.INT64
                del labels_vi.type.tensor_type.shape.dim[-1]
                g.input.append(labels_vi)

            # --- soft loss: -mean(softmax(teacher/T) * log_softmax(student/T)) * T^2 ---
            t_const = self._const_scalar(self._t)
            student_scaled = self._node("Div", [logits_name, t_const])
            teacher_scaled = self._node("Div", [teacher_logits_name, t_const])
            student_log_probs = self._node("LogSoftmax", [student_scaled], axis=-1)
            teacher_probs = self._node("Softmax", [teacher_scaled], axis=-1)
            per_token = self._node("Mul", [teacher_probs, student_log_probs])
            axes_const = self._name("axes_const")
            g.initializer.append(helper.make_tensor(axes_const, TensorProto.INT64, [1], [-1]))
            per_position = self._node("ReduceSum", [per_token, axes_const], keepdims=0)
            soft_loss = self._node("Neg", [self._node("ReduceMean", [per_position], keepdims=0)])
            t_sq_const = self._const_scalar(self._t * self._t)
            soft_loss = self._node("Mul", [soft_loss, t_sq_const])

            # --- hard loss: SoftmaxCrossEntropyLoss wants class dim at axis 1 ---
            transposed = self._node("Transpose", [logits_name], perm=[0, 2, 1])
            hard_loss_out = self._name("hard_loss")
            log_prob_out = self._name("log_prob")
            g.node.append(
                helper.make_node(
                    "SoftmaxCrossEntropyLoss",
                    [transposed, labels_name],
                    [hard_loss_out, log_prob_out],
                    reduction="mean",
                    name=self._name("SoftmaxCrossEntropyLoss"),
                )
            )

            alpha_const = self._const_scalar(self._alpha)
            one_minus_alpha_const = self._const_scalar(1.0 - self._alpha)
            weighted_soft = self._node("Mul", [soft_loss, alpha_const])
            weighted_hard = self._node("Mul", [hard_loss_out, one_minus_alpha_const])
            return self._node("Add", [weighted_soft, weighted_hard])

    return DistillationLoss


def _clamp_ir_version(path):
    model = onnx.load(path)
    if model.ir_version > MAX_SUPPORTED_IR_VERSION:
        model.ir_version = MAX_SUPPORTED_IR_VERSION
        onnx.save(model, path)


def main():
    from onnxruntime.training import artifacts

    os.makedirs(ASSETS_DIR, exist_ok=True)

    teacher = _build_causal_lm()
    teacher_path = os.path.join(ASSETS_DIR, "teacher_model.onnx")
    _export_to_onnx(teacher, teacher_path)
    _clamp_ir_version(teacher_path)
    print(f"wrote {teacher_path}")

    student = _build_causal_lm()
    student_path = os.path.join(ASSETS_DIR, "student_export.onnx")
    _export_to_onnx(student, student_path)

    student_model = onnx.load(student_path)
    requires_grad = [i.name for i in student_model.graph.initializer]
    DistillationLoss = _make_distillation_loss_class()
    artifacts.generate_artifacts(
        student_model,
        requires_grad=requires_grad,
        loss=DistillationLoss(temperature=2.0, alpha=0.5),
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=ASSETS_DIR,
        loss_input_names=["logits", "teacher_logits", "labels"],
    )
    os.remove(student_path)
    os.remove(os.path.join(ASSETS_DIR, "eval_model.onnx"))  # unused by index.html

    # "checkpoint" is ONNX Runtime's own checkpoint format, not a ModelProto
    # -- onnx.load can't parse it, and it has no ir_version field to clamp.
    for name in ("training_model.onnx", "optimizer_model.onnx"):
        _clamp_ir_version(os.path.join(ASSETS_DIR, name))
    for name in ("checkpoint", "training_model.onnx", "optimizer_model.onnx"):
        print(f"wrote {os.path.join(ASSETS_DIR, name)}")


if __name__ == "__main__":
    main()
