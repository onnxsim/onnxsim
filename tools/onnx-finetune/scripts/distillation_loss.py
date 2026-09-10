"""Custom onnxblock loss for knowledge distillation, shared by
``generate_artifacts.py --loss distillation``.

Unlike this tool's other three loss choices (mse/cross-entropy/bce, plain
``onnxruntime.training.artifacts.LossType`` enum values), distillation needs
a *second* model's output (a frozen teacher's logits) as an extra input to
the loss -- there is no enum for that, so this builds the loss graph by hand
via ``onnxruntime.training.onnxblock``.

Deliberately does NOT take the teacher model itself at artifact-generation
time: the training graph this produces declares ``teacher_logits`` as a
plain external input, left for whoever actually runs training (``onnx-
finetune`` / its WASM binding) to supply each step -- typically by running a
separate frozen teacher inference session on the same input, but nothing
here assumes that; precomputed/cached teacher logits fed from a file work
identically. This keeps generate_artifacts.py's job to exactly one thing
(build one model's gradient graph), the same as its other three loss modes.
"""

import copy


def build_distillation_loss_class():
    """Returns a ``DistillationLoss`` class subclassing ``onnxblock.blocks.
    Block`` -- built lazily inside a function (rather than at module import
    time) so importing this module doesn't require ``onnxruntime-training``
    to be installed just to read/test it.
    """
    from onnxruntime.training.onnxblock import blocks
    from onnx import TensorProto, helper

    class DistillationLoss(blocks.Block):
        """``alpha`` * soft-target cross-entropy (temperature-scaled, per
        Hinton et al.) + ``(1 - alpha)`` * hard-label cross-entropy.

        Built from raw ONNX nodes rather than composing the built-in
        ``onnxblock.loss.CrossEntropyLoss`` convenience block: that block's
        labels-input creation requires its score input to already be a
        registered graph *output*, which this loss's soft-loss branch
        produces only as an intermediate tensor.

        The individual sub-losses are also exposed as graph outputs under
        fixed, predictable names -- ``SOFT_LOSS_OUTPUT_NAME``/
        ``HARD_LOSS_OUTPUT_NAME`` -- rather than only the combined total
        this block returns, so callers can pass them to ``generate_
        artifacts(..., additional_output_names=[...])`` for a separate
        soft/hard breakdown *without* needing to call this block first to
        learn the names (every other intermediate tensor's name is a
        non-deterministic counter, generated fresh each ``build()`` call --
        ``additional_output_names`` has to be known before ``generate_
        artifacts()`` runs, since it constructs and calls this block
        internally).
        """

        SOFT_LOSS_OUTPUT_NAME = "kd_soft_loss"
        HARD_LOSS_OUTPUT_NAME = "kd_hard_loss"

        def __init__(self, temperature=2.0, alpha=0.5):
            super().__init__()
            self._t = temperature
            self._alpha = alpha
            self._counter = 0

        def _name(self, suffix):
            self._counter += 1
            return f"kd_loss/{suffix}_{self._counter}"

        def _node(self, op_type, inputs, output_name=None, **attrs):
            out = output_name or self._name(f"{op_type.lower()}_out")
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
            rank = len(logits_vi.type.tensor_type.shape.dim)
            if rank < 2:
                raise ValueError(
                    f"distillation loss needs a >=2D logits tensor (batch, ..., classes); "
                    f"{logits_name!r} has rank {rank}"
                )

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
            soft_loss = self._node("Mul", [soft_loss, t_sq_const], output_name=self.SOFT_LOSS_OUTPUT_NAME)

            # --- hard loss: SoftmaxCrossEntropyLoss wants the class dim at
            # axis 1 -- already true for a plain (batch, classes) classifier,
            # but a (batch, seq, classes)-shaped model (e.g. a causal LM)
            # needs its trailing class dim moved there first.
            hard_loss_input = logits_name
            if rank > 2:
                perm = [0, rank - 1] + list(range(1, rank - 1))
                hard_loss_input = self._node("Transpose", [logits_name], perm=perm)
            hard_loss_out = self.HARD_LOSS_OUTPUT_NAME
            log_prob_out = self._name("log_prob")
            g.node.append(
                helper.make_node(
                    "SoftmaxCrossEntropyLoss",
                    [hard_loss_input, labels_name],
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
