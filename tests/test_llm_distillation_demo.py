"""Smoke test for examples/llm_distillation/distill.py's training mechanics.

Loaded by file path rather than import: ``examples/`` sits outside the
``onnxsim`` package and outside pytest's own ``tests/`` sys.path entry (see
``tests/conftest.py``'s docstring for why nothing here adds the repo root to
``sys.path``).
"""

import importlib.util
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

_DISTILL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "llm_distillation", "distill.py"
)
_spec = importlib.util.spec_from_file_location(
    "llm_distillation_distill", _DISTILL_PATH
)
distill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distill)


def test_tiny_distillation_step_runs_and_produces_finite_losses():
    torch_mod, F, AutoModelForCausalLM, LlamaConfig = distill._lazy_imports()

    teacher = distill.build_causal_lm(
        distill.TINY_SPEC,
        vocab_size=64,
        max_position_embeddings=32,
        LlamaConfig=LlamaConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    student = distill.build_causal_lm(
        distill.TINY_SPEC,
        vocab_size=64,
        max_position_embeddings=32,
        LlamaConfig=LlamaConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    assert distill.num_parameters(student) > 0
    assert distill.num_parameters(teacher) > 0

    history = distill.run_distillation(
        teacher,
        student,
        vocab_size=64,
        steps=2,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        temperature=2.0,
        alpha=0.5,
        device="cpu",
        torch=torch_mod,
        F=F,
        seed=0,
        log_every=100,
    )

    assert len(history) == 2
    assert all(math.isfinite(loss) for loss in history)


def test_student_is_smaller_than_teacher_spec():
    # STUDENT_SPEC/TEACHER_SPEC are the two named real-scale architectures
    # the CLI defaults to (see distill.py's module docstring): the student
    # must stay meaningfully smaller than its teacher for distillation to
    # make sense.
    assert distill.STUDENT_SPEC.hidden_size < distill.TEACHER_SPEC.hidden_size
    assert (
        distill.STUDENT_SPEC.num_hidden_layers < distill.TEACHER_SPEC.num_hidden_layers
    )
