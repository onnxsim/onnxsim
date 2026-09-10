"""Real-world companion to ``tests/test_llm_distillation_demo.py``: that one
only exercises ``examples/llm_distillation/distill.py``'s training mechanics
with a synthetic ``TINY_SPEC`` architecture. This file instead downloads the
actual published ``JackFram/llama-160m`` checkpoint from the Hugging Face
Hub -- the real checkpoint ``STUDENT_SPEC`` is modeled on -- and runs a real
distillation step against it, both to confirm the download-and-run path the
example's ``--student-model-id``/``--teacher-model-id`` flags document
actually works, and to lock in that ``STUDENT_SPEC`` stays a byte-accurate
match for this checkpoint's parameter count.

Only downloads the ~650MB student, not the ~4.4GB ``TinyLlama-1.1B-Chat-v1.0``
teacher the example's README also demonstrates (manually verified once when
this test was written) -- pairing the real student with a same-vocabulary
synthetic ``TINY_SPEC`` teacher keeps this test's download to the same order
of magnitude as ``tests/test_speculative_decoding_drafter_hub_checkpoint.py``'s
~270MB.

Same heavy/optional dependencies and skip conventions as that file: torch and
transformers are not normal test dependencies, so this file skips unless
they're already importable, and skips (rather than fails) on a network error
downloading the real checkpoint from the Hub.
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

HF_REPO = "JackFram/llama-160m"
# The parameter count this checkpoint is known to have -- also what
# STUDENT_SPEC (built from literal config numbers, no download) produces;
# see test_student_spec_matches_real_checkpoint_param_count below.
EXPECTED_PARAM_COUNT = 162_417_408


@pytest.fixture(scope="module")
def real_student():
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(HF_REPO)
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not download {HF_REPO} from Hugging Face Hub: {e}")


def test_student_spec_matches_real_checkpoint_param_count(real_student):
    assert distill.num_parameters(real_student) == EXPECTED_PARAM_COUNT

    _torch, _F, AutoModelForCausalLM, LlamaConfig = distill._lazy_imports()
    synthetic_student = distill.build_causal_lm(
        distill.STUDENT_SPEC,
        vocab_size=32000,
        max_position_embeddings=2048,
        LlamaConfig=LlamaConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    assert distill.num_parameters(synthetic_student) == EXPECTED_PARAM_COUNT


def test_real_checkpoint_distills_against_synthetic_teacher(real_student):
    torch_mod, F, AutoModelForCausalLM, LlamaConfig = distill._lazy_imports()

    # Same vocabulary as the real student (so logits are comparable), but a
    # tiny synthetic teacher -- keeps this test's download to just the
    # student, per this file's module docstring.
    teacher = distill.build_causal_lm(
        distill.TINY_SPEC,
        vocab_size=32000,
        max_position_embeddings=2048,
        LlamaConfig=LlamaConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )

    history = distill.run_distillation(
        teacher,
        real_student,
        vocab_size=32000,
        steps=1,
        batch_size=1,
        seq_len=8,
        lr=1e-4,
        temperature=2.0,
        alpha=0.5,
        device="cpu",
        torch=torch_mod,
        F=F,
        seed=0,
        log_every=100,
    )

    assert len(history) == 1
    assert math.isfinite(history[0])
