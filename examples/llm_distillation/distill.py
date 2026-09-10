"""Knowledge-distillation example: train a small causal LM "student" to mimic
a larger "teacher" model.

This is a standalone PyTorch training example, unrelated to onnxsim's own
ONNX-to-ONNX simplification pipeline -- onnxsim has no training code, and
this script doesn't add any to the package. It exists to demonstrate the
distillation *mechanics* (soft-target KL divergence with temperature, per
Hinton et al., combined with the usual next-token hard-label loss) end to
end on real, named architectures, kept deliberately small so the demo is
actually runnable without a beefy GPU:

- **Student**: matches the widely-used ``JackFram/llama-160m`` shape
  (hidden_size=768, 12 layers, 12 attention heads, 32000-token vocabulary,
  no GQA) -- a real ~162M-parameter decoder-only causal LM, the same
  architecture commonly used as a speculative-decoding draft model.
- **Teacher**: matches TinyLlama-1.1B's published shape (hidden_size=2048,
  22 layers, 32 attention heads / 4 KV heads) -- a real ~1.1B-parameter
  sibling architecture, comfortably larger than the student above.

Both are randomly initialized by default and trained on synthetic random
token ids -- this exercises the training loop's shapes/losses/gradient flow
in well under a second (see ``--tiny``), it does not produce a
pretrained-quality model. Pass ``--teacher-model-id``/``--student-model-id``
to distill from/into real Hugging Face Hub checkpoints, and swap
``run_distillation``'s synthetic batches for real tokenized data, for an
actual training run.

See ``README.md`` in this directory for usage examples, including the
optional final step that ties this back into onnxsim: exporting the
distilled student to ONNX and simplifying it via
``onnxsim.export_transformers_model()``.
"""

import argparse
import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


# JackFram/llama-160m's published architecture (~162M parameters) -- small
# enough to train on CPU for this demo, while still a real architecture
# (commonly used as a speculative-decoding draft model for larger Llamas).
STUDENT_SPEC = ModelSpec(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
)

# TinyLlama-1.1B's published architecture (~1.1B parameters) -- comfortably
# larger than STUDENT_SPEC above, to act as its distillation teacher.
TEACHER_SPEC = ModelSpec(
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=4,
)

# A tiny stand-in for both specs above: same code path, no real
# parameter-count claim attached. Used by --tiny and by this example's test.
TINY_SPEC = ModelSpec(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
)


def _lazy_imports():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, LlamaConfig
    except ImportError as e:
        raise ImportError(
            "examples/llm_distillation/distill.py needs the optional 'torch' "
            "and 'transformers' packages: pip install onnxsim[transformers]"
        ) from e
    return torch, F, AutoModelForCausalLM, LlamaConfig


def build_causal_lm(spec, vocab_size, max_position_embeddings, LlamaConfig, AutoModelForCausalLM):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
    )
    return AutoModelForCausalLM.from_config(config)


def num_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def distillation_loss(F, student_logits, teacher_logits, labels, temperature, alpha):
    """Hinton et al. soft-target KL divergence (temperature-scaled), combined
    with the usual next-token hard-label cross-entropy.

    ``alpha`` weighs the two: 1.0 is pure distillation, 0.0 is plain LM
    training on ``labels`` alone (self-supervised next-token ids, same as any
    causal-LM training loop -- there is no separate "hard label" dataset).
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)

    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    hard_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return alpha * soft_loss + (1 - alpha) * hard_loss, soft_loss.detach(), hard_loss.detach()


def run_distillation(
    teacher,
    student,
    vocab_size,
    steps,
    batch_size,
    seq_len,
    lr,
    temperature,
    alpha,
    device,
    torch,
    F,
    seed=0,
    log_every=1,
):
    """Run ``steps`` synthetic-data distillation steps in place on ``student``.

    Uses random token ids rather than a real corpus -- this exercises the
    distillation mechanics end to end without requiring a dataset download.
    Swap in real batches from a tokenized corpus for an actual training run.
    """
    torch.manual_seed(seed)
    teacher.to(device).eval()
    student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    history = []
    for step in range(1, steps + 1):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits
        student_logits = student(input_ids).logits

        loss, soft_loss, hard_loss = distillation_loss(
            F, student_logits, teacher_logits, input_ids, temperature, alpha
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))
        if step % log_every == 0 or step == steps:
            print(
                f"step {step}/{steps}  loss={loss.item():.4f}  "
                f"soft={soft_loss.item():.4f}  hard={hard_loss.item():.4f}"
            )
    return history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model-id", default=None, help="Hugging Face Hub id or local directory to load a pretrained teacher from, instead of a randomly initialized TEACHER_SPEC")
    parser.add_argument("--student-model-id", default=None, help="Hugging Face Hub id or local directory to load a pretrained student from, instead of a randomly initialized STUDENT_SPEC")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on the soft (KD) loss vs. the hard next-token loss")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./distilled-student-demo")
    parser.add_argument("--tiny", action="store_true", help="shrink every model/data dimension to a tiny synthetic size and run only 2 steps, for a fast (<1s) smoke test of the training mechanics -- not a real distillation run")
    parser.add_argument("--export-onnx", action="store_true", help="after training, export the saved student checkpoint to ONNX and simplify it via onnxsim.export_transformers_model (needs 'optimum' too: pip install onnxsim[transformers])")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch, F, AutoModelForCausalLM, LlamaConfig = _lazy_imports()

    teacher_spec = TINY_SPEC if args.tiny else TEACHER_SPEC
    student_spec = TINY_SPEC if args.tiny else STUDENT_SPEC
    vocab_size = 64 if args.tiny else args.vocab_size
    max_position_embeddings = 32 if args.tiny else args.max_position_embeddings
    seq_len = 8 if args.tiny else args.seq_len
    steps = 2 if args.tiny else args.steps

    if args.teacher_model_id:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_id)
    else:
        teacher = build_causal_lm(teacher_spec, vocab_size, max_position_embeddings, LlamaConfig, AutoModelForCausalLM)

    if args.student_model_id:
        student = AutoModelForCausalLM.from_pretrained(args.student_model_id)
    else:
        student = build_causal_lm(student_spec, vocab_size, max_position_embeddings, LlamaConfig, AutoModelForCausalLM)

    print(f"teacher parameters: {num_parameters(teacher):,}")
    print(f"student parameters: {num_parameters(student):,}")

    run_distillation(
        teacher,
        student,
        vocab_size,
        steps,
        args.batch_size,
        seq_len,
        args.lr,
        args.temperature,
        args.alpha,
        args.device,
        torch,
        F,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    print(f"saved distilled student to {args.output_dir}")

    if args.export_onnx:
        from onnxsim import export_transformers_model

        onnx_dir = os.path.join(args.output_dir, "onnx")
        results = export_transformers_model(args.output_dir, onnx_dir, task="text-generation-with-past")
        print(f"exported + simplified to {onnx_dir}: {results}")


if __name__ == "__main__":
    main()
