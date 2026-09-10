"""A *real* backend for `scripts/axelera/`, wrapping Voyager SDK's actual
`axelera.compiler.quantize()` -- unlike `voyager_simulator.py` (docs-only,
no compiler involved at all), this module calls Axelera's real quantizer and
reports genuinely observed behavior.

**This corrects an earlier, wrong assumption** in this directory (see git
history / `voyager_simulator.py`'s docstring for the original framing):
`axelera-rt` and `axelera-devkit` -- the packages providing `axelera.
compiler` -- install cleanly via `pip install --extra-index-url https://
software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt
axelera-devkit[all]` (the exact command voyager-sdk's own `docs/user-guides/
sdk-install.md` documents) with **no login or token**. That is a genuinely
public Artifactory PyPI mirror, confirmed by actually installing from it.
The earlier "proprietary, credentials-only" claim was based on the
deprecated TOML-based installer's `axelera_runtime` package name/framing
(`installer_support.py`) and never actually re-tested against the current
pip-based install path -- a real mistake, not a deliberate simplification.

**What actually works here, confirmed by running it:**

- `axelera.compiler.quantize(model, calibration_dataset, config)` runs real
  INT8 post-training quantization and returns an `AxeleraQuantizedModel`
  callable on CPU -- no Metis hardware, no device driver, no extra setup
  needed beyond the two pip packages above.
- Observed: quantizing a `Conv -> BatchNormalization -> Relu` graph and its
  onnxsim-simplified `Conv -> Relu` form (BN fused into Conv) through this
  real quantizer produces **bit-identical** output on the same test input.
- Observed: a `Conv` with `auto_pad="SAME_UPPER"` (violating the
  `voyager_ops.VOYAGER_OP_SUPPORT["Conv"]["rules"]` entry `auto_pad ==
  "NOTSET"`) makes `quantize()` raise `axelera.compiler.exceptions.
  QtoolsError`, and the real error message quotes that *exact* constraint
  string back -- direct confirmation that `voyager_op_support_data.py`'s
  scraped rules are the literal predicates the real compiler's own
  compatibility checker evaluates, not just prose.
- **This refines (and partly corrects) `onnx-support.md`'s own "falls back
  to host CPU" framing**, at least for this case: CPU fallback is what
  happens for an *undocumented* op type. A *documented* "Constrained" op
  used **outside** its supported configuration was observed to be a hard
  `quantize()`-time failure, not a graceful CPU fallback -- so
  `voyager_simulator.evaluate_constraints()`'s `"violated"` verdict should
  be read as "quantization will likely fail outright", not merely "this
  node will run on the host instead".

**What still doesn't work here** (and why): `axelera.compiler.compile()`
(the step that lowers a quantized model to real deployable `.axmodel`
artifacts) failed with `axkernelcc: Cannot determine device support
directory. Set AXE_CHROOT or AXELERA_DEVICE_DIR...` -- it needs a further
"device support" package/directory this environment doesn't have (likely
tied to the driver/device install steps in `sdk-install.md`, which do need
either real hardware or additional setup this module never attempted). So:
`quantize()` (int8 correctness) is real and covered here; `compile()`
(real Metis deployment artifacts, and therefore true on-device behavior)
is not.

**Still be careful generalizing from the above**: the two observations
above are single data points (one fusion, one constraint), not a general
audit. Treat this the way `scripts/axera/pulsar2_simulator.py` treats its
own single real-hardware data point -- informative, not exhaustive.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

_INSTALL_HINT = (
    "The real Voyager SDK compiler backend needs 'axelera-rt' and "
    "'axelera-devkit' (large, optional dependencies -- pulls torch, CUDA "
    "toolkit packages, TVM, and more). Install with: pip install "
    "--extra-index-url https://software.axelera.ai/artifactory/api/pypi/"
    "axelera-pypi/simple axelera-rt axelera-devkit[all]"
)


def has_axelera_compiler() -> bool:
    """Whether `axelera.compiler` (from the `axelera-devkit` package) is
    importable in this environment."""
    try:
        import axelera.compiler  # noqa: F401
    except ImportError:
        return False
    return True


def _import_compiler():
    try:
        from axelera import compiler
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return compiler


def quantize(model: Any, calibration_dataset: Iterator, config: Optional[Any] = None):
    """Thin wrapper around `axelera.compiler.quantize()`.

    :param model: a `torch.nn.Module`, `onnx.ModelProto`, or path to a
            `.onnx` file -- forwarded as-is.
    :param calibration_dataset: an iterator yielding calibration samples
            (see Voyager SDK's `docs/reference/compiler/compiler-api.md`
            for the expected shapes). Consumed once; build a fresh one for
            each `quantize()` call, e.g. via a generator *function* (not a
            generator object) so `compare_before_after_simplify` can call
            it twice with the same seed for a fair comparison.
    :param config: an `axelera.compiler.CompilerConfig`, or `None` for the
            default.
    :returns: an `axelera.compiler.quantized_model.AxeleraQuantizedModel`,
            callable on CPU with a `torch.Tensor` input.
    :raises RuntimeError: if `axelera-devkit` is not installed.
    :raises axelera.compiler.exceptions.QtoolsError: (among other real
            compiler exceptions) if the model violates a documented
            constraint, is otherwise unsupported, or quantization
            otherwise fails -- these are Voyager SDK's own real errors,
            not wrapped or reinterpreted here.
    """
    compiler = _import_compiler()
    if config is None:
        config = compiler.CompilerConfig()
    return compiler.quantize(
        model=model, calibration_dataset=calibration_dataset, config=config
    )


def compare_before_after_simplify(
    model: Any,
    simplified_model: Any,
    calibration_dataset_fn: Callable[[], Iterator],
    test_input,
    config: Optional[Any] = None,
):
    """Quantize both `model` and `simplified_model` through the real
    Voyager SDK compiler (same calibration data, same test input) and
    report whether their outputs match.

    :param model: the original ONNX model (or path/`torch.nn.Module`).
    :param simplified_model: typically `onnxsim.simplify(model)[0]`.
    :param calibration_dataset_fn: a zero-argument callable returning a
            fresh calibration iterator -- called once per model, so both
            quantize with identical calibration data.
    :param test_input: a `torch.Tensor` (or convertible) fed to both
            quantized models for the comparison.
    :param config: forwarded to both `quantize()` calls.
    :returns: a dict with `output_before`, `output_after`,
            `max_abs_diff` (float), and `bit_identical` (bool).
    """
    import torch

    q_before = quantize(model, calibration_dataset_fn(), config)
    q_after = quantize(simplified_model, calibration_dataset_fn(), config)

    if not torch.is_tensor(test_input):
        test_input = torch.as_tensor(test_input)

    out_before = q_before(test_input)
    out_after = q_after(test_input)
    diff = (out_before - out_after).abs()
    return {
        "output_before": out_before,
        "output_after": out_after,
        "max_abs_diff": diff.max().item(),
        "bit_identical": torch.equal(out_before, out_after),
    }
