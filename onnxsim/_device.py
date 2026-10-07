"""Device and precision selection for the optional torch (CUDA / ROCm) backends.

``onnxsim.crown`` and ``onnxsim.backward_diff`` run their bound propagation on numpy
float64 by default. ``device=`` / ``precision=`` move it to a torch device:

``device``
    ``"cpu"`` (the default, ``None`` means the same): the numpy float64 path, unchanged and
    needing neither torch nor a GPU.

    ``"cuda"`` / ``"cuda:N"``: a torch accelerator. torch's ``cuda`` API also drives ROCm
    builds, so this covers AMD GPUs too (the integrated Radeon 8060S shows up as ``cuda``).

    ``"torch-cpu"``: the torch backend on the CPU. Slower than numpy for these workloads, but it
    runs the *same code path* as a GPU, which is what the CI tests (no GPU there) use to cover
    it, including the float32 soundness machinery.

    ``"auto"``: ``"cuda"`` if torch sees an accelerator, else ``"cpu"``.

``precision``
    ``"float64"`` (the default) or ``"float32"``. float32 is what makes consumer GPUs fast (the
    RTX 5050 runs float64 *slower* than the CPU), but float32 results are only trustworthy when
    the rounding error is accounted for: ``precision="float32"`` therefore runs the rigorous
    error-tracking scheme of :mod:`onnxsim._rigorous_f32` and returns *sound* bounds, at the
    cost of a shadow computation (about 2x the arithmetic) and a small loss of tightness.

torch stays optional: nothing here imports it at module import time, and ``"cpu"`` +
``"float64"`` never does.
"""

import contextlib
import dataclasses
from typing import Any, Iterator, Optional

DEVICES = ("cpu", "auto", "cuda", "torch-cpu")
PRECISIONS = ("float64", "float32")


@dataclasses.dataclass(frozen=True)
class Resolved:
    """A resolved ``(device, precision)`` pair.

    ``kind`` is ``"numpy"`` or ``"torch"``; ``torch_device`` is the torch device string
    (``"cpu"``, ``"cuda"``, ``"cuda:1"``) for the torch kind; ``precision`` is the
    arithmetic the bounds are computed in.
    """

    kind: str
    torch_device: Optional[str]
    precision: str

    @property
    def is_numpy(self) -> bool:
        return self.kind == "numpy"

    @property
    def is_f32(self) -> bool:
        return self.precision == "float32"

    @property
    def is_accelerator(self) -> bool:
        return self.torch_device is not None and self.torch_device.startswith("cuda")

    def describe(self) -> str:
        if self.is_numpy:
            return "numpy/float64"
        return f"torch:{self.torch_device}/{self.precision}"


NUMPY = Resolved("numpy", None, "float64")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "device='cuda' / 'torch-cpu' / 'auto' (with an accelerator) needs torch "
            "(pip install torch); the default device='cpu' does not"
        ) from e
    return torch


def cuda_available() -> bool:
    """True if torch is importable and sees a CUDA or ROCm device (never raises)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # a broken driver must not break an import-time probe
        return False


def resolve_device(device: Optional[str]) -> str:
    """Canonical device string: ``"cpu"``, ``"torch-cpu"``, ``"cuda"`` or ``"cuda:N"``.

    Raises ``ValueError`` for an unknown name, ``ImportError`` when torch is needed and
    missing, and ``RuntimeError`` (with a clear message, never a silent CPU fallback) when a
    CUDA device is requested but not available.
    """
    if device is None or device == "cpu":
        return "cpu"
    if not isinstance(device, str):
        raise ValueError(f"device must be a string, got {device!r}")
    if device == "torch-cpu":
        _import_torch()
        return "torch-cpu"
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    if device == "cuda" or device.startswith("cuda:"):
        torch = _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device!r} requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, hip={getattr(torch.version, 'hip', None)}, "
                f"cuda={torch.version.cuda}): install a CUDA/ROCm build of torch, or use "
                "device='cpu'. onnxsim never falls back to the CPU silently."
            )
        if device != "cuda":
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                raise ValueError(f"bad device {device!r}: expected 'cuda' or 'cuda:N'")
            if not 0 <= idx < torch.cuda.device_count():
                raise RuntimeError(
                    f"device={device!r} but torch sees {torch.cuda.device_count()} device(s)"
                )
        return device
    raise ValueError(
        f"device must be one of 'cpu', 'cuda', 'cuda:N', 'torch-cpu', 'auto' (or None), got {device!r}"
    )


def resolve_precision(precision: Optional[str], device: Optional[str] = "cpu") -> str:
    """Canonical precision for ``device``: ``"float64"`` or ``"float32"``."""
    p = "float64" if precision is None else precision
    if p not in PRECISIONS:
        raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}")
    if p == "float32" and resolve_device(device) == "cpu":
        raise ValueError(
            "precision='float32' needs a torch device: use device='cuda' (GPU) or "
            "device='torch-cpu'; the numpy path is float64 only"
        )
    return p


def resolve(device: Optional[str] = None, precision: Optional[str] = None) -> Resolved:
    """Resolve ``(device, precision)``; the defaults give :data:`NUMPY` (today's behaviour)."""
    dev = resolve_device(device)
    prec = resolve_precision(precision, dev)
    if dev == "cpu":
        return NUMPY
    return Resolved("torch", "cpu" if dev == "torch-cpu" else dev, prec)


def free_memory_bytes(torch_device: str) -> Optional[int]:
    """Free device memory in bytes for an accelerator (None for the CPU / when unknown)."""
    if not torch_device.startswith("cuda"):
        return None
    try:
        import torch

        free, _ = torch.cuda.mem_get_info(torch.device(torch_device))
        return int(free)
    except Exception:
        return None


@contextlib.contextmanager
def strict_float32() -> Iterator[None]:
    """Disable every reduced-precision shortcut for float32 matmul/conv while active.

    The error model of :mod:`onnxsim._rigorous_f32` assumes IEEE float32 multiplications and
    additions (any summation order, FMA allowed). TF32 (10-bit mantissa) breaks that, and
    cuDNN's convolution is allowed to use TF32 / Winograd / FFT by default; this context
    turns TF32 off and asks for the highest matmul precision, then restores the previous
    settings. The bound passes in :mod:`onnxsim.crown` use einsum / matmul only (no cuDNN
    convolution), so Winograd and FFT algorithms are never involved.
    """
    try:
        import torch
    except ImportError:  # nothing to configure without torch
        yield
        return
    prev_matmul = torch.get_float32_matmul_precision()
    prev_cuda_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    prev_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
    try:
        torch.set_float32_matmul_precision("highest")
        if prev_cuda_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
        if prev_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(prev_matmul)
        if prev_cuda_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_cuda_matmul
        if prev_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn
