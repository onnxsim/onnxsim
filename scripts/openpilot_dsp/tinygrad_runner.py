"""Generic tinygrad ONNX runner for Snapdragon 845 Hexagon experiments.

This is deliberately session-shaped like ONNX Runtime's Python API, so an
application can replace its SNPE/ORT model session without changing model IO
plumbing.  It does not claim to provide SNPE's Android transport: ``DSP``
requires tinygrad's runtime, while ``MOCKDSP=1`` runs generated V65 code under
qemu on a host.

Examples::

    DEV=DSP MOCKDSP=1 python tinygrad_runner.py model.onnx inputs.npz outputs.npz
    DEV=DSP python tinygrad_runner.py model.onnx inputs.npz outputs.npz --device DSP

The input archive must use ONNX input names as keys. Outputs are written using
the ONNX output names as keys in a compressed NumPy archive.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TARGET_DIR = (
    Path(__file__).resolve().parents[1]
    / "android"
    / "tinygrad_hexagon_bridge"
    / "tinygrad_codegen"
)
if str(_TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(_TARGET_DIR))
from hexagon_target import configure_tinygrad_environment  # noqa: E402


@dataclass(frozen=True)
class ValueInfo:
    name: str
    type: str
    shape: tuple[int, ...]


class TinygradSession:
    """A minimal ORT-compatible session backed by tinygrad's ``OnnxRunner``."""

    def __init__(
        self,
        model: str | Path,
        device: str = "DSP",
        target: str = "snapdragon845",
    ):
        # Import lazily: model conversion, profiling, and --help must work on
        # machines without tinygrad or a Hexagon toolchain installed.
        os.environ.update(configure_tinygrad_environment_name(target))
        from tinygrad import Tensor
        from tinygrad.nn.onnx import OnnxRunner
        import onnx

        self._tensor = Tensor
        self._model = onnx.load(str(model), load_external_data=False)
        self._inputs = tuple(
            ValueInfo(v.name, _onnx_type(v), _onnx_shape(v))
            for v in self._model.graph.input
            if v.name not in {i.name for i in self._model.graph.initializer}
        )
        self._outputs = tuple(
            ValueInfo(v.name, _onnx_type(v), _onnx_shape(v))
            for v in self._model.graph.output
        )
        self.device = device
        self._runner = OnnxRunner(str(model)).to(device)

    def get_inputs(self) -> tuple[ValueInfo, ...]:
        return self._inputs

    def get_outputs(self) -> tuple[ValueInfo, ...]:
        return self._outputs

    def run(self, _output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        result = self.run_tensors(feeds)
        return [result[output.name].numpy() for output in self._outputs]

    def run_tensors(self, feeds: dict[str, Any]) -> dict[str, Any]:
        """Run while retaining tinygrad tensors for recurrent state reuse."""
        tensors = {
            name: value
            if isinstance(value, self._tensor)
            else self._tensor(value, device=self.device)
            for name, value in feeds.items()
        }
        return self._runner(tensors)


def _onnx_type(value_info) -> str:
    import onnx

    elem = value_info.type.tensor_type.elem_type
    return onnx.TensorProto.DataType.Name(elem).lower()


def _onnx_shape(value_info) -> tuple[int, ...]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ValueError(f"dynamic model input is unsupported: {value_info.name}")
        shape.append(dim.dim_value)
    return tuple(shape)


def configure_tinygrad_environment_name(target: str) -> dict[str, str]:
    """Resolve a named target before tinygrad imports its renderer globals."""

    from hexagon_target import get_target

    return configure_tinygrad_environment(get_target(target))


def run_npz(
    model: str | Path,
    inputs: str | Path,
    device: str = "DSP",
    target: str = "snapdragon845",
) -> dict[str, np.ndarray]:
    session = TinygradSession(model, device, target)
    archive = np.load(inputs)
    missing = [item.name for item in session.get_inputs() if item.name not in archive]
    if missing:
        raise ValueError(f"missing ONNX inputs in {inputs}: {', '.join(missing)}")
    feeds = {item.name: np.asarray(archive[item.name]) for item in session.get_inputs()}
    return dict(
        zip((item.name for item in session.get_outputs()), session.run(None, feeds))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("inputs", help=".npz keyed by ONNX input names")
    parser.add_argument("outputs", help="output .npz path")
    parser.add_argument("--device", default=os.environ.get("DEV", "DSP"))
    parser.add_argument("--target", default="snapdragon845")
    args = parser.parse_args()
    outputs = run_npz(args.model, args.inputs, args.device, args.target)
    np.savez_compressed(args.outputs, **outputs)
    print(
        f"wrote {args.outputs}: "
        f"{', '.join(f'{k}={v.shape}' for k, v in outputs.items())}"
    )


if __name__ == "__main__":
    main()
