"""Expected value ranges for a model's tensors, stored in the model itself.

A range is a promise about *real use*: "this input is an image in [0, 1]",
"this output is a probability". Having it on the model means the same promise
drives the random inputs ``simplify(check_n=...)`` generates, the warning when an
output leaves its range, the input box ``onnxsim.certify`` proves rewrites over,
and the interval analysis in ``onnxsim.interval``.

Storage is **model-level** ``metadata_props``, one entry per tensor::

    onnxsim.range.<tensor name> = {"min": <number | nested list | null>,
                                   "max": <number | nested list | null>}

not per-input/per-output ``ValueInfoProto.metadata_props``: ``simplify`` drops
those (checked), while model-level entries survive. ``null`` means unbounded on
that side. Lists must broadcast to the tensor's shape, so a per-channel range for
an NCHW image is written ``[[[[0.0]], [[0.0]], [[0.0]]]]`` (shape ``[1, 3, 1, 1]``),
which :func:`set_range` builds for you from any numpy array.

Only tensors whose names survive simplification can keep an annotation in
practice -- graph inputs and outputs. Annotating an intermediate tensor works
for the model as written, but ``simplify`` may rename or remove it.

CLI: ``python -m onnxsim.ranges model.onnx --show`` or
``python -m onnxsim.ranges model.onnx --set image=0,1 --set prob=0,1 -o out.onnx``.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

KEY_PREFIX = "onnxsim.range."

Bound = Optional[object]  # number, array-like, or None for unbounded
Range = Tuple[np.ndarray, np.ndarray]


def _encode(b: Bound):
    if b is None:
        return None
    a = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("use None for an unbounded side, not inf/nan")
    return a.item() if a.ndim == 0 else a.tolist()


def _decode(v, default: float) -> np.ndarray:
    if v is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def set_range(model: onnx.ModelProto, name: str, lo: Bound, hi: Bound) -> None:
    """Annotate tensor ``name`` with ``[lo, hi]`` (``None`` = unbounded), in place."""
    if lo is not None and hi is not None and np.any(np.asarray(lo) > np.asarray(hi)):
        raise ValueError(f"empty range for {name!r}: min > max")
    payload = json.dumps({"min": _encode(lo), "max": _encode(hi)})
    clear_range(model, name)
    entry = model.metadata_props.add()
    entry.key, entry.value = KEY_PREFIX + name, payload


def clear_range(model: onnx.ModelProto, name: Optional[str] = None) -> None:
    """Remove the annotation for ``name``, or every range annotation when ``None``."""
    keep = [
        (p.key, p.value)
        for p in model.metadata_props
        if not p.key.startswith(KEY_PREFIX)
        or (name is not None and p.key != KEY_PREFIX + name)
    ]
    del model.metadata_props[:]
    for k, v in keep:
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v


def get_ranges(model: onnx.ModelProto) -> Dict[str, Range]:
    """All annotated ranges as ``{name: (lo, hi)}`` float64 arrays; unbounded sides are +-inf."""
    out: Dict[str, Range] = {}
    for p in model.metadata_props:
        if not p.key.startswith(KEY_PREFIX):
            continue
        try:
            d = json.loads(p.value)
            out[p.key[len(KEY_PREFIX) :]] = (
                _decode(d.get("min"), -np.inf),
                _decode(d.get("max"), np.inf),
            )
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"malformed range annotation {p.key!r}: {p.value!r}"
            ) from e
    return out


def sample(
    rng_: Range, shape, dtype=np.float32, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Uniform random tensor of ``shape`` inside ``rng_``.

    A side that is infinite is replaced by one unit past the finite side (or
    ``[-1, 1]`` when both are infinite), so the sample is always finite.
    """
    rng = rng or np.random.default_rng()
    lo, hi = (np.broadcast_to(b, shape).astype(np.float64) for b in rng_)
    lo, hi = (
        np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi - 1.0, -1.0)),
        np.where(np.isfinite(hi), hi, np.where(np.isfinite(lo), lo + 1.0, 1.0)),
    )
    return (lo + (hi - lo) * rng.random(shape)).astype(dtype)


def check_outputs(
    ranges: Dict[str, Range],
    outputs: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> List[str]:
    """Describe every annotated tensor in ``outputs`` that leaves its range."""
    problems = []
    for name, value in outputs.items():
        if name not in ranges or not np.issubdtype(
            np.asarray(value).dtype, np.floating
        ):
            continue
        lo, hi = ranges[name]
        v = np.asarray(value, dtype=np.float64)
        slack = atol + rtol * np.abs(v)
        below = v < np.broadcast_to(lo, v.shape) - slack
        above = v > np.broadcast_to(hi, v.shape) + slack
        if below.any() or above.any():
            problems.append(
                f"{name}: observed [{np.nanmin(v):.6g}, {np.nanmax(v):.6g}] leaves the "
                f"annotated range ({int(below.sum() + above.sum())} of {v.size} elements)"
            )
    return problems


def _parse_set(spec: str):
    name, _, rest = spec.rpartition("=")
    if not name or "," not in rest:
        raise argparse.ArgumentTypeError(
            f"expected NAME=LO,HI (use 'none' for unbounded), got {spec!r}"
        )
    lo, hi = rest.split(",", 1)
    conv = lambda s: None if s.strip().lower() in ("none", "") else float(s)  # noqa: E731
    return name, conv(lo), conv(hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onnxsim.ranges", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("model")
    ap.add_argument("--show", action="store_true", help="print the annotated ranges")
    ap.add_argument(
        "--set", action="append", default=[], type=_parse_set, metavar="NAME=LO,HI"
    )
    ap.add_argument(
        "--clear", action="store_true", help="remove every range annotation first"
    )
    ap.add_argument(
        "-o", "--output", help="write the annotated model here (default: in place)"
    )
    args = ap.parse_args(argv)
    m = onnx.load(args.model)
    if args.clear:
        clear_range(m)
    for name, lo, hi in args.set:
        set_range(m, name, lo, hi)
    if args.set or args.clear:
        onnx.save(m, args.output or args.model)
    if args.show or not (args.set or args.clear):
        for name, (lo, hi) in get_ranges(m).items():
            print(
                f"{name}: [{np.min(lo):g}, {np.max(hi):g}]"
                + ("" if lo.ndim == 0 else " (per-element)")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
