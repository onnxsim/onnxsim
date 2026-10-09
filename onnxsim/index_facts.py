"""Index-in-range facts for Gather, GatherElements and Slice.

Each fact is a claim about one node, proved over the input box given to
:func:`onnxsim.interval.propagate`:

* ``in_range``: every index the node can see lies in ``[-n, n-1]`` for the axis it
  indexes, so the node is valid for every input in the box.
* ``out_of_range``: every index lies outside that window, so every input in the box
  makes the node invalid. Only claimed for a fixed axis size, since a dynamic size may
  make the index valid.
* ``unknown``: neither holds. The index enclosure straddles the window, the index or
  the axis has no interval, or the index is unbounded.

For a dynamic axis size the lower bound is used: an index inside ``[-lo, lo-1]`` is
valid for every size at least ``lo``.

ONNX clamps Slice's starts and ends, so a Slice fact says whether each start is a valid
index. ``out_of_range`` then means the slice is empty, not that it fails. Only ``starts``
is checked; ``ends`` carries no fact. ``GatherND`` is not covered.
"""

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import onnx

from onnxsim import interval as _interval
from onnxsim import shape_ranges as _sr

IN_RANGE = "in_range"
OUT_OF_RANGE = "out_of_range"
UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class IndexFact:
    node: str  # the node's first output, which names it
    op_type: str
    axis: int
    verdict: str  # IN_RANGE, OUT_OF_RANGE or UNKNOWN
    index_hull: Optional[Tuple[float, float]]
    size: Optional[_sr.Dim]
    detail: str


@dataclasses.dataclass
class IndexFactsReport:
    facts: List[IndexFact]

    @property
    def proved(self) -> bool:
        """Every covered index is in range for every input in the box."""
        return all(f.verdict == IN_RANGE for f in self.facts)


def index_facts(
    model: onnx.ModelProto, input_ranges: Optional[dict] = None
) -> IndexFactsReport:
    res = _interval.propagate(model, input_ranges)
    facts: List[IndexFact] = []
    for node in model.graph.node:
        if node.op_type in ("Gather", "GatherElements"):
            facts.append(_gather_fact(res, node))
        elif node.op_type == "Slice":
            facts.extend(_slice_facts(res, node))
    return IndexFactsReport(facts)


def _known(res: _interval.IntervalResult, name: str) -> bool:
    return name in res.intervals or name in res.ranged


def _hull(res: _interval.IntervalResult, name: str) -> Optional[Tuple[float, float]]:
    """Hull of a tensor's values; ``(inf, -inf)`` for an empty tensor, ``None`` if unknown."""
    if not _known(res, name):
        return None
    if name in res.intervals and res.intervals[name][0].size == 0:
        return (np.inf, -np.inf)
    return res.hull(name)


def _static_ints(res: _interval.IntervalResult, name: str) -> Optional[List[int]]:
    if name not in res.intervals:
        return None
    lo, hi = res.intervals[name]
    if not np.array_equal(lo, hi) or not np.all(np.mod(lo, 1) == 0):
        return None
    return [int(v) for v in np.ravel(lo)]


def _normalize_axis(axis: int, rank: int) -> Optional[int]:
    a = axis + rank if axis < 0 else axis
    return a if 0 <= a < rank else None


def _gather_fact(res: _interval.IntervalResult, node: onnx.NodeProto) -> IndexFact:
    out = node.output[0]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    axis_attr = int(attrs.get("axis", 0))
    data, idx = node.input[0], node.input[1]
    if not _known(res, data):
        return IndexFact(
            out, node.op_type, axis_attr, UNKNOWN, None, None, "data has no interval"
        )
    shape = res.shape(data)
    axis = _normalize_axis(axis_attr, len(shape))
    if axis is None:
        return IndexFact(
            out,
            node.op_type,
            axis_attr,
            UNKNOWN,
            None,
            None,
            f"axis {axis_attr} is out of range for rank {len(shape)}",
        )
    return _judge(node.op_type, out, axis, shape[axis], _hull(res, idx), "index")


def _slice_facts(
    res: _interval.IntervalResult, node: onnx.NodeProto
) -> List[IndexFact]:
    out = node.output[0]
    data = node.input[0]
    starts = node.input[1] if len(node.input) > 1 else ""
    axes_name = node.input[3] if len(node.input) > 3 and node.input[3] else ""
    if not (_known(res, data) and starts and _known(res, starts)):
        return [
            IndexFact(
                out, "Slice", -1, UNKNOWN, None, None, "operands have no interval"
            )
        ]
    if starts not in res.intervals:
        return [
            IndexFact(
                out, "Slice", -1, UNKNOWN, None, None, "starts has a ranged shape"
            )
        ]
    shape = res.shape(data)
    s_lo, s_hi = res.intervals[starts]
    count = int(np.size(s_lo))
    if axes_name:
        axes = _static_ints(res, axes_name)
        if axes is None or len(axes) != count:
            return [
                IndexFact(
                    out,
                    "Slice",
                    -1,
                    UNKNOWN,
                    None,
                    None,
                    "axes is not a static list matching starts",
                )
            ]
    else:
        axes = list(range(count))
    facts = []
    for i, (ax, lo, hi) in enumerate(zip(axes, np.ravel(s_lo), np.ravel(s_hi))):
        axis = _normalize_axis(ax, len(shape))
        if axis is None:
            facts.append(
                IndexFact(
                    out,
                    "Slice",
                    ax,
                    UNKNOWN,
                    None,
                    None,
                    f"axis {ax} is out of range for rank {len(shape)}",
                )
            )
            continue
        facts.append(
            _judge(
                "Slice", out, axis, shape[axis], (float(lo), float(hi)), f"start[{i}]"
            )
        )
    return facts


def _judge(
    op: str,
    out: str,
    axis: int,
    dim: _sr.Dim,
    hull: Optional[Tuple[float, float]],
    what: str,
) -> IndexFact:
    if hull is None:
        return IndexFact(out, op, axis, UNKNOWN, None, dim, f"{what} has no interval")
    lo, hi = hull
    if lo > hi:
        return IndexFact(out, op, axis, IN_RANGE, None, dim, f"{what} is empty")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return IndexFact(out, op, axis, UNKNOWN, hull, dim, f"{what} is unbounded")
    n_lo = dim.lo
    if lo >= -n_lo and hi <= n_lo - 1:
        return IndexFact(
            out,
            op,
            axis,
            IN_RANGE,
            hull,
            dim,
            f"{what} in [{lo:g}, {hi:g}] is valid for axis size >= {n_lo}",
        )
    if dim.hi == dim.lo:
        n = dim.lo
        if lo > n - 1 or hi < -n:
            return IndexFact(
                out,
                op,
                axis,
                OUT_OF_RANGE,
                hull,
                dim,
                f"every {what} lies outside [-{n}, {n - 1}]",
            )
    return IndexFact(
        out,
        op,
        axis,
        UNKNOWN,
        hull,
        dim,
        f"{what} in [{lo:g}, {hi:g}] is not inside [-{n_lo}, {n_lo - 1}] for axis size {dim}",
    )
