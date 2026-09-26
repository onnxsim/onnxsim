"""Merge onnxruntime's own session profiles into onnxsim's ``profile`` trace.

onnxsim's profiler (``ONNXSIM_PROFILE``) records one ``OrtSession`` span per
constant-folding session. onnxruntime can additionally profile *inside* each of
those sessions (``ONNXSIM_ORT_PROFILE`` -> ``SessionOptions.enable_profiling``),
writing its own Chrome trace with per-operator events. This module splices the
latter into the former so the operator-level detail shows up under each
``OrtSession`` span in a single unified flame graph -- for every executor,
including the Python ``PyModelExecutor`` that ``simplify()`` uses.

It is deliberately dependency-free (standard library only) so the merge logic can
be unit-tested without building onnxsim's native extension.
"""

import json
import os
from typing import Any, Dict, Sequence, Tuple, cast

# onnxruntime events are placed on their own tid range so they render as separate
# tracks, clear of onnxsim's own span threads.
_ORT_TID_BASE = 1000


def summarize_remote_events(profile_trace: dict) -> dict:
    """Summarize remote compile/load/execute events in a Chrome trace.

    Returns phase counts and wall-time totals in microseconds. This is useful
    for CI reports and constrained UIs that cannot render the full trace.
    """
    summary = {
        "compile": {"count": 0, "duration_us": 0},
        "load": {"count": 0, "duration_us": 0},
        "execute": {"count": 0, "duration_us": 0},
        "other": {"count": 0, "duration_us": 0},
    }
    events = (
        profile_trace.get("traceEvents", []) if isinstance(profile_trace, dict) else []
    )
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        raw_args = event.get("args")
        args: dict[str, Any] = (
            cast(dict[str, Any], raw_args) if isinstance(raw_args, dict) else {}
        )
        phase = args.get("phase")
        if phase not in summary:
            if name.startswith("RemoteRPC/compile"):
                phase = "compile"
            elif name.startswith("RemoteRPC/load"):
                phase = "load"
            elif name.startswith("RemoteRPC/execute"):
                phase = "execute"
            elif name.startswith("Remote"):
                phase = "other"
            else:
                continue
        summary[phase]["count"] += 1
        summary[phase]["duration_us"] += max(0, int(event.get("dur", 0)))
    return summary


def merge_ort_traces(profile_trace: dict, ort_trace_paths: Sequence[str]) -> dict:
    """Splice onnxruntime session traces into an onnxsim ``profile`` trace.

    ``profile_trace`` is a parsed Chrome trace produced by onnxsim's profiler; it
    contains one ``OrtSession`` span per constant-folding session.
    ``ort_trace_paths`` are onnxruntime's own per-session profiling traces, in
    session (creation) order. Each onnxruntime trace is placed onto its own
    ``onnxruntime`` track and time-shifted so it lines up under the i-th
    ``OrtSession`` span: onnxruntime's timestamps are relative to that session's
    start, so offsetting them by the span's start time drops the per-operator
    events into onnxsim's timeline right where that session ran. onnxruntime's own
    thread structure is preserved (each onnxruntime thread becomes a distinct
    track) rather than flattened, which would create bogus overlaps.

    onnxruntime traces beyond the number of ``OrtSession`` spans (e.g. the
    correctness-check runs, which are not folding sessions) are ignored. The
    ``profile_trace`` dict is mutated in place and also returned.
    """
    events = profile_trace.setdefault("traceEvents", [])
    ort_spans = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X" and e.get("name") == "OrtSession" and "ts" in e
        ),
        key=lambda e: e["ts"],
    )
    if not ort_spans:
        return profile_trace

    next_tid = _ORT_TID_BASE
    track_tids: Dict[Tuple[int, int], int] = {}
    for i, path in enumerate(ort_trace_paths):
        if i >= len(ort_spans):
            break
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            continue
        ort_events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
        if not isinstance(ort_events, list):
            continue
        complete = [
            e
            for e in ort_events
            if isinstance(e, dict)
            and e.get("ph") == "X"
            and isinstance(e.get("ts"), (int, float))
            and isinstance(e.get("dur"), (int, float))
        ]
        if not complete:
            continue
        # onnxruntime timestamps are relative to this session's start; align that
        # start with the OrtSession span's start in onnxsim's timeline.
        offset = ort_spans[i]["ts"] - min(e["ts"] for e in complete)
        for e in complete:
            key = (i, int(e.get("tid", 0)))
            tid = track_tids.get(key)
            if tid is None:
                tid = next_tid
                track_tids[key] = tid
                next_tid += 1
            events.append(
                {
                    "name": e.get("name", "op"),
                    "cat": "onnxruntime",
                    "ph": "X",
                    "ts": e["ts"] + offset,
                    "dur": e["dur"],
                    "pid": 1,
                    "tid": tid,
                    "args": e.get("args", {}),
                }
            )

    # Label each onnxruntime track so the merged flame graph reads clearly.
    for (i, _ort_tid), tid in track_tids.items():
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"name": f"onnxruntime session {i}"},
            }
        )
    return profile_trace


def merge_ort_traces_into_profile(profile_path: str, ort_dir: str) -> None:
    """Read the onnxsim ``profile`` trace at ``profile_path`` and the onnxruntime
    session traces written under ``ort_dir``, merge the latter into the former
    (see :func:`merge_ort_traces`), and write the result back to ``profile_path``.

    Best-effort: a missing onnxsim trace or unreadable onnxruntime output leaves
    the onnxsim trace as-is rather than failing the caller.
    """
    if not os.path.exists(profile_path):
        return
    ort_paths = [
        os.path.join(ort_dir, name)
        for name in os.listdir(ort_dir)
        if name.endswith(".json")
    ]
    if not ort_paths:
        return
    # onnxruntime writes one file per session; mtime order is session order.
    ort_paths.sort(key=os.path.getmtime)
    with open(profile_path) as f:
        trace = json.load(f)
    merge_ort_traces(trace, ort_paths)
    with open(profile_path, "w") as f:
        json.dump(trace, f)
