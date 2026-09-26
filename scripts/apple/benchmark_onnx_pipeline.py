#!/usr/bin/env python3
"""Benchmark a staged ONNX deployment across Core ML and tinygrad Metal.

The JSON manifest keeps task-specific preprocessing outside the runtime. Each
stage is an ONNX model with an optional NPZ file containing its non-connected
inputs. Connections route named outputs from an earlier stage to named inputs
of a later stage, so the same runner can cover encoder/head, detector/backbone,
and other multi-graph pipelines.

Manifest example::

  {
    "stages": [
      {"name": "encoder", "model": "encoder.onnx", "feeds": "image.npz"},
      {"name": "head", "model": "head.onnx", "feeds": "prompts.npz"}
    ],
    "connections": [
      {"from": ["encoder", "features"], "to": ["head", "features"]}
    ],
    "backends": {"encoder": "coreml", "head": "tinygrad_metal_jit"}
  }

NPZ files can be created with ``numpy.savez(path, input_name=array, ...)``.
Without a feeds file, static inputs receive deterministic seeded values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from benchmark_sam_hybrid import (
    _coreml_runner,
    _inputs,
    _measure,
    _MetalRunner,
    _quality,
)
from onnx import compose

BACKENDS = ("coreml", "tinygrad_metal", "tinygrad_metal_jit")

# Backends tried, in order, when a stage's selected backend has no working
# runner. Metal JIT first: on the graphs measured here it is 8-20x faster than
# tinygrad's eager ONNX runner, which launches a kernel per op. Eager is the
# fallback because it still runs the graph when JIT compilation fails.
_FALLBACK_BACKENDS = ("tinygrad_metal_jit", "tinygrad_metal")


def _resolve_backends(
    names: List[str],
    backend_map: Dict[str, str],
    runners: Dict,
    fallback: bool,
) -> tuple[Dict[str, str], List[dict]]:
    """Pick each stage's backend, optionally substituting one that has a runner.

    Without ``fallback`` this is exactly the manifest's choice, so a stage whose
    backend failed to convert still fails the end-to-end measurement (the
    existing behaviour). With it, a stage whose selected backend has no runner
    takes the first available backend from :data:`_FALLBACK_BACKENDS` instead.

    Returns the per-stage choice plus one substitution record per stage that
    changed, so the report never presents a fallback as a clean measurement of
    the backend that was asked for.
    """
    selected = {name: backend_map.get(name, "coreml") for name in names}
    if any(backend not in BACKENDS for backend in selected.values()):
        raise ValueError(f"unsupported backend in {selected}")
    if not fallback:
        return selected, []
    substitutions = []
    for name in names:
        wanted = selected[name]
        if (name, wanted) in runners:
            continue
        for candidate in _FALLBACK_BACKENDS:
            if (name, candidate) in runners:
                selected[name] = candidate
                substitutions.append(
                    {
                        "stage": name,
                        "requested_backend": wanted,
                        "used_backend": candidate,
                    }
                )
                break
    return selected, substitutions


def _load_feeds(stage: dict, model_path: Path, seed: int) -> dict[str, np.ndarray]:
    feed_path = stage.get("feeds")
    if feed_path is None:
        return _inputs(onnx.load(model_path), seed)
    with np.load(model_path.parent / feed_path) as data:
        return {name: np.ascontiguousarray(data[name]) for name in data.files}


def _validate_input_connections(
    connections: List[dict],
    paths: Dict[str, Path],
    names: List[str],
    feeds: Dict[str, dict[str, np.ndarray]],
    stage_connections: List[dict],
) -> None:
    positions = {name: index for index, name in enumerate(names)}
    production_targets = {(edge["to"][0], edge["to"][1]) for edge in stage_connections}
    seen_targets = set()
    for edge in connections:
        source_stage, source_name = edge["from"]
        target_stage, target_name = edge["to"]
        if source_stage not in positions or target_stage not in positions:
            raise ValueError(f"unknown input connection stage in {edge!r}")
        if positions[source_stage] >= positions[target_stage]:
            raise ValueError("input connections must point forward in stage order")
        source_inputs = {
            value.name for value in onnx.load(paths[source_stage]).graph.input
        }
        target_inputs = {
            value.name for value in onnx.load(paths[target_stage]).graph.input
        }
        if source_name not in source_inputs or target_name not in target_inputs:
            raise ValueError(f"input connection names are not graph inputs: {edge!r}")
        if source_name not in feeds[source_stage]:
            raise ValueError(
                f"input connection source {source_name!r} is not supplied by "
                f"the {source_stage!r} feed"
            )
        target_key = (target_stage, target_name)
        if target_key in seen_targets or target_key in production_targets:
            raise ValueError(f"input connection target is already connected: {edge!r}")
        seen_targets.add(target_key)


def _runner(
    path: Path, backend: str, package: Path, compute_units: str, compute_precision: str
):
    if backend == "coreml":
        return _coreml_runner(path, compute_units, compute_precision, package)
    return _MetalRunner(path, jit=backend == "tinygrad_metal_jit")


def _ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [x.name for x in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds)))


def _fused_input_proxy(model: onnx.ModelProto, input_name: str) -> str:
    output_names = {value.name for value in model.graph.output}
    if input_name in output_names:
        return input_name
    if input_name not in {value.name for value in model.graph.input}:
        raise ValueError(f"input connection source {input_name!r} is not a graph input")
    output_name = f"{input_name}__fused_input"
    suffix = 0
    while output_name in output_names or any(
        value.name == output_name for value in model.graph.value_info
    ):
        suffix += 1
        output_name = f"{input_name}__fused_input_{suffix}"
    output = onnx.ValueInfoProto()
    output.CopyFrom(
        next(value for value in model.graph.input if value.name == input_name)
    )
    output.name = output_name
    model.graph.node.extend(
        [
            onnx.helper.make_node(
                "Identity", [input_name], [output_name], name=output_name
            )
        ]
    )
    model.graph.output.append(output)
    output_names.add(output_name)
    return output_name


def _fuse_stage_models(
    stage_names: List[str],
    models: Dict[str, onnx.ModelProto],
    connections: List[dict],
    input_connections: List[dict],
) -> Tuple[onnx.ModelProto, Dict[str, Tuple[str, str]], List[str]]:
    """Merge a linear stage chain while preserving every external input."""
    if len(stage_names) < 2:
        raise ValueError("fuse.stages must contain at least two stages")
    if len(set(stage_names)) != len(stage_names):
        raise ValueError("fuse.stages must not contain duplicates")
    if any(name not in models for name in stage_names):
        raise ValueError("fuse.stages contains an unknown stage")

    copied: Dict[str, onnx.ModelProto] = {}
    for name in stage_names:
        model = onnx.ModelProto()
        model.CopyFrom(models[name])
        del model.metadata_props[:]
        copied[name] = model

    source_proxies: Dict[Tuple[str, str], str] = {}
    for edge in input_connections:
        source_stage, source_name = edge["from"]
        if source_stage not in copied or source_stage not in stage_names:
            raise ValueError(f"unknown input connection source {source_stage!r}")
        source_proxies[(source_stage, source_name)] = _fused_input_proxy(
            copied[source_stage], source_name
        )

    prefixed: List[onnx.ModelProto] = []
    input_sources: Dict[str, Tuple[str, str]] = {}
    for index, name in enumerate(stage_names):
        prefix = f"fused_{index}_"
        model = compose.add_prefix(copied[name], prefix)
        prefixed.append(model)
        for value in model.graph.input:
            input_sources[value.name] = (name, value.name[len(prefix) :])

    result = prefixed[0]
    for index, current in enumerate(stage_names[1:], start=1):
        current_prefix = f"fused_{index}_"
        io_map = []
        target_inputs = set()
        for edge in connections:
            source_stage, source_output = edge["from"]
            target_stage, target_input = edge["to"]
            if target_stage != current:
                continue
            if source_stage not in stage_names[:index]:
                raise ValueError(
                    f"fusion connection source {source_stage!r} is not before {current!r}"
                )
            source_prefix = f"fused_{stage_names.index(source_stage)}_"
            io_map.append(
                (source_prefix + source_output, current_prefix + target_input)
            )
            target_inputs.add(current_prefix + target_input)
        for edge in input_connections:
            source_stage, source_input = edge["from"]
            target_stage, target_input = edge["to"]
            if target_stage != current:
                continue
            if source_stage not in stage_names[:index]:
                raise ValueError(
                    f"input connection source {source_stage!r} is not before {current!r}"
                )
            source_index = stage_names.index(source_stage)
            source_proxy = source_proxies[(source_stage, source_input)]
            io_map.append(
                (f"fused_{source_index}_{source_proxy}", current_prefix + target_input)
            )
            target_inputs.add(current_prefix + target_input)
        if not io_map:
            raise ValueError(f"fuse has no connection to {current!r}")
        if len(target_inputs) != len(io_map):
            raise ValueError(f"fuse has duplicate connections into {current!r}")
        result = compose.merge_models(
            result,
            prefixed[index],
            io_map,
            name=f"{result.graph.name}_{current}",
        )
        for _, target in io_map:
            input_sources.pop(target, None)

    final_output_names = [value.name for value in prefixed[-1].graph.output]
    final_outputs = [
        value for value in result.graph.output if value.name in set(final_output_names)
    ]
    if len(final_outputs) != len(final_output_names):
        raise ValueError("fusion lost one or more final outputs")
    del result.graph.output[:]
    result.graph.output.extend(final_outputs)
    onnx.checker.check_model(result)
    actual_inputs = {value.name for value in result.graph.input}
    input_sources = {
        name: source for name, source in input_sources.items() if name in actual_inputs
    }
    return (
        result,
        input_sources,
        [value.name for value in models[stage_names[-1]].graph.output],
    )


def run_pipeline(
    manifest_path: Path,
    output_path: Path,
    compute_units: str,
    compute_precision: str,
    warmup: int,
    repeats: int,
    fallback: bool = False,
) -> dict:
    spec = json.loads(manifest_path.read_text())
    stages = spec["stages"]
    if not stages:
        raise ValueError("pipeline manifest must contain at least one stage")
    names = [stage["name"] for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("stage names must be unique")
    paths = {
        stage["name"]: (manifest_path.parent / stage["model"]).resolve()
        for stage in stages
    }
    output_names = {
        name: [value.name for value in onnx.load(path).graph.output]
        for name, path in paths.items()
    }
    feeds = {
        stage["name"]: _load_feeds(stage, paths[stage["name"]], i + 1)
        for i, stage in enumerate(stages)
    }
    connections = spec.get("connections", [])
    input_connections = spec.get("input_connections", [])
    _validate_input_connections(input_connections, paths, names, feeds, connections)
    backend_map = spec.get("backends", {})
    for stage_name, backend in backend_map.items():
        if stage_name not in names or backend not in BACKENDS:
            raise ValueError(f"invalid backend assignment {stage_name}: {backend}")

    refs: dict[str, dict[str, np.ndarray]] = {}
    stage_records = {}
    package_dir = output_path.with_suffix("").with_name(output_path.stem + "_coreml")
    package_dir.mkdir(parents=True, exist_ok=True)
    runners = {}
    for stage in stages:
        name, path = stage["name"], paths[stage["name"]]
        stage_feeds = dict(feeds[name])
        for edge in connections:
            if edge["to"][0] == name:
                source_name, source_output = edge["from"]
                if source_name not in refs:
                    raise ValueError("connections must follow stage order")
                stage_feeds[edge["to"][1]] = refs[source_name][source_output]
        for edge in input_connections:
            if edge["to"][0] == name:
                source_stage, source_input = edge["from"]
                if source_stage not in feeds:
                    raise ValueError("input connections must follow stage order")
                stage_feeds[edge["to"][1]] = feeds[source_stage][source_input]
        refs[name] = _ort(path, stage_feeds)
        record = {
            "input_shapes": {k: list(v.shape) for k, v in stage_feeds.items()},
            "backends": {},
        }
        for backend in BACKENDS:
            try:
                runner = _runner(
                    path,
                    backend,
                    package_dir / f"{name}.mlpackage",
                    compute_units,
                    compute_precision,
                )
                runners[name, backend] = runner
                out, timing = _measure(runner, stage_feeds, warmup, repeats)
                reference = list(refs[name].values())
                record["backends"][backend] = {
                    **timing,
                    "vs_ort": _quality(out, reference, output_names[name]),
                }
            except Exception as exc:
                record["backends"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
        stage_records[name] = record

    selected, substitutions = _resolve_backends(names, backend_map, runners, fallback)

    fused_result = None
    fuse_spec = spec.get("fuse")
    if fuse_spec is not None:
        try:
            if not isinstance(fuse_spec, dict):
                raise ValueError("fuse must be an object")
            fuse_stages = list(fuse_spec.get("stages", []))
            fuse_backend = fuse_spec.get("backend")
            if not fuse_stages or fuse_backend not in BACKENDS:
                raise ValueError("fuse.stages and a valid fuse.backend are required")
            indices = [names.index(name) for name in fuse_stages]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("fuse.stages must be a contiguous manifest range")
            if any(selected[name] != fuse_backend for name in fuse_stages):
                raise ValueError("all fused stages must use the same backend")
            for edge in input_connections:
                if edge["to"][0] in fuse_stages and edge["from"][0] not in fuse_stages:
                    raise ValueError(
                        "an input connection into a fused stage must originate "
                        "inside fuse.stages"
                    )
            fused_input_connections = [
                edge
                for edge in input_connections
                if edge["from"][0] in fuse_stages and edge["to"][0] in fuse_stages
            ]
            models = {name: onnx.load(paths[name]) for name in fuse_stages}
            fused_model, input_sources, fused_output_names = _fuse_stage_models(
                fuse_stages, models, connections, fused_input_connections
            )
            fused_path = output_path.with_suffix(".fused.onnx")
            onnx.save(fused_model, str(fused_path))
            fused_feeds = {
                fused_name: feeds[source_stage][source_name]
                for fused_name, (source_stage, source_name) in input_sources.items()
            }
            fused_ref = list(
                ort.InferenceSession(
                    str(fused_path), providers=["CPUExecutionProvider"]
                ).run(None, fused_feeds)
            )
            fused_runner = _runner(
                fused_path,
                fuse_backend,
                package_dir / "fused.mlpackage",
                compute_units,
                compute_precision,
            )
            fused_out, fused_timing = _measure(
                fused_runner, fused_feeds, warmup, repeats
            )
            fused_result = {
                "status": "ok",
                "stages": fuse_stages,
                "backend": fuse_backend,
                "model": str(fused_path),
                **fused_timing,
                "vs_ort": _quality(fused_out, fused_ref, fused_output_names),
            }
        except Exception as exc:
            fused_result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    pipeline_result = None
    try:
        unavailable = [
            f"{name}={backend}: {stage_records[name]['backends'].get(backend, {}).get('error', 'runner unavailable')}"
            for name, backend in selected.items()
            if (name, backend) not in runners
        ]
        if unavailable:
            raise RuntimeError(
                "selected backend unavailable: " + "; ".join(unavailable)
            )
        selected_runners = {
            stage["name"]: runners[stage["name"], selected[stage["name"]]]
            for stage in stages
        }

        def execute(_):
            values: dict[str, dict[str, np.ndarray]] = {}
            for stage in stages:
                name = stage["name"]
                stage_feeds = dict(feeds[name])
                for edge in connections:
                    if edge["to"][0] == name:
                        stage_feeds[edge["to"][1]] = values[edge["from"][0]][
                            edge["from"][1]
                        ]
                for edge in input_connections:
                    if edge["to"][0] == name:
                        stage_feeds[edge["to"][1]] = feeds[edge["from"][0]][
                            edge["from"][1]
                        ]
                result = selected_runners[name](stage_feeds)
                values[name] = dict(zip(output_names[name], result))
            last = stages[-1]["name"]
            return list(values[last].values())

        out, timing = _measure(execute, {}, warmup, repeats)
        last_name = stages[-1]["name"]
        pipeline_result = {
            "status": "ok",
            "backends": selected,
            **timing,
            "vs_ort": _quality(
                out,
                list(refs[last_name].values()),
                output_names[last_name],
            ),
        }
    except Exception as exc:
        pipeline_result = {
            "status": "error",
            "backends": selected,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "pipeline": spec.get("name", manifest_path.stem),
        "compute_units": compute_units,
        "compute_precision": compute_precision,
        "backend_fallbacks": substitutions,
        "stages": stage_records,
        "end_to_end": pipeline_result,
        "fused": fused_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON ONNX pipeline manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE"],
        default="ALL",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["DEFAULT", "FLOAT16", "FLOAT32"],
        default="DEFAULT",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--fallback-to-tinygrad",
        action="store_true",
        help="when a stage's selected backend has no working runner (e.g. Core ML "
        "cannot lower a RoiAlign or NonZero), time it on a tinygrad Metal backend "
        "instead -- Metal JIT first, then eager -- and record the substitution "
        "under 'backend_fallbacks' in the report",
    )
    args = parser.parse_args()
    result = run_pipeline(
        args.manifest,
        args.output,
        args.compute_units,
        args.compute_precision,
        args.warmup,
        args.repeats,
        args.fallback_to_tinygrad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
