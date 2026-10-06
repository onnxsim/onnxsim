#!/usr/bin/env python3
"""ONNX -> NBG (Network Binary Graph) for the Allwinner NPU, as an `onnx-remote-compiler` command.

    onnx-remote-compiler --port 39510 --cache-dir ~/.cache/onnxsim-aw --target allwinner-a733 \\
      --compiler-id acuity-6.x --command 'python3 scripts/allwinner/compile_nbg.py {input} {output} {manifest} --platform a733'

The artifact is the `.nb` that `onnx-remote-viplite-worker` (tools/onnx-remote) loads and runs.

This drives VeriSilicon's Acuity toolkit (`pegasus`) the way Allwinner's awnpu_model_zoo scripts do -- import, inputmeta,
quantize, `export ovxlib --pack-nbg-unify` -- because the NBG is machine code for the Vivante VIP9000 core and Acuity is the only
compiler for it. The toolkit is not redistributable; Allwinner ships it as a Docker image (`ubuntu-npu:v2.0.x`) through its
customer portal. Point this script at either:

  * a native install:   ACUITY_PATH (dir holding `pegasus`) and VIV_SDK are set, as the zoo's scripts expect, or
  * the Docker image:   --docker-image IMAGE (or AW_NPU_DOCKER_IMAGE). The image presets ACUITY_PATH/VIV_SDK itself.

Calibration (for every --quant except `float`): --calib-dir DIR holding float32 `.npy` samples shaped like the model input.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# `pegasus export ovxlib --optimize` value per SoC (Allwinner awnpu_model_zoo, pegasus_export_ovx_nbg.sh).
PLATFORM_OPTIMIZE = {
    "v853": "VIP9000PICO_PID0XEE",
    "v85x": "VIP9000PICO_PID0XEE",
    "r853": "VIP9000PICO_PID0XEE",
    "t527": "VIP9000NANOSI_PLUS_PID0X10000016",
    "mr527": "VIP9000NANOSI_PLUS_PID0X10000016",
    "a733": "VIP9000NANODI_PLUS_PID0X1000003B",
    "t736": "VIP9000NANODI_PLUS_PID0X1000003B",
    "t536": "VIP9000NANODI_PLUS_PID0X1000003B",
    "mr536": "VIP9000NANODI_PLUS_PID0X1000003B",
}

# --quant -> (`pegasus quantize --quantizer`, --qtype, --dtype for export). `float` skips quantization.
QUANT = {
    "uint8": ("asymmetric_affine", "uint8"),
    "pcq": ("perchannel_symmetric_affine", "int8"),
    "int16": ("dynamic_fixed_point", "int16"),
    "bf16": ("qbfloat16", "qbfloat16"),
}

ONNX_DTYPE = {
    1: "float32",
    2: "uint8",
    3: "int8",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
}


class CompileError(RuntimeError):
    pass


def model_io(model):
    """(inputs, outputs) as [{name, dtype, shape}], excluding initializers; raises on a dynamic input shape."""
    init = {i.name for i in model.graph.initializer}

    def describe(v):
        tt = v.type.tensor_type
        shape = [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
        return {
            "name": v.name,
            "dtype": ONNX_DTYPE.get(tt.elem_type, str(tt.elem_type)),
            "shape": shape,
        }

    return [describe(v) for v in model.graph.input if v.name not in init], [
        describe(v) for v in model.graph.output
    ]


def parse_input_shapes(specs):
    shapes = {}
    for spec in specs or []:
        name, _, dims = spec.rpartition(":")
        if not name or not dims:
            raise CompileError(f"--input-shape wants NAME:D0,D1,...; got {spec!r}")
        shapes[name] = [int(d) for d in dims.split(",")]
    return shapes


def _load_npu_rewrite():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "npu_rewrite", Path(__file__).with_name("npu_rewrite.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_model(src, dst, shapes, simplify, rewrite_ops=True):
    """Write a static-shape, simplified, operator-rewritten copy of the model to dst.

    Returns (inputs, outputs, rewrites): the model's I/O and a {name: count} of the rewrites applied (npu_rewrite.py: LayerNormalization,
    Gelu and Div are replaced by operators Allwinner documents for the ONNX importer; a model without them is unchanged)."""
    import onnx

    model = onnx.load(src)
    if shapes:
        for v in model.graph.input:
            if v.name in shapes:
                del v.type.tensor_type.shape.dim[:]
                for d in shapes[v.name]:
                    v.type.tensor_type.shape.dim.add().dim_value = d
    if simplify:
        import onnxsim

        model, ok = onnxsim.simplify(model)
        if not ok:
            raise CompileError(
                "onnxsim could not validate the simplified model (use --no-simplify to skip)"
            )
    inputs, outputs = model_io(model)
    for i in inputs:
        if None in i["shape"] or not i["shape"]:
            raise CompileError(
                f"input {i['name']!r} has a dynamic shape {i['shape']}; Acuity needs static shapes (use --input-shape {i['name']}:1,3,224,224)"
            )
    rewrites = {}
    if rewrite_ops:
        try:
            model, stats = _load_npu_rewrite().rewrite(model)
        except ValueError as e:
            raise CompileError(str(e)) from e
        rewrites = dict(stats)
    onnx.save(model, dst)
    return inputs, outputs, rewrites


def write_calibration(calib_dir, work, count):
    """dataset.txt for `pegasus quantize`: one .npy path per line. Returns (dataset path, sample count)."""
    files = sorted(Path(calib_dir).glob("*.npy"))[:count]
    if not files:
        raise CompileError(
            f"--calib-dir {calib_dir} has no .npy samples (needed to quantize; or pass --quant float)"
        )
    ds = work / "dataset"
    ds.mkdir()
    lines = []
    for f in files:
        shutil.copy(f, ds / f.name)
        lines.append(f"./dataset/{f.name}")
    (work / "dataset.txt").write_text("\n".join(lines) + "\n")
    return work / "dataset.txt", len(files)


def stage_hybrid_layers(a, work):
    """Validate --hybrid-layers and stage it as hybrid_layer.txt, each line indented four spaces as the zoo's config_hybrid_layer.py
    expects (it is pasted under `customized_quantize_layers:` in the .quantize file). Returns the number of layers."""
    if not a.hybrid_layers:
        return 0
    if a.quant == "float":
        raise CompileError(
            "--hybrid-layers needs a quantized --quant mode: it names layers to keep at higher precision than that mode"
        )
    lines = []
    for number, raw in enumerate(Path(a.hybrid_layers).read_text().splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        name, sep, dtype = text.rpartition(": ")
        if not sep or not name.strip() or not dtype.strip():
            raise CompileError(
                f"{a.hybrid_layers}:{number}: expected 'layer_name: dtype' (e.g. 'x_output_0_12: dynamic_fixed_point-i16'), got {text!r}"
            )
        lines.append(f"    {name.strip()}: {dtype.strip()}\n")
    if not lines:
        raise CompileError(f"{a.hybrid_layers} names no layers")
    (work / "hybrid_layer.txt").write_text("".join(lines))
    return len(lines)


HYBRID_INJECT = """python3 - <<'PYEOF'
src = open("model_{q}.quantize").read()
marker = "customized_quantize_layers: {{}}"
assert marker in src, "no customized_quantize_layers placeholder in model_{q}.quantize"
layers = ("customized_quantize_layers:\\n" + open("hybrid_layer.txt").read()).rstrip("\\n")
open("model_{q}_hybrid.quantize", "w").write(src.replace(marker, layers))
PYEOF"""


def acuity_script(a, optimize, n_calib, n_hybrid=0):
    """The shell script run inside the toolkit environment, in the work dir. Mirrors the zoo's pegasus_*.sh scripts."""
    q = shlex.quote
    steps = [
        "set -e",
        'trap "chmod -R a+rwX . 2>/dev/null || true" EXIT',  # a Docker run leaves root-owned files behind
        ': "${ACUITY_PATH:?ACUITY_PATH is not set (install Acuity, or use --docker-image)}"',
        'PEGASUS="$ACUITY_PATH/pegasus"; [ -e "$PEGASUS" ] || PEGASUS="python3 $PEGASUS.py"',
        "$PEGASUS import onnx --model model.onnx --output-model model.json --output-data model.data",
        "$PEGASUS generate inputmeta --model model.json --separated-database --input-meta-output model_inputmeta.yml",
        "$PEGASUS generate postprocess-file --model model.json --postprocess-file-output model_postprocess_file.yml",
        f"{q(a.acuity_python)} acuity_inputmeta.py model"
        + (" --dataset dataset.txt" if n_calib else "")
        + a.inputmeta_args,
    ]
    export = [
        "$PEGASUS export ovxlib --pack-nbg-unify",
        f"--optimize {q(optimize)}",
        '--viv-sdk "$VIV_SDK"',
        "--model model.json --model-data model.data",
        "--target-ide-project linux64",
        "--with-input-meta model_inputmeta.yml --postprocess-file model_postprocess_file.yml",
    ]
    if a.quant == "float":
        export += ["--dtype float", "--output-path wksp/out/out"]
    else:
        quantizer, qtype = QUANT[a.quant]
        common = (
            "$PEGASUS quantize --model model.json --model-data model.data --device CPU --with-input-meta model_inputmeta.yml "
            f"--iterations {n_calib}"
        )
        tail = f"--quantizer {quantizer} --qtype {qtype}"
        steps.append(
            f"{common} --rebuild --model-quantize model_{a.quant}.quantize {tail}"
        )
        model_json, quantize_file = "model.json", f"model_{a.quant}.quantize"
        if n_hybrid:
            # The zoo's pegasus_quantize-hybrid.sh / pegasus_export_ovx_nbg-hybrid.sh: paste the layer list into a copy of the
            # .quantize file, quantize again with --hybrid, and export from the hybrid files.
            steps.append(HYBRID_INJECT.format(q=a.quant))
            steps.append(
                f"{common} --hybrid --model-quantize model_{a.quant}_hybrid.quantize {tail}"
            )
            steps.append("export VSI_NN_ENABLE_OPCHECK=0")
            model_json = f"model_{a.quant}_hybrid.quantize.json"
            quantize_file = f"model_{a.quant}_hybrid.quantize"
        export = [
            e.replace("--model model.json ", f"--model {model_json} ") for e in export
        ]
        export += [
            "--dtype quantized",
            f"--model-quantize {quantize_file}",
            "--output-path wksp/out/out",
        ]
    steps.append(" ".join(export))
    return "\n".join(steps) + "\n"


def run_toolkit(script, work, a):
    if a.docker_image:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--ipc=host",
            "-v",
            f"{work}:/workspace",
            "-w",
            "/workspace",
            a.docker_image,
            "bash",
            "-c",
            script,
        ]
    else:
        for var in ("ACUITY_PATH", "VIV_SDK"):
            if not os.environ.get(var):
                raise CompileError(
                    f"{var} is not set: install Acuity (as the awnpu_model_zoo scripts expect) or pass --docker-image"
                )
        cmd = ["bash", "-c", script]
    proc = subprocess.run(
        cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (work / "acuity.log").write_text(proc.stdout)
    if proc.returncode:
        tail = "\n".join(proc.stdout.splitlines()[-30:])
        raise CompileError(f"Acuity failed (exit {proc.returncode}):\n{tail}")


def compile_model(a):
    optimize = PLATFORM_OPTIMIZE.get(a.platform)
    if optimize is None:
        raise CompileError(
            f"unknown --platform {a.platform!r}; known: {', '.join(sorted(PLATFORM_OPTIMIZE))}"
        )
    if a.quant != "float" and a.quant not in QUANT:
        raise CompileError(f"unknown --quant {a.quant!r}")
    work = Path(tempfile.mkdtemp(prefix="onnxsim-aw-", dir=os.environ.get("TMPDIR")))
    try:
        inputs, outputs, rewrites = prepare_model(
            a.input,
            work / "model.onnx",
            parse_input_shapes(a.input_shape),
            not a.no_simplify,
            not a.no_rewrite,
        )
        n_hybrid = stage_hybrid_layers(a, work)
        n_calib = 0
        if a.quant != "float":
            if len(inputs) != 1:
                raise CompileError(
                    f"quantizing a {len(inputs)}-input model is not supported (single-input calibration only); use --quant float"
                )
            if not a.calib_dir:
                raise CompileError(
                    "--calib-dir is required to quantize (float32 .npy samples), or pass --quant float"
                )
            _, n_calib = write_calibration(a.calib_dir, work, a.calib_count)
        shutil.copy(
            Path(__file__).with_name("acuity_inputmeta.py"),
            work / "acuity_inputmeta.py",
        )
        run_toolkit(acuity_script(a, optimize, n_calib, n_hybrid), work, a)
        nbs = list((work / "wksp").rglob("network_binary.nb"))
        if len(nbs) != 1:
            raise CompileError(
                f"expected one network_binary.nb under wksp/, found {len(nbs)} (see {work / 'acuity.log'})"
            )
        shutil.copyfile(nbs[0], a.output)
        manifest = {
            "schema_version": 1,
            "compiler": {
                "name": "acuity-pegasus",
                "version": a.compiler_version or "acuity",
                "id": a.compiler_id or "unidentified",
            },
            "target": {
                "backend": "viplite",
                "device": a.platform,
                "optimize": optimize,
            },
            "artifact": {"format": "nbg", "abi": "viplite"},
            "quantization": {
                "mode": a.quant,
                "calibration_samples": n_calib,
                "hybrid_layers": n_hybrid,
            },
            "rewrites": rewrites,
            # The runner converts float32 <-> the NBG's own formats from the quantization stored in the NBG.
            "io": {"dtype": "float32", "inputs": inputs, "outputs": outputs},
            "capabilities": {"ops": [], "dtypes": ["float32", "uint8"]},
            "legalization": {"profile": "acuity", "version": 1},
        }
        Path(a.manifest).write_text(json.dumps(manifest, indent=1))
    finally:
        if os.environ.get("ONNXSIM_AW_KEEP_WORK"):
            print(f"kept work dir {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="ONNX model")
    p.add_argument("output", help="where to write the .nb artifact")
    p.add_argument("manifest", help="where to write the JSON manifest")
    p.add_argument(
        "--platform",
        default=os.environ.get("AW_PLATFORM", "a733"),
        help="SoC (default a733): " + ", ".join(sorted(PLATFORM_OPTIMIZE)),
    )
    p.add_argument(
        "--quant",
        default=os.environ.get("AW_QUANT", "pcq"),
        help="pcq (default), uint8, int16, bf16 or float",
    )
    p.add_argument(
        "--calib-dir",
        default=os.environ.get("AW_CALIB_DIR"),
        help="directory of float32 .npy calibration samples",
    )
    p.add_argument(
        "--calib-count",
        type=int,
        default=100,
        help="max calibration samples (default 100)",
    )
    p.add_argument(
        "--input-shape",
        action="append",
        metavar="NAME:D0,D1,..",
        help="fix a dynamic input shape",
    )
    p.add_argument(
        "--no-simplify", action="store_true", help="skip onnxsim before import"
    )
    p.add_argument(
        "--no-rewrite",
        action="store_true",
        help="skip npu_rewrite.py (LayerNormalization/Gelu/Div -> documented operators)",
    )
    p.add_argument(
        "--hybrid-layers",
        metavar="FILE",
        help="hybrid quantization: lines of 'layer_name: dtype' (e.g. dynamic_fixed_point-i16) naming layers of the imported "
        "graph to quantize at higher precision, as in the zoo's hybrid_layer.txt",
    )
    p.add_argument(
        "--docker-image",
        default=os.environ.get("AW_NPU_DOCKER_IMAGE"),
        help="run Acuity in this image instead of a native install",
    )
    p.add_argument(
        "--acuity-python",
        default=os.environ.get("AW_ACUITY_PYTHON", "python3"),
        help="python that has acuitylib (default python3)",
    )
    p.add_argument(
        "--inputmeta-args",
        default="",
        help="extra args for acuity_inputmeta.py, e.g. ' --mean 0,0,0 --scale 0.0039216 --preproc IMAGE_RGB'",
    )
    p.add_argument("--compiler-id", default=os.environ.get("AW_COMPILER_ID"))
    p.add_argument("--compiler-version", default=os.environ.get("AW_COMPILER_VERSION"))
    a = p.parse_args(argv)
    try:
        compile_model(a)
    except CompileError as e:
        print(f"compile_nbg: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
