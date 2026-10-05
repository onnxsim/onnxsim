#!/usr/bin/env python3
"""Fill in Acuity's generated `<model>_inputmeta.yml` (runs inside the toolkit environment, which provides `acuitylib`).

    acuity_inputmeta.py MODEL [--dataset dataset.txt] [--mean 0,0,0] [--scale 0.0039216] [--reverse-channel]
                              [--preproc IMAGE_RGB|TENSOR|...] [--postproc]

Defaults keep the ONNX contract: no preprocess node (`TENSOR`), float in / float out as seen by the runner. Pass --preproc IMAGE_RGB
(with --mean/--scale) to bake image preprocessing into the NBG, which lets the runner feed raw uint8 camera frames.
Uses the same acuitylib calls as the Allwinner awnpu_model_zoo `config_yml.py`.
"""

import argparse
import os
import sys


def floats(text):
    return [float(x) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--dataset")
    p.add_argument(
        "--dataset-type",
        default="TEXT",
        help="TEXT, NPY, H5FS, SQLITE, LMDB, GENERATOR or ZIP",
    )
    p.add_argument("--mean", type=floats)
    p.add_argument("--scale", type=floats)
    p.add_argument("--reverse-channel", action="store_true", help="swap RGB/BGR")
    p.add_argument("--preproc", default="TENSOR")
    p.add_argument(
        "--postproc",
        action="store_true",
        help="add a dequantizing output node (float32 outputs from the NBG itself)",
    )
    a = p.parse_args()

    from acuitylib.vsi_nn import VSInn

    nn = VSInn()
    net = nn.create_net()
    model, inputmeta, postprocess = (
        a.model + ".json",
        a.model + "_inputmeta.yml",
        a.model + "_postprocess_file.yml",
    )
    for f in (model, inputmeta):
        if not os.path.exists(f):
            print(f"{f} does not exist", file=sys.stderr)
            return 1
    nn.load_model(net, model)
    nn.load_model_inputmeta(net, inputmeta)

    meta = net.get_input_meta()
    for database in meta.databases:
        for port in database.ports:
            if a.mean is not None:
                port.preprocess["mean"] = a.mean
            if a.scale is not None:
                port.preprocess["scale"] = a.scale[0] if len(a.scale) == 1 else a.scale
            port.preprocess["reverse_channel"] = a.reverse_channel
            params = port.preprocess["preproc_node_params"]
            params["add_preproc_node"] = a.preproc != "TENSOR"
            params["preproc_type"] = a.preproc
    net.update_input_meta(meta)
    if a.dataset:
        nn.set_database(net, dataset_files=a.dataset, dataset_type=a.dataset_type)
    nn.save_model_inputmeta(net, inputmeta)

    if a.postproc:
        with open(postprocess) as f:
            text = f.read()
        with open(postprocess, "w") as f:
            f.write(text.replace("add_postproc_node: false", "add_postproc_node: true"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
