#!/usr/bin/env python3
"""Package Ming's model, tokenizer and native CAMPPlus speaker weights in one GGUF."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def campplus_weights(path):
    model = onnx.load(path)
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)

    weights = {}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "BatchNormalization"):
            continue
        name = re.sub(r"/(layer\d+)/\1\.(\d+)", r"/\1/\2", node.name)
        name = re.sub(r"/shortcut/shortcut\.(\d+)", r"/shortcut/\1", name)
        prefix = "speaker_encoder." + name.rsplit("/", 1)[0].strip("/").replace("/", ".")
        suffixes = ("weight", "bias") if node.op_type == "Conv" else (
            "weight", "bias", "running_mean", "running_var")
        if len(node.input) not in ((2, 3) if node.op_type == "Conv" else (5,)):
            raise ValueError(f"Unsupported parameter count: {node.name}")
        for source, suffix in zip(node.input[1:], suffixes):
            value = constants[source]
            if value.dtype != np.float32:
                raise ValueError(f"CAMPPlus parameter {source} must be float32, got {value.dtype}")
            target = prefix + "." + suffix
            if target in weights:
                raise ValueError(f"Duplicate CAMPPlus parameter: {target}")
            weights[target] = np.ascontiguousarray(value)
        if node.op_type == "BatchNormalization":
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if abs(attrs.get("epsilon", 1e-5) - 1e-5) > 1e-10 or attrs.get("training_mode", 0):
                raise ValueError(f"Unsupported CAMPPlus batch normalization: {node.name}")
    if weights["speaker_encoder.xvector.dense.linear.weight"].shape != (192, 1024, 1):
        raise ValueError("Ming requires the published 192-dimensional CAMPPlus encoder")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Published Ming checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Output GGUF file")
    parser.add_argument("--type", choices=("orig", "q8_0"), default="orig")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    source = args.source.resolve()
    weights = campplus_weights(source / "campplus.onnx")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[2] / "model_specs/ming3_tts.json"
    with tempfile.TemporaryDirectory(prefix=".ming-convert-", dir=args.output.parent) as temporary:
        root = Path(temporary)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(source / name, root / name)
        speaker = root / "speaker.safetensors"
        save_file(weights, speaker)
        command = [str(args.converter.resolve()), "--input", str(source / "model.safetensors"),
                   "--input", f"speaker={speaker}", "--root", str(root), "--model-spec", str(spec),
                   "--type", args.type, "--output", str(args.output.resolve()),
                   "--keep-type", "speaker.*=f32", "--keep-type", "stop_head*=bf16"]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
