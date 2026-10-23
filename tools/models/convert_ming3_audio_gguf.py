#!/usr/bin/env python3
"""Package Ming3 Audio Tiny, stacking routed experts for indexed GGML matmuls."""

import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import gguf
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import torch

from convert_ming3_tts_gguf import campplus_weights


def stack_thinker(source, destination, config):
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    headers = {}
    for name in dict.fromkeys(weight_map.values()):
        handles[name] = safe_open(source / name, framework="pt", device="cpu")
        with (source / name).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
            header.pop("__metadata__", None)
            headers.update(header)
    text = config["text_config"]
    layers = text["num_hidden_layers"]
    plan = []
    for name in weight_map:
        layer = re.match(r"model\.layers\.(\d+)\.", name)
        if layer and int(layer[1]) >= layers:
            continue
        if ".mlp.experts." not in name:
            plan.append((name, headers[name]["shape"], headers[name]["dtype"], name))
    for layer in range(text["first_k_dense_replace"], layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names = [f"model.layers.{layer}.mlp.experts.{i}.{projection}.weight"
                     for i in range(text["num_experts"])]
            first = headers[names[0]]
            for name in names:
                if headers[name]["shape"] != first["shape"] or headers[name]["dtype"] != first["dtype"]:
                    raise ValueError(f"Inconsistent expert bank at {name}")
            plan.append((f"model.layers.{layer}.mlp.experts.{projection}.weight",
                         [len(names), *first["shape"]], first["dtype"], names))
    writer = gguf.GGUFWriter(destination, "ming3_audio")
    writer.add_string("audiocpp.family", "ming3_audio")
    writer.add_array("audiocpp.tensor_names", [name for name, *_ in plan])
    writer.add_key_value("audiocpp.tensor_ranks", [len(shape) for _, shape, _, _ in plan],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_key_value("audiocpp.tensor_shapes", [dim for _, shape, _, _ in plan for dim in shape],
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)
    dtypes = {"BF16": (np.uint16, gguf.GGMLQuantizationType.BF16),
              "F16": (np.float16, gguf.GGMLQuantizationType.F16),
              "F32": (np.float32, gguf.GGMLQuantizationType.F32)}
    for i, (_, shape, dtype, _) in enumerate(plan):
        numpy_type, raw_type = dtypes[dtype]
        writer.add_tensor_info(f"_audiocpp.{i}", shape, np.dtype(numpy_type),
                               math.prod(shape) * np.dtype(numpy_type).itemsize, raw_dtype=raw_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    try:
        for i, (name, shape, dtype, source_names) in enumerate(plan):
            numpy_type = dtypes[dtype][0]
            if isinstance(source_names, list):
                values = np.empty(shape, dtype=numpy_type)
                for expert, source_name in enumerate(source_names):
                    tensor = handles[weight_map[source_name]].get_tensor(source_name)
                    values[expert] = tensor.view(torch.uint16).numpy() if dtype == "BF16" else tensor.numpy()
            else:
                tensor = handles[weight_map[source_names]].get_tensor(source_names)
                values = tensor.view(torch.uint16).numpy() if dtype == "BF16" else tensor.numpy()
            writer.write_tensor_data(np.ascontiguousarray(values))
            if i % 100 == 0 or isinstance(source_names, list):
                print(f"{i + 1}/{len(plan)} {name}", flush=True)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("orig", "q8_0", "q4_k"), default="orig",
                        help="Quantize Thinker experts; Q4_K also uses a Q8_0 text output head")
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--temporary-dir", type=Path, required=True,
                        help="Scratch filesystem with space for one unquantized Thinker GGUF")
    parser.add_argument("--log", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "bailing_moe_v3_audio":
        raise ValueError("Expected the official Ming3 Audio checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.temporary_dir.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[2] / "model_specs/ming3_audio.json"
    with tempfile.TemporaryDirectory(prefix="ming3-audio-", dir=args.temporary_dir) as temporary:
        root = Path(temporary)
        thinker = root / "thinker.gguf"
        stack_thinker(source, thinker, config)
        speaker = root / "speaker.safetensors"
        save_file(campplus_weights(source / "talker/campplus.onnx"), speaker)
        command = [str(args.converter.resolve()), "--input", f"thinker={thinker}",
                   "--input", f"talker={source / 'talker/model.safetensors'}",
                   "--input", f"vae={source / 'talker/vae/model.safetensors'}",
                   "--input", f"speaker={speaker}", "--root", str(source),
                   "--model-spec", str(spec), "--family", "ming3_audio",
                   "--output", str(args.output.resolve()), "--type", "orig", "--log",
                   "--keep-type", "speaker*=f32", "--keep-type", "talker*=orig",
                   "--keep-type", "vae*=orig", "--keep-type", "thinker/model.audio*=orig",
                   "--keep-type", "thinker/linear_proj_audio*=orig"]
        for layer in range(config["text_config"]["first_k_dense_replace"], config["text_config"]["num_hidden_layers"]):
            command += ["--keep-type", f"thinker/model.layers.{layer}.mlp.experts.*={args.type}"]
            command += ["--keep-type", f"thinker/model.layers.{layer}.mlp.gate*=f32"]
        if args.type == "q4_k":
            command += ["--keep-type", "thinker/lm_head.weight=q8_0"]
        for relative in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                         "chat_template.jinja", "special_tokens_map.json", "talker/config.json",
                         "talker/llm/config.json", "talker/llm/tokenizer.json", "talker/llm/tokenizer_config.json",
                         "talker/vae/config.json"):
            command += ["--sidecar", f"{source / relative}={relative}"]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
