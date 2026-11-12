#!/usr/bin/env python3
"""Convert the public non-streaming Kroko ONNX transducer package to GGUF."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
from safetensors.numpy import save_file

from convert_kroko_onnx import convert_graph, decoder_metadata

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory containing encoder/decoder/joiner.onnx and tokens.txt")
    parser.add_argument("output", type=Path, help="Output GGUF file")
    parser.add_argument("--type", choices=("f32", "f16", "q8_0"), default="f16")
    parser.add_argument("--audiocpp-gguf", type=Path, default=REPO / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {output}")
    encoder = onnx.load(root / "encoder.onnx", load_external_data=False)
    metadata = {item.key: item.value for item in encoder.metadata_props}
    if metadata.get("model_type") != "zipformer2" or metadata.get("comment") != "non-streaming zipformer2":
        raise ValueError("expected the public non-streaming Zipformer2 encoder export")
    layers = []
    for tensor in encoder.graph.initializer:
        match = re.fullmatch(r"(encoder\.encoders\.(\d+)(?:\.encoder)?\.layers\.(\d+))\.norm\.bias", tensor.name)
        if match:
            layers.append((int(match[2]), int(match[3]), match[1]))
    prefixes = [name for _, _, name in sorted(layers)]
    if not prefixes:
        raise ValueError("encoder export contains no named Zipformer layers")
    tensors = convert_graph(root / "encoder.onnx", "encoder", linear_pos_prefixes=prefixes, streaming=False)
    tensors.update(convert_graph(root / "decoder.onnx", "decoder"))
    tensors.update(convert_graph(root / "joiner.onnx", "joiner"))
    # Recover architectural dimensions from the exported parameters, not the model name.
    config = {"audiocpp_family": "kroko_asr", "model_type": "zipformer2", "streaming": False,
              "variant": root.name, "language": {"iso": "en"}, "sample_rate": 16000,
              "feature_dim": 80, "subsampling_factor": 4, **decoder_metadata(root / "decoder.onnx")}
    for key in ("encoder_dims", "num_encoder_layers", "num_heads", "query_head_dims", "value_head_dims", "cnn_module_kernels", "downsampling_factors", "left_context_len"):
        config[key] = []
    stacks = sorted({int(p.split(".")[2]) for p in prefixes})
    for stack in stacks:
        prefix = next(p for p in prefixes if int(p.split(".")[2]) == stack)
        base = "encoder." + prefix
        channels = tensors[base + ".norm.bias"].size
        heads = tensors[base + ".self_attn_weights.linear_pos.weight"].shape[0] // 4
        query = (tensors[base + ".self_attn_weights.in_proj.bias"].size // heads - 4) // 2
        values = tensors[base + ".self_attn1.in_proj.bias"].size // heads
        downsample = tensors.get(f"encoder.encoder.encoders.{stack}.downsample.weights", np.ones(1))
        for key, value in [("encoder_dims", channels), ("num_encoder_layers", sum(int(p.split('.')[2]) == stack for p in prefixes)),
                           ("num_heads", heads), ("query_head_dims", query), ("value_head_dims", values),
                           ("cnn_module_kernels", tensors[base + ".conv_module1.depthwise_conv.weight"].shape[-1]),
                           ("downsampling_factors", downsample.size), ("left_context_len", 0)]:
            config[key].append(int(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audiocpp-kroko-offline-") as temporary:
        staging = Path(temporary) / "assets"
        staging.mkdir()
        contract = json.loads((REPO / "model_specs/kroko_asr.json").read_text())
        contract["display_name"] = "Kroko XL English Offline ASR"
        contract["description"] = "Native non-streaming Zipformer2/RNN-T English transcription."
        contract["modes"] = ["offline"]
        contract["languages"] = ["en"]
        contract["capabilities"]["asr"].remove("partial_results")
        spec = Path(temporary) / "kroko_asr.json"
        spec.write_text(json.dumps(contract, indent=2) + "\n")
        save_file(tensors, str(staging / "model.safetensors"), metadata={"format": "pt", "audiocpp_family": "kroko_asr"})
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copyfile(root / "tokens.txt", staging / "tokens.txt")
        if (root / "LICENSE").is_file():
            shutil.copyfile(root / "LICENSE", staging / "LICENSE")
        command = [str(args.audiocpp_gguf.resolve()), "--input", str(staging / "model.safetensors"),
                   "--output", str(output), "--type", args.type, "--family", "kroko_asr",
                   "--model-spec", str(spec), "--root", str(staging), "--log"]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
    print(f"wrote {output} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
