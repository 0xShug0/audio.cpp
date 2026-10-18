#!/usr/bin/env python3
"""Extract the released Sidon v0.1 TorchScript weights and package a self-contained GGUF."""

import argparse
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors.torch import save_file


def extract(model_dir):
    encoder = torch.jit.load(str(model_dir / "feature_extractor_cuda.pt"), map_location="cpu")
    decoder = torch.jit.load(str(model_dir / "decoder_cuda.pt"), map_location="cpu")
    ec = encoder.code_with_constants[1].const_mapping
    dc = decoder.code_with_constants[1].const_mapping
    if len(ec) != 264 or len(dc) != 146:
        raise ValueError("Unsupported Sidon TorchScript layout: expected v0.1 encoder/decoder")
    tensors = {}
    used = set()

    def get(index, shape):
        value = ec[f"c{index}"]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Encoder c{index}: expected {shape}, got {tuple(value.shape)}")
        used.add(index)
        return value

    def norm(name, index, channels=1024):
        tensors[name + ".weight"] = get(index, (channels,))
        tensors[name + ".bias"] = get(index + 1, (channels,))

    def linear(name, index, inputs, outputs, lora=None):
        weight = get(index, (inputs, outputs))
        if lora is not None:
            weight = weight + get(12, ()).item() * (
                get(lora, (inputs, 64)) @ get(lora + 1, (64, outputs)))
        tensors[name + ".weight"] = weight.T.contiguous()
        tensors[name + ".bias"] = get(index + 1, (outputs,))

    norm("feature_projection.layer_norm", 0, 160)
    linear("feature_projection.projection", 2, 160, 1024)
    for index, expected in ((12, 0.25), (13, 0.5), (18, 8.0), (19, 64)):
        if get(index, ()).item() != expected:
            raise ValueError(f"Unexpected encoder constant c{index}")
    for layer in range(8):
        # The first layer introduces four shared scalar constants in the frozen trace.
        ids = ([4, 6, 8, 10, 14, 16, 20, 21, 23, 25, 26, 27, 29, 30, 32, 34, 36, 38]
               if layer == 0 else [40 + 32 * (layer - 1) + offset for offset in
                                  (0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 18, 19, 21, 22, 24, 26, 28, 30)])
        prefix = f"encoder.layers.{layer}."
        norm(prefix + "ffn1_layer_norm", ids[0])
        linear(prefix + "ffn1.intermediate_dense", ids[1], 1024, 4096)
        linear(prefix + "ffn1.output_dense", ids[2], 4096, 1024, ids[3])
        norm(prefix + "self_attn_layer_norm", ids[4])
        qkv = get(ids[5], (1024, 3072))
        bias = get(ids[5] + 1, (3072,))
        for part, name in enumerate(("q", "k", "v")):
            tensors[prefix + f"self_attn.linear_{name}.weight"] = qkv[:, part * 1024:(part + 1) * 1024].T.contiguous()
            tensors[prefix + f"self_attn.linear_{name}.bias"] = bias[part * 1024:(part + 1) * 1024].clone()
        tensors[prefix + "self_attn.distance_embedding.weight"] = get(ids[6], (73, 64))
        linear(prefix + "self_attn.linear_out", ids[7], 1024, 1024)
        norm(prefix + "conv_module.layer_norm", ids[8])
        tensors[prefix + "conv_module.pointwise_conv1.weight"] = get(ids[9], (2048, 1024, 1))
        tensors[prefix + "conv_module.depthwise_conv.weight"] = get(ids[10], (1024, 1, 31))
        norm(prefix + "conv_module.depthwise_layer_norm", ids[11])
        tensors[prefix + "conv_module.pointwise_conv2.weight"] = get(ids[12], (1024, 1024, 1))
        norm(prefix + "ffn2_layer_norm", ids[13])
        linear(prefix + "ffn2.intermediate_dense", ids[14], 1024, 4096)
        linear(prefix + "ffn2.output_dense", ids[15], 4096, 1024, ids[16])
        norm(prefix + "final_layer_norm", ids[17])
    if used != set(range(264)):
        raise ValueError(f"Unmapped encoder constants: {set(range(264)) - used}")
    used_decoder = set()

    def decoder_tensor(name, index, shape):
        value = dc[f"c{index}"]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Decoder c{index}: expected {shape}, got {tuple(value.shape)}")
        used_decoder.add(index)
        tensors["decoder." + name] = value

    def decoder_conv(name, index, weight_shape, bias_channels):
        decoder_tensor(name + ".weight", index, weight_shape)
        decoder_tensor(name + ".bias", index + 1, (bias_channels,))

    def decoder_snake(name, index, channels):
        decoder_tensor(name + ".alpha", index, (1, channels, 1))
        decoder_tensor(name + ".inv_alpha", index + 1, (1, channels, 1))

    decoder_conv("input", 0, (1536, 1024, 7), 1536)
    channels = 1536
    for stage, stride in enumerate((8, 5, 4, 3, 2)):
        base = 2 + 28 * stage
        prefix = f"stages.{stage}."
        decoder_snake(prefix + "activation", base, channels)
        decoder_conv(prefix + "upsample", base + 2, (channels, channels // 2, stride * 2), channels // 2)
        channels //= 2
        for unit in range(3):
            offset = base + 4 + unit * 8
            block = prefix + f"residuals.{unit}."
            decoder_snake(block + "activation1", offset, channels)
            decoder_conv(block + "conv1", offset + 2, (channels, channels, 7), channels)
            decoder_snake(block + "activation2", offset + 4, channels)
            decoder_conv(block + "conv2", offset + 6, (channels, channels, 1), channels)
    decoder_snake("output_activation", 142, channels)
    decoder_conv("output", 144, (1, channels, 7), 1)
    if used_decoder != set(range(146)):
        raise ValueError(f"Unmapped decoder constants: {set(range(146)) - used_decoder}")
    return {name: value.clone().contiguous() for name, value in tensors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", default="f32", help="GGUF tensor storage type")
    args = parser.parse_args()
    torch.set_num_threads(8)
    tensors = extract(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    staging = repo / "build/logs/sidon"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=staging) as directory:
        source = Path(directory) / "model.safetensors"
        save_file(tensors, str(source), metadata={"source": "sarulab-speech/sidon-v0.1"})
        subprocess.run([
            str(repo / "build/debug/bin/audiocpp_gguf"),
            "--input", f"weights={source}", "--output", str(args.output),
            "--type", args.type, "--family", "sidon", "--model-spec",
            str(repo / "model_specs/sidon.json"), "--no-sidecars", "--overwrite",
        ], check=True)
    print(f"Converted {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
