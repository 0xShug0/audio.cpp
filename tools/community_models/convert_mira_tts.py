#!/usr/bin/env python3
"""Convert the official MiraTTS checkpoint into audio.cpp tensor assets.

The upstream package combines a Qwen2 safetensors checkpoint, two ONNX
graphs, a DAC decoder safetensors checkpoint, and a small PyTorch 48 kHz
upsampler.  audio.cpp does not execute ONNX or pickle at runtime: this tool
extracts and gives stable names to all learned tensors, fuses upsampler weight
normalization, and writes one ordinary safetensors file per runtime namespace.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from safetensors.torch import load_file, save_file


SPEAKER_ANONYMOUS = {
    "onnx::MatMul_874": "perceiver.proj_context.weight",
    "onnx::Expand_875": "perceiver.latents",
    "onnx::MatMul_878": "perceiver.layers.0.attn.q.weight",
    "onnx::MatMul_879": "perceiver.layers.0.attn.kv.weight",
    "onnx::MatMul_883": "perceiver.layers.0.attn.out.weight",
    "onnx::MatMul_884": "perceiver.layers.0.ff.in.weight",
    "onnx::MatMul_885": "perceiver.layers.0.ff.out.weight",
    "onnx::MatMul_886": "perceiver.layers.1.attn.q.weight",
    "onnx::MatMul_887": "perceiver.layers.1.attn.kv.weight",
    "onnx::MatMul_891": "perceiver.layers.1.attn.out.weight",
    "onnx::MatMul_892": "perceiver.layers.1.ff.in.weight",
    "onnx::MatMul_893": "perceiver.layers.1.ff.out.weight",
    "onnx::MatMul_896": "quantizer.project_in.weight",
}


def processor_anonymous() -> dict[str, str]:
    names = {
        "onnx::MatMul_978": "speaker_encoder.quantizer.project_out.weight",
        "onnx::MatMul_980": "prenet.linear_pre.weight",
        "onnx::MatMul_1013": "prenet.linear.weight",
    }
    index = 981
    for downsample in range(2):
        for block in range(2):
            base = f"prenet.downsample.{downsample}.1.convnext.{block}"
            names[f"onnx::MatMul_{index}"] = f"{base}.pwconv1.weight"
            names[f"onnx::MatMul_{index + 1}"] = f"{base}.pwconv2.weight"
            index += 2
    for block in range(12):
        base = f"prenet.vocos_backbone.convnext.{block}"
        names[f"onnx::MatMul_{index}"] = f"{base}.pwconv1.weight"
        names[f"onnx::MatMul_{index + 1}"] = f"{base}.pwconv2.weight"
        index += 2
    if index != 1013:
        raise AssertionError(index)
    return names


def onnx_tensors(path: Path, namespace: str, anonymous: dict[str, str]) -> dict[str, torch.Tensor]:
    model = onnx.load(path, load_external_data=True)
    output: dict[str, torch.Tensor] = {}
    found_anonymous: set[str] = set()
    for tensor in model.graph.initializer:
        name = anonymous.get(tensor.name, tensor.name)
        if tensor.name.startswith("onnx::MatMul_"):
            if tensor.name not in anonymous:
                raise SystemExit(f"unmapped learned ONNX tensor: {tensor.name}")
        if tensor.name in anonymous:
            found_anonymous.add(tensor.name)
        value = np.asarray(numpy_helper.to_array(tensor))
        if value.ndim == 0:
            value = value.reshape(1)
        # ONNX MatMul parameters are [in, out]; audio.cpp Linear parameters
        # follow PyTorch's [out, in] convention.
        if tensor.name.startswith("onnx::MatMul_"):
            value = value.T
        output[f"{namespace}.{name}"] = torch.from_numpy(np.array(value, copy=True, order="C"))
    missing = set(anonymous) - found_anonymous
    if missing:
        raise SystemExit(f"ONNX graph is missing expected tensors: {sorted(missing)}")
    return output


def processor_context_codebook(path: Path) -> np.ndarray:
    """Recover the exported FSQ 4096x6 implicit codebook constant."""
    model = onnx.load(path, load_external_data=True)
    candidates: list[np.ndarray] = []
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            value = np.asarray(numpy_helper.to_array(attribute.t))
            if value.shape == (4096, 6):
                candidates.append(value)
    if len(candidates) != 1:
        raise SystemExit(
            f"expected one processor FSQ [4096, 6] codebook constant, found {len(candidates)}"
        )
    return np.ascontiguousarray(candidates[0].astype(np.float32))


def safetensor_tensors(path: Path, namespace: str) -> dict[str, torch.Tensor]:
    return {
        f"{namespace}.{name}": value.contiguous()
        for name, value in load_file(str(path), device="cpu").items()
    }


def upsampler_tensors(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    tensors = {
        name: value.detach().cpu().float().numpy()
        for name, value in state.items()
        if hasattr(value, "detach") and not name.startswith("optimizer.")
    }
    fused_weights: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if name.endswith(".weight_g"):
            base = name[:-len("_g")]
            direction = tensors[base + "_v"].astype(np.float32)
            axes = tuple(range(1, direction.ndim))
            norm = np.sqrt(np.sum(direction * direction, axis=axes, keepdims=True))
            fused_weights[base] = value.astype(np.float32) * direction / np.maximum(norm, 1.0e-12)

    # Use the framework FlashSR tensor contract. MiraTTS runs only residual
    # blocks 2 and 0, matching FastAudioSR.Generator.forward().
    output: dict[str, torch.Tensor] = {
        "upsampler.conv_pre.weight": torch.from_numpy(np.ascontiguousarray(fused_weights["dec.conv_pre.weight"])),
        "upsampler.conv_pre.bias": torch.from_numpy(np.ascontiguousarray(tensors["dec.conv_pre.bias"])),
        "upsampler.conv_post.weight": torch.from_numpy(np.ascontiguousarray(tensors["dec.conv_post.weight"])),
        "upsampler.activation_filter": torch.from_numpy(np.ascontiguousarray(
            tensors["dec.activation_post.upsample.filter"])),
    }
    for block in ("0", "2"):
        for group in (1, 2):
            for index in range(3):
                source = f"dec.resblocks.{block}.convs{group}.{index}"
                target = f"upsampler.resblocks.{block}.convs{group}.{index}"
                output[target + ".weight"] = torch.from_numpy(
                    np.ascontiguousarray(fused_weights[source + ".weight"]))
                output[target + ".bias"] = torch.from_numpy(
                    np.ascontiguousarray(tensors[source + ".bias"]))
        for index in range(6):
            source = f"dec.resblocks.{block}.activations.{index}.act"
            target = f"upsampler.resblocks.{block}.activations.{index}"
            output[target + ".alpha"] = torch.from_numpy(
                np.ascontiguousarray(np.exp(tensors[source + ".alpha"])).reshape(1, 32, 1))
            output[target + ".inv_beta"] = torch.from_numpy(
                np.ascontiguousarray(1.0 / (np.exp(tensors[source + ".beta"]) + 1.0e-9)).reshape(1, 32, 1))
    output["upsampler.activation_post.alpha"] = torch.from_numpy(
        np.ascontiguousarray(np.exp(tensors["dec.activation_post.act.alpha"])).reshape(1, 32, 1))
    output["upsampler.activation_post.inv_beta"] = torch.from_numpy(
        np.ascontiguousarray(1.0 / (np.exp(tensors["dec.activation_post.act.beta"]) + 1.0e-9)).reshape(1, 32, 1))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="downloaded YatharthS/MiraTTS directory")
    parser.add_argument("output", type=Path, help="output audio.cpp model directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    namespaces = ("language_model", "speaker_encoder", "processor", "decoder", "upsampler")
    destinations = [output / f"{namespace}.safetensors" for namespace in namespaces]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {existing[0]}")

    required = {
        "lm": root / "model.safetensors",
        "speaker": root / "decoders" / "s_encoder.onnx",
        "processor": root / "decoders" / "processer.onnx",
        "decoder": root / "decoders" / "detokenizer.safetensors",
        "upsampler": root / "decoders" / "upsampler.pth",
        "config": root / "config.json",
        "tokenizer": root / "tokenizer.json",
        "tokenizer_config": root / "tokenizer_config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing MiraTTS files: " + ", ".join(missing))

    tensors: dict[str, torch.Tensor] = {}
    tensors.update(safetensor_tensors(required["lm"], "language_model"))
    tensors.update(onnx_tensors(required["speaker"], "speaker_encoder", SPEAKER_ANONYMOUS))
    tensors.update(onnx_tensors(required["processor"], "processor", processor_anonymous()))
    tensors["processor.speaker_encoder.context_codebook"] = torch.from_numpy(
        processor_context_codebook(required["processor"])
    )
    tensors.update(safetensor_tensors(required["decoder"], "decoder"))
    tensors.update(upsampler_tensors(required["upsampler"]))

    output.mkdir(parents=True, exist_ok=True)
    total = 0
    for namespace in namespaces:
        prefix = namespace + "."
        scoped = {
            name[len(prefix):]: value
            for name, value in tensors.items()
            if name.startswith(prefix)
        }
        if not scoped:
            raise SystemExit(f"no tensors collected for namespace {namespace}")
        save_file(
            scoped,
            str(output / f"{namespace}.safetensors"),
            metadata={
                "format": "pt",
                "source": "YatharthS/MiraTTS",
                "audiocpp_family": "mira_tts",
                "audiocpp_namespace": namespace,
            },
        )
        total += len(scoped)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(root / name, output / name)
    print(f"wrote {len(namespaces)} tensor namespaces ({total} tensors) to {output}")


if __name__ == "__main__":
    main()
