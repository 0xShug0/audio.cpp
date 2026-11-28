#!/usr/bin/env python3
"""Convert the official Smart Turn v3.2 FP32 ONNX export to GGUF."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "build/debug/bin/audiocpp_gguf"
LAYERS = 4
CHANNELS = 384


def extract_tensors(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    arrays = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    if arrays["inner.encoder.embed_positions.weight"].shape != (400, CHANNELS):
        raise ValueError("expected the eight-second Whisper-Tiny encoder")

    matmuls = [node for node in model.graph.node
               if node.op_type == "MatMul" and len(node.input) == 2
               and node.input[1] in arrays and arrays[node.input[1]].ndim == 2]
    if len(matmuls) != LAYERS * 6 + 2:
        raise ValueError(f"expected {LAYERS * 6 + 2} learned MatMul nodes, got {len(matmuls)}")

    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    names = ("attn.query", "attn.key", "attn.value", "attn.out", "mlp.0", "mlp.2")
    bias_names = ("self_attn.q_proj.bias", None, "self_attn.v_proj.bias",
                  "self_attn.out_proj.bias", "fc1.bias", "fc2.bias")
    tensors: dict[str, np.ndarray] = {}
    for layer in range(LAYERS):
        prefix = f"encoder.blocks.{layer}"
        original = f"inner.encoder.layers.{layer}"
        for offset, (target, bias) in enumerate(zip(names, bias_names)):
            node = matmuls[layer * 6 + offset]
            array = arrays[node.input[1]]
            expected = ((CHANNELS, CHANNELS) if offset < 4 else
                        (CHANNELS, CHANNELS * 4) if offset == 4 else
                        (CHANNELS * 4, CHANNELS))
            if array.shape != expected:
                raise ValueError(f"unexpected {prefix}.{target} shape: {array.shape}")
            if bias is not None:
                uses = consumers.get(node.output[0], [])
                if not any(use.op_type == "Add" and original + "." + bias in use.input for use in uses):
                    raise ValueError(f"MatMul/bias pairing changed for {prefix}.{target}")
            tensors[f"{prefix}.{target}.weight"] = np.ascontiguousarray(array.T)
        for source, target in (("self_attn.q_proj.bias", "attn.query.bias"),
                               ("self_attn.v_proj.bias", "attn.value.bias"),
                               ("self_attn.out_proj.bias", "attn.out.bias"),
                               ("fc1.bias", "mlp.0.bias"), ("fc2.bias", "mlp.2.bias"),
                               ("self_attn_layer_norm.weight", "attn_ln.weight"),
                               ("self_attn_layer_norm.bias", "attn_ln.bias"),
                               ("final_layer_norm.weight", "mlp_ln.weight"),
                               ("final_layer_norm.bias", "mlp_ln.bias")):
            tensors[f"{prefix}.{target}"] = np.ascontiguousarray(arrays[f"{original}.{source}"])

    for source, target in (("conv1.weight", "conv1.weight"), ("conv1.bias", "conv1.bias"),
                           ("conv2.weight", "conv2.weight"), ("conv2.bias", "conv2.bias"),
                           ("embed_positions.weight", "positional_embedding"),
                           ("layer_norm.weight", "ln_post.weight"),
                           ("layer_norm.bias", "ln_post.bias")):
        tensors[f"encoder.{target}"] = np.ascontiguousarray(arrays[f"inner.encoder.{source}"])

    pool = ("head.pool.0.weight", "head.pool.2.weight")
    for index, name in enumerate(pool):
        node = matmuls[LAYERS * 6 + index]
        expected = (CHANNELS, 256) if index == 0 else (256, 1)
        if arrays[node.input[1]].shape != expected:
            raise ValueError(f"unexpected {name} shape")
        tensors[name] = np.ascontiguousarray(arrays[node.input[1]].T)
    for index in (0, 2):
        tensors[f"head.pool.{index}.bias"] = np.ascontiguousarray(arrays[f"inner.pool_attention.{index}.bias"])
    for index in (0, 1, 4, 6):
        for part in ("weight", "bias"):
            name = f"inner.classifier.{index}.{part}"
            tensors[f"head.classifier.{index}.{part}"] = np.ascontiguousarray(arrays[name])
    if any(tensor.dtype != np.float32 for tensor in tensors.values()):
        raise ValueError("the selected ONNX checkpoint is not FP32")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.onnx)
    tensors = extract_tensors(model)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smart-turn-convert-") as temporary:
        root = Path(temporary)
        save_file(tensors, root / "weights.safetensors")
        (root / "config.json").write_text(json.dumps({
            "model_type": "smart_turn_v3",
            "version": "3.2",
            "n_mels": 80,
            "n_audio_ctx": 400,
            "n_audio_state": CHANNELS,
            "n_audio_head": 6,
            "n_audio_layer": LAYERS,
        }, indent=2) + "\n")
        fd, temporary_output = tempfile.mkstemp(prefix=output.stem + "-", suffix=".gguf", dir=output.parent)
        os.close(fd)
        try:
            subprocess.run([
                str(CONVERTER), "--input", f"weights={root / 'weights.safetensors'}",
                "--root", str(root), "--output", temporary_output, "--type", "f32",
                "--family", "smart_turn", "--model-spec", str(REPO / "model_specs/smart_turn.json"),
                "--overwrite",
            ], check=True)
            os.replace(temporary_output, output)
        finally:
            Path(temporary_output).unlink(missing_ok=True)
    print(f"wrote {len(tensors)} FP32 tensors to {output}")


if __name__ == "__main__":
    main()
