#!/usr/bin/env python3
"""Convert the original checkpoint with the native converter and verify its GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import load_file, save_file


REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"


def tensor_name(name: str) -> str:
    prefix = "model.whisper_encoder."
    if not name.startswith(prefix):
        return name
    suffix = name.removeprefix(prefix)
    if suffix == "embed_positions.weight":
        return "encoder.positional_embedding"
    parts = suffix.split(".")
    if parts[0] == "layers":
        parts[0] = "blocks"
        replacements = {
            "self_attn": "attn", "q_proj": "query", "k_proj": "key",
            "v_proj": "value", "out_proj": "out", "self_attn_layer_norm": "attn_ln",
            "final_layer_norm": "mlp_ln", "fc1": "mlp.0", "fc2": "mlp.2",
        }
        parts = [replacements.get(part, part) for part in parts]
    elif parts[0] == "layer_norm":
        parts[0] = "ln_post"
    return "encoder." + ".".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--type", choices=("orig", "q8_0", "q4_k", "q4_0"), default="orig")
    parser.add_argument("--keep-type", action="append", default=[], help="Native converter tensor-type override")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(args.log, mode="w"), logging.StreamHandler()])
    checkpoint = args.model / "model-00000-of-00001.safetensors"
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the pinned upstream release")
    original = load_file(checkpoint)
    renamed = {tensor_name(name): tensor for name, tensor in original.items()}
    if len(renamed) != len(original):
        raise ValueError("Whisper tensor rename collision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moss-transcribe-diarize-") as temporary:
        root = Path(temporary)
        save_file(renamed, root / "model.safetensors")
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                     "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                     "chat_template.jinja"):
            shutil.copy2(args.model / name, root / name)
        command = [str(REPO / "build/debug/bin/audiocpp_gguf"),
                   "--input", f"weights={root / 'model.safetensors'}",
                   "--output", str(args.output.resolve()), "--type", args.type,
                   "--root", str(root), "--family", "moss_transcribe_diarize",
                   "--model-spec", str(REPO / "model_specs/moss_transcribe_diarize.json")]
        if args.overwrite:
            command.append("--overwrite")
        for rule in args.keep_type:
            command.extend(("--keep-type", rule))
        logging.info("command=%s", command)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("%s", completed.stdout)
        completed.check_returncode()

    reader = gguf.GGUFReader(str(args.output))
    embedded_spec = json.loads(reader.get_field("audiocpp.model_spec.json").contents())
    source_spec = json.loads((REPO / "model_specs/moss_transcribe_diarize.json").read_text())
    for key in ("schema_version", "family", "tasks", "modes", "options", "sources"):
        if embedded_spec.get(key) != source_spec.get(key):
            raise ValueError(f"Embedded runtime contract differs: {key}")
    expected = {"weights/" + name: tensor for name, tensor in renamed.items()}
    names = reader.get_field("audiocpp.tensor_names").contents()
    if len(names) != len(reader.tensors) or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("GGUF tensor set mismatch")
    ranks = reader.get_field("audiocpp.tensor_ranks").contents()
    shapes = reader.get_field("audiocpp.tensor_shapes").contents()
    cursor = 0
    types = Counter()
    squared_error = 0.0
    squared_reference = 0.0
    for tensor, name, rank in zip(reader.tensors, names, ranks, strict=True):
        source = expected[name]
        shape = tuple(shapes[cursor:cursor + rank])
        cursor += rank
        if source.dtype != torch.bfloat16:
            raise ValueError(f"Expected original BF16 source: {tensor.name}")
        types[tensor.tensor_type.name] += 1
        physical = tuple(tensor.shape) + (1,) * (4 - len(tensor.shape))
        if shape != tuple(source.shape) or physical != tuple(source.shape)[::-1] + (1,) * (4 - source.ndim):
            raise ValueError(f"Shape mismatch: {tensor.name}")
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            if not np.array_equal(tensor.data.reshape(-1).view(np.uint8),
                                  source.view(torch.uint8).numpy().reshape(-1)):
                raise ValueError(f"Tensor bytes differ: {tensor.name}")
        elif args.type == "orig":
            raise ValueError(f"Expected original BF16: {tensor.name}")
        else:
            actual = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(-1)
            reference = source.float().numpy().reshape(-1)
            if actual.shape != reference.shape or not np.isfinite(actual).all():
                raise ValueError(f"Invalid converted tensor: {tensor.name}")
            error = actual.astype(np.float64) - reference
            squared_error += np.sum(error * error)
            squared_reference += np.sum(reference.astype(np.float64) ** 2)
    if cursor != len(shapes):
        raise ValueError("Trailing shape metadata")
    if args.type == "orig":
        logging.info("PASS %d tensors BF16 byte-exact; SHA256=%s; GGUF bytes=%d",
                     len(expected), digest, args.output.stat().st_size)
    else:
        if not types[args.type.upper()]:
            raise ValueError("Requested quantization did not produce any quantized tensors")
        logging.info("PASS structure/finite checks; types=%s; converted-tensor relative RMSE=%g; "
                     "SHA256=%s; GGUF bytes=%d; inference quality requires separate validation",
                     dict(types), np.sqrt(squared_error / squared_reference), digest, args.output.stat().st_size)


if __name__ == "__main__":
    main()
