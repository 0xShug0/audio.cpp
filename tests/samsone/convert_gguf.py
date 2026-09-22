#!/usr/bin/env python3
"""Convert a released SAMSONE checkpoint to a standalone BF16 GGUF."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "build/debug/bin/audiocpp_gguf"


def build_token_map(tokenizer) -> dict[int, int]:
    special_ids = set(tokenizer.all_special_ids)
    output: dict[int, int] = {}
    for token, original_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", token)
        keep = original_id in special_ids or (
            not any(character.isupper() for character in cleaned)
            and not re.search(r"[\sĠĊĉ]{4,}", token)
            and not re.search(r"[^\x00-\x7FĠĊĉ\s]", token)
            and not re.search(r"[-#]{3,}", token)
        )
        if keep:
            output[len(output)] = original_id
    return output


def converted_name(name: str) -> str | None:
    name = name.removeprefix("audio_lm.")
    if name == "text_model.embedding_layer.weight":
        return None
    if name.startswith("audio_encoder.encoder.encoder."):
        suffix = name.removeprefix("audio_encoder.encoder.encoder.")
        if suffix == "embed_positions.weight":
            return "encoder.positional_embedding"
        parts = suffix.split(".")
        if parts[0] == "layers":
            parts[0] = "blocks"
            replacements = {
                "self_attn": "attn",
                "q_proj": "query",
                "k_proj": "key",
                "v_proj": "value",
                "out_proj": "out",
                "self_attn_layer_norm": "attn_ln",
                "final_layer_norm": "mlp_ln",
                "fc1": "mlp.0",
                "fc2": "mlp.2",
            }
            parts = [replacements.get(part, part) for part in parts]
        elif parts[0] == "layer_norm":
            parts[0] = "ln_post"
        return "encoder." + ".".join(parts)
    if name.startswith("text_model.model.model."):
        return "model." + name.removeprefix("text_model.model.model.")
    if name.startswith("text_model.model.lm_head."):
        return None
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    text_model_dir = args.text_model_dir.resolve()
    output = args.output.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    if not state or not all(name.startswith("audio_lm.") for name in state):
        raise RuntimeError("expected a released SAMSONE AudioLM checkpoint")
    tensors = {}
    for name, tensor in state.items():
        target = converted_name(name)
        if target is not None:
            tensors[target] = tensor.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
    token_map = build_token_map(tokenizer)
    hidden = tensors["sep_token"].shape[0]
    layers = sum(name.endswith(".self_attn.q_proj.weight") and name.startswith("model.layers.") for name in tensors)
    q_shape = tensors["model.layers.0.self_attn.q_proj.weight"].shape
    k_shape = tensors["model.layers.0.self_attn.k_proj.weight"].shape
    intermediate = tensors["model.layers.0.mlp.gate_proj.weight"].shape[0]
    text_config = json.loads((text_model_dir / "config.json").read_text())
    original_eos = int(text_config["eos_token_id"])
    original_to_pruned = {original: pruned for pruned, original in token_map.items()}
    config = {
        "model_type": "samsone",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": int(text_config["num_attention_heads"]),
        "num_key_value_heads": int(k_shape[0] // (q_shape[0] // int(text_config["num_attention_heads"]))),
        "num_hidden_layers": layers,
        "vocab_size": len(token_map),
        "max_position_embeddings": int(text_config["max_position_embeddings"]),
        "rms_norm_eps": float(text_config["rms_norm_eps"]),
        "rope_theta": float(text_config["rope_theta"]),
        "eos_token_id": original_to_pruned[original_eos],
    }
    if tensors["model.embed_tokens.weight"].shape != (len(token_map), hidden):
        raise RuntimeError("checkpoint vocabulary does not match the reconstructed token map")

    subprocess.run(
        ["cmake", "--build", str(REPO / "build/debug"), "--target", "audiocpp_gguf", "-j", "8"],
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="samsone-convert-") as temporary:
        root = Path(temporary)
        source = root / "model.safetensors"
        save_file(tensors, source)
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (root / "token_map.json").write_text(json.dumps(token_map, sort_keys=True) + "\n")
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(text_model_dir / filename, root / filename)
        fd, temporary_output = tempfile.mkstemp(prefix=output.stem + "-", suffix=".gguf", dir=output.parent)
        os.close(fd)
        temporary_output = Path(temporary_output)
        try:
            subprocess.run(
                [
                    str(CONVERTER),
                    "--input", f"weights={source}",
                    "--root", str(root),
                    "--output", str(temporary_output),
                    "--type", "bf16",
                    "--family", "samsone",
                    "--model-spec", str(REPO / "model_specs/samsone.json"),
                    "--overwrite",
                ],
                check=True,
            )
            inspection = subprocess.run(
                [str(CONVERTER), "--inspect", str(temporary_output)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if f"tensors={len(tensors)}" not in inspection.splitlines():
                raise RuntimeError(f"SAMSONE GGUF tensor count mismatch:\n{inspection}")
            os.replace(temporary_output, output)
            print(inspection, end="")
        finally:
            temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
