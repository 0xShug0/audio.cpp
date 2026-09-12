#!/usr/bin/env python3
"""Convert Hugging Face ``suno/bark-small`` into an audio.cpp GGUF package.

The generated package contains Bark's semantic, coarse, and fine transformers,
the eight EnCodec codebooks used by Bark, the EnCodec decoder, the tokenizer,
and every downloaded speaker preset.  Weight-normalized EnCodec convolutions
are materialized before export so the native runtime does not need PyTorch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "bark_tts.json"


def _load_checkpoint(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required: uv run --with torch --with safetensors " + __file__) from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint)


def _materialize_weight_norm(state: dict, prefix: str):
    import torch

    g_key, v_key = prefix + ".weight_g", prefix + ".weight_v"
    if g_key not in state or v_key not in state:
        return None
    g, v = state[g_key].float(), state[v_key].float()
    # PyTorch weight_norm(..., dim=0): one magnitude per output channel.
    dims = tuple(range(1, v.ndim))
    return (v * (g / torch.linalg.vector_norm(v, dim=dims, keepdim=True))).contiguous()


def _export_tensors(source: Path, output: Path) -> None:
    from safetensors.torch import save_file

    state = _load_checkpoint(source / "pytorch_model.bin")
    exported = {}
    for name, tensor in state.items():
        if name.endswith(".attn.bias") or name.startswith("codec_model.encoder."):
            continue
        if name.startswith("codec_model.quantizer.layers."):
            parts = name.split(".")
            if int(parts[3]) >= 8 or not name.endswith(".codebook.embed"):
                continue
        if name.endswith(".weight_g") or name.endswith(".weight_v"):
            prefix = name.rsplit(".", 1)[0]
            weight_name = prefix + ".weight"
            if weight_name not in exported:
                exported[weight_name] = _materialize_weight_norm(state, prefix)
            continue
        # Fine-model LM heads are tied to embeddings in the PyTorch object;
        # safetensors requires each exported name to own its storage.
        exported[name] = tensor.detach().clone().contiguous()
    save_file(exported, str(output))


def _collect_presets(source: Path, output: Path) -> int:
    import numpy as np

    preset_root = source / "speaker_embeddings"
    records = {}
    for semantic in sorted(preset_root.rglob("*_semantic_prompt.npy")):
        stem = semantic.name.removesuffix("_semantic_prompt.npy")
        relative = semantic.parent.relative_to(preset_root)
        key = str(relative / stem)
        coarse = semantic.with_name(stem + "_coarse_prompt.npy")
        fine = semantic.with_name(stem + "_fine_prompt.npy")
        if not coarse.is_file() or not fine.is_file():
            raise RuntimeError(f"incomplete Bark preset: {key}")
        records[key] = {
            "semantic": np.load(semantic).astype("int32").tolist(),
            "coarse": np.load(coarse).astype("int32").tolist(),
            "fine": np.load(fine).astype("int32").tolist(),
        }
    output.write_text(json.dumps({"version": 1, "presets": records}, separators=(",", ":")))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="downloaded suno/bark-small snapshot")
    parser.add_argument("--converter", type=Path, required=True, help="built audiocpp_gguf executable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", default="q8_0", choices=["orig", "f16", "bf16", "q8_0"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    required = ["pytorch_model.bin", "config.json", "generation_config.json", "tokenizer.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"Bark source is missing: {', '.join(missing)}")
    if not args.converter.is_file():
        raise SystemExit(f"converter not found: {args.converter}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audiocpp-bark-") as temp_string:
        temp = Path(temp_string)
        tensors = temp / "bark.safetensors"
        presets = temp / "speaker_presets.json"
        _export_tensors(source, tensors)
        preset_count = _collect_presets(source, presets)
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, temp / name)
        command = [
            str(args.converter), "--input", f"bark={tensors}", "--root", str(temp),
            "--family", "bark_tts", "--model-spec", str(SPEC), "--type", args.type,
            "--output", str(output),
        ]
        if args.type == "q8_0":
            # Autoregressive errors compound across Bark's semantic and coarse
            # stages. Keep both complete causal transformers, plus the waveform
            # decoder and fine lookup/output tensors, in F16. Only the fine
            # transformer's dense matrices are safe to quantize to Q8_0.
            for pattern in (
                "bark/codec_model.*", "bark/semantic.*", "bark/coarse_acoustics.*",
                "bark/fine_acoustics.input_embeds_layers.*",
                "bark/fine_acoustics.position_embeds_layer.*", "bark/fine_acoustics.layernorm_final.*",
                "bark/fine_acoustics.lm_heads.*",
            ):
                command.extend(["--keep-type", pattern + "=f16"])
        if args.overwrite:
            command.append("--overwrite")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        print(f"wrote {output} with {preset_count} speaker presets")


if __name__ == "__main__":
    main()
