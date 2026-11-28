#!/usr/bin/env python3
"""Package Voxtral TTS weights, tokenizer, and preset voices in one GGUF."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.integrations.tiktoken import TikTokenConverter


def convert_tokenizer(source, staging):
    tekken = json.loads((source / "tekken.json").read_text())
    config = tekken["config"]
    offset = config["default_num_special_tokens"]
    count = config["default_vocab_size"] - offset
    ranks = staging / "text.tiktoken"
    ranks.write_text("".join(
        f"{token['token_bytes']} {token['rank']}\n"
        for token in tekken["vocab"][:count]
    ))
    tokenizer = TikTokenConverter(vocab_file=str(ranks), pattern=config["pattern"]).converted()
    exported = json.loads(tokenizer.to_str())
    exported["model"]["vocab"] = {
        token: rank + offset for token, rank in exported["model"]["vocab"].items()
    }
    specials = {token["rank"]: token["token_str"] for token in tekken["special_tokens"]}
    exported["added_tokens"] = [
        {"id": rank, "content": specials.get(rank, f"<SPECIAL_{rank}>"),
         "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
        for rank in range(offset)
    ]
    (staging / "tokenizer.json").write_text(json.dumps(exported, ensure_ascii=False))
    (staging / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
    }))
    ranks.unlink()


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", default="native",
                        choices=("native", "bf16", "f16", "f32", "q8_0", "q4_k"))
    parser.add_argument("--converter", type=Path, default=repo / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.model_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = repo / "build/logs/voxtral_tts"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert-", dir=logs) as temporary:
        staging = Path(temporary)
        (staging / "params.json").write_bytes((source / "params.json").read_bytes())
        convert_tokenizer(source, staging)
        shard = "consolidated.safetensors"
        (staging / shard).symlink_to(source / shard)
        derived = {}
        with safe_open(source / shard, framework="pt", device="cpu") as tensors:
            weight_map = {name: shard for name in tensors.keys()}
            # Evaluate the same parametrization as upstream, preserving BF16 rounding.
            for name in list(weight_map):
                if not name.endswith(".parametrizations.weight.original1"):
                    continue
                prefix = name.removesuffix(".parametrizations.weight.original1")
                magnitude = prefix + ".parametrizations.weight.original0"
                derived[prefix + ".weight"] = torch._weight_norm(
                    tensors.get_tensor(name), tensors.get_tensor(magnitude), 0
                ).contiguous()
                del weight_map[name]
                del weight_map[magnitude]
            quantizer = "audio_tokenizer.quantizer.semantic_codebook."
            usage = tensors.get_tensor(quantizer + "cluster_usage")
            sums = tensors.get_tensor(quantizer + "embedding_sum")
            derived[quantizer + "embedding"] = (sums / usage.clamp(min=1e-5)[:, None]).contiguous()
            del weight_map[quantizer + "cluster_usage"]
            del weight_map[quantizer + "embedding_sum"]
        voices = json.loads((source / "params.json").read_text())["multimodal"]["audio_tokenizer_args"]["voice"]
        for voice in voices:
            prompt = torch.load(source / "voice_embedding" / f"{voice}.pt",
                                weights_only=True, map_location="cpu")
            if prompt.ndim != 2:
                raise ValueError(f"{voice}: expected a [frames, hidden] voice embedding")
            derived[f"voices.{voice}.prompt_embedding"] = prompt.contiguous()
        save_file(derived, staging / "derived.safetensors")
        weight_map.update({name: "derived.safetensors" for name in derived})
        (staging / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        suffix = "bf16" if args.type == "native" else args.type
        output = args.output_dir / f"voxtral-4b-tts-2603-{suffix}.gguf"
        command = [
            str(args.converter), "--input", str(staging / "model.safetensors.index.json"),
            "--root", str(staging), "--output", str(output), "--type", args.type,
            "--family", "voxtral_tts", "--model-spec", str(repo / "model_specs/voxtral_tts.json"),
            "--keep-type", "voices.*=bf16",
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        subprocess.run([str(args.converter), "--inspect", str(output)], check=True)


if __name__ == "__main__":
    main()
