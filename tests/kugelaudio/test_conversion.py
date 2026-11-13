#!/usr/bin/env python3
"""Exercise KugelAudio conversion contracts with tiny synthetic weights.

Requires torch, safetensors, numpy and gguf. This is a packaging test, not a
synthetic-model inference test; runtime validation uses the original checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from gguf import GGUFReader
import numpy as np
from safetensors.torch import save_file
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    encoder_name = "model.acoustic_tokenizer.encoder.stages.0.0.ffn.linear1.weight"
    generator_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    weights = {name: torch.arange(1024, dtype=torch.float32).reshape(32, 32).to(torch.bfloat16)
               for name in (encoder_name, generator_name)}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def run(command):
        result = subprocess.run(command, capture_output=True, text=True, creationflags=creationflags)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="kugelaudio-conversion-") as temporary:
        root = Path(temporary)
        source, tokenizer = root / "source", root / "tokenizer"
        source.mkdir()
        tokenizer.mkdir()
        save_file(weights, source / "weights.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "weights.safetensors" for name in weights}}), encoding="utf-8")
        config = {
            "model_type": "kugelaudio",
            "decoder_config": {"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 2,
                "num_key_value_heads": 1, "num_hidden_layers": 1, "vocab_size": 256,
                "max_position_embeddings": 512, "rms_norm_eps": 1e-5, "rope_theta": 10000},
            "diffusion_head_config": {"hidden_size": 64, "latent_size": 64, "head_layers": 1,
                "head_ffn_ratio": 2, "rms_norm_eps": 1e-5, "ddpm_num_inference_steps": 20},
            "acoustic_tokenizer_config": {"vae_dim": 64, "decoder_n_filters": 32,
                "decoder_ratios": [2], "encoder_depths": [1, 1], "layernorm_eps": 1e-5},
        }
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (tokenizer / name).write_text("{}", encoding="utf-8")
        (source / "voices").mkdir()
        for voice in ("default", "clear", "english_female", "english_male"):
            torch.save({"acoustic_mean": torch.zeros(1, 2, 64), "acoustic_std": torch.tensor(0.5)},
                       source / "voices" / (voice + ".pt"))

        for include_encoder in (False, True):
            output = root / ("clone" if include_encoder else "preset")
            command = [sys.executable, str(repo / "tests/kugelaudio/convert_gguf.py"),
                       "--model-dir", str(source), "--tokenizer-dir", str(tokenizer),
                       "--output-dir", str(output), "--type", "q8_0", "--converter", str(args.converter.resolve())]
            if include_encoder:
                command.append("--include-reference-encoder")
            run(command)
            package = output / ("kugelaudio-0-open-q8_0" + ("-clone" if include_encoder else "") + ".gguf")
            reader = GGUFReader(package)
            try:
                tensors = dict(zip(reader.get_field("audiocpp.tensor_names").contents(), reader.tensors))
                if tensors[generator_name].tensor_type.name != "Q8_0":
                    raise AssertionError("generator was not quantized")
                if (encoder_name in tensors) != include_encoder:
                    raise AssertionError("incorrect packaged encoder admission")
                if include_encoder:
                    encoded = tensors[encoder_name]
                    expected = weights[encoder_name].view(torch.uint16).numpy().view(np.uint8).ravel()
                    if encoded.tensor_type.name != "BF16" or not np.array_equal(expected, encoded.data.view(np.uint8).ravel()):
                        raise AssertionError("encoder source precision was not preserved")
            finally:
                # Release the mapping before TemporaryDirectory cleanup on Windows.
                reader.data._mmap.close()
            inspection = run([str(args.cli.resolve()), "--family", "kugelaudio", "--model", str(package),
                              "--model-spec-override", "embedded", "--inspect"])
            if ("task=clon " in inspection) != include_encoder:
                raise AssertionError("embedded cloning task does not match encoder inclusion")
            if ("supports_speaker_reference=true" in inspection) != include_encoder:
                raise AssertionError("embedded reference capability does not match encoder inclusion")
            print("PASS:", "clone" if include_encoder else "preset", "conversion contract and tensor types", flush=True)


if __name__ == "__main__":
    main()
