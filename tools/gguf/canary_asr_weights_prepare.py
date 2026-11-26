#!/usr/bin/env python3
"""Repackage the original Canary NeMo checkpoint without changing tensor values."""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch
import yaml
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canary_nemo_") as tmp:
        root = Path(tmp)
        with tarfile.open(args.source) as archive:
            archive.extractall(root, filter="data")
        config = yaml.safe_load((root / "model_config.yaml").read_text())
        if config["prompt_format"] != "canary2":
            raise ValueError("Expected Canary Flash with the canary2 prompt format")
        state = torch.load(root / "model_weights.ckpt", map_location="cpu", weights_only=True)
        # Cloning separates tied weights for safetensors while retaining their exact dtype and values.
        tensors = {name: value.contiguous().clone() for name, value in state.items()}
        save_file(tensors, args.output / "model.safetensors")
        offset = 0
        tokenizers = []
        for language, tokenizer in config["tokenizer"]["langs"].items():
            source = root / tokenizer["model_path"].removeprefix("nemo:")
            filename = f"tokenizer_{language}.model"
            shutil.copyfile(source, args.output / filename)
            processor = spm.SentencePieceProcessor(model_file=str(source))
            tokenizers.append({"language": language, "file": filename,
                               "offset": offset, "size": processor.vocab_size()})
            offset += processor.vocab_size()
        if offset != config["head"]["num_classes"]:
            raise ValueError(f"Tokenizer vocabulary {offset} differs from classifier size")
        config["audio_cpp_tokenizers"] = tokenizers
        config["audio_cpp_format"] = "canary_asr"
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"Converted {len(tensors)} tensors without dtype conversion; vocabulary={offset}")


if __name__ == "__main__":
    main()
