#!/usr/bin/env python3
"""Package the official MIT AST AudioSet Safetensors checkpoint as GGUF."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SOURCE_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"


def validate_source(source: Path) -> None:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    processor = json.loads((source / "preprocessor_config.json").read_text(encoding="utf-8"))
    expected = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 16,
        "frequency_stride": 10,
        "time_stride": 10,
        "max_length": 1024,
        "num_mel_bins": 128,
    }
    mismatches = [f"{key}={config.get(key)!r}, expected {value!r}" for key, value in expected.items()
                  if config.get(key) != value]
    if len(config.get("id2label", {})) != 527:
        mismatches.append("id2label must contain 527 classes")
    if processor.get("sampling_rate") != 16000 or processor.get("feature_extractor_type") != "ASTFeatureExtractor":
        mismatches.append("unexpected AST feature extractor")
    if mismatches:
        raise ValueError("not the expected MIT AST AudioSet checkpoint: " + "; ".join(mismatches))
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError(source / "model.safetensors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=["orig", "f16", "q8_0", "q4_0", "q4_k"], default="orig")
    args = parser.parse_args()
    source = args.source.resolve()
    validate_source(source)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.converter.resolve()),
        "--input", str((source / "model.safetensors").resolve()),
        "--root", str(source),
        "--family", "ast_audioset",
        "--model-spec", str(Path("model_specs/ast_audioset.json").resolve()),
        "--type", args.type,
        "--output", str(args.output.resolve()),
        "--overwrite",
    ]
    print("source_repo=" + SOURCE_REPO)
    print("+", " ".join(command))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
