#!/usr/bin/env python3
"""Convert SheetSage2 adapter + MERT2 parent into a self-contained GGUF."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECTIONS = ("query_proj", "key_proj", "value_proj", "out_proj")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def merge_adapter(adapter_dir: Path, mert_dir: Path, output_path: Path) -> None:
    adapter_weights = load_safetensors(require_file(adapter_dir / "model.safetensors"))
    mert_weights = load_safetensors(require_file(mert_dir / "model.safetensors"))
    config = json.loads(require_file(adapter_dir / "config.json").read_text(encoding="utf-8"))
    layers = int(config["backbone_config"]["num_hidden_layers"])
    scale = float(config["lora_alpha"]) / float(config["lora_rank"])
    merged: dict[str, torch.Tensor] = {name: tensor.clone() for name, tensor in mert_weights.items()}
    for layer in range(layers):
        for projection in PROJECTIONS:
            base = f"layers.{layer}.attn.{projection}.weight"
            a = adapter_weights[f"adapter.layers.{layer}.attn.{projection}.lora_A.weight"].float()
            b = adapter_weights[f"adapter.layers.{layer}.attn.{projection}.lora_B.weight"].float()
            merged[base] = merged[base].float().add((b @ a) * scale)
    for name, tensor in adapter_weights.items():
        if not name.startswith("adapter."):
            merged[name] = tensor
    config["weights_format"] = "merged"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, output_path, metadata={"format": "pt"})
    (output_path.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT / "reference" / "SheetSage2",
        help="SheetSage2 adapter snapshot containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--mert",
        type=Path,
        default=Path.home() / "Desktop" / "YuE2" / "MERT-v2-FullSong",
        help="MERT-v2-FullSong parent snapshot containing model.safetensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "SheetSage2-GGUF" / "sheetsage2-orig.gguf",
        help="Output GGUF path.",
    )
    parser.add_argument(
        "--audiocpp-gguf",
        type=Path,
        default=REPO_ROOT / "build" / "debug" / "bin" / "audiocpp_gguf",
        help="Path to audio.cpp GGUF converter.",
    )
    parser.add_argument("--type", default="orig", choices=["orig", "f16", "bf16", "q4_0", "q4_k"])
    parser.add_argument("--keep-merged", type=Path, help="Optional path to keep the merged safetensors directory.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = args.adapter.resolve()
    mert = args.mert.resolve()
    output = args.output.resolve()
    converter = require_file(args.audiocpp_gguf.resolve())
    spec = require_file(REPO_ROOT / "model_specs" / "sheetsage2.json")
    temp_owner = tempfile.TemporaryDirectory(prefix="sheetsage2-merged-")
    merged_dir = args.keep_merged.resolve() if args.keep_merged else Path(temp_owner.name)
    merged_path = merged_dir / "model.safetensors"
    merge_adapter(adapter, mert, merged_path)
    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(merged_dir / "config.json", root / "config.json")
    command = [
        str(converter),
        "--input",
        str(merged_path),
        "--root",
        str(root),
        "--model-spec",
        str(spec),
        "--family",
        "sheetsage2",
        "--output",
        str(output),
        "--type",
        args.type,
    ]
    if args.overwrite:
        command.append("--overwrite")
    run(command)
    temp_owner.cleanup()
    print("[done]", output)


if __name__ == "__main__":
    main()
