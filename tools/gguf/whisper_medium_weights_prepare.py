#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenAI Whisper medium.pt to safetensors for audio.cpp.")
    parser.add_argument(
        "--ckpt",
        default=str(Path.home() / ".cache" / "whisper" / "medium.pt"),
        help="Path to OpenAI Whisper medium.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/whisper-medium",
        help="Directory to write model.safetensors and config.json.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def require_dims(payload: dict[str, Any]) -> dict[str, int]:
    raw_dims = payload.get("dims")
    if not isinstance(raw_dims, dict):
        raise RuntimeError("Whisper checkpoint is missing dims")
    dims: dict[str, int] = {}
    for key, value in raw_dims.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise RuntimeError(f"invalid Whisper dim entry: {key!r}={value!r}")
        dims[key] = value
    return dims


def require_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise RuntimeError("Whisper checkpoint is missing model_state_dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise RuntimeError(f"invalid non-string Whisper tensor key: {key!r}")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"invalid non-tensor Whisper state entry: {key}")
        state[key] = value.detach().cpu().contiguous()
    return state


def export_config(dims: dict[str, int], tensor_count: int) -> dict[str, Any]:
    return {
        "model_type": "whisper",
        "variant": "medium",
        "source_format": "openai_whisper_pt",
        "weight_file": "model.safetensors",
        "tensor_count": tensor_count,
        "dims": dims,
        "audio_encoder": {
            "n_mels": dims["n_mels"],
            "n_audio_ctx": dims["n_audio_ctx"],
            "n_audio_state": dims["n_audio_state"],
            "n_audio_head": dims["n_audio_head"],
            "n_audio_layer": dims["n_audio_layer"],
        },
        "text_decoder": {
            "n_vocab": dims["n_vocab"],
            "n_text_ctx": dims["n_text_ctx"],
            "n_text_state": dims["n_text_state"],
            "n_text_head": dims["n_text_head"],
            "n_text_layer": dims["n_text_layer"],
        },
    }


def main() -> int:
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt)
    output_dir = resolve_path(args.output_dir)
    if not ckpt_path.is_file():
        raise RuntimeError(f"Whisper checkpoint does not exist: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected Whisper checkpoint payload type: {type(payload)}")

    dims = require_dims(payload)
    state = require_state_dict(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"
    save_file(state, str(model_path))

    reloaded = load_file(str(model_path))
    if set(reloaded.keys()) != set(state.keys()):
        raise RuntimeError("saved Whisper safetensors key set does not match source checkpoint")
    for key, value in state.items():
        if not torch.equal(value, reloaded[key]):
            raise RuntimeError(f"saved Whisper safetensors changed tensor: {key}")

    (output_dir / "config.json").write_text(
        json.dumps(export_config(dims, len(state)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"ckpt={ckpt_path}")
    print(f"output_dir={output_dir}")
    print(f"tensor_count={len(state)}")
    print(f"model_safetensors={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
