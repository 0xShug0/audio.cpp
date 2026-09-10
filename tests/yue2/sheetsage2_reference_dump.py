#!/usr/bin/env python3

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().cpu().contiguous().float().numpy().astype("<f4").tobytes())


def write_i32(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().cpu().contiguous().to(torch.int32).numpy().astype("<i4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio-frames", type=int, default=24000 * 8)
    parser.add_argument("--decoder-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str((root / "reference").resolve()))
    transformers = importlib.import_module("transformers452")
    sys.modules["transformers"] = transformers
    for name in (
        "configuration_utils",
        "dynamic_module_utils",
        "generation",
        "generation.utils",
        "modeling_outputs",
        "modeling_utils",
        "models",
        "models.auto",
        "models.auto.configuration_auto",
        "models.auto.modeling_auto",
        "models.bart",
        "models.bart.configuration_bart",
        "models.bart.modeling_bart",
        "utils",
        "utils.hub",
    ):
        try:
            sys.modules[f"transformers.{name}"] = importlib.import_module(f"transformers452.{name}")
        except ModuleNotFoundError:
            pass
    import transformers452.dynamic_module_utils as transformers452_dynamic_module_utils
    transformers452_dynamic_module_utils.transformers = transformers
    from SheetSage2.modeling_sheetsage2 import SheetSage2Model

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SheetSage2Model.from_pretrained(
        args.model,
        base_model_path=args.base_model,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()

    # A deterministic full-band synthetic waveform is long enough to exercise
    # the real MERT2 front end without relying on a tiny smoke shape.
    t = torch.arange(args.audio_frames, dtype=torch.float32) / float(model.config.sampling_rate)
    waveform = (
        0.20 * torch.sin(2.0 * torch.pi * 220.0 * t)
        + 0.07 * torch.sin(2.0 * torch.pi * 440.0 * t + 0.3)
        + 0.03 * torch.sin(2.0 * torch.pi * 880.0 * t + 0.9)
    ).unsqueeze(0)

    with torch.no_grad():
        enc = model.get_audio_features(waveform, output_hidden_states=False, return_dict=True)
        mixed = enc.mixed_hidden_state.detach().contiguous()
        ids = torch.arange(args.decoder_steps, dtype=torch.long).unsqueeze(0)
        ids = (ids * 17 + 5) % model.config.vocab_size
        ids[:, 0] = model.config.bos_token_id
        logits, _ = model.decode(mixed @ model.encoder_projection.weight.T + model.encoder_projection.bias, ids)

    write_f32(out_dir / "mixed_encoder_state.f32", mixed)
    write_i32(out_dir / "decoder_input_ids.i32", ids)
    write_f32(out_dir / "logits_ref.f32", logits)
    meta = {
        "seed": args.seed,
        "audio_frames": args.audio_frames,
        "decoder_steps": args.decoder_steps,
        "batch": int(mixed.shape[0]),
        "memory_steps": int(mixed.shape[1]),
        "encoder_hidden_size": int(mixed.shape[2]),
        "logits_shape": [int(v) for v in logits.shape],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
