#!/usr/bin/env python3
"""Convert Coqui XTTS v2 into the tensor layout consumed by audio.cpp.

The official ``model.pth`` also contains optimizer, scaler, and trainer state.  This
tool exports only the inference graph, resolves PyTorch weight parametrizations,
and stages the tokenizer/config/license sidecars without modifying the source
snapshot.

Example:
    python tools/community_models/convert_xtts_v2.py \
        --model-dir /path/to/coqui-XTTS-v2 \
        --output-dir /tmp/xtts-v2-staging \
        --run-converter build/bin/audiocpp_gguf --type f16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file


PREFIXES = {
    "gpt": "gpt.",
    "decoder": "hifigan_decoder.waveform_decoder.",
    "speaker_encoder": "hifigan_decoder.speaker_encoder.",
}


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def materialize_weight_norm(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace parametrizations.weight.original{0,1} with the effective weight.

    torch.nn.utils.parametrizations.weight_norm uses ``g * v / ||v||`` and, for
    Conv1d/ConvTranspose1d, normalizes over every axis except output channel 0.
    Coqui removes these parametrizations before inference, so the native graph must
    bind the materialized value rather than either checkpoint component.
    """
    out = dict(tensors)
    suffix = ".parametrizations.weight.original0"
    for key in tuple(out):
        if not key.endswith(suffix):
            continue
        stem = key[: -len(suffix)]
        v_key = stem + ".parametrizations.weight.original1"
        if v_key not in out:
            raise KeyError(f"weight-norm tensor {key} has no matching {v_key}")
        g = out.pop(key).float()
        v = out.pop(v_key).float()
        dims = tuple(range(1, v.ndim))
        out[stem + ".weight"] = (v * (g / torch.linalg.vector_norm(v, dim=dims, keepdim=True))).contiguous()
    return out


def extract_group(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    group: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not name.startswith(prefix) or not isinstance(value, torch.Tensor):
            continue
        short = name[len(prefix) :]
        # BatchNorm's counter is training-only and GGUF does not support scalar i64
        # tensors. Running statistics and frontend buffers remain essential.
        if short.endswith(".num_batches_tracked"):
            continue
        group[short] = value.detach().cpu().contiguous()
    if not group:
        raise RuntimeError(f"checkpoint contains no tensors under {prefix!r}")
    return materialize_weight_norm(group)


def validate_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    args = config.get("model_args", {})
    expected = {
        "gpt_layers": 30,
        "gpt_n_model_channels": 1024,
        "gpt_n_heads": 16,
        "gpt_number_text_tokens": 6681,
        "gpt_num_audio_tokens": 1026,
        "gpt_use_perceiver_resampler": True,
        "output_sample_rate": 24000,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    if mismatches:
        raise ValueError(f"unsupported XTTS checkpoint configuration: {mismatches}")


def converter_command(output_dir: Path, converter: Path, quant_type: str) -> list[str]:
    command = [str(converter)]
    for namespace in PREFIXES:
        command += ["--input", f"{namespace}={output_dir / (namespace + '.safetensors')}"]
    command += [
        "--root", str(output_dir / "root"),
        "--family", "xtts_v2",
        "--type", quant_type,
        "--output", str(output_dir / f"xtts-v2-{quant_type}.gguf"),
    ]
    # Convolutions have no quantized execution path. Keeping embeddings and the
    # small output head in F16 also avoids sampling regressions from Q8 logits.
    if quant_type.startswith("q"):
        command += [
            "--keep-type", "decoder/*=f16",
            "--keep-type", "speaker_encoder/*=f16",
            "--keep-type", "gpt/conditioning_encoder.*=f16",
            "--keep-type", "gpt/text_embedding.weight=f16",
            "--keep-type", "gpt/mel_embedding.weight=f16",
            "--keep-type", "gpt/mel_head.*=f16",
        ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-converter", type=Path)
    parser.add_argument("--type", default="f16", choices=("f32", "f16", "q8_0"))
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / "root"
    root.mkdir(parents=True, exist_ok=True)

    config = require_file(model_dir / "config.json", "config.json")
    validate_config(config)
    checkpoint = torch.load(
        require_file(model_dir / "model.pth", "model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise TypeError("XTTS model.pth does not contain a model state dictionary")

    for namespace, prefix in PREFIXES.items():
        tensors = extract_group(state, prefix)
        if namespace == "gpt":
            # Xtts.mel_stats is registered on the model rather than under gpt,
            # but it is an inference input to the conditioning encoder.
            mel_stats = state.get("mel_stats")
            if not isinstance(mel_stats, torch.Tensor) or tuple(mel_stats.shape) != (80,):
                raise RuntimeError("checkpoint is missing the 80-bin XTTS mel_stats tensor")
            tensors["mel_stats"] = mel_stats.detach().cpu().float().contiguous()
        destination = output_dir / f"{namespace}.safetensors"
        save_file(tensors, str(destination))
        parameters = sum(t.numel() for t in tensors.values())
        print(f"wrote {destination} ({len(tensors)} tensors, {parameters:,} values)")

    for filename in ("config.json", "vocab.json", "LICENSE.txt"):
        shutil.copyfile(require_file(model_dir / filename, filename), root / filename)
    print(f"staged config, tokenizer, and CPML license in {root}")

    command = converter_command(output_dir, args.run_converter or Path("audiocpp_gguf"), args.type)
    if args.run_converter:
        subprocess.run(command, check=True)
    else:
        print("next:")
        print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
