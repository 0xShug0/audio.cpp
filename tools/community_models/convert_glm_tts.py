#!/usr/bin/env python3
"""Prepare the published GLM-TTS checkpoint for audio.cpp.

The upstream snapshot already stores the Llama and Whisper-VQ weights as
Safetensors. The remaining PyTorch checkpoints are copied into deterministic
Safetensors files, and the ChatGLM4/tiktoken rank table is converted to the
vocab/merge representation consumed by audio.cpp's shared BPE tokenizer.

This tool deliberately does not download files. Point --model-dir at a complete
zai-org/GLM-TTS snapshot. The native CAMPPlus component uses the canonical
Safetensors layout also published by mlx-community/index-tts2-mlx. Place it at
frontend/campplus.safetensors or pass its path with --campplus-safetensors.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Tuple


def _byte_encoder() -> Dict[int, str]:
    """GPT-2 reversible byte-to-Unicode table used by llama.cpp BPE assets."""

    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    encoded = list(visible)
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            encoded.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(value) for value in encoded)))


def _encoded_token(token: bytes, byte_encoder: Mapping[int, str]) -> str:
    return "".join(byte_encoder[value] for value in token)


def _load_tiktoken_ranks(path: Path) -> Dict[bytes, int]:
    ranks: Dict[bytes, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected base64-token and rank"
                )
            token = base64.b64decode(fields[0], validate=True)
            rank = int(fields[1])
            if token in ranks or rank in ranks.values():
                raise ValueError(f"{path}:{line_number}: duplicate token or rank")
            ranks[token] = rank
    if not ranks:
        raise ValueError(f"empty tiktoken rank table: {path}")
    return ranks


def _merge_parts_for_token(
    token: bytes, rank: int, ranks: Mapping[bytes, int]
) -> Tuple[bytes, bytes]:
    """Recover the final binary merge that creates one tiktoken token."""

    parts = [bytes((value,)) for value in token]
    while len(parts) > 2:
        candidates = [
            (ranks.get(parts[index] + parts[index + 1], math.inf), index)
            for index in range(len(parts) - 1)
        ]
        pair_rank, index = min(candidates)
        if pair_rank >= rank:
            break
        parts[index : index + 2] = [parts[index] + parts[index + 1]]
    if len(parts) != 2 or parts[0] + parts[1] != token:
        raise ValueError(
            f"cannot reconstruct merge for rank {rank} token {token!r}"
        )
    if ranks.get(parts[0], math.inf) >= rank:
        raise ValueError(f"left merge input is not an earlier token at rank {rank}")
    if ranks.get(parts[1], math.inf) >= rank:
        raise ValueError(f"right merge input is not an earlier token at rank {rank}")
    return parts[0], parts[1]


def convert_tokenizer(model_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    tokenizer_dir = model_dir / "vq32k-phoneme-tokenizer"
    rank_path = tokenizer_dir / "tokenizer.model"
    vocab_path = tokenizer_dir / "tokenizer_vocab.json"
    merges_path = tokenizer_dir / "tokenizer_merges.txt"
    if not rank_path.is_file():
        raise FileNotFoundError(rank_path)
    if not overwrite and (vocab_path.exists() or merges_path.exists()):
        raise FileExistsError(
            "converted tokenizer assets already exist; pass --overwrite"
        )

    ranks = _load_tiktoken_ranks(rank_path)
    byte_encoder = _byte_encoder()
    ordered = sorted(ranks.items(), key=lambda item: item[1])
    expected_ranks = list(range(len(ordered)))
    actual_ranks = [rank for _, rank in ordered]
    if actual_ranks != expected_ranks:
        raise ValueError("GLM-TTS tokenizer ranks must be contiguous from zero")

    vocab = {
        _encoded_token(token, byte_encoder): rank for token, rank in ordered
    }
    merges: List[str] = []
    for token, rank in ordered:
        if len(token) == 1:
            continue
        try:
            left, right = _merge_parts_for_token(token, rank, ranks)
        except ValueError:
            # GLM4 deliberately occupies unused base-vocabulary ranks with
            # direct sentinels such as <|UNUSED_pad_256|>. They are valid
            # vocabulary entries but are not constructible BPE merges.
            if token.startswith(b"<|UNUSED_pad_") and token.endswith(b"|>"):
                continue
            raise
        merges.append(
            f"{_encoded_token(left, byte_encoder)} "
            f"{_encoded_token(right, byte_encoder)}"
        )

    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    merges_path.write_text(
        "#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8"
    )
    return vocab_path, merges_path


def _unwrap_state_dict(value: object, source: Path) -> MutableMapping[str, object]:
    if not isinstance(value, MutableMapping):
        raise TypeError(f"{source} does not contain a tensor dictionary")
    for key in ("state_dict", "model", "generator"):
        nested = value.get(key)
        if isinstance(nested, MutableMapping):
            value = nested
            break
    return value


def convert_torch_checkpoint(
    source: Path, destination: Path, overwrite: bool
) -> None:
    import torch
    from safetensors.torch import save_file

    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(checkpoint, source)
    tensors = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensors[str(name)] = value.detach().cpu().contiguous()
    if not tensors:
        raise ValueError(f"{source} contains no tensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(destination),
        metadata={
            "format": "pt",
            "source": source.name,
            "audio.cpp.component": destination.parent.name,
        },
    )


def install_campplus(
    source: Path, model_dir: Path, overwrite: bool
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = model_dir / "frontend" / "campplus.safetensors"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def write_audio_cpp_config(model_dir: Path, overwrite: bool) -> Path:
    destination = model_dir / "audio_cpp_config.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite")
    config = {
        "schema_version": 1,
        "family": "glm_tts",
        "sample_rate": 24000,
        "flow": {
            "speech_token_dim": 512,
            "vocab_size": 100000,
            "mel_dim": 80,
            "trans_dim": 768,
            "depth": 18,
            "heads": 12,
            "dim_head": 64,
            "conv_layers": 4,
            "mel_framerate": 50,
            "input_frame_rate": 25.0,
            "inference_cfg_rate": 0.7,
            "inference_steps": 10,
            "scheduler": "cosine",
            "speaker_embedding_dim": 192,
            "speaker_adaln": True,
            "speech_token_cfg": False,
            "remove_speaker_concat_condition": True,
        },
        "hift": {
            "in_channels": 80,
            "base_channels": 512,
            "nb_harmonics": 8,
            "sampling_rate": 24000,
            "nsf_alpha": 0.1,
            "nsf_sigma": 0.003,
            "nsf_voiced_threshold": 10.0,
            "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7],
            "istft_params": {"n_fft": 16, "hop_len": 4},
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "source_resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            "lrelu_slope": 0.1,
            "audio_limit": 0.99,
            "f0_num_class": 1,
            "f0_in_channels": 80,
            "f0_cond_channels": 512,
        },
    }
    destination.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Complete zai-org/GLM-TTS Hugging Face snapshot.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional zai-org/GLM-TTS source checkout used to validate provenance.",
    )
    parser.add_argument(
        "--campplus-safetensors",
        type=Path,
        help=(
            "Canonical CAMPPlus Safetensors. If omitted, "
            "<model-dir>/frontend/campplus.safetensors must already exist."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if args.reference_dir is not None:
        reference_dir = args.reference_dir.resolve()
        if not (reference_dir / "glmtts_inference.py").is_file():
            raise FileNotFoundError(
                f"{reference_dir} is not a GLM-TTS source checkout"
            )

    vocab, merges = convert_tokenizer(model_dir, args.overwrite)
    convert_torch_checkpoint(
        model_dir / "flow" / "flow.pt",
        model_dir / "flow" / "model.safetensors",
        args.overwrite,
    )
    convert_torch_checkpoint(
        model_dir / "hift" / "hift.pt",
        model_dir / "hift" / "model.safetensors",
        args.overwrite,
    )
    if args.campplus_safetensors is not None:
        campplus = install_campplus(
            args.campplus_safetensors.resolve(), model_dir, args.overwrite
        )
    else:
        campplus = model_dir / "frontend" / "campplus.safetensors"
        if not campplus.is_file():
            raise FileNotFoundError(
                "CAMPPlus Safetensors is required at "
                f"{campplus} or through --campplus-safetensors"
            )
    runtime_config = write_audio_cpp_config(model_dir, args.overwrite)

    print(f"tokenizer vocab:  {vocab}")
    print(f"tokenizer merges: {merges}")
    print(f"flow weights:     {model_dir / 'flow' / 'model.safetensors'}")
    print(f"HiFT weights:     {model_dir / 'hift' / 'model.safetensors'}")
    print(f"CAMPPlus weights: {campplus}")
    print(f"runtime config:   {runtime_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
