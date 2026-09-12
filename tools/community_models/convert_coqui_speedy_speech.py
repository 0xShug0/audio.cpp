#!/usr/bin/env python3
"""Convert Coqui LJSpeech SpeedySpeech + HiFi-GAN v2 checkpoints.

The output is an inference-only safetensors bundle consumed by audio.cpp.  It
drops the training aligner/optimizer state, materializes BatchNorm evaluation
affines, and removes HiFi-GAN weight normalization.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

GRUUT_LICENSE = """MIT License

Copyright (c) 2020 Michael Hansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

try:
    import numpy as np
    import torch
    from safetensors.numpy import save_file
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Run with `uv run --with torch --with numpy --with safetensors python "
        "tools/community_models/convert_coqui_speedy_speech.py ...`."
    ) from exc


def _json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    out: list[str] = []
    quoted = escaped = line_comment = False
    index = 0
    while index < len(text):
        char = text[index]
        if line_comment:
            if char == "\n":
                line_comment = False
                out.append(char)
        elif quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            out.append(char)
        elif char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            line_comment = True
            index += 1
        else:
            out.append(char)
        index += 1
    return json.loads("".join(out))


def _checkpoint(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    state = value.get("model", value)
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dictionary")
    return state


def _array(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def _fold_weight_norm(state: dict[str, torch.Tensor], prefix: str) -> np.ndarray:
    v = state[prefix + ".weight_v"].detach().float()
    g = state[prefix + ".weight_g"].detach().float()
    axes = tuple(range(1, v.ndim))
    return _array(v * (g / torch.linalg.vector_norm(v, dim=axes, keepdim=True)))


def _validate_configs(acoustic: dict[str, Any], vocoder: dict[str, Any]) -> None:
    args = acoustic["model_args"]
    expected = {
        "model": "speedy_speech",
        "hidden_channels": 128,
        "out_channels": 80,
        "encoder_type": "residual_conv_bn",
        "decoder_type": "residual_conv_bn",
        "use_pitch": False,
        "use_energy": False,
        "num_speakers": 0,
    }
    actual = {
        "model": acoustic.get("model"),
        **{k: args.get(k, False) if k in {"use_pitch", "use_energy"} else args.get(k)
           for k in expected if k != "model"},
    }
    if actual != expected:
        raise ValueError(f"unsupported SpeedySpeech configuration: {actual}")
    if acoustic.get("phonemizer") != "gruut" or acoustic.get("phoneme_language") != "en-us":
        raise ValueError("SpeedySpeech conversion requires the released en-us Gruut frontend")
    generator = vocoder["generator_model_params"]
    if vocoder.get("generator_model") != "hifigan_generator" or generator != {
        "resblock_type": "1",
        "upsample_factors": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_initial_channel": 128,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    }:
        raise ValueError("unsupported HiFi-GAN v2 configuration")


def convert(args: argparse.Namespace) -> None:
    acoustic_config = _json_with_comments(args.acoustic_config)
    vocoder_config = _json_with_comments(args.vocoder_config)
    _validate_configs(acoustic_config, vocoder_config)
    acoustic = _checkpoint(args.acoustic_checkpoint)
    vocoder = _checkpoint(args.vocoder_checkpoint)
    tensors: dict[str, np.ndarray] = {}

    keep = ("emb.", "encoder.", "duration_predictor.", "decoder.")
    norm_prefixes = sorted({name.rsplit(".", 1)[0] for name in acoustic if name.endswith(".running_mean")})
    for name, value in acoustic.items():
        if not name.startswith(keep) or name.endswith("num_batches_tracked"):
            continue
        if any(name.startswith(prefix + ".") for prefix in norm_prefixes):
            continue
        array = _array(value)
        if name.startswith("duration_predictor.norm_") and name.endswith((".gamma", ".beta")):
            array = np.ascontiguousarray(array.reshape(-1))
        tensors["acoustic." + name] = array

    for prefix in norm_prefixes:
        mean = acoustic[prefix + ".running_mean"].float()
        var = acoustic[prefix + ".running_var"].float()
        weight = acoustic[prefix + ".weight"].float()
        bias = acoustic[prefix + ".bias"].float()
        scale = weight * torch.rsqrt(var + 1.0e-5)
        tensors["acoustic." + prefix + ".scale"] = _array(scale)
        tensors["acoustic." + prefix + ".offset"] = _array(bias - mean * scale)

    prefixes = sorted({name[:-9] for name in vocoder if name.endswith(".weight_v")})
    for prefix in prefixes:
        tensors["vocoder." + prefix + ".weight"] = _fold_weight_norm(vocoder, prefix)
        bias = vocoder.get(prefix + ".bias")
        if bias is not None:
            tensors["vocoder." + prefix + ".bias"] = _array(bias)

    normalized = {
        "format": "coqui_speedy_speech_v1",
        "sample_rate": 22050,
        "hop_length": 256,
        "num_mels": 80,
        "vocab_size": 130,
        "hidden_channels": 128,
        "encoder_dilations": acoustic_config["model_args"]["encoder_params"]["dilations"],
        "decoder_dilations": acoustic_config["model_args"]["decoder_params"]["dilations"],
        "phoneme_language": "en-us",
        "phonemes": acoustic_config["characters"]["phonemes"],
        "punctuations": acoustic_config["characters"]["punctuations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"family": "coqui_speedy_speech", "license": "Apache-2.0"})
    args.output_config.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    if args.gruut_lexicon:
        lexicon_output = args.output_config.parent / "gruut-lexicon.tsv"
        with sqlite3.connect(args.gruut_lexicon) as connection, lexicon_output.open("w", encoding="utf-8") as stream:
            rows = connection.execute(
                "SELECT word, phonemes FROM word_phonemes WHERE pron_order = 0 ORDER BY word"
            )
            for word, phonemes in rows:
                # Coqui's Gruut wrapper drops stress, flattens phoneme clusters,
                # and maps ASCII g to IPA script-g.
                value = phonemes.replace(" ", "").replace("ˈ", "").replace("ˌ", "").replace("g", "ɡ")
                stream.write(f"{word}\t{value}\n")
        (args.output_config.parent / "GRUUT_LICENSE.txt").write_text(GRUUT_LICENSE, encoding="utf-8")
    print(f"wrote {len(tensors)} tensors to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--acoustic-config", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-config", type=Path, required=True)
    parser.add_argument(
        "--gruut-lexicon",
        type=Path,
        required=True,
        help="gruut-lang-en lexicon.db used by the checkpoint's training frontend",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
