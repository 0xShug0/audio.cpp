#!/usr/bin/env python3
"""Prepare an official IndexTTS-2.5 checkpoint for audio.cpp's GGUF converter.

Point --model-dir at a complete IndexTeam/IndexTTS-2.5 snapshot. The script writes
a staging directory whose Safetensors layout matches the tensor namespaces the
index_tts2_5 engine expects, plus a root/ directory with the sidecar files
(config, tokenizer, auxiliary model configs) that audiocpp_gguf embeds into the
final GGUF.

The w2v-bert-2.0, CAMPPlus and BigVGAN checkpoints are not part of the official
snapshot; they are auto-detected under <model-dir>/hf_cache/ (where the official
downloader places them) and each can be overridden explicitly.

This tool does not download anything and never writes into --model-dir.

Example:
    python tools/convert_index_tts2_5.py \
        --model-dir /path/to/IndexTTS-2.5 \
        --output-dir /path/to/staging

Then run the printed audiocpp_gguf command, or let the script run it:

    python tools/convert_index_tts2_5.py \
        --model-dir /path/to/IndexTTS-2.5 \
        --output-dir /path/to/staging \
        --run-converter /path/to/audiocpp_gguf --type f16

Add --native-dir /path/to/native-model to also emit a directly loadable
native Safetensors model directory (no GGUF conversion needed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

# GGUF tensor namespaces (must match the index_tts2 model spec) and the
# staging file each one is produced from.
TENSOR_OUTPUTS = [
    ("gpt", "gpt.safetensors"),
    ("s2mel", "s2mel.safetensors"),
    ("speaker_matrix", "speaker_matrix.safetensors"),
    ("emotion_matrix", "emotion_matrix.safetensors"),
    ("wav2vec2bert_stats", "wav2vec2bert_stats.safetensors"),
    ("wav2vec2bert", "wav2vec2bert.safetensors"),
    ("semantic_codec", "semantic_codec.safetensors"),
    ("campplus", "campplus.safetensors"),
    ("bigvgan", "bigvgan.safetensors"),
    ("qwen_emotion", "qwen_emotion.safetensors"),
]

QWEN_SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _flatten(obj, prefix: str = "", out: Dict[str, torch.Tensor] | None = None) -> Dict[str, torch.Tensor]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(value, f"{prefix}{key}.", out)
    elif hasattr(obj, "shape"):
        out[prefix.rstrip(".")] = obj.contiguous()
    else:
        raise TypeError(f"unexpected non-tensor leaf at {prefix!r}: {type(obj)}")
    return out


def _save_safetensors(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))
    print(f"wrote {path} ({len(tensors)} tensors)")


def _convert_gpt(model_dir: Path, output_dir: Path) -> None:
    # gpt.pth is already a flat tensor dict (includes spk_emb_proj and
    # lang_embedding; the 2.5 checkpoint has no conditioning_encoder/speed_emb).
    obj = _load_checkpoint(_require_file(model_dir / "gpt.pth", "gpt.pth"))
    _save_safetensors(_flatten(obj), output_dir / "gpt.safetensors")


def _convert_s2mel(model_dir: Path, output_dir: Path) -> None:
    # s2mel.pth wraps the state dict under "net" (cfm.*/length_regulator.*/gpt_layer.*).
    obj = _load_checkpoint(_require_file(model_dir / "s2mel.pth", "s2mel.pth"))
    if isinstance(obj, dict) and isinstance(obj.get("net"), dict):
        obj = obj["net"]
    _save_safetensors(_flatten(obj), output_dir / "s2mel.safetensors")


def _convert_semantic_codec(model_dir: Path, output_dir: Path) -> None:
    # codec.pth wraps the state dict under "model" (encoder.*/decoder.*/quantizer.*/down/up).
    obj = _load_checkpoint(_require_file(model_dir / "codec.pth", "codec.pth"))
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        obj = obj["model"]
    _save_safetensors(_flatten(obj), output_dir / "semantic_codec.safetensors")


def _convert_emotion_matrices(model_dir: Path, output_dir: Path) -> None:
    # feat1.pt/feat2.pt hold a single root-level tensor each: (73, 192) speaker
    # matrix and (73, 1280) emotion matrix.
    speaker = _load_checkpoint(_require_file(model_dir / "feat1.pt", "feat1.pt"))
    emotion = _load_checkpoint(_require_file(model_dir / "feat2.pt", "feat2.pt"))
    _save_safetensors({"tensor": speaker.float().contiguous()}, output_dir / "speaker_matrix.safetensors")
    _save_safetensors({"tensor": emotion.float().contiguous()}, output_dir / "emotion_matrix.safetensors")


def _convert_wav2vec2bert_stats(model_dir: Path, output_dir: Path) -> None:
    obj = _load_checkpoint(_require_file(model_dir / "wav2vec2bert_stats.pt", "wav2vec2bert_stats.pt"))
    flat = {key: value.float().contiguous() for key, value in _flatten(obj).items()}
    _save_safetensors(flat, output_dir / "wav2vec2bert_stats.safetensors")


def _convert_campplus(campplus_checkpoint: Path, output_dir: Path) -> None:
    # The engine binds CAMPPlus weights under the "speaker_encoder." prefix.
    obj = _load_checkpoint(_require_file(campplus_checkpoint, "campplus checkpoint"))
    flat = _flatten(obj)
    flat = {key if key.startswith("speaker_encoder.") else f"speaker_encoder.{key}": value for key, value in flat.items()}
    _save_safetensors(flat, output_dir / "campplus.safetensors")


def _convert_bigvgan(bigvgan_dir: Path, output_dir: Path) -> None:
    # bigvgan_generator.pt stores keys with a "generator." prefix; the engine
    # expects bare names (conv_pre/ups.N/...).
    obj = _load_checkpoint(_require_file(bigvgan_dir / "bigvgan_generator.pt", "bigvgan generator"))
    flat = _flatten(obj)
    flat = {key[len("generator."):] if key.startswith("generator.") else key: value for key, value in flat.items()}
    _save_safetensors(flat, output_dir / "bigvgan.safetensors")


def _copy(src: Path, dst: Path, label: str) -> None:
    _require_file(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"copied {src} -> {dst}")


def _stage_config_v2_5(src: Path, dst: Path) -> None:
    """Stage config.yaml with the version field normalized to "2.5".

    The official IndexTTS-2.5 snapshot ships config.yaml with `version: 2.0`
    (inherited from IndexTTS-2). audio.cpp selects the IndexTTS2 family variant
    from this field, so the staged copy must declare 2.5 explicitly.
    """
    _require_file(src, "config.yaml")
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith("version:"):
            lines[i] = 'version: "2.5"'
            replaced = True
            break
    if not replaced:
        lines.append('version: "2.5"')
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"staged {src} -> {dst} (version normalized to \"2.5\")")


def build_converter_command(output_dir: Path, converter: str, quant_type: str) -> list[str]:
    command = [converter]
    for namespace, filename in TENSOR_OUTPUTS:
        command += ["--input", f"{namespace}={output_dir / filename}"]
    command += [
        "--root", str(output_dir / "root"),
        "--family", "index_tts2",
        "--type", quant_type,
        "--output", str(output_dir / f"index-tts2_5-{quant_type}.gguf"),
    ]
    return command


# Native Safetensors model directory layout: the spec's safetensors source maps
# logical tensor groups to these paths under the model root.
NATIVE_TENSOR_LAYOUT = [
    # (staging filename, relative path in the native model directory)
    ("gpt.safetensors", "gpt.safetensors"),
    ("s2mel.safetensors", "s2mel.safetensors"),
    ("speaker_matrix.safetensors", "feat1.safetensors"),
    ("emotion_matrix.safetensors", "feat2.safetensors"),
    ("wav2vec2bert_stats.safetensors", "wav2vec2bert_stats.safetensors"),
    ("wav2vec2bert.safetensors", "w2v-bert-2.0/model.safetensors"),
    ("semantic_codec.safetensors", "semantic_codec_model.safetensors"),
    ("campplus.safetensors", "campplus.safetensors"),
    ("bigvgan.safetensors", "bigvgan/model.safetensors"),
    ("qwen_emotion.safetensors", "qwen0.6bemo4-merge/model.safetensors"),
]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def write_native_layout(output_dir: Path, native_dir: Path) -> None:
    """Assemble the directly loadable native Safetensors model directory."""
    for staging_name, relative in NATIVE_TENSOR_LAYOUT:
        _link_or_copy(_require_file(output_dir / staging_name, staging_name), native_dir / relative)
    root_dir = output_dir / "root"
    for name in ("config.yaml", "multilingual_zh_ja_yue_char_del.tiktoken",
                 "w2v-bert-2.0/config.json", "w2v-bert-2.0/preprocessor_config.json",
                 "bigvgan/config.json"):
        _link_or_copy(_require_file(root_dir / name, name), native_dir / name)
    for name in QWEN_SIDECARS:
        _link_or_copy(_require_file(root_dir / "qwen0.6bemo4-merge" / name, name),
                      native_dir / "qwen0.6bemo4-merge" / name)
    print(f"native model directory written to {native_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage an official IndexTTS-2.5 snapshot for audio.cpp's GGUF converter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to the official IndexTTS-2.5 snapshot (gpt.pth, s2mel.pth, ...).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Staging directory to write; created if missing. Never inside --model-dir.")
    parser.add_argument("--w2v-bert-dir", type=Path, default=None,
                        help="Directory with w2v-bert-2.0 model.safetensors/config.json/preprocessor_config.json. "
                             "Default: <model-dir>/hf_cache/w2v-bert-2.0")
    parser.add_argument("--campplus-checkpoint", type=Path, default=None,
                        help="Path to campplus_cn_common.bin. Default: <model-dir>/hf_cache/campplus/campplus_cn_common.bin")
    parser.add_argument("--bigvgan-dir", type=Path, default=None,
                        help="Directory with bigvgan_generator.pt and config.json. Default: <model-dir>/hf_cache/bigvgan")
    parser.add_argument("--run-converter", type=str, default=None,
                        help="Path to the audiocpp_gguf executable; when given, run the GGUF conversion right away.")
    parser.add_argument("--native-dir", type=Path, default=None,
                        help="Also write a directly loadable native Safetensors model directory (the layout of "
                             "the spec's safetensors source: feat1/feat2.safetensors, semantic_codec_model.safetensors, "
                             "bigvgan/model.safetensors, w2v-bert-2.0/, qwen0.6bemo4-merge/). Files are hardlinked "
                             "from the staging directory when possible, copied otherwise.")
    parser.add_argument("--type", dest="quant_type", default="f16",
                        choices=("orig", "f16", "bf16", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"),
                        help="GGUF weight type used with --run-converter and in the printed command (default: f16).")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not model_dir.is_dir():
        print(f"error: --model-dir does not exist: {model_dir}", file=sys.stderr)
        return 1
    if output_dir == model_dir or model_dir in output_dir.parents:
        print("error: --output-dir must not be inside --model-dir", file=sys.stderr)
        return 1

    hf_cache = model_dir / "hf_cache"
    w2v_bert_dir = (args.w2v_bert_dir or hf_cache / "w2v-bert-2.0").resolve()
    campplus_checkpoint = (args.campplus_checkpoint or hf_cache / "campplus" / "campplus_cn_common.bin").resolve()
    bigvgan_dir = (args.bigvgan_dir or hf_cache / "bigvgan").resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = output_dir / "root"
    root_dir.mkdir(parents=True, exist_ok=True)

    _convert_gpt(model_dir, output_dir)
    _convert_s2mel(model_dir, output_dir)
    _convert_semantic_codec(model_dir, output_dir)
    _convert_emotion_matrices(model_dir, output_dir)
    _convert_wav2vec2bert_stats(model_dir, output_dir)
    _convert_campplus(campplus_checkpoint, output_dir)
    _convert_bigvgan(bigvgan_dir, output_dir)

    # Auxiliary checkpoints ship as Safetensors already; copy them through.
    _copy(w2v_bert_dir / "model.safetensors", output_dir / "wav2vec2bert.safetensors", "w2v-bert-2.0 weights")
    _copy(model_dir / "qwen0.6bemo4-merge" / "model.safetensors", output_dir / "qwen_emotion.safetensors",
          "qwen emotion weights")

    # Sidecar files embedded into the GGUF via --root.
    _stage_config_v2_5(model_dir / "config.yaml", root_dir / "config.yaml")
    _copy(model_dir / "multilingual_zh_ja_yue_char_del.tiktoken",
          root_dir / "multilingual_zh_ja_yue_char_del.tiktoken", "tiktoken vocabulary")
    _copy(w2v_bert_dir / "config.json", root_dir / "w2v-bert-2.0" / "config.json", "w2v-bert-2.0 config")
    _copy(w2v_bert_dir / "preprocessor_config.json", root_dir / "w2v-bert-2.0" / "preprocessor_config.json",
          "w2v-bert-2.0 preprocessor config")
    _copy(bigvgan_dir / "config.json", root_dir / "bigvgan" / "config.json", "bigvgan config")
    for name in QWEN_SIDECARS:
        _copy(model_dir / "qwen0.6bemo4-merge" / name, root_dir / "qwen0.6bemo4-merge" / name, f"qwen sidecar {name}")

    if args.native_dir is not None:
        native_dir = args.native_dir.resolve()
        if native_dir == model_dir or model_dir in native_dir.parents:
            print("error: --native-dir must not be inside --model-dir", file=sys.stderr)
            return 1
        native_dir.mkdir(parents=True, exist_ok=True)
        write_native_layout(output_dir, native_dir)

    converter = args.run_converter or "audiocpp_gguf"
    command = build_converter_command(output_dir, converter, args.quant_type)
    lines = [command[0]]
    index = 1
    while index < len(command):
        flag = command[index]
        if flag.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            lines.append(f"    {flag} {command[index + 1]} \\")
            index += 2
        else:
            lines.append(f"    {flag} \\")
            index += 1
    lines[-1] = lines[-1].rstrip(" \\")
    print()
    print("staging complete. Convert to GGUF with:")
    print("\n".join(lines))

    if args.run_converter is not None:
        print()
        print("running converter...")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"error: converter exited with {result.returncode}", file=sys.stderr)
            return result.returncode
        print(f"GGUF written to {output_dir / f'index-tts2_5-{args.quant_type}.gguf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
