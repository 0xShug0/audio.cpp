#!/usr/bin/env python3
"""Convert local ControlFoley PyTorch weights into one audio.cpp GGUF package.

Example:
  build/debug/bin/audiocpp_gguf must exist before running this script.

  python scripts/controlfoley/convert_controlfoley_gguf.py \
    --model-dir models/ControlFoley \
    --bigvgan-dir /home/leo/.cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb \
    --output-dir models/ControlFoley-GGUF
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_state_dict(path: Path, *, trusted: bool = False) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=not trusted)
    if isinstance(obj, dict):
        for key in ("generator", "state_dict", "model", "best_state"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a state dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if isinstance(value, torch.Tensor):
            out[str(key)] = value.detach().cpu().contiguous()
    if not out:
        raise RuntimeError(f"{path} did not contain tensor weights")
    return out


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def strip_prefix(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    needle = prefix + "."
    out = {key[len(needle):]: value for key, value in tensors.items() if key.startswith(needle)}
    if not out:
        raise RuntimeError(f"no tensors matched prefix {prefix}")
    return out


def pack_synchformer_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(tensors)
    key = "vfeat_extractor.patch_embed_3d.proj.weight"
    weight = out.get(key)
    if weight is not None and weight.ndim == 5:
        out[key] = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4]).contiguous()
    return out


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


def write_open_clip_tokenizer_files(bpe_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(bpe_path, "rt", encoding="utf-8") as handle:
        merges = handle.read().splitlines()[1:49152 - 256 - 2 + 1]
    byte_encoder = bytes_to_unicode()
    vocab = {value: index for index, value in enumerate(byte_encoder.values())}
    vocab.update({value + "</w>": index + 256 for index, value in enumerate(byte_encoder.values())})
    for merge in merges:
        vocab["".join(merge.split())] = len(vocab)
    vocab["<start_of_text>"] = len(vocab)
    vocab["<end_of_text>"] = len(vocab)

    vocab_path = output_dir / "open_clip_vocab.json"
    merges_path = output_dir / "open_clip_merges.txt"
    config_path = output_dir / "open_clip_tokenizer_config.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    merges_path.write_text("\n".join(merges) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps({
        "added_tokens_decoder": {
            str(vocab["<start_of_text>"]): {"content": "<start_of_text>", "special": True},
            str(vocab["<end_of_text>"]): {"content": "<end_of_text>", "special": True},
        }
    }, indent=2), encoding="utf-8")
    return vocab_path, merges_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bigvgan-dir", type=Path, required=True)
    parser.add_argument("--open-clip-weights", type=Path, required=True)
    parser.add_argument("--open-clip-bpe", type=Path, required=True)
    parser.add_argument("--musicgen-style-dir", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audiocpp-gguf", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--type", default="f32", choices=["orig", "f32", "f16", "bf16", "q8_0", "q4_k", "q5_k", "q6_k"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    bigvgan_dir = args.bigvgan_dir
    staging = args.output_dir / "_safetensors"
    output = args.output_dir / f"controlfoley-large-44k-{args.type}.gguf"

    flow = load_state_dict(model_dir / "weights" / "controlfoley.pth")
    vae = load_state_dict(model_dir / "ext_weights" / "v1-44.pth")
    bigvgan = load_state_dict(bigvgan_dir / "bigvgan_generator.pt")
    open_clip = load_state_dict(args.open_clip_weights)
    cav_mae = load_state_dict(model_dir / "ext_weights" / "cav_mae_st.pth")
    synchformer = pack_synchformer_tensors(load_state_dict(model_dir / "ext_weights" / "synchformer_state_dict.pth"))
    clap = strip_prefix(
        load_state_dict(model_dir / "ext_weights" / "music_speech_audioset_epoch_15_esc_89.98.pt", trusted=True),
        "module")
    mert = load_state_dict(args.mert_dir / "pytorch_model.bin")
    musicgen_style = strip_prefix(
        load_state_dict(args.musicgen_style_dir / "state_dict.bin", trusted=True),
        "condition_provider.conditioners.self_wav")

    flow_path = staging / "flow.safetensors"
    vae_path = staging / "vae.safetensors"
    bigvgan_path = staging / "bigvgan.safetensors"
    open_clip_path = staging / "open_clip.safetensors"
    cav_mae_path = staging / "cav_mae.safetensors"
    synchformer_path = staging / "synchformer.safetensors"
    clap_path = staging / "clap.safetensors"
    mert_path = staging / "mert.safetensors"
    musicgen_style_path = staging / "musicgen_style.safetensors"
    write_safetensors(flow_path, flow)
    write_safetensors(vae_path, vae)
    write_safetensors(bigvgan_path, bigvgan)
    write_safetensors(open_clip_path, open_clip)
    write_safetensors(cav_mae_path, cav_mae)
    write_safetensors(synchformer_path, synchformer)
    write_safetensors(clap_path, clap)
    write_safetensors(mert_path, mert)
    write_safetensors(musicgen_style_path, musicgen_style)
    tokenizer_paths = write_open_clip_tokenizer_files(args.open_clip_bpe, args.output_dir)

    cmd = [
        str(args.audiocpp_gguf),
        "--input",
        f"flow={flow_path}",
        "--input",
        f"vae={vae_path}",
        "--input",
        f"bigvgan={bigvgan_path}",
        "--input",
        f"open_clip={open_clip_path}",
        "--input",
        f"cav_mae={cav_mae_path}",
        "--input",
        f"synchformer={synchformer_path}",
        "--input",
        f"clap={clap_path}",
        "--input",
        f"mert={mert_path}",
        "--input",
        f"musicgen_style={musicgen_style_path}",
        "--output",
        str(output),
        "--type",
        args.type,
        "--family",
        "controlfoley",
        "--model-spec",
        "model_specs/controlfoley.json",
        "--root",
        str(args.output_dir),
        "--sidecar",
        f"{tokenizer_paths[0]}=open_clip_vocab.json",
        "--sidecar",
        f"{tokenizer_paths[1]}=open_clip_merges.txt",
        "--sidecar",
        f"{tokenizer_paths[2]}=open_clip_tokenizer_config.json",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
