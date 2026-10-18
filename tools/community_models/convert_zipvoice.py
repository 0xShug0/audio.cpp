#!/usr/bin/env python3
"""Convert ZipVoice / ZipVoice-Distill torch checkpoints to audio.cpp packages.

ZipVoice (https://github.com/k2-fsa/ZipVoice, k2-fsa/ZipVoice on HuggingFace)
ships `model.pt` checkpoints (a pickled dict with a "model" state dict), a
`model.json` architecture config, and a `tokens.txt` phone vocab. audio.cpp
cannot read the nested pickle directly, so this script first flattens the
state dict to safetensors (the development format) and then packages it into
a self-contained GGUF with two tensor namespaces:

  model.*   — the ZipVoice flow-matching model (raw torch names)
  vocos.*   — the Vocos mel-24kHz vocoder used to decode generated features

`tokens.txt` and `model.json` are copied next to the output so the GGUF plus
those two files form a complete model directory for
`audiocpp_cli --family zipvoice`.

Examples:
  # ZipVoice-Distill (8 steps, guidance-scale embedding) with bundled vocoder
  python3 tools/community_models/convert_zipvoice.py \
      --model-dir /models/ZipVoice/zipvoice_distill \
      --vocos /models/vocos-mel-24khz/vocos.safetensors \
      --converter build/bin/audiocpp_gguf

  # safetensors only (development format, no GGUF step)
  python3 tools/community_models/convert_zipvoice.py \
      --model-dir /models/ZipVoice/zipvoice_distill --safetensors-only
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def load_state_dict(model_dir: Path, checkpoint_name: str) -> dict:
    ckpt_path = model_dir / checkpoint_name
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"unexpected checkpoint layout in {ckpt_path}")
    return {name: tensor.contiguous().to(torch.float32)
            for name, tensor in state.items()
            if isinstance(tensor, torch.Tensor)}


def write_safetensors(state: dict, output: Path) -> None:
    from safetensors.torch import save_file
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, "
          f"{len(state)} tensors)")


def convert_gguf(converter: Path, safetensors: Path, vocos: Path,
                 output: Path, quant_type: str, overwrite: bool,
                 root: Path) -> None:
    command = [
        str(converter),
        "--input", f"model={safetensors}",
        "--input", f"vocos={vocos}",
        "--root", str(root),
        "--family", "zipvoice",
        "--type", quant_type,
        "--output", str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="ZipVoice model dir with model.pt / model.json / tokens.txt")
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--vocos", type=Path,
                        default=Path("/models/vocos-mel-24khz/vocos.safetensors"),
                        help="vocos.safetensors (bundled into the GGUF)")
    parser.add_argument("--converter", type=Path,
                        default=Path("build/bin/audiocpp_gguf"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <model-dir>/converted")
    parser.add_argument("--type", default="orig",
                        choices=["orig", "f16", "bf16", "q8_0", "q2_k", "q3_k",
                                 "q4_k", "q5_k", "q6_k"],
                        help="GGUF storage type (default orig = keep f32)")
    parser.add_argument("--name", default=None,
                        help="package base name (default: model-dir name, lowercased)")
    parser.add_argument("--safetensors-only", action="store_true",
                        help="stop after writing the flattened safetensors")
    parser.add_argument("--no-vocos", action="store_true",
                        help="do not bundle a vocoder namespace into the GGUF")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    for required in ("model.json", "tokens.txt"):
        if not (model_dir / required).is_file():
            raise SystemExit(f"{required} missing in {model_dir}")
    with (model_dir / "model.json").open() as f:
        config = json.load(f)
    has_guidance_embed = (
        config.get("distill", {}).get("guidance_scale_embed", False)
        if isinstance(config.get("distill"), dict) else False
    )
    name = args.name or model_dir.name.lower().replace("_", "-")
    out_dir = (args.output_dir or model_dir / "converted").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state_dict(model_dir, args.checkpoint_name)
    print(f"loaded {len(state)} tensors from {model_dir / args.checkpoint_name}")

    # Fixed name matching model_specs/zipvoice.json sources[format=safetensors]
    # ("model:zipvoice-orig.safetensors"); the GGUF keeps the -<type> naming.
    safetensors_path = out_dir / "zipvoice-orig.safetensors"
    write_safetensors(state, safetensors_path)

    # dev-format directory: safetensors + config + vocab
    shutil.copyfile(model_dir / "tokens.txt", out_dir / "tokens.txt")
    shutil.copyfile(model_dir / "model.json", out_dir / "model.json")
    # Chinese frontend sidecars (pypinyin tables + jieba dictionaries), when
    # the model directory was staged by export_zipvoice_zh_dict.py. Staged
    # into out_dir BEFORE packaging: the converter root is scanned for
    # sidecars, so these ride inside the GGUF and a converted package is a
    # single self-sufficient file (loose copies below remain for the
    # directory layout).
    for sidecar in sorted(model_dir.glob("zh_*")):
        shutil.copyfile(sidecar, out_dir / sidecar.name)
    staged_zh = sorted(p.name for p in out_dir.glob("zh_*"))
    print(f"copied tokens.txt / model.json -> {out_dir}"
          + (f" + {len(staged_zh)} zh frontend sidecars" if staged_zh else ""))

    if args.safetensors_only:
        return

    converter = args.converter.resolve()
    if not converter.is_file():
        raise SystemExit(f"converter not found: {converter} (build target audiocpp_gguf)")

    gguf_dir = out_dir / name
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"{name}-{args.type}.gguf"
    # NOTE: --root must point at a directory with real (non-symlinked)
    # sidecar files; HF-cache snapshots use blob symlinks that the sidecar
    # scanner does not enumerate, so pass the flattened dev directory that
    # already carries tokens.txt / model.json copies.
    root = out_dir if (out_dir / "tokens.txt").is_file() else model_dir
    if args.no_vocos:
        command = [
            str(converter),
            "--input", f"model={safetensors_path}",
            "--root", str(root),
            "--family", "zipvoice",
            "--type", args.type,
            "--output", str(gguf_path),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    else:
        vocos = args.vocos.resolve()
        if not vocos.is_file():
            raise SystemExit(f"vocos checkpoint not found: {vocos} "
                             "(pass --no-vocos or --vocos <path>)")
        convert_gguf(converter, safetensors_path, vocos, gguf_path,
                     args.type, args.overwrite, root)
    shutil.copyfile(model_dir / "tokens.txt", gguf_dir / "tokens.txt")
    shutil.copyfile(model_dir / "model.json", gguf_dir / "model.json")
    # Chinese frontend sidecars (pypinyin tables + jieba dictionaries), when
    # the model directory was staged by export_zipvoice_zh_dict.py.
    for sidecar in sorted(model_dir.glob("zh_*")):
        shutil.copyfile(sidecar, gguf_dir / sidecar.name)
    print(f"package ready: {gguf_path}")


if __name__ == "__main__":
    main()
