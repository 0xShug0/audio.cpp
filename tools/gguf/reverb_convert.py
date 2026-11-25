#!/usr/bin/env python3
"""Convert a Reverb-TTS checkpoint to an audio.cpp `reverb` package and, optionally, GGUF.

    hf download software-mansion/Reverb-TTS --revision v1.0.0 \
        --include "model/safetensors/fp16/*" --include "model/tokenizer*" --include "model/chat_template.jinja" --include "voices/*" \
        --local-dir Reverb-TTS
    hf download eustlb/higgs-audio-v2-tokenizer --revision 0e3a9177c2991016b8e5e17d23aa474ee665bef2 \
        --local-dir higgs-audio-v2-tokenizer
    python tools/gguf/reverb_convert.py --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer \
        --output Reverb-TTS-package \
        --audiocpp-gguf build/bin/audiocpp_gguf --gguf Reverb-TTS-GGUF/reverb-tts-q8_0.gguf --type q8_0

The package directory holds the variant's config.json and weights (int8/int4 variants are expanded
to fp16, as reverb_tts.model.dequantize does), the tokenizer, the Higgs Audio v2 codec under
audio_tokenizer/, and the 16 preset voices merged into voices.json. It loads directly as a
safetensors `reverb` model. With --audiocpp-gguf and --gguf it is then packed into one GGUF.
Needs numpy and safetensors; torch is not required.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

CODEC_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
SCALE = "_scale"
QUANTIZED = "llm.layers."


def dequantize(weights: dict, quantization: dict) -> dict:
    """reverb_tts.model.dequantize, in numpy: each code times its scale, kept as fp16."""
    if quantization.get("rotation") or quantization.get("activations"):
        raise SystemExit("rotated or activation-quantized checkpoints are not supported; use fp16 or fp32")
    bits, group = quantization["bits"], quantization.get("group_size")
    for key in [k for k in weights if k.endswith(SCALE)]:
        name, scale = key[: -len(SCALE)], weights.pop(key).astype(np.float32)
        codes = weights[name]
        projection = name.startswith(QUANTIZED)
        if projection and bits == 4:
            # Two codes per byte, the first in the low nibble, each stored as code + 8.
            low, high = codes & 0xF, codes >> 4
            codes = np.stack([low, high], axis=-1).reshape(*codes.shape[:-1], -1).astype(np.int16) - 8
        codes = codes.astype(np.float32)
        if projection and group:
            shaped = codes.reshape(*codes.shape[:-1], -1, group) * scale[..., None]
            weights[name] = shaped.reshape(*codes.shape).astype(np.float16)
        else:
            weights[name] = (codes * (scale if scale.ndim == codes.ndim else scale[..., None])).astype(np.float16)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reverb", required=True, type=Path, help="software-mansion/Reverb-TTS snapshot")
    parser.add_argument("--codec", required=True, type=Path, help="eustlb/higgs-audio-v2-tokenizer snapshot")
    parser.add_argument("--output", required=True, type=Path, help="package directory to write")
    parser.add_argument("--variant", default="fp16", choices=("fp32", "fp16", "int8", "int4"))
    parser.add_argument("--audiocpp-gguf", type=Path, help="audiocpp_gguf binary, to also write a GGUF")
    parser.add_argument("--gguf", type=Path, help="GGUF output path")
    parser.add_argument("--type", default="q8_0", help="audiocpp_gguf --type (f16, q8_0, ...)")
    parser.add_argument("--model-spec", type=Path,
                        default=Path(__file__).resolve().parents[2] / "model_specs" / "reverb.json")
    args = parser.parse_args()
    if (args.audiocpp_gguf is None) != (args.gguf is None):
        parser.error("--audiocpp-gguf and --gguf go together")

    source = args.reverb / "model" / "safetensors" / args.variant
    out = args.output
    (out / "audio_tokenizer").mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if "reverb" not in config:
        raise SystemExit(f"{source}/config.json is not a Reverb-TTS config")
    weights = load_file(str(source / "model.safetensors"))
    quantization = config.pop("quantization", None)
    if quantization:
        weights = dequantize(weights, quantization)
    weights.pop("codebook_layer_offsets", None)  # fixed c * 1025 offsets, applied in code
    save_file(weights, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (args.reverb / "model" / name).is_file():
            shutil.copyfile(args.reverb / "model" / name, out / name)
    for name in CODEC_FILES:
        shutil.copyfile(args.codec / name, out / "audio_tokenizer" / name)

    voices = {}
    for path in sorted((args.reverb / "voices").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        codes = np.asarray(data["codes"], dtype=np.int64).reshape(8, -1)
        voices[path.stem] = {
            "language": data["language"],
            "transcript": data["transcript"],
            "speech_frames": data.get("speech_frames"),
            "codes": codes.reshape(-1).tolist(),
        }
    if not voices:
        raise SystemExit(f"no preset voices under {args.reverb / 'voices'}")
    (out / "voices.json").write_text(json.dumps(voices, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(weights)} tensors ({args.variant}), {len(voices)} voices")

    if args.gguf is not None:
        args.gguf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(args.audiocpp_gguf), "--family", "reverb", "--model-spec", str(args.model_spec),
                        "--root", str(out),
                        "--input", f"weights={out / 'model.safetensors'}",
                        "--input", f"audio_tokenizer_weights={out / 'audio_tokenizer' / 'model.safetensors'}",
                        "--output", str(args.gguf), "--type", args.type, "--overwrite"], check=True)
        print(f"wrote {args.gguf}")


if __name__ == "__main__":
    main()
