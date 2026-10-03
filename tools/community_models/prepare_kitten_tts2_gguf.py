#!/usr/bin/env python3
"""Convert local Kitten TTS 2 assets into a self-contained audio.cpp GGUF.

No network access or Python model loading. The native converter preserves the
HF tensor names and embeds the native decoder, tokenizer, and speaker weights.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path


def read_voice_array(path: Path, name: str, dtype: str) -> list:
    """Read upstream's flat numeric NPY arrays without NumPy or pickle."""
    with zipfile.ZipFile(path) as archive:
        data = archive.read(name + ".npy")
    if data[:6] != b"\x93NUMPY" or data[6:8] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError(f"unsupported NPY header: {path}:{name}")
    width = 2 if data[6] == 1 else 4
    offset = 8 + width
    length = int.from_bytes(data[8:offset], "little")
    header = ast.literal_eval(data[offset:offset + length].decode("utf-8"))
    shape = header["shape"]
    if (header["descr"] not in ("<" + dtype, ">" + dtype) or
            header["fortran_order"] or not isinstance(shape, tuple) or len(shape) != 1 or
            not isinstance(shape[0], int) or shape[0] <= 0):
        raise ValueError(f"expected a flat {dtype} array: {path}:{name}")
    payload = data[offset + length:]
    if len(payload) != shape[0] * 4:
        raise ValueError(f"incorrect NPY payload size: {path}:{name}")
    fmt = header["descr"][0] + ("f" if dtype == "f4" else "i")
    values = [item[0] for item in struct.iter_unpack(fmt, payload)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite voice array: {path}:{name}")
    return values


def prepare_voices(args, model: Path, s3gen: Path, root: Path, required: set[str]) -> Path:
    base = (args.prepared_voices or model / "cpp/default/voices.json").resolve()
    prepared = json.loads(base.read_text(encoding="utf-8"))
    if not args.prepared_voices:
        index_path = model / "voices/voices.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = {name: entry for name, entry in index.items() if name not in prepared}
        if missing:
            preparer = args.voice_preparer or args.converter.with_name(
                "audiocpp_kitten_tts2_prepare_voices" + args.converter.suffix)
            if not preparer.is_file():
                raise FileNotFoundError(f"build audiocpp_kitten_tts2_prepare_voices or pass --voice-preparer: {preparer}")
            manifest = {}
            for name, entry in missing.items():
                artifacts = index_path.parent / entry["artifacts"]
                identity = read_voice_array(artifacts, "embedding", "f4")
                tokens = read_voice_array(artifacts, "reference_tokens", "i4")
                if len(identity) != 512 or any(token < 0 or token >= 6561 for token in tokens):
                    raise ValueError(f"invalid upstream voice artifacts: {name}")
                manifest[name] = {
                    "reference": str((index_path.parent / entry["reference"]).resolve()),
                    "transcript": entry["transcript"], "identity": identity, "reference_tokens": tokens,
                }
            manifest_path = root / "voice-manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            output = root / "voices.json"
            subprocess.run([str(preparer.resolve()), str(model), str(s3gen), str(manifest_path),
                            str(output), args.voice_backend, str(args.threads)], check=True)
            additions = json.loads(output.read_text(encoding="utf-8"))
            if additions.keys() != missing.keys():
                raise ValueError("native preparer returned an unexpected set of voices")
            # Retain existing decimal values verbatim as Python numbers. Only
            # newly prepared voices pass through the native JSON serializer.
            prepared.update(additions)
            output.write_text(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            base = output
    absent = required - prepared.keys()
    if absent:
        raise ValueError(f"prepared voice index is missing registered presets: {', '.join(sorted(absent))}")
    if args.voices_output:
        args.voices_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, args.voices_output)
    print(f"Packaging {len(prepared)} prepared voices.", flush=True)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output GGUF.")
    parser.add_argument("--type", choices=("orig", "f16", "q8_0"), default="q8_0")
    parser.add_argument("--s3gen", type=Path, help="Official s3gen_meanflow.safetensors (defaults to MODEL/native/).")
    parser.add_argument("--s3-license", type=Path, help="Chatterbox LICENSE (defaults to MODEL/native/LICENSE).")
    parser.add_argument("--voice-preparer", type=Path, help="Native voice preparer (defaults to the converter's sibling executable).")
    parser.add_argument("--voice-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepared-voices", type=Path, help="Use an already prepared, complete voice index.")
    parser.add_argument("--voices-output", type=Path, help="Also save the complete prepared voice index for reuse or validation.")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    if args.voices_output and args.voices_output.exists() and not args.overwrite:
        parser.error("--voices-output exists; pass --overwrite to replace it")
    if args.threads < 1:
        parser.error("--threads must be positive")
    model = args.model.resolve()
    decoder_dir = model / "cpp" / "default"
    s3gen = (args.s3gen or model / "native" / "s3gen_meanflow.safetensors").resolve()
    s3_license = (args.s3_license or model / "native" / "LICENSE").resolve()
    speaker = model / "speaker" / "model.safetensors"
    source = model / "lm" / "model.safetensors"
    spec_path = Path(__file__).resolve().parents[2] / "model_specs" / "kitten_tts2.json"
    if json.loads((model / "config.json").read_text(encoding="utf-8"))["type"] != "KITTEN2":
        parser.error("--model must name a Kitten TTS 2 repository")
    sidecars = {
        "config.json": model / "config.json",
        "lm/config.json": model / "lm" / "config.json",
        "lm/tokenizer_config.json": model / "lm" / "tokenizer_config.json",
        "lm/tokenizer.json": model / "lm" / "tokenizer.json",
        "cpp/default/voices.json": args.prepared_voices or decoder_dir / "voices.json",
        "LICENSE.md": model / "LICENSE.md",
        "speaker/LICENSE": model / "speaker" / "LICENSE",
        "native/LICENSE": s3_license,
    }
    for path in [source, s3gen, speaker, args.converter, *sidecars.values()]:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kitten-gguf-", dir=args.output.parent) as directory:
        root = Path(directory)
        required = set(next(option["values"] for option in spec["options"]["request"] if option["name"] == "voice_id"))
        sidecars["cpp/default/voices.json"] = prepare_voices(args, model, s3gen, root, required)
        # Keep preparation manifests (including local WAV paths) outside the
        # converter root: only publish the explicit runtime sidecars below.
        package_root = root / "package"
        package_root.mkdir()
        prepared_spec = package_root / "kitten_tts2.json"
        prepared_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        notice = package_root / "NOTICE"
        notice.write_text(
            "This Stellon Labs Model is licensed under the Stellon Labs Community License, "
            "Copyright © Stellon Labs Inc. All Rights Reserved\n"
            "Powered by Stellon Labs.\n"
            "Packaging changes: language-model tensors converted to audio.cpp GGUF; "
            "native S3 meanflow and speaker encoder tensors included with prepared voices.\n"
            "S3 decoder: ResembleAI/chatterbox-turbo; speaker encoder: pyannote/embedding.\n"
            "See native/LICENSE and speaker/LICENSE for their respective terms.\n",
            encoding="utf-8",
        )
        sidecars["NOTICE"] = notice
        command = [str(args.converter.resolve()), "--input", f"language_model={source}",
                   "--input", f"s3gen={s3gen}", "--input", f"speaker={speaker}",
                   "--output", str(args.output.resolve()), "--type", args.type,
                   "--keep-type", "language_model/model.embed_tokens.weight=f16",
                   "--keep-type", "language_model/spk_proj.*=orig",
                   # These checkpoints are F32, with scalar I64 batch counters.
                   # Preserve both; forcing scalar counters to F32 is unsupported.
                   "--keep-type", "s3gen/*=orig", "--keep-type", "speaker/*=orig",
                   "--family", "kitten_tts2", "--model-spec", str(prepared_spec), "--root", str(package_root)]
        for destination, path in sidecars.items():
            command += ["--sidecar", f"{path}={destination}"]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
