#!/usr/bin/env python3
"""Package the complete native MuLaCover pipeline as one FP16 or Q8_0 GGUF.

Run in conda qwen3-tts. --model-dir contains MuLaCover/, HeartCodec-oss/,
Qwen3-Embedding-0.6B/, and SymbolicTranscriptor/. Prepare raw transcription
checkpoints with mulacover/convert_transcriber.py (--chord-root for five folds).
The existing audiocpp_gguf primitive owns quantization and package writing.

Every package includes generator, style encoder, codec, YourMT3, five chord
folds, and the canonical chord vocabulary. Runtime uses native package storage;
Q8 mode quantizes eligible matrices across components, retaining FP16 lookup
tables/convolutions and FP32 vector parameters.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors import safe_open


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = REPOSITORY / "model_specs" / "mulacover.json"
VOCABULARY = REPOSITORY / "tools/models/mulacover/chord_vocabulary.txt"
NOTICES = REPOSITORY / "tests/mulacover"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path,
                        default=REPOSITORY / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--log-file", type=Path, required=True,
                        help="Save converter stdout/stderr; --log is always enabled")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.model_dir.resolve()
    converter = args.converter.resolve()
    if not converter.is_file():
        parser.error(f"converter missing: {converter}")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}")
    spec = json.loads(SPEC.read_text())
    source = next(item for item in spec["sources"] if item["format"] == "gguf")
    stage_files = {}
    tensor_inputs = {
        "mula_weights": "MuLaCover/model.safetensors.index.json",
        "codec_weights": "HeartCodec-oss/model.safetensors.index.json",
        "qwen_weights": "Qwen3-Embedding-0.6B/model.safetensors",
        "transcriber_weights": "SymbolicTranscriptor/yourmt3/model.safetensors",
        **{f"chord_weights_{fold}": f"SymbolicTranscriptor/chord/fold{fold}.safetensors"
           for fold in range(5)},
    }
    for name in ("LICENSE.model", "LICENSE.apache-2.0", "LICENSE.yourmt3", "LICENSE.chordnet",
                 "LICENSE.librosa", "LICENSE.rotary-embedding", "NOTICE"):
        stage_files[f"licenses/{name}"] = NOTICES / name
    for name, relative in tensor_inputs.items():
        path = root / relative
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
        stage_files[relative] = path
        if path.name.endswith(".index.json"):
            index = json.loads(path.read_text())
            for shard in sorted(set(index["weight_map"].values())):
                shard_path = path.parent / shard
                if not shard_path.is_file():
                    parser.error(f"missing checkpoint shard: {shard_path}")
                stage_files[(Path(relative).parent / shard).as_posix()] = shard_path
    for name, reference in source["files"].items():
        relative = reference.removeprefix("model:")
        path = VOCABULARY if name == "chord_vocabulary" else root / relative
        if not path.is_file():
            parser.error(f"missing resource: {path}")
        stage_files[relative] = path

    output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    # Copy instead of symlinking: the native writer records paths relative to
    # --root after canonicalization. A symlink (notably HeartCodec) would leak a
    # machine-specific path and root scanning would admit raw training files.
    with tempfile.TemporaryDirectory(prefix="mulacover-package-", dir=output.parent) as temporary:
        staged = Path(temporary) / "MuLaCover"
        for relative, original in stage_files.items():
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        command = [str(converter), "--family", "mulacover", "--model-spec", str(SPEC),
                   "--root", str(staged), "--type", args.type, "--output", str(output), "--log"]
        for namespace, relative in tensor_inputs.items():
            command += ["--input", f"{namespace}={staged / relative}"]
        # Sidecars are already at canonical relative paths under the clean root.
        # Do not embed the raw .ckpt/.sdict files or source safetensors metadata.
        for namespace, relative in tensor_inputs.items():
            path = staged / relative
            if path.name.endswith(".index.json"):
                weight_map = json.loads(path.read_text())["weight_map"]
                shards = [path.parent / name for name in sorted(set(weight_map.values()))]
            else:
                weight_map = None
                shards = [path]
            for shard in shards:
                with safe_open(shard, framework="numpy") as checkpoint:
                    for name in sorted(checkpoint.keys()):
                        if weight_map is not None and weight_map.get(name) != shard.name:
                            continue
                        tensor = checkpoint.get_slice(name)
                        shape, dtype = tensor.get_shape(), tensor.get_dtype()
                        storage = "orig"
                        if dtype in ("F32", "F16", "BF16"):
                            storage = "f32" if len(shape) <= 1 else "f16"
                            if (args.type == "q8_0"
                                    and len(shape) == 2 and name.endswith(".weight")
                                    and "embed" not in name.lower()
                                    and "codebook" not in name.lower()
                                    and "additive_pe" not in name.lower()
                                    and shape[-1] % 32 == 0):
                                storage = "q8_0"
                        command += ["--keep-type", f"{namespace}/{name}={storage}"]
        if args.overwrite:
            command.append("--overwrite")
        with args.log_file.open("w", encoding="utf-8") as log:
            log.write("command=" + repr(command) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
