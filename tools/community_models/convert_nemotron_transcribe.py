#!/usr/bin/env python3
"""Run NVIDIA's pinned SALM conversion without backend patches or Windows symlinks.

Use a full BF16 checkpoint and an explicitly supplied, unpatched llama-quantize.
The output and source must share a filesystem for hardlink staging. This wrapper
does not download, build a quantizer, patch a dependency, or overwrite outputs.
Run prepare_nemotron_transcribe.py afterwards to embed the audio.cpp package.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REFERENCE_REVISION = "66d50d91b75a406becd95fbaf39d2c9c13e95d7c"
LLAMA_REVISION = "bd4f514db14d87fded667787a7a963bfbaa98e89"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Pinned NVIDIA/NeMo-Speech.cpp checkout")
    parser.add_argument("--quantizer", type=Path, required=True, help="Pristine llama-quantize executable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint-name", default="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    args = parser.parse_args()
    source, reference, quantizer, output = (p.resolve() for p in
        (args.source, args.reference, args.quantizer, args.output))
    if args.threads < 1 or not quantizer.is_file() or not (source / "model.safetensors").is_file():
        parser.error("source model.safetensors, explicit quantizer and positive threads are required")
    if Path(args.checkpoint_name).name != args.checkpoint_name or not args.checkpoint_name:
        parser.error("checkpoint-name must be a simple filename stem")
    for directory, expected in ((reference, REFERENCE_REVISION), (reference / "llama.cpp", LLAMA_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"unexpected reference revision in {directory}: {actual}, expected {expected}")
        if subprocess.check_output(["git", "-C", str(directory), "diff", "HEAD", "--name-only"], text=True).strip():
            raise ValueError(f"reference source has local changes: {directory}")
    work = output.with_name(output.name + ".work")
    if output.exists() or work.exists() or output.with_name(output.name + ".quantizing").exists():
        raise ValueError("output/work files already exist; inspect them before another conversion")
    if source == work or source.is_relative_to(work) or work.is_relative_to(source):
        raise ValueError("conversion work must be separate from the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(reference))
    from conversion import salm
    encoder = output.with_name(salm.encoder_file_name(args.checkpoint_name))
    if encoder.exists():
        raise ValueError(f"encoder output already exists: {encoder}")
    original_stage = salm.stage

    def stage(checkpoint: Path, staging: Path, weights: bool = True) -> None:
        if checkpoint.resolve() != source or staging.resolve() != (work / "hf").resolve() or staging.exists():
            raise ValueError("unexpected or previously used staging directory")
        # The original helper receives weights=False, so it only writes the
        # small tokenizer/config files. Large source weights are never copied.
        original_stage(checkpoint, staging, weights=False)
        if weights:
            for path in sorted(checkpoint.glob("*.safetensors")) + sorted(checkpoint.glob("*.safetensors.index.json")):
                try:
                    os.link(path, staging / path.name)
                except OSError as error:
                    raise RuntimeError("hardlink staging failed; source and output must share a filesystem") from error

    def quantize(bf16: Path, outfile: Path, profile: str, executable: Path, dense: str = "bf16") -> None:
        if outfile.resolve() != output or executable.resolve() != quantizer:
            raise ValueError("unexpected quantization destination or executable")
        temporary = output.with_name(output.name + ".quantizing")
        if temporary.exists():
            raise ValueError("quantization temporary already exists")
        command = [str(executable)]
        for rule in salm.tensor_rules(profile, dense):
            command += ["--tensor-type", rule]
        command += [str(bf16), str(temporary), "Q8_0", str(args.threads)]
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("quantizer produced no output")
        temporary.replace(output)

    salm.stage = stage
    salm.quantize_decoder = quantize
    salm.convert(source, output, profile="iq4_nl", llama_cpp=reference / "llama.cpp",
                 quantizer=quantizer, auto_build_quantizer=False, encoder="f16",
                 checkpoint_name=args.checkpoint_name)


if __name__ == "__main__":
    main()
