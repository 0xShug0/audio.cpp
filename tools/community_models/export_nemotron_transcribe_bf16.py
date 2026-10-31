#!/usr/bin/env python3
"""Export one reusable BF16 decoder and its paired F16/F32 speech encoder.

Uses the pinned, unmodified NVIDIA/llama.cpp converter. Large source tensors are
hardlinked into an isolated staging directory, never copied or requantized.
Keep the completed BF16 decoder for quantize_nemotron_transcribe.py.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from convert_nemotron_transcribe import REFERENCE_REVISION, LLAMA_REVISION


def publish(temporary: Path, output: Path) -> None:
    if os.name == "nt":
        os.rename(temporary, output)
    else:
        os.link(temporary, output)
        temporary.unlink()


def check_encoder(path: Path, identity: str) -> None:
    import gguf
    reader = gguf.GGUFReader(str(path))
    try:
        if reader.fields["general.architecture"].contents() != "pee" or \
                reader.fields["nemo_speech.checkpoint_id"].contents() != identity or \
                not base64.b64decode(reader.fields["ctc.tokenizer.spm_model"].contents(), validate=True) or \
                not reader.tensors:
            raise ValueError("encoder is not the checkpoint's paired speech/CTC component")
    finally:
        reader.data._mmap.close()


def export_encoder(source: Path, salm, output: Path, checkpoint_name: str, identity: str, work: Path) -> Path:
    speech = output.parent / salm.encoder_file_name(checkpoint_name)
    if speech.exists():
        check_encoder(speech, identity)
        return speech
    temporary = speech.with_name(speech.name + ".exporting")
    if temporary.exists() or temporary.with_name(temporary.name + ".writing").exists():
        raise FileExistsError("uninspected encoder export exists")
    from conversion import pee, ctc_timestamp
    from conversion.salm_encoder import write_encoder_file
    ctc_source = salm.ctc_checkpoint(source)
    if ctc_source is None:
        raise ValueError("the full source checkpoint must include its CTC timestamp head")
    encoder_work = work / "encoder"
    encoder_work.mkdir(parents=True, exist_ok=False)
    pee.convert(source, encoder_work / "speech.gguf", "fp16")
    ctc_timestamp.convert(ctc_source, encoder_work / "ctc.gguf", checkpoint_name)
    write_encoder_file(encoder_work / "speech.gguf", encoder_work / "ctc.gguf", temporary,
        name=checkpoint_name + " Encoder F16", checkpoint_id=identity)
    check_encoder(temporary, identity)
    publish(temporary, speech)
    return speech


def export(source: Path, reference: Path, output: Path,
           checkpoint_name: str = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", encoder_only: bool = False) -> dict:
    source, reference, output = (p.resolve() for p in (source, reference, output))
    if not (source / "model.safetensors").is_file():
        raise ValueError("the full source model.safetensors is required")
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint-name must be a simple filename stem")
    for directory, expected in ((reference, REFERENCE_REVISION), (reference / "llama.cpp", LLAMA_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected or subprocess.check_output(
                ["git", "-C", str(directory), "diff", "HEAD", "--name-only"], text=True).strip():
            raise ValueError(f"reference must be clean at {expected}: {directory}")
    work = output.with_name(output.name + (".encoder-export-work" if encoder_only else ".export-work"))
    temporary = output.with_name(output.name + ".exporting")
    if work.exists() or (not encoder_only and any(p.exists() for p in (output, temporary))):
        raise ValueError("output or previous export work exists; inspect it before another export")
    if output.is_relative_to(source) or source.is_relative_to(work):
        raise ValueError("export output/work must be separate from the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=False)
    sys.path.insert(0, str(reference))
    from conversion import salm
    identity = salm.checkpoint_id(source)
    speech = export_encoder(source, salm, output, checkpoint_name, identity, work)
    if encoder_only:
        return {"checkpoint_id": identity, "encoder": speech.name, "encoder_bytes": speech.stat().st_size,
            "encoder_only": True, "reference_revision": REFERENCE_REVISION}
    staging = work / "hf"
    salm.stage(source, staging, weights=False)
    for path in sorted(source.glob("*.safetensors")) + sorted(source.glob("*.safetensors.index.json")):
        try:
            os.link(path, staging / path.name)
        except OSError as error:
            raise RuntimeError("source and output must share a filesystem for hardlink staging") from error
    command = [sys.executable, str(reference / "conversion/salm_decoder_hf.py"), str(reference / "llama.cpp"),
        str(staging), "--outfile", str(temporary), "--outtype", "bf16", "--model-name", checkpoint_name + " BF16"]
    for key in salm.decoder_keys(checkpoint_name, identity):
        command += ["--nemo-speech-key", key]
    subprocess.run(command, check=True)
    # Inspect only metadata, without loading the full model into RAM.
    import gguf
    reader = gguf.GGUFReader(str(temporary))
    try:
        if reader.fields["general.architecture"].contents() != "nemotron_h_moe" or \
                reader.fields["nemo_speech.checkpoint_id"].contents() != identity:
            raise RuntimeError("export architecture/checkpoint identity is inconsistent")
        allowed = {gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
        if not reader.tensors or any(t.tensor_type not in allowed for t in reader.tensors):
            raise RuntimeError("export must contain floating-point source tensors only")
        if not any(t.name.endswith("ffn_up_exps.weight") for t in reader.tensors):
            raise RuntimeError("export contains no routed expert banks")
        tensor_count = len(reader.tensors)
        # Full tensors are checked later by the quantizer and runtime smoke tests.
    finally:
        reader.data._mmap.close()
    publish(temporary, output)
    return {"checkpoint_id": identity, "decoder": output.name, "decoder_bytes": output.stat().st_size,
        "tensor_count": tensor_count, "reference_revision": REFERENCE_REVISION,
        "encoder": speech.name, "encoder_bytes": speech.stat().st_size,
        "llama_revision": LLAMA_REVISION, "staging_retained": True,
        "source_weight_policy": "full source BF16, no quantized input"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    parser.add_argument("--encoder-only", action="store_true", help="Export/check only the paired F16 speech + F32 CTC GGUF")
    args = parser.parse_args()
    receipt = args.output.with_suffix(".encoder-export.json" if args.encoder_only else ".export.json")
    if receipt.exists():
        raise FileExistsError("an export receipt already exists")
    result = export(args.source, args.reference, args.output, args.checkpoint_name, args.encoder_only)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
