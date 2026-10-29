#!/usr/bin/env python3
"""Prepare ConwayResearch/Underdog-Bark-3.0-Pro for the native Parakeet runtime.

The original BF16 checkpoint is never modified. Conversion cancels the existing
loader's input projection scaling and variance-floor convention; both are the
same compatibility transforms already used by the Phonon-2 converter.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from convert_phonon2 import compensate_projection, compensate_batch_norm

SOURCE_REPO = "ConwayResearch/Underdog-Bark-3.0-Pro"
SOURCE_REVISION = "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4"
SOURCE_SHA256 = "6f84b33e63ebd1851e910e8b2fe299d29f5e0efc6f57e84d92d2f79af464e0e5"
VARIANT = "underdog_bark3_pro"


def validate_config(config):
    enc = config.get("encoder_config", {})
    expected = {"hidden_size": 1024, "num_hidden_layers": 24,
                "num_attention_heads": 8, "subsampling_factor": 8,
                "conv_kernel_size": 9, "scale_input": False}
    if config.get("model_type") != "parakeet_tdt" or config.get("vocab_size") != 8193:
        raise ValueError("expected the pinned Bark 3.0 Pro Parakeet-TDT architecture")
    if any(enc.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported Bark encoder configuration")
    if config.get("durations") != [0, 1, 2, 3, 4] or config.get("num_decoder_layers") != 2:
        raise ValueError("unsupported Bark decoder configuration")


def package_spec(repository):
    spec = json.loads((repository / "model_specs/parakeet_tdt.json").read_text(encoding="utf-8"))
    spec.update(display_name="Underdog Bark 3.0 Pro", status="experimental", languages=["en"],
                description="Conway Research Bark 3.0 Pro English Parakeet-TDT ASR; includes the reference 250 ms input prefix.",
                packages=[])
    spec.pop("package_defaults", None)
    spec["ui"].pop("recommended_package", None)
    spec["ui"]["docs"] = ["docs/models/underdog_bark3.md"]
    spec["capabilities"]["asr"] = ["word_timestamps", "partial_results"]
    spec["sources"] = [s for s in spec["sources"] if s["format"] == "gguf"]
    spec["options"]["request"] = [o for o in spec["options"]["request"]
                                   if not o["name"].startswith("hotword") and o["name"] != "audio_chunk_duration_sec"]
    for option in spec["options"]["request"]:
        if option["name"] == "audio_chunk_mode":
            option["values"] = ["auto", "fixed", "none"]
            option["description"] = "Bark offline scheduling: auto follows offline_mode; fixed uses bounded windows; none uses full-context. VAD is not admitted for this variant."
    spec["options"]["session"] = [o for o in spec["options"]["session"] if o["name"] != "vad_model_path"]
    for option in spec["options"]["session"]:
        if option["name"] == "offline_mode":
            option["default"] = "full_context"
            option["description"] = "Bark defaults to the publisher's whole-utterance context. Explicit auto/long_form uses bounded windows and can change the transcript."
    return spec


def prepare(source, staging):
    import torch
    from safetensors.torch import load_file
    from safetensors.numpy import save_file

    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_config(cfg)
    checksum = hashlib.sha256()
    with (source / "model.safetensors").open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(16 * 1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != SOURCE_SHA256:
        raise ValueError("checkpoint differs from the pinned revision; update provenance and validate the new source first")
    if (staging / "model.safetensors").exists():
        raise FileExistsError("staging already contains weights; use a new staging directory or --pack-only")
    staging.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    raw = load_file(source / "model.safetensors", device="cpu")
    tensors = {name: np.ascontiguousarray(value.float().numpy()) for name, value in raw.items()
               if not name.endswith(".num_batches_tracked")}
    if len(tensors) != 699:
        raise ValueError("unexpected inference tensor inventory")
    records = []
    for name in sorted(tensors):
        values = tensors[name]
        if not np.isfinite(values).all():
            raise ValueError("non-finite source tensor: " + name)
        values = compensate_projection(name, values)
        values = compensate_batch_norm(name, values, tensors)
        tensors[name] = np.ascontiguousarray(values, dtype=np.float32)
        records.append({"name": name, "shape": list(values.shape),
                        "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    cfg["variant"] = VARIANT
    (staging / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for name in ["processor_config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "NOTICE", "README.md"]:
        shutil.copyfile(source / name, staging / name)
    # The same pinned repository supplies its full CC-BY-4.0 license here.
    shutil.copyfile(source / "coreml/LICENSE", staging / "LICENSE")
    provenance = {"source": SOURCE_REPO, "revision": SOURCE_REVISION, "model_sha256": SOURCE_SHA256, "variant": VARIANT,
                  "input_prefix_samples": 4000, "sample_rate": 16000,
                  "subsampling_projection_scale": 1 / 32,
                  "batch_norm_gamma_correction": "sqrt((max(var,1e-5)+1e-5)/(var+1e-5))",
                  "dropped_training_counters": 24, "tensors": records}
    (staging / "conversion.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    save_file(tensors, staging / "model.safetensors")
    repository = Path(__file__).resolve().parents[2]
    (staging / "model_spec.json").write_text(json.dumps(package_spec(repository), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--gguf-tool", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--type", choices=["f32", "bf16", "q8_0"], default="q8_0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing generated GGUF")
    args = parser.parse_args()
    if args.prepare_only and args.pack_only:
        parser.error("--prepare-only and --pack-only are mutually exclusive")
    if not args.pack_only:
        prepare(args.source.resolve(), args.staging.resolve())
    if args.prepare_only:
        return
    if not args.gguf_tool or not args.output:
        parser.error("packing requires --gguf-tool and --output")
    stage = args.staging.resolve()
    provenance = json.loads((stage / "conversion.json").read_text(encoding="utf-8"))
    if (provenance.get("source") != SOURCE_REPO or provenance.get("revision") != SOURCE_REVISION or
            provenance.get("model_sha256") != SOURCE_SHA256 or provenance.get("variant") != VARIANT):
        raise ValueError("staging does not belong to this model/revision")
    command = [str(args.gguf_tool.resolve()), "--input", str(stage / "model.safetensors"),
               "--root", str(stage), "--family", "parakeet_tdt", "--model-spec", str(stage / "model_spec.json"),
               "--type", args.type, "--keep-type", "*conv.norm.weight=f32", "--output", str(args.output.resolve())]
    if args.overwrite:
        command.append("--overwrite")
    for name in ["config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json", "NOTICE", "LICENSE", "README.md", "conversion.json"]:
        command += ["--sidecar", f"{stage / name}={name}"]
    import os
    subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


if __name__ == "__main__":
    main()
