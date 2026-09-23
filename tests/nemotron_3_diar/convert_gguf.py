#!/usr/bin/env python3
"""Convert NVIDIA Nemotron 3 Diarization from its original NeMo checkpoint.

The converter reads the official ``.nemo`` archive directly, stages canonical
Safetensors plus the model/frontend configuration, and can package a
self-contained GGUF with audio.cpp's converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REPO = "nvidia/Nemotron-3-Diarization"
SOURCE_REVISION = "723e19c601d99b7e58fba6a14e32153e0afe48d9"
SOURCE_LICENSE = "openmdw-1.1"
SOURCE_LICENSE_URL = "https://openmdw.ai/license/1-1/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import yaml

    if not source.is_file():
        raise SystemExit(f"checkpoint not found: {source}")
    with tarfile.open(source, mode="r:*") as archive:
        config_member = archive.extractfile("model_config.yaml")
        weights_member = archive.extractfile("model_weights.ckpt")
        if config_member is None or weights_member is None:
            raise SystemExit(f"NeMo archive is missing model_config.yaml or model_weights.ckpt: {source}")
        config = yaml.safe_load(config_member.read().decode("utf-8"))
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as checkpoint:
            while block := weights_member.read(16 * 1024 * 1024):
                checkpoint.write(block)
            checkpoint.flush()
            loaded = torch.load(checkpoint.name, map_location="cpu", weights_only=False)
    state = loaded.get("state_dict", loaded)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint did not contain a non-empty state dictionary")
    return config, state


def assert_model_shape(config: dict[str, Any]) -> None:
    encoder = config["encoder"]
    modules = config["sortformer_modules"]
    checks = {
        "sample_rate": (config["sample_rate"], 16000),
        "max_num_of_spks": (config["max_num_of_spks"], 8),
        "encoder.n_layers": (encoder["n_layers"], 31),
        "encoder.d_model": (encoder["d_model"], 512),
        "encoder.n_heads": (encoder["n_heads"], 8),
        "encoder.subsampling": (encoder["subsampling"], "feature_stacking"),
        "encoder.subsampling_factor": (encoder["subsampling_factor"], 8),
        "encoder.self_attention_model": (encoder["self_attention_model"], "rope"),
        "sortformer_modules.num_spks": (modules["num_spks"], 8),
        "sortformer_modules.tf_d_model": (modules["tf_d_model"], 192),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("not the expected Nemotron 3 Diarization architecture: " + "; ".join(mismatches))


def processor_config(config: dict[str, Any]) -> dict[str, Any]:
    pre = config["preprocessor"]
    return {
        "feature_extractor": {
            "feature_size": int(pre["features"]),
            "sampling_rate": int(pre["sample_rate"]),
            "n_fft": int(pre["n_fft"]),
            "win_length": int(round(float(pre["window_size"]) * int(pre["sample_rate"]))),
            "hop_length": int(round(float(pre["window_stride"]) * int(pre["sample_rate"]))),
            "window": str(pre["window"]),
            "dither": float(pre["dither"]),
            "normalize": str(pre["normalize"]),
            "preemphasis": float(pre.get("preemph", 0.97)),
            "return_attention_mask": True,
        }
    }


def model_config(config: dict[str, Any], source_hash: str) -> dict[str, Any]:
    encoder = config["encoder"]
    modules = config["sortformer_modules"]
    pre = config["preprocessor"]
    return {
        "model_type": "nemotron_3_diar",
        "architectures": ["Nemotron3Diarization"],
        "audiocpp_family": "nemotron_3_diar",
        "source_model": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "sample_rate": int(config["sample_rate"]),
        "num_speakers": int(config["max_num_of_spks"]),
        "high_resolution": bool(config["high_resolution"]),
        "output_subsampling_factor": int(config["output_subsampling_factor"]),
        "frontend": {
            "num_mels": int(pre["features"]),
            "n_fft": int(pre["n_fft"]),
            "win_length": int(round(float(pre["window_size"]) * int(pre["sample_rate"]))),
            "hop_length": int(round(float(pre["window_stride"]) * int(pre["sample_rate"]))),
            "window": str(pre["window"]),
            "dither": float(pre["dither"]),
            "normalize": str(pre["normalize"]),
            "preemphasis": float(pre.get("preemph", 0.97)),
        },
        "encoder": {
            "feat_in": int(encoder["feat_in"]),
            "hidden_size": int(encoder["d_model"]),
            "intermediate_size": int(float(encoder["ff_expansion"]) * int(encoder["d_model"])),
            "num_attention_heads": int(encoder["n_heads"]),
            "num_hidden_layers": int(encoder["n_layers"]),
            "subsampling_factor": int(encoder["subsampling_factor"]),
            "pre_block_norm": bool(encoder["pre_block_norm"]),
            "qkv_bias": bool(encoder["qkv_bias"]),
            "qk_norm": bool(encoder["qk_norm"]),
            "rope_theta": float(encoder.get("rope_base", 10000.0)),
            "rotary_fraction": float(encoder.get("rotary_fraction", 1.0)),
            "layer_norm_eps": 1.0e-5,
        },
        "head": {
            "hidden_size": int(modules["tf_d_model"]),
            "upsample_factor": int(encoder["subsampling_factor"]),
            "upsample_kernel_size": 3,
        },
        "streaming": {
            "spkcache_len": int(modules["spkcache_len"]),
            "fifo_len": int(modules["fifo_len"]),
            "chunk_len": int(modules["chunk_len"]),
            "chunk_left_context": int(modules["chunk_left_context"]),
            "chunk_right_context": int(modules["chunk_right_context"]),
            "spkcache_update_period": int(modules["spkcache_update_period"]),
            "spkcache_sil_frames_per_spk": int(modules["spkcache_sil_frames_per_spk"]),
            "pred_score_threshold": float(modules["pred_score_threshold"]),
            "scores_boost_latest": float(modules["scores_boost_latest"]),
            "sil_threshold": float(modules["sil_threshold"]),
            "strong_boost_rate": float(modules["strong_boost_rate"]),
            "weak_boost_rate": float(modules["weak_boost_rate"]),
            "min_pos_scores_rate": float(modules["min_pos_scores_rate"]),
            "max_index": int(modules["max_index"]),
        },
    }


def staging_spec() -> dict[str, Any]:
    return json.loads(Path("model_specs/nemotron_3_diar.json").read_text(encoding="utf-8"))


def convert(checkpoint: Path, output_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    config, state = load_checkpoint(checkpoint)
    assert_model_shape(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(checkpoint)

    tensors: dict[str, torch.Tensor] = {}
    manifest: list[dict[str, Any]] = []
    for source_name, value in sorted(state.items()):
        if source_name == "preprocessor.featurizer.window":
            continue
        if source_name == "preprocessor.featurizer.fb":
            value = value.squeeze(0)
            destination = "preprocessor.fb"
        else:
            destination = source_name
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected tensor for {source_name}, got {type(value).__name__}")
        if value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"unsupported tensor dtype for {source_name}: {value.dtype}")
        tensor = value.detach().cpu().contiguous()
        tensors[destination] = tensor
        manifest.append({
            "source": source_name,
            "destination": destination,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        })

    save_file(tensors, str(output_dir / "model.safetensors"), metadata={
        "format": "nemotron-3-diarization-audiocpp-staging",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    })
    (output_dir / "config.json").write_text(
        json.dumps(model_config(config, source_hash), indent=2) + "\n", encoding="utf-8")
    (output_dir / "processor_config.json").write_text(
        json.dumps(processor_config(config), indent=2) + "\n", encoding="utf-8")
    (output_dir / "model_spec.json").write_text(
        json.dumps(staging_spec(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "tensor_manifest.json").write_text(json.dumps({
        "source": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "source_tensor_count": len(state),
        "emitted_tensor_count": len(tensors),
        "tensors": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_license_url": SOURCE_LICENSE_URL,
        "source_sha256": source_hash,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"source tensors: {len(state)}")
    print(f"emitted tensors: {len(tensors)}")
    print(f"wrote staging package: {output_dir}")


def package(args: argparse.Namespace) -> None:
    if args.converter is None or args.gguf_output is None:
        return
    command = [
        str(args.converter.resolve()),
        "--input", str((args.output_dir / "model.safetensors").resolve()),
        "--root", str(args.output_dir.resolve()),
        "--family", "nemotron_3_diar",
        "--model-spec", str((args.output_dir / "model_spec.json").resolve()),
        "--type", args.type,
        "--keep-type", "preprocessor.fb=f32",
        "--output", str(args.gguf_output.resolve()),
        "--overwrite",
    ]
    args.gguf_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--gguf-output", type=Path)
    parser.add_argument("--type", choices=["orig", "f16", "q8_0"], default="orig")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    convert(args.checkpoint.resolve(), args.output_dir)
    package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
