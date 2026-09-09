#!/usr/bin/env python3
"""Stage NVIDIA Sortformer v2.1 weights for audio.cpp.

This converter reads the official NVIDIA ``.nemo`` checkpoint, not a GGUF
round-trip, and emits a v2-specific audio.cpp staging directory:

    model.safetensors       canonical ``enc.*``, ``tf.*``, and ``diar.*`` names
    config.json             versioned v2.1 architecture and streaming config
    processor_config.json   exact frontend parameters
    model_spec.json         temporary self-contained audio.cpp package spec
    tensor_manifest.json    source/destination names, shapes, and dtypes
    provenance.json         source revision, hashes, and license information

The canonical tensor names intentionally match the validated
handy-computer/transcribe.cpp v2.1 port. They must not be renamed to the v1
``fc_encoder.*`` / ``tf_encoder.*`` / ``sortformer_modules.*`` namespaces.
The v2 audio.cpp loader will consume this contract after the graph adapter is
implemented.

The mapping tables are adapted from handy-computer/transcribe.cpp's
``scripts/convert-sortformer.py`` (MIT License; Copyright (c) 2026 The
transcribe.cpp authors). See ``SORTFORMER_THIRD_PARTY_NOTICES.md``.

Dependencies are intentionally small and work without installing NeMo:

    uv run --python 3.12 --with torch --with pyyaml --with safetensors \
      tools/community_models/convert_sortformer_v2_1.py \
      --checkpoint path/to/diar_streaming_sortformer_4spk-v2.1.nemo \
      --output-dir models/diar_streaming_sortformer_4spk-v2.1-audiocpp

To package the staged F32 tensors after building ``audiocpp_gguf``:

    uv run --python 3.12 --with torch --with pyyaml --with safetensors \
      tools/community_models/convert_sortformer_v2_1.py ... \
      --converter build/debug/bin/audiocpp_gguf \
      --gguf-output models/Sortformer-Diar-4spk-v2.1-GGUF/sortformer-v2.1-f32.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
SOURCE_REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"
SOURCE_LICENSE = "nvidia-open-model-license"
SOURCE_LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"

PRE_ENCODE_TABLE = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("encoder.pre_encode.conv.0.bias", "enc.pre_encode.conv.0.bias"),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("encoder.pre_encode.conv.2.bias", "enc.pre_encode.conv.2.bias"),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight"),
    ("encoder.pre_encode.conv.3.bias", "enc.pre_encode.conv.3.bias"),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight"),
    ("encoder.pre_encode.conv.5.bias", "enc.pre_encode.conv.5.bias"),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight"),
    ("encoder.pre_encode.conv.6.bias", "enc.pre_encode.conv.6.bias"),
    ("encoder.pre_encode.out.weight", "enc.pre_encode.out.weight"),
    ("encoder.pre_encode.out.bias", "enc.pre_encode.out.bias"),
]

CONFORMER_BLOCK_TABLE = [
    ("norm_feed_forward1.weight", "norm_ff1.weight"),
    ("norm_feed_forward1.bias", "norm_ff1.bias"),
    ("feed_forward1.linear1.weight", "ff1.linear1.weight"),
    ("feed_forward1.linear1.bias", "ff1.linear1.bias"),
    ("feed_forward1.linear2.weight", "ff1.linear2.weight"),
    ("feed_forward1.linear2.bias", "ff1.linear2.bias"),
    ("norm_self_att.weight", "norm_attn.weight"),
    ("norm_self_att.bias", "norm_attn.bias"),
    ("self_attn.linear_q.weight", "attn.linear_q.weight"),
    ("self_attn.linear_q.bias", "attn.linear_q.bias"),
    ("self_attn.linear_k.weight", "attn.linear_k.weight"),
    ("self_attn.linear_k.bias", "attn.linear_k.bias"),
    ("self_attn.linear_v.weight", "attn.linear_v.weight"),
    ("self_attn.linear_v.bias", "attn.linear_v.bias"),
    ("self_attn.linear_out.weight", "attn.linear_out.weight"),
    ("self_attn.linear_out.bias", "attn.linear_out.bias"),
    ("self_attn.linear_pos.weight", "attn.linear_pos.weight"),
    ("self_attn.pos_bias_u", "attn.pos_bias_u"),
    ("self_attn.pos_bias_v", "attn.pos_bias_v"),
    ("norm_conv.weight", "norm_conv.weight"),
    ("norm_conv.bias", "norm_conv.bias"),
    ("conv.pointwise_conv1.weight", "conv.pointwise1.weight"),
    ("conv.pointwise_conv1.bias", "conv.pointwise1.bias"),
    ("conv.depthwise_conv.weight", "conv.depthwise.weight"),
    ("conv.depthwise_conv.bias", "conv.depthwise.bias"),
    ("conv.batch_norm.weight", "conv.bn.weight"),
    ("conv.batch_norm.bias", "conv.bn.bias"),
    ("conv.batch_norm.running_mean", "conv.bn.running_mean"),
    ("conv.batch_norm.running_var", "conv.bn.running_var"),
    ("conv.pointwise_conv2.weight", "conv.pointwise2.weight"),
    ("conv.pointwise_conv2.bias", "conv.pointwise2.bias"),
    ("norm_feed_forward2.weight", "norm_ff2.weight"),
    ("norm_feed_forward2.bias", "norm_ff2.bias"),
    ("feed_forward2.linear1.weight", "ff2.linear1.weight"),
    ("feed_forward2.linear1.bias", "ff2.linear1.bias"),
    ("feed_forward2.linear2.weight", "ff2.linear2.weight"),
    ("feed_forward2.linear2.bias", "ff2.linear2.bias"),
    ("norm_out.weight", "norm_out.weight"),
    ("norm_out.bias", "norm_out.bias"),
]

TRANSFORMER_BLOCK_TABLE = [
    ("layer_norm_1.weight", "norm_1.weight"),
    ("layer_norm_1.bias", "norm_1.bias"),
    ("first_sub_layer.query_net.weight", "attn.q.weight"),
    ("first_sub_layer.query_net.bias", "attn.q.bias"),
    ("first_sub_layer.key_net.weight", "attn.k.weight"),
    ("first_sub_layer.key_net.bias", "attn.k.bias"),
    ("first_sub_layer.value_net.weight", "attn.v.weight"),
    ("first_sub_layer.value_net.bias", "attn.v.bias"),
    ("first_sub_layer.out_projection.weight", "attn.out.weight"),
    ("first_sub_layer.out_projection.bias", "attn.out.bias"),
    ("layer_norm_2.weight", "norm_2.weight"),
    ("layer_norm_2.bias", "norm_2.bias"),
    ("second_sub_layer.dense_in.weight", "ff.in.weight"),
    ("second_sub_layer.dense_in.bias", "ff.in.bias"),
    ("second_sub_layer.dense_out.weight", "ff.out.weight"),
    ("second_sub_layer.dense_out.bias", "ff.out.bias"),
]

HEAD_TABLE = [
    ("sortformer_modules.encoder_proj.weight", "diar.encoder_proj.weight"),
    ("sortformer_modules.encoder_proj.bias", "diar.encoder_proj.bias"),
    ("sortformer_modules.first_hidden_to_hidden.weight", "diar.fc1.weight"),
    ("sortformer_modules.first_hidden_to_hidden.bias", "diar.fc1.bias"),
    ("sortformer_modules.hidden_to_spks.weight", "diar.spk_head.weight"),
    ("sortformer_modules.hidden_to_spks.bias", "diar.spk_head.bias"),
    ("sortformer_modules.single_hidden_to_spks.weight", "diar.single_spk_head.weight"),
    ("sortformer_modules.single_hidden_to_spks.bias", "diar.single_spk_head.bias"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(source: Path) -> tuple[dict[str, Any], dict[str, Any], Path, bool]:
    """Return YAML config, state dict, checkpoint path, and temporary flag."""
    import torch
    import yaml

    if source.is_dir():
        config_path = source / "model_config.yaml"
        checkpoint_path = source / "model_weights.ckpt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise SystemExit(f"checkpoint directory needs model_config.yaml and model_weights.ckpt: {source}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return config, torch.load(checkpoint_path, map_location="cpu", weights_only=False), checkpoint_path, False

    if not source.exists():
        raise SystemExit(f"checkpoint not found: {source}")
    with tarfile.open(source, mode="r:*") as archive:
        config_member = archive.extractfile("model_config.yaml")
        weights_member = archive.extractfile("model_weights.ckpt")
        if config_member is None or weights_member is None:
            raise SystemExit(f"Nemo archive is missing model_config.yaml or model_weights.ckpt: {source}")
        config_text = config_member.read().decode("utf-8")
        weights = weights_member.read()

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as handle:
        handle.write(weights)
        checkpoint_path = Path(handle.name)
    config = yaml.safe_load(config_text)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        checkpoint_path.unlink(missing_ok=True)
    return config, state, checkpoint_path, True


def canonicalize_normalize(value: Any) -> str:
    """Map NeMo's ``NA`` spelling to the explicit no-normalization value."""
    normalized = str(value).strip().lower()
    return "none" if normalized in {"", "na", "none", "null"} else normalized


def assert_model_shape(config: dict[str, Any]) -> None:
    encoder = config["encoder"]
    transformer = config["transformer_encoder"]
    modules = config["sortformer_modules"]
    checks = {
        "sample_rate": (config["sample_rate"], 16000),
        "max_num_of_spks": (config["max_num_of_spks"], 4),
        "encoder.n_layers": (encoder["n_layers"], 17),
        "encoder.d_model": (encoder["d_model"], 512),
        "encoder.feat_in": (encoder["feat_in"], 128),
        "transformer_encoder.num_layers": (transformer["num_layers"], 18),
        "transformer_encoder.hidden_size": (transformer["hidden_size"], 192),
        "transformer_encoder.num_attention_heads": (transformer["num_attention_heads"], 8),
        "sortformer_modules.num_spks": (modules["num_spks"], 4),
    }
    mismatches = [f"{name}={actual!r}, expected {expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError("not the expected Sortformer v2.1 architecture: " + "; ".join(mismatches))


def processor_config(config: dict[str, Any]) -> dict[str, Any]:
    pre = config["preprocessor"]
    return {
        "feature_extractor": {
            "feature_extractor_type": "SortformerFeatureExtractor",
            "feature_size": int(pre["features"]),
            "hop_length": int(round(float(pre["window_stride"]) * int(pre["sample_rate"]))),
            "n_fft": int(pre["n_fft"]),
            "padding_side": "right",
            "padding_value": 0.0,
            "preemphasis": float(pre.get("preemph", 0.97)),
            "processor_class": "SortformerProcessor",
            "return_attention_mask": True,
            "sampling_rate": int(pre["sample_rate"]),
            "win_length": int(round(float(pre["window_size"]) * int(pre["sample_rate"]))),
            "window": str(pre.get("window", "hann")),
            "dither": float(pre.get("dither", 0.0)),
            "normalize": canonicalize_normalize(pre.get("normalize", "NA")),
        },
        "processor_class": "SortformerProcessor",
    }


def model_config(config: dict[str, Any], source_sha256: str | None) -> dict[str, Any]:
    encoder = config["encoder"]
    transformer = config["transformer_encoder"]
    modules = config["sortformer_modules"]
    pre = config["preprocessor"]
    return {
        "model_type": "sortformer_streaming",
        "architectures": ["SortformerStreamingV2_1"],
        "audiocpp_family": "sortformer_diar_v2",
        "audiocpp_variant": "streaming_v2_1",
        "source_model": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_sha256,
        "num_speakers": int(config["max_num_of_spks"]),
        "loss_weights": {
            "pil": float(config["pil_weight"]),
            "ats": float(config["ats_weight"]),
        },
        "frontend": {
            "sample_rate": int(pre["sample_rate"]),
            "num_mels": int(pre["features"]),
            "n_fft": int(pre["n_fft"]),
            "hop_length": int(round(float(pre["window_stride"]) * int(pre["sample_rate"]))),
            "win_length": int(round(float(pre["window_size"]) * int(pre["sample_rate"]))),
            "window": str(pre.get("window", "hann")),
            "normalize": canonicalize_normalize(pre.get("normalize", "NA")),
            "preemphasis": float(pre.get("preemph", 0.97)),
            "dither": float(pre.get("dither", 0.0)),
        },
        "encoder": {
            "n_layers": int(encoder["n_layers"]),
            "d_model": int(encoder["d_model"]),
            "n_heads": int(encoder["n_heads"]),
            "d_ff": int(encoder["d_model"]) * int(encoder["ff_expansion_factor"]),
            "feat_in": int(encoder["feat_in"]),
            "subsampling_factor": int(encoder["subsampling_factor"]),
            "subsampling_channels": int(encoder["subsampling_conv_channels"]),
            "conv_kernel": int(encoder["conv_kernel_size"]),
            "conv_norm_type": str(encoder.get("conv_norm_type", "batch_norm")),
            "pos_emb_max_len": int(encoder.get("pos_emb_max_len", 5000)),
            "attention_bias": True,
            "scale_input": bool(encoder.get("xscaling", True)),
        },
        "transformer": {
            "n_layers": int(transformer["num_layers"]),
            "d_model": int(transformer["hidden_size"]),
            "n_heads": int(transformer["num_attention_heads"]),
            "d_ff": int(transformer["inner_size"]),
            "activation": str(transformer.get("hidden_act", "relu")),
            "pre_ln": bool(transformer.get("pre_ln", False)),
        },
        "streaming": {
            "chunk_len": int(modules["chunk_len"]),
            "chunk_left_context": int(modules["chunk_left_context"]),
            "chunk_right_context": int(modules["chunk_right_context"]),
            "fifo_len": int(modules["fifo_len"]),
            "spkcache_len": int(modules["spkcache_len"]),
            "spkcache_update_period": int(modules["spkcache_update_period"]),
            "spkcache_sil_frames_per_spk": int(modules["spkcache_sil_frames_per_spk"]),
            "scores_add_rnd": float(modules["scores_add_rnd"]),
            "pred_score_threshold": float(modules["pred_score_threshold"]),
            "max_index": int(modules["max_index"]),
            "scores_boost_latest": float(modules["scores_boost_latest"]),
            "sil_threshold": float(modules["sil_threshold"]),
            "strong_boost_rate": float(modules["strong_boost_rate"]),
            "weak_boost_rate": float(modules["weak_boost_rate"]),
            "min_pos_scores_rate": float(modules["min_pos_scores_rate"]),
            "causal_attn_rate": float(modules["causal_attn_rate"]),
            "causal_attn_rc": int(modules["causal_attn_rc"]),
            "frame_hop_samples": int(round(float(pre["window_stride"]) * int(pre["sample_rate"]))) * int(encoder["subsampling_factor"]),
        },
    }


def staging_model_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": "sortformer_diar_v2",
        "display_name": "Sortformer Diarization v2.1",
        "description": "NVIDIA Streaming Sortformer v2.1; four-speaker arrival-order diarization.",
        "category": "speech_analysis",
        "status": "supported",
        "tasks": ["diar"],
        "modes": ["offline", "streaming"],
        "languages": ["multilingual"],
        "capabilities": {"diar": ["speaker_turns"]},
        "dependencies": [],
        "options": {"request": [], "session": [], "load": []},
        "runtime": {"tags": ["gguf"]},
        "ui": {
            "recommended_package": "sortformer_diar_4spk_v2_1_f32",
            "tags": ["Diar", "GGUF"],
            "docs": ["docs/speech_analysis.md", "README.md"],
        },
        "package_defaults": {
            "download": {
                "kind": "huggingface_snapshot",
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "gated": True,
            }
        },
        "packages": [
            {
                "id": "sortformer_diar_4spk_v2_1_f32",
                "display_name": "Sortformer Diar 4spk v2.1 F32 GGUF",
                "default": True,
                "format": "gguf",
                "precision": "f32",
                "target_directory": ".",
                "files": ["sortformer-v2.1-f32.gguf"],
            }
        ],
        "sources": [
            {
                "format": "gguf",
                "roots": {"model": ".", "weights": "$gguf"},
                "files": {"config": "model:config.json", "processor": "model:processor_config.json"},
                "tensors": {"weights": "weights:"},
            },
            {
                "format": "safetensors",
                "roots": {"model": "."},
                "files": {"config": "model:config.json", "processor": "model:processor_config.json"},
                "tensors": {"weights": "model:model.safetensors"},
            },
        ],
    }


def convert(checkpoint: Path, output_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    config, state, extracted_checkpoint, extracted_checkpoint_is_temp = load_checkpoint(checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint did not contain a non-empty state dictionary")
    assert_model_shape(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_hash = sha256(checkpoint) if checkpoint.is_file() else sha256(extracted_checkpoint)
    out: dict[str, torch.Tensor] = {}
    manifest: list[dict[str, Any]] = []
    used: set[str] = set()

    def emit(src: str, dst: str) -> None:
        if src not in state:
            raise KeyError(f"missing expected tensor: {src}")
        if dst in out:
            raise KeyError(f"duplicate destination tensor: {dst}")
        tensor = state[src]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected tensor for {src}, got {type(tensor).__name__}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"expected fp32 tensor for {src}, got {tensor.dtype}")
        tensor = tensor.detach().cpu().contiguous()
        out[dst] = tensor
        used.add(src)
        manifest.append({"source": src, "destination": dst, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})

    filterbank_source = "preprocessor.featurizer.fb"
    if filterbank_source not in state:
        raise KeyError(f"missing expected tensor: {filterbank_source}")
    filterbank = state[filterbank_source]
    if not isinstance(filterbank, torch.Tensor):
        raise TypeError(f"expected tensor for {filterbank_source}, got {type(filterbank).__name__}")
    if filterbank.ndim == 3 and filterbank.shape[0] == 1:
        filterbank = filterbank.squeeze(0)
    if filterbank.ndim != 2:
        raise ValueError(
            f"expected rank-2 or [1, rank-2] filterbank, got shape {tuple(filterbank.shape)}"
        )
    if filterbank.shape[0] != int(config["preprocessor"]["features"]):
        raise ValueError(
            f"filterbank mel dimension {filterbank.shape[0]} does not match preprocessor.features"
        )
    filterbank = filterbank.detach().cpu().float().contiguous()
    out["preprocessor.fb"] = filterbank
    used.add(filterbank_source)
    manifest.append({
        "source": filterbank_source,
        "destination": "preprocessor.fb",
        "shape": list(filterbank.shape),
        "dtype": str(filterbank.dtype),
        "storage": "f32",
    })

    encoder_layers = int(config["encoder"]["n_layers"])
    transformer_layers = int(config["transformer_encoder"]["num_layers"])
    for src, dst in PRE_ENCODE_TABLE:
        emit(src, dst)
    for i in range(encoder_layers):
        for src_suffix, dst_suffix in CONFORMER_BLOCK_TABLE:
            emit(f"encoder.layers.{i}.{src_suffix}", f"enc.blocks.{i}.{dst_suffix}")
    for i in range(transformer_layers):
        for src_suffix, dst_suffix in TRANSFORMER_BLOCK_TABLE:
            emit(f"transformer_encoder.layers.{i}.{src_suffix}", f"tf.blocks.{i}.{dst_suffix}")
    for src, dst in HEAD_TABLE:
        emit(src, dst)

    unexpected = [
        key for key in set(state) - used
        if not key.startswith("preprocessor.") and not key.endswith(".num_batches_tracked")
    ]
    if unexpected:
        raise ValueError(f"{len(unexpected)} unmapped state_dict tensors, e.g. {sorted(unexpected)[:8]}")

    save_file(out, str(output_dir / "model.safetensors"), metadata={
        "format": "sortformer-v2.1-audiocpp-staging",
        "tensor_namespace": "enc/tf/diar",
        "auxiliary_tensors": "preprocessor.fb",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    })
    (output_dir / "config.json").write_text(json.dumps(model_config(config, source_hash), indent=2) + "\n", encoding="utf-8")
    (output_dir / "processor_config.json").write_text(json.dumps(processor_config(config), indent=2) + "\n", encoding="utf-8")
    (output_dir / "model_spec.json").write_text(json.dumps(staging_model_spec(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "tensor_manifest.json").write_text(json.dumps({
        "source": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "source_tensor_count": len(state),
        "emitted_tensor_count": len(out),
        "skipped_tensor_count": len(state) - len(out),
        "tensors": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_license_url": SOURCE_LICENSE_URL,
        "source_sha256": source_hash,
        "mapping_reference": "handy-computer/transcribe.cpp/scripts/convert-sortformer.py",
        "mapping_reference_commit": "e2f82cb6702315a1194f3bf1a6fee67cd2678447",
        "tensor_namespace": "enc/tf/diar",
        "auxiliary_tensors": "preprocessor.fb",
    }, indent=2) + "\n", encoding="utf-8")

    if extracted_checkpoint_is_temp:
        extracted_checkpoint.unlink(missing_ok=True)
    print(f"source tensors: {len(state)}")
    print(f"emitted tensors: {len(out)}")
    print(f"skipped tensors: {len(state) - len(out)} (preprocessor + BN counters)")
    print(f"wrote staging package: {output_dir}")


def package_with_audiocpp(args: argparse.Namespace) -> None:
    if args.converter is None or args.gguf_output is None:
        return
    converter = args.converter.resolve()
    if not converter.exists():
        raise SystemExit(f"audiocpp_gguf not found: {converter}")
    spec = args.model_spec.resolve() if args.model_spec else (args.output_dir / "model_spec.json").resolve()
    if not spec.exists():
        raise SystemExit(f"model spec not found: {spec}")
    output = args.gguf_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(converter),
        "--input", str(args.output_dir / "model.safetensors"),
        "--root", str(args.output_dir),
        "--family", "sortformer_diar_v2",
        "--model-spec", str(spec),
        "--type", args.type,
        "--keep-type", "preprocessor.fb=f32",
    ]
    for override in args.keep_type:
        command.extend(["--keep-type", override])
    if args.keep_nonmatrix_f32:
        manifest_path = args.output_dir / "tensor_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for tensor in manifest["tensors"]:
            if len(tensor["shape"]) != 2:
                command.extend(["--keep-type", f'{tensor["destination"]}=f32'])
    command.extend(["--output", str(output), "--overwrite"])
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="official .nemo file or extracted checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="staging directory to create")
    parser.add_argument("--converter", type=Path, help="optional audiocpp_gguf executable")
    parser.add_argument("--model-spec", type=Path, help="optional audio.cpp model spec; defaults to staging model_spec.json")
    parser.add_argument("--gguf-output", type=Path, help="optional final GGUF output path")
    parser.add_argument("--type", choices=["orig", "f16", "q8_0"], default="orig", help="GGUF storage type; use orig for the F32 gate")
    parser.add_argument(
        "--keep-type",
        action="append",
        default=[],
        metavar="PREFIX*=TYPE",
        help="additional audiocpp_gguf tensor-type override; repeated, first matching rule wins",
    )
    parser.add_argument(
        "--keep-nonmatrix-f32",
        action="store_true",
        help="keep every non-rank-2 emitted tensor as F32 (mixed-F16 profile)",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    convert(args.checkpoint.resolve(), args.output_dir)
    package_with_audiocpp(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
