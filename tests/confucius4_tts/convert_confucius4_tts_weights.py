#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Confucius4-TTS reference checkpoints to an AudioCPP safetensors bundle."
    )
    parser.add_argument("--model-root", default="models/Confucius4-TTS")
    parser.add_argument("--w2v-root", default="models/facebook-w2v-bert-2.0")
    parser.add_argument("--bigvgan-root", default="models/bigvgan_v2_22khz_80band_256x")
    parser.add_argument("--campplus", default="models/funasr-campplus/campplus_cn_common.bin")
    parser.add_argument("--reference-root", default="reference/Confucius4-TTS")
    parser.add_argument("--output-dir", default="models/Confucius4-TTS/audio_cpp")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-semantic-shaw", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--write-gguf", action="store_true")
    parser.add_argument("--gguf-output", default="model.gguf")
    parser.add_argument("--gguf-type", default="orig")
    parser.add_argument("--gguf-tool", default="build/debug/bin/audiocpp_gguf")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise RuntimeError(f"required directory does not exist: {path}")
    return path


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def prepare_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def as_tensor_dict(payload: Any, label: str) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a dict-like state payload")
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(payload):
        value = payload[key]
        if not isinstance(key, str):
            raise RuntimeError(f"{label} contains a non-string key: {key!r}")
        if not torch.is_tensor(value):
            continue
        tensors[key] = prepare_tensor(value)
    if not tensors:
        raise RuntimeError(f"{label} did not contain tensors")
    return tensors


def checkpoint_tensors(path: Path, label: str) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for wrapper_key in ("state_dict", "generator", "model"):
            wrapped = payload.get(wrapper_key)
            if isinstance(wrapped, dict):
                return as_tensor_dict(wrapped, f"{label}.{wrapper_key}")
    return as_tensor_dict(payload, label)


def prefixed_tensors(tensors: OrderedDict[str, torch.Tensor], prefix: str) -> OrderedDict[str, torch.Tensor]:
    if not prefix:
        raise RuntimeError("prefixed_tensors requires a non-empty prefix")
    return OrderedDict((f"{prefix}.{key}", value) for key, value in tensors.items())


def effective_weight_norm(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if v.ndim < 2 or g.shape[0] != v.shape[0]:
        raise RuntimeError(f"unsupported weight_norm tensor shapes: v={tuple(v.shape)} g={tuple(g.shape)}")
    dims = tuple(range(1, v.ndim))
    norm = torch.linalg.vector_norm(v.float(), ord=2, dim=dims, keepdim=True)
    if torch.any(norm == 0):
        raise RuntimeError("weight_norm source contains a zero norm row")
    g_view = g.float().reshape((g.shape[0],) + (1,) * (v.ndim - 1))
    return prepare_tensor((v.float() * (g_view / norm)).to(v.dtype))


def materialize_weight_norm(state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    consumed: set[str] = set()
    for key, value in state.items():
        if key in consumed:
            continue
        if key.endswith(".parametrizations.weight.original1"):
            prefix = key[: -len(".parametrizations.weight.original1")]
            g_key = prefix + ".parametrizations.weight.original0"
            if g_key in state:
                out[prefix + ".weight"] = effective_weight_norm(value, state[g_key])
                consumed.add(key)
                consumed.add(g_key)
                continue
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            g_key = prefix + ".weight_g"
            if g_key in state:
                out[prefix + ".weight"] = effective_weight_norm(value, state[g_key])
                consumed.add(key)
                consumed.add(g_key)
                continue
        if key.endswith(".parametrizations.weight.original0") or key.endswith(".weight_g"):
            continue
        out[key] = value
    if not out:
        raise RuntimeError("weight_norm materialization produced an empty state")
    return out


def safetensors_info(path: Path) -> dict[str, Any]:
    tensor_count = 0
    total_tensor_bytes = 0
    dtypes: dict[str, int] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            total_tensor_bytes += tensor_bytes(tensor)
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "tensor_bytes": total_tensor_bytes,
        "tensor_count": tensor_count,
        "dtypes": dtypes,
        "metadata": metadata,
    }


def verify_saved(path: Path, expected: OrderedDict[str, torch.Tensor]) -> None:
    actual = load_file(str(path), device="cpu")
    if set(actual.keys()) != set(expected.keys()):
        raise RuntimeError(f"saved safetensors key set changed: {path}")
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key]
        if actual_tensor.shape != expected_tensor.shape:
            raise RuntimeError(
                f"saved tensor shape changed for {key}: {tuple(expected_tensor.shape)} -> {tuple(actual_tensor.shape)}"
            )
        if actual_tensor.dtype != expected_tensor.dtype:
            raise RuntimeError(f"saved tensor dtype changed for {key}: {expected_tensor.dtype} -> {actual_tensor.dtype}")
        if not torch.equal(actual_tensor, expected_tensor):
            raise RuntimeError(f"saved tensor values changed for {key}")


def write_safetensors(
    tensors: OrderedDict[str, torch.Tensor],
    output_path: Path,
    metadata: dict[str, str],
    overwrite: bool,
    verify: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(output_path), metadata=metadata)
    if verify:
        verify_saved(output_path, tensors)
    info = safetensors_info(output_path)
    info["source_format"] = metadata.get("source_format", "")
    info["source_file"] = metadata.get("source_file", "")
    return info


def copy_safetensors(source_path: Path, output_path: Path, metadata_label: str, overwrite: bool) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    info = safetensors_info(output_path)
    info["source_format"] = "safetensors"
    info["source_file"] = str(source_path)
    info["component"] = metadata_label
    return info


def copy_required_asset(source_path: Path, output_path: Path, overwrite: bool) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return {"file": output_path.name, "bytes": output_path.stat().st_size, "source_file": str(source_path)}


def wav2vec_stats_tensors(path: Path) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"wav2vec2bert stats must be a dict: {path}")
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in ("mean", "var"):
        value = payload.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"wav2vec2bert stats missing tensor: {key}")
        tensors[key] = prepare_tensor(value)
    return tensors


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gguf_bundle(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    gguf_tool = require_file(resolve_path(args.gguf_tool))
    gguf_output = Path(args.gguf_output)
    if not gguf_output.is_absolute():
        gguf_output = output_dir / gguf_output
    command = [
        str(gguf_tool),
        "--input",
        f"t2s={output_dir / 't2s.safetensors'}",
        "--input",
        f"s2a={output_dir / 's2a.safetensors'}",
        "--input",
        f"semantic_encoder={output_dir / 'semantic_encoder.safetensors'}",
        "--input",
        f"semantic_encoder_shaw={output_dir / 'semantic_encoder_shaw.safetensors'}",
        "--input",
        f"semantic_stats={output_dir / 'semantic_stats.safetensors'}",
        "--input",
        f"style_encoder={output_dir / 'style_encoder.safetensors'}",
        "--input",
        f"vocoder={output_dir / 'vocoder.safetensors'}",
        "--output",
        str(gguf_output),
        "--type",
        args.gguf_type,
        "--family",
        "confucius4_tts",
        "--model-spec",
        str(REPO_ROOT / "model_specs" / "confucius4_tts.json"),
        "--root",
        str(output_dir),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return {
        "file": gguf_output.name,
        "bytes": gguf_output.stat().st_size,
        "type": args.gguf_type,
    }


def main() -> None:
    args = parse_args()
    model_root = require_dir(resolve_path(args.model_root))
    w2v_root = require_dir(resolve_path(args.w2v_root))
    bigvgan_root = require_dir(resolve_path(args.bigvgan_root))
    reference_root = require_dir(resolve_path(args.reference_root))
    campplus_path = require_file(resolve_path(args.campplus))
    output_dir = resolve_path(args.output_dir)
    verify = not args.no_verify

    components: dict[str, Any] = {}
    assets: dict[str, Any] = {}

    components["t2s"] = copy_safetensors(
        require_file(model_root / "t2s_model.safetensors"),
        output_dir / "t2s.safetensors",
        "t2s",
        args.overwrite,
    )
    components["semantic_encoder"] = copy_safetensors(
        require_file(w2v_root / "model.safetensors"),
        output_dir / "semantic_encoder.safetensors",
        "semantic_encoder",
        args.overwrite,
    )
    if not args.skip_semantic_shaw:
        components["semantic_encoder_shaw"] = write_safetensors(
            checkpoint_tensors(require_file(w2v_root / "conformer_shaw.pt"), "semantic_encoder_shaw"),
            output_dir / "semantic_encoder_shaw.safetensors",
            {
                "format": "pt",
                "audio_cpp_model": "confucius4_tts",
                "component": "semantic_encoder_shaw",
                "source_format": "pytorch",
                "source_file": str(w2v_root / "conformer_shaw.pt"),
            },
            args.overwrite,
            verify,
        )

    components["semantic_stats"] = write_safetensors(
        wav2vec_stats_tensors(require_file(model_root / "wav2vec2bert_stats.pt")),
        output_dir / "semantic_stats.safetensors",
        {
            "format": "pt",
            "audio_cpp_model": "confucius4_tts",
            "component": "semantic_stats",
            "source_format": "pytorch",
            "source_file": str(model_root / "wav2vec2bert_stats.pt"),
        },
        args.overwrite,
        verify,
    )
    components["s2a"] = write_safetensors(
        checkpoint_tensors(require_file(model_root / "s2a_model.pt"), "s2a"),
        output_dir / "s2a.safetensors",
        {
            "format": "pt",
            "audio_cpp_model": "confucius4_tts",
            "component": "s2a",
            "source_format": "pytorch",
            "source_file": str(model_root / "s2a_model.pt"),
        },
        args.overwrite,
        verify,
    )
    components["style_encoder"] = write_safetensors(
        prefixed_tensors(checkpoint_tensors(campplus_path, "style_encoder"), "speaker_encoder"),
        output_dir / "style_encoder.safetensors",
        {
            "format": "pt",
            "audio_cpp_model": "confucius4_tts",
            "component": "style_encoder",
            "source_format": "pytorch",
            "source_file": str(campplus_path),
        },
        args.overwrite,
        verify,
    )
    components["vocoder"] = write_safetensors(
        checkpoint_tensors(require_file(bigvgan_root / "bigvgan_generator.pt"), "vocoder"),
        output_dir / "vocoder.safetensors",
        {
            "format": "pt",
            "audio_cpp_model": "confucius4_tts",
            "component": "vocoder",
            "source_format": "pytorch",
            "source_file": str(bigvgan_root / "bigvgan_generator.pt"),
        },
        args.overwrite,
        verify,
    )

    for name in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        assets[name] = copy_required_asset(require_file(model_root / name), output_dir / name, args.overwrite)
    assets["w2v_preprocessor_config.json"] = copy_required_asset(
        require_file(w2v_root / "preprocessor_config.json"),
        output_dir / "w2v_preprocessor_config.json",
        args.overwrite,
    )
    assets["bigvgan_config.json"] = copy_required_asset(
        require_file(bigvgan_root / "config.json"),
        output_dir / "bigvgan_config.json",
        args.overwrite,
    )
    assets["inference_config.yaml"] = copy_required_asset(
        require_file(reference_root / "config" / "inference_config.yaml"),
        output_dir / "inference_config.yaml",
        args.overwrite,
    )
    manifest = {
        "format": "audio_cpp_confucius4_tts_safetensors_bundle",
        "version": 1,
        "components": components,
        "assets": assets,
    }
    if args.write_gguf:
        manifest["gguf"] = write_gguf_bundle(args, output_dir)
    write_json(output_dir / "manifest.json", manifest, args.overwrite)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
