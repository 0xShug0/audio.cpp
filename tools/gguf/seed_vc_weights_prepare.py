#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Seed-VC reference checkpoints to audio.cpp safetensors.")
    parser.add_argument(
        "--source-root",
        default="models/SeedVC",
        help="Directory containing downloaded Seed-VC, ASTRAL, CAMPPlus, RMVPE, BigVGAN, Whisper, and HuBERT assets.",
    )
    parser.add_argument(
        "--reference-root",
        default="models/SeedVC-reference",
        help="Directory containing the Seed-VC reference source configs.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/SeedVC-MLX",
        help="Directory to write the converted audio.cpp Seed-VC asset bundle.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
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


def as_tensor_dict(payload: Any, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a dict-like state payload")
    out: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise RuntimeError(f"{label} contains a non-string tensor key: {key!r}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"{label} contains a non-tensor value at key: {key}")
        out[key] = value.detach().cpu().contiguous()
    return out


def strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):] if key.startswith(prefix) else key


def prefixed_state(payload: Any, label: str, prefix: str, strip_module: bool = False) -> dict[str, torch.Tensor]:
    state = as_tensor_dict(payload, label)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        local_key = strip_prefix(key, "module.") if strip_module else key
        out[f"{prefix}.{local_key}"] = value
    return out


def stripped_state(payload: Any, label: str, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    state = as_tensor_dict(payload, label)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        local_key = key
        for prefix in prefixes:
            local_key = strip_prefix(local_key, prefix)
        out[local_key] = value
    return out


def effective_weight_norm(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if v.ndim < 2 or g.shape[0] != v.shape[0]:
        raise RuntimeError(f"unsupported weight_norm tensor shapes: v={tuple(v.shape)} g={tuple(g.shape)}")
    dims = tuple(range(1, v.ndim))
    norm = torch.linalg.vector_norm(v.float(), ord=2, dim=dims, keepdim=True)
    if torch.any(norm == 0):
        raise RuntimeError("weight_norm source contains a zero norm row")
    return (v.float() * (g.float().reshape((g.shape[0],) + (1,) * (v.ndim - 1)) / norm)).to(v.dtype)


def materialize_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for key, value in state.items():
        if key.endswith(".weight_v"):
            prefix = key[:-len(".weight_v")]
            g_key = prefix + ".weight_g"
            if g_key in state:
                out[prefix + ".weight"] = effective_weight_norm(value, state[g_key]).detach().cpu().contiguous()
                consumed.add(key)
                consumed.add(g_key)
                continue
        if key in consumed or key.endswith(".weight_g"):
            continue
        out[key] = value
    return out


def save_checked(state: dict[str, torch.Tensor], path: Path) -> dict[str, Any]:
    if not state:
        raise RuntimeError(f"refusing to write empty safetensors file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    reloaded = load_file(str(path))
    if set(reloaded.keys()) != set(state.keys()):
        raise RuntimeError(f"saved safetensors key set changed: {path}")
    for key, value in state.items():
        if not torch.equal(value, reloaded[key]):
            raise RuntimeError(f"saved safetensors changed tensor {key}: {path}")
    return {
        "path": str(path.relative_to(path.parents[1])),
        "tensor_count": len(state),
        "keys": sorted(state.keys()),
    }


def convert_v2_ar(source_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = torch.load(require_file(source_root / "seed-vc/v2/ar_base.pth"), map_location="cpu")
    net = payload.get("net") if isinstance(payload, dict) else None
    if not isinstance(net, dict):
        raise RuntimeError("Seed-VC V2 AR checkpoint is missing net")
    state: dict[str, torch.Tensor] = {}
    state.update(prefixed_state(net.get("ar"), "v2 ar", "ar", strip_module=True))
    state.update(prefixed_state(net.get("length_regulator"), "v2 ar length regulator", "length_regulator", strip_module=True))
    return save_checked(state, output_dir / "v2/ar.safetensors")


def convert_v2_cfm(source_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = torch.load(require_file(source_root / "seed-vc/v2/cfm_small.pth"), map_location="cpu")
    net = payload.get("net") if isinstance(payload, dict) else None
    if not isinstance(net, dict):
        raise RuntimeError("Seed-VC V2 CFM checkpoint is missing net")
    state: dict[str, torch.Tensor] = {}
    state.update(prefixed_state(net.get("cfm"), "v2 cfm", "cfm", strip_module=True))
    state.update(prefixed_state(net.get("length_regulator"), "v2 cfm length regulator", "length_regulator", strip_module=True))
    return save_checked(state, output_dir / "v2/cfm.safetensors")


def convert_v1_svc(source_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = torch.load(
        require_file(source_root / "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
        map_location="cpu",
    )
    net = payload.get("net") if isinstance(payload, dict) else None
    if not isinstance(net, dict):
        raise RuntimeError("Seed-VC V1 SVC checkpoint is missing net")
    state: dict[str, torch.Tensor] = {}
    for component in ("cfm", "style_encoder", "vq", "length_regulator"):
        state.update(prefixed_state(net.get(component), f"v1 svc {component}", component, strip_module=True))
    return save_checked(state, output_dir / "v1/svc.safetensors")


def convert_v1_variant(
    source_root: Path,
    output_dir: Path,
    checkpoint_name: str,
    output_name: str,
    label: str) -> dict[str, Any]:
    payload = torch.load(require_file(source_root / "seed-vc" / checkpoint_name), map_location="cpu")
    net = payload.get("net") if isinstance(payload, dict) else None
    if not isinstance(net, dict):
        raise RuntimeError(f"Seed-VC {label} checkpoint is missing net")
    state: dict[str, torch.Tensor] = {}
    for component in ("cfm", "style_encoder", "vq", "length_regulator"):
        state.update(prefixed_state(net.get(component), f"{label} {component}", component, strip_module=True))
    return save_checked(materialize_weight_norm(state), output_dir / "v1" / f"{output_name}.safetensors")


def convert_raw_checkpoint(source_path: Path, output_path: Path, label: str, prefix: str = "") -> dict[str, Any]:
    payload = torch.load(require_file(source_path), map_location="cpu")
    state = as_tensor_dict(payload, label)
    if prefix:
        state = {f"{prefix}.{key}": value for key, value in state.items()}
    return save_checked(state, output_path)


def convert_stripped_checkpoint(
    source_path: Path,
    output_path: Path,
    label: str,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    payload = torch.load(require_file(source_path), map_location="cpu")
    return save_checked(stripped_state(payload, label, prefixes), output_path)


def convert_weight_norm_checkpoint(source_path: Path, output_path: Path, label: str) -> dict[str, Any]:
    payload = torch.load(require_file(source_path), map_location="cpu")
    return save_checked(materialize_weight_norm(as_tensor_dict(payload, label)), output_path)


def convert_bigvgan(source_path: Path, output_path: Path, label: str) -> dict[str, Any]:
    payload = torch.load(require_file(source_path), map_location="cpu")
    generator = payload.get("generator") if isinstance(payload, dict) else None
    return save_checked(as_tensor_dict(generator, label), output_path)


def copy_tree(source: Path, destination: Path) -> None:
    require_dir(source)
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(".cache", "*.bin", "*.pt", "*.pth")
    shutil.copytree(source, destination, ignore=ignore)


def copy_hf_snapshot(repo_id: str, destination: Path) -> Path:
    source = Path(snapshot_download(repo_id=repo_id, local_files_only=False))
    copy_tree(source, destination)
    return source


def copy_normalized_config(source: Path, yaml_destination: Path, json_destination: Path) -> None:
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:
            return True

    yaml_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(require_file(source).read_text(encoding="utf-8"))
    yaml_destination.write_text(
        yaml.dump(payload, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    json_destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source_root = resolve_path(args.source_root)
    reference_root = resolve_path(args.reference_root)
    output_dir = resolve_path(args.output_dir)
    require_dir(source_root)
    require_dir(reference_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model_type": "seed_vc",
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "weights": {},
        "copied_assets": {},
    }
    weights = manifest["weights"]
    weights["v2_ar"] = convert_v2_ar(source_root, output_dir)
    weights["v2_cfm"] = convert_v2_cfm(source_root, output_dir)
    weights["v1_svc"] = convert_v1_svc(source_root, output_dir)
    weights["v1_whisper_bigvgan"] = convert_v1_variant(
        source_root,
        output_dir,
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "whisper_bigvgan",
        "V1 Whisper BigVGAN",
    )
    weights["v1_xlsr_hift"] = convert_v1_variant(
        source_root,
        output_dir,
        "DiT_uvit_tat_xlsr_ema.pth",
        "xlsr_hift",
        "V1 XLSR HiFT",
    )
    weights["astral_bsq32"] = convert_raw_checkpoint(
        source_root / "astral-quantization/bsq32/bsq32_light.pth",
        output_dir / "astral/bsq32.safetensors",
        "ASTRAL BSQ32",
    )
    weights["astral_bsq2048"] = convert_raw_checkpoint(
        source_root / "astral-quantization/bsq2048/bsq2048_light.pth",
        output_dir / "astral/bsq2048.safetensors",
        "ASTRAL BSQ2048",
    )
    weights["campplus"] = convert_raw_checkpoint(
        source_root / "campplus/campplus_cn_common.bin",
        output_dir / "campplus/model.safetensors",
        "CAMPPlus",
        prefix="speaker_encoder",
    )
    weights["rmvpe"] = convert_raw_checkpoint(
        source_root / "rmvpe/rmvpe.pt",
        output_dir / "rmvpe/model.safetensors",
        "RMVPE",
    )
    weights["bigvgan_22k"] = convert_bigvgan(
        source_root / "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt",
        output_dir / "bigvgan/v2_22khz_80band_256x/model.safetensors",
        "BigVGAN 22 kHz",
    )
    weights["bigvgan_44k"] = convert_bigvgan(
        source_root / "bigvgan_v2_44khz_128band_512x/bigvgan_generator.pt",
        output_dir / "bigvgan/v2_44khz_128band_512x/model.safetensors",
        "BigVGAN 44 kHz",
    )
    weights["hift"] = convert_weight_norm_checkpoint(
        source_root / "seed-vc/hift.pt",
        output_dir / "hift/model.safetensors",
        "HiFT",
    )

    copy_tree(source_root / "whisper-small", output_dir / "whisper-small")
    copy_tree(source_root / "hubert-large-ll60k", output_dir / "hubert-large-ll60k")
    xlsr_source = copy_hf_snapshot("facebook/wav2vec2-xls-r-300m", output_dir / "wav2vec2-xls-r-300m")
    weights["hubert_large_ll60k"] = convert_raw_checkpoint(
        source_root / "hubert-large-ll60k/pytorch_model.bin",
        output_dir / "hubert-large-ll60k/model.safetensors",
        "HuBERT-large-ll60k",
    )
    weights["wav2vec2_xlsr_300m"] = convert_stripped_checkpoint(
        xlsr_source / "pytorch_model.bin",
        output_dir / "wav2vec2-xls-r-300m/model.safetensors",
        "Wav2Vec2 XLS-R 300M",
        ("wav2vec2.",),
    )
    copy_normalized_config(
        reference_root / "configs/v2/vc_wrapper.yaml",
        output_dir / "v2/vc_wrapper.yaml",
        output_dir / "v2/vc_wrapper.json")
    copy_normalized_config(
        reference_root / "configs/astral_quantization/default_32.yml",
        output_dir / "astral/bsq32.yml",
        output_dir / "astral/bsq32.json")
    copy_normalized_config(
        reference_root / "configs/astral_quantization/default_2048.yml",
        output_dir / "astral/bsq2048.yml",
        output_dir / "astral/bsq2048.json")
    copy_normalized_config(
        source_root / "seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        output_dir / "v1/svc.yml",
        output_dir / "v1/svc.json")
    copy_normalized_config(
        source_root / "seed-vc/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        output_dir / "v1/whisper_bigvgan.yml",
        output_dir / "v1/whisper_bigvgan.json")
    copy_normalized_config(
        source_root / "seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml",
        output_dir / "v1/xlsr_hift.yml",
        output_dir / "v1/xlsr_hift.json")
    copy_normalized_config(
        source_root / "seed-vc/hifigan.yml",
        output_dir / "hift/config.yml",
        output_dir / "hift/config.json")
    shutil.copy2(require_file(source_root / "bigvgan_v2_22khz_80band_256x/config.json"), output_dir / "bigvgan/v2_22khz_80band_256x/config.json")
    shutil.copy2(require_file(source_root / "bigvgan_v2_44khz_128band_512x/config.json"), output_dir / "bigvgan/v2_44khz_128band_512x/config.json")

    manifest["copied_assets"] = {
        "whisper_small": "whisper-small",
        "hubert_large_ll60k": "hubert-large-ll60k",
        "v2_wrapper_config": "v2/vc_wrapper.yaml",
        "v2_wrapper_json": "v2/vc_wrapper.json",
        "astral_bsq32_config": "astral/bsq32.yml",
        "astral_bsq32_json": "astral/bsq32.json",
        "astral_bsq2048_config": "astral/bsq2048.yml",
        "astral_bsq2048_json": "astral/bsq2048.json",
        "v1_svc_config": "v1/svc.yml",
        "v1_svc_json": "v1/svc.json",
        "v1_whisper_bigvgan_config": "v1/whisper_bigvgan.yml",
        "v1_whisper_bigvgan_json": "v1/whisper_bigvgan.json",
        "v1_xlsr_hift_config": "v1/xlsr_hift.yml",
        "v1_xlsr_hift_json": "v1/xlsr_hift.json",
        "hift_config": "hift/config.yml",
        "hift_json": "hift/config.json",
        "bigvgan_22k_config": "bigvgan/v2_22khz_80band_256x/config.json",
        "bigvgan_44k_config": "bigvgan/v2_44khz_128band_512x/config.json",
        "wav2vec2_xlsr_300m": "wav2vec2-xls-r-300m",
    }
    (output_dir / "seed_vc_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"source_root={source_root}")
    print(f"reference_root={reference_root}")
    print(f"output_dir={output_dir}")
    for name, info in weights.items():
        print(f"{name}: {info['tensor_count']} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
