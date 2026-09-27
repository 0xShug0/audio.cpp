#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "demucs"
REMOTE_ROOT = REFERENCE_ROOT / "demucs" / "remote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert original Demucs .th checkpoints into audio.cpp-friendly safetensors + JSON configs."
    )
    parser.add_argument(
        "--name",
        default="htdemucs",
        help="Reference model or bag name, e.g. htdemucs, htdemucs_ft, or htdemucs_6s.",
    )
    parser.add_argument("--repo", default="models/htdemucs", help="Directory containing original .th checkpoints.")
    parser.add_argument("--output-dir", default="models/htdemucs", help="Directory to write converted artifacts.")
    return parser.parse_args()


def install_import_stubs() -> None:
    if "openunmix.filtering" not in sys.modules:
        filtering = types.ModuleType("openunmix.filtering")

        def _wiener_stub(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("openunmix.filtering.wiener stub should not be called during Demucs conversion")

        filtering.wiener = _wiener_stub
        openunmix = types.ModuleType("openunmix")
        openunmix.filtering = filtering
        sys.modules["openunmix"] = openunmix
        sys.modules["openunmix.filtering"] = filtering

    if "julius" not in sys.modules:
        julius = types.ModuleType("julius")

        def _resample_frac_stub(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("julius.resample_frac stub should not be called during Demucs conversion")

        julius.resample_frac = _resample_frac_stub
        sys.modules["julius"] = julius

    if "dora.log" not in sys.modules:
        log = types.ModuleType("dora.log")

        def _fatal_stub(message: str) -> None:
            raise RuntimeError(message)

        log.fatal = _fatal_stub
        dora = types.ModuleType("dora")
        dora.log = log
        sys.modules["dora"] = dora
        sys.modules["dora.log"] = log


def load_package(path: Path) -> dict[str, Any]:
    install_import_stubs()
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            package = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        try:
            sys.path.remove(str(REFERENCE_ROOT))
        except ValueError:
            pass
    if not isinstance(package, dict):
        raise RuntimeError(f"unexpected Demucs package type in {path}: {type(package)}")
    required = {"klass", "args", "kwargs", "state"}
    missing = required.difference(package)
    if missing:
        raise RuntimeError(f"Demucs package {path} is missing keys: {sorted(missing)}")
    return package


def find_checkpoint(repo_dir: Path, signature: str) -> Path:
    matches = sorted(repo_dir.glob(f"{signature}*.th"))
    if not matches:
        raise RuntimeError(f"Could not find checkpoint for signature '{signature}' in {repo_dir}")
    if len(matches) > 1:
        exact = [path for path in matches if path.stem == signature or path.stem.startswith(signature + "-")]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"Ambiguous checkpoints for signature '{signature}' in {repo_dir}: {matches}")
    return matches[0]


def normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return repr(value)


def export_model_config(package: dict[str, Any], checkpoint_path: Path, signature: str) -> dict[str, Any]:
    klass = package["klass"]
    kwargs = {str(key): normalize_value(value) for key, value in package["kwargs"].items()}
    return {
        "model_type": "demucs",
        "class_name": getattr(klass, "__name__", str(klass)),
        "signature": signature,
        "checkpoint_file": checkpoint_path.name,
        "args": [normalize_value(value) for value in package["args"]],
        "kwargs": kwargs,
        "sources": normalize_value(kwargs.get("sources")),
        "samplerate": normalize_value(kwargs.get("samplerate")),
        "audio_channels": normalize_value(kwargs.get("audio_channels")),
        "segment": normalize_value(kwargs.get("segment")),
    }


def export_state_tensors(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = package["state"]
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected Demucs state payload type: {type(state)}")
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "__quantized":
            raise RuntimeError("Demucs quantized checkpoints are not supported by this converter yet")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"unexpected non-tensor entry in Demucs state: {key} -> {type(value)}")
        converted[str(key)] = value.detach().cpu().contiguous()
    return converted


def load_bag_manifest(name: str) -> dict[str, Any] | None:
    yaml_path = REMOTE_ROOT / f"{name}.yaml"
    if not yaml_path.is_file():
        return None
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "models" not in payload:
        raise RuntimeError(f"invalid Demucs bag manifest: {yaml_path}")
    return payload


def convert_single(signature: str, repo_dir: Path, output_root: Path) -> dict[str, Any]:
    checkpoint_path = find_checkpoint(repo_dir, signature)
    package = load_package(checkpoint_path)
    state = export_state_tensors(package)

    model_dir = output_root / signature
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(
        json.dumps(export_model_config(package, checkpoint_path, signature), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "signature": signature,
        "checkpoint_file": checkpoint_path.name,
        "output_dir": str(model_dir.relative_to(output_root)),
        "tensor_count": len(state),
    }


def main() -> int:
    args = parse_args()
    repo_dir = REPO_ROOT / args.repo if not Path(args.repo).is_absolute() else Path(args.repo)
    output_root = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not repo_dir.is_dir():
        raise RuntimeError(f"checkpoint repo directory does not exist: {repo_dir}")

    bag = load_bag_manifest(args.name)
    output_dir = output_root / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if bag is None:
        converted = [convert_single(args.name, repo_dir, output_dir)]
        manifest = {
            "model_type": "demucs_single",
            "name": args.name,
            "model": converted[0],
        }
    else:
        signatures = bag["models"]
        if not isinstance(signatures, list) or not signatures:
            raise RuntimeError(f"Demucs bag manifest has invalid model list: {args.name}")
        converted = [convert_single(str(signature), repo_dir, output_dir) for signature in signatures]
        weights = normalize_value(bag.get("weights"))
        segment = normalize_value(bag.get("segment"))
        if len(converted) == 1 and weights is None and segment is None:
            manifest = {
                "model_type": "demucs_single_alias",
                "name": args.name,
                "alias_of": converted[0]["signature"],
                "model": converted[0],
            }
        else:
            manifest = {
                "model_type": "demucs_bag",
                "name": args.name,
                "models": converted,
                "weights": weights,
                "segment": segment,
            }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"name={args.name}")
    print(f"repo_dir={repo_dir}")
    print(f"output_dir={output_dir}")
    print(f"model_count={len(converted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
