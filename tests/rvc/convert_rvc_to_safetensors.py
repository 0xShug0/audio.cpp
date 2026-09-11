#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official RVC PyTorch checkpoints to safetensors.")
    parser.add_argument("--source-root", default="models/RVC/raw")
    parser.add_argument("--output-root", default="models/RVC/safetensors")
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


def tensor_state_dict(payload: Any, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a dict-like checkpoint")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise RuntimeError(f"{label} contains a non-string key: {key!r}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"{label} contains a non-tensor value at key: {key}")
        tensors[key] = value.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError(f"{label} contains no tensors")
    return tensors


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def save_checked(state: dict[str, torch.Tensor], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_path))
    reloaded = load_file(str(output_path))
    if set(reloaded.keys()) != set(state.keys()):
        raise RuntimeError(f"saved safetensors key set changed: {output_path}")
    for key, value in state.items():
        if not torch.equal(value, reloaded[key]):
            raise RuntimeError(f"saved safetensors changed tensor {key}: {output_path}")
    return {
        "path": str(output_path),
        "tensor_count": len(state),
        "keys": sorted(state.keys()),
    }


def checkpoint_tensors_and_metadata(source_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(require_file(source_path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = tensor_state_dict(payload["model"], str(source_path))
        metadata = {key: json_safe(value) for key, value in payload.items() if key != "model"}
        return state, metadata
    if isinstance(payload, dict) and isinstance(payload.get("weight"), dict):
        state = tensor_state_dict(payload["weight"], str(source_path))
        metadata = {key: json_safe(value) for key, value in payload.items() if key != "weight"}
        return state, metadata
    return tensor_state_dict(payload, str(source_path)), {}


def convert_checkpoint(source_path: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    relative = source_path.relative_to(source_root)
    output_path = output_root / relative.with_suffix(".safetensors")
    state, metadata = checkpoint_tensors_and_metadata(source_path)
    result = save_checked(state, output_path)
    if metadata:
        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result["metadata_path"] = str(metadata_path)
    return result


def convert_onnx(source_path: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    relative = source_path.relative_to(source_root)
    output_path = output_root / relative.with_suffix(".onnx.safetensors")
    model = onnx.load(str(require_file(source_path)))
    state: dict[str, torch.Tensor] = {}
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer).copy()
        state[initializer.name] = torch.from_numpy(array).detach().cpu().contiguous()
    return save_checked(state, output_path)


def convert_faiss_index(source_path: Path, output_path: Path) -> str:
    import faiss
    import numpy as np

    index = faiss.read_index(str(source_path))
    if index.__class__.__name__ != "IndexIVFFlat":
        raise RuntimeError(f"RVC native retrieval sidecar supports only IndexIVFFlat: {source_path}")
    quantizer = faiss.downcast_index(index.quantizer)
    if not quantizer.__class__.__name__.startswith("IndexFlat"):
        raise RuntimeError(f"RVC native retrieval sidecar requires a flat quantizer: {source_path}")
    centroids = np.ascontiguousarray(quantizer.reconstruct_n(0, index.nlist), dtype="float32")
    vectors: list[np.ndarray] = []
    offsets: list[float] = []
    lengths: list[float] = []
    cursor = 0
    for list_id in range(index.nlist):
        size = int(index.invlists.list_size(list_id))
        offsets.append(float(cursor))
        lengths.append(float(size))
        if size:
            codes = faiss.rev_swig_ptr(index.invlists.get_codes(list_id), size * index.code_size)
            vectors.append(np.frombuffer(codes, dtype="float32").reshape(size, index.d).copy())
        cursor += size
    all_vectors = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, index.d), dtype="float32")
    save_checked(
        {
            "centroids": torch.from_numpy(centroids).contiguous(),
            "vectors": torch.from_numpy(all_vectors).contiguous(),
            "list_offsets": torch.tensor(offsets, dtype=torch.float32),
            "list_lengths": torch.tensor(lengths, dtype=torch.float32),
            "metric_type": torch.tensor([float(index.metric_type)], dtype=torch.float32),
            "nprobe": torch.tensor([float(index.nprobe)], dtype=torch.float32),
        },
        output_path,
    )
    return str(output_path)


def copy_and_convert_indices(source_root: Path, output_root: Path) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    converted: list[str] = []
    for source_path in sorted(source_root.rglob("*.index")):
        relative = source_path.relative_to(source_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        copied.append(str(output_path))
        converted.append(convert_faiss_index(source_path, output_path.with_suffix(".ivf.safetensors")))
    return copied, converted


def main() -> int:
    args = parse_args()
    source_root = resolve_path(args.source_root)
    output_root = resolve_path(args.output_root)
    if not source_root.is_dir():
        raise RuntimeError(f"source root does not exist: {source_root}")

    converted = []
    for source_path in sorted([*source_root.rglob("*.pt"), *source_root.rglob("*.pth")]):
        if ".cache" in source_path.parts:
            continue
        converted.append(convert_checkpoint(source_path, source_root, output_root))
    onnx_converted = []
    for source_path in sorted(source_root.rglob("*.onnx")):
        if ".cache" in source_path.parts:
            continue
        onnx_converted.append(convert_onnx(source_path, source_root, output_root))
    copied_indices, converted_indices = copy_and_convert_indices(source_root, output_root)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "checkpoint_count": len(converted),
        "onnx_count": len(onnx_converted),
        "index_count": len(copied_indices),
        "index_sidecar_count": len(converted_indices),
        "checkpoints": converted,
        "onnx": onnx_converted,
        "indices": copied_indices,
        "index_sidecars": converted_indices,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"converted_checkpoints={len(converted)}")
    print(f"converted_onnx={len(onnx_converted)}")
    print(f"copied_indices={len(copied_indices)}")
    print(f"converted_index_sidecars={len(converted_indices)}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
