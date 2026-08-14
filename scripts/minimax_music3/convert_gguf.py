#!/usr/bin/env python3
"""Convert MiniMax-Music3 diffusers-format checkpoints to audio.cpp component GGUFs.

Reads the HF snapshot of MiniMaxAI/MiniMax-Music3 and writes one GGUF per component:

  lm                 language_model/ (Qwen3-8B), with the lm_head sliced to the
                     16385 sampleable rows (row 0 = audio end token 151670,
                     rows 1..16384 = semantic codes at offset 151675)
  depth_decoder      rvq_depth_decoder/
  dit                transformer/ (flow-matching transformer)
  condition_encoder  condition_encoder/
  vocoder            vocoder/, with torch weight_norm weight_g/weight_v pairs
                     folded into plain conv weights

Example:

  scripts/minimax_music3/convert_gguf.py \
    --component lm --snapshot models/MiniMax-Music3-hf \
    --output models/MiniMax-Music3-Q4-GGUF/lm_q4_k.gguf --type q4_k
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

GGML_MAX_NAME = 64
AUDIO_END_TOKEN_ID = 151670
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384
LM_HEAD_KEY = "lm_head.weight"
LM_HEAD_SLICED_KEY = "lm_head_sliced.weight"

COMPONENT_DIRS = {
    "lm": "language_model",
    "depth_decoder": "rvq_depth_decoder",
    "dit": "transformer",
    "condition_encoder": "condition_encoder",
    "vocoder": "vocoder",
}


@dataclass(frozen=True)
class TypeOverride:
    pattern: str
    qtype: gguf.GGMLQuantizationType | None


def normalize_type_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def parse_ggml_type(value: str) -> gguf.GGMLQuantizationType | None:
    name = normalize_type_name(value)
    if name in {"native", "orig", "original"}:
        return None
    for item in gguf.GGMLQuantizationType:
        if normalize_type_name(item.name) == name:
            return item
    raise argparse.ArgumentTypeError(f"unknown GGML tensor type: {value}")


def parse_override(value: str) -> TypeOverride:
    if "=" not in value:
        raise argparse.ArgumentTypeError("override must be PATTERN=TYPE")
    pattern, type_text = value.split("=", 1)
    if not pattern:
        raise argparse.ArgumentTypeError("override pattern cannot be empty")
    return TypeOverride(pattern=pattern, qtype=parse_ggml_type(type_text))


def is_quantized_type(qtype: gguf.GGMLQuantizationType) -> bool:
    return qtype not in {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.I8,
        gguf.GGMLQuantizationType.I16,
        gguf.GGMLQuantizationType.I32,
        gguf.GGMLQuantizationType.I64,
        gguf.GGMLQuantizationType.F64,
    }


def can_quantize(shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> bool:
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]
    return len(shape) > 0 and shape[-1] % block_size == 0


def load_component_tensors(component_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor of one component, resolving sharded checkpoints."""
    index_files = sorted(component_dir.glob("*.safetensors.index.json"))
    tensors: dict[str, torch.Tensor] = {}
    if index_files:
        index = json.loads(index_files[0].read_text())
        shard_keys: dict[str, list[str]] = {}
        for key, shard in index["weight_map"].items():
            shard_keys.setdefault(shard, []).append(key)
        for shard, keys in sorted(shard_keys.items()):
            with safe_open(component_dir / shard, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensors[key] = handle.get_tensor(key)
        return tensors
    shard_files = sorted(component_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors found in {component_dir}")
    for shard in shard_files:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensors:
                    raise ValueError(f"duplicate tensor {key} across shards in {component_dir}")
                tensors[key] = handle.get_tensor(key)
    return tensors


def slice_lm_head(tensors: dict[str, torch.Tensor]) -> None:
    """Replace the full 200k-row lm_head with the 16385 sampleable rows."""
    head = tensors.pop(LM_HEAD_KEY)
    rows = [AUDIO_END_TOKEN_ID] + list(range(AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE))
    index = torch.tensor(rows, dtype=torch.int64)
    tensors[LM_HEAD_SLICED_KEY] = head.index_select(0, index).contiguous()


def fold_weight_norm(tensors: dict[str, torch.Tensor]) -> None:
    """Fold torch weight_norm (dim=0) weight_g/weight_v pairs into plain weights."""
    bases = sorted({name[: -len(".weight_g")] for name in tensors if name.endswith(".weight_g")})
    for base in bases:
        weight_g = tensors.pop(base + ".weight_g").to(torch.float64)
        weight_v = tensors.pop(base + ".weight_v").to(torch.float64)
        norm_dims = tuple(range(1, weight_v.dim()))
        norm = weight_v.pow(2).sum(dim=norm_dims, keepdim=True).sqrt()
        tensors[base + ".weight"] = (weight_g * weight_v / norm).to(torch.float32)


def to_f32_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).contiguous().numpy()


def native_payload(tensor: torch.Tensor) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    if tensor.dtype == torch.bfloat16:
        return gguf.quants.quantize(to_f32_array(tensor), gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16
    if tensor.dtype in {torch.float16, torch.float32, torch.float64}:
        return tensor.detach().cpu().contiguous().numpy(), None
    raise ValueError(f"no native GGUF storage for torch dtype {tensor.dtype}")


def resolve_target(
    name: str,
    requested: gguf.GGMLQuantizationType | None,
    overrides: list[TypeOverride],
) -> gguf.GGMLQuantizationType | None | str:
    for override in overrides:
        if fnmatch.fnmatchcase(name, override.pattern):
            return override.qtype
    return requested


def ggml_quantize_with_helper(
    data: np.ndarray,
    qtype: gguf.GGMLQuantizationType,
    helper: Path,
) -> np.ndarray:
    if not helper.is_file():
        raise FileNotFoundError(
            f"gguf-py cannot quantize {qtype.name} and the ggml helper was not found: {helper} "
            "(build the ggml-quantize-raw target and pass --ggml-quantize-helper)")
    process = subprocess.run(
        [str(helper), str(int(qtype)), str(int(data.shape[-1])), "256"],
        input=np.ascontiguousarray(data, dtype=np.float32).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
    byte_shape = gguf.quants.quant_shape_to_byte_shape(data.shape, qtype)
    expected = int(np.prod(byte_shape))
    if len(process.stdout) != expected:
        raise RuntimeError(f"ggml helper returned {len(process.stdout)} bytes, expected {expected}")
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(byte_shape).copy()


def convert_tensor(
    name: str,
    tensor: torch.Tensor,
    target: gguf.GGMLQuantizationType | None,
    helper: Path,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    """Return (payload, raw_dtype) for the writer; raw_dtype None means numpy-native."""
    if target is None:
        return native_payload(tensor)
    shape = tuple(tensor.shape)
    if is_quantized_type(target):
        eligible = len(shape) == 2 and name.endswith(".weight") and can_quantize(shape, target)
        if not eligible:
            return native_payload(tensor)
        try:
            return gguf.quants.quantize(to_f32_array(tensor), target), target
        except NotImplementedError:
            return ggml_quantize_with_helper(to_f32_array(tensor), target, helper), target
    if target == gguf.GGMLQuantizationType.F32:
        return to_f32_array(tensor), None
    if target == gguf.GGMLQuantizationType.F16:
        return to_f32_array(tensor).astype(np.float16), None
    if target == gguf.GGMLQuantizationType.BF16:
        return gguf.quants.quantize(to_f32_array(tensor), target), target
    raise ValueError(f"unsupported target type {target.name} for {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", choices=sorted(COMPONENT_DIRS), required=True)
    parser.add_argument("--snapshot", type=Path, required=True, help="MiniMax-Music3 HF snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--type", default=None, type=parse_ggml_type, help="native, f16, bf16, q8_0, q4_k, ...")
    parser.add_argument("--override", action="append", type=parse_override, default=[], help="PATTERN=TYPE")
    parser.add_argument(
        "--ggml-quantize-helper",
        type=Path,
        default=Path("build/linux-cuda-release/bin/ggml-quantize-raw"),
        help="helper binary for tensor types gguf-py cannot quantize (K-quants)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    component_dir = args.snapshot / COMPONENT_DIRS[args.component]
    if not component_dir.is_dir():
        raise SystemExit(f"component directory does not exist: {component_dir}")
    output: Path = args.output
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tensors = load_component_tensors(component_dir)
    if args.component == "lm":
        slice_lm_head(tensors)
    if args.component == "vocoder":
        fold_weight_norm(tensors)

    logical_names = sorted(tensors)
    physical_names: list[str] = []
    used: set[str] = set()
    for index, name in enumerate(logical_names):
        physical = name if len(name) < GGML_MAX_NAME and name not in used else f"_standalone.{index}"
        if physical in used:
            raise ValueError(f"duplicate physical tensor name: {physical}")
        used.add(physical)
        physical_names.append(physical)

    tmp = output.with_name(output.name + ".tmp")
    writer = gguf.GGUFWriter(tmp, "audiocpp", use_temp_file=True)
    try:
        writer.add_name(args.name or output.stem)
        writer.add_string("audiocpp.tensor_name_format", "native")
        writer.add_string("audiocpp.family", "minimax_music3")
        writer.add_string("audiocpp.component", args.component)

        logical_shapes: list[tuple[int, ...]] = []
        for index, name in enumerate(logical_names):
            tensor = tensors.pop(name)
            target = resolve_target(name, args.type, args.override)
            payload, raw_dtype = convert_tensor(name, tensor, target, args.ggml_quantize_helper)
            logical_shapes.append(tuple(int(dim) for dim in tensor.shape))
            writer.add_tensor(physical_names[index], payload, raw_dtype=raw_dtype)
            print(f"[{index + 1}/{len(logical_names)}] {name} shape={list(tensor.shape)} bytes={payload.nbytes}", flush=True)
            del tensor, payload

        writer.add_array("audiocpp.tensor_names", logical_names)
        writer.add_key_value(
            "audiocpp.tensor_ranks",
            [len(shape) for shape in logical_shapes],
            gguf.GGUFValueType.ARRAY,
            sub_type=gguf.GGUFValueType.INT32,
        )
        writer.add_key_value(
            "audiocpp.tensor_shapes",
            [dim for shape in logical_shapes for dim in shape],
            gguf.GGUFValueType.ARRAY,
            sub_type=gguf.GGUFValueType.INT64,
        )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()
    tmp.replace(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
