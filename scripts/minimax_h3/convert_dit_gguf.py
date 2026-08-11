#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gguf
import numpy as np
import torch
from safetensors import safe_open


HELPER_SUFFIXES = (
    ".absmax",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_map",
    ".quant_state.bitsandbytes__nf4",
)
GGML_MAX_NAME = 64


@dataclass(frozen=True)
class InputSpec:
    prefix: str
    path: Path


@dataclass(frozen=True)
class TensorEntry:
    logical_name: str
    source_index: int
    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    is_bnb_nf4: bool


@dataclass(frozen=True)
class SyntheticTensor:
    logical_name: str
    shape: tuple[int, ...]
    data: np.ndarray
    raw_dtype: gguf.GGMLQuantizationType | None
    stored_type: gguf.GGMLQuantizationType


@dataclass(frozen=True)
class TypeOverride:
    pattern: str
    qtype: gguf.GGMLQuantizationType | None


class ManualTensorSlice:
    def __init__(self, dtype: str, shape: tuple[int, ...]):
        self.dtype = dtype
        self.shape = shape

    def get_dtype(self) -> str:
        return self.dtype

    def get_shape(self) -> tuple[int, ...]:
        return self.shape


class ManualSafetensorsHandle:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            self.data_start = 8 + int(header_len)
            self.header = json.loads(handle.read(header_len))
        self.index = {key: value for key, value in self.header.items() if key != "__metadata__"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self.index.keys())

    def get_slice(self, key: str) -> ManualTensorSlice:
        item = self.index[key]
        return ManualTensorSlice(item["dtype"], tuple(int(dim) for dim in item["shape"]))

    def get_tensor(self, key: str) -> torch.Tensor:
        item = self.index[key]
        dtype = item["dtype"]
        start, end = (int(x) for x in item["data_offsets"])
        with self.path.open("rb") as handle:
            handle.seek(self.data_start + start)
            data = handle.read(end - start)
        shape = tuple(int(dim) for dim in item["shape"])
        if dtype == "BF16":
            values = np.frombuffer(data, dtype=np.uint16).copy()
            return torch.from_numpy(values).view(torch.bfloat16).reshape(shape)
        numpy_dtype = {
            "F64": np.float64,
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
        }.get(dtype)
        if numpy_dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {dtype}")
        return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(shape)


def open_safetensors_compat(path: Path):
    try:
        handle = safe_open(path, framework="pt", device="cpu")
        handle.keys()
        return handle
    except Exception:
        return ManualSafetensorsHandle(path)


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


def parse_input(value: str) -> InputSpec:
    if "=" in value:
        prefix, raw_path = value.split("=", 1)
    else:
        prefix, raw_path = "", value
    path = Path(raw_path).expanduser()
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    return InputSpec(prefix=prefix, path=path)


def parse_override(value: str) -> TypeOverride:
    if "=" not in value:
        raise argparse.ArgumentTypeError("override must be PATTERN=TYPE")
    pattern, type_text = value.split("=", 1)
    if not pattern:
        raise argparse.ArgumentTypeError("override pattern cannot be empty")
    return TypeOverride(pattern=pattern, qtype=parse_ggml_type(type_text))


def torch_dtype_to_native_type(dtype: torch.dtype) -> gguf.GGMLQuantizationType:
    mapping = {
        torch.float32: gguf.GGMLQuantizationType.F32,
        torch.float16: gguf.GGMLQuantizationType.F16,
        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
        torch.int8: gguf.GGMLQuantizationType.I8,
        torch.int16: gguf.GGMLQuantizationType.I16,
        torch.int32: gguf.GGMLQuantizationType.I32,
        torch.int64: gguf.GGMLQuantizationType.I64,
        torch.float64: gguf.GGMLQuantizationType.F64,
    }
    if dtype not in mapping:
        raise ValueError(f"no native GGUF storage for torch dtype {dtype}")
    return mapping[dtype]


def safetensors_dtype_to_torch(dtype: str) -> torch.dtype:
    mapping = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return mapping[dtype]


def tensor_to_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).contiguous().numpy()


def tensor_to_native_array(tensor: torch.Tensor) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    dtype = tensor.dtype
    if dtype == torch.bfloat16:
        values = tensor_to_f32(tensor)
        return gguf.quants.quantize(values, gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16
    if dtype == torch.float16:
        return tensor.detach().cpu().contiguous().numpy(), None
    if dtype == torch.float32:
        return tensor.detach().cpu().contiguous().numpy(), None
    if dtype == torch.float64:
        return tensor.detach().cpu().contiguous().numpy(), None
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        return tensor.detach().cpu().contiguous().numpy(), None
    raise ValueError(f"no native GGUF storage for torch dtype {dtype}")


def load_tensor(path: Path, key: str) -> torch.Tensor:
    with open_safetensors_compat(path) as handle:
        return handle.get_tensor(key)


def load_u8(path: Path, key: str) -> np.ndarray:
    tensor = load_tensor(path, key)
    if tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 tensor for {key}, got {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy()


def load_f32(path: Path, key: str) -> np.ndarray:
    tensor = load_tensor(path, key)
    if tensor.dtype != torch.float32:
        raise ValueError(f"expected float32 tensor for {key}, got {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy()


def handle_u8(handle, key: str) -> np.ndarray:
    tensor = handle.get_tensor(key)
    if tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 tensor for {key}, got {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy()


def handle_f32(handle, key: str) -> np.ndarray:
    tensor = handle.get_tensor(key)
    if tensor.dtype != torch.float32:
        raise ValueError(f"expected float32 tensor for {key}, got {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy()


def checked_elements(name: str, shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"tensor shape contains a non-positive dimension: {name}")
        count *= int(dim)
    return count


def load_bnb_nf4(path: Path, key: str) -> np.ndarray:
    with open_safetensors_compat(path) as handle:
        return load_bnb_nf4_from_handle(handle, key)


def load_bnb_nf4_from_handle(handle, key: str) -> np.ndarray:
    state_bytes = handle_u8(handle, key + ".quant_state.bitsandbytes__nf4").tobytes()
    state = json.loads(state_bytes.decode("utf-8"))
    if state.get("quant_type") != "nf4":
        raise ValueError(f"BNB quant_state is not NF4: {key}")
    shape = tuple(int(x) for x in state["shape"])
    blocksize = int(state["blocksize"])
    nested_blocksize = int(state["nested_blocksize"])
    nested_offset = float(state["nested_offset"])
    if blocksize <= 0 or nested_blocksize <= 0:
        raise ValueError(f"BNB NF4 quant_state contains invalid block sizes: {key}")

    elements = checked_elements(key, shape)
    packed_tensor = handle.get_tensor(key)
    if packed_tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 BNB NF4 payload for {key}, got {packed_tensor.dtype}")
    packed = packed_tensor.detach().cpu().contiguous().numpy().reshape(-1)
    if tuple(packed_tensor.shape) != ((elements + 1) // 2, 1):
        raise ValueError(f"BNB NF4 packed tensor shape mismatch: {key}")
    if packed.size != (elements + 1) // 2:
        raise ValueError(f"BNB NF4 packed byte count mismatch: {key}")

    absmax = handle_u8(handle, key + ".absmax").reshape(-1)
    nested_absmax = handle_f32(handle, key + ".nested_absmax").reshape(-1)
    nested_quant_map = handle_f32(handle, key + ".nested_quant_map").reshape(-1)
    quant_map = handle_f32(handle, key + ".quant_map").reshape(-1)
    if nested_quant_map.size != 256 or quant_map.size != 16:
        raise ValueError(f"BNB NF4 quant maps have unexpected size: {key}")
    expected_blocks = (elements + blocksize - 1) // blocksize
    expected_nested = (absmax.size + nested_blocksize - 1) // nested_blocksize
    if absmax.size != expected_blocks:
        raise ValueError(f"BNB NF4 absmax block count mismatch: {key}")
    if nested_absmax.size != expected_nested:
        raise ValueError(f"BNB NF4 nested absmax length mismatch: {key}")

    scales = nested_quant_map[absmax.astype(np.int64)]
    scales *= nested_absmax[np.arange(absmax.size, dtype=np.int64) // nested_blocksize]
    scales += nested_offset

    nibbles = np.empty(elements, dtype=np.uint8)
    nibbles[0::2] = packed >> 4
    lows = nibbles[1::2]
    lows[:] = packed[: lows.size] & 0x0F

    values = quant_map[nibbles].astype(np.float32, copy=False)
    full = elements // blocksize
    if full:
        values[: full * blocksize].reshape(full, blocksize)[:] *= scales[:full, None]
    if full * blocksize < elements:
        values[full * blocksize :] *= scales[full]
    return values.reshape(shape)


def can_quantize(shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> bool:
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]
    return len(shape) > 0 and shape[-1] % block_size == 0


def ggml_quantize_with_helper(
    data: np.ndarray,
    qtype: gguf.GGMLQuantizationType,
    helper: Path,
    chunk_rows: int,
) -> np.ndarray:
    if not helper.is_file():
        raise FileNotFoundError(f"ggml quantize helper not found: {helper}")
    if not can_quantize(tuple(data.shape), qtype):
        raise ValueError(f"can't quantize shape {data.shape} to {qtype.name}")
    process = subprocess.run(
        [str(helper), str(int(qtype)), str(int(data.shape[-1])), str(chunk_rows)],
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


def quantize_array(
    data: np.ndarray,
    qtype: gguf.GGMLQuantizationType,
    helper: Path,
    helper_chunk_rows: int,
    quantizer: str,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    if qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
        return gguf.quants.quantize(data, qtype), None
    if qtype == gguf.GGMLQuantizationType.BF16:
        return gguf.quants.quantize(data, qtype), qtype
    if not is_quantized_type(qtype):
        dtype_map = {
            gguf.GGMLQuantizationType.I8: np.int8,
            gguf.GGMLQuantizationType.I16: np.int16,
            gguf.GGMLQuantizationType.I32: np.int32,
            gguf.GGMLQuantizationType.I64: np.int64,
            gguf.GGMLQuantizationType.F64: np.float64,
        }
        if qtype not in dtype_map:
            raise ValueError(f"explicit conversion to {qtype.name} is not implemented")
        return data.astype(dtype_map[qtype], copy=False), None
    if quantizer in {"auto", "python"}:
        try:
            return gguf.quants.quantize(data, qtype), qtype
        except NotImplementedError:
            if quantizer == "python":
                raise
    return ggml_quantize_with_helper(data, qtype, helper, helper_chunk_rows), qtype


def helper_base_name(keys: set[str], key: str) -> str | None:
    for suffix in HELPER_SUFFIXES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in keys and base + ".quant_state.bitsandbytes__nf4" in keys:
                return base
    return None


def source_tensors(inputs: list[InputSpec]) -> list[TensorEntry]:
    entries: list[TensorEntry] = []
    logical_names: set[str] = set()
    for source_index, spec in enumerate(inputs):
        if not spec.path.is_file():
            raise FileNotFoundError(f"input safetensors file does not exist: {spec.path}")
        with open_safetensors_compat(spec.path) as handle:
            keys = set(handle.keys())
            for key in handle.keys():
                tensor = handle.get_slice(key)
                logical = spec.prefix + key
                if logical in logical_names:
                    raise ValueError(f"duplicate logical tensor name: {logical}")
                logical_names.add(logical)
                entries.append(
                    TensorEntry(
                        logical_name=logical,
                        source_index=source_index,
                        key=key,
                        shape=tuple(int(dim) for dim in tensor.get_shape()),
                        dtype=safetensors_dtype_to_torch(tensor.get_dtype()),
                        is_bnb_nf4=tensor.get_dtype() == "U8"
                        and key + ".quant_state.bitsandbytes__nf4" in keys,
                    )
                )
    return entries


def include_tensor(name: str, include: list[str], exclude_prefix: list[str]) -> bool:
    if include and not any(fnmatch.fnmatchcase(name, pattern) for pattern in include):
        return False
    return not any(prefix and name.startswith(prefix) for prefix in exclude_prefix)


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def physical_names(logical_names: list[str], alias_max_len: int) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for index, name in enumerate(logical_names):
        physical = name
        if alias_max_len > 0 and (len(physical) >= alias_max_len or physical in used):
            physical = f"_standalone.{index}"
        if physical in used:
            raise ValueError(f"duplicate physical tensor name: {physical}")
        used.add(physical)
        out.append(physical)
    return out


def matching_override(name: str, overrides: list[TypeOverride]) -> gguf.GGMLQuantizationType | None | str:
    for override in overrides:
        if fnmatch.fnmatchcase(name, override.pattern):
            return override.qtype
    return "no-match"


def should_quantize_by_scope(entry: TensorEntry, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "none":
        return False
    return len(entry.shape) == 2 and entry.logical_name.endswith(".weight")


def convert_regular_tensor(
    entry: TensorEntry,
    tensor: torch.Tensor,
    requested_type: gguf.GGMLQuantizationType | None,
    overrides: list[TypeOverride],
    quantize_scope: str,
    ineligible: str,
    helper: Path,
    helper_chunk_rows: int,
    quantizer: str,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None, gguf.GGMLQuantizationType]:
    override = matching_override(entry.logical_name, overrides)
    target = requested_type if override == "no-match" else override
    if target is None:
        data, raw_dtype = tensor_to_native_array(tensor)
        return data, raw_dtype, torch_dtype_to_native_type(tensor.dtype)

    source_is_float = tensor.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if not source_is_float:
        data, raw_dtype = tensor_to_native_array(tensor)
        return data, raw_dtype, torch_dtype_to_native_type(tensor.dtype)

    if not is_quantized_type(target):
        data = tensor_to_f32(tensor)
        out, raw_dtype = quantize_array(data, target, helper, helper_chunk_rows, quantizer)
        return out, raw_dtype, target

    if should_quantize_by_scope(entry, quantize_scope) and can_quantize(entry.shape, target):
        data = tensor_to_f32(tensor)
        out, raw_dtype = quantize_array(data, target, helper, helper_chunk_rows, quantizer)
        return out, raw_dtype, target

    if ineligible == "error":
        raise ValueError(f"tensor is not eligible for {target.name}: {entry.logical_name} shape={entry.shape}")
    if ineligible == "native":
        data, raw_dtype = tensor_to_native_array(tensor)
        return data, raw_dtype, torch_dtype_to_native_type(tensor.dtype)

    data = tensor_to_f32(tensor)
    fallback = parse_ggml_type(ineligible)
    if fallback is None:
        raise ValueError(f"invalid ineligible target: {ineligible}")
    out, raw_dtype = quantize_array(data, fallback, helper, helper_chunk_rows, quantizer)
    return out, raw_dtype, fallback


def convert_synthetic_tensor(
    entry: SyntheticTensor,
    requested_type: gguf.GGMLQuantizationType | None,
    overrides: list[TypeOverride],
    quantize_scope: str,
    ineligible: str,
    helper: Path,
    helper_chunk_rows: int,
    quantizer: str,
) -> SyntheticTensor:
    override = matching_override(entry.logical_name, overrides)
    target = requested_type if override == "no-match" else override
    if target is None:
        return entry
    if not is_quantized_type(target):
        out, raw_dtype = quantize_array(entry.data, target, helper, helper_chunk_rows, quantizer)
        return SyntheticTensor(entry.logical_name, entry.shape, out, raw_dtype, target)
    tensor_entry = TensorEntry(entry.logical_name, 0, entry.logical_name, entry.shape, torch.float32, False)
    if should_quantize_by_scope(tensor_entry, quantize_scope) and can_quantize(entry.shape, target):
        out, raw_dtype = quantize_array(entry.data, target, helper, helper_chunk_rows, quantizer)
        return SyntheticTensor(entry.logical_name, entry.shape, out, raw_dtype, target)
    if ineligible == "error":
        raise ValueError(f"synthetic tensor is not eligible for {target.name}: {entry.logical_name} shape={entry.shape}")
    if ineligible == "native":
        return entry
    fallback = parse_ggml_type(ineligible)
    if fallback is None:
        raise ValueError(f"invalid ineligible target: {ineligible}")
    out, raw_dtype = quantize_array(entry.data, fallback, helper, helper_chunk_rows, quantizer)
    return SyntheticTensor(entry.logical_name, entry.shape, out, raw_dtype, fallback)


def convert_bnb_tensor(
    entry: TensorEntry,
    path: Path,
    target: gguf.GGMLQuantizationType,
    helper: Path,
    helper_chunk_rows: int,
    quantizer: str,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None, gguf.GGMLQuantizationType, tuple[int, ...]]:
    values = load_bnb_nf4(path, entry.key)
    out, raw_dtype = quantize_array(values, target, helper, helper_chunk_rows, quantizer)
    return out, raw_dtype, target, tuple(int(dim) for dim in values.shape)


def h3_timestep_embedding(points: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    weight_in = tensors["time_embedder.proj_in.weight"].to(torch.float64)
    bias_in = tensors["time_embedder.proj_in.bias"].to(torch.float64)
    weight_out = tensors["time_embedder.proj_out.weight"].to(torch.float64)
    bias_out = tensors["time_embedder.proj_out.bias"].to(torch.float64)
    half = weight_in.shape[1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    args = points.to(torch.float64).view(-1, 1) * freqs.view(1, -1)
    features = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    hidden = torch.nn.functional.silu(features @ weight_in.transpose(0, 1) + bias_in)
    return hidden @ weight_out.transpose(0, 1) + bias_out


def h3_curve_grid(grid: int) -> torch.Tensor:
    if grid < 2:
        raise ValueError("H3 AdaLN curve grid must be at least 2")
    return torch.linspace(0.0, 1.0, grid, dtype=torch.float64)


def h3_fit_curve_basis(curve: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    grid, dim = curve.shape
    if not 1 <= rank <= min(grid, dim):
        raise ValueError(f"H3 AdaLN curve rank must be in [1, {min(grid, dim)}], got {rank}")
    _, _, vh = torch.linalg.svd(curve.to(torch.float64), full_matrices=False)
    basis = vh[:rank].contiguous()
    table = curve.to(torch.float64) @ basis.transpose(0, 1)
    return basis, table


def h3_adaln_curve_tensors(input_path: Path, grid: int, rank: int) -> tuple[list[SyntheticTensor], set[str]]:
    needed = {
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
    }
    with open_safetensors_compat(input_path) as handle:
        keys = set(handle.keys())
        tensors = {name: handle.get_tensor(name) for name in needed}
        curve = torch.nn.functional.silu(h3_timestep_embedding(h3_curve_grid(grid), tensors))
        basis, table = h3_fit_curve_basis(curve, rank)
        synthetic = [
            SyntheticTensor(
                "adaln_t_table",
                tuple(int(dim) for dim in table.shape),
                table.to(torch.float32).contiguous().numpy(),
                None,
                gguf.GGMLQuantizationType.F32,
            )
        ]
        exclude = set(needed)
        for key in handle.keys():
            if not key.endswith("adaln_proj.linear.weight"):
                continue
            if handle.get_slice(key).get_dtype() == "U8" and key + ".quant_state.bitsandbytes__nf4" in keys:
                weight = torch.from_numpy(load_bnb_nf4_from_handle(handle, key)).to(torch.float64)
            else:
                weight = handle.get_tensor(key).to(torch.float64)
            if weight.ndim != 2 or weight.shape[1] != basis.shape[1]:
                raise ValueError(f"H3 AdaLN weight shape does not match curve basis: {key}")
            projected = (weight @ basis.transpose(0, 1)).to(torch.float32).contiguous().numpy()
            synthetic.append(
                SyntheticTensor(
                    key,
                    tuple(int(dim) for dim in projected.shape),
                    projected,
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.F32,
                )
            )
            exclude.add(key)
    return synthetic, exclude


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone safetensors to audio.cpp-compatible GGUF converter.")
    parser.add_argument("--input", action="append", type=parse_input, required=True, help="[PREFIX=]PATH.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--arch", default="audiocpp")
    parser.add_argument("--type", default=None, type=parse_ggml_type, help="native, f16, bf16, q8_0, q4_k, ...")
    parser.add_argument("--bnb-nf4-type", default=None, type=parse_ggml_type)
    parser.add_argument("--override", action="append", type=parse_override, default=[], help="PATTERN=TYPE")
    parser.add_argument("--include", action="append", default=[], help="fnmatch pattern for logical tensor names")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--overlay-input", type=Path, default=None)
    parser.add_argument("--overlay-include", action="append", default=[], help="fnmatch tensor name pattern to take from overlay input")
    parser.add_argument("--quantize-scope", choices=("all", "weights-2d", "none"), default="all")
    parser.add_argument("--ineligible", choices=("error", "native", "f16", "bf16"), default="error")
    parser.add_argument("--quantizer", choices=("auto", "python", "ggml"), default="auto")
    parser.add_argument("--ggml-quantize-helper", type=Path, default=Path("build/debug/ggml-quantize-raw"))
    parser.add_argument("--helper-chunk-rows", type=int, default=256)
    parser.add_argument("--alias-max-len", type=int, default=GGML_MAX_NAME)
    parser.add_argument("--h3-adaln-curve-grid", type=int, default=0)
    parser.add_argument("--h3-adaln-curve-rank", type=int, default=64)
    parser.add_argument("--h3-video-vae-decode-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if any(spec.path.suffix != ".safetensors" for spec in args.input):
        raise ValueError("this standalone converter currently accepts safetensors inputs")
    if args.bnb_nf4_type is None:
        bnb_type = None
    else:
        bnb_type = args.bnb_nf4_type

    output = args.output.expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = source_tensors(args.input)
    key_sets: list[set[str]] = []
    for spec in args.input:
        with open_safetensors_compat(spec.path) as handle:
            key_sets.append(set(handle.keys()))

    synthetic: list[SyntheticTensor] = []
    synthetic_exclude: set[str] = set()
    if args.h3_adaln_curve_grid:
        if len(args.input) != 1 or args.input[0].prefix:
            raise ValueError("H3 AdaLN curve conversion requires one unprefixed DiT safetensors input")
        synthetic, synthetic_exclude = h3_adaln_curve_tensors(
            args.input[0].path,
            args.h3_adaln_curve_grid,
            args.h3_adaln_curve_rank,
        )

    overlay_keys: set[str] = set()
    if args.overlay_input is not None:
        if args.overlay_input.suffix != ".safetensors":
            raise ValueError("overlay input must be a safetensors file")
        with open_safetensors_compat(args.overlay_input) as handle:
            overlay_keys = set(handle.keys())
        if not args.overlay_include:
            raise ValueError("--overlay-input requires at least one --overlay-include pattern")

    selected: list[TensorEntry] = []
    overlay_sidecars: list[TensorEntry] = []
    selected_logical_names: set[str] = set()
    for entry in entries:
        if entry.logical_name in synthetic_exclude:
            continue
        if args.h3_video_vae_decode_only and not (
            entry.logical_name.startswith("decoder.") or entry.logical_name.startswith("post_quant_conv.")
        ):
            continue
        if not include_tensor(entry.logical_name, args.include, args.exclude_prefix):
            continue
        if bnb_type is not None and helper_base_name(key_sets[entry.source_index], entry.key) is not None:
            continue
        selected.append(entry)
        selected_logical_names.add(entry.logical_name)
        if args.overlay_input is not None and matches_any(entry.logical_name, args.overlay_include):
            if entry.logical_name not in overlay_keys:
                raise ValueError(f"overlay input is missing {entry.logical_name}")
            for suffix in (".weight_scale",):
                sidecar = entry.logical_name.removesuffix(".weight") + suffix
                if sidecar in overlay_keys and sidecar not in selected_logical_names:
                    with open_safetensors_compat(args.overlay_input) as handle:
                        tensor = handle.get_slice(sidecar)
                    overlay_sidecars.append(
                        TensorEntry(
                            logical_name=sidecar,
                            source_index=-1,
                            key=sidecar,
                            shape=tuple(int(dim) for dim in tensor.get_shape()),
                            dtype=safetensors_dtype_to_torch(tensor.get_dtype()),
                            is_bnb_nf4=False,
                        )
                    )
                    selected_logical_names.add(sidecar)
    if not selected and not synthetic:
        raise ValueError("no tensors selected for conversion")

    ordered_entries = [("source", entry) for entry in selected] + [("source", entry) for entry in overlay_sidecars] + [("synthetic", entry) for entry in synthetic]
    ordered_entries.sort(key=lambda item: item[1].logical_name)
    logical_names: list[str] = []
    logical_shapes: list[tuple[int, ...]] = []
    selected_names = [entry.logical_name for _, entry in ordered_entries]
    names = physical_names(selected_names, args.alias_max_len)

    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = gguf.GGUFWriter(tmp, args.arch, use_temp_file=True)

    try:
        writer.add_name(args.name or output.stem)
        writer.add_string("audiocpp.tensor_name_format", "native")
        writer.add_string("audiocpp.source_format", "safetensors" if len(args.input) == 1 else "packed")
        writer.add_string("audiocpp.weight_type", "orig" if args.type is None else args.type.name.lower())
        if bnb_type is not None:
            writer.add_string("audiocpp.bnb_nf4_type", bnb_type.name.lower())
        writer.add_array("audiocpp.tensor_sources.names", [spec.prefix.removesuffix(".") for spec in args.input])
        writer.add_array("audiocpp.tensor_sources.paths", [str(spec.path) for spec in args.input])
        writer.add_string("standalone_gguf_converter.version", "1")

        total_tensors = len(ordered_entries)
        for index, (kind, entry) in enumerate(ordered_entries):
            if kind == "source":
                use_overlay = args.overlay_input is not None and (
                    entry.source_index < 0 or matches_any(entry.logical_name, args.overlay_include)
                )
                spec = InputSpec("", args.overlay_input) if use_overlay else args.input[entry.source_index]
                if not use_overlay and entry.is_bnb_nf4 and bnb_type is not None:
                    override = matching_override(entry.logical_name, args.override)
                    target = bnb_type if override == "no-match" else override
                    if target is None:
                        raise ValueError(f"BNB NF4 tensor cannot use native output: {entry.logical_name}")
                    data, raw_dtype, stored_type, logical_shape = convert_bnb_tensor(
                        entry, spec.path, target, args.ggml_quantize_helper, args.helper_chunk_rows, args.quantizer
                    )
                else:
                    tensor = load_tensor(spec.path, entry.key)
                    if use_overlay:
                        entry = TensorEntry(
                            entry.logical_name,
                            entry.source_index,
                            entry.key,
                            tuple(int(dim) for dim in tensor.shape),
                            tensor.dtype,
                            False,
                        )
                    if args.h3_video_vae_decode_only and entry.logical_name == "post_quant_conv.weight":
                        if tuple(tensor.shape[-3:]) != (1, 1, 1):
                            raise ValueError("MiniMax-H3 video VAE post_quant_conv.weight must be a 1x1x1 convolution")
                        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1])
                        entry = TensorEntry(
                            entry.logical_name,
                            entry.source_index,
                            entry.key,
                            tuple(int(dim) for dim in tensor.shape),
                            entry.dtype,
                            entry.is_bnb_nf4,
                        )
                    data, raw_dtype, stored_type = convert_regular_tensor(
                        entry,
                        tensor,
                        args.type,
                        args.override,
                        args.quantize_scope,
                        args.ineligible,
                        args.ggml_quantize_helper,
                        args.helper_chunk_rows,
                        args.quantizer,
                    )
                    logical_shape = entry.shape
                logical_name = entry.logical_name
            else:
                converted = convert_synthetic_tensor(
                    entry,
                    args.type,
                    args.override,
                    args.quantize_scope,
                    args.ineligible,
                    args.ggml_quantize_helper,
                    args.helper_chunk_rows,
                    args.quantizer,
                )
                logical_name = converted.logical_name
                logical_shape = converted.shape
                data = converted.data
                raw_dtype = converted.raw_dtype
                stored_type = converted.stored_type
            logical_names.append(logical_name)
            logical_shapes.append(tuple(int(dim) for dim in logical_shape))
            writer.add_tensor(names[index], data, raw_dtype=raw_dtype)
            print(
                f"[{index + 1}/{total_tensors}] {logical_name} -> {stored_type.name} "
                f"shape={list(logical_shape)} bytes={data.nbytes}",
                flush=True,
            )

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
        writer.write_tensors_to_file(progress=args.progress)
        writer.close()
        os.replace(tmp, output)
    except Exception:
        writer.close()
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"gguf={output.resolve()}")
    print(f"tensors={len(selected) + len(synthetic)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"convert_dit_gguf.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
