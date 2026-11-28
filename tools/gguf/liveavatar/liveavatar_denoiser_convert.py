#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open


DEFAULT_DENOISER = "Wan2.2-S2V-14B-Q4_K_M.gguf"
BLOCK_ATTN_FFN_RE = re.compile(r"^blocks\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q|k|v|o)|ffn\.(?:0|2))\.weight$")
BLOCK_ATTN_RE = re.compile(r"^blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)\.weight$")
BLOCK_FFN_RE = re.compile(r"^blocks\.\d+\.ffn\.(?:0|2)\.weight$")


def gguf_scalar(reader: gguf.GGUFReader, key: str) -> int | str:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"template GGUF is missing field: {key}")
    value = field.parts[field.data[-1]]
    field_type = field.types[-1]
    if isinstance(value, bytes):
        if field_type == gguf.GGUFValueType.STRING:
            return value.decode("utf-8")
        return int.from_bytes(value, "little", signed=False)
    if isinstance(value, np.ndarray):
        if field_type == gguf.GGUFValueType.STRING:
            return value.tobytes().decode("utf-8")
        if value.ndim == 0:
            return int(value)
        return int.from_bytes(value.tobytes(), "little", signed=False)
    return value.item() if hasattr(value, "item") else value


def load_weight_map(index_path: Path) -> dict[str, Path]:
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    return {name: index_path.parent / shard for name, shard in index["weight_map"].items()}


class SafeTensorSet:
    def __init__(self, weight_map: dict[str, Path]) -> None:
        self._weight_map = weight_map
        self._local = threading.local()

    def close(self) -> None:
        handles = getattr(self._local, "handles", {})
        for handle in handles.values():
            handle.__exit__(None, None, None)
        handles.clear()

    def _handles(self) -> dict[Path, object]:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
        return handles

    def shape(self, name: str) -> tuple[int, ...]:
        path = self._weight_map[name]
        handles = self._handles()
        handle = handles.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            handle.__enter__()
            handles[path] = handle
        return tuple(int(dim) for dim in handle.get_slice(name).get_shape())

    def get(self, name: str) -> torch.Tensor:
        path = self._weight_map[name]
        handles = self._handles()
        handle = handles.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            handle.__enter__()
            handles[path] = handle
        return handle.get_tensor(name)


class LoRATensors:
    def __init__(self, path: Path | None, lora_alpha: float | None, lora_rank: int | None) -> None:
        self._path = path
        self._lora_alpha = lora_alpha
        self._lora_rank = lora_rank
        self._local = threading.local()
        self._main_handle = None
        self.pairs: dict[str, tuple[str, str, float]] = {}
        if path is not None:
            self._main_handle = safe_open(str(path), framework="pt", device="cpu")
            self._main_handle.__enter__()
            self.pairs = self._load_pairs()

    def close(self) -> None:
        if self._main_handle is not None:
            self._main_handle.__exit__(None, None, None)
            self._main_handle = None
        handle = getattr(self._local, "handle", None)
        if handle is not None:
            handle.__exit__(None, None, None)
            self._local.handle = None

    def _handle(self):
        if self._path is None:
            return None
        handle = getattr(self._local, "handle", None)
        if handle is None:
            handle = safe_open(str(self._path), framework="pt", device="cpu")
            handle.__enter__()
            self._local.handle = handle
        return handle

    def _load_pairs(self) -> dict[str, tuple[str, str, float]]:
        pairs: dict[str, dict[str, str]] = defaultdict(dict)
        for key in self._main_handle.keys():
            if key.endswith(".lora_A.default.weight"):
                pairs[key.removesuffix(".lora_A.default.weight") + ".weight"]["down"] = key
            elif key.endswith(".lora_B.default.weight"):
                pairs[key.removesuffix(".lora_B.default.weight") + ".weight"]["up"] = key
            elif key.endswith(".alpha"):
                pairs[key.removesuffix(".alpha") + ".weight"]["alpha"] = key
            else:
                raise ValueError(f"unsupported LoRA tensor key: {key}")

        out: dict[str, tuple[str, str, float]] = {}
        for base, values in pairs.items():
            if "down" not in values or "up" not in values:
                raise ValueError(f"incomplete LoRA pair for {base}")
            down_shape = tuple(self._main_handle.get_slice(values["down"]).get_shape())
            up_shape = tuple(self._main_handle.get_slice(values["up"]).get_shape())
            if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1]:
                raise ValueError(f"unsupported LoRA shapes for {base}: A={down_shape} B={up_shape}")
            rank = int(down_shape[0])
            if self._lora_rank is not None and self._lora_rank != rank:
                raise ValueError(f"LoRA rank mismatch for {base}: expected {self._lora_rank}, file has {rank}")
            if "alpha" in values:
                scale = float(self._main_handle.get_tensor(values["alpha"]).item()) / float(rank)
            else:
                if self._lora_alpha is None:
                    raise ValueError("LoRA file has no alpha tensors; pass --lora-alpha to match the training config")
                scale = float(self._lora_alpha) / float(rank)
            out[base] = (values["down"], values["up"], scale)
        return out

    def merge(self, name: str, base: torch.Tensor) -> torch.Tensor:
        if name not in self.pairs:
            return base
        handle = self._handle()
        down_key, up_key, scale = self.pairs[name]
        down = handle.get_tensor(down_key).to(torch.float32)
        up = handle.get_tensor(up_key).to(torch.float32)
        delta = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(base.shape)
        merged = base.to(torch.float32)
        merged.add_(delta, alpha=scale)
        return merged


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


def quantize_with_helper(
    values: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    quant_module,
) -> np.ndarray:
    data = values.to(torch.float32).contiguous().numpy()
    if not is_quantized_type(qtype):
        return gguf.quants.quantize(data, qtype)
    return quant_module.quantize(data, int(qtype))


def raw_dtype_for_type(qtype: gguf.GGMLQuantizationType) -> np.dtype:
    if is_quantized_type(qtype):
        return np.dtype(np.uint8)
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.dtype(np.float32)
    if qtype == gguf.GGMLQuantizationType.F16:
        return np.dtype(np.float16)
    if qtype == gguf.GGMLQuantizationType.BF16:
        return np.dtype(np.uint16)
    if qtype == gguf.GGMLQuantizationType.I8:
        return np.dtype(np.int8)
    if qtype == gguf.GGMLQuantizationType.I16:
        return np.dtype(np.int16)
    if qtype == gguf.GGMLQuantizationType.I32:
        return np.dtype(np.int32)
    if qtype == gguf.GGMLQuantizationType.I64:
        return np.dtype(np.int64)
    if qtype == gguf.GGMLQuantizationType.F64:
        return np.dtype(np.float64)
    raise NotImplementedError(f"unsupported tensor type: {qtype.name}")


def tensor_info_for_shape(
    shape: tuple[int, ...],
    qtype: gguf.GGMLQuantizationType,
) -> tuple[tuple[int, ...], np.dtype, int]:
    if is_quantized_type(qtype):
        byte_shape = tuple(int(dim) for dim in gguf.quants.quant_shape_to_byte_shape(shape, qtype))
        return byte_shape, np.dtype(np.uint8), int(np.prod(byte_shape))
    dtype = raw_dtype_for_type(qtype)
    return shape, dtype, int(np.prod(shape) * dtype.itemsize)


def load_quant_module(module_dir: Path):
    sys.path.insert(0, str(module_dir))
    try:
        return importlib.import_module("_audiocpp_ggml_quant")
    except Exception as error:
        raise RuntimeError(
            f"failed to import _audiocpp_ggml_quant from {module_dir}; "
            "build it with: cmake --build build/debug --target _audiocpp_ggml_quant -j$(nproc)"
        ) from error


def parse_ggml_type(name: str | None) -> gguf.GGMLQuantizationType | None:
    if name is None:
        return None
    normalized = name.upper()
    if not hasattr(gguf.GGMLQuantizationType, normalized):
        valid = ", ".join(sorted(item.name for item in gguf.GGMLQuantizationType))
        raise ValueError(f"unknown GGML quantization type {name!r}; valid types: {valid}")
    return getattr(gguf.GGMLQuantizationType, normalized)


def convert_tensor(
    name: str,
    qtype: gguf.GGMLQuantizationType,
    sources: SafeTensorSet,
    lora: LoRATensors,
    quant_module,
) -> np.ndarray:
    base = sources.get(name)
    merged = lora.merge(name, base)
    values = native_values(name, merged)
    return quantize_with_helper(values, qtype, quant_module)


def gguf_reader_shape_to_cpp_shape(tensor: gguf.ReaderTensor) -> tuple[int, ...]:
    return tuple(reversed(tuple(int(dim) for dim in tensor.shape)))


def native_shape(name: str, source_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(source_shape) == 5:
        out_channels, in_channels, kernel_depth, kernel_height, kernel_width = source_shape
        return (out_channels * in_channels, kernel_depth, kernel_height, kernel_width)
    return source_shape


def native_values(name: str, values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 5:
        return values.contiguous().reshape(
            values.shape[0] * values.shape[1],
            values.shape[2],
            values.shape[3],
            values.shape[4],
        )
    return values.contiguous()


def validate_template_policy(
    template_tensors: list[gguf.ReaderTensor],
    weight_map: dict[str, Path],
    lora: LoRATensors,
) -> None:
    template_names = {tensor.name for tensor in template_tensors}
    missing_base = sorted(template_names - set(weight_map))
    if missing_base:
        raise ValueError(f"template tensors missing from official base shards: {missing_base[:8]}")
    missing_template = sorted(set(lora.pairs) - template_names)
    if missing_template:
        raise ValueError(f"LoRA targets missing from template GGUF: {missing_template[:8]}")
    missing_base_lora = sorted(set(lora.pairs) - set(weight_map))
    if missing_base_lora:
        raise ValueError(f"LoRA targets missing from official base shards: {missing_base_lora[:8]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LiveAvatar official denoiser safetensors to an audio.cpp-native GGUF.")
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--template-gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument(
        "--quantized-tensor-type",
        help="Override every quantized template tensor, e.g. NVFP4.",
    )
    parser.add_argument("--quant-module-dir", type=Path, default=Path("build/debug/bin"))
    parser.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--q8-block-attn-ffn",
        action="store_true",
        help="Store repeated denoiser block attention and FFN weights as Q8_0 while keeping the template policy for all other tensors.",
    )
    parser.add_argument(
        "--block-attn-type",
        help="Override repeated denoiser block self/cross attention projection weights that are Q4_K in the template, e.g. Q4_0 or NVFP4.",
    )
    parser.add_argument(
        "--block-ffn-type",
        help="Override repeated denoiser block FFN projection weights that are Q4_K in the template, e.g. Q4_0 or NVFP4.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    block_attn_type = parse_ggml_type(args.block_attn_type)
    block_ffn_type = parse_ggml_type(args.block_ffn_type)
    quantized_tensor_type = parse_ggml_type(args.quantized_tensor_type)
    if args.q8_block_attn_ffn and (block_attn_type is not None or block_ffn_type is not None or quantized_tensor_type is not None):
        raise ValueError("--q8-block-attn-ffn cannot be combined with quantization override options")

    weight_map = load_weight_map(args.base_index)
    template = gguf.GGUFReader(str(args.template_gguf))
    lora = LoRATensors(args.lora, args.lora_alpha, args.lora_rank)
    sources = SafeTensorSet(weight_map)
    quant_module = load_quant_module(args.quant_module_dir)
    tmp = args.output.with_name(args.output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None

    try:
        validate_template_policy(list(template.tensors), weight_map, lora)
        arch = str(gguf_scalar(template, "general.architecture"))
        qversion = int(gguf_scalar(template, "general.quantization_version"))
        file_type = gguf.LlamaFileType(int(gguf_scalar(template, "general.file_type")))

        writer = gguf.GGUFWriter(tmp, arch, use_temp_file=True)
        writer.add_quantization_version(qversion)
        writer.add_file_type(file_type)
        tensor_infos: list[tuple[str, tuple[int, ...], gguf.GGMLQuantizationType, np.dtype, int]] = []
        exact_names: list[str] = []
        exact_ranks: list[int] = []
        exact_shapes: list[int] = []

        print(f"source tensors: {len(weight_map)}", flush=True)
        print(f"template policy tensors: {len(template.tensors)}", flush=True)
        print(f"lora targets: {len(lora.pairs)}", flush=True)
        if args.lora is not None:
            print(f"lora alpha: {args.lora_alpha}", flush=True)
            print(f"lora rank override: {args.lora_rank}", flush=True)
        print(f"jobs: {args.jobs}", flush=True)
        print(f"torch threads per job: {args.torch_threads}", flush=True)
        print("layout: audio.cpp native, 5D conv3d tensors folded to 4D", flush=True)

        override_counts = defaultdict(int)
        for tensor in template.tensors:
            base_shape = sources.shape(tensor.name)
            cpp_template_shape = gguf_reader_shape_to_cpp_shape(tensor)
            shape = native_shape(tensor.name, base_shape)
            if shape != native_shape(tensor.name, cpp_template_shape):
                raise ValueError(
                    f"shape policy mismatch for {tensor.name}: base={base_shape} "
                    f"template={cpp_template_shape} native={shape}")
            qtype = tensor.tensor_type
            if quantized_tensor_type is not None and is_quantized_type(qtype):
                qtype = quantized_tensor_type
                override_counts["quantized_tensor"] += 1
            if args.q8_block_attn_ffn and BLOCK_ATTN_FFN_RE.match(tensor.name):
                qtype = gguf.GGMLQuantizationType.Q8_0
                override_counts["q8_block_attn_ffn"] += 1
            if block_attn_type is not None and tensor.tensor_type == gguf.GGMLQuantizationType.Q4_K and BLOCK_ATTN_RE.match(tensor.name):
                qtype = block_attn_type
                override_counts["block_attn"] += 1
            if block_ffn_type is not None and tensor.tensor_type == gguf.GGMLQuantizationType.Q4_K and BLOCK_FFN_RE.match(tensor.name):
                qtype = block_ffn_type
                override_counts["block_ffn"] += 1
            writer_shape, dtype, nbytes = tensor_info_for_shape(shape, qtype)
            tensor_infos.append((tensor.name, writer_shape, qtype, dtype, nbytes))
            exact_names.append(tensor.name)
            exact_ranks.append(len(shape))
            exact_shapes.extend(shape)
        for name, count in sorted(override_counts.items()):
            print(f"{name} overrides: {count}", flush=True)

        writer.add_key_value("audiocpp.tensor_names", exact_names, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING)
        writer.add_key_value(
            "audiocpp.tensor_ranks",
            exact_ranks,
            gguf.GGUFValueType.ARRAY,
            gguf.GGUFValueType.INT32,
        )
        writer.add_key_value(
            "audiocpp.tensor_shapes",
            exact_shapes,
            gguf.GGUFValueType.ARRAY,
            gguf.GGUFValueType.INT64,
        )
        for name, writer_shape, qtype, dtype, nbytes in tensor_infos:
            writer.add_tensor_info(name, writer_shape, dtype, nbytes, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        jobs = [(name, qtype) for name, _, qtype, _, _ in tensor_infos]
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            in_flight: dict[int, Future[np.ndarray]] = {}
            next_submit = 0
            next_write = 0

            def submit_more() -> None:
                nonlocal next_submit
                while next_submit < len(jobs) and len(in_flight) < args.jobs:
                    name, qtype = jobs[next_submit]
                    in_flight[next_submit] = executor.submit(
                        convert_tensor,
                        name,
                        qtype,
                        sources,
                        lora,
                        quant_module,
                    )
                    next_submit += 1

            submit_more()
            while next_write < len(jobs):
                name, qtype = jobs[next_write]
                start = time.perf_counter()
                data = in_flight.pop(next_write).result()
                writer.write_tensor_data(data)
                print(
                    f"[{next_write + 1:04d}/{len(jobs):04d}] write {name} "
                    f"type={qtype.name} bytes={data.nbytes} seconds={time.perf_counter() - start:.3f}",
                    flush=True,
                )
                next_write += 1
                submit_more()

        writer.close()
        os.replace(tmp, args.output)
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        sources.close()
        lora.close()

    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
