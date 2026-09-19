#!/usr/bin/env python3
"""Build audio.cpp-native LiveAvatar component GGUFs.

The DiT denoiser stays separate so different denoiser quantizations can be
swapped independently. This script creates the smaller runtime components:
support GGUF (text encoder + audio encoder + tokenizer sidecar) and VAE GGUF.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

from convert_denoiser_gguf import (
    is_quantized_type,
    load_quant_module,
    native_values,
    parse_ggml_type,
    raw_dtype_for_type,
    tensor_info_for_shape,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/home/leo/Desktop/Wan2.2-S2V")
DEFAULT_SUPPORT_OUTPUT = DEFAULT_ROOT / "Wan2.2-S2V-Support-Q4_K_S-F16.gguf"
DEFAULT_VAE_OUTPUT = DEFAULT_ROOT / "Wan2.2-S2V-VAE-F16.gguf"
GGML_MAX_NAME = 64


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--audio-encoder", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--support-output", type=Path, default=DEFAULT_SUPPORT_OUTPUT)
    parser.add_argument("--vae-output", type=Path, default=DEFAULT_VAE_OUTPUT)
    parser.add_argument("--support-type", default="orig")
    parser.add_argument("--vae-type", default="orig")
    parser.add_argument("--quant-module-dir", type=Path, default=REPO_ROOT / "build/debug/bin")
    parser.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--skip-support", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_tensor(values: torch.Tensor, qtype: gguf.GGMLQuantizationType, quant_module) -> np.ndarray:
    values = native_values("", values)
    if is_quantized_type(qtype):
        return quant_module.quantize(values.to(torch.float32).contiguous().numpy(), int(qtype))
    if qtype == gguf.GGMLQuantizationType.F32:
        return values.to(torch.float32).contiguous().numpy()
    if qtype == gguf.GGMLQuantizationType.F16:
        return values.to(torch.float16).contiguous().numpy()
    if qtype == gguf.GGMLQuantizationType.BF16:
        return values.to(torch.bfloat16).view(torch.uint16).contiguous().numpy()
    dtype = raw_dtype_for_type(qtype)
    return values.to(torch.float32).contiguous().numpy().astype(dtype)


def encode_safetensor_tensor(path: Path, name: str, qtype: gguf.GGMLQuantizationType, quant_module) -> np.ndarray:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return encode_tensor(handle.get_tensor(name), qtype, quant_module)


def parse_vae_type(value: str) -> gguf.GGMLQuantizationType:
    if value in {"orig", "native"}:
        return gguf.GGMLQuantizationType.F16
    parsed = parse_ggml_type(value)
    if parsed is None:
        return gguf.GGMLQuantizationType.F16
    return parsed


def read_model_spec() -> tuple[str, str]:
    model_spec_json = (REPO_ROOT / "model_specs/liveavatar.json").read_text(encoding="utf-8")
    return json.loads(model_spec_json)["family"], model_spec_json


def add_audiocpp_metadata(
    writer: gguf.GGUFWriter,
    model_spec_family: str,
    model_spec_json: str,
    exact_names: list[str],
    exact_ranks: list[int],
    exact_shapes: list[int],
) -> None:
    writer.add_quantization_version(2)
    writer.add_uint32("audiocpp.model_spec.version", 1)
    writer.add_string("audiocpp.model_spec.family", model_spec_family)
    writer.add_string("audiocpp.model_spec.json", model_spec_json)
    writer.add_key_value("audiocpp.tensor_names", exact_names, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING)
    writer.add_key_value("audiocpp.tensor_ranks", exact_ranks, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    writer.add_key_value("audiocpp.tensor_shapes", exact_shapes, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT64)


def add_tokenizer_sidecar(writer: gguf.GGUFWriter, tokenizer: Path) -> None:
    payload = tokenizer.read_bytes()
    writer.add_key_value(
        "audiocpp.embedded_files.names",
        ["tokenizer/spiece.model"],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.STRING,
    )
    writer.add_key_value(
        "audiocpp.embedded_files.offsets",
        [0, len(payload)],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT64,
    )
    writer.add_key_value(
        "audiocpp.embedded_files.data",
        payload,
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT8,
    )


def choose_physical_name(logical_name: str, index: int, physical_names: set[str]) -> str:
    physical_name = logical_name
    if len(physical_name) >= GGML_MAX_NAME or physical_name in physical_names:
        physical_name = f"_audiocpp.{index}"
    if physical_name in physical_names:
        raise ValueError(f"failed to create a unique GGUF tensor alias for {logical_name}")
    physical_names.add(physical_name)
    return physical_name


def convert_support_gguf(args: argparse.Namespace, text_encoder: Path, audio_encoder: Path, tokenizer: Path) -> None:
    if args.support_output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.support_output}")
    if args.support_type not in {"orig", "native"}:
        raise ValueError("support GGUF conversion currently preserves source tensor types; use --support-type orig")
    args.support_output.parent.mkdir(parents=True, exist_ok=True)
    quant_module = load_quant_module(args.quant_module_dir)
    model_spec_family, model_spec_json = read_model_spec()
    tmp = args.support_output.with_name(args.support_output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    text_reader = gguf.GGUFReader(str(text_encoder))
    text_tensors = sorted(text_reader.tensors, key=lambda tensor: tensor.name)
    with safe_open(str(audio_encoder), framework="pt", device="cpu") as handle:
        audio_names = sorted(handle.keys())
        audio_shapes = {name: tuple(int(dim) for dim in handle.get_slice(name).get_shape()) for name in audio_names}

    tensor_infos = []
    exact_names: list[str] = []
    exact_ranks: list[int] = []
    exact_shapes: list[int] = []
    physical_names: set[str] = set()
    for tensor in text_tensors:
        tensor_name = f"text_encoder/{tensor.name}"
        physical_name = choose_physical_name(tensor_name, len(tensor_infos), physical_names)
        data = tensor.data
        writer_shape = tuple(int(dim) for dim in data.shape) if data.dtype == np.uint8 else tuple(int(dim) for dim in tensor.shape)
        qtype = tensor.tensor_type
        tensor_infos.append(("text", tensor, tensor_name, physical_name, writer_shape, qtype, data.dtype, data.nbytes))
        native_shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
        exact_names.append(tensor_name)
        exact_ranks.append(len(native_shape))
        exact_shapes.extend(native_shape)
    for name in audio_names:
        native_shape = audio_shapes[name]
        tensor_name = f"audio_encoder/{name}"
        physical_name = choose_physical_name(tensor_name, len(tensor_infos), physical_names)
        qtype = gguf.GGMLQuantizationType.F16
        writer_shape, dtype, nbytes = tensor_info_for_shape(native_shape, qtype)
        tensor_infos.append(("audio", name, tensor_name, physical_name, writer_shape, qtype, dtype, nbytes))
        exact_names.append(tensor_name)
        exact_ranks.append(len(native_shape))
        exact_shapes.extend(native_shape)

    writer = None
    try:
        writer = gguf.GGUFWriter(tmp, "liveavatar", use_temp_file=True)
        add_audiocpp_metadata(writer, model_spec_family, model_spec_json, exact_names, exact_ranks, exact_shapes)
        add_tokenizer_sidecar(writer, tokenizer)
        for _, _, _, physical_name, writer_shape, qtype, dtype, nbytes in tensor_infos:
            writer.add_tensor_info(physical_name, writer_shape, dtype, nbytes, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            in_flight: dict[int, Future[np.ndarray]] = {}
            next_submit = 0
            next_write = 0

            def submit_more() -> None:
                nonlocal next_submit
                while next_submit < len(tensor_infos) and len(in_flight) < args.jobs:
                    kind, source, _, _, _, qtype, _, _ = tensor_infos[next_submit]
                    if kind == "text":
                        in_flight[next_submit] = executor.submit(np.ascontiguousarray, source.data)
                    else:
                        in_flight[next_submit] = executor.submit(encode_safetensor_tensor, audio_encoder, source, qtype, quant_module)
                    next_submit += 1

            submit_more()
            while next_write < len(tensor_infos):
                _, _, tensor_name, _, _, qtype, _, _ = tensor_infos[next_write]
                start = time.perf_counter()
                data = in_flight.pop(next_write).result()
                writer.write_tensor_data(data)
                print(
                    f"[{next_write + 1:04d}/{len(tensor_infos):04d}] write {tensor_name} "
                    f"type={qtype.name} bytes={data.nbytes} seconds={time.perf_counter() - start:.3f}",
                    flush=True,
                )
                next_write += 1
                submit_more()
        writer.close()
        os.replace(tmp, args.support_output)
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"wrote {args.support_output}", flush=True)


def convert_vae_gguf(args: argparse.Namespace, vae: Path) -> None:
    if args.vae_output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.vae_output}")
    args.vae_output.parent.mkdir(parents=True, exist_ok=True)
    qtype = parse_vae_type(args.vae_type)
    quant_module = load_quant_module(args.quant_module_dir)
    model_spec_family, model_spec_json = read_model_spec()
    tmp = args.vae_output.with_name(args.vae_output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with safe_open(str(vae), framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        shapes = {name: tuple(int(dim) for dim in handle.get_slice(name).get_shape()) for name in names}

    tensor_infos = []
    exact_names: list[str] = []
    exact_ranks: list[int] = []
    exact_shapes: list[int] = []
    for name in names:
        source_shape = shapes[name]
        native_shape = (
            (source_shape[0] * source_shape[1], source_shape[2], source_shape[3], source_shape[4])
            if len(source_shape) == 5
            else source_shape
        )
        tensor_name = f"vae/{name}"
        writer_shape, dtype, nbytes = tensor_info_for_shape(native_shape, qtype)
        tensor_infos.append((name, tensor_name, writer_shape, qtype, dtype, nbytes))
        exact_names.append(tensor_name)
        exact_ranks.append(len(native_shape))
        exact_shapes.extend(native_shape)

    writer = None
    try:
        writer = gguf.GGUFWriter(tmp, "liveavatar", use_temp_file=True)
        add_audiocpp_metadata(writer, model_spec_family, model_spec_json, exact_names, exact_ranks, exact_shapes)
        for _, tensor_name, writer_shape, qtype, dtype, nbytes in tensor_infos:
            writer.add_tensor_info(tensor_name, writer_shape, dtype, nbytes, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            in_flight: dict[int, Future[np.ndarray]] = {}
            next_submit = 0
            next_write = 0

            def submit_more() -> None:
                nonlocal next_submit
                while next_submit < len(tensor_infos) and len(in_flight) < args.jobs:
                    name, _, _, qtype, _, _ = tensor_infos[next_submit]
                    in_flight[next_submit] = executor.submit(encode_safetensor_tensor, vae, name, qtype, quant_module)
                    next_submit += 1

            submit_more()
            while next_write < len(tensor_infos):
                name, tensor_name, _, qtype, _, _ = tensor_infos[next_write]
                start = time.perf_counter()
                data = in_flight.pop(next_write).result()
                writer.write_tensor_data(data)
                print(
                    f"[{next_write + 1:04d}/{len(tensor_infos):04d}] write {tensor_name} "
                    f"type={qtype.name} bytes={data.nbytes} seconds={time.perf_counter() - start:.3f}",
                    flush=True,
                )
                next_write += 1
                submit_more()
        writer.close()
        os.replace(tmp, args.vae_output)
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"wrote {args.vae_output}", flush=True)


def main() -> None:
    args = parse_args()
    root = args.root
    text_encoder = require_file(
        args.text_encoder or root / "models/text_encoders/Wan/umt5-xxl-encoder-Q4_K_S.gguf",
        "LiveAvatar text encoder",
    )
    audio_encoder = require_file(
        args.audio_encoder or root / "_hf/Comfy-Org-Wan2.2/split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
        "LiveAvatar audio encoder",
    )
    vae = require_file(
        args.vae or root / "_hf/Comfy-Org-Wan2.2/split_files/vae/wan_2.1_vae.safetensors",
        "LiveAvatar VAE",
    )
    tokenizer = require_file(
        args.tokenizer or root / "models/text_encoders/Wan/google-umt5-xxl/spiece.model",
        "LiveAvatar SentencePiece tokenizer",
    )
    if not args.skip_support:
        convert_support_gguf(args, text_encoder, audio_encoder, tokenizer)
    if not args.skip_vae:
        convert_vae_gguf(args, vae)


if __name__ == "__main__":
    main()
