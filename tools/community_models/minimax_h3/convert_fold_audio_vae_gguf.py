#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gguf
import numpy as np


def array_strings(reader: gguf.GGUFReader, key: str) -> list[str]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    out: list[str] = []
    index = 5
    for _ in range(count):
        size = int(parts[index][0])
        out.append(bytes(parts[index + 1][:size]).decode("utf-8"))
        index += 2
    return out


def array_i32(reader: gguf.GGUFReader, key: str) -> list[int]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    return [int(parts[5 + index][0]) for index in range(count)]


def array_i64(reader: gguf.GGUFReader, key: str) -> list[int]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    return [int(parts[5 + index][0]) for index in range(count)]


def logical_shapes(reader: gguf.GGUFReader) -> dict[str, tuple[int, ...]]:
    names = array_strings(reader, "audiocpp.tensor_names")
    ranks = array_i32(reader, "audiocpp.tensor_ranks")
    flat_shapes = array_i64(reader, "audiocpp.tensor_shapes")
    if len(names) != len(ranks):
        raise ValueError("GGUF tensor name/rank metadata length mismatch")
    out: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, rank in zip(names, ranks):
        shape = tuple(flat_shapes[offset:offset + rank])
        if len(shape) != rank:
            raise ValueError("GGUF tensor shape metadata is truncated")
        out[name] = shape
        offset += rank
    if offset != len(flat_shapes):
        raise ValueError("GGUF tensor shape metadata has trailing dimensions")
    return out


def dequant_f32(tensor: gguf.ReaderTensor, shape: tuple[int, ...]) -> np.ndarray:
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False).reshape(shape)


def fold_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray) -> np.ndarray:
    if weight_v.ndim != 3 or weight_g.ndim != 3 or weight_g.shape[1:] != (1, 1):
        raise ValueError(f"unexpected weight-norm shapes: weight_v={weight_v.shape}, weight_g={weight_g.shape}")
    if weight_v.shape[0] != weight_g.shape[0]:
        raise ValueError(f"weight-norm output dimension mismatch: weight_v={weight_v.shape}, weight_g={weight_g.shape}")
    flat = weight_v.reshape(weight_v.shape[0], -1)
    scale = weight_g.reshape(weight_g.shape[0]) / np.sqrt(np.sum(flat * flat, axis=1, dtype=np.float64)).astype(np.float32)
    return (weight_v * scale[:, None, None]).astype(np.float32, copy=False)


def tensor_map(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    tensors: dict[str, gguf.ReaderTensor] = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise ValueError(f"duplicate tensor name in source GGUF: {tensor.name}")
        tensors[tensor.name] = tensor
    return tensors


def folded_name(name: str) -> str | None:
    if name.startswith("decoder.") and name.endswith(".weight_v"):
        return name[:-len("_v")]
    return None


def copy_tensor(
    writer: gguf.GGUFWriter,
    tensor: gguf.ReaderTensor,
    shape: tuple[int, ...],
) -> tuple[str, tuple[int, ...]]:
    raw_dtype = tensor.tensor_type if tensor.data.dtype == np.uint8 else None
    writer.add_tensor(tensor.name, np.array(tensor.data, copy=True), raw_dtype=raw_dtype)
    return tensor.name, shape


def add_folded_tensor(
    writer: gguf.GGUFWriter,
    name: str,
    values: np.ndarray,
    folded_type: str,
) -> tuple[str, tuple[int, ...]]:
    if folded_type == "f16":
        data = np.ascontiguousarray(values, dtype=np.float16)
    elif folded_type == "f32":
        data = np.ascontiguousarray(values, dtype=np.float32)
    else:
        raise ValueError(f"unsupported folded tensor type: {folded_type}")
    shape = tuple(int(dim) for dim in values.shape)
    writer.add_tensor(name, data, raw_shape=shape)
    return name, tuple(int(dim) for dim in data.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold MiniMax-H3 Audio VAE BigVGAN weight-norm tensors into direct GGUF weights.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folded-type", choices=("f32", "f16"), default="f32")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = args.input.expanduser()
    output_path = args.output.expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(source_path)
    tensors = tensor_map(reader)
    shapes = logical_shapes(reader)
    tmp = output_path.with_name(output_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    writer = gguf.GGUFWriter(tmp, "audiocpp", use_temp_file=True)
    logical_names: list[str] = []
    output_shapes: list[tuple[int, ...]] = []
    skipped: set[str] = set()
    folded_count = 0
    try:
        writer.add_name(output_path.stem)
        writer.add_string("audiocpp.tensor_name_format", "native")
        writer.add_string("audiocpp.source_format", "gguf_folded")
        writer.add_string("audiocpp.weight_type", f"folded_{args.folded_type}")
        writer.add_string("audiocpp.folded_from", str(source_path))
        writer.add_string("standalone_gguf_converter.version", "1")
        writer.add_string("minimax_h3.audio_vae.bigvgan_weight_norm", "folded")

        for tensor in sorted(reader.tensors, key=lambda item: item.name):
            if tensor.name in skipped:
                continue
            direct_name = folded_name(tensor.name)
            if direct_name is not None:
                g_name = direct_name + "_g"
                if g_name not in tensors:
                    raise ValueError(f"missing weight_g pair for {tensor.name}")
                logical_name, shape = add_folded_tensor(
                    writer,
                    direct_name,
                    fold_weight_norm(dequant_f32(tensor, shapes[tensor.name]), dequant_f32(tensors[g_name], shapes[g_name])),
                    args.folded_type,
                )
                skipped.add(g_name)
                folded_count += 1
            elif tensor.name.startswith("decoder.") and tensor.name.endswith(".weight_g"):
                continue
            else:
                logical_name, shape = copy_tensor(writer, tensor, shapes[tensor.name])
            logical_names.append(logical_name)
            output_shapes.append(shape)

        writer.add_array("audiocpp.tensor_names", logical_names)
        writer.add_key_value(
            "audiocpp.tensor_ranks",
            [len(shape) for shape in output_shapes],
            gguf.GGUFValueType.ARRAY,
            sub_type=gguf.GGUFValueType.INT32,
        )
        writer.add_key_value(
            "audiocpp.tensor_shapes",
            [dim for shape in output_shapes for dim in shape],
            gguf.GGUFValueType.ARRAY,
            sub_type=gguf.GGUFValueType.INT64,
        )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(tmp, output_path)
    except Exception:
        writer.close()
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"gguf={output_path.resolve()}")
    print(f"folded_bigvgan_tensors={folded_count}")
    print(f"tensors={len(logical_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
