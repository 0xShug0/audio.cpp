#!/usr/bin/env python3
"""Repack a MiniMax Music 3 GGUF with an exact compact semantic LM head.

The operation never dequantizes the head. It copies the already-quantized rows
used by the native MiniMax semantic sampler and leaves every other tensor byte
sequence unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


LM_HEAD_NAME = "lm_head.weight"
SOURCE_VOCAB_SIZE = 200_000
HIDDEN_SIZE = 4_096
SEMANTIC_OFFSET = 151_675
SEMANTIC_VOCAB_SIZE = 16_384
EOS_TOKEN_ID = 151_670
LAYOUT = "semantic_compact_v1"


def compact_token_ids() -> list[int]:
    return list(range(SEMANTIC_OFFSET, SEMANTIC_OFFSET + SEMANTIC_VOCAB_SIZE)) + [EOS_TOKEN_ID]


def compact_lm_head_rows(source: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if source.ndim != 2 or source.shape[0] != SOURCE_VOCAB_SIZE:
        raise ValueError(
            f"MiniMax Music 3 full lm_head must have {SOURCE_VOCAB_SIZE} rows; got {source.shape}"
        )
    return np.ascontiguousarray(source[compact_token_ids()])


def compact_logical_shapes(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    if shapes.get(LM_HEAD_NAME) != (SOURCE_VOCAB_SIZE, HIDDEN_SIZE):
        raise ValueError(
            f"expected logical {LM_HEAD_NAME} shape {(SOURCE_VOCAB_SIZE, HIDDEN_SIZE)}; "
            f"got {shapes.get(LM_HEAD_NAME)}"
        )
    out = dict(shapes)
    out[LM_HEAD_NAME] = (SEMANTIC_VOCAB_SIZE + 1, HIDDEN_SIZE)
    return out


def _array_strings(reader: Any, key: str) -> list[str]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    out: list[str] = []
    index = 5
    for _ in range(count):
        size = int(parts[index][0])
        out.append(bytes(parts[index + 1][:size]).decode("utf-8"))
        index += 2
    return out


def _array_i32(reader: Any, key: str) -> list[int]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    return [int(parts[5 + index][0]) for index in range(count)]


def _array_i64(reader: Any, key: str) -> list[int]:
    parts = reader.fields[key].parts
    count = int(parts[4][0])
    return [int(parts[5 + index][0]) for index in range(count)]


def _field_string(reader: Any, key: str) -> str:
    field = reader.fields[key]
    size = int(field.parts[-2][0])
    return bytes(field.parts[-1][:size]).decode("utf-8")


def _logical_shapes(reader: Any) -> dict[str, tuple[int, ...]]:
    names = _array_strings(reader, "audiocpp.tensor_names")
    ranks = _array_i32(reader, "audiocpp.tensor_ranks")
    flat = _array_i64(reader, "audiocpp.tensor_shapes")
    if len(names) != len(ranks):
        raise ValueError("GGUF tensor name/rank metadata length mismatch")
    out: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, rank in zip(names, ranks):
        shape = tuple(flat[offset : offset + rank])
        if len(shape) != rank:
            raise ValueError("GGUF logical tensor shapes are truncated")
        out[name] = shape
        offset += rank
    if offset != len(flat):
        raise ValueError("GGUF logical tensor shapes have trailing dimensions")
    return out


def _sha256_array(data: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    view = memoryview(np.ascontiguousarray(data)).cast("B")
    stride = 64 * 1024 * 1024
    for offset in range(0, len(view), stride):
        digest.update(view[offset : offset + stride])
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    import gguf

    args = _parse_args()
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(source_path, "r")
    shapes = _logical_shapes(reader)
    output_shapes = compact_logical_shapes(shapes)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if LM_HEAD_NAME not in tensors:
        raise ValueError(f"source GGUF is missing {LM_HEAD_NAME}")
    source_head = tensors[LM_HEAD_NAME]
    if source_head.data.dtype != np.uint8:
        raise ValueError(f"expected raw quantized lm_head bytes; got {source_head.data.dtype}")
    compact_head = compact_lm_head_rows(source_head.data)

    tmp = output_path.with_name(output_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = gguf.GGUFWriter(tmp, "audiocpp", use_temp_file=True)
    try:
        writer.add_name(output_path.stem)
        writer.add_string("audiocpp.tensor_name_format", _field_string(reader, "audiocpp.tensor_name_format"))
        writer.add_string("audiocpp.source_format", "gguf_compact_lm_head")
        writer.add_string("audiocpp.weight_type", _field_string(reader, "audiocpp.weight_type"))
        writer.add_array("audiocpp.tensor_sources.names", _array_strings(reader, "audiocpp.tensor_sources.names"))
        writer.add_array("audiocpp.tensor_sources.paths", _array_strings(reader, "audiocpp.tensor_sources.paths"))
        writer.add_string("standalone_gguf_converter.version", "1")
        writer.add_string("minimax_music3.lm_head.layout", LAYOUT)
        writer.add_uint64("minimax_music3.lm_head.source_vocab_size", SOURCE_VOCAB_SIZE)
        writer.add_array("minimax_music3.lm_head.token_ids", compact_token_ids())

        logical_names: list[str] = []
        logical_shape_list: list[tuple[int, ...]] = []
        for tensor in sorted(reader.tensors, key=lambda item: item.name):
            data = compact_head if tensor.name == LM_HEAD_NAME else tensor.data
            raw_dtype = tensor.tensor_type if data.dtype == np.uint8 else None
            writer.add_tensor(tensor.name, data, raw_dtype=raw_dtype)
            logical_names.append(tensor.name)
            logical_shape_list.append(output_shapes[tensor.name])

        writer.add_array("audiocpp.tensor_names", logical_names)
        writer.add_key_value(
            "audiocpp.tensor_ranks",
            [len(shape) for shape in logical_shape_list],
            gguf.GGUFValueType.ARRAY,
            sub_type=gguf.GGUFValueType.INT32,
        )
        writer.add_key_value(
            "audiocpp.tensor_shapes",
            [dim for shape in logical_shape_list for dim in shape],
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

    verify = gguf.GGUFReader(output_path, "r")
    output_head = next(tensor for tensor in verify.tensors if tensor.name == LM_HEAD_NAME)
    if output_head.data.shape != compact_head.shape or not np.array_equal(output_head.data, compact_head):
        raise RuntimeError("output compact lm_head does not match retained source rows byte-for-byte")

    manifest = {
        "schema_version": 1,
        "layout": LAYOUT,
        "source": str(source_path),
        "output": str(output_path),
        "source_lm_head_shape": list(shapes[LM_HEAD_NAME]),
        "output_lm_head_shape": list(output_shapes[LM_HEAD_NAME]),
        "source_lm_head_type": str(source_head.tensor_type),
        "source_lm_head_bytes": int(source_head.data.nbytes),
        "output_lm_head_bytes": int(output_head.data.nbytes),
        "source_lm_head_sha256": _sha256_array(source_head.data),
        "output_lm_head_sha256": _sha256_array(output_head.data),
        "token_ids": compact_token_ids(),
        "row_bytes_equal": True,
        "output_size_bytes": output_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print(f"manifest={manifest_path}")
    print(f"lm_head_bytes={source_head.data.nbytes}->{output_head.data.nbytes}")
    print("row_bytes_equal=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
