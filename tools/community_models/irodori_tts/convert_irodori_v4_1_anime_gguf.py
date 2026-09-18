#!/usr/bin/env python3
"""Convert Irodori-TTS-v4.1-Anime int8 weights to audio.cpp native GGUF.

The fine-tune ships TorchAO-style int8 packed tensors. audio.cpp should not
learn that fine-tune-specific layout at runtime, so this script dequantizes the
packed pairs back to the native Irodori tensor names and then lets
``audiocpp_gguf`` write the normal self-contained q8_0 GGUF.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


QDATA_SUFFIX = "._weight_qdata"
SCALE_SUFFIX = "._weight_scale"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    return path


def dequantize_int8_weight_only(input_path: Path, output_path: Path) -> None:
    converted: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for key in sorted(keys):
            if key.endswith(SCALE_SUFFIX):
                continue
            if key.endswith(QDATA_SUFFIX):
                base = key[: -len(QDATA_SUFFIX)]
                scale_key = base + SCALE_SUFFIX
                if scale_key not in keys:
                    raise RuntimeError(f"missing int8 scale tensor for {key}")
                name = base + ".weight"
                qdata = source.get_tensor(key).to(torch.float32)
                scale = source.get_tensor(scale_key).to(torch.float32)
                converted[name] = (qdata * scale).to(torch.bfloat16).contiguous()
                continue
            if key in converted:
                raise RuntimeError(f"duplicate converted tensor: {key}")
            converted[key] = source.get_tensor(key).contiguous()
    save_file(converted, output_path)


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
    out: dict[str, tuple[int, ...]] = {}
    offset = 0
    for name, rank in zip(names, ranks):
        out[name] = tuple(flat_shapes[offset : offset + rank])
        offset += rank
    if offset != len(flat_shapes):
        raise RuntimeError("GGUF logical shape metadata is inconsistent")
    return out


def tensor_values(tensor: gguf.ReaderTensor, shape: tuple[int, ...]) -> np.ndarray:
    if tensor.data.dtype == np.uint8:
        return gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False).reshape(shape)
    return np.asarray(tensor.data).reshape(shape)


def extract_codec_weights(base_gguf: Path, output_path: Path) -> None:
    reader = gguf.GGUFReader(str(base_gguf))
    shapes = logical_shapes(reader)
    prefix = "codec_weights/"
    codec: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith(prefix):
            continue
        name = tensor.name[len(prefix) :]
        values = np.array(tensor_values(tensor, shapes[tensor.name]), copy=True)
        codec[name] = torch.from_numpy(values).contiguous()
    if not codec:
        raise RuntimeError(f"base GGUF contains no {prefix} tensors: {base_gguf}")
    save_file(codec, output_path)


def materialize_sidecars(base_gguf: Path, output_dir: Path) -> None:
    reader = gguf.GGUFReader(str(base_gguf))
    names = list(reader.fields["audiocpp.embedded_files.names"].contents())
    offsets = [int(value) for value in reader.fields["audiocpp.embedded_files.offsets"].contents()]
    data = bytes(int(value) for value in reader.fields["audiocpp.embedded_files.data"].contents())
    if len(offsets) != len(names) + 1:
        raise RuntimeError("GGUF embedded sidecar offsets are inconsistent")
    for index, name in enumerate(names):
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data[offsets[index] : offsets[index + 1]])


def run_converter(args: argparse.Namespace, staged_model: Path, staged_codec: Path, sidecar_root: Path) -> None:
    command = [
        str(args.converter),
        "--input",
        f"model_weights={staged_model}",
        "--input",
        f"codec_weights={staged_codec}",
        "--root",
        str(sidecar_root),
        "--family",
        "irodori_tts",
        "--type",
        "q8_0",
        "--output",
        str(args.output),
    ]
    tokenizer_json = args.source_dir / "tokenizer" / "tokenizer.json"
    tokenizer_config = args.source_dir / "tokenizer" / "tokenizer_config.json"
    if tokenizer_json.is_file():
        command += ["--sidecar", f"{tokenizer_json}=tokenizer/tokenizer.json"]
    if tokenizer_config.is_file():
        command += ["--sidecar", f"{tokenizer_config}=tokenizer/tokenizer_config.json"]
    if args.overwrite:
        command.append("--overwrite")

    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert phasefield-audio/Irodori-TTS-v4.1-Anime int8-weight-only to q8_0 GGUF."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Downloaded phasefield-audio/Irodori-TTS-v4.1-Anime directory.",
    )
    parser.add_argument(
        "--base-gguf",
        type=Path,
        required=True,
        help="audio.cpp native Irodori v4 Small GGUF used for codec weights and base sidecars.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("build/debug/bin/audiocpp_gguf"),
        help="Path to audiocpp_gguf.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output q8_0 GGUF path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-staged",
        type=Path,
        default=None,
        help="Keep the dequantized temporary model safetensors at this path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.base_gguf = require_file(args.base_gguf.resolve())
    args.converter = require_file(args.converter.resolve())
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    int8_weights = require_file(args.source_dir / "int8-weight-only" / "model.safetensors")

    if args.keep_staged is not None:
        staged_model = args.keep_staged.resolve()
        staged_dir = staged_model.parent
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_codec = staged_dir / "codec.safetensors"
        sidecar_root = staged_dir / "sidecars"
        dequantize_int8_weight_only(int8_weights, staged_model)
        extract_codec_weights(args.base_gguf, staged_codec)
        materialize_sidecars(args.base_gguf, sidecar_root)
        run_converter(args, staged_model, staged_codec, sidecar_root)
        print(f"staged_model={staged_model}")
        print(f"staged_codec={staged_codec}")
        print(f"sidecar_root={sidecar_root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="irodori_v4_1_anime_") as tmp:
        staged_root = Path(tmp)
        staged_model = staged_root / "model.safetensors"
        staged_codec = staged_root / "codec.safetensors"
        sidecar_root = staged_root / "sidecars"
        dequantize_int8_weight_only(int8_weights, staged_model)
        extract_codec_weights(args.base_gguf, staged_codec)
        materialize_sidecars(args.base_gguf, sidecar_root)
        run_converter(args, staged_model, staged_codec, sidecar_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
