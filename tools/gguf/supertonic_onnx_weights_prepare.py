#!/usr/bin/env python3
"""Convert Supertonic 3 ONNX initializers into a single safetensors package."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


GRAPH_FILES = {
    "duration_predictor": "duration_predictor.onnx",
    "text_encoder": "text_encoder.onnx",
    "vector_estimator": "vector_estimator.onnx",
    "vocoder": "vocoder.onnx",
}

GENERATED_INITIALIZER_RENAMES = {
    "text_encoder": {
        "onnx::MatMul_3680": "tts.ttl.speech_prompted_text_encoder.attention1.W_query.linear.weight",
        "onnx::MatMul_3681": "tts.ttl.speech_prompted_text_encoder.attention1.W_key.linear.weight",
        "onnx::MatMul_3682": "tts.ttl.speech_prompted_text_encoder.attention1.W_value.linear.weight",
        "onnx::MatMul_3683": "tts.ttl.speech_prompted_text_encoder.attention1.out_fc.linear.weight",
        "onnx::MatMul_3684": "tts.ttl.speech_prompted_text_encoder.attention2.W_query.linear.weight",
        "onnx::MatMul_3685": "tts.ttl.speech_prompted_text_encoder.attention2.W_key.linear.weight",
        "onnx::MatMul_3686": "tts.ttl.speech_prompted_text_encoder.attention2.W_value.linear.weight",
        "onnx::MatMul_3687": "tts.ttl.speech_prompted_text_encoder.attention2.out_fc.linear.weight",
    },
    "vector_estimator": {
        "onnx::MatMul_3384": "vector_estimator.tts.ttl.vector_field.main_blocks.1.linear.linear.weight",
        "onnx::MatMul_3429": "vector_estimator.tts.ttl.vector_field.main_blocks.7.linear.linear.weight",
        "onnx::MatMul_3474": "vector_estimator.tts.ttl.vector_field.main_blocks.13.linear.linear.weight",
        "onnx::MatMul_3519": "vector_estimator.tts.ttl.vector_field.main_blocks.19.linear.linear.weight",
        "onnx::MatMul_3390": "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.weight",
        "onnx::MatMul_3391": "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.W_key.linear.weight",
        "onnx::MatMul_3392": "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.W_value.linear.weight",
        "onnx::MatMul_3399": "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.out_fc.linear.weight",
        "onnx::MatMul_3435": "vector_estimator.tts.ttl.vector_field.main_blocks.9.attn.W_query.linear.weight",
        "onnx::MatMul_3436": "vector_estimator.tts.ttl.vector_field.main_blocks.9.attn.W_key.linear.weight",
        "onnx::MatMul_3437": "vector_estimator.tts.ttl.vector_field.main_blocks.9.attn.W_value.linear.weight",
        "onnx::MatMul_3444": "vector_estimator.tts.ttl.vector_field.main_blocks.9.attn.out_fc.linear.weight",
        "onnx::MatMul_3480": "vector_estimator.tts.ttl.vector_field.main_blocks.15.attn.W_query.linear.weight",
        "onnx::MatMul_3481": "vector_estimator.tts.ttl.vector_field.main_blocks.15.attn.W_key.linear.weight",
        "onnx::MatMul_3482": "vector_estimator.tts.ttl.vector_field.main_blocks.15.attn.W_value.linear.weight",
        "onnx::MatMul_3489": "vector_estimator.tts.ttl.vector_field.main_blocks.15.attn.out_fc.linear.weight",
        "onnx::MatMul_3525": "vector_estimator.tts.ttl.vector_field.main_blocks.21.attn.W_query.linear.weight",
        "onnx::MatMul_3526": "vector_estimator.tts.ttl.vector_field.main_blocks.21.attn.W_key.linear.weight",
        "onnx::MatMul_3527": "vector_estimator.tts.ttl.vector_field.main_blocks.21.attn.W_value.linear.weight",
        "onnx::MatMul_3534": "vector_estimator.tts.ttl.vector_field.main_blocks.21.attn.out_fc.linear.weight",
        "onnx::MatMul_3405": "vector_estimator.tts.ttl.vector_field.main_blocks.5.attention.W_query.linear.weight",
        "onnx::MatMul_3406": "vector_estimator.tts.ttl.vector_field.main_blocks.5.attention.W_key.linear.weight",
        "onnx::MatMul_3407": "vector_estimator.tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.weight",
        "onnx::MatMul_3408": "vector_estimator.tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.weight",
        "onnx::MatMul_3450": "vector_estimator.tts.ttl.vector_field.main_blocks.11.attention.W_query.linear.weight",
        "onnx::MatMul_3451": "vector_estimator.tts.ttl.vector_field.main_blocks.11.attention.W_key.linear.weight",
        "onnx::MatMul_3452": "vector_estimator.tts.ttl.vector_field.main_blocks.11.attention.W_value.linear.weight",
        "onnx::MatMul_3453": "vector_estimator.tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.weight",
        "onnx::MatMul_3495": "vector_estimator.tts.ttl.vector_field.main_blocks.17.attention.W_query.linear.weight",
        "onnx::MatMul_3496": "vector_estimator.tts.ttl.vector_field.main_blocks.17.attention.W_key.linear.weight",
        "onnx::MatMul_3497": "vector_estimator.tts.ttl.vector_field.main_blocks.17.attention.W_value.linear.weight",
        "onnx::MatMul_3498": "vector_estimator.tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.weight",
        "onnx::MatMul_3540": "vector_estimator.tts.ttl.vector_field.main_blocks.23.attention.W_query.linear.weight",
        "onnx::MatMul_3541": "vector_estimator.tts.ttl.vector_field.main_blocks.23.attention.W_key.linear.weight",
        "onnx::MatMul_3542": "vector_estimator.tts.ttl.vector_field.main_blocks.23.attention.W_value.linear.weight",
        "onnx::MatMul_3543": "vector_estimator.tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.weight",
    },
    "vocoder": {
        "onnx::Conv_1441": "tts.ae.decoder.embed.net.weight",
        "onnx::Conv_1442": "tts.ae.decoder.embed.net.bias",
        "onnx::PRelu_1506": "tts.ae.decoder.head.act.weight",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Supertonic 3 ONNX assets to clean safetensors."
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("models/supertonic-3/onnx"),
        help="Directory containing the four Supertonic ONNX files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/supertonic-3/ggml"),
        help="Output directory for supertonic.safetensors.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted files.",
    )
    return parser.parse_args()


def require_real_onnx(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not an ONNX model; run git lfs pull first"
        )


def sanitize_name(graph_name: str, tensor_name: str) -> str:
    name = tensor_name.strip()
    if name.startswith("/"):
        name = name[1:]
    name = name.replace("::", ".")
    name = name.replace("/", ".")
    name = re.sub(r"[^0-9A-Za-z_.-]+", ".", name)
    name = re.sub(r"\.+", ".", name).strip(".")
    if not name:
        name = "tensor"
    return f"{graph_name}.{name}"


def is_generated_initializer_name(tensor_name: str) -> bool:
    lowered = tensor_name.lower()
    return "onnx" in lowered or "constant" in lowered


def semantic_generated_name(graph_name: str, tensor_name: str) -> str | None:
    return GENERATED_INITIALIZER_RENAMES.get(graph_name, {}).get(tensor_name)


def unique_name(base: str, used: Counter[str]) -> str:
    used[base] += 1
    if used[base] == 1:
        return base
    return f"{base}.{used[base] - 1}"


def convert_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float64:
        return array.astype(np.float32)
    if array.dtype == np.uint64:
        max_i64 = np.iinfo(np.int64).max
        if array.size and int(array.max()) > max_i64:
            raise RuntimeError("uint64 initializer cannot be represented as int64")
        return array.astype(np.int64)
    if array.dtype == np.uint32:
        return array.astype(np.int64)
    if array.dtype == np.uint16:
        return array.astype(np.int32)
    if array.dtype == np.uint8:
        return array.astype(np.int16)
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    return np.ascontiguousarray(array)


def load_graph(
    graph_name: str,
    path: Path,
    tensors: dict[str, np.ndarray],
    used_names: Counter[str],
) -> None:
    require_real_onnx(path)
    model = onnx.load(path, load_external_data=False)

    for initializer in model.graph.initializer:
        array = convert_array(numpy_helper.to_array(initializer))
        if is_generated_initializer_name(initializer.name):
            semantic_name = semantic_generated_name(graph_name, initializer.name)
            if semantic_name is None:
                continue
            clean_name = unique_name(sanitize_name(graph_name, semantic_name), used_names)
        else:
            clean_name = unique_name(sanitize_name(graph_name, initializer.name), used_names)
        tensors[clean_name] = np.ascontiguousarray(array)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    weights_path = output_dir / "supertonic.safetensors"
    if not args.force and weights_path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing converted files under {output_dir}; pass --force"
        )

    tensors: dict[str, np.ndarray] = {}
    used_names: Counter[str] = Counter()
    for graph_name, filename in GRAPH_FILES.items():
        graph_path = args.onnx_dir / filename
        if not graph_path.is_file():
            raise RuntimeError(f"missing Supertonic ONNX graph: {graph_path}")
        load_graph(graph_name, graph_path, tensors, used_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "minitts-supertonic-ggml",
        "source": str(args.onnx_dir),
        "tensor_count": str(len(tensors)),
    }
    save_file(tensors, weights_path, metadata=metadata)
    print(f"wrote {weights_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
