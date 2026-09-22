#!/usr/bin/env python3
"""Convert KittenTTS ONNX initializers into a native GGML safetensors package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


DEFAULT_MODEL = Path("models/kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx")
DEFAULT_OUTPUT_DIR = Path("models/kitten-tts-mini-0.8/ggml")
DEFAULT_OUTPUT_NAME = "kitten_tts.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KittenTTS ONNX initializers to a safetensors file for native GGML graphs."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the KittenTTS ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the converted safetensors package.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output safetensors filename.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    parser.add_argument(
        "--voices",
        type=Path,
        default=Path("models/kitten-tts-mini-0.8/voices.npz"),
        help="Path to the KittenTTS voices.npz file.",
    )
    return parser.parse_args()


def require_real_onnx(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(f"{path} is a Git LFS pointer, not an ONNX model; run git lfs pull first")


def convert_array(array: np.ndarray, tensor_name: str) -> np.ndarray:
    if array.dtype == np.float64:
        return np.ascontiguousarray(array.astype(np.float32))
    if array.dtype == np.uint64:
        max_i64 = np.iinfo(np.int64).max
        if array.size and int(array.max()) > max_i64:
            raise RuntimeError(f"{tensor_name} contains uint64 values that cannot be represented as int64")
        return np.ascontiguousarray(array.astype(np.int64))
    if array.dtype == np.uint32:
        return np.ascontiguousarray(array.astype(np.int64))
    if array.dtype == np.uint16:
        return np.ascontiguousarray(array.astype(np.int32))
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8))
    if array.dtype.kind == "O":
        raise RuntimeError(f"{tensor_name} has unsupported object dtype")
    return np.ascontiguousarray(array)


def dequantize_weight(initializers: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    q = initializers[f"{prefix}_quantized"]
    scale = initializers[f"{prefix}_scale"].astype(np.float32)
    zero_point = initializers[f"{prefix}_zero_point"].astype(np.float32)
    while scale.ndim < q.ndim:
        scale = np.expand_dims(scale, axis=-1)
        zero_point = np.expand_dims(zero_point, axis=-1)
    return np.ascontiguousarray((q.astype(np.float32) - zero_point) * scale)


def linear_weight(initializers: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    if f"{prefix}_quantized" in initializers:
        weight = dequantize_weight(initializers, prefix)
    else:
        weight = initializers[prefix]
    if weight.ndim != 2:
        raise RuntimeError(f"expected rank-2 linear weight for {prefix}, got {weight.shape}")
    return np.ascontiguousarray(weight.T)


def conv_weight(initializers: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    if f"{prefix}_quantized" in initializers:
        weight = dequantize_weight(initializers, prefix)
    else:
        weight = initializers[prefix]
    if weight.ndim != 3:
        raise RuntimeError(f"expected rank-3 convolution weight for {prefix}, got {weight.shape}")
    return np.ascontiguousarray(weight)


def add_weight_norm_conv(tensors: dict[str, np.ndarray], name: str, weight: np.ndarray, transpose_leading: bool = False) -> None:
    leading = weight.shape[0] if not transpose_leading else weight.shape[1]
    flat = weight.reshape(leading, -1) if not transpose_leading else weight.transpose(1, 0, 2).reshape(leading, -1)
    g = np.linalg.norm(flat.astype(np.float64), axis=1).astype(np.float32)
    tensors[f"{name}.weight_v"] = np.ascontiguousarray(weight.astype(np.float32, copy=False))
    tensors[f"{name}.weight_g"] = np.ascontiguousarray(g.reshape(leading, 1, 1))


def add_linear(
    tensors: dict[str, np.ndarray],
    initializers: dict[str, np.ndarray],
    dst: str,
    weight_prefix: str,
    bias_name: str | None,
) -> None:
    tensors[f"{dst}.weight"] = linear_weight(initializers, weight_prefix)
    if bias_name is not None:
        tensors[f"{dst}.bias"] = np.ascontiguousarray(initializers[bias_name])


def add_conv(
    tensors: dict[str, np.ndarray],
    initializers: dict[str, np.ndarray],
    dst: str,
    weight_prefix: str,
    bias_name: str | None,
    weight_norm: bool,
    transpose_leading: bool = False,
) -> None:
    weight = conv_weight(initializers, weight_prefix)
    if weight_norm:
        add_weight_norm_conv(tensors, dst, weight, transpose_leading=transpose_leading)
    else:
        tensors[f"{dst}.weight"] = weight
    if bias_name is not None:
        tensors[f"{dst}.bias"] = np.ascontiguousarray(initializers[bias_name])


def reorder_lstm_gates(value: np.ndarray) -> np.ndarray:
    hidden = value.shape[-2] // 4 if value.ndim == 3 else value.shape[-1] // 4
    if value.ndim == 3:
        parts = [value[:, i * hidden:(i + 1) * hidden, :] for i in range(4)]
        return np.ascontiguousarray(np.concatenate([parts[0], parts[2], parts[3], parts[1]], axis=1))
    parts = [value[:, i * hidden:(i + 1) * hidden] for i in range(4)]
    return np.ascontiguousarray(np.concatenate([parts[0], parts[2], parts[3], parts[1]], axis=1))


def add_lstm(
    tensors: dict[str, np.ndarray],
    initializers: dict[str, np.ndarray],
    dst: str,
    w_prefix: str,
    r_prefix: str,
    b_name: str,
) -> None:
    w = reorder_lstm_gates(np.transpose(dequantize_weight(initializers, w_prefix), (0, 2, 1)))
    r = reorder_lstm_gates(np.transpose(dequantize_weight(initializers, r_prefix), (0, 2, 1)))
    hidden4 = w.shape[1]
    b_raw = initializers[b_name].astype(np.float32, copy=False)
    b_ih = reorder_lstm_gates(b_raw[:, :hidden4])
    b_hh = reorder_lstm_gates(b_raw[:, hidden4:])
    for direction, suffix in [(0, ""), (1, "_reverse")]:
        tensors[f"{dst}.weight_ih_l0{suffix}"] = np.ascontiguousarray(w[direction])
        tensors[f"{dst}.weight_hh_l0{suffix}"] = np.ascontiguousarray(r[direction])
        tensors[f"{dst}.bias_ih_l0{suffix}"] = np.ascontiguousarray(b_ih[direction])
        tensors[f"{dst}.bias_hh_l0{suffix}"] = np.ascontiguousarray(b_hh[direction])


def add_voice_tensors(tensors: dict[str, np.ndarray], voices_path: Path) -> None:
    voices = np.load(voices_path)
    for name in sorted(voices.files):
        tensors[f"voices.{name}"] = np.ascontiguousarray(voices[name].astype(np.float32, copy=False))


def strip_kmodel(name: str) -> str:
    return name[len("kmodel."):] if name.startswith("kmodel.") else name


def add_native_named_tensors(tensors: dict[str, np.ndarray], initializers: dict[str, np.ndarray]) -> None:
    for src, array in initializers.items():
        if not src.startswith("kmodel."):
            continue
        dst = strip_kmodel(src)
        if dst.endswith(".weight_quantized") or dst.endswith(".weight_scale") or dst.endswith(".weight_zero_point"):
            continue
        if dst.endswith(".bias") or dst.endswith(".gamma") or dst.endswith(".beta") or ".alpha" in dst:
            tensors[dst] = np.ascontiguousarray(array)
        elif dst.endswith(".weight"):
            tensors[dst] = np.ascontiguousarray(array)

    add_linear(tensors, initializers, "bert.encoder.embedding_hidden_mapping_in", "onnx::MatMul_5883", "kmodel.bert.encoder.embedding_hidden_mapping_in.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query", "onnx::MatMul_5884", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key", "onnx::MatMul_5887", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value", "onnx::MatMul_5890", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense", "onnx::MatMul_5894", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn", "onnx::MatMul_5895", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias")
    add_linear(tensors, initializers, "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output", "onnx::MatMul_5896", "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias")
    add_linear(tensors, initializers, "bert_encoder", "onnx::MatMul_6040", "kmodel.bert_encoder.bias")
    add_linear(tensors, initializers, "predictor.duration_proj.linear_layer", "onnx::MatMul_6245", "kmodel.predictor.duration_proj.linear_layer.bias")

    add_lstm(tensors, initializers, "text_encoder.lstm", "onnx::LSTM_5874", "onnx::LSTM_5875", "onnx::LSTM_5873")
    add_lstm(tensors, initializers, "predictor.text_encoder.lstms.0", "onnx::LSTM_6094", "onnx::LSTM_6095", "onnx::LSTM_6093")
    add_lstm(tensors, initializers, "predictor.text_encoder.lstms.2", "onnx::LSTM_6144", "onnx::LSTM_6145", "onnx::LSTM_6143")
    add_lstm(tensors, initializers, "predictor.text_encoder.lstms.4", "onnx::LSTM_6194", "onnx::LSTM_6195", "onnx::LSTM_6193")
    add_lstm(tensors, initializers, "predictor.lstm", "onnx::LSTM_6243", "onnx::LSTM_6244", "onnx::LSTM_6242")
    add_lstm(tensors, initializers, "predictor.shared", "onnx::LSTM_6292", "onnx::LSTM_6293", "onnx::LSTM_6291")

    plain_conv_prefixes = {
        "predictor.F0_proj",
        "predictor.N_proj",
        "decoder.generator.noise_convs.0",
        "decoder.generator.noise_convs.1",
    }
    for src in sorted(initializers):
        if not src.startswith("kmodel.") or not src.endswith(".weight_quantized"):
            continue
        dst_prefix = strip_kmodel(src[:-len(".weight_quantized")])
        weight = dequantize_weight(initializers, src[:-len("_quantized")])
        if weight.ndim == 2:
            tensors[f"{dst_prefix}.weight"] = np.ascontiguousarray(weight.T)
        elif weight.ndim == 3:
            add_conv(
                tensors,
                initializers,
                dst_prefix,
                src[:-len("_quantized")],
                f"kmodel.{dst_prefix}.bias" if f"kmodel.{dst_prefix}.bias" in initializers else None,
                weight_norm=dst_prefix not in plain_conv_prefixes,
                transpose_leading=False,
            )

    folded_weight_norm_plain = [
        ("predictor.F0.1.pool", False),
        ("predictor.N.1.pool", False),
        ("decoder.decode.3.pool", False),
        ("decoder.F0_conv", False),
        ("decoder.N_conv", False),
        ("decoder.generator.ups.0", False),
        ("decoder.generator.ups.1", False),
    ]
    for dst_prefix, transpose_leading in folded_weight_norm_plain:
        src_prefix = f"kmodel.{dst_prefix}.weight"
        if src_prefix in initializers:
            add_weight_norm_conv(
                tensors,
                dst_prefix,
                initializers[src_prefix].astype(np.float32, copy=False),
                transpose_leading=transpose_leading,
            )
        bias_name = f"kmodel.{dst_prefix}.bias"
        if bias_name in initializers:
            tensors[f"{dst_prefix}.bias"] = np.ascontiguousarray(initializers[bias_name])

    tensors["decoder.generator.m_source.l_linear.weight"] = linear_weight(initializers, "onnx::MatMul_6388")
    tensors["decoder.generator.m_source.l_linear.bias"] = np.ascontiguousarray(initializers["kmodel.decoder.generator.m_source.l_linear.bias"])


def convert_onnx(path: Path, voices_path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    require_real_onnx(path)
    model = onnx.load(path, load_external_data=False)
    initializers = {
        initializer.name: convert_array(numpy_helper.to_array(initializer), initializer.name)
        for initializer in model.graph.initializer
    }
    tensors: dict[str, np.ndarray] = {}
    add_native_named_tensors(tensors, initializers)
    add_voice_tensors(tensors, voices_path)

    metadata = {
        "format": "audiocpp-kitten-tts-ggml",
        "source": str(path),
        "voices": str(voices_path),
        "tensor_count": str(len(tensors)),
    }
    return tensors, metadata


def main() -> None:
    args = parse_args()
    if args.output_name.endswith(".safetensors"):
        output_name = args.output_name
    else:
        output_name = f"{args.output_name}.safetensors"
    weights_path = args.output_dir / output_name
    if not args.force and weights_path.exists():
        raise RuntimeError(f"refusing to overwrite existing converted file {weights_path}; pass --force")

    tensors, metadata = convert_onnx(args.onnx, args.voices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, weights_path, metadata=metadata)
    print(f"wrote {weights_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
