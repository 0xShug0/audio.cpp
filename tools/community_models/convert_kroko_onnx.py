#!/usr/bin/env python3
"""Convert a free Kroko Zipformer2 package into audio.cpp tensor assets.

Kroko community ``.data`` files contain a small JSON header followed by four
little-endian length-prefixed blobs: encoder ONNX, decoder ONNX, joiner ONNX,
and tokens.txt.  The ONNX graphs published by Kroko are dynamically quantized.
This converter recovers named floating-point weights from those graphs so the
native audio.cpp runtime and the normal GGUF converter can consume them.

Commercial/encrypted Kroko packages are intentionally unsupported.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


LAYER_PREFIXES = [
    *(f"encoder.encoders.0.layers.{i}" for i in range(2)),
    *(f"encoder.encoders.1.encoder.layers.{i}" for i in range(2)),
    *(f"encoder.encoders.2.encoder.layers.{i}" for i in range(4)),
    *(f"encoder.encoders.3.encoder.layers.{i}" for i in range(5)),
    *(f"encoder.encoders.4.encoder.layers.{i}" for i in range(4)),
    *(f"encoder.encoders.5.encoder.layers.{i}" for i in range(2)),
]


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated Kroko package")
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def unpack_package(source: Path, destination: Path) -> dict:
    data = source.read_bytes()
    header_size, offset = read_u32(data, 0)
    header_end = offset + header_size
    if header_end > len(data):
        raise ValueError("truncated Kroko JSON header")
    header = json.loads(data[offset:header_end].decode("utf-8"))
    if not header.get("free", False):
        raise ValueError(
            "commercial/encrypted Kroko packages are not supported; "
            "use a community package with free=true"
        )
    offset = header_end
    names = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        length, offset = read_u32(data, offset)
        end = offset + length
        if end > len(data):
            raise ValueError(f"truncated Kroko package block: {name}")
        (destination / name).write_bytes(data[offset:end])
        offset = end
    if offset != len(data):
        raise ValueError(f"unexpected {len(data) - offset} trailing package bytes")
    return header


def tensor_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }


def consumers(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            result[name].append(node)
    return result


def downstream_bias(
    node: onnx.NodeProto,
    initializers: dict[str, np.ndarray],
    by_input: dict[str, list[onnx.NodeProto]],
) -> str | None:
    frontier = list(node.output)
    seen: set[int] = set()
    for _ in range(5):
        next_frontier: list[str] = []
        for value in frontier:
            for consumer in by_input.get(value, ()):
                identity = id(consumer)
                if identity in seen:
                    continue
                seen.add(identity)
                if consumer.op_type == "Add":
                    for input_name in consumer.input:
                        if input_name in initializers:
                            return input_name
                next_frontier.extend(consumer.output)
        frontier = next_frontier
    return None


def dequantized_matmul_weights(
    model: onnx.ModelProto,
    *,
    linear_pos_prefixes: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    initializers = tensor_map(model)
    by_input = consumers(model)
    linear_pos = iter(linear_pos_prefixes)
    output: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "MatMulInteger":
            continue
        if len(node.input) < 4:
            raise ValueError("Kroko MatMulInteger is missing quantization inputs")
        weight_name = node.input[1]
        zero_name = node.input[3]
        scale_name = weight_name.removesuffix("_quantized") + "_scale"
        if weight_name not in initializers or zero_name not in initializers or scale_name not in initializers:
            raise ValueError(f"missing quantization tensors for {weight_name}")
        quantized = initializers[weight_name].astype(np.float32)
        zero = initializers[zero_name].astype(np.float32)
        scale = initializers[scale_name].astype(np.float32)
        # ONNX MatMul stores [in, out], whereas audio.cpp Linear weights follow
        # the PyTorch [out, in] convention.
        weight = ((quantized - zero) * scale).T.copy()
        bias_name = downstream_bias(node, initializers, by_input)
        if bias_name is not None:
            semantic_name = bias_name.removesuffix(".bias") + ".weight"
        else:
            try:
                semantic_name = next(linear_pos) + ".self_attn_weights.linear_pos.weight"
            except StopIteration as exc:
                raise ValueError(f"cannot name bias-free MatMul weight {weight_name}") from exc
        if semantic_name in output:
            raise ValueError(f"duplicate recovered weight name: {semantic_name}")
        output[semantic_name] = weight
    try:
        unexpected = next(linear_pos)
    except StopIteration:
        unexpected = None
    if unexpected is not None:
        raise ValueError(f"not all expected positional weights were recovered; first missing layer: {unexpected}")
    return output


def recover_zipformer_constants(
    model: onnx.ModelProto,
) -> dict[str, np.ndarray]:
    """Give semantic names to parameters folded by the ONNX exporter.

    ``SimpleDownsample.bias.softmax()`` is exported as a constant weight, and
    the two learned chunkwise-convolution edge scales are anonymous
    ``onnx::Concat_*`` initializers.  They are model parameters, not disposable
    graph helpers, so preserve them under stable audio.cpp names.
    """
    initializers = tensor_map(model)
    recovered: dict[str, np.ndarray] = {}

    downsample_weights: list[np.ndarray] = []
    for node in model.graph.node:
        if (
            node.op_type == "Mul"
            and node.name.startswith("/downsample")
            and len(node.input) >= 2
            and node.input[1] in initializers
        ):
            value = initializers[node.input[1]]
            if value.ndim == 3 and value.shape[1:] == (1, 1):
                downsample_weights.append(
                    np.ascontiguousarray(value.reshape(-1).astype(np.float32))
                )
    if len(downsample_weights) != 6:
        raise ValueError(
            "expected five Zipformer stack downsamplers and one output "
            f"downsampler, found {len(downsample_weights)}"
        )
    for stack, value in enumerate(downsample_weights[:-1], start=1):
        recovered[
            f"encoder.encoders.{stack}.downsample.weights"
        ] = value
    recovered["encoder.downsample_output.weights"] = downsample_weights[-1]

    nodes = list(model.graph.node)
    chunk_scales = 0
    for index, node in enumerate(nodes):
        if (
            node.op_type != "Conv"
            or len(node.input) < 2
            or not node.input[1].endswith(
                ".depthwise_conv.chunkwise_conv.weight"
            )
        ):
            continue
        semantic_base = node.input[1].removesuffix(
            ".chunkwise_conv.weight"
        )
        conv_weight = initializers[node.input[1]]
        if conv_weight.ndim != 3:
            raise ValueError(
                f"unexpected chunkwise convolution weight shape for {semantic_base}"
            )
        scale_shape = (conv_weight.shape[0], conv_weight.shape[2])
        candidates: list[str] = []
        # Exporters use either a nearby Concat (128-frame packages) or
        # dynamic Slice nodes (64-frame packages) to select the learned left
        # and right edge scales.  In both layouts the two source
        # initializers have the unambiguous [channels, kernel] shape.
        for follower in nodes[index + 1 : index + 13]:
            for input_name in follower.input:
                value = initializers.get(input_name)
                if value is not None and value.shape == scale_shape:
                    candidates.append(input_name)
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 2:
            raise ValueError(
                "cannot recover two chunkwise edge scales for "
                f"{semantic_base}; found {candidates}"
            )
        recovered[f"{semantic_base}.chunk_scale_left"] = np.ascontiguousarray(
            initializers[candidates[0]].astype(np.float32)
        )
        recovered[f"{semantic_base}.chunk_scale_right"] = np.ascontiguousarray(
            initializers[candidates[1]].astype(np.float32)
        )
        chunk_scales += 1
    if chunk_scales != len(LAYER_PREFIXES) * 2:
        raise ValueError(
            "expected two chunkwise convolution scales per Zipformer layer; "
            f"recovered {chunk_scales}"
        )
    return recovered


def convert_graph(
    path: Path,
    namespace: str,
    *,
    linear_pos_prefixes: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    model = onnx.load(path, load_external_data=True)
    initializers = tensor_map(model)
    recovered = dequantized_matmul_weights(
        model, linear_pos_prefixes=linear_pos_prefixes
    )
    if namespace == "encoder":
        recovered.update(recover_zipformer_constants(model))
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3:
            continue
        quant_name, scale_name, zero_name = node.input[:3]
        if quant_name not in initializers:
            producer = producers.get(quant_name)
            if (
                producer is not None
                and producer.op_type == "Gather"
                and producer.input
                and producer.input[0] in initializers
            ):
                quant_name = producer.input[0]
        if (
            quant_name not in initializers
            or scale_name not in initializers
            or zero_name not in initializers
        ):
            continue
        semantic_name = quant_name.removesuffix("_quantized")
        recovered[semantic_name] = np.ascontiguousarray(
            (initializers[quant_name].astype(np.float32)
             - initializers[zero_name].astype(np.float32))
            * initializers[scale_name].astype(np.float32)
        )
    # Skip only tensors that belong to an ONNX quantization tuple.  A blanket
    # ``*_scale`` filter would incorrectly remove real Zipformer parameters
    # such as ``bypass_scale`` and ``norm.log_scale``.
    quantization_helpers: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "MatMulInteger" and len(node.input) >= 4:
            weight_name = node.input[1]
            quantization_helpers.update(
                {
                    weight_name,
                    node.input[3],
                    weight_name.removesuffix("_quantized") + "_scale",
                }
            )
        elif node.op_type == "DequantizeLinear" and len(node.input) >= 3:
            quantization_helpers.update(node.input[:3])
            quant_name = node.input[0]
            producer = producers.get(quant_name)
            if (
                producer is not None
                and producer.op_type == "Gather"
                and producer.input
                and producer.input[0] in initializers
            ):
                quantization_helpers.add(producer.input[0])
    result: dict[str, np.ndarray] = {}
    for name, value in initializers.items():
        if name in quantization_helpers or name.startswith("onnx::MatMul_"):
            continue
        # Safetensors represents rank-0 values as a one-element tensor so the
        # same output can also be packed into GGUF.
        if value.ndim == 0:
            value = value.reshape(1)
        result[f"{namespace}.{name}"] = np.ascontiguousarray(value)
    for name, value in recovered.items():
        result[f"{namespace}.{name}"] = np.ascontiguousarray(value)
    return result


def metadata_from_encoder(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}

    def ints(key: str) -> list[int]:
        return [int(part) for part in values[key].split(",")]

    left_context_len = ints("left_context_len")
    base_left_context = left_context_len[0]
    if any(
        length <= 0 or base_left_context % length != 0
        for length in left_context_len
    ):
        raise ValueError(
            "Kroko Zipformer left-context lengths do not describe integral "
            "encoder downsampling factors"
        )

    return {
        "model_type": values.get("model_type", "zipformer2"),
        "sample_rate": 16000,
        "feature_dim": 80,
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "dither": 0.0,
        "snip_edges": False,
        "normalize_samples": True,
        "chunk_size": int(values["T"]),
        "chunk_shift": int(values["decode_chunk_len"]),
        "subsampling_factor": 4,
        "encoder_dims": ints("encoder_dims"),
        "query_head_dims": ints("query_head_dims"),
        "value_head_dims": ints("value_head_dims"),
        "num_heads": ints("num_heads"),
        "num_encoder_layers": ints("num_encoder_layers"),
        "cnn_module_kernels": ints("cnn_module_kernels"),
        "left_context_len": left_context_len,
        "downsampling_factors": [
            base_left_context // length for length in left_context_len
        ],
    }


def decoder_metadata(path: Path) -> dict[str, int]:
    model = onnx.load(path, load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}
    return {
        "vocab_size": int(values["vocab_size"]),
        "context_size": int(values["context_size"]),
        "blank_id": 0,
        "unk_id": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Free Kroko .data package or extracted directory")
    parser.add_argument("output", type=Path, help="Output audio.cpp model directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    weights_path = output / "model.safetensors"
    if weights_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {weights_path}")

    temporary: TemporaryDirectory[str] | None = None
    if args.input.is_dir():
        extracted = args.input.resolve()
        header = {
            "type": "zipformer2",
            "free": True,
            "language": {"name": "Unknown", "iso": "auto"},
        }
    else:
        temporary = TemporaryDirectory(prefix="audiocpp-kroko-")
        extracted = Path(temporary.name)
        header = unpack_package(args.input.resolve(), extracted)

    required = [extracted / name for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing Kroko resources: " + ", ".join(missing))

    tensors: dict[str, np.ndarray] = {}
    tensors.update(
        convert_graph(
            extracted / "encoder.onnx",
            "encoder",
            linear_pos_prefixes=LAYER_PREFIXES,
        )
    )
    tensors.update(convert_graph(extracted / "decoder.onnx", "decoder"))
    tensors.update(convert_graph(extracted / "joiner.onnx", "joiner"))

    config = metadata_from_encoder(extracted / "encoder.onnx")
    config.update(decoder_metadata(extracted / "decoder.onnx"))
    config["audiocpp_family"] = "kroko_asr"
    config["variant"] = "Kroko-Community-Streaming"
    config["language"] = header.get("language", {})
    config["package_id"] = header.get("id", "")
    config["source_format"] = "kroko-free-data"

    output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(weights_path),
        metadata={
            "format": "pt",
            "source": "Banafo/Kroko-ASR",
            "audiocpp_family": "kroko_asr",
        },
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "tokens.txt").write_bytes((extracted / "tokens.txt").read_bytes())
    if temporary is not None:
        temporary.cleanup()
    print(f"wrote {weights_path} ({len(tensors)} tensors)")
    print(f"wrote {output / 'config.json'}")
    print(f"wrote {output / 'tokens.txt'}")


if __name__ == "__main__":
    main()
