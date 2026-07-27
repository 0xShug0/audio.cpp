#!/usr/bin/env python3
"""Convert the official Inflect v2 ONNX exports to audio.cpp safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np  # ty: ignore[unresolved-import]
    import onnx  # ty: ignore[unresolved-import]
    from onnx import numpy_helper  # ty: ignore[unresolved-import]
    from safetensors.numpy import save_file  # ty: ignore[unresolved-import]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Inflect v2 conversion requires onnx and safetensors. Run through "
        "`uv run --with onnx --with safetensors python ...`."
    ) from exc


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != "inflect_v2_inference_config_v1":
        raise ValueError("config is not an Inflect v2 inference config")

    data = config.get("data", {})
    model = config.get("model", {})
    common = {
        "sampling_rate": data.get("sampling_rate"),
        "hop_length": data.get("hop_length"),
        "n_heads": model.get("n_heads"),
        "n_layers": model.get("n_layers"),
        "upsample_rates": model.get("upsample_rates"),
        "upsample_kernel_sizes": model.get("upsample_kernel_sizes"),
        "resblock_kernel_sizes": model.get("resblock_kernel_sizes"),
        "resblock_dilation_sizes": model.get("resblock_dilation_sizes"),
        "use_sdp": model.get("use_sdp"),
        "inference_only": model.get("inference_only"),
    }
    expected_common = {
        "sampling_rate": 24000,
        "hop_length": 256,
        "n_heads": 2,
        "n_layers": 3,
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5]] * 3,
        "use_sdp": False,
        "inference_only": True,
    }
    if common != expected_common:
        raise ValueError("config uses an unsupported Inflect v2 architecture")

    dimensions = (
        model.get("inter_channels"),
        model.get("hidden_channels"),
        model.get("filter_channels"),
        model.get("upsample_initial_channel"),
        model.get("n_layers_q"),
    )
    variants = {
        (192, 96, 768, 320, 3): "micro-v2",
        (128, 72, 384, 192, 2): "nano-v2",
    }
    try:
        variant = variants[dimensions]
    except KeyError as exc:
        raise ValueError(
            "config dimensions do not match Inflect Micro v2 or Nano v2"
        ) from exc
    return config, variant


def _pair(
    expected: dict[str, tuple[int, ...]],
    prefix: str,
    weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
) -> None:
    expected[f"{prefix}.weight"] = weight_shape
    expected[f"{prefix}.bias"] = bias_shape


def _expected_tensors(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    model = config["model"]
    hidden = int(model["hidden_channels"])
    inter = int(model["inter_channels"])
    filters = int(model["filter_channels"])
    upsample = int(model["upsample_initial_channel"])
    half = inter // 2
    heads = int(model["n_heads"])
    expected: dict[str, tuple[int, ...]] = {
        "model.enc_p.emb.weight": (178, hidden),
    }

    for layer in range(3):
        prefix = f"model.enc_p.encoder.attn_layers.{layer}"
        expected[f"{prefix}.emb_rel_k"] = (1, 9, hidden // heads)
        expected[f"{prefix}.emb_rel_v"] = (1, 9, hidden // heads)
        for projection in ("conv_q", "conv_k", "conv_v", "conv_o"):
            _pair(
                expected,
                f"{prefix}.{projection}",
                (hidden, hidden, 1),
                (hidden,),
            )
        for norm_group in ("norm_layers_1", "norm_layers_2"):
            norm = f"model.enc_p.encoder.{norm_group}.{layer}"
            expected[f"{norm}.gamma"] = (hidden,)
            expected[f"{norm}.beta"] = (hidden,)
        ffn = f"model.enc_p.encoder.ffn_layers.{layer}"
        _pair(expected, f"{ffn}.conv_1", (filters, hidden, 3), (filters,))
        _pair(expected, f"{ffn}.conv_2", (hidden, filters, 3), (hidden,))

    _pair(
        expected,
        "model.enc_p.proj",
        (2 * inter, hidden, 1),
        (2 * inter,),
    )
    _pair(expected, "model.dp.conv_1", (256, hidden, 3), (256,))
    expected["model.dp.norm_1.gamma"] = (256,)
    expected["model.dp.norm_1.beta"] = (256,)
    _pair(expected, "model.dp.conv_2", (256, 256, 3), (256,))
    expected["model.dp.norm_2.gamma"] = (256,)
    expected["model.dp.norm_2.beta"] = (256,)
    _pair(expected, "model.dp.proj", (1, 256, 1), (1,))

    _pair(
        expected,
        "model.dec.conv_pre",
        (upsample, inter, 7),
        (upsample,),
    )
    channels = upsample
    for stage, (rate, kernel) in enumerate(
        zip(model["upsample_rates"], model["upsample_kernel_sizes"], strict=True)
    ):
        output_channels = channels // 2
        _pair(
            expected,
            f"model.dec.ups.{stage}",
            (channels, output_channels, int(kernel)),
            (output_channels,),
        )
        for block, block_kernel in enumerate(model["resblock_kernel_sizes"]):
            block_index = stage * 3 + block
            for stack in ("convs1", "convs2"):
                for layer in range(3):
                    _pair(
                        expected,
                        f"model.dec.resblocks.{block_index}.{stack}.{layer}",
                        (output_channels, output_channels, int(block_kernel)),
                        (output_channels,),
                    )
        channels = output_channels
    expected["model.dec.conv_post.weight"] = (1, channels, 7)

    for flow in (0, 2, 4, 6):
        prefix = f"model.flow.flows.{flow}"
        _pair(expected, f"{prefix}.pre", (hidden, half, 1), (hidden,))
        for layer in range(4):
            _pair(
                expected,
                f"{prefix}.enc.in_layers.{layer}",
                (2 * hidden, hidden, 5),
                (2 * hidden,),
            )
            output_channels = hidden if layer == 3 else 2 * hidden
            _pair(
                expected,
                f"{prefix}.enc.res_skip_layers.{layer}",
                (output_channels, hidden, 1),
                (output_channels,),
            )
        _pair(expected, f"{prefix}.post", (half, hidden, 1), (half,))

    if len(expected) != 302:
        raise AssertionError(f"internal tensor inventory has {len(expected)} entries")
    return expected


def _read_initializers(paths: list[Path]) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for path in paths:
        model = onnx.load(path, load_external_data=True)
        for initializer in model.graph.initializer:
            if initializer.name in tensors:
                raise ValueError(f"duplicate ONNX initializer: {initializer.name}")
            value = numpy_helper.to_array(initializer)
            if value.dtype != np.float32:
                raise ValueError(
                    f"{initializer.name} is {value.dtype}; only FP32 is supported"
                )
            tensors[initializer.name] = np.ascontiguousarray(value)
    return tensors


def convert(
    duration_path: Path,
    decode_path: Path,
    config_path: Path,
    output_path: Path,
) -> str:
    config, variant = _load_config(config_path)
    expected = _expected_tensors(config)
    tensors = _read_initializers([duration_path, decode_path])

    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("invalid ONNX tensor inventory; " + "; ".join(details))
    for name, shape in expected.items():
        actual = tuple(int(value) for value in tensors[name].shape)
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, expected {shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output_path,
        metadata={
            "format": "pt",
            "family": "inflect_v2",
            "variant": variant,
            "precision": "fp32",
            "source": "official ONNX export",
        },
    )
    return variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variant = convert(args.duration, args.decode, args.config, args.output)
    print(f"Wrote {variant} FP32 weights to {args.output}")


if __name__ == "__main__":
    main()
