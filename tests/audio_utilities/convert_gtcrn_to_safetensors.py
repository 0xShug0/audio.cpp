#!/usr/bin/env python3
"""Convert official GTCRN checkpoints and parity fixtures to safetensors."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
HALF_CHANNELS = 8
CHANNELS = 16

CHECKPOINTS = {
    "gtcrn_dns3": "checkpoints/model_trained_on_dns3.tar",
    "gtcrn_vctk": "checkpoints/model_trained_on_vctk.tar",
    "gtcrn_streaming": "stream/onnx_models/model_trained_on_dns3.tar",
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folded_batch_norm(state: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    weight = state[f"{prefix}.weight"].float()
    bias = state[f"{prefix}.bias"].float()
    mean = state[f"{prefix}.running_mean"].float()
    var = state[f"{prefix}.running_var"].float()
    scale = weight / torch.sqrt(var + 1.0e-5)
    return scale.contiguous(), (bias - mean * scale).contiguous()


def conv_transpose_as_conv2d_weight(weight: torch.Tensor, in_channels: int, out_channels: int, groups: int) -> torch.Tensor:
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    converted = torch.empty(out_channels, in_per_group, weight.shape[2], weight.shape[3], dtype=weight.dtype)
    for group in range(groups):
        source = weight[group * in_per_group : (group + 1) * in_per_group]
        converted[group * out_per_group : (group + 1) * out_per_group] = source.permute(1, 0, 2, 3)
    return torch.flip(converted, dims=(-2, -1)).contiguous()


def tensor_map_for_runtime(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        tensors[name] = tensor.detach().cpu().float().contiguous().clone()

    for name in list(state):
        if name.endswith(".bn.weight") or name.endswith(".point_bn1.weight") or name.endswith(".depth_bn.weight") or name.endswith(".point_bn2.weight"):
            prefix = name[: -len(".weight")]
            scale, bias = folded_batch_norm(state, prefix)
            tensors[f"{prefix}.__bn_scale"] = scale
            tensors[f"{prefix}.__bn_bias"] = bias

    for dp in ("dpgrnn1", "dpgrnn2"):
        for rnn in ("rnn1", "rnn2"):
            for kind in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                source = f"{dp}.intra_rnn.{rnn}.{kind}_reverse"
                target = f"{dp}.intra_rnn.{rnn}_reverse.{kind}"
                tensors[target] = state[source].detach().cpu().float().contiguous().clone()

    for index in range(3):
        prefix = f"decoder.de_convs.{index}"
        tensors[f"{prefix}.point_conv1.__stream_weight"] = conv_transpose_as_conv2d_weight(
            state[f"{prefix}.point_conv1.weight"],
            HALF_CHANNELS * 3,
            CHANNELS,
            1,
        )
        tensors[f"{prefix}.point_conv2.__stream_weight"] = conv_transpose_as_conv2d_weight(
            state[f"{prefix}.point_conv2.weight"],
            CHANNELS,
            HALF_CHANNELS,
            1,
        )
        tensors[f"{prefix}.depth_conv.__stream_weight"] = conv_transpose_as_conv2d_weight(
            state[f"{prefix}.depth_conv.weight"],
            CHANNELS,
            CHANNELS,
            CHANNELS,
        )
    tensors["decoder.de_convs.3.conv.__stream_weight"] = conv_transpose_as_conv2d_weight(
        state["decoder.de_convs.3.conv.weight"],
        CHANNELS,
        CHANNELS,
        2,
    )
    tensors["decoder.de_convs.4.conv.__stream_weight"] = conv_transpose_as_conv2d_weight(
        state["decoder.de_convs.4.conv.weight"],
        CHANNELS,
        2,
        1,
    )
    return tensors


def make_case(samples: int, base_hz: float, noise_hz: float) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float32) / float(SAMPLE_RATE)
    clean = (
        0.18 * torch.sin(2.0 * torch.pi * base_hz * t)
        + 0.06 * torch.sin(2.0 * torch.pi * (base_hz * 1.9) * t + 0.27)
    )
    noise = (
        0.045 * torch.sin(2.0 * torch.pi * noise_hz * t + 0.63)
        + 0.018 * torch.sin(2.0 * torch.pi * (noise_hz * 1.7) * t + 1.41)
    )
    envelope = torch.linspace(0.35, 1.0, samples, dtype=torch.float32)
    return ((clean + noise) * envelope).contiguous()


def stream_reference_outputs(stream_model, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.hann_window(WIN_LENGTH).pow(0.5)
    spec = torch.stft(
        waveform,
        N_FFT,
        HOP_LENGTH,
        WIN_LENGTH,
        window,
        return_complex=False,
    )[None]
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)
    outputs = []
    with torch.no_grad():
        for frame in range(spec.shape[2]):
            yi, conv_cache, tra_cache, inter_cache = stream_model(
                spec[:, :, frame : frame + 1, :],
                conv_cache,
                tra_cache,
                inter_cache,
            )
            outputs.append(yi)
    enhanced = torch.cat(outputs, dim=2)
    wav = torch.istft(
        torch.view_as_complex(enhanced.contiguous()),
        N_FFT,
        HOP_LENGTH,
        WIN_LENGTH,
        window,
        length=waveform.numel(),
    )
    return (
        wav.detach().cpu().contiguous(),
        spec[:, :, 0, :].detach().cpu().contiguous(),
        outputs[0][:, :, 0, :].detach().cpu().contiguous(),
    )


def load_stream_model(reference_root: Path, state: dict[str, torch.Tensor]):
    sys.path.insert(0, str(reference_root / "stream"))
    gtcrn_offline = import_module(reference_root / "gtcrn.py", "gtcrn_official")
    gtcrn_stream = import_module(reference_root / "stream" / "gtcrn_stream.py", "gtcrn_stream_official")
    convert_module = import_module(reference_root / "stream" / "modules" / "convert.py", "gtcrn_convert_official")
    offline = gtcrn_offline.GTCRN().eval()
    offline.load_state_dict(state)
    stream_model = gtcrn_stream.StreamGTCRN().eval()
    convert_module.convert_to_stream(stream_model, offline)
    return stream_model


def convert(reference_root: Path, output_dir: Path, fixture_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    cases = (make_case(4096, 220.0, 1900.0), make_case(12000, 315.0, 2750.0))
    for model_name, checkpoint_relative in CHECKPOINTS.items():
        checkpoint = reference_root / checkpoint_relative
        payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"GTCRN checkpoint is missing model state: {checkpoint}")
        state = payload["model"]
        if not isinstance(state, dict):
            raise RuntimeError(f"GTCRN model state is not a dict: {checkpoint}")
        save_file(
            tensor_map_for_runtime(state),
            str(output_dir / f"{model_name}.safetensors"),
            metadata={
                "model": "GTCRN",
                "checkpoint": model_name,
                "source": "https://github.com/Xiaobin-Rong/gtcrn",
                "license": "Apache-2.0",
                "sample_rate": str(SAMPLE_RATE),
                "streaming_frame": "1 STFT frame",
            },
        )

        stream_model = load_stream_model(reference_root, state)
        for index, waveform in enumerate(cases):
            output, spec_frame0, output_frame0 = stream_reference_outputs(stream_model, waveform)
            save_file(
                {
                    "input": waveform.reshape(1, -1).contiguous(),
                    "output": output.reshape(1, -1).contiguous(),
                    "spec_frame0": spec_frame0,
                    "output_frame0": output_frame0,
                },
                str(fixture_dir / f"{model_name}_case{index}.safetensors"),
                metadata={
                    "source": "GTCRN official StreamGTCRN PyTorch reference",
                    "checkpoint": model_name,
                    "case": str(index),
                    "sample_rate": str(SAMPLE_RATE),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/framework/audio_utilities/gtcrn"))
    parser.add_argument("--fixture-dir", type=Path, default=Path("tests/assets/framework/audio_utilities/gtcrn"))
    args = parser.parse_args()
    convert(args.reference_root, args.output_dir, args.fixture_dir)


if __name__ == "__main__":
    main()
