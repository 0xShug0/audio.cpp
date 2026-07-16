#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

# Prefer the bundled CUDA libraries in the active conda env over any
# incompatible system cuDNN injected through LD_LIBRARY_PATH.
os.environ.pop("LD_LIBRARY_PATH", None)

import torch
import torch.nn.functional as F
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERF_CASES = REPO_ROOT / "tools" / "perf" / "model_perf_cases.json"
DEFAULT_REQUEST_CASES = REPO_ROOT / "tools" / "perf" / "model_perf_request_cases.json"


@dataclass
class PerfCase:
    id: str
    family: str
    case_id: str
    request_id: str
    warmup_request_id: str | None = None
    request_overrides: dict[str, Any] | None = None
    warmup_request_overrides: dict[str, Any] | None = None
    request_remove_keys: list[str] | None = None
    warmup_request_remove_keys: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial Python performance harness for audio.cpp model references.")
    parser.add_argument("--perf-cases", type=Path, default=DEFAULT_PERF_CASES)
    parser.add_argument("--request-cases", type=Path, default=DEFAULT_REQUEST_CASES)
    parser.add_argument("--id", default="")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_absolute_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def shorten_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut < max(8, limit // 2):
        cut = limit
    return text[:cut].rstrip()


def read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    return pcm, sample_rate


def write_wav_mono_f32(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def write_clipped_wav(source: Path, temp_dir: Path, seconds: float) -> Path:
    samples, sample_rate = read_wav_mono_f32(source)
    limit = max(1, min(len(samples), int(round(sample_rate * seconds))))
    target = temp_dir / source.name
    write_wav_mono_f32(target, samples[:limit], sample_rate)
    return target


def resolve_request_paths(request: dict[str, Any]) -> dict[str, Any]:
    out = dict(request)
    for key in ("audio", "voice_ref", "source_audio", "target_voice", "prosody_ref", "style_ref"):
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = str(ensure_absolute_path(value))
    return out


def derive_warmup_request(request: dict[str, Any], temp_dir: Path) -> dict[str, Any]:
    out = resolve_request_paths(request)
    for key, limit in (
        ("text", 80),
        ("target_text", 80),
        ("style_ref_text", 80),
        ("reference_text", 80),
        ("lyrics", 80),
        ("instruct", 64),
    ):
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = shorten_text(value, limit)
    for key in ("audio", "voice_ref", "source_audio", "target_voice", "prosody_ref", "style_ref"):
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = str(write_clipped_wav(Path(value), temp_dir, 2.0))
    if isinstance(out.get("duration_seconds"), (int, float)):
        out["duration_seconds"] = min(float(out["duration_seconds"]), 4.0)
    if isinstance(out.get("max_tokens"), (int, float)):
        out["max_tokens"] = min(int(out["max_tokens"]), 64)
    if isinstance(out.get("num_inference_steps"), (int, float)):
        out["num_inference_steps"] = min(int(out["num_inference_steps"]), 8)
    return out


def audio_duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    return float(len(np.asarray(audio).reshape(-1))) / float(sample_rate) if sample_rate > 0 else 0.0


def input_audio_duration_sec(request: dict[str, Any]) -> float:
    for key in ("audio", "source_audio"):
        value = request.get(key)
        if isinstance(value, str) and value:
            samples, sample_rate = read_wav_mono_f32(Path(value))
            return audio_duration_sec(samples, sample_rate)
    return 0.0


def load_perf_cases(path: Path) -> list[PerfCase]:
    root = read_json(path)
    return [
        PerfCase(
            id=item["id"],
            family=item["family"],
            case_id=item["case_id"],
            request_id=item["request_id"],
            warmup_request_id=item.get("warmup_request_id"),
            request_overrides=item.get("request_overrides"),
            warmup_request_overrides=item.get("warmup_request_overrides"),
            request_remove_keys=item.get("request_remove_keys"),
            warmup_request_remove_keys=item.get("warmup_request_remove_keys"),
        )
        for item in root["cases"]
    ]


def find_case(request_cases: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in request_cases["cases"]:
        if item["id"] == case_id:
            return item
    raise KeyError(case_id)


def find_request(case: dict[str, Any], request_id: str) -> dict[str, Any]:
    for item in case["requests"]:
        if item["id"] == request_id:
            return item
    raise KeyError(f"{case['id']}:{request_id}")


def materialize_perf_request(
    request: dict[str, Any],
    overrides: dict[str, Any] | None,
    remove_keys: list[str] | None,
) -> dict[str, Any]:
    out = dict(request)
    for key in remove_keys or []:
        out.pop(key, None)
    if overrides:
        out.update(overrides)
    return out


def set_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(max(1, threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, threads))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def use_transformers457() -> None:
    import transformers457 as _transformers457
    import transformers457.dynamic_module_utils as _dynamic_module_utils

    sys.modules["transformers"] = _transformers457
    _dynamic_module_utils.transformers = _transformers457


def enable_legacy_llama_config_positional_args() -> None:
    import transformers

    original = transformers.LlamaConfig
    if getattr(original, "_audiocpp_legacy_positional_enabled", False):
        return

    class LegacyLlamaConfig(original):
        _audiocpp_legacy_positional_enabled = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if args:
                if len(args) != 5:
                    raise TypeError(f"Legacy LlamaConfig positional compatibility expects 5 args, got {len(args)}")
                vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads = args
                kwargs = {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "num_hidden_layers": num_hidden_layers,
                    "num_attention_heads": num_attention_heads,
                    **kwargs,
                }
            super().__init__(**kwargs)

    transformers.LlamaConfig = LegacyLlamaConfig


def force_attention_implementation(module: object, value: str) -> None:
    if hasattr(module, "attn_implementation"):
        setattr(module, "attn_implementation", value)
    if hasattr(module, "config"):
        config = getattr(module, "config")
        if hasattr(config, "attn_implementation"):
            setattr(config, "attn_implementation", value)
        if hasattr(config, "local_transformer_attn_implementation"):
            setattr(config, "local_transformer_attn_implementation", value)
        if hasattr(config, "gpt2_config") and getattr(config, "gpt2_config") is not None:
            gpt2_config = getattr(config, "gpt2_config")
            if hasattr(gpt2_config, "_attn_implementation"):
                setattr(gpt2_config, "_attn_implementation", value)
            if hasattr(gpt2_config, "attn_implementation"):
                setattr(gpt2_config, "attn_implementation", value)
    named_children = getattr(module, "named_children", None)
    if callable(named_children):
        for _, child in named_children():
            force_attention_implementation(child, value)


def resample_linear_mono(waveform: np.ndarray, input_sample_rate: int, output_sample_rate: int) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return waveform
    scale = output_sample_rate / input_sample_rate
    output_samples = int(round(len(waveform) * scale))
    pos = np.arange(output_samples, dtype=np.float64) / scale
    left = np.floor(pos).astype(np.int64)
    right = np.minimum(left + 1, len(waveform) - 1)
    frac = (pos - left).astype(np.float32)
    return waveform[left] * (1.0 - frac) + waveform[right] * frac


class RefCitrinet:
    LOG_EPS = float(np.ldexp(1.0, -24))
    BN_EPS = 1.0e-3
    NORM_EPS = 1.0e-5

    def __init__(self, model_dir: Path, device: torch.device):
        model_path = model_dir / "citrinet_256.safetensors"
        self.device = device
        self.tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(model_path), framework="pt", device="cpu") as f:
            self.metadata = {
                key: json.loads(value)
                if value.startswith(("[", "{", "\"")) or value in {"true", "false", "null"} or value.lstrip("-").isdigit()
                else value
                for key, value in f.metadata().items()
            }
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key).float().contiguous().to(device)
        self.sample_rate = int(self.metadata["sample_rate"])
        self.n_mels = int(self.metadata["n_mels"])
        self.n_fft = int(self.metadata["n_fft"])
        self.hop = int(round(float(self.metadata["window_stride"]) * self.sample_rate))
        self.win = int(round(float(self.metadata["window_size"]) * self.sample_rate))
        self.pad_to = int(self.metadata["pad_to"])
        self.blank_id = int(self.metadata["blank_id"])
        self.jasper = list(self.metadata["jasper"])
        self.window = self.tensors["preprocessor.featurizer.window"]
        self.fb = self.tensors["preprocessor.featurizer.fb"].view(self.n_mels, self.n_fft // 2 + 1)
        vocab_path = model_dir / str(self.metadata["vocab_file"])
        self.vocab = [piece[:-1] if piece.endswith("\r") else piece for piece in vocab_path.read_text().split("\n")]

    def bn(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = self.tensors[f"{prefix}.weight"].view(1, -1, 1)
        b = self.tensors[f"{prefix}.bias"].view(1, -1, 1)
        mean = self.tensors[f"{prefix}.running_mean"].view(1, -1, 1)
        var = self.tensors[f"{prefix}.running_var"].view(1, -1, 1)
        return (x - mean) / torch.sqrt(var + self.BN_EPS) * w + b

    def conv1d(self, x: torch.Tensor, prefix: str, stride: int, dilation: int, padding: int, groups: int, bias: bool) -> torch.Tensor:
        weight = self.tensors[f"{prefix}.weight"]
        if weight.ndim == 2:
            weight = weight.unsqueeze(-1)
        bias_t = self.tensors[f"{prefix}.bias"] if bias else None
        return F.conv1d(x, weight, bias=bias_t, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def squeeze_excite(self, x: torch.Tensor, block: int, repeat: int) -> torch.Tensor:
        se_index = 3 if repeat == 1 else repeat * 5 - 2
        y = x.mean(dim=-1, keepdim=True)
        y = self.conv1d(y, f"encoder.encoder.{block}.mconv.{se_index}.fc.0", 1, 1, 0, 1, False)
        y = F.relu(y)
        y = self.conv1d(y, f"encoder.encoder.{block}.mconv.{se_index}.fc.2", 1, 1, 0, 1, False)
        return x * torch.sigmoid(y)

    def block(self, x: torch.Tensor, block_index: int, cfg: dict) -> torch.Tensor:
        residual_x = x
        repeat = int(cfg["repeat"])
        kernel = int(cfg["kernel"])
        dilation = int(cfg["dilation"])
        for r in range(repeat):
            base = r * 5 if cfg["separable"] else r * 3
            stride = int(cfg["stride"]) if r + 1 == repeat else 1
            padding = dilation * (kernel - 1) // 2
            if cfg["separable"]:
                x = self.conv1d(x, f"encoder.encoder.{block_index}.mconv.{base}.conv", stride, dilation, padding, x.shape[1], False)
                x = self.conv1d(x, f"encoder.encoder.{block_index}.mconv.{base + 1}.conv", 1, 1, 0, 1, False)
                x = self.bn(x, f"encoder.encoder.{block_index}.mconv.{base + 2}")
            else:
                x = self.conv1d(x, f"encoder.encoder.{block_index}.mconv.{base}.conv", stride, dilation, padding, 1, False)
                x = self.bn(x, f"encoder.encoder.{block_index}.mconv.{base + 1}")
            if r + 1 != repeat:
                x = F.relu(x)
        if cfg.get("se", False):
            x = self.squeeze_excite(x, block_index, repeat)
        if cfg["residual"]:
            residual_stride = int(cfg["stride"]) if cfg.get("residual_mode") == "stride_add" else 1
            res = self.conv1d(residual_x, f"encoder.encoder.{block_index}.res.0.0.conv", residual_stride, 1, 0, 1, False)
            res = self.bn(res, f"encoder.encoder.{block_index}.res.0.1")
            x = x + res
        return F.relu(x)

    def compute_features(self, waveform: np.ndarray) -> tuple[torch.Tensor, int]:
        xt = torch.from_numpy(waveform.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        stft = torch.stft(xt, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window, center=True, pad_mode="constant", normalized=False, onesided=True, return_complex=True)
        mel = torch.matmul(self.fb, stft.abs().pow(2.0)[0]).transpose(0, 1)
        feats = torch.log(mel + self.LOG_EPS)
        raw_frames = int(feats.shape[0])
        mean = feats.mean(dim=0, keepdim=True)
        var = ((feats - mean) ** 2).sum(dim=0, keepdim=True) / max(raw_frames - 1, 1)
        feats = (feats - mean) / (torch.sqrt(var) + self.NORM_EPS)
        if self.pad_to > 1:
            padded = ((raw_frames + self.pad_to - 1) // self.pad_to) * self.pad_to
            if padded != raw_frames:
                feats = F.pad(feats, (0, 0, 0, padded - raw_frames))
        return feats, raw_frames

    def output_frames(self, input_frames: int) -> int:
        frames = input_frames
        for cfg in self.jasper:
            stride = int(cfg["stride"])
            if stride > 1:
                frames = (frames + stride - 1) // stride
        return frames

    def infer_features(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float().t().unsqueeze(0)
        for block_index, cfg in enumerate(self.jasper):
            x = self.block(x, block_index, cfg)
        logits = self.conv1d(x, "decoder.decoder_layers.0", 1, 1, 0, 1, True)
        return logits.squeeze(0).transpose(0, 1).contiguous()

    def greedy_decode(self, logits: torch.Tensor) -> str:
        rows = logits.detach().cpu().numpy()
        ids: list[int] = []
        prev = -1
        for row in rows:
            best = int(np.argmax(row))
            if best == prev:
                continue
            prev = best
            if best != self.blank_id:
                ids.append(best)
        text = ""
        for idx in ids:
            piece = self.vocab[idx]
            if piece.startswith("##"):
                text += piece[2:]
            elif not text or text[-1] in {"'", "-"} or piece in {".", ",", "!", "?", ":", ";", "'", "\"", ")", "]", "}", "-", "/", "\\"}:
                text += piece
            else:
                text += " " + piece
        return text

    def transcribe(self, audio_path: Path) -> str:
        waveform, sample_rate = read_wav_mono_f32(audio_path)
        waveform = resample_linear_mono(waveform, sample_rate, self.sample_rate)
        features, raw_frames = self.compute_features(waveform)
        logits = self.infer_features(features)
        return self.greedy_decode(logits[: self.output_frames(raw_frames)])


class RefMarbleNet:
    FEATURE_DIM = 80
    TARGET_SR = 16000
    N_FFT = 512
    HOP = 160
    WIN = 400
    LOG_EPS = float(np.ldexp(1.0, -24))
    BN_EPS = 1.0e-3
    OUTPUT_STRIDE = 2

    def __init__(self, model_dir: Path, device: torch.device):
        self.device = device
        self.tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(model_dir / "marblenet_vad.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key).float().contiguous().to(device)
        self.window = self.tensors["preprocessor.featurizer.window"]
        self.fb = self.tensors["preprocessor.featurizer.fb"].view(self.FEATURE_DIM, self.N_FFT // 2 + 1)

    def bn(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = self.tensors[f"{prefix}.weight"].view(1, -1, 1)
        b = self.tensors[f"{prefix}.bias"].view(1, -1, 1)
        mean = self.tensors[f"{prefix}.running_mean"].view(1, -1, 1)
        var = self.tensors[f"{prefix}.running_var"].view(1, -1, 1)
        return (x - mean) / torch.sqrt(var + self.BN_EPS) * w + b

    def conv1d(self, x: torch.Tensor, prefix: str, stride: int, dilation: int, padding: int, groups: int, bias: bool) -> torch.Tensor:
        weight = self.tensors[f"{prefix}.weight"]
        if weight.ndim == 2:
            weight = weight.unsqueeze(-1)
        bias_t = self.tensors[f"{prefix}.bias"] if bias else None
        return F.conv1d(x, weight, bias=bias_t, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def block(self, x: torch.Tensor, block: int, repeat: int, kernel: int, stride: int, dilation: int, residual: bool, separable: bool) -> torch.Tensor:
        residual_x = x
        for r in range(repeat):
            if separable:
                base = r * 5
                x = self.conv1d(x, f"encoder.encoder.{block}.mconv.{base}.conv", stride, dilation, dilation * (kernel - 1) // 2, x.shape[1], False)
                x = self.conv1d(x, f"encoder.encoder.{block}.mconv.{base + 1}.conv", 1, 1, 0, 1, False)
                x = self.bn(x, f"encoder.encoder.{block}.mconv.{base + 2}")
            else:
                base = r * 3
                x = self.conv1d(x, f"encoder.encoder.{block}.mconv.{base}.conv", stride, dilation, dilation * (kernel - 1) // 2, 1, False)
                x = self.bn(x, f"encoder.encoder.{block}.mconv.{base + 1}")
            if r + 1 != repeat:
                x = F.relu(x)
        if residual:
            res = self.conv1d(residual_x, f"encoder.encoder.{block}.res.0.0.conv", 1, 1, 0, 1, False)
            res = self.bn(res, f"encoder.encoder.{block}.res.0.1")
            x = x + res
        return F.relu(x)

    def detect(self, audio_path: Path, threshold: float = 0.5) -> list[dict[str, float | int]]:
        waveform, sample_rate = read_wav_mono_f32(audio_path)
        waveform = resample_linear_mono(waveform, sample_rate, self.TARGET_SR)
        xt = torch.from_numpy(waveform.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        stft = torch.stft(xt, n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN, window=self.window, center=True, pad_mode="constant", normalized=False, onesided=True, return_complex=True)
        mel = torch.matmul(self.fb, stft.abs().pow(2.0)[0]).transpose(0, 1)
        feats = torch.log(mel + self.LOG_EPS)
        if feats.shape[0] % 2 != 0:
            feats = F.pad(feats, (0, 0, 0, 1))
        x = feats.float().t().unsqueeze(0)
        x = self.block(x, 0, repeat=1, kernel=11, stride=2, dilation=1, residual=False, separable=True)
        x = self.block(x, 1, repeat=2, kernel=13, stride=1, dilation=1, residual=True, separable=True)
        x = self.block(x, 2, repeat=2, kernel=15, stride=1, dilation=1, residual=True, separable=True)
        x = self.block(x, 3, repeat=2, kernel=17, stride=1, dilation=1, residual=True, separable=True)
        x = self.block(x, 4, repeat=1, kernel=29, stride=1, dilation=2, residual=False, separable=True)
        x = self.block(x, 5, repeat=1, kernel=1, stride=1, dilation=1, residual=False, separable=False)
        logits = self.conv1d(x, "decoder.layer0", 1, 1, 0, 1, True).squeeze(0).transpose(0, 1).contiguous().detach().cpu().numpy()
        if logits.shape[1] == 1:
            probabilities = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        else:
            shifted = logits[:, :2] - logits[:, :2].max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probabilities = exp[:, 1] / exp.sum(axis=1)
        segments: list[dict[str, float | int]] = []
        active = False
        start_frame = 0
        confidence_sum = 0.0
        confidence_count = 0
        for frame, probability in enumerate(probabilities):
            if float(probability) >= threshold:
                if not active:
                    active = True
                    start_frame = frame
                    confidence_sum = 0.0
                    confidence_count = 0
                confidence_sum += float(probability)
                confidence_count += 1
                continue
            if active:
                segments.append({
                    "start_sample": int(start_frame * self.HOP * self.OUTPUT_STRIDE * sample_rate / self.TARGET_SR),
                    "end_sample": int(frame * self.HOP * self.OUTPUT_STRIDE * sample_rate / self.TARGET_SR),
                    "confidence": confidence_sum / confidence_count,
                })
                active = False
        if active:
            segments.append({
                "start_sample": int(start_frame * self.HOP * self.OUTPUT_STRIDE * sample_rate / self.TARGET_SR),
                "end_sample": int(len(probabilities) * self.HOP * self.OUTPUT_STRIDE * sample_rate / self.TARGET_SR),
                "confidence": confidence_sum / confidence_count,
            })
        return segments


RunnerFn = Callable[[dict[str, Any], argparse.Namespace], tuple[float, float]]
LoaderFn = Callable[[dict[str, Any], argparse.Namespace], tuple[Any, RunnerFn]]


def qwen3_asr_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    use_transformers457()
    sys.path.insert(0, str(REPO_ROOT / "reference" / "Qwen3-ASR"))
    from qwen_asr import Qwen3ASRModel

    device_map = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model = Qwen3ASRModel.from_pretrained(
        str(ensure_absolute_path(case["model"])),
        device_map=device_map,
    )

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        model.transcribe(
            audio=request["audio"],
            context=request.get("context", ""),
            language=request.get("language") or None,
            return_time_stamps=False,
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def ace_step_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "ACE-Step-1.5"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    backend = "cuda" if args.backend == "cuda" else "cpu"
    checkpoint_dir = ensure_absolute_path(case["model"])
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoint_dir)
    dit_model_path = case.get("load_options", {}).get("ace_step.dit_model_path", "acestep-v15-turbo")
    lm_model_path = case.get("load_options", {}).get("ace_step.lm_model_path", "acestep-5Hz-lm-1.7B")

    dit_handler = AceStepHandler()
    init_message, ok = dit_handler.initialize_service(
        project_root=str(checkpoint_dir),
        config_path=dit_model_path,
        device=backend,
        force_dtype=None,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step DiT init failed: {init_message}")

    llm_handler = LLMHandler()
    init_message, ok = llm_handler.initialize(
        checkpoint_dir=str(checkpoint_dir),
        lm_model_path=lm_model_path,
        backend="pt",
        device=backend,
        offload_to_cpu=False,
        dtype=None,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step LM init failed: {init_message}")

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        options = dict(request.get("options", {}))
        params_dict = {
            "task_type": request.get("task_route", "text2music"),
            "caption": request.get("text", ""),
            "lyrics": request.get("lyrics", ""),
            "vocal_language": request.get("language", "unknown"),
            "duration": request.get("duration_seconds"),
            "inference_steps": request.get("num_inference_steps"),
            "guidance_scale": request.get("guidance_scale"),
            "seed": request.get("seed"),
            "src_audio": request.get("audio"),
            "reference_audio": request.get("reference_audio"),
            "instruction": request.get("instruction"),
            "audio_codes": request.get("audio_codes", ""),
            "repainting_start": request.get("repaint_start"),
            "repainting_end": request.get("repaint_end"),
            "lm_cfg_scale": options.get("lm_cfg_scale"),
            "lm_temperature": options.get("lm_temperature"),
            "lm_top_k": options.get("lm_top_k"),
            "lm_top_p": options.get("lm_top_p"),
        }
        negative_prompt = request.get("negative_prompt")
        if negative_prompt is not None:
            params_dict["lm_negative_prompt"] = negative_prompt
        params_dict = {key: value for key, value in params_dict.items() if value is not None}
        params = GenerationParams(**params_dict)
        config = GenerationConfig(
            batch_size=int(request.get("batch_size", 1)),
            allow_lm_batch=False,
            use_random_seed=not ("seed" in request),
            seeds=[int(request.get("seed", 0))],
            noise_file=None,
            lm_batch_chunk_size=1,
            constrained_decoding_debug=False,
            audio_format="wav",
        )
        started = time.perf_counter()
        results = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        if not results.success or not results.audios:
            raise RuntimeError(results.error or "ACE-Step returned no output")
        first_audio = results.audios[0]
        audio = np.asarray(first_audio["tensor"], dtype=np.float32).reshape(-1)
        sample_rate = int(first_audio["sample_rate"])
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, sample_rate)

    return (dit_handler, llm_handler), run


def qwen3_forced_aligner_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    use_transformers457()
    sys.path.insert(0, str(REPO_ROOT / "reference" / "Qwen3-ASR"))
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    device_map = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model = Qwen3ForcedAligner.from_pretrained(
        str(ensure_absolute_path(case["model"])),
        device_map=device_map,
    )

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        model.align(
            audio=request["audio"],
            text=request["text"],
            language=request.get("language") or "English",
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def qwen3_tts_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    use_transformers457()
    sys.path.insert(0, str(REPO_ROOT / "reference" / "Qwen3-TTS"))
    from qwen_tts import Qwen3TTSModel

    device_map = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        str(ensure_absolute_path(case["model"])),
        device_map=device_map,
    )

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        if case["task"] == "vdes":
            audio_list, sample_rate = model.generate_voice_design(
                text=request["text"],
                instruct=request["instruct"],
                language=request.get("language", "English"),
                max_new_tokens=int(request.get("max_tokens", 512)),
                do_sample=bool(request.get("do_sample", True)),
            )
        elif request.get("speaker"):
            audio_list, sample_rate = model.generate_custom_voice(
                text=request["text"],
                speaker=request["speaker"],
                instruct=request.get("instruct", ""),
                language=request.get("language", "English"),
                max_new_tokens=int(request.get("max_tokens", 512)),
                do_sample=bool(request.get("do_sample", True)),
            )
        else:
            prompt = model.create_voice_clone_prompt(
                ref_audio=request["voice_ref"],
                ref_text=request["reference_text"],
            )
            audio_list, sample_rate = model.generate_voice_clone(
                text=request["text"],
                voice_clone_prompt=prompt,
                language=request.get("language", "English"),
                max_new_tokens=int(request.get("max_tokens", 512)),
                do_sample=bool(request.get("do_sample", True)),
            )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(np.asarray(audio_list[0], dtype=np.float32), int(sample_rate))

    return model, run


def chatterbox_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    torch.backends.cudnn.enabled = False
    sys.path.insert(0, str(REPO_ROOT / "reference" / "chatterbox" / "src"))
    language = "en"
    request_language = ""
    if case["requests"]:
        request_language = case["requests"][0].get("language", "en")
    if request_language.lower() == "en":
        from chatterbox.tts import ChatterboxTTS
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTS
        language = request_language
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model = ChatterboxTTS.from_local(ensure_absolute_path(case["model"]), device=device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        if language == "en":
            wav = model.generate(request["text"], audio_prompt_path=request["voice_ref"])
        else:
            wav = model.generate(request["text"], language_id=language, audio_prompt_path=request["voice_ref"])
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio = wav.detach().cpu().numpy().astype(np.float32)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, int(model.sr))

    return model, run


def miocodec_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    sys.path.insert(0, str(REPO_ROOT / "reference" / "MioCodec" / "src"))
    from miocodec import MioCodecModel, load_audio

    model_dir = ensure_absolute_path(case["model"])
    model = MioCodecModel.from_pretrained(
        config_path=str(model_dir / "config.yaml"),
        weights_path=str(model_dir / "model.safetensors"),
    ).eval()
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model = model.to(device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        source = load_audio(request["audio"], sample_rate=model.config.sample_rate).to(device)
        target = load_audio(request["voice_ref"], sample_rate=model.config.sample_rate).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            decoded = model.voice_conversion(source, target)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio = decoded.detach().cpu().float().numpy()
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, int(model.config.sample_rate))

    return model, run


def miotts_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "MioTTS-Inference"
    codec_root = REPO_ROOT / "reference" / "MioCodec" / "src"
    if str(codec_root) not in sys.path:
        sys.path.insert(0, str(codec_root))
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from miocodec import MioCodecModel
    from miotts_server.audio import load_reference_audio_path
    from miotts_server.codec import MioCodecService
    from miotts_server.text import normalize_text
    from miotts_server.token_parser import parse_speech_tokens

    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model_path = ensure_absolute_path(case["model"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    codec = MioCodecService(model_id=str(REPO_ROOT / "models" / "MioCodec-25Hz-44.1kHz-v2"), device=device, presets_dir=REPO_ROOT / "presets")
    codec_model_dir = REPO_ROOT / "models" / "MioCodec-25Hz-44.1kHz-v2"
    codec._codec = MioCodecModel.from_pretrained(
        config_path=str(codec_model_dir / "config.yaml"),
        weights_path=str(codec_model_dir / "model.safetensors"),
    ).eval().to(device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        text = normalize_text(request["text"])
        messages = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        reference_waveform = load_reference_audio_path(request["voice_ref"], codec.sample_rate).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=bool(request.get("do_sample", True)),
                max_new_tokens=int(request.get("max_tokens", 700)),
                temperature=0.8,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            generated_text = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=False)
            speech_tokens = parse_speech_tokens(generated_text)
            audio = codec.synthesize(speech_tokens, reference_waveform=reference_waveform)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.tensor(audio)
        audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio_np, codec.sample_rate)

    return (model, tokenizer, codec), run


def voxcpm2_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    sys.path.insert(0, str(REPO_ROOT / "reference" / "VoxCPM" / "src"))
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(str(ensure_absolute_path(case["model"])), load_denoiser=False)
    if args.backend == "cuda":
        torch.cuda.set_device(args.device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        kwargs = {
            "text": request["text"],
            "cfg_value": float(request.get("guidance_scale", 2.0)),
            "inference_timesteps": int(request.get("num_inference_steps", 10)),
        }
        if request.get("voice_ref"):
            kwargs["reference_wav_path"] = request["voice_ref"]
        started = time.perf_counter()
        wav = model.generate(**kwargs)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio = np.asarray(wav, dtype=np.float32)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, int(model.tts_model.sample_rate))

    return model, run


def kokoro_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch
    from safetensors.torch import load_file

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "kokoro"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from kokoro.pipeline import KPipeline
    from kokoro.model import KModel

    model_dir = ensure_absolute_path(case["model"])
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    checkpoint_path = model_dir / "kokoro-v1_0.safetensors"
    original_torch_load = torch.load

    def patched_torch_load(path, *load_args, **load_kwargs):
        if Path(path) == checkpoint_path:
            flat_state = load_file(str(checkpoint_path), device="cpu")
            grouped: dict[str, dict[str, object]] = {}
            for name, tensor in flat_state.items():
                prefix, rest = name.split(".", 1)
                grouped.setdefault(prefix, {})[rest] = tensor
            return grouped
        return original_torch_load(path, *load_args, **load_kwargs)

    torch.load = patched_torch_load
    try:
        model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(model_dir / "config.json"),
            model=str(checkpoint_path),
        ).to(device).eval()
    finally:
        torch.load = original_torch_load
    voice_id = case.get("session_options", {}).get("voice", "af_heart")
    lang_code = voice_id[0]
    pipeline = KPipeline(lang_code=lang_code, model=model, repo_id="hexgrad/Kokoro-82M")
    voices = json.loads((model_dir / "voices.json").read_text(encoding="utf-8"))
    voice_info = voices[voice_id]
    voice_pack = np.fromfile(model_dir / "voices" / voice_info["path"], dtype=np.float32).reshape(
        int(voice_info["rows"]),
        int(voice_info["cols"]),
    )
    voice_tensor = torch.from_numpy(voice_pack).unsqueeze(1)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        outputs = [
            item
            for item in pipeline(
                request["text"],
                voice=voice_tensor,
                speed=1.0,
                split_pattern=r"\n+",
            )
            if getattr(item, "audio", None) is not None
        ]
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        if not outputs:
            raise RuntimeError("Kokoro produced no audio")
        audio = np.concatenate([np.asarray(item.audio, dtype=np.float32) for item in outputs])
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, 24000)

    return (pipeline, voice_tensor), run


def pocket_tts_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import yaml
    import torch

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "pocket-tts"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from pocket_tts import TTSModel

    model_dir = ensure_absolute_path(case["model"])
    language = case.get("session_options", {}).get("language", "english")
    config_src = ref_root / "pocket_tts" / "config" / f"{language}.yaml"
    config = yaml.safe_load(config_src.read_text(encoding="utf-8"))
    bundle_dir = model_dir / "languages" / language if (model_dir / "languages" / language).exists() else model_dir
    config["weights_path"] = str(bundle_dir / "model.safetensors")
    config["weights_path_without_voice_cloning"] = str(bundle_dir / "model.safetensors")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(bundle_dir / "tokenizer.model")
    config_dir = REPO_ROOT / "build" / "perf" / "pocket_tts"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{language}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model = TTSModel.load_model(config=str(config_path)).to(device).eval()

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        voice_state = model.get_state_for_audio_prompt(request["voice_ref"])
        started = time.perf_counter()
        audio = model.generate_audio(voice_state, request["text"])
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio_np, int(model.sample_rate))

    return model, run


def omnivoice_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "OmniVoice"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from omnivoice.models.omnivoice import OmniVoice

    device_map = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model = OmniVoice.from_pretrained(str(ensure_absolute_path(case["model"])), device_map=device_map)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        audio = model.generate(
            text=request["text"],
            language=request.get("language", "en"),
            ref_audio=request.get("voice_ref"),
            ref_text=request.get("reference_text"),
            instruct=request.get("instruct"),
            num_step=int(request.get("num_inference_steps", 16)),
            guidance_scale=float(request.get("guidance_scale", 2.0)),
            audio_chunk_duration=float(request.get("options", {}).get("audio_chunk_duration", 15.0)),
        )[0]
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio_np = np.asarray(audio, dtype=np.float32)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio_np, int(model.sampling_rate))

    return model, run


def moss_tts_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch
    use_transformers457()
    from transformers import AutoConfig, AutoModelForCausalLM

    set_threads(args.threads)
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model_path = ensure_absolute_path(case["model"])
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    attention_impl = "sdpa" if args.backend == "cuda" else "eager"
    config.attn_implementation = attention_impl
    config.local_transformer_attn_implementation = attention_impl
    if getattr(config, "gpt2_config", None) is not None:
        config.gpt2_config._attn_implementation = attention_impl
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    force_attention_implementation(model, attention_impl)
    audio_tokenizer_path = str(REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano")
    text_tokenizer_path = str(model_path / "tokenizer.model") if (model_path / "tokenizer.model").is_file() else str(model_path)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        out_path = REPO_ROOT / "build" / "perf" / "moss_tts_request.wav"
        started = time.perf_counter()
        model.inference(
            text=request["text"],
            output_audio_path=str(out_path),
            mode="voice_clone",
            prompt_audio_path=request["voice_ref"],
            text_tokenizer_path=text_tokenizer_path,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_path,
            device=device,
            nq=16,
            max_new_frames=int(request.get("max_tokens", 120)),
            do_sample=bool(request.get("do_sample", False)),
            use_kv_cache=True,
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio, sample_rate = read_wav_mono_f32(out_path)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio, sample_rate)

    return model, run


def demucs_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "demucs"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from demucs.api import Separator

    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    model_path = ensure_absolute_path(case["model"])
    model_name = "htdemucs"
    repo_arg: Path | None = model_path
    manifest_path = model_path / "manifest.json"
    if manifest_path.is_file():
        repo_arg = None
    separator = Separator(
        model=model_name,
        repo=repo_arg,
        device=device,
        shifts=0,
        overlap=0.25,
        split=True,
        segment=None,
        jobs=0,
        progress=False,
    )

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        separator.separate_audio_file(Path(request["audio"]))
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return separator, run


def roformer_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import types
    import yaml
    import soundfile as sf
    import torch

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "Mel-Band-Roformer-Vocal-Model"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    if "ml_collections" not in sys.modules:
        class ConfigDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__()
                payload = dict(*args, **kwargs)
                for key, value in payload.items():
                    self[key] = self._convert(value)

            @classmethod
            def _convert(cls, value):
                if isinstance(value, dict):
                    return cls(value)
                if isinstance(value, list):
                    return [cls._convert(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(cls._convert(item) for item in value)
                return value

            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = self._convert(value)
        module = types.ModuleType("ml_collections")
        module.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = module
    from ml_collections import ConfigDict
    from inference import run_folder
    from utils import get_model_from_config

    ckpt_path = ensure_absolute_path("models/melbandroformer/MelBandRoformer.ckpt")
    config_path = REPO_ROOT / "reference" / "Mel-Band-Roformer-Vocal-Model" / "configs" / "config_vocals_mel_band_roformer.yaml"
    config = ConfigDict(yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader))
    model = get_model_from_config("mel_band_roformer", config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model = model.to(device=device).eval()

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        work_dir = REPO_ROOT / "build" / "perf" / "roformer"
        input_dir = work_dir / "input"
        store_dir = work_dir / "store"
        input_dir.mkdir(parents=True, exist_ok=True)
        store_dir.mkdir(parents=True, exist_ok=True)
        staged = input_dir / Path(request["audio"]).name
        shutil.copyfile(request["audio"], staged)
        started = time.perf_counter()
        run_folder(model, argparse.Namespace(input_folder=str(input_dir), store_dir=str(store_dir)), config, device, verbose=False)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def seed_vc_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import os
    import importlib
    import torch
    from types import SimpleNamespace

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "SeedVC"
    sys.path.insert(0, str(ref_root))
    bigvgan_module = importlib.import_module("modules.bigvgan.bigvgan")
    bigvgan_from_pretrained = bigvgan_module.BigVGAN._from_pretrained.__func__

    def patched_bigvgan_from_pretrained(
        cls,
        *,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        map_location="cpu",
        strict=False,
        use_cuda_kernel=False,
        **model_kwargs,
    ):
        return bigvgan_from_pretrained(
            cls,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            use_cuda_kernel=use_cuda_kernel,
            **model_kwargs,
        )

    bigvgan_module.BigVGAN._from_pretrained = classmethod(patched_bigvgan_from_pretrained)
    cwd = os.getcwd()
    os.chdir(ref_root)
    try:
        module = load_module("seedvc_reference_inference_v2_perf", ref_root / "inference_v2.py")
    finally:
        os.chdir(cwd)
    model_root = ensure_absolute_path(case["model"])
    if (model_root / "v2" / "ar.safetensors").is_file():
        official_root = REPO_ROOT / "models" / "SeedVC"
        if official_root.is_dir():
            model_root = official_root
    device = f"cuda:{args.device}" if args.backend == "cuda" else "cpu"
    load_args = SimpleNamespace(
        ar_checkpoint_path=str(model_root / "seed-vc/v2/ar_base.pth"),
        cfm_checkpoint_path=str(model_root / "seed-vc/v2/cfm_small.pth"),
        compile=False,
        output=str(REPO_ROOT / "build" / "perf" / "seedvc"),
    )
    cwd = os.getcwd()
    os.chdir(ref_root)
    try:
        wrapper = module.load_v2_models(load_args)
        module.vc_wrapper_v2 = wrapper
    finally:
        os.chdir(cwd)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        run_args = SimpleNamespace(
            diffusion_steps=int(request.get("num_inference_steps", 30)),
            length_adjust=float(request.get("options", {}).get("length_adjust", 1.0)),
            intelligibility_cfg_rate=float(request.get("options", {}).get("intelligibility_cfg_rate", 0.7)),
            similarity_cfg_rate=float(request.get("options", {}).get("similarity_cfg_rate", 0.7)),
            top_p=0.9,
            temperature=1.0,
            repetition_penalty=1.0,
            convert_style=False,
            anonymization_only=False,
        )
        started = time.perf_counter()
        cwd_inner = os.getcwd()
        os.chdir(ref_root)
        try:
            audio = module.convert_voice_v2(request["audio"], request["voice_ref"], run_args)
        finally:
            os.chdir(cwd_inner)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        sample_rate, waveform = audio
        audio_np = np.asarray(waveform, dtype=np.float32).reshape(-1)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio_np, int(sample_rate))

    return wrapper, run


def vevo2_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    use_transformers457()
    enable_legacy_llama_config_positional_args()
    ref_root = REPO_ROOT / "reference" / "Amphion"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline

    model_root = ensure_absolute_path(case["model"])
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    cwd = os.getcwd()
    os.chdir(ref_root)
    try:
        pipeline = Vevo2InferencePipeline(
            prosody_tokenizer_ckpt_path=str(model_root / "tokenizer" / "prosody_fvq512_6.25hz"),
            content_style_tokenizer_ckpt_path=str(model_root / "tokenizer" / "contentstyle_fvq16384_12.5hz"),
            ar_cfg_path=str(model_root / "contentstyle_modeling" / "posttrained" / "amphion_config.json"),
            ar_ckpt_path=str(model_root / "contentstyle_modeling" / "posttrained"),
            fmt_cfg_path=str(model_root / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa" / "config.json"),
            fmt_ckpt_path=str(model_root / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa"),
            vocoder_cfg_path=str(model_root / "vocoder" / "config.json"),
            vocoder_ckpt_path=str(model_root / "vocoder"),
            device=device,
        )
    finally:
        os.chdir(cwd)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        cwd_inner = os.getcwd()
        os.chdir(ref_root)
        try:
            audio = pipeline.inference_ar_and_fm(
                target_text=request["target_text"],
                prosody_wav_path=request.get("style_ref"),
                style_ref_wav_path=request.get("style_ref"),
                style_ref_wav_text=request.get("style_ref_text", ""),
                timbre_ref_wav_path=request.get("target_voice"),
                top_k=int(request.get("top_k", 25)),
                top_p=float(request.get("top_p", 0.8)),
                temperature=float(request.get("temperature", 1.0)),
                flow_matching_steps=int(request.get("num_inference_steps", 32)),
                display_audio=False,
            )
        finally:
            os.chdir(cwd_inner)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(audio_np, 24000)

    return pipeline, run


def parakeet_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    set_threads(args.threads)
    driver = load_module("parakeet_test_driver_perf", REPO_ROOT / "reference" / "parakeet-tdt" / "tools" / "test_driver.py")
    driver_args = argparse.Namespace(
        model=str(ensure_absolute_path(case["model"])),
        audio="",
        backend=args.backend,
        device=args.device,
        mode="perf",
        run_mode="offline",
        trace_log=None,
        timing_log=None,
        batch_size=1,
        timestamps=False,
        timestamp_levels="char,word,segment",
        self_attention_model="rel_pos_local_attn",
        att_context_left=256,
        att_context_right=256,
        chunk_secs=2.0,
        left_context_secs=10.0,
        right_context_secs=2.0,
        clean_groundtruth_text=False,
        langid="en",
    )
    driver.configure_runtime(driver_args)
    nemo_asr, open_dict, torch, _timing_log_scalar, _trace_log_scalar = driver.import_runtime_modules()
    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    model = driver.resolve_model(nemo_asr, driver_args.model)
    driver.configure_device(torch, model, driver_args.backend)
    driver.configure_timestamps(model, open_dict)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        model.transcribe(
            [request["audio"]],
            batch_size=1,
            return_hypotheses=False,
            timestamps=False,
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def sortformer_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import librosa
    import torch

    set_threads(args.threads)
    ref_root = REPO_ROOT / "reference" / "nemo"
    if str(ref_root) not in sys.path:
        sys.path.insert(0, str(ref_root))
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig

    map_location = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model_path = ensure_absolute_path(case["model"])
    model_name = str(model_path)
    if model_name.endswith(".nemo"):
        model = SortformerEncLabelModel.restore_from(model_name, map_location=map_location).to(map_location)
    elif model_path.is_dir():
        model = SortformerEncLabelModel.from_pretrained(model_name=f"nvidia/{model_path.name}", map_location=map_location).to(map_location)
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name=model_name, map_location=map_location).to(map_location)
    model.eval()

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        audio_np, sample_rate = librosa.load(request["audio"], sr=None, mono=False)
        started = time.perf_counter()
        diarize_cfg = DiarizeConfig(
            session_len_sec=20.0,
            batch_size=1,
            num_workers=0,
            sample_rate=sample_rate,
            verbose=False,
            include_tensor_outputs=False,
        )
        model.diarize(
            audio=audio_np,
            sample_rate=sample_rate,
            override_config=diarize_cfg,
        )
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def citrinet_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    set_threads(args.threads)
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model = RefCitrinet(ensure_absolute_path(case["model"]), device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        model.transcribe(Path(request["audio"]))
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def marblenet_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    set_threads(args.threads)
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    model = RefMarbleNet(ensure_absolute_path(case["model"]), device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        started = time.perf_counter()
        model.detect(Path(request["audio"]))
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, input_audio_duration_sec(request)

    return model, run


def silero_vad_loader(case: dict[str, Any], args: argparse.Namespace) -> tuple[Any, RunnerFn]:
    import torch

    set_threads(args.threads)
    sys.path.insert(0, str(REPO_ROOT / "reference" / "silero-vad" / "src"))
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=False)
    device = torch.device(f"cuda:{args.device}" if args.backend == "cuda" else "cpu")
    if args.backend == "cuda":
        model = model.to(device)

    def run(request: dict[str, Any], _args: argparse.Namespace) -> tuple[float, float]:
        waveform, sample_rate = read_wav_mono_f32(Path(request["audio"]))
        audio = torch.from_numpy(waveform.astype(np.float32, copy=False)).to(device)
        started = time.perf_counter()
        get_speech_timestamps(audio, model, sampling_rate=sample_rate, return_seconds=False, visualize_probs=False)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return wall_ms, audio_duration_sec(waveform, sample_rate)

    return model, run


LOADERS: dict[str, LoaderFn] = {
    "ace_step": ace_step_loader,
    "qwen3_asr": qwen3_asr_loader,
    "qwen3_forced_aligner": qwen3_forced_aligner_loader,
    "qwen3_tts": qwen3_tts_loader,
    "chatterbox": chatterbox_loader,
    "miocodec": miocodec_loader,
    "miotts": miotts_loader,
    "voxcpm2": voxcpm2_loader,
    "kokoro_tts": kokoro_loader,
    "pocket_tts": pocket_tts_loader,
    "omnivoice": omnivoice_loader,
    "moss_tts_nano": moss_tts_loader,
    "seed_vc": seed_vc_loader,
    "vevo2": vevo2_loader,
    "parakeet_tdt": parakeet_loader,
    "sortformer_diar": sortformer_loader,
    "citrinet_asr": citrinet_loader,
    "marblenet_vad": marblenet_loader,
    "htdemucs": demucs_loader,
    "mel_band_roformer": roformer_loader,
    "silero_vad": silero_vad_loader,
}


def measure_cold(loader: LoaderFn, case: dict[str, Any], request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    wall_ms: list[float] = []
    duration_sec = 0.0
    for _ in range(3):
        started = time.perf_counter()
        _model, run = loader(case, args)
        run_wall_ms, duration_sec = run(request, args)
        total_ms = (time.perf_counter() - started) * 1000.0
        if total_ms < run_wall_ms:
            total_ms = run_wall_ms
        wall_ms.append(total_ms)
    average_wall_ms = float(sum(wall_ms) / len(wall_ms))
    return {
        "wall_ms": wall_ms,
        "average_wall_ms": average_wall_ms,
        "duration_sec": duration_sec,
        "rtf": (average_wall_ms / 1000.0 / duration_sec) if duration_sec > 0.0 else 0.0,
    }


def measure_warm(loader: LoaderFn, case: dict[str, Any], warmup_request: dict[str, Any], request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _model, run = loader(case, args)
    run(warmup_request, args)
    wall_ms: list[float] = []
    duration_sec = 0.0
    for _ in range(3):
        run_wall_ms, duration_sec = run(request, args)
        wall_ms.append(run_wall_ms)
    average_wall_ms = float(sum(wall_ms) / len(wall_ms))
    return {
        "wall_ms": wall_ms,
        "average_wall_ms": average_wall_ms,
        "duration_sec": duration_sec,
        "rtf": (average_wall_ms / 1000.0 / duration_sec) if duration_sec > 0.0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    perf_cases = load_perf_cases(args.perf_cases)
    request_cases = read_json(args.request_cases)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_out = args.summary_out or (REPO_ROOT / "build" / "logs" / "perf" / f"python_{args.backend}_{timestamp}" / "summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for perf_case in perf_cases:
        if args.id and args.id != perf_case.id:
            continue
        if perf_case.family not in LOADERS:
            raise RuntimeError(f"python perf harness does not support family yet: {perf_case.family}")
        case = find_case(request_cases, perf_case.case_id)
        request = resolve_request_paths(materialize_perf_request(
            find_request(case, perf_case.request_id),
            perf_case.request_overrides,
            perf_case.request_remove_keys,
        ))
        if perf_case.warmup_request_id:
            warmup_request = resolve_request_paths(materialize_perf_request(
                find_request(case, perf_case.warmup_request_id),
                perf_case.warmup_request_overrides,
                perf_case.warmup_request_remove_keys,
            ))
        else:
            temp_dir = (summary_out.parent / "_warmup_temp" / perf_case.id).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            warmup_request = derive_warmup_request(request, temp_dir)
        cold = measure_cold(LOADERS[perf_case.family], case, request, args)
        warm = measure_warm(LOADERS[perf_case.family], case, warmup_request, request, args)
        summary = {
            "id": perf_case.id,
            "family": perf_case.family,
            "case_id": perf_case.case_id,
            "request_id": perf_case.request_id,
            "task": case["task"],
            "mode": case["mode"],
            "cold": cold,
            "warm": warm,
        }
        summaries.append(summary)
        print(
            f"{perf_case.id} cold_avg_ms={cold['average_wall_ms']} cold_rtf={cold['rtf']} "
            f"warm_avg_ms={warm['average_wall_ms']} warm_rtf={warm['rtf']}",
            flush=True,
        )

    payload = {
        "backend": args.backend,
        "device": args.device,
        "threads": args.threads,
        "cases": summaries,
    }
    summary_out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary_json={summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
