#!/usr/bin/env python3
"""Run the unmodified upstream acoustic encoder as a cloning feature reference.

Upstream's public generation processor only takes prepared voice caches; this
harness exercises the original encoder component still present in its weights.
It does not claim that upstream exposes a supported raw-audio cloning API.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file


def load_original_encoder(upstream, config_path, weights):
    # Import only the audited config/tokenizer modules, without UI, hosted API,
    # watermark downloads or the full 7B generation model's package initializer.
    for suffix, path in [("", upstream), (".models", upstream / "models"), (".configs", upstream / "configs")]:
        module = types.ModuleType("kugelaudio_open" + suffix)
        module.__path__ = [str(path)]
        sys.modules[module.__name__] = module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    configs = load("kugelaudio_open.configs.model_config", upstream / "configs/model_config.py")
    for name in ["KugelAudioAcousticTokenizerConfig", "KugelAudioSemanticTokenizerConfig"]:
        setattr(sys.modules["kugelaudio_open.configs"], name, getattr(configs, name))
    tokenizer = load("kugelaudio_open.models.tokenizer", upstream / "models/tokenizer.py")
    config = configs.KugelAudioAcousticTokenizerConfig(**json.loads(config_path.read_text())["acoustic_tokenizer_config"])
    encoder_config = copy.deepcopy(config)
    encoder_config.dimension = config.vae_dim
    encoder_config.n_filters = config.encoder_n_filters
    encoder_config.ratios = config.encoder_ratios
    encoder_config.depths = [int(x) for x in config.encoder_depths.split("-")] if isinstance(config.encoder_depths, str) else config.encoder_depths
    encoder_config.norm = config.conv_norm
    encoder_config.bias = config.conv_bias
    encoder = tokenizer.TokenizerEncoder(encoder_config).float().eval()
    state = load_file(str(weights), device="cpu")
    prefix = "model.acoustic_tokenizer.encoder."
    encoder.load_state_dict({name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}, strict=True)
    return encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--encoder", required=True, type=Path)
    p.add_argument("--audio", required=True, nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    p.add_argument("--tf32", action="store_true", help="compare CUDA's reduced-precision FP32 math policy")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_original_encoder(args.upstream, args.config, args.encoder)
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    encoder = encoder.to(device=args.device, dtype=dtype)
    reports = []
    for path in args.audio:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 24000:
            raise ValueError("reference comparisons use 24 kHz fixtures to isolate encoding from resampler differences")
        audio = audio.mean(axis=1, dtype=np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        audio = audio * (10 ** (-25 / 20) / (rms + 1e-6))
        peak = np.max(np.abs(audio))
        if peak > 1:
            audio = audio / (peak + 1e-6)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        start = time.perf_counter()
        with torch.inference_mode():
            mean = encoder(torch.from_numpy(audio)[None, None, :].to(device=args.device, dtype=dtype))
            mean = mean.transpose(1, 2).contiguous().float().cpu().numpy()
        row = dict(audio=str(path.resolve()), samples=len(audio), shape=list(mean.shape), seconds=time.perf_counter()-start,
                   device=args.device, dtype=args.dtype, tf32=args.tf32)
        np.save(args.output_dir / (path.stem + ".mean.npy"), mean)
        np.save(args.output_dir / (path.stem + ".input.npy"), audio)
        reports.append(row)
        print(path.name, row["shape"], flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
