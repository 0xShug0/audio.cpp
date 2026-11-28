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


def load_original_encoder(upstream, config_path, weights):
    from safetensors.torch import load_file
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


def compare(args):
    # Acceptance from docs/reports/kugelaudio_reference_cloning.md, F32 reference.
    rows = json.loads((args.output_dir / "report.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty or malformed reference report")
    names = set()
    for row in rows:
        name = Path(row["audio"]).stem
        if not name or name in (".", "..") or name in names or row["dtype"] != "f32" or row["tf32"] is not False:
            raise ValueError("comparison requires unique fixture names and strict F32 references")
        names.add(name)
        prefix = args.cpp_dir / name
        meta = json.loads(Path(str(prefix) + ".json").read_text())
        if type(row["sample_rate"]) is not int or row["sample_rate"] != 24000:
            raise ValueError(f"{name}: expected 24 kHz reference")
        shape = row["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or
                any(type(n) is not int or n <= 0 for n in shape) or
                shape[0] != 1 or shape[2] != 64 or
                type(row["samples"]) is not int or row["samples"] <= 0 or
                type(meta["samples"]) is not int or type(meta["frames"]) is not int or
                type(meta["dim"]) is not int or meta["samples"] != row["samples"] or
                [1, meta["frames"], meta["dim"]] != shape):
            raise ValueError(f"{name}: invalid or mismatched metadata shape")
        reference = np.load(args.output_dir / (name + ".mean.npy"), allow_pickle=False)
        prepared = np.load(args.output_dir / (name + ".input.npy"), allow_pickle=False)
        arrays = []
        for suffix, expected in (("input", (row["samples"],)), ("mean", tuple(shape))):
            path = Path(str(prefix) + f".{suffix}.f32")
            if path.stat().st_size != int(np.prod(expected)) * 4:
                raise ValueError(f"{path}: byte count does not match shape")
            arrays.append(np.fromfile(path, dtype=np.float32).reshape(expected))
        actual_input, actual = arrays
        if (reference.shape != tuple(shape) or prepared.shape != (row["samples"],) or
                any(a.dtype.kind != "f" or not np.isfinite(a).all()
                    for a in (reference, prepared, actual_input, actual))):
            raise ValueError(f"{name}: invalid, nonfinite or mismatched reference data")
        input_error = float(np.max(np.abs(prepared.astype(np.float64) - actual_input)))
        if input_error > args.max_input_abs_error:
            raise ValueError(f"{name}: prepared input mismatch ({input_error})")
        a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            raise ValueError(f"{name}: zero feature norm")
        relative_l2 = float(np.linalg.norm(a - b) / norm_b)
        cosine = float(np.dot(a / norm_a, b / norm_b))
        limit, minimum = (1e-3, 0.999999) if args.backend == "cuda" else (1e-5, 0.99999999)
        print(f"{name}: input_max_abs={input_error:.9g} relative_l2={relative_l2:.9g} cosine={cosine:.12g}")
        if not np.isfinite(relative_l2) or not np.isfinite(cosine) or relative_l2 >= limit or cosine <= minimum:
            raise ValueError(f"{name}: encoder acceptance failed for {args.backend}")
    print(f"PASS: {len(rows)} encoder fixtures")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", action="store_true", help="compare existing reference output with native probe outputs")
    p.add_argument("--cpp-dir", type=Path, help="native prefixes must be CPP_DIR/<audio stem>")
    p.add_argument("--backend", choices=("cpu", "cuda", "vulkan"))
    p.add_argument("--max-input-abs-error", type=float,
                   help="required explicit prepared-audio acceptance; report observations are not a bound")
    p.add_argument("--upstream", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--encoder", type=Path)
    p.add_argument("--audio", nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    p.add_argument("--tf32", action="store_true", help="compare CUDA's reduced-precision FP32 math policy")
    args = p.parse_args()
    if args.compare:
        if (args.cpp_dir is None or args.backend is None or args.max_input_abs_error is None or
                not np.isfinite(args.max_input_abs_error) or args.max_input_abs_error < 0):
            p.error("--compare requires --cpp-dir, --backend and finite nonnegative --max-input-abs-error")
        compare(args)
        return
    if any(value is None for value in (args.upstream, args.config, args.encoder, args.audio)):
        p.error("generation requires --upstream, --config, --encoder and --audio")
    if args.threads <= 0 or len({path.stem for path in args.audio}) != len(args.audio):
        p.error("threads must be positive and audio stems must be unique")
    import soundfile as sf
    import torch

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
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"{path}: empty or nonfinite audio")
        audio = audio.mean(axis=1, dtype=np.float32)
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if not np.isfinite(rms):
            raise ValueError(f"{path}: invalid input RMS")
        audio = audio * (10 ** (-25 / 20) / (rms + 1e-6))
        peak = np.max(np.abs(audio))
        if peak > 1:
            audio = audio / (peak + 1e-6)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        start = time.perf_counter()
        with torch.inference_mode():
            mean = encoder(torch.from_numpy(audio)[None, None, :].to(device=args.device, dtype=dtype))
            mean = mean.transpose(1, 2).contiguous().float().cpu().numpy()
        if mean.ndim != 3 or mean.shape[0] != 1 or mean.shape[1] <= 0 or mean.shape[2] != 64 or not np.isfinite(mean).all():
            raise ValueError(f"{path}: invalid encoder output")
        row = dict(audio=str(path.resolve()), sample_rate=rate, samples=len(audio), shape=list(mean.shape), seconds=time.perf_counter()-start,
                   device=args.device, dtype=args.dtype, tf32=args.tf32)
        np.save(args.output_dir / (path.stem + ".mean.npy"), mean)
        np.save(args.output_dir / (path.stem + ".input.npy"), audio)
        reports.append(row)
        print(path.name, row["shape"], flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
