#!/usr/bin/env python3
"""Generate an official MOSS-SoundEffect v2 diffusion/waveform fixture.

Uses official engine components directly: the public wrapper enables BF16 autocast.
No compiled model, TeaCache, CFG merge, or alternative sampler is enabled.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch


TOKENIZER_CASES = [
    "Steady rain on a wooden roof.",
    "清晨的森林里，鸟儿鸣叫。",
    "Ｆｕｌｌ－ｗｉｄｔｈ！",
    "A cafe\u0301 with soft background chatter.",
    "“Rustling leaves” and a bird’s call.",
    "A\u00a0forest\u2003with\nwind.",
    "A ﬁre crackles beside a ﬂowing stream.",
    "Waves &amp; wind, then &amp;amp; rain.",
    "A bell &quot;rings&quot; &#x1F514; twice.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("reference/MOSS-TTS"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="Steady rain pattering on a metal roof with distant rolling thunder.")
    parser.add_argument("--negative-prompt", default="")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.model_dir.is_dir() or not (args.reference / "moss_soundeffect_v2").is_dir():
        parser.error("local model directory and official source directory must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / "reference.safetensors"
    manifest_path = args.output_dir / "manifest.json"
    if fixture_path.exists() or manifest_path.exists():
        parser.error("refusing to overwrite an existing reference fixture")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.log, mode="x")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[file_handler, logging.StreamHandler()])
    started = time.perf_counter()
    try:
        # Capture official loader prints and warnings in the requested log too.
        with contextlib.redirect_stdout(file_handler.stream), contextlib.redirect_stderr(file_handler.stream):
            import torch
            from safetensors.torch import save_file
            from torch.nn.attention import SDPBackend, sdpa_kernel

            sys.path.insert(0, str(args.reference.resolve()))
            from moss_soundeffect_v2.pipeline_moss_soundeffect import MossSoundEffectPipeline
            from moss_soundeffect_v2.diffsynth.models import wan_video_dit

            if not torch.cuda.is_available():
                raise RuntimeError("strict reference requires CUDA; no CPU fallback")
            torch.set_num_threads(8)
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            pipeline = MossSoundEffectPipeline.from_pretrained(
                str(args.model_dir.resolve()), torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32, device="cuda")
            engine = pipeline.engine
            if not hasattr(engine.model_fn, "_torchdynamo_orig_callable"):
                raise RuntimeError("expected official torch.compile model_fn wrapper with _torchdynamo_orig_callable")
            model_fn = engine.model_fn._torchdynamo_orig_callable
            # Module.float() preserves complex RoPE buffers; .to(dtype=float32) would not.
            engine.eval()
            if args.dtype == "f32":
                engine.float()

            seconds, guidance, shift = 10.0, 4.0, 5.0
            prompt = args.prompt.strip()
            formatted_prompt = f"{prompt} duration: {seconds:.1f}s"
            hop_length = int(engine.vae.hop_length)
            frames = pipeline.sample_rate * pipeline.max_inference_seconds // hop_length
            shape = (1, int(engine.audio_latent_dim), frames)
            # Official compatibility_mode uses PyTorch SDPA, which supports F32.
            # Force math attention instead of optional Flash/Sage implementations.
            reference_attention = functools.partial(wan_video_dit.flash_attention, compatibility_mode=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.dtype == "bf16"), \
                    torch.autocast("cpu", enabled=False), \
                    (sdpa_kernel(SDPBackend.MATH) if args.dtype == "f32" else contextlib.nullcontext()), \
                    (patch.object(wan_video_dit, "flash_attention", reference_attention)
                     if args.dtype == "f32" else contextlib.nullcontext()):
                positive = engine.prompter.encode_prompt(formatted_prompt, positive=True, device=engine.device)
                negative = engine.prompter.encode_prompt(args.negative_prompt, positive=False, device=engine.device)
                positive_ids, positive_mask = engine.prompter.tokenizer(
                    formatted_prompt, return_mask=True, add_special_tokens=True)
                negative_ids, negative_mask = engine.prompter.tokenizer(
                    args.negative_prompt, return_mask=True, add_special_tokens=True)
                initial_noise = engine.generate_noise(shape, seed=args.seed, rand_device="cpu")
                latent = initial_noise.clone()
                engine.scheduler.set_timesteps(args.steps, denoising_strength=1.0, shift=shift)
                timesteps = engine.scheduler.timesteps.detach().cpu().clone()
                sigmas = torch.cat((engine.scheduler.sigmas.detach().cpu(), torch.zeros(1)))
                for index, timestep in enumerate(engine.scheduler.timesteps):
                    timestep = timestep.unsqueeze(0).to(device=engine.device)
                    pos = model_fn(dit=engine.dit, latents=latent, timestep=timestep, context=positive)
                    neg = model_fn(dit=engine.dit, latents=latent, timestep=timestep, context=negative)
                    velocity = neg.float() + guidance * (pos.float() - neg.float())
                    latent = engine.scheduler.step(velocity, engine.scheduler.timesteps[index], latent)
                    logging.info("reference.diffusion.step=%d/%d", index + 1, args.steps)
                with torch.autocast("cuda", enabled=False):
                    waveform = engine.vae.decode(latent.float())

            tensors = {
                "context.positive": positive,
                "context.negative": negative,
                "initial_noise": initial_noise,
                "schedule.timesteps": timesteps,
                "schedule.sigmas": sigmas,
                "final_latent": latent,
                "waveform": waveform,
            }
            for name, tensor in tensors.items():
                if not torch.isfinite(tensor).all().item():
                    raise RuntimeError(f"fixture tensor must be finite: {name}: {tensor.dtype}")
                tensors[name] = tensor.float()
            tensors.update({
                "text.positive.ids": positive_ids.to(torch.int32),
                "text.positive.mask": positive_mask.to(torch.int32),
                "text.positive.valid_count": positive_mask.gt(0).sum(dim=1).to(torch.int32),
                "text.negative.ids": negative_ids.to(torch.int32),
                "text.negative.mask": negative_mask.to(torch.int32),
                "text.negative.valid_count": negative_mask.gt(0).sum(dim=1).to(torch.int32),
            })
            if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 1):
                raise RuntimeError(f"expected batch-one mono waveform, got {waveform.shape}")
            output_samples = int(pipeline.sample_rate * seconds)
            if waveform.shape[-1] < output_samples:
                raise RuntimeError("decoded waveform is shorter than the requested crop")
            save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()},
                      str(fixture_path), metadata={"format": "moss_soundeffect_v2.parity.v1"})
            source_files = [
                "pipeline_moss_soundeffect.py", "diffsynth/pipelines/wan_audio.py",
                "diffsynth/models/wan_audio_dit.py", "diffsynth/models/wan_video_dit.py",
                "diffsynth/models/dac_vae.py", "diffsynth/models/qwen3_text_encoder.py",
                "diffsynth/prompters/wan_prompter.py", "diffsynth/schedulers/flow_match.py",
            ]
            source_root = args.reference.resolve() / "moss_soundeffect_v2"
            tokenizer_cases = []
            for text in TOKENIZER_CASES:
                ids, mask = engine.prompter.tokenizer(text, return_mask=True, add_special_tokens=True)
                tokenizer_cases.append({"text": text, "ids": ids[0][mask[0].bool()].tolist()})
            manifest = {
                "tokenizer_cases": tokenizer_cases,
                "format": "moss_soundeffect_v2.parity.v1",
                "model_dir": str(args.model_dir.resolve()), "source_root": str(source_root),
                "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                  for name in source_files},
                "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0),
                "dtype": "bfloat16" if args.dtype == "bf16" else "float32",
                "autocast": args.dtype == "bf16", "tf32": False,
                "attention": "official" if args.dtype == "bf16" else "sdpa_math",
                "compiled_model": False,
                "prompt": prompt, "formatted_prompt": formatted_prompt, "negative_prompt": args.negative_prompt,
                "seconds": seconds, "output_samples": output_samples,
                "max_duration_sec": pipeline.max_inference_seconds, "sample_rate": pipeline.sample_rate,
                "hop_length": hop_length, "frames": frames, "latent_dim": shape[1],
                "text_length": int(positive.shape[1]), "text_dim": int(positive.shape[2]),
                "steps": args.steps, "seed": args.seed, "guidance_scale": guidance, "sigma_shift": shift,
                "noise_device": "cpu", "full_waveform_samples": int(waveform.shape[-1]),
                "transformer_config": json.loads((args.model_dir / "transformer/config.json").read_text()),
                "scope": "full native text conditioning, DiT trajectory and DAC waveform with aligned noise; oracle contexts retained for isolation",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            logging.info("reference.fixture=%s", fixture_path)
    except Exception:
        logging.exception("reference generation failed")
        raise
    finally:
        logging.info("session.wall_ms=%.3f", (time.perf_counter() - started) * 1000)


if __name__ == "__main__":
    main()
