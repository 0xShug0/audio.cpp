"""Export a controlled full U-Net prediction from the official LatentSync model."""

import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--cache-interval", type=int, default=1)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--benchmark-runs", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_dir.resolve()))
    from latentsync.models.unet import UNet3DConditionModel

    config = json.loads((args.native_dir / "config.json").read_text())
    model = UNet3DConditionModel(**config["unet"])
    with safe_open(args.native_dir / "unet.safetensors", framework="pt", device="cpu") as source:
        model.load_state_dict({key.removeprefix("unet."): source.get_tensor(key) for key in source.keys()}, strict=True)
    model = (model.half() if args.dtype == "f16" else model.float()).cuda().eval()
    torch.manual_seed(42)
    image = torch.randn(args.batch, 13, args.frames, args.latent_size, args.latent_size, device="cuda")
    audio = torch.randn(args.batch, args.frames, 50, 384, device="cuda")
    if args.dtype == "f16":
        image = image.half()
        audio = audio.half()
    timesteps = torch.full((args.batch,), 501, dtype=torch.long, device="cuda")
    extra = {}
    with torch.inference_mode():
        if args.steps:
            from diffusers import DDIMScheduler
            if args.batch != 1:
                raise ValueError("Denoising fixture represents one video clip; use --batch 1")
            scheduler = DDIMScheduler.from_config(config["scheduler"])
            scheduler.set_timesteps(args.steps, device="cuda")
            if args.cache_interval > 1:
                from DeepCache import DeepCacheSDHelper
                cache = DeepCacheSDHelper(pipe=SimpleNamespace(unet=model, scheduler=scheduler))
                cache.set_params(cache_interval=args.cache_interval, cache_branch_id=0)
                cache.enable()
            initial = image[:, :4, :1].repeat(1, 1, args.frames, 1, 1)
            mask = (image[:, 4:5] > 0).float()
            masked, reference = image[:, 5:9], image[:, 9:13]
            conditions = torch.cat((mask, masked, reference), dim=1)
            cfg = args.guidance > 1
            model_audio = torch.cat((torch.zeros_like(audio), audio)) if cfg else audio
            output = initial.clone()
            for index, timestep in enumerate(scheduler.timesteps):
                model_input = torch.cat((output, conditions), dim=1)
                if cfg:
                    model_input = torch.cat((model_input, model_input))
                prediction = model(model_input, timestep, encoder_hidden_states=model_audio).sample
                if args.cache_interval > 1:
                    extra[f"step_input.{index}"] = model_input.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu()
                    extra[f"step_output.{index}"] = prediction.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu()
                if cfg:
                    uncond, cond = prediction.chunk(2)
                    prediction = uncond + args.guidance * (cond - uncond)
                output = scheduler.step(prediction, timestep, output, eta=0).prev_sample
            image = initial
            for name, value in (("mask", mask), ("masked_image", masked), ("reference_image", reference)):
                extra[name] = value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu()
        else:
            output = model(image, timesteps, encoder_hidden_states=audio).sample
            if args.benchmark_runs:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.benchmark_runs):
                    output = model(image, timesteps, encoder_hidden_states=audio).sample
                torch.cuda.synchronize()
                print(f"session.wall_ms {(time.perf_counter() - start) * 1000:.3f}", flush=True)
        time_features = model.time_proj(timesteps)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({"frames": args.frames, "steps": args.steps, "guidance": args.guidance,
                                                       "cache_interval": args.cache_interval, "weight_type": args.dtype,
                                                       "benchmark_runs": args.benchmark_runs}))
    link = args.output / "weights.safetensors"
    if not link.exists():
        link.symlink_to((args.native_dir / "unet.safetensors").resolve())
    save_file({
        "input": image.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu(),
        "output": output.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu(),
        "audio": audio.flatten(0, 1).contiguous().cpu(),
        "timestep_features": time_features.contiguous().cpu(),
        **extra,
    }, args.output / "reference.safetensors")
    print(f"U-Net: {tuple(image.shape)} -> {tuple(output.shape)}", flush=True)


if __name__ == "__main__":
    main()
