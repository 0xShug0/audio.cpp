"""Export the official DDIM schedule and timestep features for numerical checks."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from diffusers.models.embeddings import Timesteps
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = DDIMScheduler.from_config(json.loads(args.scheduler_config.read_text()))
    schedule.set_timesteps(20)
    torch.manual_seed(42)
    initial = torch.randn(64)
    noises = torch.randn(20, 64)
    sample = initial.clone()
    outputs = []
    for timestep, noise in zip(schedule.timesteps, noises):
        sample = schedule.step(noise, timestep, sample, eta=0).prev_sample
        outputs.append(sample.clone())
    features = Timesteps(320, True, 0)(schedule.timesteps)
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"initial": initial, "noises": noises, "outputs": torch.stack(outputs),
               "timesteps": schedule.timesteps, "features": features}, args.output / "reference.safetensors")
    print(f"DDIM timesteps: {schedule.timesteps.tolist()}")


if __name__ == "__main__":
    main()
