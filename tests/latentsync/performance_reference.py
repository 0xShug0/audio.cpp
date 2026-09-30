"""Time repeated official inference requests, without precision or RNG overrides."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    for name in ("reference", "checkpoint", "video", "audio", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference))
    os.chdir(args.reference)
    from scripts import inference

    original_call = inference.LipsyncPipeline.__call__

    def timed_call(pipeline, **kwargs):
        totals = {}

        def timed(name, function):
            def call(*positional, **keywords):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = function(*positional, **keywords)
                torch.cuda.synchronize()
                totals[name] = totals.get(name, 0) + (time.perf_counter() - start) * 1000
                return result
            return call

        pipeline.unet.forward = timed("unet.wall_ms", pipeline.unet.forward)
        pipeline.vae.encode = timed("vae.encode.wall_ms", pipeline.vae.encode)
        pipeline.vae.decode = timed("vae.decode.wall_ms", pipeline.vae.decode)
        for index in range(args.repeats):
            totals.clear()
            set_seed(1247)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            kwargs["video_out_path"] = str(args.output / f"request-{index}.mp4")
            result = original_call(pipeline, **kwargs)
            torch.cuda.synchronize()
            totals["session.wall_ms"] = (time.perf_counter() - start) * 1000
            totals["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1048576
            print(json.dumps({"request": index, "deepcache": args.deepcache, **totals}), flush=True)
        return result

    inference.LipsyncPipeline.__call__ = timed_call
    inference.main(OmegaConf.load("configs/unet/stage2_512.yaml"), argparse.Namespace(
        video_path=str(args.video), audio_path=str(args.audio),
        video_out_path=str(args.output / "output.mp4"),
        inference_ckpt_path=str(args.checkpoint), enable_deepcache=args.deepcache,
        seed=1247, inference_steps=20, guidance_scale=1.5,
        temp_dir=str(args.output / "temp")))


if __name__ == "__main__":
    main()
