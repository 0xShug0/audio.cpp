"""Run the official LatentSync pipeline in controlled F32 for output parity, not timing."""

import argparse
import functools
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--frames-per-chunk", type=int, default=16)
    parser.add_argument("--cache-interval", type=int, default=1)
    args = parser.parse_args()
    for key in ("reference", "checkpoints", "video", "audio", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference))
    os.chdir(args.reference)

    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines import lipsync_pipeline
    from latentsync.utils import image_processor
    from latentsync.whisper.audio2feature import Audio2Feature

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.load("configs/unet/stage2_512.yaml")
    audio = Audio2Feature(model_path=str(args.checkpoints / "whisper/tiny.pt"), device="cuda")
    audio.model.float()
    audio.model.transcribe = functools.partial(audio.model.transcribe, fp16=False)
    vae = AutoencoderKL.from_pretrained(args.checkpoints / "sd-vae-ft-mse", torch_dtype=torch.float32)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model),
        str(args.checkpoints / "latentsync_unet.pt"), device="cpu")
    pipeline = lipsync_pipeline.LipsyncPipeline(vae=vae, audio_encoder=audio, unet=unet.float(),
        scheduler=DDIMScheduler.from_pretrained("configs")).to("cuda")
    if args.cache_interval > 1:
        from DeepCache import DeepCacheSDHelper
        cache = DeepCacheSDHelper(pipe=pipeline)
        cache.set_params(cache_interval=args.cache_interval, cache_branch_id=0)
        cache.enable()
    original_write = lipsync_pipeline.write_video

    def write_video(path, frames, fps):
        frames = np.asarray(frames, dtype=np.uint8)
        frames.tofile(args.output / "output.rgb")
        (args.output / "output.json").write_text(json.dumps({"shape": list(frames.shape), "fps": fps}) + "\n")
        original_write(path, frames, fps)

    torch.manual_seed(args.seed)
    with patch.object(image_processor, "AlignRestore", functools.partial(image_processor.AlignRestore, dtype=torch.float32)), \
            patch.object(lipsync_pipeline, "write_video", write_video):
        pipeline(video_path=str(args.video), audio_path=str(args.audio),
            video_out_path=str(args.output / "output.mp4"), temp_dir=str(args.output / "temp"),
            num_frames=args.frames_per_chunk, num_inference_steps=20, guidance_scale=1.5,
            weight_dtype=torch.float32, width=512, height=512,
            mask_image_path="latentsync/utils/mask.png")
    shutil.copyfile("temp/video.mp4", args.output / "decoded-input.mp4")
    print("Controlled F32 reference completed; do not use this run as a performance baseline.")


if __name__ == "__main__":
    main()
