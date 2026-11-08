#!/usr/bin/env python3
"""Measure the official MOSS-SoundEffect v2 public pipeline after warmup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("reference/MOSS-TTS"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--attention", choices=("sdpa", "auto"), default="sdpa")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    import torch
    import soundfile as sf
    sys.path.insert(0, str(args.reference.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/audiocpp_cli"))
    from moss_soundeffect_v2 import MossSoundEffectPipeline
    from moss_soundeffect_v2.diffsynth.models import wan_video_dit
    from run_audiocpp_cli_path_tests import ResourceMonitor
    if args.attention == "sdpa":
        # Compare native BF16 attention, rather than optional quantized SageAttention.
        wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
        wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
        wan_video_dit.SAGE_ATTN_AVAILABLE = False
    import os

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reference measurement")
    torch.set_num_threads(8)
    pipe = MossSoundEffectPipeline.from_pretrained(str(args.model_dir.resolve()), torch_dtype=torch.bfloat16, device="cuda")
    prompts = [
        ("rain", "Steady rain pattering on a wooden roof, with distant thunder.", 42),
        ("birds", "清晨的森林里，鸟儿鸣叫，微风轻轻吹过树叶。", 2026),
    ]
    pipe(prompt=prompts[0][1], seconds=10, num_inference_steps=10, seed=0)
    torch.cuda.synchronize()
    results = []
    for name, prompt, seed in prompts:
        monitor = ResourceMonitor(os.getpid(), 0, 50)
        stop = threading.Event()
        def sample():
            while not stop.wait(0.05):
                monitor.sample()
        pool = ThreadPoolExecutor(max_workers=1)
        sampler = pool.submit(sample)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            audio = pipe(prompt=prompt, seconds=10, num_inference_steps=100, cfg_scale=4, sigma_shift=5, seed=seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            try:
                sampler.result()
                monitor.sample()
            finally:
                pool.shutdown()
                monitor.close()
        values = audio[0, 0].detach().float().cpu().numpy()
        sf.write(args.output_dir / f"{name}.wav", values, pipe.sample_rate, subtype="FLOAT")
        result = {"id": name, "seed": seed, "steps": 100, "seconds": len(values) / pipe.sample_rate,
                  "attention": args.attention, "compiled_model": not torch._dynamo.config.disable,
                  "wall_ms": elapsed * 1000, "rtf": elapsed / 10,
                  "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "resources": monitor.metrics.to_json()}
        results.append(result)
        logging.info("session.wall_ms=%f", result["wall_ms"])
        logging.info("result=%s", json.dumps(result))
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
