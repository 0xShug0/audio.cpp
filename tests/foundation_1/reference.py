#!/usr/bin/env python3
"""Run the model author's unmodified Python pipeline with local checkpoints."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import soundfile as sf
import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--text-encoder', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--duration-sec', type=float, default=10)
    p.add_argument('--num-steps', type=int, default=100)
    p.add_argument('--guidance-scale', type=float, default=7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--log', type=Path, required=True)
    a = p.parse_args()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(a.log), logging.StreamHandler()])
    sys.path.insert(0, str(a.upstream.resolve()))
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.conditioners import T5Conditioner
    from stable_audio_tools.inference.sampling import sample
    config = json.loads((a.source / 'model_config.json').read_text())
    # Resolve only the pretrained model identifier locally; inference is unchanged.
    text_path = str(a.text_encoder.resolve())
    T5Conditioner.T5_MODELS.append(text_path)
    T5Conditioner.T5_MODEL_DIMS[text_path] = 768
    config['model']['conditioning']['configs'][0]['config']['t5_model_name'] = text_path
    model = create_model_from_config(config)
    model.load_state_dict(load_file(a.source / 'Foundation_1.safetensors'), strict=True)
    model.eval().requires_grad_(False).to('cuda')
    frames = int((a.duration_sec * config['sample_rate'] + 2047) // 2048) * 2048
    for i in range(a.repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            torch.manual_seed(a.seed)
            noise = torch.randn(1, model.io_channels, frames // 2048, device='cuda')
            conditions = model.conditioner([{'prompt': a.prompt, 'seconds_start': 0,
                'seconds_total': a.duration_sec}], 'cuda')
            latent = sample(model.model, noise, a.num_steps, 0, cfg_scale=a.guidance_scale,
                batch_cfg=True, rescale_cfg=True, **model.get_conditioning_inputs(conditions))
            output = model.pretransform.decode(latent)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        logging.info('request=%d session.wall_ms=%.6f rtf=%.6f peak_allocated_mib=%.3f peak_reserved_mib=%.3f',
            i, elapsed, elapsed / (a.duration_sec * 1000),
            torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20)
        audio = output[0, :, :round(a.duration_sec * config['sample_rate'])].float().cpu().T.numpy()
        sf.write(a.output if i == 0 else a.output.with_stem(a.output.stem + f'-{i}'),
                 audio, config['sample_rate'], subtype='FLOAT')


if __name__ == '__main__':
    main()
