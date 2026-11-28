#!/usr/bin/env python3
"""Run the official single-device FlashHead pipeline with repeatable inputs."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--variant', choices=('lite','pro'),default='lite')
    p.add_argument('--audio-encoder', type=Path, required=True)
    p.add_argument('--portrait', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    p.add_argument('--color-strength', type=float, default=1)
    p.add_argument('--export', type=Path)
    args = p.parse_args()
    for name in ("source", "audio_encoder", "portrait", "audio", "output"):
        setattr(args, name, getattr(args, name).resolve())
    import flash_head.src.pipeline.flash_head_pipeline as official
    if args.export: args.export = args.export.resolve()
    os.chdir(Path(official.__file__).resolve().parents[3])
    official.COMPILE_MODEL = args.compile
    official.COMPILE_VAE = args.compile
    pipeline = official.FlashHeadPipeline(str(args.source), args.variant, str(args.audio_encoder), device='cuda',
                                          param_dtype=getattr(torch, args.dtype))
    if args.export:
        from safetensors.torch import save_file
        export_root = args.export.resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        def capture(method, filename, input_names):
            count = 0
            def wrapped(*inputs, **kwargs):
                nonlocal count
                output = method(*inputs, **kwargs)
                name = filename if filename != 'dit.safetensors' or count == 0 else 'dit' + str(count) + '.safetensors'
                count += 1
                target = export_root / name
                if not target.exists():
                    tensors = {key: (value.squeeze(0) if value.ndim == 5 else value).detach().cpu().contiguous() for key, value in zip(input_names, inputs)}
                    tensors.update({key: (value.squeeze(0) if value.ndim == 5 else value).detach().cpu().contiguous() for key, value in kwargs.items() if isinstance(value, torch.Tensor)})
                    tensors['output'] = (output.squeeze(0) if output.ndim == 5 else output).detach().cpu().contiguous()
                    save_file(tensors, target)
                return output
            return wrapped
        pipeline.vae.encode = capture(pipeline.vae.encode, 'encode.safetensors', ['pixels'])
        pipeline.vae.decode = capture(pipeline.vae.decode, 'decode.safetensors', ['latents'])
        pipeline.model.forward = capture(pipeline.model.forward, 'dit.safetensors', [])
    torch.manual_seed(args.seed)
    session_started=time.perf_counter()
    motion_frames=9 if args.variant=="lite" else 5
    block=(33-motion_frames)*16000//25
    pipeline.prepare_params(str(args.portrait), (512, 512), 33, motion_frames, 4, args.seed, 5, args.color_strength, False)
    samples, rate = sf.read(args.audio, dtype='float32')
    if rate != 16000 or samples.ndim != 1:
        raise ValueError('Reference test requires mono 16 kHz audio')
    samples = np.pad(samples, (0, (-len(samples)) % block))
    history = deque([0.] * 128000, maxlen=128000)
    from flash_head.inference import get_audio_embedding, run_pipeline
    frames = []
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, chunk in enumerate(samples.reshape(-1, block)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        history.extend(chunk.tolist())
        embedding = get_audio_embedding(pipeline, np.asarray(history), 167, 200)
        if args.export and i == 0:
            save_file({'wave': torch.from_numpy(np.asarray(history, dtype=np.float32)),
                       'output': embedding.squeeze(0).cpu().contiguous()}, export_root / 'audio.safetensors')
        output = run_pipeline(pipeline, embedding)[motion_frames:]
        chunk_frames=output.cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append({'chunk': i, 'wall_ms': elapsed * 1000, 'rtf': elapsed / (block/16000)})
        frames.append(chunk_frames)
    session_ms=(time.perf_counter()-session_started)*1000
    print(f"session.wall_ms {session_ms}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, np.concatenate(frames), fps=25, codec='libx264', quality=8)
    np.save(args.output.with_suffix('.npy'), np.concatenate(frames))
    args.output.with_suffix('.json').write_text(json.dumps({'chunks': rows, 'session.wall_ms':session_ms, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
