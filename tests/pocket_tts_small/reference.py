#!/usr/bin/env python3
"""Official PocketTTS small fixture export, waveform validation, and isolated timing."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

PIN = "4cc5753319ceddc5cbf70598d9011f1f22858ba0"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['export', 'compare', 'performance'], default='export')
    p.add_argument('--upstream', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--voice', type=Path)
    p.add_argument('--text', action='append', help='Repeat for varied performance requests')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=50)
    p.add_argument('--frames-after-eos', type=int)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--candidate', type=Path, help='C++ float WAV or safetensors with waveform key')
    p.add_argument('--log', action='store_true', required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(args.output_dir / (args.mode + '.log'))], force=True)
    import torch
    import soundfile as sf
    from safetensors.torch import load_file, save_file

    if args.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.mode == 'compare':
        if not args.fixture or not args.candidate:
            p.error('compare requires --fixture and --candidate')
        ref = load_file(str(args.fixture))['waveform'].double().flatten()
        if args.candidate.suffix == '.safetensors':
            candidate = load_file(str(args.candidate))['waveform'].double().flatten()
        else:
            samples, rate = sf.read(args.candidate, dtype='float64', always_2d=True)
            if rate != 24000 or samples.shape[1] != 1:
                raise ValueError('Expected mono 24000 Hz candidate')
            candidate = torch.from_numpy(samples[:, 0].copy())
        if ref.shape != candidate.shape:
            raise ValueError(f'Waveform length mismatch: {ref.shape} vs {candidate.shape}')
        if not torch.isfinite(candidate).all() or not torch.isfinite(ref).all():
            raise ValueError('Nonfinite waveform')
        cosine = torch.nn.functional.cosine_similarity(ref, candidate, dim=0).item()
        report = {'cosine': cosine, 'threshold': 0.99999, 'passed': cosine > 0.99999,
                  'samples': ref.numel(), 'max_abs_error': (ref-candidate).abs().max().item()}
        (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        logging.info('%s', report)
        if not report['passed']:
            raise SystemExit(1)
        return
    if not all([args.upstream, args.config, args.voice, args.text]):
        p.error('generation requires --upstream --config --voice --text')
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError(f'Upstream revision {revision} != {PIN}')
    sys.path.insert(0, str(args.upstream.resolve()))
    from pocket_tts import TTSModel
    import pocket_tts
    # Upstream import sets intra-op threads to 1; restore the requested budget.
    torch.set_num_threads(args.threads)
    if not Path(pocket_tts.__file__).resolve().is_relative_to(args.upstream.resolve()):
        raise RuntimeError('Imported PocketTTS outside requested checkout')
    if args.mode == 'performance' and (len(set(args.text)) < 2 or any(len(t) < 80 for t in args.text)):
        p.error('performance requires at least two distinct meaningful texts (80+ characters)')
    if args.mode == 'export' and len(args.text) != 1:
        p.error('export accepts exactly one text')
    torch.manual_seed(args.seed)
    model = TTSModel.load_model(config=args.config).to(args.device).eval()
    voice = model.get_state_for_audio_prompt(args.voice)
    metadata = dict(upstream_revision=revision, config=str(args.config.resolve()),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    voice=str(args.voice.resolve()), voice_sha256=hashlib.sha256(args.voice.read_bytes()).hexdigest(),
                    device=args.device, threads=args.threads, seed=args.seed,
                    max_tokens=args.max_tokens, frames_after_eos=args.frames_after_eos,
                    sample_rate=model.sample_rate, torch_version=torch.__version__)

    def sync():
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)

    def generate(text):
        return model.generate_audio_stream(voice, text, max_tokens=args.max_tokens,
                                          frames_after_eos=args.frames_after_eos)

    if args.mode == 'performance':
        # No wrappers/hooks, no artifact copies inside timing. Run this process exclusively.
        with torch.no_grad():
            for text in args.text:
                torch.manual_seed(args.seed)
                list(generate(text))
            sync()
            rows = []
            for index, text in enumerate(args.text):
                torch.manual_seed(args.seed)
                if torch.device(args.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(args.device)
                sync()
                start = time.perf_counter()
                first = None
                chunks = []
                for chunk in generate(text):
                    sync()
                    if first is None:
                        first = (time.perf_counter()-start)*1000
                    chunks.append(chunk)
                sync()
                wall = (time.perf_counter()-start)*1000
                duration = sum(c.numel() for c in chunks)/model.sample_rate
                rows.append({'text': text, 'session.wall_ms': wall, 'ttfa_ms': first,
                             'audio.duration_sec': duration, 'rtf': wall/1000/duration})
                if torch.device(args.device).type == 'cuda':
                    rows[-1]['cuda.peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
                    rows[-1]['cuda.peak_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
                # Save CPU copies after timing, without allocating a combined CUDA waveform.
                waveform = torch.cat([c.detach().cpu().flatten() for c in chunks])
                sf.write(args.output_dir / f'request-{index:04d}.wav', waveform.numpy(), model.sample_rate, subtype='FLOAT')
                logging.info('%s', rows[-1])
        metadata.update(warmup_requests=len(args.text), requests=rows,
                        ttfa_definition='First chunk yielded by official streaming API; includes upstream queue scheduling')
        (args.output_dir/'performance.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return

    tensors = {'voice.'+k: v.detach().cpu().contiguous().clone()
               for k, v in load_file(str(args.voice)).items()}
    calls, prepared, states = [], [], []
    original_decode = model.mimi.decode_from_latent
    original_prepare = model.flow_lm.conditioner.prepare

    def capture_decode(latent, state):
        index = len(calls)
        reset = not states or state is not states[-1]
        if reset:
            states.append(state)  # retain identity so Python cannot reuse its address
        if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[2] != 32:
            raise ValueError(f'Unexpected Mimi latent shape: {latent.shape}')
        tensors[f'decoder.{index:04d}.latent'] = latent.detach().cpu().contiguous().clone()
        output = original_decode(latent, state)
        calls.append({'index': index, 'reset': reset, 'frames': latent.shape[1],
                      'samples': output.shape[-1], 'fade_in_samples': model.sample_rate//200 if reset else 0})
        return output

    def capture_prepare(text):
        tokens = original_prepare(text)
        index = len(prepared)
        tensors[f'text.{index:04d}.token_ids'] = tokens.detach().cpu().contiguous().clone()
        prepared.append(text)
        return tokens

    model.mimi.decode_from_latent = capture_decode
    model.flow_lm.conditioner.prepare = capture_prepare
    try:
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            chunks = list(generate(args.text[0]))
        sync()
        wall = (time.perf_counter()-start)*1000
    finally:
        model.mimi.decode_from_latent = original_decode
        model.flow_lm.conditioner.prepare = original_prepare
    waveform = torch.cat(chunks).detach().cpu().contiguous()
    if not torch.isfinite(waveform).all() or not calls:
        raise ValueError('Invalid or empty export')
    if hashlib.sha256(args.voice.read_bytes()).hexdigest() != metadata['voice_sha256']:
        raise RuntimeError('Voice file changed during export; rerun with stable inputs')
    tensors['waveform'] = waveform
    tensors['latents'] = torch.cat([tensors[f'decoder.{i:04d}.latent'] for i in range(len(calls))], dim=1)
    metadata.update(text=args.text[0], prepared_text=prepared, decoder_calls=calls,
                    **{'session.wall_ms': wall}, timing_kind='instrumented correctness export; not performance',
                    latent_layout='B,T,32; denormalized; do not renormalize',
                    validation_status='exported; C++ teacher-forced waveform comparison pending')
    save_file(tensors, str(args.output_dir/'fixture.safetensors'), metadata={'schema': 'pocket_tts_small.v1', 'sample_rate': str(model.sample_rate),
                  'latent_layout': 'B,T,32', 'text': args.text[0], 'max_tokens': str(args.max_tokens),
                  'prepared_text': json.dumps(prepared),
                  'decoder_calls': json.dumps(calls)})
    sf.write(args.output_dir/'reference.wav', waveform.numpy(), model.sample_rate, subtype='FLOAT')
    (args.output_dir/'fixture.json').write_text(json.dumps(metadata, indent=2)+'\n')
    logging.info('fixture=%s session.wall_ms=%.3f samples=%d', args.output_dir/'fixture.safetensors', wall, waveform.numel())


if __name__ == '__main__':
    main()
