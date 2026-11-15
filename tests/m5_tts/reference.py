#!/usr/bin/env python3
"""Official M5 reference execution and model-boundary parity fixtures.

Export mode is an FP32 correctness control, never a production benchmark.
The production decoder runtime contains no fixture or debugging hooks.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['export', 'performance'], required=True)
    parser.add_argument('--runtime', choices=['torch', 'onnx'], default='torch')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--language', default='Chinese')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-text', default='')
    parser.add_argument('--speaker-id', type=int, default=1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-duration-sec', type=float, default=40)
    parser.add_argument('--log', action='store_true', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(args.output_dir / (args.mode + '.log'))], force=True)
    sys.path.insert(0, str(args.upstream.resolve()))
    import numpy as np
    import soundfile as sf
    if args.runtime == 'onnx':
        if args.mode != 'performance':
            raise ValueError('ONNX is the performance baseline; FP32 fixture export uses PyTorch')
        from infer_onnx import M5Onnx
        model = M5Onnx(str(args.model), threads=args.threads, device=args.device,
            local_files_only=True, verify=True)
        options = dict(prompt_wav=str(args.reference), prompt_text=args.reference_text,
            lang=args.language, seed=args.seed, max_frames=int(args.max_duration_sec * 12.5),
            stream=args.stream)
        warm = '这是一次准备。' if args.language == 'Chinese' else 'This is a warmup.'
        model.infer(warm, **options)
        results = []
        for index in range(2):
            begin = time.perf_counter()
            first = None
            chunks = []
            for chunk in model.generate(args.text, **options):
                if first is None and chunk.size:
                    first = (time.perf_counter() - begin) * 1000
                chunks.append(chunk)
            elapsed = time.perf_counter() - begin
            audio = np.concatenate(chunks)
            seconds = len(audio) / model.sample_rate
            result = dict(runtime='onnx', device=args.device, precision='fp32',
                threads=args.threads, seed=args.seed, streaming=args.stream,
                checkpoint_sha256=model.meta['checkpoint_sha256'],
                wall_ms=elapsed * 1000, ttfa_ms=first, audio_seconds=seconds,
                rtf=elapsed / seconds, status=model.stats['status'])
            logging.info('session.wall_ms %.6f', result['wall_ms'])
            logging.info('rtf %.6f', result['rtf'])
            results.append(result)
            sf.write(args.output_dir / f'output-{index}.wav', audio, model.sample_rate, subtype='FLOAT')
        (args.output_dir / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        return
    if args.stream:
        raise ValueError('The official PyTorch API is offline; use ONNX for streaming')
    import torch
    from safetensors.torch import save_file
    from inference import M5TTS

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    model = M5TTS.from_pretrained(str(args.model), codec=str(args.codec),
        device='cpu' if args.mode == 'export' else args.device, attention='sdpa', local_files_only=True)
    if args.mode == 'export':
        # Construct in FP32 first: upcasting a BF16 rotary buffer cannot restore
        # the original FP32 frequencies for a numerical correctness control.
        model.generator._model.to(device=args.device, dtype=torch.float32)
        model.generator._audio_tokenizer.model.to(device=args.device, dtype=torch.float32)
        model.generator._audio_tokenizer.device = torch.device(args.device)
        model.generator.device = torch.device(args.device)
        if args.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    metadata = dict(mode=args.mode, device=args.device, threads=args.threads,
        seed=args.seed, model=str(args.model.resolve()), codec=str(args.codec.resolve()),
        lm_dtype=str(next(model.generator._model.parameters()).dtype),
        codec_dtype=str(next(model.generator._audio_tokenizer.model.parameters()).dtype),
        attention='sdpa', text=args.text, language=args.language)
    infer_options = dict(prompt_wav=str(args.reference) if args.reference else None,
        prompt_text=args.reference_text if args.reference else None,
        speaker_id=args.speaker_id, lang=args.language, temperature=0.9, topk=10,
        max_audio_length_ms=args.max_duration_sec * 1000)

    fixtures = {}
    hooks = []
    selections = []
    if args.mode == 'export':
        import m5.model.model as reference_model
        original_sample = reference_model.sample_topk

        def sample(logits, topk, temperature, need_finish):
            index = len(selections)
            fixtures[f'logits.{index}'] = logits.detach().float().cpu().clone().contiguous()
            token = original_sample(logits, topk, temperature, need_finish)
            selections.append(token.detach().cpu().reshape(-1).to(torch.int32))
            return token

        reference_model.sample_topk = sample

        def boundary(name):
            count = 0

            def capture(module, inputs, kwargs):
                nonlocal count
                if count < (8 if name == 'backbone' else 8 * 15):
                    fixtures[f'{name}.input.{count}'] = kwargs['hidden_states'].detach().float().cpu().contiguous()
                    fixtures[f'{name}.positions.{count}'] = kwargs['position_ids'].detach().cpu().to(torch.int32).contiguous()
                count += 1
            return capture

        network = model.generator._model
        original_frame = network.generate_frame

        def frame(tokens, mask, *values, **kwargs):
            if 'prompt.tokens' not in fixtures:
                fixtures['prompt.tokens'] = tokens.detach().cpu().to(torch.int32).contiguous()
                fixtures['prompt.mask'] = mask.detach().cpu().to(torch.int32).contiguous()
            return original_frame(tokens, mask, *values, **kwargs)

        network.generate_frame = frame
        hooks.append(network.backbone.register_forward_pre_hook(boundary('backbone'), with_kwargs=True))
        hooks.append(network.decoder.register_forward_pre_hook(boundary('local'), with_kwargs=True))
        original_decode = model.generator._audio_tokenizer.decode

        def decode(codes, *values, **kwargs):
            fixtures['codec.codes'] = codes['audio_codes'][0].detach().cpu().to(torch.int32).contiguous()
            return original_decode(codes, *values, **kwargs)

        model.generator._audio_tokenizer.decode = decode
        original_encode = model.generator._audio_tokenizer.encode

        def encode(waveform, *values, **kwargs):
            fixtures['reference.waveform'] = torch.from_numpy(np.asarray(waveform).copy())
            result = original_encode(waveform, *values, **kwargs)
            fixtures['reference.codes'] = result.audio_codes[0].detach().cpu().to(torch.int32).contiguous()
            return result

        model.generator._audio_tokenizer.encode = encode
    else:
        # The upstream model and tokenizer caches are warmed before measurement.
        warm = '这是一次准备。' if args.language == 'Chinese' else 'This is a warmup.'
        model.infer(warm, **infer_options)
        torch.manual_seed(args.seed)

    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    audio = model.infer(args.text, **infer_options)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    metadata.update(wall_ms=elapsed * 1000, audio_seconds=audio.numel() / model.sample_rate,
                    rtf=elapsed / (audio.numel() / model.sample_rate))
    logging.info('session.wall_ms %.6f', metadata['wall_ms'])
    logging.info('rtf %.6f', metadata['rtf'])
    if args.device == 'cuda':
        # Allocator statistics complement, but do not replace, process VRAM sampling.
        metadata['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    sf.write(args.output_dir / 'output.wav', audio.numpy(), model.sample_rate, subtype='FLOAT')
    if args.mode == 'export':
        for hook in hooks:
            hook.remove()
        fixtures['sampled.tokens'] = torch.cat(selections).reshape(-1, 16).contiguous()
        fixtures['waveform'] = audio.contiguous()
        ids = fixtures['prompt.tokens'][0, :, -1][fixtures['prompt.mask'][0, :, -1].bool()].tolist()
        prepared_text = model.generator._text_tokenizer.decode(ids[1:-1], skip_special_tokens=False)
        save_file(fixtures, args.output_dir / 'reference.safetensors', metadata={'conditioned_text': prepared_text})
    (args.output_dir / 'result.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
