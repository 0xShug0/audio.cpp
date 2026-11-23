#!/usr/bin/env python3
"""Generate independent FP32 parity inputs/outputs with the official Python model.

Excitation is zeroed only in the decoder-isolation control. Normal output keeps
the unmodified upstream synthesis path and its natural random source.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--log', action='store_true', required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from kittenml.kittentts2_tiny.model import KittenTTS2Tiny
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = KittenTTS2Tiny(str(args.model.resolve()), backend='torch', device=args.device, threads=8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ('english', 'en', 'Eleanor', 'Hello there. This little kitten can speak clearly and naturally.'),
        ('german', 'de', 'Lena', 'Guten Morgen. Heute scheint die Sonne und wir gehen zusammen spazieren.'),
        ('spanish', 'es', 'Lucia', 'Hola. Esta pequeña voz puede hablar con claridad y naturalidad.'),
    ]
    manifest = []
    for name, lang, voice, text in cases:
        torch.manual_seed(42)
        sf.write(args.output_dir / f'{name}-python.wav', m.generate(text, voice=voice, language=lang), m.sample_rate)
        chunks = m.phonemize(text, language=lang)
        if len(chunks) != 1:
            raise ValueError(f'{name}: expected one reference chunk')
        ps = chunks[0]
        tokens = m._tokens(ps)
        _, key = m.resolve_voice(voice, lang)
        style = m._pack(lang, key)[min(len(tokens) - 1, 509)][None]
        net = m._torch_model(lang).net
        captured = {}
        def capture(module, inputs):
            captured['decoder_x'] = inputs[0].detach().cpu().numpy()
            captured['f0'] = inputs[2].detach().cpu().numpy()
        hook = net.decoder.generator.register_forward_pre_hook(capture)
        with torch.no_grad():
            _, durations = m._run(lang, tokens, style, 1.0)
        hook.remove()
        # Isolate the generator against identical input, independent of both
        # platforms' RNG implementations and upstream predictor rounding.
        source = net.decoder.generator.m_source.l_linear
        weight, bias = source.weight.detach().clone(), source.bias.detach().clone()
        with torch.no_grad():
            source.weight.zero_()
            source.bias.zero_()
            waveform = net.decoder.generator(
                torch.from_numpy(captured['decoder_x']).to(args.device),
                torch.from_numpy(style[:, :128]).to(args.device),
                torch.from_numpy(captured['f0']).to(args.device))
            source.weight.copy_(weight)
            source.bias.copy_(bias)
        for label, values in captured.items():
            np.asarray(values, np.float32).tofile(args.output_dir / f'{name}-{label}.f32')
        waveform.float().cpu().numpy().tofile(args.output_dir / f'{name}-decoder.f32')
        manifest.append(dict(name=name, language=lang, voice=voice, text=text, phonemes=ps,
                             tokens=tokens, style=style.reshape(-1).tolist(), durations=durations.tolist()))
        print('REFERENCE', name, 'tokens', len(tokens), 'duration_frames', int(durations.sum()), flush=True)
    (args.output_dir / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
