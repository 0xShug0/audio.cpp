#!/usr/bin/env python3
"""Check XTTS conditioning against Torch operations used by upstream Coqui.

Uses published GGUF tensors, not a second checkpoint. Compares the existing
framework tensor trace samples; this is component evidence, not full WAV parity.
Upstream: TTS/tts/layers/xtts/{latent_encoder,perceiver_encoder}.py.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import gguf
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream-hifigan', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    reader = gguf.GGUFReader(str(args.model))
    weights = {}
    for tensor in reader.tensors:
        if tensor.name.startswith(('gpt.cond_enc.', 'gpt.perc.')):
            data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            weights[tensor.name] = torch.from_numpy(data.copy().reshape(tuple(reversed(tensor.shape)))).float()
    def linear(x, prefix, bias=True):
        weight = weights[prefix + '.weight']
        if weight.ndim == 3:
            weight = weight.squeeze(-1)
        return F.linear(x, weight, weights[prefix + '.bias'] if bias else None)
    waveform, sr = sf.read(args.reference, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(waveform).mean(-1).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, sr, 22050)[..., :22050 * 6]
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=256,
        win_length=1024, n_mels=80, f_min=0, f_max=8000, norm='slaney', mel_scale='htk')(waveform)
    mel = mel.clamp_min(1e-5).log()
    expected = {'xtts_v2.conditioning.mel': mel}
    norms = torch.from_numpy(np.fromfile(args.model.parent / 'mel_norms.bin', dtype='<f4').copy())
    x = linear((mel / norms[None, :, None]).transpose(1, 2), 'gpt.cond_enc.init')
    with torch.inference_mode():
        for layer in range(6):
            prefix = f'gpt.cond_enc.attn.{layer}'
            x = F.group_norm(x.transpose(1, 2), 32, weights[prefix + '.norm.weight'],
                             weights[prefix + '.norm.bias'], eps=1e-5).transpose(1, 2)
            qkv = linear(x, prefix + '.qkv').transpose(1, 2).reshape(16, 3 * 64, -1)
            q, k, v = qkv.chunk(3, dim=1)
            scale = 64 ** -.25
            attention = torch.einsum('bct,bcs->bts', q * scale, k * scale).softmax(-1)
            context = torch.einsum('bts,bcs->bct', attention, v).reshape(1, 1024, -1).transpose(1, 2)
            x = x + linear(context, prefix + '.proj_out')
        expected['xtts_v2.conditioning.encoder'] = x
        latents = weights['gpt.perc.latents'].unsqueeze(0)
        for layer in range(2):
            prefix = f'gpt.perc.layers.{layer}'
            q = linear(latents, prefix + '.0.to_q', False).reshape(1, 32, 8, 64).transpose(1, 2)
            k, v = linear(torch.cat([latents, x], 1), prefix + '.0.to_kv', False).chunk(2, dim=-1)
            k = k.reshape(1, -1, 8, 64).transpose(1, 2)
            v = v.reshape(1, -1, 8, 64).transpose(1, 2)
            attention = ((q @ k.transpose(-2, -1)) * (64 ** -.5)).softmax(-1)
            context = (attention @ v).transpose(1, 2).reshape(1, 32, 512)
            latents = latents + linear(context, prefix + '.0.to_out', False)
            value, gate = linear(latents, prefix + '.1.0').chunk(2, dim=-1)
            latents = latents + linear(value * F.gelu(gate), prefix + '.1.2')
        latents = F.normalize(latents, dim=-1) * (1024 ** .5) * weights['gpt.perc.norm.gamma']
        expected['xtts_v2.conditioning.latents'] = latents
    if args.upstream_hifigan:
        # Load upstream classes without its checkpoint I/O dependency. No
        # architecture or forward-method source is changed.
        tree = ast.parse(args.upstream_hifigan.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == 'TTS.utils.io')]
        upstream = {'__name__': 'xtts_upstream_reference'}
        exec(compile(tree, str(args.upstream_hifigan), 'exec'), upstream)
        encoder = upstream['ResNetSpeakerEncoder'](log_input=True).eval()
        speaker_weights = {}
        vocoder = gguf.GGUFReader(str(args.model.parent / 'xtts-v2-vocoder-f16.gguf'))
        for tensor in vocoder.tensors:
            if not tensor.name.startswith('hifigan_decoder.se.'):
                continue
            name = tensor.name.removeprefix('hifigan_decoder.se.')
            if name.endswith('.rm'): name = name[:-3] + '.running_mean'
            if name.endswith('.rv'): name = name[:-3] + '.running_var'
            data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            speaker_weights[name] = torch.from_numpy(data.copy().reshape(tuple(reversed(tensor.shape)))).float()
        # Batch counters are training-only; all inference parameters and running
        # statistics must be supplied by the GGUF.
        missing, unexpected = encoder.load_state_dict(speaker_weights, strict=False)
        assert all(name.endswith('.num_batches_tracked') for name in missing), missing
        assert not unexpected, unexpected
        raw, sr = sf.read(args.reference, dtype='float32', always_2d=True)
        raw = torch.from_numpy(raw).mean(-1).unsqueeze(0)
        raw = torchaudio.functional.resample(raw, sr, 22050)[..., :22050 * 30]
        raw = torchaudio.functional.resample(raw, 22050, 16000)
        emphasized = F.conv1d(F.pad(raw.unsqueeze(1), (1, 0), mode='reflect'), torch.tensor([[[-.97, 1.]]])).squeeze(1)
        power_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
            hop_length=160, n_mels=64, window_fn=torch.hamming_window)(emphasized)
        expected['xtts_v2.speaker.power_mel'] = power_mel
        with torch.inference_mode():
            expected['xtts_v2.speaker.embedding'] = encoder(power_mel.clone(), l2_norm=True)
    report = {}
    for key, tensor in expected.items():
        line = next(line for line in args.log.read_text().splitlines() if key + ' shape=' in line)
        pairs = re.findall(r'(\d+):([-+\deE.]+)', line.split('samples=')[1])
        indices = torch.tensor([int(i) for i, _ in pairs])
        actual = torch.tensor([float(value) for _, value in pairs])
        reference = tensor.flatten()[indices]
        delta = actual - reference
        row = {'samples': len(pairs), 'max_abs': delta.abs().max().item(),
               'relative_l2': (delta.norm() / reference.norm().clamp_min(1e-12)).item()}
        report[key] = row
        if row['relative_l2'] > 2e-3 or (not key.endswith('power_mel') and row['max_abs'] > .05):
            raise AssertionError(f'{key}: {row}')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
