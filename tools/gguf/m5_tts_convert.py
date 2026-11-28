#!/usr/bin/env python3
"""Package M5-TTS and its Qwen3 codec using the framework GGUF converter."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--variant', choices=['voice_clone', 'custom_voice'], required=True)
    parser.add_argument('--type', default='native', choices=['native', 'f32', 'f16', 'bf16', 'q8_0', 'q4_0'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter', type=Path, default=Path('build/debug/bin/audiocpp_gguf'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    config = json.loads((args.source / 'config.json').read_text())
    conditioning = json.loads((args.source / 'conditioning.json').read_text())
    # These are the published architecture definitions, independent of tensor names.
    flavors = {'llama-30M': 8, 'llama-20M': 5}
    normalized = dict(variant=args.variant, text_vocab_size=conditioning['vocab_size'],
                      audio_vocab_size=config['audio_vocab_size'],
                      audio_num_codebooks=config['audio_num_codebooks'], max_position_embeddings=2048)
    for key in ['backbone', 'decoder']:
        normalized[key] = dict(layers=flavors[config[key + '_flavor']], hidden_size=512,
                               intermediate_size=2048, num_attention_heads=8,
                               num_key_value_heads=2, head_dim=64,
                               rms_norm_eps=1e-5, rope_theta=500000.0)
    state = load_file(str(args.source / 'model.safetensors'), device='cpu')
    heads = state.pop('audio_head')
    for index, head in enumerate(heads):
        state[f'audio_heads.{index}.weight'] = head.T.contiguous()
    for key in ['backbone', 'decoder']:
        c = normalized[key]
        base = 1.0 / (c['rope_theta'] ** (torch.arange(0, c['head_dim'], 2).float() / c['head_dim']))
        wavelength = 2.0 * torch.pi / base
        smooth = (2048.0 / wavelength - 1.0) / 3.0
        scaled = torch.where(wavelength > 2048.0, base / 32.0, base)
        medium = (wavelength >= 512.0) & (wavelength <= 2048.0)
        scaled = torch.where(medium, (1.0 - smooth) * base / 32.0 + smooth * base, scaled)
        state[key + '.rope_factors'] = (base / scaled).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[2] / 'model_specs/m5_tts.json'
    with tempfile.TemporaryDirectory(prefix='.m5-convert-', dir=args.output.parent) as temporary:
        root = Path(temporary) / ('M5-TTS-50M-VoiceClone' if args.variant == 'voice_clone' else 'M5-TTS-50M-CustomVoice')
        root.mkdir()
        (root / 'config.json').write_text(json.dumps(normalized, indent=2) + '\n')
        shutil.copyfile(args.codec / 'config.json', root / 'codec_config.json')
        shutil.copyfile(args.source / 'conditioning.json', root / 'conditioning.json')
        shutil.copytree(args.source / 'tokenizer', root / 'tokenizer')
        save_file(state, root / 'model.safetensors')
        shutil.copyfile(args.codec / 'model.safetensors', root / 'codec.safetensors')
        command = [str(args.converter.resolve()), '--input', f'model={root / "model.safetensors"}',
                   '--input', f'codec={root / "codec.safetensors"}', '--root', str(root),
                   '--family', 'm5_tts', '--model-spec', str(spec), '--type', args.type,
                   '--keep-type', 'codec/*=f32',
                   '--keep-type', 'model/backbone.rope_factors=f32',
                   '--keep-type', 'model/decoder.rope_factors=f32',
                   '--output', str(args.output.resolve())]
        if args.overwrite:
            command.append('--overwrite')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
