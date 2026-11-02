#!/usr/bin/env python3
"""Package FlashHead DiT, variant-specific VAE and Wav2Vec2 weights in one GGUF."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from contextlib import nullcontext


# Checkpoint mapping from SoulX-FlashHead, Apache-2.0.
VAE_KEYS_RENAME_DICT = {
    "decoder.up_blocks.3.conv_in": "decoder.up_blocks.7",
    "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.8",
    "decoder.up_blocks.3": "decoder.up_blocks.9",
    "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
    "decoder.up_blocks.2.conv_in": "decoder.up_blocks.4",
    "decoder.up_blocks.2": "decoder.up_blocks.6",
    "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.2",
    "decoder.up_blocks.1": "decoder.up_blocks.3",
    "decoder.up_blocks.0": "decoder.up_blocks.1",
    "decoder.mid_block": "decoder.up_blocks.0",
    "encoder.down_blocks.3": "encoder.down_blocks.8",
    "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.7",
    "encoder.down_blocks.2": "encoder.down_blocks.6",
    "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.4",
    "encoder.down_blocks.1.conv_out": "encoder.down_blocks.5",
    "encoder.down_blocks.1": "encoder.down_blocks.3",
    "encoder.down_blocks.0.conv_out": "encoder.down_blocks.2",
    "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.mid_block": "encoder.down_blocks.9",
    "conv_shortcut.conv": "conv_shortcut",
    "resnets": "res_blocks",
    "norm3": "norm3.norm",
    "latents_mean": "per_channel_statistics.mean-of-means",
    "latents_std": "per_channel_statistics.std-of-means",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audio-encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('lite', 'pro', 'pretrained'), default='lite')
    parser.add_argument('--converter', type=Path, default=Path('build/debug/bin/audiocpp_gguf'))
    parser.add_argument('--type', choices=('orig', 'bf16', 'q8_0'), default='bf16')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Write normalized safetensors and config to the output directory without creating a GGUF')
    args = parser.parse_args()
    source = args.source.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = Path(__file__).resolve().parents[3] / 'model_specs/soulx_flashhead.json'
    if args.prepare_only:
        args.output.mkdir(parents=True,exist_ok=False)
    context = (nullcontext(args.output) if args.prepare_only else
               tempfile.TemporaryDirectory(prefix='.flashhead-',dir=args.output.parent))
    with context as temporary:
        root = Path(temporary)
        model_dir = source / {'lite': 'Model_Lite', 'pro': 'Model_Pro', 'pretrained': 'teacher'}[args.variant]
        config = json.loads((model_dir / 'config.json').read_text())
        config['variant'] = args.variant
        config['vae'] = (json.loads((source / 'VAE_LTX/config.json').read_text())
                         if args.variant == 'lite' else {'architecture': 'wan2.1'})
        config['audio_encoder'] = json.loads((args.audio_encoder / 'config.json').read_text())
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        # Preserve each VAE's upstream names and explicit BF16 storage.
        import torch
        from safetensors.torch import load_file, save_file
        tensors = (load_file(source / 'VAE_LTX/diffusion_pytorch_model.safetensors')
                   if args.variant == 'lite' else
                   torch.load(source / 'VAE_Wan/Wan2.1_VAE.pth', map_location='cpu', weights_only=True))
        renamed = {}
        for name, value in tensors.items():
            if args.variant == 'lite':
                for before, after in VAE_KEYS_RENAME_DICT.items():
                    name = name.replace(before, after)
            if name in renamed:
                raise ValueError('Duplicate VAE tensor: ' + name)
            if value.ndim == 5:
                value = value.flatten(0, 1)
            renamed[name] = value.to(torch.bfloat16)
        vae = root / 'vae.safetensors'
        save_file(renamed, vae)
        dit = root / 'dit.safetensors'
        dit_tensors = load_file(model_dir / 'diffusion_pytorch_model.safetensors')
        save_file({k: (v.flatten(0, 1) if v.ndim == 5 else v).to(torch.bfloat16)
                   for k, v in dit_tensors.items()}, dit)
        del tensors, renamed, dit_tensors
        audio = root / 'audio.safetensors'
        audio_tensors = load_file(args.audio_encoder / 'model.safetensors')
        save_file({k.removeprefix('wav2vec2.'): v for k, v in audio_tensors.items() if k.startswith('wav2vec2.')}, audio)
        if args.prepare_only:
            return
        command = [str(args.converter.resolve()), '--input', 'dit=' + str(dit),
                   '--input', 'vae=' + str(vae),
                   '--input', 'audio=' + str(audio),
                   '--root', str(root), '--model-spec', str(spec), '--type', args.type,
                   '--keep-type', 'vae/*=bf16', '--keep-type', 'audio/*=f32',
                   '--output', str(args.output.resolve())]
        if args.overwrite:
            command.append('--overwrite')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
