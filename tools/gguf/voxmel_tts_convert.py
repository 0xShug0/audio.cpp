#!/usr/bin/env python3
"""Package VoxMel, its Qwen tokenizer and Vocos in a self-contained spec-v1 GGUF.

Run in conda qwen3-tts. --source is a public VoxMel HF snapshot; --upstream
is the official VoxMel source checkout. Quantization/embedding are performed
by the existing audiocpp_gguf tool. No Python dependency is needed at inference.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torchaudio
from omegaconf import OmegaConf
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--vocos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--type', choices=['f32', 'f16', 'q8_0'], default='f16')
    p.add_argument('--converter', type=Path, default=REPO / 'build/debug/bin/audiocpp_gguf')
    p.add_argument('--log-file', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.output.exists() and not a.overwrite:
        p.error(f'output exists: {a.output}')
    sys.path.insert(0, str(a.upstream.resolve()))
    from voxcore.tokenizer.qwen_tokenizer import get_qwen_tokenizer
    tokenizer = get_qwen_tokenizer(str(a.upstream / 'examples/voxmel/qwen_tokenizer'), True).tokenizer
    c = OmegaConf.to_container(OmegaConf.load(a.source / 'config.yaml'), resolve=True)
    backbone = json.loads((a.upstream / 'examples/voxmel/qwen_tokenizer/config.json').read_text())
    weights = torch.load(a.source / 'model_L_final.pt', map_location='cpu', weights_only=False, mmap=True)
    if c.get('use_ema') and 'ema' in weights:
        state = dict(weights['ema'])
        online = weights.get('state_dict', {})
        online = {k.removeprefix('module.'): v for k, v in online.items()}
        for k in state:
            if k.rsplit('.', 1)[-1] in ('detail_mean', 'detail_var') and k.removeprefix('module.') in online:
                state[k] = online[k.removeprefix('module.')]
    else:
        state = weights.get('state_dict', weights)
        state = state.get('model_state_dict', state)
        state = state.get('model', state)
    # Qwen is a feature extractor: the unused vocabulary projection is not an
    # inference operand. Keep the actual token embedding, including tied data.
    state = {k.removeprefix('module.'): v.contiguous().clone()
             for k, v in state.items() if k.removeprefix('module.') != 'llm.model.lm_head.weight'}
    state['frontend.window'] = torch.hann_window(1024)
    state['frontend.mel_filterbank'] = torchaudio.functional.melscale_fbanks(
        n_freqs=513, f_min=0, f_max=12000, n_mels=c['model_conf']['n_mels'],
        sample_rate=24000, norm=None, mel_scale='htk').T.contiguous()
    vocos = torch.load(a.vocos / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    vocos = {k: v.contiguous().clone() for k, v in vocos.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.log_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='voxmel-') as temp:
        root = Path(temp)
        tokenizer.save_pretrained(root)
        (root / 'config.json').write_text(json.dumps({'model': c['model_conf'], 'backbone': backbone}, indent=2))
        save_file(state, str(root / 'model.safetensors'))
        save_file(vocos, str(root / 'vocos.safetensors'))
        cmd = [str(a.converter.resolve()), '--input', f'model={root}/model.safetensors',
               '--input', f'vocos={root}/vocos.safetensors', '--root', str(root), '--family', 'voxmel_tts',
               '--model-spec', str(REPO / 'model_specs/voxmel_tts.json'), '--type', a.type,
               '--output', str(a.output.resolve()), '--keep-type', 'vocos/*=f32',
               '--keep-type', 'model/frontend.*=f32', '--log']
        if a.overwrite:
            cmd.append('--overwrite')
        with a.log_file.open('w') as log:
            log.write('command=' + repr(cmd) + '\n')
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(a.output, a.output.stat().st_size)


if __name__ == '__main__':
    main()
