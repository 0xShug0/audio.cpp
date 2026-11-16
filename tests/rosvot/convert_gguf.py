#!/usr/bin/env python3
"""Convert ROSVOT, RWBD and their RMVPE frontend to one spec-v1 GGUF package."""
import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

import torch
import yaml
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Directory containing rosvot/, rwbd/, rmvpe/')
    p.add_argument('--staging', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--converter', type=Path, default=ROOT/'build/debug/bin/audiocpp_gguf')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--log', type=Path, required=True)
    a = p.parse_args()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(a.log), logging.StreamHandler()])
    a.staging.mkdir(parents=True, exist_ok=True)
    config = {}
    for name in ['rosvot', 'rwbd']:
        cfg = yaml.safe_load((a.source/name/'config.yaml').read_text())
        if (cfg['bkb_net'] != 'conformer' or set(cfg['channel_multiples'].split('-')) != {'1'}
                or set(cfg['updown_rates'].split('-')) != {'2'}):
            raise ValueError('Expected the published relative-Conformer checkpoint')
        keys = ['hidden_size', 'use_mel_bins', 'conformer_kernel', 'bkb_layers', 'note_num',
                'note_start', 'note_pitch_temperature', 'pitch_attn_num_head', 'note_bd_temperature',
                'note_bd_min_gap', 'note_bd_ref_min_gap', 'word_bd_temperature', 'word_bd_min_gap']
        config[name] = {k: cfg[k] for k in keys if k in cfg}
        config[name]['down_layers'] = len(cfg['updown_rates'].split('-'))
        state = torch.load(a.source/name/'model.pt', map_location='cpu', weights_only=False)['state_dict']['model']
        state = {k: v.contiguous() for k, v in state.items() if not k.endswith('num_batches_tracked')}
        save_file(state, a.staging/f'{name}.safetensors')
        logging.info('%s: %d tensors', name, len(state))
    rmvpe = torch.load(a.source/'rmvpe/model.pt', map_location='cpu', weights_only=False)['model']
    save_file({k: v.contiguous() for k, v in rmvpe.items() if not k.endswith('num_batches_tracked')}, a.staging/'rmvpe.safetensors')
    (a.staging/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for filename in ['LICENSE', 'LICENSE.rmvpe', 'NOTICE']:
        shutil.copyfile(ROOT/'tests/rosvot'/filename, a.staging/filename)
    cmd = [str(a.converter.resolve()), '--root', str(a.staging.resolve()), '--family', 'rosvot',
           '--model-spec', str(ROOT/'model_specs/rosvot.json'), '--type', 'f32',
           '--output', str(a.output.resolve()), '--log']
    for name in ['rosvot', 'rwbd', 'rmvpe']:
        cmd += ['--input', f'{name}={a.staging.resolve()}/{name}.safetensors']
    # Keep convolutions, embeddings and normalization in their original precision.
    # There is no lossy global conversion of the pitch extraction frontend.
    for pattern in ['rmvpe/*', 'rosvot/*', 'rwbd/*']:
        cmd += ['--keep-type', pattern+'=f32']
    if a.overwrite: cmd += ['--overwrite']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.log.open('a') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
