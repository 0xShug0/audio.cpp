#!/usr/bin/env python3
"""Pack all four official Dettato checkpoints into a standalone spec-v1 GGUF.

The upstream checkout supplies the exact frontend constants. Inference uses
no Python code or external model sidecars.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]



def prepare_native_state(stage, state):
    """Normalize checkpoint-only layouts without changing inference arithmetic."""
    for key in list(state):
        if key.endswith('.running_var'):
            prefix = key[:-len('.running_var')]
            gamma = state[prefix + '.weight'].numpy().astype(np.float32)
            beta = state[prefix + '.bias'].numpy().astype(np.float32)
            mean = state[prefix + '.running_mean'].numpy().astype(np.float32)
            variance = state[key].numpy().astype(np.float32)
            scale = gamma / np.sqrt(variance + np.float32(1e-5))
            bias = beta - mean * scale
            state[prefix + '.eval_scale'] = torch.from_numpy(scale.copy())
            state[prefix + '.eval_bias'] = torch.from_numpy(bias.copy())
            for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
                del state[prefix + suffix]
    if stage in ('transcriber', 'grouper'):
        offsets = np.array([-2,-1,1,2,-12,12,-19,19,-24,24,-28,28,-31,31], dtype=np.int32)
        order = np.argsort(offsets)
        for key in list(state):
            if key.endswith(('.pmix.taps', '.pm.taps')):
                prefix = key[:-len('.taps')]
                state[prefix + '.coefficients'] = state.pop(key)[:, order].contiguous()
                positions = np.zeros((len(offsets), 2), dtype=np.int32)
                positions[:, 0] = offsets[order]
                state[prefix + '.offsets'] = torch.from_numpy(positions)
            elif key.endswith('.rel_bias'):
                lanes = 88 if stage == 'transcriber' else 96
                table = state[key]
                positions = torch.arange(lanes)
                indices = positions[None, :] - positions[:, None] + table.shape[-1] // 2
                state[key] = table[:, indices].contiguous()
    if stage == 'transcriber':
        for key in list(state):
            if key.endswith(('.pw1.weight', '.pw2.weight')):
                weight = state[key]
                if weight.ndim != 4 or tuple(weight.shape[2:]) != (1, 1):
                    raise ValueError(f'Expected pointwise projection: {key} {tuple(weight.shape)}')
                state[key] = weight[:, :, 0, 0].contiguous()
    return state


def prepare_native_config(configs):
    configs['weight_layout'] = 'audiocpp_v1'
    config = configs['drums']['drum_kit']
    layers, index = [], 0
    for token in config['stem'].split(','):
        if token.startswith('c'):
            layers.append({'kind':'conv', 'index':index}); index += 3
        elif token == 'r':
            layers.append({'kind':'residual', 'index':index}); index += 1
        elif token.startswith('p') and len(token) >= 3:
            layers.append({'kind':'pool', 'index':index, 'time':int(token[1]), 'frequency':int(token[2:])}); index += 1
        else:
            raise ValueError(f'Unknown drum stem token: {token}')
    config['stem_layers'] = layers
    del config['stem']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter', type=Path, default=REPO / 'build/debug/bin/audiocpp_gguf')
    parser.add_argument('--log-file', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--weight-type', choices=['f32', 'f16'], default='f16')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f'Output exists: {args.output}')
    sys.path.insert(0, str(args.upstream.resolve() / 'src'))
    from dettato.transcriber.model import LaneNet
    from dettato.grouper.model import NoteNet
    from dettato.grouper.instruments import CREDIT
    from dettato.drums.features import _filters
    stages = {'transcriber':'dettato', 'evidence':'dettato-evidence', 'grouper':'dettato-grouper', 'drums':'dettato-drums'}
    configs = {stage:json.loads((args.source / name / 'config.json').read_text()) for stage, name in stages.items()}
    prepare_native_config(configs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dettato-convert-') as directory:
        root = Path(directory)
        (root / 'config.json').write_text(json.dumps(configs, indent=2))
        inputs = []
        for stage, name in stages.items():
            state = {k:v.contiguous() for k,v in load_file(str(args.source / name / 'model.safetensors')).items()
                     if not k.endswith('.num_batches_tracked')}
            if stage in ('transcriber','grouper'):
                model = (LaneNet if stage=='transcriber' else NoteNet)(**configs[stage]['model'])
                model.load_state_dict(load_file(str(args.source / name / 'model.safetensors')))
                for key, value in model.front.named_buffers():
                    state['front.' + key] = value.contiguous()
            if stage=='drums':
                for name, nfft in [('lo',512),('hi',1411),('cq',6144)]:
                    state['front.' + name + '.filterbank'] = _filters(name).contiguous()
                    state['front.' + name + '.window'] = torch.hann_window(nfft)
            if stage=='grouper':
                state['instrument_credit'] = torch.from_numpy(CREDIT).float().contiguous()
            state = prepare_native_state(stage, state)
            # Keep frontend constants, normalization, evidence and host-recurrent
            # matrices in F32. Only the actual GEMM/convolution weights use F16.
            if args.weight_type == 'f16' and stage != 'evidence':
                for key, value in state.items():
                    projection = key.endswith('.weight') or 'weight_ih_l' in key or (stage == 'transcriber' and 'weight_hh_l' in key)
                    depthwise = value.ndim == 4 and value.shape[1] == 1
                    if projection and value.ndim >= 2 and not depthwise and not key.startswith(('front.', 'pitch_emb.')):
                        state[key] = value.half()
            save_file(state, str(root / (stage + '.safetensors')))
            inputs += ['--input', f'{stage}={root}/{stage}.safetensors']
        command = [str(args.converter.resolve()), *inputs, '--root', str(root), '--family', 'dettato',
                   '--model-spec', str(REPO / 'model_specs/dettato.json'), '--type', 'native',
                   '--output', str(args.output.resolve()), '--log']
        if args.overwrite:
            command.append('--overwrite')
        with args.log_file.open('w') as log:
            log.write('command=' + repr(command) + '\n');log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(args.output, args.output.stat().st_size)


if __name__ == '__main__':
    main()
