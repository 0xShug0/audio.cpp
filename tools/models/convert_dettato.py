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

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]


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
