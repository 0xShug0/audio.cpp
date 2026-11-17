#!/usr/bin/env python3
"""Package SoulX-Singer SVS or SVC with its inference dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--whisper', type=Path)
    p.add_argument('--variant', choices=['svc','svs'], default='svc')
    p.add_argument('--phoneset', type=Path)
    p.add_argument('--rmvpe', type=Path)
    p.add_argument('--staging', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--type', choices=['f32', 'f16', 'q8_0'], default='f16')
    p.add_argument('--converter', type=Path, default=Path('build/debug/bin/audiocpp_gguf'))
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    a.staging.mkdir(parents=True, exist_ok=True)
    notices = ['LICENSE', 'LICENSE.amphion', 'NOTICE']
    if a.variant == 'svc': notices += ['LICENSE.whisper', 'LICENSE.rmvpe']
    for name in notices:
        shutil.copyfile(Path(__file__).resolve().parent/name, a.staging/name)
    state = torch.load(a.source, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    model = {k.removeprefix('cfm_decoder.model.diff_estimator.'): v.contiguous()
             for k, v in state.items() if k.startswith('cfm_decoder.model.diff_estimator.')}
    model.update({k: state[k].contiguous() for k in ['f0_encoder.weight']})
    for field in ['weight', 'bias']:
        model[f'cond_emb.{field}'] = state[f'cfm_decoder.model.cond_emb.{field}'].contiguous()
    layers = 0
    while f'layers.{layers}.mlp.gate_proj.weight' in model:
        base = f'layers.{layers}.mlp.'
        model[base + 'in.weight'] = torch.cat([model.pop(base + 'up_proj.weight'), model.pop(base + 'gate_proj.weight')])
        layers += 1
    if a.variant == 'svs':
        model.update({k:v.contiguous() for k,v in state.items() if k.startswith(('note_', 'preflow.'))})
        if a.phoneset is None: p.error('--phoneset is required for SVS')
        shutil.copyfile(a.phoneset,a.staging/'phone_set.json')
    save_file(model, a.staging/'model.safetensors')
    vocos = {k.removeprefix('vocoder.model.'): v.contiguous() for k, v in state.items()
             if k.startswith('vocoder.model.')}
    save_file(vocos, a.staging/'vocos.safetensors')
    if a.variant == 'svc':
        if a.whisper is None or a.rmvpe is None: p.error('--whisper and --rmvpe are required for SVC')
        whisper = load_file(a.whisper/'model.safetensors')
        normalized = {}
        for key, value in whisper.items():
            key = key.removeprefix("model.")
            if not key.startswith('encoder.'): continue
            name = key.replace('encoder.embed_positions.weight', 'encoder.positional_embedding')
            name = name.replace('.layers.', '.blocks.').replace('.self_attn_layer_norm.', '.attn_ln.')
            name = name.replace('.final_layer_norm.', '.mlp_ln.').replace('encoder.layer_norm.', 'encoder.ln_post.')
            for old, new in [('q_proj', 'query'), ('k_proj', 'key'), ('v_proj', 'value'), ('out_proj', 'out')]:
                name = name.replace('.self_attn.'+old+'.', '.attn.'+new+'.')
            name = name.replace('.fc1.', '.mlp.0.').replace('.fc2.', '.mlp.2.')
            normalized[name] = value.contiguous()
        save_file(normalized, a.staging/'whisper.safetensors')
        rmvpe = torch.load(a.rmvpe, map_location='cpu', weights_only=True)
        save_file({k: v.contiguous() for k,v in rmvpe.items() if not k.endswith('num_batches_tracked')}, a.staging/'rmvpe.safetensors')
    config = dict(task="svc" if a.variant == "svc" else "gen", layers=layers, hidden_size=1024, intermediate_size=4096, heads=16,
                  sample_rate=24000, hop_length=480, n_fft=1920, mel_bins=128,
                  mel_mean=-4.92, mel_variance=8.14, whisper_hidden=512, whisper_heads=8, whisper_layers=6)
    (a.staging/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    family = 'soulx_singer'
    spec = Path(__file__).resolve().parents[2]/f'model_specs/{family}.json'
    cmd = [str(a.converter.resolve()), '--root', str(a.staging.resolve()), '--family', family,
           '--model-spec', str(spec), '--type', a.type, '--output', str(a.output.resolve()), '--log']
    for name in (['model','vocos','whisper','rmvpe'] if a.variant == 'svc' else ['model','vocos']): cmd += ['--input', f'{name}={a.staging.resolve()}/{name}.safetensors']
    for pattern in ['rmvpe/*', 'whisper/*', 'model/f0_encoder.weight', 'model/cond_emb*',
                    'model/diff_step_mlp*', 'model/norm.to_weight*', 'model/preflow*', 'model/note_*']:
        cmd += ['--keep-type', pattern+'=f32']
    for name in vocos:
        matrix = name.endswith(('pwconv1.weight', 'pwconv2.weight', 'head.out.weight'))
        storage = 'f16' if matrix and a.type != 'f32' else 'f32'
        cmd += ['--keep-type', f'vocos/{name}={storage}']
    for layer in range(layers):
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            cmd += ['--keep-type', f'model/layers.{layer}.{norm}*=f32']
    if a.overwrite: cmd.append('--overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__': main()
