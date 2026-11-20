#!/usr/bin/env python3
"""Capture official VoxMel component boundaries for independent native replay.

No fixtures/weights are stored in the source repository. Use upstream's pinned
Python dependencies. This captures correctness evidence, not performance data.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import soundfile as sf
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'model', 'vocos', 'reference-audio', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--reference-text', required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--log', action='store_true', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.upstream.resolve()))
    from voxcore import VoxMel
    tokens = a.upstream / 'examples/voxmel/qwen_tokenizer'
    m = VoxMel.from_pretrained(a.model, tokenizer_path=tokens, vocos_path=a.vocos,
                              llm_config=str(tokens), device='cuda', ncpu=8)
    records = {}
    def keep(name, value):
        records[name] = value.detach().cpu().contiguous().clone()
    ar = m.model.llm.forward_one_step
    ar_count = 0
    def forward(xs, masks, cache=None):
        nonlocal ar_count
        out, new_cache = ar(xs, masks, cache)
        keep(f'ar.{ar_count}.input', xs)
        keep(f'ar.{ar_count}.hidden', out[:, -1])
        keep(f'ar.{ar_count}.stop', m.model.stop_token_head(out[:, -1]))
        ar_count += 1
        return out, new_cache
    m.model.llm.forward_one_step = forward
    def capture_flow(flow, prefix):
        sample, velocity = flow.sample, flow.velocity
        count = 0
        def run(cond, *args, **kwargs):
            nonlocal count
            key = f'{prefix}.{count}'
            keep(key + '.condition', cond)
            for source, target in [('cond_dropT', 'drop_text'), ('cond_dropTS', 'drop_both')]:
                if kwargs.get(source) is not None: keep(key + '.' + target, kwargs[source])
            first = True
            def predict(phi, t, condition, *rest, **kw):
                nonlocal first
                y = velocity(phi, t, condition, *rest, **kw)
                if first:
                    keep(key + '.noise', phi[:cond.size(0)])
                    keep(key + '.velocity0', y)
                    first = False
                return y
            flow.velocity = predict
            try: result = sample(cond, *args, **kwargs)
            finally: flow.velocity = velocity
            keep(key + '.output', result)
            count += 1
            return result
        flow.sample = run
    capture_flow(m.model.coarse_flow, 'coarse')
    for i, tier in enumerate(m.model.up_tiers):
        context = tier._context
        def conditioned(x, fn=context, index=i):
            keep(f'tier.{index}.known', x)
            y = fn(x)
            keep(f'tier.{index}.context', y)
            return y
        tier._context = conditioned
        capture_flow(tier.flow, f'residual{i}')
    inference = m.model.inference_with_prompt
    def infer(*args, **kwargs):
        keep('reference.mel', args[0])
        keep('reference.ids', args[1].to(torch.int32))
        keep('target.ids', args[2].to(torch.int32))
        result = inference(*args, **kwargs)
        keep('mel', result)
        return result
    m.model.inference_with_prompt = infer
    wav = m.generate(a.text, prompt_wav_path=a.reference_audio,
                     prompt_text=a.reference_text, seed=a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    keep('waveform', torch.from_numpy(wav))
    save_file(records, str(a.output))
    sf.write(a.output.with_suffix('.wav'), wav, m.sample_rate, subtype='FLOAT')
    metadata = dict(text=a.text, reference_text=a.reference_text, seed=a.seed,
                    ar_steps=ar_count, waveform_samples=len(wav), tensors=len(records))
    a.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metadata)


if __name__ == '__main__':
    main()
