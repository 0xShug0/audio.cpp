#!/usr/bin/env python3
"""Package Foundation-1 and T5-base as a self-contained spec-v1 GGUF.

The checkpoint already uses safetensors. Preserve its tensor names and let
audiocpp_gguf perform storage conversion; convolution and conditioning operands
remain F32. No upstream inference code is needed during conversion.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--text-encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--type', choices=['f32', 'f16', 'q8_0'], default='f16')
    parser.add_argument('--converter', type=Path, default=ROOT / 'build/debug/bin/audiocpp_gguf')
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='foundation-1-') as staging:
        root = Path(staging)
        shutil.copyfile(args.source / 'model_config.json', root / 'config.json')
        shutil.copyfile(args.source / 'LICENSE.md', root / 'LICENSE.md')
        text = root / 'text_encoder'
        text.mkdir()
        for filename in ['config.json', 'tokenizer.json', 'spiece.model']:
            shutil.copyfile(args.text_encoder / filename, text / filename)
        command = [str(args.converter.resolve()), '--family', 'foundation_1',
                   '--model-spec', str(ROOT / 'model_specs/foundation_1.json'),
                   '--root', str(root), '--type', args.type,
                   '--input', f'model={args.source.resolve()}/Foundation_1.safetensors',
                   '--input', f'text_encoder={args.text_encoder.resolve()}/model.safetensors',
                   '--exclude-prefix', 'text_encoder/decoder.',
                   '--exclude-prefix', 'text_encoder/lm_head.',
                   '--keep-type', 'model/pretransform.*=f32',
                   '--keep-type', 'model/conditioner.*=f32',
                   '--keep-type', 'model/model.model.timestep_features.*=f32',
                   '--output', str(args.output.resolve()), '--log']
        if args.overwrite:
            command.append('--overwrite')
        with args.log.open('w') as log:
            log.write('command=' + repr(command) + '\n')
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
