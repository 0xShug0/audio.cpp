"""Check all updated packed LM matrices against the full upstream weights."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/community_models'))
import kitten_tts2_weights as converter
from test_updated_weights import dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = {}
    with safe_open(args.model / 'lm/model.safetensors', framework='pt') as full:
        with safe_open(args.model / 'lm/model-ternary.safetensors', framework='pt') as source:
            names = json.loads(source.metadata()['packed'])
            for name, info in names.items():
                reference = full.get_tensor(name)
                actual = converter.unpack_ternary(source.get_tensor(name+'::codes'), source.get_tensor(name+'::scale'), info['shape'])
                if not torch.equal(actual, reference): raise RuntimeError(f'unpacking changed {name}')
                for bits in (4, 8):
                    encoded = converter.exact_ternary_blocks(reference, bits)
                    if not np.array_equal(dequantize(encoded, bits).reshape(reference.shape), reference.float().numpy()):
                        raise RuntimeError(f'Q{bits}_0 packing changed {name}')
            if len(names) != 196: raise RuntimeError('unexpected ternary body matrix count')
            if set(names) != {name for name in full.keys() if converter.is_body_projection(name)}:
                raise RuntimeError('full and packed body projection sets differ')
            report = {'exact_body_matrices': len(names), 'q8_repacking_exact': True, 'q4_repacking_exact': True}
            print('196 matrices match full BF16 exactly before and after Q8_0/Q4_0 packing', flush=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
