"""Validate hybrid Q4 tensor storage and values against the updated BF16 source."""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch
from safetensors import safe_open


def tensor_index(path):
    with path.open('rb') as stream:
        def read(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))

        def string():
            return stream.read(read('Q')[0]).decode('utf-8')

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(kind):
            if kind == 8:
                stream.seek(read('Q')[0], 1)
            elif kind == 9:
                subtype, count = read('IQ')
                if subtype in (8, 9):
                    for _ in range(count):
                        skip(subtype)
                else:
                    stream.seek(count * sizes[subtype], 1)
            else:
                stream.seek(sizes[kind], 1)

        magic, version, count, metadata = read('4sIQQ')
        if magic != b'GGUF' or version != 3:
            raise ValueError('expected GGUF v3')
        for _ in range(metadata):
            string()
            skip(read('I')[0])
        tensors = {}
        for _ in range(count):
            name = string()
            shape = read('Q' * read('I')[0])[::-1]
            kind, offset = read('IQ')
            tensors[name] = (shape, kind, offset)
        return (stream.tell() + 31) // 32 * 32, tensors


def decode(payload, kind):
    if kind == 30:
        return (np.frombuffer(payload, '<u2').astype(np.uint32) << 16).view(np.float32)
    if kind == 44:
        blocks = np.frombuffer(payload, np.uint8).reshape(-1, 68)
        scales = blocks[:, 64:].copy().view('<f2').astype(np.float32)
        codes = blocks[:, :64].reshape(-1, 2, 32)
        values = np.stack([(codes >> shift) & 3 for shift in (0, 2, 4, 6)], axis=2)
        return ((values.astype(np.float32).reshape(-1, 2, 128) - 1) * scales[..., None]).ravel()
    blocks = np.frombuffer(payload, np.uint8).reshape(-1, 18 if kind == 2 else 34)
    scales = blocks[:, :2].copy().view('<f2').astype(np.float32)
    codes = (np.concatenate((blocks[:, 2:] & 15, blocks[:, 2:] >> 4), 1).astype(np.float32) - 8
             if kind == 2 else blocks[:, 2:].copy().view(np.int8).astype(np.float32))
    return (codes * scales).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start, tensors = tensor_index(args.gguf)
    mapping = json.loads((args.model / 'config.json').read_text())['token_map']
    audio_start, audio_end = mapping['audio_id_base'], mapping['audio_id_base'] + mapping['num_audio_tokens']
    report = {'exact_body_matrices': 0, 'exact_dense_tensors': 0}
    errors = {'embedding': [0.0, 0.0], 'speech_tokens': [0.0, 0.0]}
    with safe_open(args.model / 'lm/model.safetensors', framework='pt') as full, args.gguf.open('rb') as stream:
        for name, (shape, kind, offset) in tensors.items():
            if not name.startswith('language_model/'):
                continue
            source_name = name.removeprefix('language_model/')
            stream.seek(start + offset)
            if source_name == 'model.embed_tokens.weight':
                if kind != 8:
                    raise RuntimeError('hybrid embedding must be Q8_0')
                source = full.get_slice(source_name)
                if tuple(source.get_shape()) != shape:
                    raise RuntimeError('embedding shape differs')
                row_bytes = shape[1] // 32 * 34
                for begin in range(0, shape[0], 512):
                    end = min(begin + 512, shape[0])
                    actual = decode(stream.read((end - begin) * row_bytes), kind).reshape(end - begin, shape[1])
                    expected = source[begin:end].float().numpy()
                    if not np.isfinite(actual).all():
                        raise RuntimeError('non-finite embedding')
                    for label, first, last in [('embedding', begin, end), ('speech_tokens', max(begin, audio_start), min(end, audio_end))]:
                        if first >= last:
                            continue
                        reference = expected[first-begin:last-begin].astype(np.float64)
                        restored = actual[first-begin:last-begin].astype(np.float64)
                        errors[label][0] += float(np.sum((restored-reference)**2))
                        errors[label][1] += float(np.sum(reference**2))
            else:
                if kind not in (2, 30):
                    raise RuntimeError(f'unexpected LM storage: {name}')
                size = math.prod(shape) * 2 if kind == 30 else math.prod(shape) // 32 * 18
                actual = decode(stream.read(size), kind).reshape(shape)
                expected = full.get_tensor(source_name).float().numpy()
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f'conversion changed {name}')
                report['exact_body_matrices' if kind == 2 else 'exact_dense_tensors'] += 1
    if report['exact_body_matrices'] != 196:
        raise RuntimeError('expected 196 Q4 body matrices')
    for label, (error, norm) in errors.items():
        report[label + '_relative_l2_vs_bf16'] = (error / norm) ** .5
        if report[label + '_relative_l2_vs_bf16'] > .005:
            raise RuntimeError(f'{label} Q8 error exceeds 0.5%')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
