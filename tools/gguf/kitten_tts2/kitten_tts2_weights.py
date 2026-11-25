"""Offline Kitten ternary checkpoint reader for GGML Q8_0/Q4_0/TQ2_1 storage.

Packing follows KittenML/KittenTTS kittenml/kittentts2/ternary.py.
Updated exports require NumPy, PyTorch and safetensors during conversion only.
TQ2_1 uses audio.cpp's type ID 44, not the Kitten llama.cpp fork's ID 43.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def unpack_ternary(codes, scale, shape):
    if len(shape) != 2 or shape[1] % 128 or tuple(scale.shape) != (shape[0], shape[1] // 128):
        raise ValueError("invalid ternary matrix geometry")
    packed = codes.numpy()
    count = math.prod(shape)
    if packed.dtype != np.uint8 or packed.size != (count + 4) // 5 or np.any(packed > 242):
        raise ValueError("invalid base-3 ternary codes")
    digits = ((packed[:, None].astype(np.int16) // np.array([1, 3, 9, 27, 81])) % 3 - 1)
    values = torch.from_numpy(digits.reshape(-1)[:count].copy()).to(torch.bfloat16)
    return (values.reshape(shape[0], -1, 128) * scale[..., None]).reshape(shape)


def exact_ternary_blocks(tensor, bits):
    """Encode BF16 ternary values exactly in ordinary Q4_0 or Q8_0 blocks."""
    if bits not in (4, 8) or tensor.ndim != 2 or tensor.shape[-1] % 32:
        raise ValueError("invalid GGML block geometry")
    values = tensor.float().numpy().reshape(-1, 32)
    gamma = np.max(np.abs(values), axis=1)
    if not np.all(np.isfinite(values)) or np.any((values != 0) & (np.abs(values) != gamma[:, None])):
        raise ValueError("matrix is not ternary within each GGML block")
    magnitude = np.full(gamma.shape, 4 if bits == 4 else 64, dtype=np.float32)
    scale = (gamma / magnitude).astype('<f2')
    while np.any(scale.astype(np.float32) * magnitude != gamma):
        bad = scale.astype(np.float32) * magnitude != gamma
        if np.any(magnitude[bad] == 1):
            raise ValueError("ternary scale is not exactly representable in standard GGML")
        magnitude[bad] /= 2
        scale = (gamma / magnitude).astype('<f2')
    codes = (np.sign(values) * magnitude[:, None]).astype(np.int8)
    if bits == 4:
        codes = (codes + 8).astype(np.uint8)
        codes = codes[:, :16] | (codes[:, 16:] << 4)
    else:
        codes = codes.view(np.uint8)
    return np.concatenate((scale.view(np.uint8).reshape(-1, 2), codes), axis=1).tobytes()


def exact_ternary_tq2_1(tensor):
    """Pack two independently scaled 128-weight groups into each 68-byte block."""
    if tensor.ndim != 2 or tensor.shape[-1] % 256:
        raise ValueError('TQ2_1 rows must be divisible by 256')
    values = tensor.float().numpy().reshape(-1, 2, 128)
    scales = np.max(np.abs(values), axis=2)
    if not np.all(np.isfinite(values)) or np.any((values != 0) & (np.abs(values) != scales[..., None])):
        raise ValueError('TQ2_1 requires ternary values within each 128-weight group')
    stored_scales = scales.astype('<f2')
    if not np.array_equal(stored_scales.astype(np.float32), scales):
        raise ValueError('TQ2_1 scales are not exactly representable as FP16')
    codes = (np.sign(values).astype(np.int8) + 1).astype(np.uint8).reshape(-1, 2, 4, 32)
    packed = codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)
    return np.concatenate((packed.reshape(-1, 64), stored_scales.view(np.uint8).reshape(-1, 4)), axis=1).tobytes()


def write_tensor_gguf(path, tensors):
    """Stream a temporary tensor-only GGUF; audiocpp_gguf packages the sidecars."""
    def string(text):
        raw = text.encode('utf-8')
        return struct.pack('<Q', len(raw)) + raw
    entries = []
    with tempfile.TemporaryFile(dir=path.parent) as data:
        for name, shape, dtype, payload in tensors:
            if not shape or len(shape) > 4 or any(dim <= 0 for dim in shape):
                raise ValueError(f'invalid tensor shape: {name}')
            offset = data.tell()
            entries.append((name, shape, dtype, offset))
            data.write(payload)
            data.write(bytes(-data.tell() % 32))
        with path.open('wb') as target:
            target.write(struct.pack('<4sIQQ', b'GGUF', 3, len(entries), 0))
            for name, shape, dtype, offset in entries:
                target.write(string(name) + struct.pack('<I', len(shape)))
                target.write(struct.pack('<' + 'Q' * len(shape), *reversed(shape)))
                target.write(struct.pack('<IQ', dtype, offset))
            target.write(bytes(-target.tell() % 32))
            data.seek(0)
            shutil.copyfileobj(data, target, 8 * 1024 * 1024)


def is_body_projection(name):
    return re.fullmatch(r'model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight', name) is not None


def stage_language_model(source: Path, output: Path, export_type: str):
    def tensors():
        with safe_open(source, framework='pt') as handle:
            metadata = handle.metadata() or {}
            if export_type == 'q8_0':
                if metadata.get('format') != 'kittenml-ternary-1.58' or metadata.get('group_size') != '128':
                    raise ValueError('unsupported ternary checkpoint schema')
                packed = json.loads(metadata['packed'])
                for name in sorted(set(handle.keys()) - {key + suffix for key in packed for suffix in ('::codes', '::scale')}):
                    value = handle.get_tensor(name).contiguous()
                    if value.dtype != torch.bfloat16:
                        raise ValueError(f'expected BF16 dense tensor: {name}')
                    yield name, tuple(value.shape), 30, value.view(torch.uint16).numpy().tobytes()
                for name, info in sorted(packed.items()):
                    value = unpack_ternary(handle.get_tensor(name + '::codes'), handle.get_tensor(name + '::scale'), info['shape'])
                    yield name, tuple(value.shape), 8, exact_ternary_blocks(value, 8)
            elif export_type in ('q4_0', 'tq2_1'):
                body_count = 0
                for name in sorted(handle.keys()):
                    value = handle.get_tensor(name).contiguous()
                    if value.dtype != torch.bfloat16:
                        raise ValueError(f'expected BF16 source tensor: {name}')
                    if is_body_projection(name):
                        body_count += 1
                        if export_type == 'tq2_1':
                            yield name, tuple(value.shape), 44, exact_ternary_tq2_1(value)
                        else:
                            yield name, tuple(value.shape), 2, exact_ternary_blocks(value, 4)
                    else:
                        # Preserve dense tensors here; the packaging step
                        # selects the embedding storage independently.
                        yield name, tuple(value.shape), 30, value.view(torch.uint16).numpy().tobytes()
                if body_count != 196:
                    raise ValueError(f'expected 196 Kitten body projections, got {body_count}')
            else:
                raise ValueError('expected q8_0, q4_0 or tq2_1 export type')
    write_tensor_gguf(output, tensors())


def refresh_voice_projections(model: Path, voices: dict):
    """Project the released identity vectors with this checkpoint's LM weights."""
    import torch.nn.functional as functional
    index = json.loads((model / 'voices/voices.json').read_text(encoding='utf-8'))
    with safe_open(model / 'lm/model.safetensors', framework='pt') as handle:
        linear = handle.get_tensor('spk_proj.0.weight')
        bias = handle.get_tensor('spk_proj.0.bias')
        norm = handle.get_tensor('spk_proj.1.weight')
        norm_bias = handle.get_tensor('spk_proj.1.bias')
    for name, entry in index.items():
        if name not in voices:
            continue
        with np.load(model / 'voices' / entry['artifacts'], allow_pickle=False) as arrays:
            identity = torch.from_numpy(arrays['embedding'].copy()).to(torch.bfloat16)
        if identity.shape != (512,) or not torch.isfinite(identity).all():
            raise ValueError(f'invalid preset identity: {name}')
        projected = functional.linear(identity, linear, bias)
        projected = functional.layer_norm(projected, (2048,), norm, norm_bias, 1e-5)
        voices[name]['speaker'] = projected.float().tolist()
    # Keep the registered preset alias in sync with its source voice.
    if 'Bruno' in voices:
        voices['PreparedBruno'] = dict(voices['Bruno'])
