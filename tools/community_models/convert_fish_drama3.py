#!/usr/bin/env python3
"""Convert the official Drama 3 FP8 checkpoint to audio.cpp's Fish variant.

Uses bounded staging files and the drama3_quantize executable from a build with
ENGINE_BUILD_MODEL_TESTS=ON. Expert FP8 block scales are multiplied, not divided.
The speech head contains exactly EOS + the semantic band; no text output rows
are needed for this TTS-only runtime. The original checkpoint is left intact.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open


class Converter:
    def __init__(self, args):
        self.args = args
        self.source = args.source.resolve()
        self.stage = args.staging.resolve()
        self.stage.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((self.source / 'config.json').read_text())
        if self.config['model_type'] != 'fish_omni':
            raise ValueError('not a Drama 3 checkpoint')
        self.index = json.loads((self.source / 'model.safetensors.index.json').read_text())['weight_map']
        identity = {'precision':args.precision, 'config_sha256':hashlib.sha256((self.source/'config.json').read_bytes()).hexdigest(),
                    'index_sha256':hashlib.sha256((self.source/'model.safetensors.index.json').read_bytes()).hexdigest(),
                    'files':{name:[(self.source/name).stat().st_size,(self.source/name).stat().st_mtime_ns]
                             for name in sorted(set(self.index.values())|{'codec.pth'})}}
        manifest = self.stage / '.source-manifest'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('staging belongs to a different source checkpoint or precision; choose a new staging directory')
        manifest.write_text(json.dumps(identity))
        self.identity = identity
        self.handles = {}
        self.entries = []
        self.consumed = set()
        self.norms = 0
        torch.set_num_threads(4)

    def tensor(self, name):
        self.consumed.add(name)
        filename = self.index[name]
        if filename not in self.handles:
            self.handles[filename] = safe_open(self.source / filename, framework='pt', device='cpu')
        return self.handles[filename].get_tensor(name)

    def dequant_expert(self, name):
        w = self.tensor(name)
        scales = self.tensor(name.replace('.weight', '.weight_scale_inv')).float()
        rows, cols = w.shape
        if rows % 128 or cols % 128 or tuple(scales.shape) != (rows // 128, cols // 128):
            raise ValueError(f'unsupported FP8 block shape: {name}')
        return (w.reshape(rows // 128, 128, cols // 128, 128).float() *
                scales[:, None, :, None]).reshape(rows, cols).numpy()

    def emit(self, name, shape, kind, chunks):
        shape = tuple(int(v) for v in shape)
        if len(name.encode()) >= 128:
            raise ValueError(f'GGML tensor name too long: {name}')
        token = hashlib.sha256(name.encode()).hexdigest()[:20]
        raw = self.stage / (token + '.input.f32')
        payload = self.stage / (token + '.' + kind)
        descriptor = self.stage / (token + '.json')
        description = {'name': name, 'shape': shape, 'kind': kind}
        # Staging is reusable only when the descriptor matches this conversion.
        if descriptor.exists() and payload.exists() and json.loads(descriptor.read_text()) == json.loads(json.dumps(description)):
            for _ in chunks: pass  # account for consumed source keys
        else:
            count = 0
            with raw.open('wb') as output:
                for chunk in chunks:
                    chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                    if not np.isfinite(chunk).all():
                        raise ValueError(f'non-finite values: {name}')
                    count += chunk.size
                    chunk.tofile(output)
            if count != math.prod(shape):
                raise ValueError(f'tensor element count mismatch: {name}')
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            subprocess.run([str(self.args.quantizer.resolve()), kind, str(shape[-1]),
                            str(math.prod(shape[:-1])), str(raw), str(payload)],
                           check=True, creationflags=flags)
            raw.unlink()
            descriptor.write_text(json.dumps(description))
        dtype = getattr(gguf.GGMLQuantizationType, kind.upper())
        block, size = gguf.GGML_QUANT_SIZES[dtype]
        if shape[-1] % block:
            raise ValueError(f'invalid quantized row: {name}')
        byte_shape = (*shape[:-1], shape[-1] // block * size)
        expected = math.prod(byte_shape)
        if payload.stat().st_size != expected:
            raise ValueError(f'truncated quantized payload: {name}')
        self.entries.append((name, shape, byte_shape, dtype, payload, expected))

    def ordinary(self, name, tensor, prefix='model_weights/'):
        shape = tuple(tensor.shape)
        # Norms/routers/short row projections and codec convolutions remain F32.
        kind = 'f32'
        if len(shape) == 2 and shape[-1] % 32 == 0:
            if prefix == 'codec_weights/':
                kind = 'q8_0' if shape[-1] >= 32 else 'f32'
            elif 'embedding' in name or name.endswith('embed_tokens.weight') or '.heads.' in name or 'candidate_lm_head' in name:
                kind = 'bf16'
            elif '.mlp.gate.weight' not in name and 'shared_expert_gate' not in name:
                kind = self.args.precision if shape[-1] % 256 == 0 or self.args.precision != 'q4_k' else 'q8_0'
        if len(shape) == 1:
            shape = (shape[0],)
        self.emit(prefix + name, shape, kind,
                  (tensor[start:start+128].float().numpy() for start in range(0, shape[0], 128)))

    def experts(self):
        t = self.config['text_config']
        for layer in range(t['num_hidden_layers']):
            prefix = f'model.language_model.layers.{layer}.mlp.experts'
            def gu():
                for expert in range(t['num_experts']):
                    yield self.dequant_expert(f'{prefix}.{expert}.gate_proj.weight')
                    yield self.dequant_expert(f'{prefix}.{expert}.up_proj.weight')
            self.emit('model_weights/' + prefix + '.gate_up.weight',
                      (t['num_experts'], 2*t['moe_intermediate_size'], t['hidden_size']), self.args.precision, gu())
            self.emit('model_weights/' + prefix + '.down.weight',
                      (t['num_experts'], t['hidden_size'], t['moe_intermediate_size']), self.args.precision,
                      (self.dequant_expert(f'{prefix}.{expert}.down_proj.weight') for expert in range(t['num_experts'])))
            print(f'Experts {layer+1}/{t["num_hidden_layers"]}', flush=True)

    def codec(self):
        raw = torch.load(self.source / 'codec.pth', map_location='cpu', weights_only=True, mmap=True)
        for name, tensor in sorted(raw.items()):
            if name.endswith(('causal_mask', 'freqs_cis', '.parametrizations.weight.original0', '.weight_g')):
                continue
            if name.endswith('.parametrizations.weight.original1') or name.endswith('.weight_v'):
                suffix = '.parametrizations.weight.original1' if '.parametrizations.' in name else '.weight_v'
                base = name[:-len(suffix)]
                g = raw[base + ('.parametrizations.weight.original0' if '.parametrizations.' in name else '.weight_g')]
                axes = tuple(range(1, tensor.ndim))
                norm = tensor.double().square().sum(dim=axes, keepdim=True).sqrt()
                if not torch.all(norm > 0): raise ValueError(f'zero codec weight norm: {name}')
                tensor = (g.double() * tensor.double() / norm).float()
                name = base + '.weight'
                self.norms += 1
            if name.endswith('.wqkv.weight'):
                parts = tensor.chunk(3, dim=0)
                if len(parts) != 3 or any(p.shape[0] != tensor.shape[1] for p in parts):
                    raise ValueError(f'unsupported codec QKV: {name}')
                for letter, part in zip('qkv', parts):
                    self.ordinary(name[:-len('.wqkv.weight')] + f'.{letter}_proj.weight', part, 'codec_weights/')
            else:
                self.ordinary(name, tensor, 'codec_weights/')
        print('Codec: folded', self.norms, 'weight norms', flush=True)

    def run(self):
        if self.args.pack_only:
            codec = torch.load(self.source / 'codec.pth', map_location='cpu', weights_only=True, mmap=True)
            self.norms = sum(name.endswith(('.parametrizations.weight.original1', '.weight_v')) for name in codec)
            del codec
            for descriptor in sorted(self.stage.glob('*.json')):
                item = json.loads(descriptor.read_text())
                if item['kind'] not in [self.args.precision, 'bf16', 'f32', 'q8_0']:
                    raise ValueError('staging precision mismatch')
                shape = tuple(item['shape'])
                dtype = getattr(gguf.GGMLQuantizationType, item['kind'].upper())
                block, size = gguf.GGML_QUANT_SIZES[dtype]
                byte_shape = (*shape[:-1], shape[-1] // block * size)
                path = descriptor.with_suffix('.' + item['kind'])
                expected = math.prod(byte_shape)
                if path.stat().st_size != expected: raise ValueError('staging payload size mismatch')
                self.entries.append((item['name'], shape, byte_shape, dtype, path, expected))
            if not self.entries: raise ValueError('empty staging directory')
            self.write()
            return
        self.experts()
        for name in sorted(self.index):
            if re.search(r'\.mlp\.experts\.\d+\.', name): continue
            if name == 'lm_head.weight':
                source = self.tensor(name)
                indices = [self.config['eos_token_id'], *range(self.config['semantic_start_token_id'], self.config['semantic_end_token_id']+1)]
                self.ordinary('drama3.candidate_lm_head.weight', source[indices])
            else:
                self.ordinary(name, self.tensor(name))
        if self.consumed != self.index.keys():
            raise ValueError(f'unconsumed checkpoint tensors: {self.index.keys()-self.consumed}')
        self.codec()
        self.write()

    def write(self):
        out = self.args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix + '.partial')
        writer = gguf.GGUFWriter(str(partial), 'audiocpp')
        writer.add_string('general.name', 'Fish Audio Drama 3')
        writer.add_string('general.license', 'fish-audio-community-license')
        writer.add_string('audiocpp.tensor_name_format', 'native')
        writer.add_string('audiocpp.weight_type', self.args.precision)
        writer.add_array('audiocpp.tensor_sources.names', ['model_weights', 'codec_weights'])
        writer.add_array('audiocpp.tensor_sources.paths', ['model.safetensors.index.json', 'codec.pth'])
        spec_path = Path(__file__).resolve().parents[2] / 'model_specs' / 'fish_audio.json'
        spec = json.loads(spec_path.read_text())
        spec['schema_version'] = 1
        spec['dependencies'] = []
        spec.update(display_name='Fish Audio Drama 3', status='experimental', description='Native Drama 3 MoE text-to-speech variant.', packages=[], languages=['auto','en'])
        spec['sources'] = [s for s in spec['sources'] if s['format'] == 'gguf']
        spec.pop('download', None)
        spec.pop('package_defaults', None)
        spec['ui'].pop('recommended_package', None)
        spec['ui']['docs'] = ['docs/models/fish_drama3.md']
        writer.add_uint32('audiocpp.model_spec.version', 1)
        writer.add_string('audiocpp.model_spec.family', 'fish_audio')
        writer.add_string('audiocpp.model_spec.json', json.dumps(spec, ensure_ascii=False))
        self.config['audiocpp_context_size'] = self.args.context_size
        files = {name: (self.source / name).read_text(encoding='utf-8') for name in
                 ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'LICENSE', 'README.md', 'chat_template.jinja']}
        files['config.json'] = json.dumps(self.config)
        files['conversion.json'] = json.dumps({'source': 'fishaudio/drama-3', 'source_revision':self.args.source_revision,
            'source_identity':self.identity, 'fp8_block_size':128, 'expert_precision':self.args.precision,
            'candidate_head':'EOS + semantic band', 'codec_weight_norms':self.norms})
        writer.add_array('audiocpp.embedded_files.names', list(files))
        writer.add_array('audiocpp.embedded_files.contents', list(files.values()))
        writer.add_array('audiocpp.tensor_names', [e[0] for e in self.entries])
        writer.add_key_value('audiocpp.tensor_ranks', [len(e[1]) for e in self.entries], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
        writer.add_key_value('audiocpp.tensor_shapes', [v for e in self.entries for v in e[1]], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT64)
        for index, (name, shape, byte_shape, dtype, path, size) in enumerate(self.entries):
            # GGML names are limited to 63 bytes. The full source name stays in
            # audiocpp.tensor_names; the framework resolves that parallel map.
            writer.add_tensor_info(f'tensor_{index}', byte_shape, np.dtype(np.uint8), size, raw_dtype=dtype)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
        for name, shape, byte_shape, dtype, path, size in self.entries:
            data = np.memmap(path, mode='r', dtype=np.uint8, shape=byte_shape)
            writer.write_tensor_data(data)
            del data
        writer.close()
        os.replace(partial, out)
        print('Wrote', out, round(out.stat().st_size/2**30, 3), 'GiB;', len(self.entries), 'tensors', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-revision', help='Pinned source revision for the embedded conversion record')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--quantizer', type=Path, required=True)
    parser.add_argument('--precision', choices=['q4_k', 'q8_0', 'bf16'], default='q4_k')
    parser.add_argument('--context-size', type=int, default=4096)
    parser.add_argument('--pack-only', action='store_true', help='repack already converted staging payloads')
    args = parser.parse_args()
    if not 128 <= args.context_size <= 8192: parser.error('context size must be 128..8192')
    Converter(args).run()


if __name__ == '__main__': main()
