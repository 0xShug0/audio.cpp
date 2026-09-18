"""Create Kokoro GGUFs from the official hexgrad/Kokoro-82M source checkout.

The default GGUF contains model weights, config, vocab, and voice packs only.
Use --embed-multilingual-resources to build a larger fully bundled package.
Python is not used for inference.
"""
import argparse
import importlib.metadata
import json
import struct
from pathlib import Path
import numpy as np
import gguf

class ResourceWriter(gguf.GGUFWriter):
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        # Avoid a Python function call per byte for large embedded dictionaries.
        if vtype == gguf.GGUFValueType.ARRAY and isinstance(val, bytes):
            return (struct.pack('<I', int(vtype)) if add_vtype else b'') + struct.pack('<IQ', 0, len(val)) + val
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def flatten_torch_state(prefix, value, out):
    import torch
    if isinstance(value, dict):
        for key, item in value.items():
            key = str(key)
            if key == 'module' and prefix:
                name = prefix
            else:
                if prefix and key.startswith('module.'):
                    key = key[len('module.'):]
                name = key if not prefix else prefix + '.' + key
            flatten_torch_state(name, item, out)
        return
    if not torch.is_tensor(value):
        raise RuntimeError(f'unsupported Kokoro checkpoint value at {prefix}: {type(value)!r}')
    out[prefix] = value.detach().cpu().contiguous()


def load_official_weights(source):
    checkpoint = source / 'kokoro-v1_0.pth'
    if not checkpoint.is_file():
        raise RuntimeError('official Kokoro source must contain kokoro-v1_0.pth')
    import torch
    state = torch.load(checkpoint, map_location='cpu')
    flat = {}
    flatten_torch_state('', state, flat)
    return flat


def make_vocab_tsv(config):
    vocab = config.get('vocab')
    if not isinstance(vocab, dict):
        raise RuntimeError('Kokoro config.json is missing vocab object')
    lines = [f'{symbol}\t{int(index)}\n' for symbol, index in sorted(vocab.items(), key=lambda item: int(item[1]))]
    return ''.join(lines).encode('utf-8')


def add_file(files, name, data):
    files[name] = data if isinstance(data, bytes) else data.encode('utf-8')


def add_tree(files, root, prefix):
    for path in sorted(root.rglob('*')):
        if path.is_file():
            files[prefix + '/' + path.relative_to(root).as_posix()] = path.read_bytes()


def load_official_voices(source):
    voice_files = sorted((source / 'voices').glob('*.pt'))
    if not voice_files:
        raise RuntimeError('official Kokoro source must contain voices/*.pt')
    import torch
    voices = {}
    files = {}
    for path in voice_files:
        tensor = torch.load(path, map_location='cpu')
        if not torch.is_tensor(tensor):
            raise RuntimeError(f'Kokoro voice file is not a tensor: {path}')
        array = tensor.detach().cpu().to(torch.float32).numpy()
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        if array.ndim != 2:
            raise RuntimeError(f'Kokoro voice tensor must be rank 2 or [rows,1,cols]: {path}')
        array = np.ascontiguousarray(array, dtype=np.float32)
        out_name = path.stem + '.bin'
        files['voices/' + out_name] = array.tobytes()
        voices[path.stem] = {'rows': int(array.shape[0]), 'cols': int(array.shape[1]), 'path': out_name}
    return voices, files


def load_g2p_tables():
    import jieba
    from pypinyin import pinyin_dict, phrases_dict, Style, lazy_pinyin
    from misaki.zh import ZHG2P
    from misaki.cutlet import HEPBURN, JA_WORDS
    files = {}
    add_file(files, 'g2p/ja.json', json.dumps({'kana': HEPBURN, 'words': sorted(JA_WORDS)},
                                              ensure_ascii=False, separators=(',', ':')))
    from jieba.finalseg import start_P, trans_P, emit_P
    jieba.initialize()
    syllables = set()
    chars = {}
    for cp in pinyin_dict.pinyin_dict:
        value = lazy_pinyin(chr(cp), style=Style.TONE3, neutral_tone_with_five=True)[0]
        chars[chr(cp)] = value
        syllables.add(value)
    phrases = {}
    for word in phrases_dict.phrases_dict:
        value = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
        phrases[word] = value
        syllables.update(value)
    ipa = {}
    for value in sorted(syllables):
        try:
            ipa[value] = ZHG2P.py2ipa(value).replace('\u032f', '')
        except (ValueError, IndexError, KeyError, AssertionError):
            continue
    add_file(files, 'g2p/zh.json', json.dumps({'chars': chars, 'phrases': phrases, 'ipa': ipa,
        'frequency': {k: v for k, v in jieba.dt.FREQ.items() if v > 0}, 'total': jieba.dt.total,
        'start': start_P, 'transition': trans_P, 'emission': emit_P},
        ensure_ascii=False, separators=(',', ':')))
    return files


def load_multilingual_resources():
    import espeakng_loader
    import unidic
    if not (Path(unidic.DICDIR) / 'sys.dic').is_file():
        raise RuntimeError('Download the Japanese dictionary with: python -m unidic download')
    files = {}
    add_tree(files, Path(espeakng_loader.get_data_path()), 'espeak-ng-data')
    for name in ['sys.dic', 'unk.dic', 'char.bin', 'matrix.bin', 'dicrc']:
        files['unidic/' + name] = (Path(unidic.DICDIR) / name).read_bytes()
    files.update(load_g2p_tables())
    notices = {}
    add_tree(notices, Path(unidic.DICDIR) / 'licenses', 'licenses/unidic-dictionary')
    readme = Path(unidic.DICDIR) / 'README'
    if readme.is_file():
        notices['licenses/unidic-dictionary-README'] = readme.read_bytes()
    for dist_name in ['misaki', 'unidic', 'espeakng-loader', 'jieba', 'pypinyin']:
        dist = importlib.metadata.distribution(dist_name)
        for file in dist.files or []:
            if any(x in file.name.lower() for x in ['license', 'copying', 'notice']):
                src = Path(dist.locate_file(file))
                if src.is_file():
                    notices['licenses/' + dist_name + '-' + file.name] = src.read_bytes()
    files.update(notices)
    add_file(files, 'g2p/versions.json', json.dumps({n: importlib.metadata.version(n)
        for n in ['misaki', 'unidic', 'espeakng-loader', 'jieba', 'pypinyin']},
        ensure_ascii=False, separators=(',', ':')))
    return files


def load_official_source(source, embed_multilingual_resources):
    config_path = source / 'config.json'
    if not config_path.is_file():
        raise RuntimeError('official Kokoro source must contain config.json')
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode('utf-8'))
    voices, voice_files = load_official_voices(source)
    files = {
        'config.json': config_bytes,
        'voices.json': json.dumps(voices, ensure_ascii=False, separators=(',', ':')).encode('utf-8'),
        'vocab.tsv': make_vocab_tsv(config),
    }
    files.update(voice_files)
    files.update(load_g2p_tables())
    if embed_multilingual_resources:
        files.update(load_multilingual_resources())
    return config, voices, files, load_official_weights(source)


def convert(source, output, precision, overwrite=False, model_spec=None, embed_multilingual_resources=False):
    if output.exists() and not overwrite: raise FileExistsError(output)
    _, voices, files, weights = load_official_source(source, embed_multilingual_resources)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.gguf.partial')
    writer = ResourceWriter(str(temporary), 'kokoro_tts')
    writer.add_name('Kokoro v1.0 multilingual ' + precision)
    writer.add_array('kokoro.languages', ['en-us', 'en-gb', 'es', 'fr-fr', 'hi', 'it', 'ja', 'pt-br', 'zh'])
    writer.add_array('kokoro.voices', sorted(voices))
    if model_spec is not None:
        spec_text = model_spec.read_text(encoding='utf-8')
        spec_json = json.loads(spec_text)
        if spec_json.get('family') != 'kokoro_tts' or spec_json.get('schema_version') != 1:
            raise RuntimeError(f'not a Kokoro schema-v1 model spec: {model_spec}')
        writer.add_uint32('audiocpp.model_spec.version', 1)
        writer.add_string('audiocpp.model_spec.family', 'kokoro_tts')
        writer.add_string('audiocpp.model_spec.json', json.dumps(spec_json, ensure_ascii=False, separators=(',', ':')))
    names, offsets, data = [], [0], bytearray()
    for name in sorted(files):
        names.append(name)
        data.extend(files[name])
        offsets.append(len(data))
    writer.add_array('audiocpp.embedded_files.names', names)
    writer.add_key_value('audiocpp.embedded_files.offsets', offsets, gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.UINT64)
    writer.add_key_value('audiocpp.embedded_files.data', bytes(data), gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.UINT8)
    counts = {}
    source_names = sorted(weights)
    writer.add_array('kokoro.tensor_names', source_names)
    writer.add_array('audiocpp.tensor_names', ['kokoro/' + name for name in source_names])
    ranks, shapes = [], []
    for index, name in enumerate(source_names):
        array = weights[name].float().numpy()
        ranks.append(len(array.shape))
        shapes.extend(int(dim) for dim in array.shape)
        writer.add_key_value('kokoro.tensor_shape.' + name, list(array.shape),
                             gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT64)
        # Retain scalar/vector/norm and Snake parameters in F32. Weight-norm
        # convolution tensors use BF16 on disk and are reconstructed in F32.
        kind = gguf.GGMLQuantizationType.F32
        if array.ndim >= 2 and all(d > 1 for d in array.shape):
            kind = gguf.GGMLQuantizationType.BF16
            if precision == 'q8_0' and array.ndim == 2 and array.shape[-1] % 32 == 0:
                kind = gguf.GGMLQuantizationType.Q8_0
        encoded = gguf.quants.quantize(np.ascontiguousarray(array, dtype=np.float32), kind)
        writer.add_tensor('kokoro.' + str(index), encoded, raw_dtype=kind)
        counts[kind.name] = counts.get(kind.name, 0) + 1
    writer.add_key_value('audiocpp.tensor_ranks', ranks, gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.INT32)
    writer.add_key_value('audiocpp.tensor_shapes', shapes, gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.INT64)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    temporary.replace(output)
    print(json.dumps({'path': str(output), 'bytes': output.stat().st_size,
                      'tensors': counts, 'resources': len(names), 'resource_bytes': len(data)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-spec', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'model_specs' / 'kokoro_tts.json')
    p.add_argument('--embed-multilingual-resources', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--type', choices=['q8_0', 'bf16', 'both'], default='both')
    args = p.parse_args()
    for precision in (['q8_0', 'bf16'] if args.type == 'both' else [args.type]):
        convert(args.source, args.output_dir / ('kokoro-82m-' + precision + '.gguf'),
                precision, args.overwrite, args.model_spec, args.embed_multilingual_resources)
