"""Create standalone Kokoro GGUFs, preserving every voice and pronunciation resource.

Conversion dependencies: numpy, safetensors, gguf, misaki[ja,zh], espeakng-loader,
unidic (run `python -m unidic download` once). Python is not used for inference.
"""
import argparse
import importlib.metadata
import json
import struct
import shutil
from pathlib import Path
import numpy as np
import gguf
from safetensors import safe_open

class ResourceWriter(gguf.GGUFWriter):
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        # Avoid a Python function call per byte for large embedded dictionaries.
        if vtype == gguf.GGUFValueType.ARRAY and isinstance(val, bytes):
            return (struct.pack('<I', int(vtype)) if add_vtype else b'') + struct.pack('<IQ', 0, len(val)) + val
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def prepare(source, root):
    import espeakng_loader
    import unidic
    import jieba
    from pypinyin import pinyin_dict, phrases_dict, Style, lazy_pinyin
    from misaki.zh import ZHG2P
    from misaki.cutlet import HEPBURN, JA_WORDS
    if not (Path(unidic.DICDIR) / 'sys.dic').is_file():
        raise RuntimeError('Download the Japanese dictionary with: python -m unidic download')
    shutil.copytree(source, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.cache', '.gitattributes', '*.gguf'))
    shutil.copytree(espeakng_loader.get_data_path(), root / 'espeak-ng-data', dirs_exist_ok=True)
    dic = root / 'unidic'
    dic.mkdir(exist_ok=True)
    for name in ['sys.dic', 'unk.dic', 'char.bin', 'matrix.bin', 'dicrc']:
        shutil.copy2(Path(unidic.DICDIR) / name, dic / name)
    dump(root / 'g2p' / 'ja.json', {'kana': HEPBURN, 'words': sorted(JA_WORDS)})
    # Export the complete pypinyin dictionaries and Jieba DAG/HMM data, not sample phrases.
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
    dump(root / 'g2p' / 'zh.json', {'chars': chars, 'phrases': phrases, 'ipa': ipa,
        'frequency': {k: v for k, v in jieba.dt.FREQ.items() if v > 0}, 'total': jieba.dt.total,
        'start': start_P, 'transition': trans_P, 'emission': emit_P})
    # Keep redistribution notices with the resources.
    notices = root / 'licenses'
    notices.mkdir(exist_ok=True)
    shutil.copytree(Path(unidic.DICDIR) / 'licenses', notices / 'unidic-dictionary', dirs_exist_ok=True)
    shutil.copy2(Path(unidic.DICDIR) / 'README', notices / 'unidic-dictionary-README')
    for dist_name in ['misaki', 'unidic', 'espeakng-loader', 'jieba', 'pypinyin', 'mecab-python3']:
        dist = importlib.metadata.distribution(dist_name)
        for file in dist.files or []:
            if any(x in file.name.lower() for x in ['license', 'copying', 'notice']):
                src = Path(dist.locate_file(file))
                if src.is_file(): shutil.copy2(src, notices / (dist_name + '-' + file.name))
    dump(root / 'g2p' / 'versions.json', {n: importlib.metadata.version(n)
        for n in ['misaki', 'unidic', 'espeakng-loader', 'jieba', 'pypinyin']})


def convert(root, output, precision, overwrite=False):
    if output.exists() and not overwrite: raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.gguf.partial')
    writer = ResourceWriter(str(temporary), 'kokoro_tts')
    writer.add_name('Kokoro v1.0 multilingual ' + precision)
    writer.add_array('kokoro.languages', ['en-us', 'en-gb', 'es', 'fr-fr', 'hi', 'it', 'ja', 'pt-br', 'zh'])
    writer.add_array('kokoro.voices', sorted(json.loads((root / 'voices.json').read_text())))
    names, offsets, data = [], [0], bytearray()
    for file in sorted(root.rglob('*')):
        if not file.is_file() or file.suffix in ['.safetensors', '.gguf']: continue
        if 'misaki_en' in file.relative_to(root).parts: continue
        names.append(file.relative_to(root).as_posix())
        data.extend(file.read_bytes())
        offsets.append(len(data))
    writer.add_array('audiocpp.embedded_files.names', names)
    writer.add_key_value('audiocpp.embedded_files.offsets', offsets, gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.UINT64)
    writer.add_key_value('audiocpp.embedded_files.data', bytes(data), gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.UINT8)
    counts = {}
    with safe_open(root / 'kokoro-v1_0.safetensors', framework='numpy') as model:
        writer.add_array('kokoro.tensor_names', list(model.keys()))
        for index, name in enumerate(model.keys()):
            array = model.get_tensor(name).astype(np.float32)
            writer.add_key_value('kokoro.tensor_shape.' + name, list(array.shape),
                                 gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT64)
            # Retain scalar/vector/norm and Snake parameters in F32. Weight-norm
            # convolution tensors use BF16 on disk and are reconstructed in F32.
            kind = gguf.GGMLQuantizationType.F32
            if array.ndim >= 2 and all(d > 1 for d in array.shape):
                kind = gguf.GGMLQuantizationType.BF16
                if precision == 'q8_0' and array.ndim == 2 and array.shape[-1] % 32 == 0:
                    kind = gguf.GGMLQuantizationType.Q8_0
            encoded = gguf.quants.quantize(array, kind)
            writer.add_tensor('kokoro.' + str(index), encoded, raw_dtype=kind)
            counts[kind.name] = counts.get(kind.name, 0) + 1
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
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--skip-prepare', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--type', choices=['q8_0', 'bf16', 'both'], default='both')
    args = p.parse_args()
    if not args.skip_prepare: prepare(args.source, args.resources)
    for precision in (['q8_0', 'bf16'] if args.type == 'both' else [args.type]):
        convert(args.resources, args.output_dir / ('kokoro-82m-' + precision + '.gguf'), precision, args.overwrite)
