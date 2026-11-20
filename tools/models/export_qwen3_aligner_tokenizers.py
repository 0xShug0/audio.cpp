#!/usr/bin/env python3
"""Export the upstream Japanese/Korean tokenizers for Qwen3 Forced Aligner.

Only segmentation parameters are exported; POS tagging does not affect word
boundaries. Run in the environment containing the reference nagisa package.
"""
import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import nagisa
import numpy as np
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--korean-dictionary', type=Path, required=True,
                        help='qwen_asr/inference/assets/korean_dict_jieba.dict from official qwen-asr')
    args = parser.parse_args()
    dictionary = args.korean_dictionary.resolve(strict=True)
    tagger = nagisa.tagger
    hp = tagger._hp
    if hp['LAYERS'] != 1:
        raise ValueError('Only the official single-layer Nagisa segmenter is supported')
    model = tagger._model
    tensors = {name: np.ascontiguousarray(getattr(model, attr).as_array(), dtype=np.float32)
               for name, attr in [('unigram', 'UNI'), ('bigram', 'BI'),
                                  ('word', 'WORD'), ('chartype', 'CTYPE'),
                                  ('transitions', 'trans')]}
    hidden = hp['DIM_HIDDEN'] // 2
    # DyNet gates are IFoG; framework LSTM gates are IFGo. DyNet adds a
    # separate +1 forget bias at inference; incorporate it into the export.
    order = np.concatenate([np.arange(i*hidden, (i+1)*hidden) for i in (0, 1, 3, 2)])
    for direction, builder in zip(('forward', 'reverse'), model.ws_model.builder_layers[0]):
        for name, parameter in zip(('weight_ih', 'weight_hh', 'bias_ih'), builder.get_parameters()[0]):
            value = parameter.as_array()[order].copy()
            if name == 'bias_ih':
                value[hidden:2*hidden] += 1.0
            tensors[f'{direction}.{name}'] = np.ascontiguousarray(value, dtype=np.float32)
        tensors[f'{direction}.bias_hh'] = np.zeros(hidden*4, dtype=np.float32)
    tensors['projection.weight'] = np.ascontiguousarray(model.w_ws.as_array(), dtype=np.float32)
    tensors['projection.bias'] = np.ascontiguousarray(model.b_ws.as_array(), dtype=np.float32)
    config = {'nagisa_version': importlib.metadata.version('nagisa'),
              'window_size': hp['WINDOW_SIZE'], 'hidden_size': hidden,
              'unigram': tagger._uni2id, 'bigram': tagger._bi2id, 'word': tagger._word2id}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_dir/'nagisa.safetensors'))
    (args.output_dir/'nagisa.json').write_text(json.dumps(config, ensure_ascii=False)+'\n', encoding='utf-8')
    shutil.copyfile(dictionary, args.output_dir/'korean_dict_jieba.dict')
    print(f'Exported Nagisa {config["nagisa_version"]} segmentation assets to {args.output_dir}')


if __name__ == '__main__':
    main()
