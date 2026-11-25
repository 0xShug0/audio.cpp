# F5-TTS component checks

Build only the required optional targets in the existing Debug tree:

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build/debug --parallel 12 --target f5_parity f5_cfg_parity f5_tokenizer f5_e2e
```

## Independent reference generation

Use a clean checkout of the official [F5-TTS](https://github.com/SWivid/F5-TTS)
repository at `600e5fdfbf319f84735557aca80a7aaf60d487e6`. Install its dependencies
in the model environment. `reference.py` checks the revision and imported source
location before running the official PyTorch DiT; it does not reproduce the C++
network in Python. It expects the Habibi/F5-v1 architecture used by these probes
(1024 hidden, 22 layers, 16 heads, 2731 embedding rows), an original EMA
safetensors checkpoint, and its matching `vocab.txt`.

```sh
python tests/f5_tts/reference.py \
  --upstream /path/to/F5-TTS --checkpoint /path/to/model.safetensors \
  --vocab /path/to/vocab.txt --output /path/to/external/f5-reference --device cpu
```

This saves one controlled FP32 input, conditional/final and batched-CFG outputs,
and tokenizer cases. The final-output probe reads token IDs stored as F32;
the CFG probe reads the same IDs stored as I32. Randomness is fixed at seed 42;
CUDA TF32 is disabled in the reference. A manifest records checkpoint/vocabulary
hashes, upstream revision, Torch version, precision, device and output hashes.
Use `--tokens-only` when only prompt construction changed; it does not load or
require a DiT checkpoint. A small explicit vocabulary can exercise those
formatting branches without model weights. Generated tensors remain outside the source tree.

The tokenizer fixture uses explicit expected assembled strings for an empty
reference, an ASCII-ended reference, an Arabic-ended reference and an unknown
character/dialect. Official `list_str_to_idx` produces the expected IDs.

```sh
build/debug/bin/f5_tokenizer /path/to/external/f5-reference/tokens /path/to/model.safetensors --log
build/debug/bin/f5_parity /path/to/model.safetensors /path/to/external/f5-reference --log
build/debug/bin/f5_cfg_parity /path/to/model.safetensors /path/to/external/f5-reference --log
```

The checkpoint's parent directory must contain its `vocab.txt`. Both velocity
checks retain their existing cosine gate of 0.999. CPU is the default; set
`F5_CUDA=1` and pass a device index as the last positional argument for CUDA.
Device zero is the default, not a developer's GPU one.

`f5_parity` normally validates the final output. Its older optional stage mode
(`F5_RAW_TAPS=1`) additionally requires the original stage fixtures and populated
C++ taps. Missing requested taps now fail instead of being reported as passed.
The new generator does not claim to recreate those historical stage captures;
no new production instrumentation was added for them.

## End-to-end request

All paths and the transcript matching the reference recording are explicit:

```sh
build/debug/bin/f5_e2e /path/to/model.safetensors /path/to/vocos.safetensors \
  /path/to/reference.wav /path/to/output.wav 'Transcript of the reference recording' --log
```

The existing `F5_TEXT`, `F5_DIALECT`, `F5_STEPS`, `F5_LONG` and `F5_CUDA`
controls remain available. Supply a final device argument when needed. This
probe produces audio; it is not by itself an ASR or Python waveform-parity gate.
