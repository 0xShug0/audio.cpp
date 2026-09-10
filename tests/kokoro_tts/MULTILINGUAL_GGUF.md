# Kokoro multilingual GGUF preparation and validation

The Kokoro loader accepts the original extracted model directory or a standalone
GGUF produced by `tools/prepare_kokoro_gguf.py`. Specify `--family kokoro_tts` for
GGUF loading. This is a Kokoro-specific container; arbitrary third-party Kokoro
GGUF layouts are not supported.

Each package includes all 54 source voices, configuration, English pronunciation
resources, eSpeak data, the full UniDic dictionary, and Chinese pronunciation and
segmentation dictionaries. Executable libraries are not embedded in the model.

| Language | CLI language | Example voice |
| --- | --- | --- |
| American English | en-us | af_heart |
| British English | en-gb | bf_emma |
| Spanish | es | ef_dora |
| French | fr-fr | ff_siwis |
| Hindi | hi | hf_alpha |
| Italian | it | if_sara |
| Japanese | ja | jf_alpha |
| Brazilian Portuguese | pt-br | pf_dora |
| Mandarin Chinese | zh | zf_xiaobei |

## Runtime dependencies

English retains the existing native frontend. Spanish, French, Hindi, Italian,
and Portuguese use the native eSpeak library. Japanese uses native MeCab with
embedded UniDic data. Chinese uses native dictionary/DAG/HMM processing.
Python is used only for conversion and upstream comparison, never inference.

On Windows, place `espeak-ng.dll` and `libmecab.dll` beside the executable, or set
`AUDIOCPP_ESPEAK_LIBRARY` / `AUDIOCPP_MECAB_LIBRARY` to absolute library paths.
On other platforms these variables can select installed native libraries.
Embedded data is extracted to a temporary directory for the loaded model's
lifetime and removed when its assets are released.

## Conversion

Use a preparation environment containing numpy, safetensors, gguf (tested with
0.19), misaki[ja,zh] (tested with 0.9.4), espeakng-loader, unidic, and Jieba.
Run `python -m unidic download` first. Start from the original extracted Kokoro
directory containing weights, config, all voices, and English resources.

```powershell
python tools/prepare_kokoro_gguf.py --source ../models_v3_test/kokoro-82m-v1_0-ggml --resources ../models_v3_test/Kokoro-multilingual-resources --output-dir ../models_v3_test/Kokoro-GGUF
```

Outputs are `kokoro-v1.0-q8_0.gguf` and `kokoro-v1.0-bf16.gguf`. Q8 quantizes
eligible matrices; unsupported weight layouts remain BF16, while sensitive
small tensors remain F32 in both packages. Q8 does not mean every tensor is Q8.
The full embedded dictionaries dominate file size: approximately 943 MB for Q8
and 965 MB for BF16. Language resources are identical in both.

## Synthesis

```powershell
.\build\windows-cpu-release\bin\audiocpp_cli.exe --task tts --family kokoro_tts --model ..\models_v3_test\Kokoro-GGUF\kokoro-v1.0-q8_0.gguf --backend cpu --threads 8 --language en-us --voice-id af_heart --text "Hello, this is a native Kokoro TTS test." --out kokoro-q8.wav
```

For non-Latin text on Windows, use a UTF-8 text file with
`--batch-text-file input.txt --batch-merge-audio concat` instead of `--text`.

CPU thread count should be tuned to the machine. On a Ryzen 9 7950X3D,
`--threads 16` reduced Q8 request times by 18–23% versus eight threads in two
opposite-order runs of the seven-request English benchmark. All corresponding
WAV hashes were identical. This is a runtime configuration improvement; it
does not change the model file or establish an optimal setting for other CPUs.

## Validation

`compare_multilingual_g2p.py` compares `kokoro_g2p_probe` against installed Misaki.
`validate_multilingual_packages.py` synthesizes each of the nine language variants
with both precisions and saves WAV files, command logs, and `validation.json`.

Initial Windows CPU results: 12/12 pronunciation cases matched upstream exactly;
18/18 synthesis cases produced non-silent 24 kHz audio. These are smoke tests,
not perceptual-equivalence measurements or GPU coverage. Reported process times
include model loading and dictionary extraction, not just generation.

The new Japanese/Chinese frontend is not a claim of complete upstream text
normalization parity: unusual numbers, mixed scripts, and Unicode normalization
edge cases need broader coverage. Review all bundled resource and library
licenses before redistributing a package; model data does not replace the
separate native-library redistribution requirements.
