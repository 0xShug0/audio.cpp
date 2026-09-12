# Kokoro multilingual GGUF preparation and validation

The Kokoro loader accepts the original extracted model directory or a standalone
GGUF produced by `tools/prepare_kokoro_gguf.py`. Specify `--family kokoro_tts` for
GGUF loading. This is a Kokoro-specific container; arbitrary third-party Kokoro
GGUF layouts are not supported.

Release packages include all 54 source voices, configuration, vocabulary,
weights, and the generated `g2p/ja.json` / `g2p/zh.json` tables. Heavy frontend
runtime resources such as eSpeak data and UniDic are not included by default to
keep the GGUFs small. Users who want a fully bundled local package can pass
`--embed-multilingual-resources` during conversion.

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

English, Spanish, French, Hindi, Italian, and Portuguese use the shared native
eSpeak library and runtime eSpeak data. Chinese uses the bundled `g2p/zh.json`
dictionary/DAG/HMM data. Japanese uses the bundled `g2p/ja.json` plus native
MeCab and UniDic data when a fully bundled package is exported.
Python is used only for conversion and upstream comparison, never inference.

Kokoro uses the shared `engine::audio::EspeakPhonemizer` introduced in PR #502.
Its caret-tied IPA mode, punctuation restoration, language selection, and
Kokoro phoneme mapping remain in the model frontend. The shared runtime
serializes eSpeak calls with SanoTTS and Inflect v2 and reselects the voice on
each call.

Build CLI and server with `AUDIOCPP_STATIC_ESPEAK=ON` to statically link eSpeak.
Both automatically locate `espeak-ng-data.bin` beside their executable and use
the shared extraction cache. This mode does not require an eSpeak shared library.
See [shared eSpeak documentation](../../docs/espeak_phonemizer.md) for build,
data packaging, and licensing details.

With the default dynamic build, place `espeak-ng.dll` beside the executable on
Windows or install the platform library. `AUDIOCPP_ESPEAK_LIBRARY` selects an
explicit library and retains the existing model-local eSpeak data default.
`AUDIOCPP_ESPEAK_DATA` overrides the data location with a directory or a
`.bin`/`.gguf` data package in either build mode.

Japanese still requires `libmecab.dll` on Windows or an installed MeCab library;
`AUDIOCPP_MECAB_LIBRARY` selects its absolute path. The small release GGUF does
not include UniDic. Export a local GGUF with `--embed-multilingual-resources`
for the full multilingual resource bundle. Bundled data is extracted to a
temporary directory for the loaded model's lifetime and removed when its assets
are released.

## Conversion

Use a preparation environment containing numpy, gguf, and torch. Start from the
official hexgrad/Kokoro-82M source checkout containing `config.json`,
`kokoro-v1_0.pth`, and `voices/*.pt`.

```powershell
python tools/prepare_kokoro_gguf.py --source ../models_v3_test/Kokoro-82M --output-dir ../models_v3_test/Kokoro-GGUF
```

Outputs are `kokoro-82m-q8_0.gguf` and `kokoro-82m-bf16.gguf`. Q8 quantizes
eligible matrices; unsupported weight layouts remain BF16, while sensitive
small tensors remain F32 in both packages. Q8 does not mean every tensor is Q8.

To build a large fully bundled multilingual package for local redistribution or
offline deployment, also install misaki[ja,zh] (tested with 0.9.4),
espeakng-loader, unidic, and Jieba, then run `python -m unidic download` first:

```powershell
python tools/prepare_kokoro_gguf.py --source ../models_v3_test/Kokoro-82M --output-dir ../models_v3_test/Kokoro-GGUF --embed-multilingual-resources
```

## Synthesis

```powershell
.\build\windows-cpu-release\bin\audiocpp_cli.exe --task tts --family kokoro_tts --model ..\models_v3_test\Kokoro-GGUF\kokoro-82m-q8_0.gguf --backend cpu --threads 8 --language en-us --voice-id af_heart --text "Hello, this is a native Kokoro TTS test." --out kokoro-q8.wav
```

For non-Latin text on Windows, use a UTF-8 text file with
`--batch-text-file input.txt --batch-merge-audio concat` instead of `--text`.

CPU thread count should be tuned to the machine. On a Ryzen 9 7950X3D,
`--threads 16` reduced Q8 request times by 18–23% versus eight threads in two
opposite-order runs of the seven-request English benchmark. All corresponding
WAV hashes were identical. This is a runtime configuration improvement; it
does not change the model file or establish an optimal setting for other CPUs.

## Validation

`compare_shared_espeak.py` compares the pre-migration Kokoro probe against both
shared dynamic and static eSpeak modes. All 12 multilingual pronunciation cases
match exactly in both modes on Windows, including punctuation and number cases.
The script exits with failure on any mismatch or frontend error.

`compare_multilingual_g2p.py` compares `kokoro_g2p_probe` against installed
eSpeak and Misaki-compatible Japanese/Chinese frontends.
`validate_multilingual_packages.py` synthesizes each of the nine language
variants with both precisions from fully bundled packages and saves WAV files,
command logs, and `validation.json`.

Initial Windows CPU results: 12/12 pronunciation cases matched upstream exactly;
18/18 synthesis cases produced non-silent 24 kHz audio. These are smoke tests,
not perceptual-equivalence measurements or GPU coverage. Reported process times
include model loading and dictionary extraction, not just generation.

The new Japanese/Chinese frontend is not a claim of complete upstream text
normalization parity: unusual numbers, mixed scripts, and Unicode normalization
edge cases need broader coverage. Review all bundled resource and library
licenses before redistributing a package; model data does not replace the
separate native-library redistribution requirements.
