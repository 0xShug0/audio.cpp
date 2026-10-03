# Kitten TTS 2

Powered by Stellon Labs.

`kitten_tts2` supports offline multilingual TTS, preset voices, reference-audio cloning,
and chunked long-form speech. The complete inference path is native C++/GGML:
Qwen3, S3 meanflow, HiFT, S3 tokenizer, CAMPPlus, and XVectorSincNet.
Python, LibTorch, TorchScript, and ONNX Runtime are not runtime dependencies.

The family is experimental; a compatible community GGUF is available below.
CPU and NVIDIA CUDA are validated. Other GPU backends, streaming, automatic
reference transcription, the upstream text normalizer, and the smaller
`student_w4`/`student_w8` decoders are not implemented or validated by this port.
The full default decoder is used.

## Build

Shared S3 code is included through the `chatterbox` dependency. No Kitten-specific
build flag or Torch package is needed.

```sh
cmake -S . -B build-kitten2 -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=kitten_tts2 -DENGINE_ENABLE_CUDA=OFF
cmake --build build-kitten2 --config Release \
  --target audiocpp_cli audiocpp_server audiocpp_gguf audiocpp_kitten_tts2_prepare_voices --parallel 8
```

Visual Studio binaries are in `build-kitten2/bin/Release/`; single-configuration
builds use `build-kitten2/bin/`.

For NVIDIA, configure with `-DENGINE_ENABLE_CUDA=ON` and select
`--backend cuda` at runtime. The Kitten networks use the selected GGML backend;
reference resampling, sampling, and some small host computations remain on CPU.
Windows Release builds and preset/cloned speech were tested on a 16 GB RTX 4060 Ti
with CUDA Toolkit 13.3 and driver 610.88. Set `-DCMAKE_CUDA_ARCHITECTURES=89` when
building specifically for this card; choose the architecture for other cards.
CUDA 13.3 places its runtime DLLs in `CUDA_PATH/bin/x64`; ensure that directory
is on `PATH` when launching the executable on Windows.

The speaker's temporal max pooling uses a CUDA-supported 3x1 window. F32 speaker,
S3, and HiFT matrix operations explicitly preserve full precision: TF32 error in
the first speaker convolution otherwise changes the identity embedding. The CUDA
backend honors this precision request automatically; no environment override is
needed. See the validation record for measured RTF and component errors.

The WebUI shows **7 GB estimated VRAM** for the native Q8 package. A local CUDA
server run with a 238-character passage peaked approximately 4.7 GiB above the
desktop baseline for preset speech and 5.9 GiB for cloning. The estimate includes
headroom; it is guidance, not a tested minimum for every input or weight type.

## Download and preset speech

The ready-to-run [community package](https://huggingface.co/dignome/kitten_tts2)
contains the full native decoder, speaker encoders, and all 48 prepared voice
entries. Install it through the WebUI model manager or from the command line:

```sh
python tools/model_manager_v2.py install kitten_tts2 --models-root models
```

The explicit package ID is `kitten_tts2_q8_0`. It installs the GGUF and readable
license/NOTICE files under `models/kitten-tts2/`. No Hugging Face token is needed.

Direct download:
[kitten-tts2-native-q8-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-native-q8-multilingual.gguf)
(3,282,123,776 bytes, approximately 3.28 GB). The published file's SHA-256 is
`e97920ca5053f9fcd4de638dcd8114ed2510d4291a93257473a8843c3ff349ad`.
Separate model components or reference recordings are not needed at runtime.

```sh
audiocpp_cli --family kitten_tts2 \
  --model models/kitten-tts2/kitten-tts2-native-q8-multilingual.gguf \
  --backend cpu --threads 8 --task tts --voice-id Bruno \
  --request-option seed=1234 --text "Hello there. This is native Kitten speech." \
  --out hello.wav
```

### Original source assets

Use the original [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2)
layout, including the full `lm/model.safetensors`, LM configuration/tokenizer,
`config.json`, `cpp/default/voices.json`, and `speaker/model.safetensors`.
Preserve `LICENSE.md` and `speaker/LICENSE`.

Add the official [S3 meanflow checkpoint](https://huggingface.co/ResembleAI/chatterbox-turbo/blob/main/s3gen_meanflow.safetensors)
as `native/s3gen_meanflow.safetensors`, and the
[Chatterbox MIT license](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE)
as `native/LICENSE`. Validated decoder revision:
`749d1c1a46eb10492095d68fbcf55691ccf137cd`; SHA-256:
`d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f`.
The runtime never downloads assets.

The upstream custom `cpp/model-tq2_1.gguf` and packed ternary/emb4 safetensors
are not inputs to this loader. Use the full safetensors or convert to audio.cpp GGUF.

```sh
audiocpp_cli --family kitten_tts2 --model /path/to/kitten-tts-2 \
  --backend cpu --threads 8 --task tts --voice-id Bruno \
  --request-option seed=1234 --text "Hello there. This is native Kitten speech." \
  --out hello.wav
```

The complete native package contains all 47 upstream voices and `PreparedBruno`.
Output is mono 24 kHz audio, written with audio.cpp's existing WAV support.

## Languages

Use the language-named preset for Arabic, Chinese, French, German, Hindi,
Italian, Portuguese, Russian, or Spanish. The 38 named English voices remain
available. These ten languages are registered in the model spec.

```sh
audiocpp_cli --family kitten_tts2 --model models/kitten-tts2/kitten-tts2-native-q8-multilingual.gguf \
  --backend cpu --task tts --voice-id German \
  --text "Guten Morgen. Dies ist ein Test der deutschen Sprachausgabe." --out german.wav
```

The server accepts `"voice": "German"` (or another language preset) with UTF-8
text in `input`. A reference clip and transcript in the target language can also
be supplied for cloning. There is no separate language switch: the voice
reference supplies the accent and pronunciation context.

Text reaches the original Qwen tokenizer without English normalization, matching
upstream's recommendation to use `normalize=False` for other languages. Write
numbers, dates and abbreviations as spoken words in the target language.
The shared chunker preserves Unicode codepoints and supports CJK punctuation.

Upstream's README advertises 20 languages, but its published voice index provides
presets for English and the nine languages above. The remaining ten languages
have no shipped language presets in this snapshot and are not validated here.
Generation and tokenization checks do not establish pronunciation quality;
see the validation record for exactly what was exercised.

## Voice cloning

Supply a one-to-thirty-second reference clip and its transcript:

```sh
audiocpp_cli --family kitten_tts2 --model models/kitten-tts2/kitten-tts2-native-q8-multilingual.gguf \
  --backend cpu --threads 8 --task clon --voice-ref reference.wav \
  --reference-text "The exact words spoken in the reference clip." \
  --text "The new words to say in this voice." --request-option seed=1234 \
  --out cloned.wav
```

Cloning also works in TTS sessions. The C++ request accepts `voice.speaker.audio`
or `audio_input`, with `reference_text` in request options. Reference audio
takes precedence over a preset ID. The server speech endpoint accepts
`voice_ref` (path or base64) and `reference_text`.

Channels are averaged. Native sinc resampling supplies 16 kHz speaker/tokenizer
inputs and the 24 kHz decoder reference. The identity encoder uses the complete
clip. Decoder conditioning repeats short clips to at least six seconds and takes
at most ten seconds, following the upstream default decoder policy.
Upstream uses librosa/soxr for some S3 preprocessing, so reference tokens need not
be byte-identical with this native sinc resampler.

Reference encoders load on the first clone request. A session caches its last
clip and transcript; changing either invalidates conditioning and the LM prefix.
This in-memory cache does not register persistent voice IDs.

## Sampling and long-form text

Stable sampling defaults to temperature 0.8, top-p 0.8, top-k 50.
`preset=expressive` uses temperature/top-p 0.9. Explicit request options override
the preset; temperature zero selects greedy decoding.

The sampler uses the upstream 50-token repetition window, silence/end exemptions,
and token-run penalty. The tied head projects only speech codes and the two end
markers, deliberately excluding stray text tokens. Native RNG and arithmetic
differ from PyTorch CPU; equal seeds do not promise Python-identical waveforms.

Text defaults to 380-codepoint chunks in `tag_aware` mode. Each chunk reuses the
reference prefix and starts fresh target generation. Joins trim edge silence,
apply short fades, and insert 160 ms gaps. Expression tags reach the model
verbatim and enable its emotion prefix. Expand numbers/abbreviations when needed.

Qwen projections default to Q8_0. With original full safetensors,
`--session-option kitten_tts2.weight_type=q4_0` packs folded ternary weights
exactly into standard GGML Q4_0 blocks. Already quantized sources may fail that
exact-packing check; use `native` or `q8_0` with the Q8 package.
Embeddings/the tied head stay F16; decoder and reference encoders use F32.

## Self-contained GGUF

```sh
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /path/to/kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --type q8_0 --output kitten-tts2-native-q8-multilingual.gguf
```

This helper uses Python's standard library, the native converter, and the
`audiocpp_kitten_tts2_prepare_voices` executable beside it. The upstream prepared
`cpp/default/voices.json` omits the nine language presets; the helper fills those
from `voices/voices.json`, the original NPZ embeddings/tokens, and reference WAVs.
It preserves existing English conditioning and computes only missing decoder
conditioning and speaker projections through native GGML. No Torch or NumPy is
needed for preparation or inference.

Preparation uses CPU by default; `--voice-backend cuda` selects a CUDA-enabled
preparer. `--voice-preparer` can select an executable elsewhere. Save the complete
index with `--voices-output voices.json`, then reuse it with
`--prepared-voices voices.json` on later conversions. Incomplete indexes are
rejected so a multilingual spec cannot silently ship English-only voices.
For direct safetensors loading, install that complete index as
`MODEL/cpp/default/voices.json`; the original upstream index alone lacks the
language presets. The GGUF embeds it and needs no adjacent reference clips.

`--s3gen` and `--s3-license` can select assets outside the model directory.
All three tensor sources, preset conditioning, configuration, tokenizer, spec,
licenses, and NOTICE are packaged. No `.pt` file is embedded.
The package supports cloning without adjacent model files. Use the GGUF path as
`--model`.

Weights retain the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md),
including commercial registration/limits and attribution requirements.
Decoder and speaker components retain their respective upstream terms.
No model weights are committed with the integration.

See [validation instructions](../../tests/kitten_tts2/README.md) and
[measured results](../../tests/kitten_tts2/validation.md).
