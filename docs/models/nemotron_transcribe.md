# NVIDIA Nemotron 3.5 Transcribe

`nemotron_transcribe` is an experimental community port of
[NVIDIA Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B).
It accepts complete recordings through the `asr` task. Instructions select
transcription, translation, audio Q&A or language identification. Its CTC head
provides source-language word timestamps and speaker labels. This family is
separate from the smaller `nemotron_asr` streaming model.

The port uses stock GGML, framework weight stores, the mel frontend and tokenizer,
with a model-local parallel speech/speaker encoder and Nemotron-H decoder.
NVIDIA's patched serving kernels, MTP and DFlash acceleration are outside this port.

## Build

```sh
cmake -S . -B build/nemotron -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=nemotron_transcribe -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
  -DENGINE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/nemotron --target audiocpp_cli --parallel 4
```

Omit CUDA for CPU builds. On Windows, use an x64 Visual Studio developer shell
and the `.exe` suffix. Select the destination GPU architecture when distributing
a binary.

## Prepare weights

Use the full BF16 checkpoint, not the separately packed NVFP4 release. The initial
profile has IQ4_NL routed experts, Q8_0 dense decoder linears, BF16/F32 remaining
decoder weights, an F16 speech encoder and an F32 CTC head. Native weight storage
preserves this mixture.

The package also embeds the exact trained BF16 Hann window. It does not regenerate
that window in FP32. Frontend validation uses the actual
[HF feature extractor](https://github.com/huggingface/transformers/blob/0c38601b69653fa2f6f4c88b287e7a8d7c57fb4e/src/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py)
and the checkpoint's trained filterbank.

The wrappers require the pinned
[NVIDIA reference](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/66d50d91b75a406becd95fbaf39d2c9c13e95d7c)
and its pinned llama.cpp submodule. Build a pristine CPU `llama-quantize` and pass
its path explicitly. The wrapper never applies backend patches or builds a
quantizer automatically.

```sh
git clone https://github.com/NVIDIA/NeMo-Speech.cpp reference/NeMo-Speech.cpp
git -C reference/NeMo-Speech.cpp checkout 66d50d91b75a406becd95fbaf39d2c9c13e95d7c
git -C reference/NeMo-Speech.cpp submodule update --init --depth 1 llama.cpp
cmake -S reference/NeMo-Speech.cpp/llama.cpp -B build/nemotron-quantizer \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build/nemotron-quantizer --target llama-quantize --parallel 4
```

Create a Python 3.12 environment with uv and install the pinned reference's
`requirements.txt`. CPU PyTorch suffices for conversion. The package wrapper
uses `gguf==0.19.0` and `numpy==1.26.4`.

Download model revision `a99e33f698257b8b69ca7834f19c73582eb1fa7e`, including
configuration, tokenizer and license files. The checkpoint is about 67.5 GB.
Allow about 160 GB free for source, temporary BF16 decoder and GGUF outputs.
Source and output must share a filesystem for hardlink staging; Windows does
not need symbolic-link privileges.

```sh
uv run --no-project --python .venv-nemotron python \
  tools/community_models/convert_nemotron_transcribe.py \
  --source models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --reference reference/NeMo-Speech.cpp \
  --quantizer build/nemotron-quantizer/bin/llama-quantize \
  --output models/native/nemotron-transcribe-iq4_nl.gguf --threads 16

uv run --no-project --python .venv-nemotron python \
  tools/community_models/prepare_nemotron_transcribe.py \
  --source models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --decoder models/native/nemotron-transcribe-iq4_nl.gguf \
  --encoder models/native/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf \
  --output models/audiocpp/nemotron-transcribe-iq4_nl.gguf
```

The package is two GGUFs in one directory. Pass the decoder to audio.cpp; it
names its encoder sibling. The decoder embeds its v1 spec, configs, tokenizer
and notices. The encoder includes its CTC tokenizer. Checkpoint IDs must match,
and packaging checks source identity, config and tokenizer semantics. Keep the
encoder filename unchanged. No downloadable package is advertised yet.

## Run

```sh
build/nemotron/bin/audiocpp_cli --task asr --family nemotron_transcribe \
  --model models/audiocpp/nemotron-transcribe-iq4_nl.gguf \
  --backend cuda --audio recording.wav --text-out transcript.txt
```

Use 16 kHz mono WAV for matched comparisons. The runtime session requires that
format; the CLI's audio loader handles decoding and resampling.

Pass each task option with `--request-option key=value`:

| Task | Options |
| --- | --- |
| Transcription | Default; optionally `language=ja` |
| Translation | `template_name=translation`, `target_language=English` |
| Audio Q&A | `template_name=audio_qa`, `instruct=What is discussed in this recording?` |
| Language identification | `template_name=language_id` |
| Word timestamps | `return_timestamps=true` |
| Speaker-tagged transcription and timings | `diarization=true` |

`instruct` replaces the template instruction, including its language hint and
speaker-tag request. `--text` supplies the instruction when `instruct` is absent.
Response formatting is preserved. Translation and Q&A templates reject timestamps
and diarization because their output need not match the source words.

Generation is greedy by default. `max_tokens` overrides the checkpoint setting or
the native duration budget `max(24, ceil(6 * seconds - 1e-9) + 24)`. Prompt, audio
and output must fit `nemotron_transcribe.max_context`, default 73,728. The custom
result records token-limit termination. Sampling options are in the model spec.

The native silence gate returns empty text only when peak amplitude is below
0.001 and RMS is below 0.00003. It does not trim audio or repair generated text.

## Timestamps and speakers

`TaskResult.word_timestamps` uses sample offsets in the input audio clock.
`speaker_turns` contains 10 ms speaker-activity segments. Custom output
`nemotron_transcribe.result.v1` includes word times in seconds, speaker column,
speaker confidence, original transcript tag, alignment status and termination.
`speaker_N` denotes Sortformer column N. The transcript's `spk:N` tag can map to
a different column. Chinese/Japanese alignment units are characters; Thai uses
the reference's combining-mark-aware units.

Unavailable or partially failed alignment retains the transcript and available
speaker activity. Failed streams produce no invented word rows. Generic word/turn
confidence zero means unavailable; speaker confidence is reported separately.

## Long recordings and validation

The input limit is one hour. Bounded encoder windows and a stateful speaker
cache/FIFO match the pinned native defaults: 500 core frames plus 50 left/right
frames on an 80 ms grid, with cache/FIFO/update 188/40/300. These differ from the
HF processor's 30-second chunk setting and the diarizer's training geometry.
Session options expose the choices. A native C++ comparison does not establish
complete Python long-form parity.

Local validation used MSVC 14.44, CUDA 12.8.61, an RTX PRO 6000 Blackwell 96 GB,
and an i9-13900K. The final package passed 11 CUDA requests covering English and
Japanese transcription, translation, Q&A, language identification, two-speaker
dialogue, repeated requests, silence and token-limit termination. All nine paired
text outputs matched the pinned native reference after removing speaker markers
and normalizing whitespace. CPU transcription with timestamps also passed.

| Measured request | Audio | Processing time, model already loaded |
| --- | --- | --- |
| CUDA English with timestamps | 7.33 s | 0.556 s |
| CUDA distinct two-speaker dialogue with timestamps | 49.055 s | 1.232 s |
| CPU English with timestamps, four threads | 7.33 s | 6.250 s |

These are individual requests on a small, self-generated synthetic speech set,
with other GPU workloads active. CUDA model loading took 14.35 seconds separately.
They are not a held-out accuracy or throughput benchmark. Component checks also
cover official HF frontend parity, full real encoder weights, decoder math,
actual CTC weights, Unicode alignment and 3,522 tokenizer cases.

Word times are not identical to the native reference. Maximum observed differences
were 119 ms in the Japanese clip and 424 ms at one dialogue pause. The largest
outlier follows the native C++ frontend's legacy Hann/window-placement/valid-frame
policy, which changes a speaker-activity gap. This port follows the checkpoint/HF
frontend. Both runtimes also omit one repeated sentence pair in a four-repeat
56.44-second fixture; the distinct 49-second dialogue includes all eight turns.
These results do not establish hour-long accuracy. Full-model Metal and Vulkan
execution, broad multilingual accuracy and actual hour-long audio remain untested.
See the [validation record](nemotron_transcribe_validation.md) for pinned inputs,
settings, commands and observed outputs.

Weights retain OpenMDW-1.1 and component notices. Runtime alignment adaptations
and Unicode/numerical attributions are in the model's `THIRD_PARTY_NOTICES.md`
and `LICENSE.apache-2.0`.
