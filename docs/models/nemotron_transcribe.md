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

## Download packages

[Community GGUF packages](https://huggingface.co/Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF)
are available for this experimental port. Select one decoder and download the
shared `mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf` beside it.
The encoder is 1,670,410,080 bytes and includes the F32 CTC timestamp head.
It is a required auxiliary component, not a standalone F16 version of the 30B model.
Speech encoder weights use FP16 by default; the CTC head stays FP32. The
`mmproj-` prefix identifies the audio encoder/projection sidecar and keeps it
out of the Hub's standalone-model quantization selector. The updated loader
also accepts the original native filename without the prefix, preferring it
when both names are present. No decoder or encoder tensor data was changed.

| Routed expert profile | Decoder file | Decoder bytes |
| --- | --- | ---: |
| IQ4_NL | `nemotron-transcribe-30b-a4b-experts-IQ4_NL.gguf` | 20,370,389,728 |
| Q4_K | `nemotron-transcribe-30b-a4b-experts-Q4_K_S.gguf` | 21,672,721,184 |
| Q3_K | `nemotron-transcribe-30b-a4b-experts-Q3_K_S.gguf` | 17,294,261,024 |
| Q2_K | `nemotron-transcribe-30b-a4b-experts-Q2_K.gguf` | 13,946,026,784 |

These labels describe the routed experts. All four packages use Q8_0 dense
decoder linears and preserve other BF16/F32 source types. The encoder stays
floating-point. File size is not peak runtime memory. Quantization changes
predictions; the number of bits does not establish an accuracy ranking.

The Q3_K and Q4_K expert files use `_K_S` filename suffixes so the Hugging Face
selector recognizes them. These custom mixed profiles do not use the stock
whole-model llama.cpp `_S` recipes. The tensor types and weights are unchanged;
the conversion flags remain `--profile q3_k` and `--profile q4_k`.

The model-manager entries download the decoder and its shared encoder. Pass the
chosen decoder file as the model path, rather than the directory containing
multiple GGUFs. Specs, configs, tokenizer and license notices are embedded.
The repository also contains readable licenses and a per-tensor format inventory.

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
rest of the encoder filename unchanged. Native conversion writes the original
filename; the runtime accepts that name or the same basename prefixed with
`mmproj-`, as used by the public downloads.

### Q4_K, Q3_K and Q2_K expert profiles

The K profiles use genuine Q4_K, Q3_K or Q2_K routed expert tensors. Dense decoder
linears use Q8_0; embeddings, output projection and other floating-point tensors
retain their source types. They share the F16 speech encoder and F32 CTC head.
The names describe the routed banks, rather than every tensor in the package.

The logical widths 2,688 and 1,856 do not fit a 256-element K block. A standard
quantization request can therefore fall back to a different type. The converter
pads only routed-bank input columns to 2,816 and 2,048 with zeros, and audits the
actual output types. The model adds matching zero input activations. Logical
output widths, routing, normalization and dense projections remain unchanged.
Four-bit K storage can be larger than IQ4_NL because of these padded columns.

These files require this community model's metadata-controlled row-padding
support. Stock NeMo-Speech.cpp and llama.cpp inference do not load this layout.
The existing unpadded IQ4_NL path remains available.

Build the pristine pinned quantizer above. Export one floating-point decoder and
its paired encoder directly from the full checkpoint:

```sh
uv run --no-project --python .venv-nemotron python \
  tools/community_models/export_nemotron_transcribe_bf16.py \
  --source models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --reference reference/NeMo-Speech.cpp \
  --output models/native/nemotron-transcribe-source-BF16.gguf
```

The exporter reuses an existing encoder only when its checkpoint identity and CTC
component agree. `--encoder-only` exports/checks that sibling without writing the
decoder. Conversion work is retained for inspection. Source tensors use hardlinks,
so the source and export destination must share a filesystem.

Create a self-contained package, embedding the spec, configs, tokenizer and exact
trained frontend window while streaming one shared padded cache:

```sh
uv run --no-project --python .venv-nemotron python \
  tools/community_models/quantize_nemotron_transcribe.py \
  --source-bf16-gguf models/native/nemotron-transcribe-source-BF16.gguf \
  --quantizer build/nemotron-quantizer/bin/llama-quantize \
  --profile q4_k --threads 16 \
  --padded-cache models/native/nemotron-transcribe-row256-package-cache.gguf \
  --package-source models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --encoder models/native/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf \
  --output models/audiocpp/nemotron-transcribe-30b-a4b-experts-Q4_K_S.gguf
```

Repeat with `--profile q3_k` and `q2_k`, changing the output filename accordingly.
Keep the source, cache and package arguments identical to reuse the cache and
encoder. No separate packaging copy is needed. The optional `--model-spec` selects
a spec file; freeze it while creating a set of variants. A cache with different
source or package inputs is rejected.

The converter never requantizes an existing quantized model. It checks the pinned
quantizer, every tensor's actual type and physical shape, pairing metadata and
unchanged floating-point payloads. It retains a quantization receipt and refuses
to overwrite existing or partial artifacts. Allow roughly 270 GB for the full
source, floating-point export, shared padded cache, three packages and encoder.

The self-generated converter tests do not need the checkpoint or GPU. Set
`NEMOTRON_TEST_QUANTIZER` to the pinned executable to include real quantization
and package round-trips, then run:

```sh
uv run --no-project --python .venv-nemotron python \
  tests/community_models/nemotron_transcribe/test_quantize.py
```

In PowerShell, set the executable with
`$env:NEMOTRON_TEST_QUANTIZER = 'build/nemotron-quantizer/bin/llama-quantize.exe'`.

## Run

```sh
build/nemotron/bin/audiocpp_cli --task asr --family nemotron_transcribe \
  --model models/audiocpp/nemotron-transcribe-iq4_nl.gguf \
  --backend cuda --audio recording.wav --text-out transcript.txt
```

Use 16 kHz mono WAV for matched comparisons. Other PCM rates and channel counts
are averaged to mono and resampled inside this model, using the framework's
Soxr helper with its linear fallback. The shared CLI/server loaders are unchanged.
Matched 16 kHz mono inputs pass through without resampling.

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

The server accepts the same options in JSON at `/v1/audio/transcriptions`.
Use `/v1/audio/transcriptions/details` for word timestamps and speaker turns.
The generic `/v1/tasks/run` route also returns `custom_schema_output`, including
termination and alignment diagnostics. Multipart WAV uploads support the ordinary transcription fields;
use JSON `options` for task templates, questions and timestamp requests. For example:

```json
{"model":"nemotron","audio":"recording.wav","options":{"template_name":"audio_qa","instruct":"What is discussed in this recording?"}}
```

Replace `nemotron` with the configured server model ID. The JSON audio path is
local to the server. Invalid request options return a client error, including
when the recording is silent. `top_p=0` keeps the highest-ranked candidate.

The native silence gate returns empty text only when peak amplitude is below
0.001 and RMS is below 0.00003. It does not trim audio or repair generated text.

## CPU weight offload

`nemotron_transcribe.disk_experts=true` keeps routed expert banks in their native
GGUF format in host memory and uploads selected experts into a bounded cache.
Expert arithmetic, dense decoder layers and the speech encoder still run on
the selected backend. Offload is disabled by default and requires
`nemotron_transcribe.weight_type=native`.

```sh
build/debug/bin/audiocpp_cli --task asr --family nemotron_transcribe \
  --model models/nemotron-transcribe-30b-a4b-experts-Q4_K_S.gguf \
  --backend cuda --audio recording.wav --log \
  --session-option nemotron_transcribe.disk_experts=true \
  --session-option nemotron_transcribe.expert_cache_size=8 \
  --session-option nemotron_transcribe.prefill_batch_size=1024
```

The required encoder GGUF must be beside the decoder. The same offload controls
apply to Vulkan. Eight resident expert slots per layer is the default; larger
caches trade device memory for fewer uploads. The cache must cover the selected
experts per token and cannot exceed the checkpoint's expert count. The existing
`prefill_batch_size` control also applies to offload: larger batches amortize
expert transfers, use more activation memory and can change numerical rounding.

`nemotron_transcribe.pin_experts=true` optionally retains complete expert banks
in pinned host RAM for direct uploads. This consumes host memory approximately
equal to the native expert payload and requires backend pinned-buffer support.
With pinning disabled, `disk_expert_read_threads` controls host staging copies
(default four); parallel staging requires OpenMP or Apple dispatch. Set it to
one in builds without either facility.

Graphs, recurrent state, staging allocations and the bounded expert cache are
session-owned. Independent requests reset recurrent state and expert-slot
placement. Offloading does not change the router, expert activation or routing
weight formulas, but grouped expert matrix multiplication can change floating
point rounding and therefore generated text.

## Timestamps and speakers

The CTC head is loaded on the first timestamp or diarization request and reused
by later requests in the same session. Plain transcription does not load it.

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

For longer recordings, caller-managed clips matching the source processor's
30-second setting are a useful starting point. Shift each result's times by its
source offset. Separate requests reset speaker identities, and cuts can split
words. The one-hour acceptance limit describes capacity; it does not guarantee
recognition quality for one uninterrupted request.

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
Loading depends on disk cache and system load; this is not a general cold-load estimate.
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

All four downloadable profiles also passed CUDA task checks on the synthetic
fixtures, including English/Japanese transcription, translation, audio Q&A,
language identification, speaker-tagged dialogue, silence, token-limit reporting
and repeated requests. The K converter's six self-generated tests passed with
the pristine pinned quantizer, and the C++ padding test checks floating-point
parity. These checks establish the exercised routes, not broad accuracy.
CTC can attach a later turn's leading word to earlier speaker activity even when
the transcript is complete. Treat precise word times and speaker boundaries as
experimental.

A follow-up CPU server audit passed 24 HTTP checks, including malformed requests,
multipart upload, 48 kHz stereo input and unload/reload. Focused safety tests also
passed on Windows and WSL GCC. Storage overrides are priced from tensor metadata
before uploads; the server's existing opt-in memory guard can reject an expanded
model before allocation. These checks do not establish full-model accuracy at
every storage type.

Weights retain OpenMDW-1.1 and component notices. Runtime alignment adaptations
and Unicode/numerical attributions are in the model's `THIRD_PARTY_NOTICES.md`
and `LICENSE.apache-2.0`.
