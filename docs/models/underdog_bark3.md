# Underdog Bark 3.0 Pro

[Conway Research's Bark 3.0 Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro)
is an **English ASR** fine-tune of Parakeet-TDT v3, not a Bark speech-generation
model. It reuses the `parakeet_tdt` community runtime; no new GGML kernels or
shared frontend behavior are needed. The converted package selects the
`underdog_bark3_pro` variant explicitly. Stock Parakeet and Phonon are unchanged.

## Convert

The source revision is pinned to
`4b5c51e04aa8d16dda9ecdff71bc29f7360149e4`. The converter checks the checkpoint
SHA-256 and architecture before preparing 699 inference tensors, omitting 24
training-only BatchNorm counters. Install `numpy`, `torch`, `safetensors` and
`huggingface_hub` in a Python environment, then run from the repository root:

```bash
hf download ConwayResearch/Underdog-Bark-3.0-Pro \
  --revision 4b5c51e04aa8d16dda9ecdff71bc29f7360149e4 \
  --include model.safetensors config.json processor_config.json tokenizer.json \
    tokenizer_config.json generation_config.json NOTICE README.md coreml/LICENSE \
    runtime/transcribe_bark_pro.py runtime/requirements.txt \
  --local-dir models/Underdog-Bark-3.0-Pro-source

cmake -S . -B build/bark3 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=parakeet_tdt \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON \
  -DENGINE_BUILD_WARMBENCH=ON
cmake --build build/bark3 --config Release --target \
  audiocpp_cli audiocpp_server audiocpp_gguf parakeet_warm_bench \
  parakeet_bark3_session_test parakeet_variant_config_test parakeet_parity_dump

python tools/community_models/convert_underdog_bark3.py \
  --source models/Underdog-Bark-3.0-Pro-source \
  --staging models/Underdog-Bark-3.0-Pro-prepared \
  --gguf-tool build/bark3/bin/audiocpp_gguf \
  --type q8_0 --output models/underdog-bark-3.0-pro-q8_0.gguf
```

Adjust executable paths for your CMake generator/platform (`.exe` on Windows).
For CPU-only builds disable both GPU flags. To pack another storage type, reuse
the same staging directory with `--pack-only --type f32` or `--type bf16` and a
different output path. The source checkpoint is BF16; F32 means lossless expanded
storage with conversion compensation, not additional source precision.

Every GGUF embeds the tokenizer, configs, model spec v1, conversion provenance,
upstream NOTICE and full CC-BY-4.0 license. Conversion cancels the runtime's
existing 32x input-projection scaling and corrects its BatchNorm variance floor.
Corrected BatchNorm gamma stays F32 in all packages. Q8 is mixed storage: only
supported matrices are quantized; convolution tensors retain floating-point
storage. These packages are manually converted, not advertised as already
published model-manager downloads.

## CLI and server

```bash
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model-spec-override embedded \
  --model models/underdog-bark-3.0-pro-q8_0.gguf --backend cuda \
  --audio input-16k-mono.wav --text-out transcript.txt --words-out words.json --metrics

# Original-runner comparison with lossless expanded storage.
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model-spec-override embedded \
  --model models/underdog-bark-3.0-pro-f32.gguf --backend cpu \
  --session-option parakeet_tdt.offline_mode=full_context --audio input-16k-mono.wav

# Buffered streaming from raw PCM.
ffmpeg -i input.wav -ar 16000 -ac 1 -f f32le - | \
  build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
    --model-spec-override embedded \
    --model models/underdog-bark-3.0-pro-q8_0.gguf --backend cuda \
    --mode streaming --audio - --input-format f32le --input-rate 16000 --input-channels 1
```

For HTTP, edit paths in [the server example](../../examples/server/underdog-bark3-asr.json):

```bash
build/bark3/bin/audiocpp_server --config examples/server/underdog-bark3-asr.json
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=bark3 -F file=@input-16k-mono.wav
```

Select the embedded contract explicitly: source/deployment builds otherwise
prefer the current family catalog, which describes multilingual stock Parakeet.
The embedded contract advertises Bark's English-only options and capabilities.

Use `--backend cpu` or `--backend vulkan` with a suitable device for other
backends. The streaming entry also supports the server's
[`/v1/audio/transcriptions/live`](../../app/server/api.md#post-v1audiotranscriptionslive)
raw-PCM/SSE route. No experimental parallel-runtime flag is involved.

## Input, context and timestamps

The adapter adds the publisher's **4000 zero samples at 16 kHz (250 ms)** once
per request or stream, after resampling/downmixing. Callers supply the original,
unpadded audio. Reported word timestamps subtract that prefix and stay within
the original recording. Timestamps use the existing Parakeet emission-boundary
policy, not an independently validated forced aligner.

Offline input accepts the framework's downmixing/linear resampling; for reference
parity use identical mono 16 kHz inputs (the publisher uses a different resampler
for other rates). Streaming requires contiguous mono 16 kHz PCM chunks.
Empty audio and VAD preprocessing are rejected. Silence is valid.

Bark defaults to `offline_mode=full_context`, matching the publisher's
whole-utterance context. Whole-recording attention grows quadratically and is
unsuitable for arbitrarily long inputs. Explicit `offline_mode=auto` selects
bounded windows beyond the 30-second threshold (including the prefix);
`offline_mode=long_form` always uses them. For CPU long recordings, select one
of these bounded policies to avoid large graph allocations. Windowed offline
and buffered streaming may produce different text
from full-context decoding. Buffered streaming re-encodes bidirectional windows;
it is **not native cache-aware streaming**. No Bark hotword-biasing guarantee is
made. Metal, HIP and large-corpus accuracy are not validated by this port.

On the tested five-minute recording, native GPU full context with F32/BF16
storage exactly matched the original PyTorch F32 transcript, while opt-in
windowed decoding changed punctuation/case and several word spans. Native CPU
full context failed a 35.2 GiB RAM allocation on that recording. See the
[complete comparison](../validation/underdog_bark3_original_comparison.md) for
the context settings, precision differences, timing and GPU-memory tradeoff.

## Validation

[The validation record](../validation/underdog_bark3.md) records tested backends,
storage sizes, parity, lifecycle checks and timing. Reproduce reference checks
using the original, unmodified source snapshot (Transformers 5.19.0,
PyTorch 2.8.0 CPU, float32):

```bash
pip install transformers==5.19.0 soundfile librosa
python tests/parakeet_tdt/bark3_reference.py \
  --source models/Underdog-Bark-3.0-Pro-source \
  --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --output build/bark3/reference.json
python tests/parakeet_tdt/validate_bark3.py --bin-dir build/bark3/bin \
  --model models/underdog-bark-3.0-pro-q8_0.gguf --backend cuda \
  --reference build/bark3/reference.json --output-dir build/bark3/validation
python -m unittest discover -s tests/parakeet_tdt -p test_bark3_conversion.py
ctest --test-dir build/bark3 -C Release \
  -R 'parakeet_bark3_session_test|parakeet_variant_config_test' --output-on-failure
```

The unit test covers prefix/capacity handling, original-sample timestamps,
repeated requests, stereo resampling, streaming callbacks and exception recovery.
The native reference harness reuses one session across different recording
lengths and repeats. Matching these fixtures is not a corpus-level WER claim.

Weights are CC-BY-4.0. Retain the upstream NOTICE/license and credit Conway
Research, NVIDIA Parakeet v3 and Moondream Parakeet Ultra when distributing
derivatives. Conversion changes are recorded in the embedded `conversion.json`.
