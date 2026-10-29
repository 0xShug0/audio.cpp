# Underdog Bark 3.0

Conway Research's **Standard, Plus and Pro are English ASR variants** of the
existing `parakeet_tdt` community family, not Bark speech-generation models.
All share the 0.6B FastConformer/TDT architecture, publisher's 250 ms prefix,
original-sample timestamps and native runtime. Stock Parakeet and Phonon keep
their behavior; no shared GGML kernels or parallel server runtime are needed.

| Variant | Pinned Hugging Face source / revision | Source weights | Converter | Recommended GGUF |
|---|---|---|---|---|
| Standard | [ConwayResearch/Underdog-Bark-3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) / `85db2ad18ab41e534a222ee253c29191d77ed919` | 163 MB `model.bark` | `convert_underdog_bark3_standard.py` | `underdog-bark-3.0-mixed-f16.gguf` |
| Plus | [ConwayResearch/Underdog-Bark-3.0-Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus) / `d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb` | 180 MB `model.bark` | `convert_underdog_bark3_plus.py` | `underdog-bark-3.0-plus-mixed-f16.gguf` |
| Pro | [ConwayResearch/Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) / `4b5c51e04aa8d16dda9ecdff71bc29f7360149e4` | About 1.25 GB BF16 safetensors | `convert_underdog_bark3.py` | `underdog-bark-3.0-pro-q8_0.gguf` |

The compressed download sizes are not expanded GGUF/RAM/VRAM requirements.
Packages explicitly select `underdog_bark3_standard`, `underdog_bark3_plus` or
`underdog_bark3_pro`. This guide covers GGUF; native compressed `.bark` loading
is proposed separately in PR #1133.

## Convert

Build shared binaries once. Disable both GPU flags for CPU-only builds; adjust
paths for your generator/platform (`.exe` on Windows).

```bash
cmake -S . -B build/bark3 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=parakeet_tdt \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DENGINE_BUILD_WARMBENCH=ON
cmake --build build/bark3 --config Release --target \
  audiocpp_cli audiocpp_server audiocpp_gguf parakeet_warm_bench \
  parakeet_bark3_session_test parakeet_variant_config_test
```

Conversion needs Python, `numpy`, `safetensors` and `huggingface_hub`; add
`constriction==0.5.0` for Standard/Plus or `torch` for Pro. Native inference
requires none of these Python dependencies.

### Standard or Plus

Use the Standard assignments below, or replace all five with the Plus values.
These assignments use Bash syntax; in PowerShell use `$NAME = '...'` syntax or
substitute the literal values.

```bash
NAME=Underdog-Bark-3.0
REVISION=85db2ad18ab41e534a222ee253c29191d77ed919
CONVERTER=convert_underdog_bark3_standard.py
OUTPUT=underdog-bark-3.0-mixed-f16.gguf
TYPE=mixed-f16
# Plus: NAME=Underdog-Bark-3.0-Plus
#       REVISION=d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb
#       CONVERTER=convert_underdog_bark3_plus.py
#       OUTPUT=underdog-bark-3.0-plus-mixed-f16.gguf
#       TYPE=mixed-f16

hf download "ConwayResearch/$NAME" --revision "$REVISION" \
  --include model.bark bark_loader.py mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir "models/$NAME-source"
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir "models/$NAME-source"
python "tools/community_models/$CONVERTER" \
  --source "models/$NAME-source" --staging "models/$NAME-prepared" \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type "$TYPE" --output "models/$OUTPUT"
```

The converters verify checkpoint SHA256, pinned NVIDIA sidecars and vocabulary,
decode ANS/FP16 levels without executing downloaded Python, and prepare 699
inference tensors after dropping 24 training counters. Immutable profiles reject
staging from the other checkpoint. No NVIDIA model weights are downloaded.

### Pro

```bash
hf download ConwayResearch/Underdog-Bark-3.0-Pro \
  --revision 4b5c51e04aa8d16dda9ecdff71bc29f7360149e4 \
  --include model.safetensors config.json processor_config.json tokenizer.json \
    tokenizer_config.json generation_config.json NOTICE README.md coreml/LICENSE \
    runtime/transcribe_bark_pro.py runtime/requirements.txt \
  --local-dir models/Underdog-Bark-3.0-Pro-source
python tools/community_models/convert_underdog_bark3.py \
  --source models/Underdog-Bark-3.0-Pro-source --staging models/Underdog-Bark-3.0-Pro-prepared \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type q8_0 \
  --output models/underdog-bark-3.0-pro-q8_0.gguf
```

All converters support `--prepare-only`, or reuse staging with `--pack-only`
and another `--type`/`--output`. `--overwrite` replaces generated GGUFs; sources
are never changed. Every GGUF embeds the English model spec v1, tokenizer/configs,
source/conversion provenance, NOTICE and full CC-BY-4.0 license. No unpublished
model-manager download is advertised. Projection scaling and BatchNorm variance
floor compensation match the existing loader; corrected gamma stays F32.

| Variant | Precision policy |
|---|---|
| Standard / Plus | `mixed-f16` preserves every staged FP32 value, keeping non-representable row-scale products, normalization and compensated projections in F32. Q8 keeps feed-forward/pointwise matrices in exact F16 and quantizes eligible attention matrices. Q8/BF16 are rounded derivatives. |
| Standard BF16 | Additionally retains predictor, encoder-to-joint projection and output head in F32 to protect case/duration decisions; Plus's BF16 default is unchanged. |
| Pro | BF16 is source precision; F32 is lossless expanded storage, not extra precision. Protected Q8 retains predictor/projector/output head in exact source BF16 and keeps convolutions floating point. |

Keep `matmul_weight_type`/`conv_weight_type` at `native`: explicit overrides can
round protected tensors again.

## CLI and server

Use any recommended GGUF with the same flags:

```bash
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model-spec-override embedded --backend cuda --threads 8 \
  --model models/underdog-bark-3.0-mixed-f16.gguf \
  --audio input-16k-mono.wav --text-out transcript.txt --words-out words.json --metrics

ffmpeg -i input.wav -ar 16000 -ac 1 -f f32le - | \
  build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
    --model-spec-override embedded --backend cuda \
    --model models/underdog-bark-3.0-mixed-f16.gguf \
    --mode streaming --audio - --input-format f32le --input-rate 16000 --input-channels 1
```

Select the embedded contract explicitly; the family catalog otherwise describes
multilingual stock Parakeet. Use `--backend cpu` or `vulkan` for other devices and
`--device` to choose the intended GPU when needed.

The single [server example](../../examples/server/underdog-bark3-asr.json) has
offline and buffered-streaming entries for all three variants. Replace the paths,
or keep only the entries you use. Lazy loading and `max_loaded_models: 1` avoid
keeping all three resident together. Pro's existing IDs remain available.

| Variant | Offline model ID | Streaming model ID |
|---|---|---|
| Standard | `bark3standard` | `bark3standard-stream` |
| Plus | `bark3plus` | `bark3plus-stream` |
| Pro | `bark3` | `bark3-stream` |

```bash
build/bark3/bin/audiocpp_server --config examples/server/underdog-bark3-asr.json
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=bark3standard -F file=@input-16k-mono.wav
```

Streaming entries support the raw-PCM/SSE
[`/v1/audio/transcriptions/live`](../../app/server/api.md#post-v1audiotranscriptionslive)
route. These use the ordinary legacy server, without parallel-runtime flags.

## Input, context and timestamps

Supply original unpadded audio. The adapter adds **4000 zero samples at 16 kHz
(250 ms)** once per request/stream after resampling/downmixing. Word timestamps
subtract the prefix and stay within the recording; they follow native TDT emission
boundaries, not an independently validated forced aligner. For source parity use
identical mono 16 kHz inputs: the publisher uses a different resampler at other
rates. Streaming requires contiguous mono 16 kHz PCM chunks. Empty audio is
rejected; silence is valid.

Full context is the default. Bidirectional attention grows quadratically with
duration and retains the encoder frame limit. Explicit `offline_mode=auto`
switches to bounded windows after 30 seconds including the prefix; `long_form`
always uses windows. Windowing and offline VAD can change punctuation/case and
word spans. Enable offline-only VAD with:

```text
--request-option audio_chunk_mode=vad
--request-option audio_chunk_duration_sec=30
--session-option parakeet_tdt.vad_model_path=assets/framework/models/silero_vad
```

The bundled default is Silero VAD. Each speech span receives its own prefix and
keeps the original sample clock. Optional `hotwords`/`hotwords_score` provide
native token-logit bias (also HTTP `prompt`), with comma-separated terms or JSON
word/spoken forms. Empty lists or zero score disable bias; this is an extension
to the original runner. Standard emits `Yeah.` on one-second silence; unfiltered
native inference preserves it, while VAD can skip non-speech.

Streaming is buffered, not native cache-aware: bidirectional attention and
whole-utterance normalization change earlier activations when future audio
arrives. The frontend only caches raw log-mel frames whose complete
FFT/preemphasis support fits both windows. Edges and F64 whole-window normalization
are recomputed and caches reset for every stream; encoder/context policy is unchanged.

## Validation

Evidence remains in the separate [Standard](../validation/underdog_bark3_standard.md),
[Plus](../validation/underdog_bark3_plus.md) and [Pro](../validation/underdog_bark3.md)
validation records, plus the [initial Pro comparison](../validation/underdog_bark3_original_comparison.md).
CPU/CUDA/Vulkan are tested; Metal/HIP and corpus WER are unvalidated. Fixture
matches do not imply universal rounded parity. Plus CPU Q8 omitted one comma on
the five-minute recording; mixed-F16 is recommended for Standard/Plus fidelity.
The CPU graph fix reduced the tested five-minute allocation from 35.2 GiB to
2.62 GiB without changing full-context recognition.

For references use a separate Python 3.11 environment with PyTorch 2.8.0,
Transformers 5.19.0, soundfile, soxr and librosa, plus conversion dependencies.
Original loaders may fetch processor/generation metadata, not NVIDIA weights.
The reference scripts are `bark3_standard_reference.py`, `bark3_plus_reference.py`
and `bark3_reference.py` for Standard, Plus and Pro respectively:

```bash
python tests/parakeet_tdt/bark3_standard_reference.py \
  --source models/Underdog-Bark-3.0-source \
  --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --device cpu --check-official-transcribe --output build/bark3/reference.json
python tests/parakeet_tdt/validate_bark3.py --bin-dir build/bark3/bin \
  --model models/underdog-bark-3.0-mixed-f16.gguf --backend cpu \
  --reference build/bark3/reference.json --output-dir build/bark3/validation
python -m unittest discover -s tests/parakeet_tdt -p 'test_bark3*conversion.py'
ctest --test-dir build/bark3 -C Release \
  -R 'parakeet_bark3_session_test|parakeet_variant_config_test' --output-on-failure
```

The harness reuses a session across input lengths and repeats. Unit/path tests
cover prefix clocks, streaming callbacks, request reuse and exception recovery.
Weights are CC-BY-4.0: credit Conway Research, NVIDIA Parakeet v3 and Moondream
Parakeet Ultra, retain NOTICE/license and disclose conversion/rounding changes.
