# KugelAudio reference-audio cloning validation

Tested locally on Windows, 2026-11-13: RTX 3090 24 GiB, Ryzen 7950X3D,
64 GiB RAM; 8 CPU threads. Initial measurements used base `eeeca9e8`; the final
audit used `eb8281c9` and repeated the runtime checks after moving encoder
configuration parsing into the cloning path.
No shared backend, GGML kernel, generator or server-runtime changes are required.

## Scope and reference

This is an experimental raw-audio conditioning extension, using the acoustic
encoder present in the original weights. The upstream open processor exposes
presets/prepared voice caches, not a supported raw-WAV cloning interface.
The comparison below exercises the unmodified upstream encoder component, not
full PyTorch-versus-native TTS waveform parity.

- Original code: `Kugelaudio/kugelaudio-open`, revision
  `64c75284a4bbeaafc470e85dc9dcdc94fdcb6065`.
- Original weights: `kugelaudio/kugelaudio-0-open`, revision
  `55a171ae0208a57fa8853209989e8c0d95572e89`.
- Preset Q8_0 GGUF: `audio-cpp/audio.cpp-gguf`, revision
  `73ce4162f6cce370dd533c6782b772135c1b2a8e`.
- PyTorch 2.8, Transformers 5.19. Original BF16 encoder values are expanded to
  F32 for the strict component reference; native encoding also stores them in F32.

Feature fixtures: six seconds from `2086-149220-0033.wav` (LibriSpeech reader),
six seconds from local `test.mp3` at 3 seconds (non-speech stress case), and
2.0015 seconds from local `We need to talk about 5min.mp4` at 30 seconds.
All are mono 24 kHz; the last exercises non-aligned right padding.
The original Python normalization and native input differ by at most `5.97e-8`.

| Backend | Maximum feature RMSE | Maximum relative L2 error | Minimum cosine |
| --- | ---: | ---: | ---: |
| CPU | 0.00001551 | 0.000297% | 0.999999999995 |
| CUDA | 0.004355 | 0.083365% | 0.999999653 |
| Vulkan | 0.00001991 | 0.000382% | 0.999999999992 |

Acceptance: CPU/Vulkan relative L2 below `1e-5`, cosine above `0.99999999`;
CUDA relative L2 below `1e-3`, cosine above `0.999999`. All nine cases pass.
CUDA is not bit-identical to strict F32: its existing cuBLAS TF32 policy explains
the difference. A diagnostic with TF32 disabled reduces RMSE to `3.05e-6`–`1.60e-5`.
The original PyTorch encoder with TF32 enabled also differs by approximately
`0.0010`–`0.0039` RMSE; its native CUDA BF16 execution differs by `0.051`–`0.119`.
The implementation does not change global math settings. Vulkan selects the
existing F32 matmul lowering only on the new encoder graph.

## Session, quality and performance

Both cloning references are six-second speech clips: the reader fixture and the
video at 30 seconds. `test.mp3` is excluded from speaker-quality evidence because
ASR found no speech in that excerpt. The generation text is:

> Hello, this is a test of the cloned voice.

Seed 1234, default 20 diffusion steps, guidance 3, greedy AR control tokens,
`max_tokens=64`, native packaged weight types. Model loading and an initial preset
warmup are outside these timers. First clone includes reference preparation and
synthesis; warm clone
reuses the reference. Backends ran consecutively, without concurrent benchmark
requests. These are short single-case measurements, not a throughput benchmark.

| Backend | Generated audio | First clone | Warm clone | Sampled peak GPU VRAM, model included |
| --- | ---: | ---: | ---: | ---: |
| CPU | 3.067 s | 17.12 s | 14.78 s | — |
| CUDA | 3.200 s | 2.13 s | 0.76 s | 10.84 GiB |
| Vulkan | 3.200 s | 2.47 s | 0.91 s | 11.21 GiB |

VRAM is total device usage sampled every 100 ms, including a 0.244 GiB baseline;
it is not an exact allocator peak. CPU process peak RSS was 20.09 GiB, including
model storage/mappings. Encoder F32 weights occupy about 1.28 GiB temporarily;
their graphs/weights are released before synthesis. Only the latest normalized
waveform and mean features are cached. Timing/memory are specific to these clips.

Each backend passes the long-lived session probe: seeded cached repeat;
reference change; silent-reference failure without changing the prepared voice;
explicit preset restoration and empty preparation; clone task missing-reference
recovery; two text chunks; and streamed chunks matching final decoder audio.
Preset samples before/after cloning are bit-identical within each backend.
Streaming matches offline synthesis after applying the existing offline-only
peak-normalization rule. Cross-backend waveforms are not identical.

Independent speaker evidence uses
[Microsoft WavLM Base Plus SV](https://huggingface.co/microsoft/wavlm-base-plus-sv),
revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, with mono 16 kHz inputs.
Scores below compare each reference against its generated clone and the unchanged
English male preset. They support these samples; they do not establish universal
speaker accuracy or listening quality.

| Backend | Reader similarity: clone / preset | Video speaker similarity: clone / preset |
| --- | ---: | ---: |
| CPU | 0.941 / 0.681 | 0.955 / 0.839 |
| CUDA | 0.937 / 0.673 | 0.961 / 0.824 |
| Vulkan | 0.936 / 0.676 | 0.936 / 0.844 |

Independent Bark ASR recovered the requested words on five of six short clones;
CPU's video-speaker clone transcribed “clone” for “cloned.” All three two-chunk
outputs recovered both complete sentences. ASR is an auxiliary content check.

CUDA/Vulkan legacy server checks pass for path/base64 references with identical
decoded audio; cached repeat; HTTP 400 for a silent reference; successful recovery;
and bit-identical preset restoration. CPU was exercised via CLI/C++ sessions.
Standalone encoder safetensors and an encoder-bearing external GGUF both load;
a preset-only package without an encoder fails with an actionable error.
Legacy embedded contracts remain preset-only; the augmented package advertises
TTS and cloning, each offline/streaming. Metal has not been tested.

The augmented GGUF contains 932 tensors. All 656 tensors from the original
preset package retain their types, shapes and bytes; all 276 added encoder
tensors match the original BF16 safetensors bytes exactly. File sizes are
9,752,398,658 bytes before and 10,439,833,634 bytes after. The source package is
preserved; no new package is advertised as an available hosted download.

## Reproduce

From the workspace root in an MSVC x64 shell, configure the tested custom build:

```powershell
cmake -S outputs/ub3p/source -B outputs/ub3p/build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8" -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=kugelaudio -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DENGINE_BUILD_WARMBENCH=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DGGML_CCACHE=OFF -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build outputs/ub3p/build-gpu -j 4 --target audiocpp_cli audiocpp_server audiocpp_gguf kugelaudio_reference_encoder_probe kugelaudio_cloning_session_probe kugelaudio_reference_audio_test gguf_tensor_source_test model_spec_system_test
```

For the CPU build, change the build directory to `build-cpu` and both backend
enable flags to `OFF`. CUDA 12.4, Vulkan SDK 1.4.350 and MSVC 14.43 were used.
The matching source encoder and locally augmented GGUF are described in
[the model documentation](../models/kugelaudio.md#conversion).

```powershell
$env:OMP_NUM_THREADS="8"
$env:OMP_WAIT_POLICY="PASSIVE"
$env:GGML_VK_VISIBLE_DEVICES="1" # this machine: hide the AMD integrated GPU
outputs/ub3p/build-gpu/bin/kugelaudio_cloning_session_probe.exe models/KugelAudio-0-Open-GGUF/kugelaudio-0-open-q8_0-clone.gguf outputs/kugelaudio-clone/session-validation/cuda outputs/kugelaudio-clone/fixtures/reader.wav outputs/kugelaudio-clone/fixtures/movie-speaker.wav cuda
ctest --test-dir outputs/ub3p/build-gpu -R "^(kugelaudio_reference_audio_test|gguf_tensor_source_test|model_spec_system_test)$" --output-on-failure
```

Repeat the probe with `vulkan`; for `cpu`, use the CPU binary. Both build trees
pass all three selected CTests. Frontend cases include downmix/resampling,
normalization, clipping, silence, malformed frames, NaN/Inf and the 30-second cap;
invalid frontend inputs use the framework's request-error type. The exact
30-second boundary is accepted. Configuration cases cover string/array depths,
unsupported architecture flags, invalid strides, channel overflow, graph block
limits and invalid normalization. A preset GGUF with encoder configuration
removed still produces bit-identical preset audio through the original route.

The packaging smoke test uses tiny synthetic weights to exercise the real
converter and embedded-contract inspection; it does not claim synthetic-model
inference. It requires `torch`, `safetensors`, `numpy` and `gguf`:

```powershell
python outputs/ub3p/source/tests/kugelaudio/test_conversion.py --converter outputs/ub3p/build-cpu/bin/audiocpp_gguf.exe --cli outputs/ub3p/build-cpu/bin/audiocpp_cli.exe
```

Both packaging modes pass: preset conversion excludes the encoder and cloning
capability; encoder-enabled conversion retains its BF16 bytes while quantizing
the generator to Q8_0, and advertises the cloning capability. After the final
audit, every session-probe waveform and both longer normal/cloned comparison
samples are bit-identical to the earlier outputs. Detailed audit records are
under `outputs/kugelaudio-clone/pr-audit/`.

Component reference and native feature dumps:

```powershell
python outputs/ub3p/source/tests/kugelaudio/reference_encoder.py --upstream outputs/kugelaudio-audit-20261113/src/kugelaudio_open --config models/KugelAudio-0-Open-source/config.json --encoder models/KugelAudio-0-Open-source/acoustic_encoder.safetensors --audio outputs/kugelaudio-clone/fixtures/reader.wav outputs/kugelaudio-clone/fixtures/test-speaker.wav outputs/kugelaudio-clone/fixtures/movie-short.wav --output-dir outputs/kugelaudio-clone/original-encoder --threads 8
outputs/ub3p/build-gpu/bin/kugelaudio_reference_encoder_probe.exe models/KugelAudio-0-Open-source/acoustic_encoder.safetensors outputs/kugelaudio-clone/fixtures/reader.wav cuda outputs/kugelaudio-clone/encoder-parity/cuda-reader
```

The reference harness also accepts `--device cuda --tf32` and
`--device cuda --dtype bf16` for the precision controls above. Local JSON/log/audio
evidence resides in `outputs/kugelaudio-clone/`: encoder parity, session results,
generation timings, speaker similarities, ASR transcripts and server validation.
