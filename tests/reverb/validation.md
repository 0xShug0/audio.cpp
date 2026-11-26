# Reverb-TTS validation record

## Initial port (2026-11-25)

Base: audio.cpp `94df9f29` (developed on `5631473a`, rebased). Upstream: reverb-tts 1.0.1 (code, software-mansion-labs/Reverb-TTS `2c72a7f9`), software-mansion/Reverb-TTS
`v1.0.0` (weights), eustlb/higgs-audio-v2-tokenizer `0e3a9177` (codec). Linux x86-64, Intel Xeon
@ 2.10 GHz (AVX-512), 2 threads, CPU backend, GCC 13.3, Release.

### Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=omnivoice,reverb -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_CUDA=OFF \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build -j2 --target audiocpp_cli audiocpp_server audiocpp_gguf reverb_rules_test reverb_frontend_probe
```

Builds without warnings. The OmniVoice changes are additive: `load_omnivoice_assets` gains an
overload taking another family's package spec, and `OmniVoiceReferenceAudioOptions` gains
`normalize_rms` (default `true`, so OmniVoice's behaviour is unchanged).

### Packages

```sh
for type in f16 q8_0; do
  python tools/gguf/reverb_convert.py --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer \
    --output Reverb-TTS-package \
    --audiocpp-gguf build/bin/audiocpp_gguf --gguf Reverb-TTS-GGUF/reverb-tts-$type.gguf --type $type
done
```

| Package | Bytes |
|---|---|
| `reverb-tts-f16.gguf` | 999,722,272 |
| `reverb-tts-q8_0.gguf` | 875,481,568 |

### Parity

- `reverb_rules_test`: pass.
- `check_frontend.py`: 0 failures (300 texts; resampling 48k/44.1k/22.05k/16k to 24k and 24k to
  16k, max abs error 1.2e-7 to 1.8e-7; `raise_quiet` 6e-8; `speech_length` and `trim_silence` exact).
- `check_parity.py` (safetensors package, `reverb.weight_type=f32`, `end_cut=0.2` for upstream's fixed end cut): `en_female_narrator` seed 0,
  `en_male_narrator_1` seed 7 and `fr_female_standard` seed 3 all give the same sample count and
  waveform correlation 1.00000000. The prompt token ids, first-pass log-probabilities (max abs error
  3.2e-3, every argmax equal) and all 840 generated codes of the first case were also compared
  directly and match.
- Voice cloning, `assets/resources/b.wav` (24 kHz, 14.07 s): 352 reference frames and
  `speech_frames` 352 in both. Reference codes agree with the upstream codec per codebook
  0.991, 0.989, 0.980, 0.980, 0.972, 0.952, 0.929, 0.926 with the F16 GGUF codec (upstream runs the
  codec in fp32).
- F16 GGUF: 96.5% of the first case's codes equal the fp32 run (codebook 0: 100%). Q8_0: 36%
  (codebook 0: 71%). Masked-diffusion sampling amplifies small logit differences into a different
  take, so these are expected; intelligibility is unaffected (below).

### Intelligibility (`check_quality.py`, Whisper small.en)

| Run | WER |
|---|---|
| reverb-tts 1.0.1, fp32 CPU | 0.00% (0/140) |
| audio.cpp F16 GGUF | 0.00% (0/140) |
| audio.cpp Q8_0 GGUF | 0.00% (0/140) |

Long form (Q8_0, seed 3, a 103-word paragraph, split into pieces by the estimate): 36.6 s of audio,
WER 0.0%.

### Speed and memory (CPU, 2 threads)

First case (4.0 s of audio, 69 prompt tokens + 372 reference frames, 105 target frames):

| Run | Generate | Codec decode | Wall | Peak RSS |
|---|---|---|---|---|
| audio.cpp F16 GGUF | 9.08 s (prefix 2.59 s) | 2.63 s | 11.7 s | 1243 MB |
| audio.cpp Q8_0 GGUF | 9.24 s (prefix 2.66 s) | 2.65 s | 11.9 s | 1232 MB |
| reverb-tts fp32 (PyTorch, generation only) | 5.89 s | | | |

On this two-core machine generation is compute-bound (about 85 GFLOP/s across the ten target
passes); PyTorch's CPU kernels are faster here. Long-lived sessions (`--request-sequence`) ran at
0.29x to 0.45x real time.

### CUDA (RTX 4090 24 GB, Arch Linux)

`gpu_test.sh`-style run: `-DENGINE_ENABLE_CUDA=ON`, `--backend cuda`, default session options, the
same 12 requests (6 preset, 6 cloned from `assets/resources/b.wav`) in one `--request-sequence`
session, the 103-word long-form paragraph, and a French preset, for each package. Weights and the codec
load on `CUDA0`; `reverb_rules_test` passes.

| Package | Warm requests (x real time, min / median / max) | First request (cold) | Long form (36.7 s) | WER, 12 requests | WER, long form | GPU memory above idle |
|---|---|---|---|---|---|---|
| Q8_0 | 34.5 / 45.1 / 58.2 | 4.6x (0.84 s, includes graph setup) | 0.92 s (39.8x) | 0.00% (0/140) | 0.0% (0/103) | about 2.3 GB peak |
| F16 | 28.2 / 61.4 / 72.2 | 4.7x | 0.96 s (38.1x) | 0.00% (0/140) | 0.0% (0/103) | about 2.0 GB peak |

A warm 4 to 6 s request takes about 95 ms: generation about 87 ms (prefix about 9 ms), codec decode
7 to 11 ms. GPU memory was sampled every 0.5 s with `nvidia-smi` around the whole run, so it
includes model load. WER from Whisper small.en as in `check_quality.py`.

### Vulkan (same RTX 4090, Arch Linux)

`-DENGINE_ENABLE_VULKAN=ON` (needs the `spirv-headers` package), `--backend vulkan`, the same runs as
CUDA. Weights and the codec load on `Vulkan0`; `reverb_rules_test` passes.

| Package | Warm requests (x real time, median / max) | First requests | Long form (36.7 s) | WER, 12 requests | WER, long form | GPU memory above idle |
|---|---|---|---|---|---|---|
| Q8_0 | 35.6 / 49.9 | 0.45x first preset, 0.9x first clone (pipeline compilation) | 0.93 s (39.3x) | 0.00% (0/140) | 0.0% (0/103) | about 2.2 GB peak |
| F16 | 38.5 / 53.6 | 4.0x | 0.99 s (36.9x) | 0.00% (0/140) | 0.0% (0/103) | about 1.7 GB peak |

A warm request takes about 120 ms. The first request of each new graph shape pays Vulkan pipeline
compilation (the first cloned request also builds the codec encoder); later sessions reuse the
driver's pipeline cache, as the F16 run's 4.0x first request shows.

### Server

`audiocpp_server` with the Q8_0 GGUF: `POST /v1/audio/speech` with `voice` `en_male_narrator_1`
and `fr_female_standard` returned 200 and WAV audio.

### Known limitations

- No Verbalis text normalization: numbers, dates and symbols are passed through as written.
- Offline only (upstream has no streaming either); no batching of several requests.
- Validated on CPU, CUDA and Vulkan; Metal untested.
