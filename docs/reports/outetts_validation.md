# OuteTTS validation

This procedure exercises OuteTTS 1.0 1B in one long-lived audio.cpp session.
It covers normal TTS, framework long-form text chunking, voice cloning, repeated
reference-profile cache hits, cached-step graph reuse, stage timings, and
default-versus-`mem_saver` memory behavior.

## Model setup

Install the safetensors model, IBM DAC, and Qwen3 Forced Aligner resources:

```bash
python tools/model_manager.py install outetts_1_0_1b --models-dir models
```

The model-manager package id is `outetts_1_0_1b`. It creates these model
roots:

- `models/Llama-OuteTTS-1.0-1B`
- `models/DAC.speech.v1.0`
- `models/Qwen3-ForcedAligner-0.6B`

The standalone GGUF command is documented in [TTS](tts.md#outetts). The packed
file used below contains the OuteTTS language model, IBM DAC, Qwen aligner,
tokenizers, configuration, and package specification.

## Python reference setup

The parity run used the official OuteTTS repository at commit
`f5eac6e70d792844c6a6959d900a47af2c061a5b` (`outetts` 0.4.4), Python
3.10, PyTorch 2.5.1+cu121, and Transformers 4.52.3:

```powershell
git clone https://github.com/edwko/OuteTTS build\reference\OuteTTS
git -C build\reference\OuteTTS checkout f5eac6e70d792844c6a6959d900a47af2c061a5b
python -m venv --system-site-packages build\reference\venv
build\reference\venv\Scripts\python.exe -m pip install -e build\reference\OuteTTS --no-deps
build\reference\venv\Scripts\python.exe -m pip install transformers==4.52.3 llama-cpp-python==0.3.9 polars natsort mecab-python3 unidic-lite uroman openai-whisper ftfy pyloudnorm
```

The reference loaded `models/Llama-OuteTTS-1.0-1B` through
`outetts.Backend.HF` with FP32 CUDA weights and loaded the official IBM DAC
checkpoint from
`models/DAC.speech.v1.0/weights_24khz_1.5kbps_v1.0.pth`. Both implementations
used temperature `0.4`, repetition penalty `1.1` over the latest 64 tokens,
top-k `40`, top-p `0.9`, min-p `0.05`, the request-file seeds, and the same
maximum number of new tokens.

Temperature zero was deliberately not used. OuteTTS 1.0's official
`generation_config.json`, README example, and Python API all select sampled
generation with temperature `0.4`; changing to greedy generation would test a
different inference contract. OuteTTS has no diffusion or latent-noise input.
Its only request-time randomness is language-model token sampling.

For a controlled clone comparison, Qwen word timings were generated once and
passed to the official Python `AudioProcessor.create_speaker_from_dict` path:

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task align --family qwen3_forced_aligner `
  --model ..\models\Qwen3-ForcedAligner-0.6B_Q8\Qwen3-ForcedAligner-0.6B_Q8.gguf `
  --backend cuda --audio assets\resources\b.wav `
  --text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." `
  --language en --words-out build\reference\b_words.json
```

Python still performed the official DAC encoding and feature extraction. This
keeps different alignment algorithms from contaminating the TTS comparison.
The audio.cpp cold-clone timing includes its Qwen alignment, while the Python
number starts from the supplied word timings.

Run the committed official-reference driver after creating the alignment JSON:

```powershell
build\reference\venv\Scripts\python.exe `
  tools\audiocpp_cli\outetts_reference.py `
  --model ..\models\Llama-OuteTTS-1.0-1B `
  --dac ..\models\DAC.speech.v1.0\weights_24khz_1.5kbps_v1.0.pth `
  --alignment-json build\reference\b_words.json `
  --request-file tests\outetts\warm_bench_requests.json `
  --device cuda --dtype fp32 `
  --out-dir build\reference\official_script_fp32
```

The driver resets both `torch.manual_seed` and
`torch.cuda.manual_seed_all` immediately before every `model.generate` call,
after all prompt, alignment, and DAC-profile work. It writes native 24 kHz
WAVs plus `boundaries.json`, `memory.json`, `stdout.log`, and `command.json`.

## Exact build commands

The Windows validation used the project build script to establish the compiler
and backend configuration, then explicitly enabled the warm-benchmark targets.

Windows CPU:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -ConfigureOnly
cmake -S . -B build\windows-cpu-release -DENGINE_BUILD_WARMBENCH=ON
cmake --build build\windows-cpu-release --parallel 4 --target audiocpp_cli audiocpp_server outetts_warm_bench
```

Windows CUDA 12.4:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
.\scripts\build_windows.ps1 -Preset windows-cuda-release -ConfigureOnly
cmake -S . -B build\windows-cuda-release -DENGINE_BUILD_WARMBENCH=ON
cmake --build build\windows-cuda-release --parallel 4 --target audiocpp_cli audiocpp_server outetts_warm_bench
```

## Exact standalone-GGUF path tests

The tested GGUF was deliberately outside the PR worktree and its directory
contained only one file:

```text
..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf
```

Normal TTS:

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task tts --family outetts `
  --model ..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf `
  --backend cuda `
  --text "This is the standalone GGUF path test." `
  --max-tokens 256 --request-option seed=1234 `
  --out ..\outputs\outetts_path_test_tts.wav
```

Voice cloning using the aligner and DAC embedded in the same GGUF:

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task clon --family outetts `
  --model ..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf `
  --backend cuda `
  --voice-ref assets\resources\b.wav `
  --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." `
  --request-option reference_language=en `
  --text "This is the standalone GGUF cloning path test." `
  --max-tokens 256 --request-option seed=42 `
  --out ..\outputs\outetts_path_test_clone.wav
```

Both commands loaded successfully without an external package spec, tokenizer,
DAC directory, aligner directory, or other sidecar. The generated 24 kHz mono
WAVs passed ffmpeg decoding:

| Artifact | Duration | Bytes | SHA-256 |
|---|---:|---:|---|
| `..\outputs\outetts_path_test_tts.wav` | 1.399667s | 67228 | `92DA5D8438D4E6A79FBDDBD8EDA77D1AFD2A9B47F61085ECD3AD45C913102904` |
| `..\outputs\outetts_path_test_clone.wav` | 1.453000s | 69788 | `E5B46E62C2DB89A654F7EC21A7BE31F575D6F667F8578224386E7C552492B05E` |

## Exact long-lived-session runs

CUDA default:

```powershell
build\windows-cuda-release\bin\outetts_warm_bench.exe `
  --model ..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf `
  --backend cuda --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --hold-seconds 5 `
  --audio-out-dir ..\outputs\outetts_review_cuda_default `
  --log-file build\logs\warmbench\outetts-cuda-default.log
```

Expected trace evidence:

- `outetts.text_chunk_count` is greater than one for `tts_longform`.
- `outetts.llama.step.graph_reused=1` appears after the first compatible
  generation request or chunk.
- the second identical reference reports `outetts.reference_cache.hit=1`.
- `tts_cold` and `tts_repeat` are byte-identical, as are `clone_cold`
  and `clone_repeat`; this verifies that warm graph/profile reuse does not
  change deterministic output.
- `outetts.aligner.runtime_reused=1` is observable for uncached references
  while the default session retains the aligner.
- only one active OuteTTS Llama runtime is retained; switching between native
  TTS weights and the CUDA F32 cloning fallback replaces the previous runtime.

Run the same request sequence in a fresh process with memory saver enabled:

```powershell
build\windows-cuda-release\bin\outetts_warm_bench.exe `
  --model ..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf `
  --backend cuda --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --session-option outetts.mem_saver=true `
  --hold-seconds 5 `
  --audio-out-dir ..\outputs\outetts_review_cuda_mem_saver `
  --log-file build\logs\warmbench\outetts-cuda-mem_saver.log
```

CPU default:

```powershell
build\windows-cpu-release\bin\outetts_warm_bench.exe `
  --model ..\models\Llama-OuteTTS-1.0-1B_Q8\Llama-OuteTTS-1.0-1B_Q8.gguf `
  --backend cpu --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --hold-seconds 5 `
  --audio-out-dir ..\outputs\outetts_review_cpu_default `
  --log-file build\logs\warmbench\outetts-cpu-default-final.log
```

Safetensors CUDA parity run (DAC is resolved by the package spec; the aligner
path is supplied because it is not embedded in the safetensors directory):

```powershell
build\windows-cuda-release\bin\outetts_warm_bench.exe `
  --model ..\models\Llama-OuteTTS-1.0-1B `
  --backend cuda --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --session-option outetts.weight_type=f32 `
  --session-option outetts.aligner_model_path=..\models\Qwen3-ForcedAligner-0.6B `
  --audio-out-dir build\reference\cpp_f32_final_explicit `
  --log-file build\reference\cpp_f32_final_explicit.log
```

The memory-saver trace reports a positive
`outetts.llama.step.released_cache_capacity` after generation and
`outetts.aligner.runtime_released=1` after an uncached reference.

Sample total-device VRAM every 250 ms while each fresh benchmark runs. Ensure
that no unrelated CUDA workload is active:

```powershell
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250
```

Compare peak and final resident VRAM, request wall time, output duration, and RTF
between the two runs. Keep model, backend, device, seed, requests, and
quantization identical.

## Generated artifacts

Each benchmark directory contains:

```text
tts_cold_1.wav
tts_repeat_1.wav
tts_longform_1.wav
clone_cold_1.wav
clone_repeat_1.wav
```

The exact generated directories and logs were:

- `..\outputs\outetts_review_cuda_default`
- `..\outputs\outetts_review_cuda_mem_saver`
- `..\outputs\outetts_review_cpu_default`
- `build\logs\warmbench\outetts-cuda-default.log`
- `build\logs\warmbench\outetts-cuda-mem_saver.log`
- `build\logs\warmbench\outetts-cpu-default-final.log`
- `build\logs\warmbench\outetts-ab-torch-sampler\default-{1,2,3}-audio`
- `build\logs\warmbench\outetts-ab-torch-sampler\mem_saver-{1,2,3}-audio`
- `build\reference\cpp_f32_final_explicit`
- `build\reference\cpp_q8_torch_sampler`
- `build\reference\official_script_fp32\outputs`
- `build\reference\official_script_fp32\boundaries.json`

These validation artifacts are reproducible local outputs and are not committed
to the repository.

## Measured validation

The committed five-request sequence was measured with the packed Q8 GGUF on an
NVIDIA GeForce RTX 3090 (CUDA 12.4). Each CUDA mode was run in three fresh
processes in alternating order. Values below are mean +/- sample standard
deviation. The Python HF and audio.cpp safetensors columns are single loaded-
session runs; model load is excluded from every per-request measurement.

| Request | Python HF FP32 wall / RTF | C++ FP32 wall / RTF | C++ Q8 wall / RTF |
|---|---:|---:|---:|
| `tts_cold` | 4599.12 ms / 3.485 | 5366.24 ms / 4.066 | 3211.84 +/- 17.30 ms / 2.434 |
| `tts_repeat` | 3777.69 ms / 2.863 | 3864.98 ms / 2.929 | 3030.79 +/- 18.60 ms / 2.297 |
| `tts_longform` | 15129.91 ms / 2.690 | 15547.80 ms / 2.764 | 13015.83 +/- 171.29 ms / 2.342 |
| `clone_cold` | 5762.51 ms / 3.792 | 5988.01 ms / 3.940 | 6432.68 +/- 65.11 ms / 4.233 |
| `clone_repeat` | 5522.67 ms / 3.634 | 4231.19 ms / 2.784 | 4220.23 +/- 44.07 ms / 2.777 |

These are per-request timings from a single loaded session; model loading is
excluded. Python and C++ FP32 generated 1.31967, 1.31967, 5.62533, 1.51967,
and 1.51967 seconds respectively. Q8 generated 5.55867 seconds for the
long-form request and matched the other four durations.

The memory-saver A/B is also reported per request rather than inferred from a
single aggregate run:

| Request | Default CUDA Q8 | Memory saver CUDA Q8 | Wall delta | Default RTF | Memory-saver RTF |
|---|---:|---:|---:|---:|---:|
| `tts_cold` | 3211.84 +/- 17.30 ms | 3223.24 +/- 12.89 ms | +0.35% | 2.434 | 2.443 |
| `tts_repeat` | 3030.79 +/- 18.60 ms | 3044.58 +/- 18.76 ms | +0.45% | 2.297 | 2.307 |
| `tts_longform` | 13015.83 +/- 171.29 ms | 13070.17 +/- 105.32 ms | +0.42% | 2.342 | 2.351 |
| `clone_cold` | 6432.68 +/- 65.11 ms | 6467.00 +/- 46.38 ms | +0.53% | 4.233 | 4.256 |
| `clone_repeat` | 4220.23 +/- 44.07 ms | 4238.27 +/- 17.07 ms | +0.43% | 2.777 | 2.789 |

Memory saver was slightly slower on every request in this controlled repeat:
0.35% to 0.53%, and 0.44% over the mean five-request wall total. These small
differences are near normal run-to-run variance, so there is no evidence that
memory saver is a speed optimization. It remains useful
because the independently sampled peak VRAM fell from 17653 MiB to 5780 MiB
(67.3%) with the same 294 MiB post-session resident VRAM. It is therefore a
memory-versus-speed option, not a speed optimization.

The Windows CPU Q8 run is likewise reported per request:

| Request | Wall | Audio | RTF |
|---|---:|---:|---:|
| `tts_cold` | 10517.2 ms | 1.31967s | 7.970 |
| `tts_repeat` | 10349.9 ms | 1.31967s | 7.843 |
| `tts_longform` | 41604.0 ms | 5.67867s | 7.326 |
| `clone_cold` | 22849.3 ms | 1.58633s | 14.404 |
| `clone_repeat` | 18205.6 ms | 1.58633s | 11.477 |

All generated WAV files passed an ffmpeg decode check. The traces confirmed
four framework chunks for the long-form request, compatible step-graph reuse in
default mode, explicit graph release in memory-saver mode, and a reference-
profile cache hit on the repeated clone. In the original instrumented default
CUDA run, cold reference preparation took about 890 ms (109 ms alignment,
139 ms DAC encoding, and 214 ms profile construction), while the repeated
cached reference took 0.29 ms.

## Path and parity results

- The standalone path test loaded the one-file GGUF from outside the worktree;
  no adjacent sidecars or `model_specs` directory were present.
- Within CUDA, cold and repeated TTS were byte-identical with SHA-256
  `9D1AF0D25212129B3C26880F7C928CFFA813A4B9EF22F12F390A4C372D51D406`.
- Within CUDA, cold and repeated cloning were byte-identical with SHA-256
  `EB7E6B5A1330F845F45D3DA21C8AC28490572F95218A6F0D3DA2A83CB797EFE6`.
- The CUDA default and memory-saver outputs had the same hashes.
- Within CPU, cold and repeated TTS were byte-identical with SHA-256
  `A315C0AD425C99910597FCC4BAB9D80CB5F4C893176EBD9099BA6C6AA7B66BFD`.
- Within CPU, cold and repeated cloning were byte-identical with SHA-256
  `1A2009C0D512773A4B5082BF37EDEDD6A52A37CC6D556772C62EA26550A83B4D`.
- CPU and CUDA WAV bytes are not identical because backend floating-point
  execution differs; deterministic reuse parity was therefore checked within
  each backend rather than asserted across backends.

Python parity uses the repository's
`tools/audiocpp_cli/compare_audiocpp_cli_path_results.py` implementation: PCM
WAV cosine plus an 80-band `log1p` mel-spectrogram cosine (`n_fft=1024`,
`hop=256`). Both sides write the DAC-native 24 kHz signal directly. Raw WAV
cosine is highly phase-sensitive; log-mel cosine is the more informative value
when the DAC encoders are numerically close but not bit-identical.

The CUDA sampler now uses the framework's PyTorch-compatible exponential-race
random stream, equivalent to the `torch.multinomial` path used by Transformers.
Temporary trace-level instrumentation compared prompt strings, prompt token
IDs, generated token IDs, codec codes, and WAV frame counts at component
boundaries. It was removed after validation to avoid normal-log spam. The
permanent Python driver records the Python side in `boundaries.json`.

Boundary results under the controlled FP32 run:

| Request | Chunks | Python/C++ prompt IDs | Generated IDs per chunk | Python/C++ frames |
|---|---:|---:|---:|---:|
| `tts_cold` | 1 | 22 / 22 | 256 | 31672 / 31672 |
| `tts_repeat` | 1 | 22 / 22 | 256 | 31672 / 31672 |
| `tts_longform` | 4 | 20,22,14,21 / same | 256 each | 135008 / 135008 |
| `clone_cold` | 1 | 2367 / 2367 | 256 | 36472 / 36472 |
| `clone_repeat` | 1 | 2367 / 2367 | 256 | 36472 / 36472 |

The sampled prompt/generated boundary values for all six ordinary-TTS chunks
had zero mismatches. Clone prompt construction also matches the official word
labels and token count. The C++ DAC reference encoder still differs slightly
from Python: among 1046 reference-profile frames, C1 differs on 48 and C2 on
118. That numerical boundary explains why cloning is not waveform-identical.

Primary native/FP32 parity, measured only after the controlled random stream
and frame counts matched:

| Request | WAV cosine | Log-mel cosine | C++ / Python frames |
|---|---:|---:|---:|
| `tts_cold` | 0.999812247 | 0.999997481 | 31672 / 31672 |
| `tts_repeat` | 0.999812238 | 0.999997483 | 31672 / 31672 |
| `tts_longform` | 0.997324530 | 0.999294328 | 135008 / 135008 |
| `clone_cold` | -0.007753064 | 0.889403845 | 36472 / 36472 |
| `clone_repeat` | -0.007753064 | 0.889403845 | 36472 / 36472 |

Only after that path was clean was the packed Q8 model compared with the same
FP32 Python reference. Quantization is expected to perturb autoregressive
sampling, so this is a secondary characterization rather than the correctness
gate:

| Request | WAV cosine | Log-mel cosine | C++ Q8 / Python frames |
|---|---:|---:|---:|
| `tts_cold` | -0.016326171 | 0.696795792 | 31672 / 31672 |
| `tts_repeat` | -0.016326591 | 0.696792751 | 31672 / 31672 |
| `tts_longform` | 0.577927575 | 0.920741352 | 133408 / 135008 |
| `clone_cold` | -0.033990775 | 0.882715855 | 36472 / 36472 |
| `clone_repeat` | -0.033990775 | 0.882715855 | 36472 / 36472 |

## Backend coverage

| Backend | Coverage |
|---|---|
| Windows CPU | CLI/server/warm-benchmark build; normal TTS, long-form TTS, cloning, cache reuse, graph reuse, WAV decode, timing, and RSS run |
| Windows CUDA 12.4 | CLI/server/warm-benchmark build; normal TTS, long-form TTS, cloning, standalone-path loading, cache/graph reuse, three-run default/memory-saver A/B, Python HF parity, WAV decode, timing, RTF, and VRAM run |
| Linux CPU | GitHub Actions compile check passed for CLI, server, and GGUF converter; no model runtime measurement claimed |
| Linux Vulkan | GitHub Actions compile check passed for CLI, server, and GGUF converter; no model runtime measurement claimed |
| macOS CPU | GitHub Actions compile check passed for CLI, server, and GGUF converter; no model runtime measurement claimed |
| Metal | Not enabled or runtime-tested in this validation |

## Known limitations

- OuteTTS is offline-only in this implementation.
- Voice cloning requires an accurate transcript of the reference WAV and rejects
  references longer than 20 seconds.
- Quantized CUDA cloning expands the OuteTTS language-model weights to F32 for
  generation correctness. Default mode can therefore have a high transient
  peak while the aligner is retained; `outetts.mem_saver=true` releases the
  aligner and cached-step graph between phases.
- Long-form output concatenates independently generated chunks. `max_tokens`
  applies to each chunk; the runtime splits text further to fit an explicit cap
  and retries a chunk that reaches the cap without EOS/audio-end as smaller
  chunks rather than returning silently truncated speech. It reports an error
  only when the remaining text cannot be split any further.
- CPU and CUDA outputs are deterministic within each tested backend but are not
  expected to be byte-identical across backends.
- The controlled CUDA sampling stream is PyTorch-compatible. CPU sampling is
  deterministic for a fixed seed but still uses the C++ standard-library
  distribution and is not claimed to reproduce PyTorch token-for-token.
- Quantized weights can change sampled tokens and output length even when the
  seed and sampler implementation match, as shown by the Q8 long-form frame
  drift above.
