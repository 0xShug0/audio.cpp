# Kitten TTS 2 native validation record

Powered by Stellon Labs.

Validated locally on 2026-10-03. The full default decoder and reference-audio
cloning run natively through the CLI, C++ session API, and server. The port remains
experimental. CPU and NVIDIA CUDA builds,
preset synthesis, and reference cloning pass. Other GPU backends have not been
validated.

The validated multilingual GGUF is published at
[dignome/kitten_tts2](https://huggingface.co/dignome/kitten_tts2), with SHA-256
`e97920ca5053f9fcd4de638dcd8114ed2510d4291a93257473a8843c3ff349ad`.
The `kitten_tts2_q8_0` package installs it through the normal model manager.

## Environment and assets

- Windows x64, Visual Studio 2022 Build Tools, Release.
- AMD Ryzen AI 9 HX 470, 12 physical cores, approximately 52 GiB RAM.
- CPU and CUDA backends, eight host inference threads.
- NVIDIA RTX 4060 Ti, 16 GiB VRAM, compute capability 8.9, driver 610.88,
  CUDA Toolkit 13.3.73. CUDA Release build targets architecture 89.
- Initial numerical/performance validation used audio.cpp base `2892ed3e`;
  the port was subsequently applied to `2ba9fa80`. KittenTTS source: `226cb35`.
- Kitten snapshot `baa41e5d2c5f64be0095365672a7858542261271`.
- S3 checkpoint from ResembleAI/chatterbox-turbo revision
  `749d1c1a46eb10492095d68fbcf55691ccf137cd`, SHA-256
  `d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f`.
- Native GGUF: Q8_0 LM projections, F16 embedding/head, original BF16 speaker
  projection, F32 S3 and speaker encoders. The initial English package is
  3,276,893,696 bytes; the complete multilingual package is 3,282,123,776 bytes.
- Python/PyTorch were used only for offline numerical comparisons. The
  converter helper uses standard-library Python and the native converter and
  voice preparer; no Torch/NumPy is needed for packaging or runtime inference.

Commands are in [README.md](README.md). Logs and samples remain local under the
ignored build directories named below; they are not distributed with this patch.
No weights or generated WAVs are committed.

## Integration on base 2ba9fa80

The local `kittentts2` branch was checked again after applying the port to
`2ba9fa80`, retaining the newer OWSM models and Windows UTF-8 process-code-page
support. This pass used a fresh CPU-only Release build in `build-kitten2-pr/`:

- CLI, server, converter, native voice preparer, both probes and model-spec test
  compiled successfully. The new preparer/probes use the UTF-8 executable manifest.
- Model-spec tests, loader/catalog synchronization, all four asset-reader tests,
  Svelte checks (zero errors/warnings) and the regenerated interface build passed.
- The session probe passed seeded repeats, preset switching, three chunks,
  reference cloning/cache changes, stored tensor precision and invalid-input checks.
- All 13 multilingual server cases passed, including nine language presets,
  English Bruno, Chinese chunking and German/Chinese cloning. The API exposed
  all 48 configured voice names.
- Russian text passed through Windows CLI arguments with exact upstream BPE,
  producing finite, non-silent mono 24 kHz audio.
- The native voice preparer accepted accented input/output file paths and
  reproduced the validated German preset exactly.

Logs and WAVs are local to `build-kitten2-pr/`. The multilingual report is
`build-kitten2-pr/multilingual-validation/report.json`.
CUDA numerical and timing results below come from the earlier validation on
`2892ed3e`; the transferred model and shared arithmetic sources are identical
apart from line endings. CUDA was not rebuilt during this branch-integration pass.

## Passed checks

- CLI, server, converter, native session/component probes, model-spec test build.
- `model_spec_system_test` and loader/catalog synchronization pass.
- CPU PE imports contain only Windows/MSVC/OpenMP dependencies. The CUDA binary
  additionally imports `cublas64_13.dll`; neither uses Torch, Python, or ONNX Runtime.
- Original safetensors generate native preset speech and native clones.
- The GGUF has 2,856 tensors, three namespaces, ten sidecars including the model
  spec, licenses/NOTICE, and no TorchScript decoder resource.
- A single loaded GGUF session on each backend generates exactly equal float32 WAV samples on
  seeded repeats, switches preset voices, joins three text chunks, clones two
  reference voices, reuses cached clone conditioning, and rejects unknown voices,
  missing reference transcripts, and clips too short for the speaker encoder.
- A temporary CPU loopback server returns HTTP 200 `audio/wav` for both preset
  and reference-audio speech. Both outputs are finite, non-silent mono 24 kHz.
- The CUDA `audiocpp_server` Release target builds and reports `cpu,cuda`, with
  the RTX 4060 Ti available as CUDA device 0. A loopback HTTP check using the
  Q8 GGUF reports `backend=cuda` and returns HTTP 200 `audio/wav` for both
  preset speech (52,320 samples, RMS 0.15923) and cloning (56,160 samples,
  RMS 0.14871), mono 24 kHz. The test server was stopped. Configuration, log,
  and response measurements are saved under `build-kitten2-cuda/` as
  `server-kitten2.json`, `server-smoke.log`, and `server-validation.json`.

## Numerical parity

Q8 GGUF CUDA LM trace: exactly 301 prompt tokens; maximum logit error
0.253082, mean 0.052854, cosine 0.99999946; native and reference top speech token
2957. Generation emitted 72 codes and 71,520 waveform samples.

Component comparisons use controlled inputs and random draws to isolate
arithmetic from the native/PyTorch RNG difference:

| Component | CPU maximum error | CUDA maximum error |
|---|---:|---:|
| XVectorSincNet normalized identity | 0.000001563 | 0.000003476 |
| BF16 speaker projection (same identity input) | 0 | 0 |
| S3 encoder hidden states | 0.000001907 | 0.000002146 |
| Two-step meanflow mel, zero noise | 0.000842094 | 0.003841341 |
| HiFT pitch, same native mel | 0.000000358 | 0.000000298 |
| HiFT waveform, same mel and zero random draws | 0.000184685 | 0.000020705 |

Both backends pass the same error limits. CPU and CUDA decoder comparisons use
their respective generated speech codes, so these columns are independent
upstream parity checks, not direct CPU-versus-CUDA waveform comparisons.

The waveform comparison excludes the first 960 samples, where the upstream
wrapper applies its fade outside the generic HiFT component. These checks caught
and fixed two shared S3 encoder errors: relative attention must prepend its zero
column, and attention/feed-forward layer norms require epsilon 1e-12.
Other encoder and decoder norms retain their original epsilon.

The speaker comparison uses the exact native-resampled waveform. It does not
claim equality between native sinc and upstream librosa/soxr reference
preprocessing. All tests compare the specified components; they do not establish
perceptual speaker similarity or Python-identical autoregressive sampling.

## CPU timings

Q8 package, measured on the CPU listed above, eight threads, seed 1234.
Model/session load is excluded. The first clone includes lazy reference encoder
loading and conditioning. These are individual short samples, not averages.

| Request | Wall seconds | Audio seconds | RTF |
|---|---:|---:|---:|
| Bruno, first request | 8.858 | 3.10 | 2.857 |
| Bruno, cached repeat | 5.992 | 3.10 | 1.933 |
| Switch to Bella | 9.600 | 3.52 | 2.727 |
| Three-chunk passage | 26.450 | 13.11 | 2.018 |
| Clone Bruno reference, first | 11.237 | 3.62 | 3.104 |
| Clone, cached repeat | 6.616 | 3.62 | 1.828 |
| Change clone reference to Bella | 11.332 | 4.20 | 2.698 |

RTF 1.93 means about 1.93 seconds of work per second of generated audio.
The native full decoder is not real-time on this tested CPU configuration.

## CUDA timings

Q8 package, full-precision speaker/S3/HiFT operations, eight host
threads, seed 1234. No `NVIDIA_TF32_OVERRIDE` or fusion override was set. Timings
exclude model/session loading; the first clone includes lazy encoder loading.
These are individual desktop runs, not isolated benchmark averages.

| Request | Wall seconds | Audio seconds | RTF |
|---|---:|---:|---:|
| Bruno, first request | 2.287 | 2.98 | 0.767 |
| Bruno, cached repeat | 2.127 | 2.98 | 0.714 |
| Switch to Bella | 3.245 | 3.52 | 0.922 |
| Three-chunk passage | 10.818 | 13.70 | 0.790 |
| Clone Bruno reference, first | 4.328 | 3.10 | 1.396 |
| Clone, cached repeat | 2.736 | 3.10 | 0.883 |
| Change clone reference to Bella | 3.832 | 4.28 | 0.895 |

Cached preset and clone requests are faster than real time in this run.
GPU utilization was 59% with 7,666 MiB in use after
the probe exited, indicating other GPU activity; this is not the model's VRAM
requirement or peak. Logs: `cuda-validated-session.log`, `cuda-final-parity.log`,
and `cuda-lm-parity.json` under `build-kitten2/`.

## Multilingual presets

The complete package adds Arabic, Chinese, French, German, Hindi, Italian,
Portuguese, Russian and Spanish. Together with 38 English voices and
`PreparedBruno`, it contains 48 prepared entries. The original 39 entries are
preserved exactly as parsed JSON values. The nine additions retain the original
upstream transcripts, 512-value NPZ speaker identities and reference codec tokens.
Native CPU preparation computes the speaker projection and full-decoder reference
conditioning using the same reference preprocessing as live cloning.

The standard-library NPZ reader matched NumPy exactly on all nine shipped assets.
Prepared speaker projections were compared with the original BF16 PyTorch
linear/layer-norm calculation: eight matched exactly and Spanish's maximum
absolute error was 0.00048828125, below the existing 0.032 component limit.

`check_multilingual.py` passed 13 cases on each backend using the final
3,282,123,776-byte multilingual GGUF and the rebuilt CUDA-capable server:

| Check | CPU | NVIDIA CUDA |
|---|---|---|
| Nine language presets plus English Bruno | Pass | Pass |
| Original reference codec tokens, reference and target BPE | Exact | Exact |
| Finite, non-silent mono 24 kHz output; generation ends before token limit | Pass | Pass |
| Chinese text split into three Unicode chunks | Pass (15.04 s audio) | Pass (13.46 s audio) |
| German reference-audio cloning | Pass | Pass |
| Chinese reference-audio cloning | Pass | Pass |

These check execution and text preservation, not pronunciation, intelligibility
or speaker similarity. No ASR scoring or native-speaker listening assessment was
performed. The remaining ten languages advertised by the upstream README have no
language presets in this snapshot and were not tested. CPU/CUDA sampling may
produce different audio; matching waveforms across backends is not required.
The run was not a controlled performance benchmark.

Reports, WAVs and traces are under
`build-kitten2-cuda/multilingual-validation-{cpu,cuda}/`. Preparation comparisons
are in `build-kitten2-cuda/multilingual-preparation-validation.json`.
The complete voice index is `build-kitten2/kitten-tts2-multilingual-voices.json`;
the GGUF is `build-kitten2/kitten-tts2-native-q8-multilingual.gguf`.
It embeds ten runtime sidecars, excluding the temporary preparation manifest.

A local server configured with all 48 voices confirmed the registration,
the multilingual model path, the rebuilt interface and the 7 GB estimate.
The model-spec test, loader/catalog synchronization check, four asset-reader
regressions, Svelte checks (zero errors/warnings) and UI/server builds passed.

## VRAM estimate

The model spec and WebUI catalog show 7 GB estimated VRAM for the native Q8
package with the full decoder. A fresh CUDA server was sampled every 200 ms
using `nvidia-smi` during a 238-character preset request and a clone request
with the same text. The server retained both requests in one session.

| Measurement | GPU memory (MiB) |
|---|---:|
| Desktop baseline before starting the server | 3,497 |
| Peak during preset speech | 8,316 |
| Peak during cloning | 9,530 |
| After stopping the server | 3,523 |

Subtracting the baseline gives approximately 4.71 GiB for preset speech and
5.89 GiB for cloning. These are sampled total-device deltas, not isolated
per-process allocations; desktop activity and brief unsampled peaks can affect
them. Seven GB allows headroom for temporary buffers and longer inputs, but is
not a guaranteed maximum or a verified minimum GPU capacity. The estimate does
not cover F32 language-model weights or every allowed chunk/reference length.
Raw measurements are in `build-kitten2-cuda/vram-measurement.json`.

## Remaining limits

- CUDA was tested on one RTX 4060 Ti. Other NVIDIA generations, other GPU
  backends, sustained-load memory stability, and worst-case VRAM remain unvalidated.
- Full default decoder only; quantized student decoders are not ported.
- Cloning needs a transcript and a 1–30 second clip; no automatic ASR or saved
  voice artifact format is provided.
- Streaming and upstream written-to-spoken normalization are not implemented.
- Broad perceptual quality and exhaustive testing of all English presets still
  need evaluation. Multilingual execution checks are recorded above; they do not
  establish pronunciation accuracy or coverage of all 20 upstream-advertised languages.
- Shared S3 arithmetic fixes are covered by the Kitten upstream component test;
  separate Chatterbox end-to-end model runs were not performed.

## Shared arithmetic and CUDA precision

The speaker encoder uses a 3x1 two-dimensional max-pooling operation because this
CUDA backend does not implement POOL_1D. CPU speaker parity remains unchanged.

TF32 arithmetic in the first SincNet convolution changes the speaker identity
embedding. F32 speaker/S3/HiFT matrix operations therefore request full precision;
the CUDA backend honors that request in single and batched cuBLAS operations and
bypasses its TF32 matrix kernel for explicit F32 requests. The same upstream
checks now pass without environment overrides. Default-precision operations keep
their previous behavior. These shared changes have not been benchmarked across
other model families.

The converter matches namespace/tensor names with slash separators. This
preserves F16 token embeddings, original BF16 speaker projections, and original
F32 speech/speaker components. Stored types were inspected directly, and the
session probe checks them before inference. Adding the multilingual presets
does not change tensor contents or types.
