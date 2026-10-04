# Kitten TTS 2 native validation record

Powered by Stellon Labs.

## Reviewed scope and prerequisite

Validated locally on 2026-10-03 after removing the shared precision changes.
The Kitten implementation depends on
[Chatterbox correctness PR #778](https://github.com/0xShug0/audio.cpp/pull/778),
commit `f96a8e78a923aabd4874f06eef60b170358fdc80`. That separate change corrects
relative-shift padding and the attention/feed-forward LayerNorm epsilon.

The numerical and synthesis results below use the cleaned Kitten runtime
together with that prerequisite. The starting Kitten checkout was `c41cc827`
on base `2ba9fa80`; the shared CUDA, S3 and HiFT precision edits and Kitten's
speaker precision override were removed before testing. The three correctness
hunks from #778 were applied for the builds and then removed from the Kitten
working tree. These results do not claim that the Kitten branch on its original
base passes encoder parity without #778.

The final Kitten diff contains no changes to `src/framework/`,
`include/engine/framework/`, `external/ggml/`, or `src/models/chatterbox/`.
Its implementation and headers live in `src/community_models/kitten_tts2/`
and `include/engine/community_models/kitten_tts2/`. Model-specific tests,
documentation, specification and preparation tools accompany it.

Changes outside those model-specific files are limited to:

- `CMakeLists.txt`: model registration and Kitten preparer/probe targets.
- Shared documentation indexes: entries linking to Kitten's model guide.
- `webui/configs/models_catalog.json`: the Kitten model-manager entry.
- `webui/native/dist/index.html`: the generated interface containing that
  catalog entry; the WebUI imports the catalog at build time.

## Environment and assets

- Windows x64, Visual Studio 2022 Build Tools, MSVC 19.44.35228.0, Release.
- AMD Ryzen AI 9 HX 470, 12 physical cores, approximately 52 GiB RAM.
- NVIDIA RTX 4060 Ti, 16 GiB VRAM, compute capability 8.9, driver 610.88,
  CUDA Toolkit 13.3.73. Eight host inference threads.
- Python 3.12.10 and PyTorch 2.13.0 for offline reference comparisons.
- KittenTTS source `226cb35`; Kitten model snapshot
  `baa41e5d2c5f64be0095365672a7858542261271`.
- S3 checkpoint from ResembleAI/chatterbox-turbo revision
  `749d1c1a46eb10492095d68fbcf55691ccf137cd`, SHA-256
  `d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f`.

The tested [published multilingual GGUF](https://huggingface.co/dignome/kitten_tts2)
is 3,282,123,776 bytes, SHA-256
`e97920ca5053f9fcd4de638dcd8114ed2510d4291a93257473a8843c3ff349ad`.
It contains Q8_0 LM projections, F16 embedding/head, original BF16 speaker
projection, and F32 S3/speaker weights. The package has 48 prepared voice entries,
three tensor namespaces and ten runtime sidecars, including licenses/NOTICE.

The full inference path is native C++/GGML. Python, PyTorch, TorchScript and
ONNX Runtime are not runtime dependencies. The default `decoder.pt` export is
used only by the Python comparison script.

## Build and test setup

The existing CUDA build was reconfigured/rebuilt with these settings (PowerShell
line continuations), with #778 applied in the validation checkout:

```powershell
cmake -S . -B build-kitten2-cuda -G "Visual Studio 17 2022" -A x64 `
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=kitten_tts2 `
  -DENGINE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DENGINE_ENABLE_LLAMAFILE=OFF -DENGINE_BUILD_TESTS=ON `
  -DCMAKE_SUPPRESS_REGENERATION=ON
cmake --build build-kitten2-cuda --config Release --parallel 8 `
  --target audiocpp_cli audiocpp_server audiocpp_gguf `
    audiocpp_kitten_tts2_prepare_voices kitten_tts2_session_probe `
    kitten_tts2_components_probe model_spec_system_test
```

Binaries are in `build-kitten2-cuda/bin/Release/`. The same build supports CPU
and CUDA selection. Both backends were exercised by the native session probe;
the server language checks used CUDA. Commands and reference assets are
documented in [README.md](README.md).

After moving all Kitten headers from `src/` into `include/`, both the CUDA
build above and the existing CPU-only `build-kitten2-pr/` build were reconfigured
and rebuilt successfully for the same seven targets. Both model-spec tests
passed. These builds also included #778 temporarily; its source changes were
then removed from the Kitten working tree. The move changes include paths and
removes the preparer's private `src/` include directory without changing the
inference logic covered by the numerical runs.

For strict CUDA component parity, `NVIDIA_TF32_OVERRIDE=0` was set before
launching both the session probe that captured speaker intermediates and the
component probe. All ordinary CPU/CUDA synthesis runs and the multilingual
server run used an environment with this override removed. The production
implementation leaves shared precision and dispatch policy unchanged.

## Passed checks after precision cleanup

- Release build of CLI, server, converter, voice preparer, both native probes
  and model-spec test.
- Model-spec test, loader/catalog synchronization and four preparation tests.
- On both CPU and CUDA: packaged tensor types, exact seeded repeats, preset
  switching, three-chunk synthesis, native cloning, cached clones and changed
  references; rejection of unknown voices, missing transcripts and short clips.
- LM parity: 301 exact prompt tokens; maximum logit error 0.253082, mean
  0.052854, cosine 0.99999946; matching top speech token 2957. The controlled
  first preset produced 72 codes and 71,520 waveform samples.
- All six controlled CUDA component comparisons, and CPU S3/HiFT comparisons.
- All 13 CUDA multilingual server cases, with 48 registered voices:
  English Bruno, nine additional language presets, Chinese chunking, and
  German/Chinese cloning. Reference codec tokens and reference/target BPE
  matched the upstream assets. Audio was finite, non-silent mono 24 kHz,
  and generation ended before the token limit. The temporary server was stopped.

## Controlled component results

The CPU and CUDA decoder comparisons below use the same speech codes and
weights. Random draws are fixed to zero for the meanflow solver; HiFT receives
identical native mel within each backend comparison, with zero phase/noise draws.
CUDA comparisons use the test-only TF32 setting above.

| Component | CPU maximum error | CUDA maximum error |
|---|---:|---:|
| XVectorSincNet normalized identity | Not repeated in this pass | 0.000003476 |
| BF16 speaker projection, same identity input | Not repeated in this pass | 0 |
| S3 encoder hidden states | 0.000001907 | 0.000002146 |
| Two-step meanflow mel, zero noise | 0.000592232 | 0.003253222 |
| HiFT pitch, same native mel | 0.000000298 | 0.000000268 |
| HiFT waveform, same mel and zero random draws | 0.000013851 | 0.000015877 |

These pass the unchanged tolerances documented in the test README. The waveform
comparison excludes the first 960 samples where the upstream wrapper applies
its fade. Speaker parity uses the exact native-resampled input, so it does not
measure equality with upstream librosa/soxr preprocessing.

A CUDA ablation with identical inputs measured encoder maximum errors of
0.318668097 without the correctness fixes, 0.000009358 with the relative-shift
fix only, and 0.000002146 with both fixes. The original code fails the 0.0001
encoder tolerance; the shift-only variant passes, with the matching epsilon
further reducing error. This evidence is also recorded in #778.

## Session timings without a TF32 override

The published Q8 package, eight threads and seed 1234 were used. Model/session
loading is excluded; the first clone includes lazy reference-encoder loading.
Tracing was enabled. These are individual desktop measurements with other
activity, not isolated benchmark averages or a performance comparison.

| Request | CPU wall s | CPU audio s | CPU RTF | CUDA wall s | CUDA audio s | CUDA RTF |
|---|---:|---:|---:|---:|---:|---:|
| Bruno, first request | 8.992 | 3.10 | 2.901 | 1.199 | 2.98 | 0.402 |
| Bruno, cached repeat | 5.531 | 3.10 | 1.784 | 0.963 | 2.98 | 0.323 |
| Switch to Bella | 9.168 | 3.52 | 2.604 | 1.323 | 3.52 | 0.376 |
| Three-chunk passage | 25.500 | 13.11 | 1.945 | 4.802 | 13.70 | 0.351 |
| Clone Bruno reference, first | 10.710 | 3.62 | 2.959 | 2.535 | 3.50 | 0.724 |
| Clone, cached repeat | 6.636 | 3.62 | 1.833 | 1.196 | 3.50 | 0.342 |
| Change clone reference to Bella | 10.783 | 4.20 | 2.567 | 1.541 | 3.80 | 0.406 |

RTF is wall time divided by generated audio duration. CPU and CUDA can generate
different speech codes and durations. These checks establish execution and
repeatability within a backend; they do not measure perceptual quality.

## Language coverage and earlier packaging checks

The package provides Arabic, Chinese, French, German, Hindi, Italian,
Portuguese, Russian and Spanish presets, plus 38 English voices and
`PreparedBruno`. The 39 original prepared entries are preserved as parsed JSON;
the nine additions retain upstream transcripts, NPZ identities and codec tokens.

Earlier packaging checks compared all nine NPZ assets with NumPy and the BF16
speaker projections with PyTorch. Eight projections matched exactly; Spanish's
maximum error was 0.00048828125, below the 0.032 tolerance. Earlier CPU and CUDA
server runs also passed 13 multilingual cases. The current cleanup repeats the
CUDA server cases; it does not claim a new CPU multilingual server run.

Earlier Svelte checks and interface builds passed. The header/precision cleanup
does not change the UI catalog or its generated bundle. Existing CLI/server
import checks found no Torch, Python or ONNX Runtime DLL dependency.

## VRAM estimate (earlier measurement)

The model spec and WebUI retain the 7 GB estimate for the native Q8 package.
This estimate came from the earlier implementation with shared precision edits,
before the cleanup. It has not been remeasured after those edits were removed.

A server was sampled every 200 ms during a 238-character preset request and
a clone request with the same text, retaining both in one session:

| Measurement | Total GPU memory (MiB) |
|---|---:|
| Desktop baseline before starting the server | 3,497 |
| Peak during preset speech | 8,316 |
| Peak during cloning | 9,530 |
| After stopping the server | 3,523 |

The baseline-subtracted deltas were approximately 4.71 GiB for preset speech
and 5.89 GiB for cloning. These are sampled total-device deltas, not isolated
per-process allocations. Seven GB provides guidance with headroom, not a
guaranteed maximum or verified minimum for all inputs and weight types.

## Evidence and limits

Current logs, reports, WAVs and traces remain local under the ignored
`build-kitten2-cuda/review-split/` directory: `results.json`,
`combined-build.log`, `cpu-session.log`, `cuda-session.log`,
`controlled-cuda-reference.log`, `both-fixes-cpu-reference.log`,
`lm-parity.json`, and `multilingual/report.json`. Baseline comparison failures
in this directory are intentional ablations. No weights or generated audio are
committed. Earlier logs remain in `build-kitten2-pr/` and `build-kitten2-cuda/`.
The header-move rebuild results are in `header-rebuild.json` and the
`headers-{cpu,cuda}-{configure,build,model-spec}.log` files in the same directory.

- #778 must be incorporated before validating the final integrated branch.
- CUDA was tested on one RTX 4060 Ti; other GPU backends, NVIDIA generations,
  sustained-load stability and worst-case memory remain unvalidated.
- Only the full default decoder is ported; student decoders are unsupported.
- Cloning requires a transcript and a 1–30 second clip. Automatic ASR,
  streaming and upstream written-to-spoken normalization are not implemented.
- Language checks establish execution and text preservation. Pronunciation,
  intelligibility, speaker similarity, every English voice, and the ten other
  upstream-advertised languages have not been comprehensively evaluated.
- The component harness exercises reused Chatterbox S3 arithmetic. Separate
  end-to-end Chatterbox speech/voice-conversion quality tests were not performed.
