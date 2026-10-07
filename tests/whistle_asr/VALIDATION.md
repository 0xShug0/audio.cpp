# Whistle ASR validation report

## Scope

- Move Whistle sources and headers into the community-model directories and namespace.
- Preserve the public `whistle_asr` family, schema-v1 specification, and spec-backed loader.
- Add community catalog entries, setup instructions, limitations, and source-specific licensing records.
- Keep model packages empty. No weights, extracted assets, logs, or recordings ship in this change.

Affected areas are `src/community_models/whistle_asr/`,
`include/engine/community_models/whistle_asr/`, `tools/community_models/whistle_asr/`,
`tests/whistle_asr/`, `model_specs/whistle_asr.json`, loader registration,
`CMakeLists.txt`, `README.md`, and ASR/community/licensing documentation.

Preparation follows [CONTRIBUTING.md](../../CONTRIBUTING.md), the
[community guidance](../../docs/community_models/models.md), and
[issue #54](https://github.com/0xShug0/audio.cpp/issues/54).

## Model setup and build

Use the published FP32 checkpoint and `.cact` assets at model revision
`b358ddadd89b7a713b5aa131f23032d3cca1b251`. Put `whistle.safetensors`,
`config.json`, and `LICENSE` in a dedicated source directory. Keep `.cact` outside it.
The converter verifies the pinned input hashes, extracts four assets, and embeds
those assets, configuration, license, and the current model specification.

The following exact PowerShell commands ran from the repository root unless noted.
The existing `build/whistle` configuration used Ninja, MSVC Release, CPU AVX2,
`GGML_NATIVE=OFF`, OpenMP, deployment mode, custom model set `whistle_asr`,
and `ENGINE_BUILD_TESTS=ON`. This configuration reused existing dependencies.

```powershell
python tools/check_loader_catalog_sync.py --self-test
python tools/check_loader_catalog_sync.py
& 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
cmake --build build/whistle --target audiocpp_cli audiocpp_gguf whistle_frontend_test whistle_assets_test --parallel 16
New-Item -ItemType Directory -Force build/whistle/pr-prep/source,build/whistle/pr-prep/model,build/whistle/pr-prep/standalone
Copy-Item -LiteralPath C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/source/whistle.safetensors,C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/source/config.json,C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/source/LICENSE -Destination build/whistle/pr-prep/source
python tools/community_models/whistle_asr/convert.py --source build/whistle/pr-prep/source --cact C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/whistle.cact --converter build/whistle/bin/audiocpp_gguf.exe --output build/whistle/pr-prep/model/whistle-f32.gguf
build/whistle/bin/audiocpp_gguf.exe --inspect build/whistle/pr-prep/model/whistle-f32.gguf
cmake -S . -B build/whistle -DAUDIOCPP_WHISTLE_TEST_MODEL=C:/Users/adityasharma/Projects/audio.cpp/build/whistle/pr-prep/model/whistle-f32.gguf
cmake --build build/whistle --target audiocpp_cli audiocpp_gguf whistle_frontend_test whistle_assets_test --parallel 16
ctest --test-dir build/whistle -R whistle_ --output-on-failure
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --backend cpu --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --audio assets/resources/sample_16k.wav --language en --metrics --log
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --backend cpu --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --audio C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/german_16k.wav --language de --metrics --log
```

## Observed validation

Measured on 2026-10-06, local Pacific time.

- Catalog self-test passed both tests. Catalog synchronization passed with 115 loaders and 113 specs.
  Existing unrelated Safetensors package warnings remained; no unrelated fixes were made.
- Both builds succeeded. Reconfiguration initially ran without the MSVC environment and failed with
  `LINK : fatal error LNK1104: cannot open file 'kernel32.lib'`. Repeating after
  `Launch-VsDevShell.ps1` succeeded. Existing dependency/compiler warnings remained.
- Fresh conversion succeeded. Inspection reported 116 tensors, six embedded sidecars,
  an embedded model specification, and `model_spec_family=whistle_asr`.
- CTest passed `whistle_frontend_test`, `whistle_asr_integration`, and
  `whistle_asr_full_integration`, 3/3 in 3.59 seconds, using the freshly converted GGUF.
  Frontend vectors are independent feature checks, not full-model FP32 reference parity.
- Integration exercises decoder reuse, quiet speech, constant-RMS speech, digital silence,
  the 30-second limit, and non-finite input rejection. Decode exhaustion is inspected in code
  and explicitly throws; it was not forced in an executed regression test.
- Actual CLI ASR passed with forced English and German. A separate same-session CLI batch
  detected English twice and produced identical transcripts.
- CLI digital silence produced `text_output=`. A 480001-sample, 16 kHz mono recording failed
  with `Whistle audio exceeds the 30-second input limit`.
  Streaming failed with `Whistle currently supports offline ASR only`.

Exact CLI limit commands used the ignored generated input script:

```powershell
python build/whistle/pr-prep/create-inputs.py
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --model build/whistle/pr-prep/model/whistle-f32.gguf --audio build/whistle/pr-prep/silence.wav --threads 4
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --model build/whistle/pr-prep/model/whistle-f32.gguf --audio build/whistle/pr-prep/over-limit.wav --threads 4
build/whistle/bin/audiocpp_cli.exe --task asr --mode streaming --family whistle_asr --model build/whistle/pr-prep/model/whistle-f32.gguf --audio assets/resources/sample_16k.wav --threads 4
```

The combined command invocation exited 1 after the expected streaming rejection.
Individual exit codes were not captured. Successful generation/silence and failed duration/streaming
are established by the generated inputs and actual CLI output above.

The standalone model directory contained only `whistle-f32.gguf` (220831648 bytes).
With a fresh repository-local `TEMP` cache, CLI reconstruction extracted `config.json`,
`LICENSE`, `tokenizer.blob`, `mel_filterbank_80.f32`, `hadamard_perm1.f32`, and
`hadamard_perm2.f32`. Extracted files matched the conversion source hashes.
CLI recognition also succeeded from `C:/Windows`, without adjacent sidecars or a working-directory
`model_specs` tree. Source inspection of `find_contract_spec_path()` confirms that the embedded
v1 contract takes precedence over workspace and builtin fallback.
An additional metadata check measured the new community documentation path inside the GGUF,
no stale core documentation path, and matching hashes for all six extracted sidecars.

```powershell
python build/whistle/pr-prep/check_embedded_assets.py | Tee-Object build/whistle/pr-prep/embedded-assets-check.log
```

This exited successfully with seven `PASS` lines. The local check script remains ignored.

```powershell
# Run with working directory C:/Windows. All writes remain in the ignored build directory.
$env:TEMP='C:/Users/adityasharma/Projects/audio.cpp/build/whistle/pr-prep/standalone'
$env:TMP=$env:TEMP
& C:/Users/adityasharma/Projects/audio.cpp/build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --backend cpu --threads 4 --model C:/Users/adityasharma/Projects/audio.cpp/build/whistle/pr-prep/model/whistle-f32.gguf --audio C:/Users/adityasharma/Projects/audio.cpp/assets/resources/sample_16k.wav --language en --metrics --log
```

## Transcript comparison and reference limitations

Measured inputs are 16 kHz mono WAVs. English is the complete repository sample, 14.0719 seconds.
German is the existing local `german_16k.wav`, 2.11594 seconds; its recording provenance is not established.
Their SHA-256 hashes are respectively
`a8096a0c2ac136bcaf70143dc80ba576ade8dfbf7f9f30ecdec29b920ab3e3cb` and
`82c41f2dd56e21fb9407b9d4abf7692d9db8806b41e7134607ea4748411803b1`.

FP32 audio.cpp English output was:

```text
Some call me Nature. Others call me Mother Nature. I've been here for over 4.5 billion years, twenty two thousand five hundred times longer than you.
```

The local quantized official `.cact` reference produced:

```text
Some call me Nature. Others call me Mother Nature. I've been here for over 4.5 billion years, 22,500 times longer than you.
```

Both produced this German transcript:

```text
Guten Morgen, wie wird das Wetter heute?
```

Exact reference commands used automatic language detection, four threads, full/default audio decoder
depth, and no keywords, timestamps, or streaming:

```powershell
& C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/needle.exe --help
& C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/needle.exe --model C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/whistle.cact --audio assets/resources/sample_16k.wav --threads 4
& C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/needle.exe --model C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/whistle.cact --audio C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/german_16k.wav --threads 4
```

The `.cact` hash matches the pinned converter asset hash
`b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb`.
The local executable SHA-256 is
`e8863ca0c06a47d406777077f1ba728b58e57d77fedff252dbe6827d588acc8c`.
Its source revision cannot be established from the available files.
The audio.cpp decoder uses greedy selection, a 320-text-token ceiling, and no sampling.
This is a transcript comparison with quantized weights, not exact FP32 numerical parity.
No FP32 reference runtime or seven-language validation corpus was available.

## Performance and process memory

Measured hardware is AMD Ryzen AI MAX+ 395, 16 cores/32 logical processors.
OS is Windows 11 Enterprise Insider Preview, build 26704.
Compiler is MSVC 19.51.36257.0. Final configure used CMake 4.4 and Ninja 1.13.2.
These are local observations, not a performance guarantee.

- Four-thread CLI English request took 319.928 ms for 14.0719 seconds, RTF 0.0227352.
  Peak process working set was 446 MiB; reported peak Windows commit was 232 MiB.
- Four-thread CLI German request took 78.3542 ms for 2.11594 seconds, RTF 0.0370305.
  Peak working set was 432 MiB; reported peak Windows commit was 219 MiB.
- One loaded CLI session ran two identical English requests.
  First request took 467.397 ms, RTF 0.0332149, peak working set 447 MiB.
  Repeat took 479.373 ms, RTF 0.0340659, peak working set 448 MiB.
  These are first/repeated requests, not OS-cache-cold measurements.
- An ignored one-thread scratch probe using the existing test objects separately measured
  asset loading at 23.7752 ms and runtime initialization at 150.822 ms.
  Inference then took 640.984 ms and 661.587 ms, RTF 0.0455505 and 0.0470146.
  `GetProcessMemoryInfo` reported peak working set 467447808 bytes for both requests.
  The scratch probe initially requested four threads from test objects built without OpenMP.
  That unsupported probe failed; the successful probe uses one thread. Production CLI uses OpenMP.

CLI metrics time preparation/inference after model/session creation and input loading.
They exclude model loading. Scratch load timings are separate and use a different one-thread harness.
RTF is inference seconds divided by input-audio seconds. All measured RTFs are below 1.0.
Memory figures describe CPU process memory, not GPU memory.

```powershell
$batch = @(@{id='cold';audio='C:/Users/adityasharma/Projects/audio.cpp/assets/resources/sample_16k.wav'},@{id='repeat';audio='C:/Users/adityasharma/Projects/audio.cpp/assets/resources/sample_16k.wav'})
$batch | ConvertTo-Json | Set-Content build/whistle/pr-prep/batch.json
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --backend cpu --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --request-sequence build/whistle/pr-prep/batch.json --metrics
```

The scratch source, compile logs, exact probe command, model, recordings, input/sidecar hashes,
and run logs remain under ignored `build/whistle/pr-prep/`; they are not PR source artifacts.
The successful probe command is preserved in `benchmark-command.ps1`, executed from `build/whistle`.
Original cached inputs remain unchanged under
`C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/`.

## Provenance, licensing, and limitations

See [provenance](../../tools/community_models/whistle_asr/PROVENANCE.md),
[license table](../../docs/model_licenses.md), and
[model guide](../../docs/community_models/whistle_asr.md).
The converter preserves the pinned upstream `LICENSE` inside the GGUF.
The repository license does not license the weights or extracted assets.
The contributor confirms that no code was copied or adapted from reference implementations.
Independent extracted-asset provenance remains unverified in the upstream release.
No independently audited redistribution rights or maintainer approval is claimed.

Backend coverage is offline CPU and Metal with FP32 weights; see the
[GGML encoder section](#ggml-encoder-rewrite-2026-10-07) below. Supported language codes are
`en`, `de`, `fr`, `es`, `it`, `nl`, `pl`. English and German inputs above were checked.
The other five codes each have one [synthetic CLI spot check](MULTILINGUAL_VALIDATION.md),
with ten successful automatic/forced-language runs and zero normalized prompt word edits.
The quantized reference matches four clips; Polish differs by `niebo` versus
`nie bo`. This is documented without a parity claim.
Their measured four-thread inference RTF ranges from 0.0392522 to 0.0570413;
peak process working set ranges from 432 to 434 MiB. These are not a broad accuracy corpus.
Inputs must be 16 kHz mono, at most 30 seconds. There is no streaming, resampling,
long-form recognition, GPU inference, quantization, keyword biasing, or word timestamps.
Noise-only audio may produce words. This family is not a speech-activity detector.
No downloadable package is advertised.

## Pre-submission adversarial review and regression checks

A fresh-context review of the port found a byte-fallback decoding defect.
Invalid byte tokens could return invalid UTF-8, and byte-composed metaspace
markers were not converted to spaces. The model-local decoder now concatenates
visible bytes, applies UTF-8 replacement decoding, then replaces metaspace globally.
No framework or GGML code changes are included.

`whistle_tokenizer_test` adds literal regressions for valid multibyte fallback,
invalid and incomplete UTF-8, overlong/surrogate/out-of-range sequences,
byte-composed metaspace, special-token handling, malformed hex, and ID bounds.
The focused re-review found no remaining Critical or Important issue.

The following checks ran after the fix, in a VS developer shell where applicable:

```powershell
cmake -S . -B build/whistle
cmake --build build/whistle --target whistle_tokenizer_test whistle_frontend_test whistle_assets_test audiocpp_cli --config Release -j 4
build/whistle/bin/whistle_tokenizer_test.exe
ctest --test-dir build/whistle -R whistle_ --output-on-failure
python -X utf8 build/whistle/pr-prep/tokenizer-fix/run_cli_regressions.py | Tee-Object build/whistle/pr-prep/tokenizer-fix/cli-regressions.log
```

Measured results are a successful build, tokenizer-test pass, and CTest **4/4**.
All ten French/Spanish/Italian/Dutch/Polish CLI requests exit 0 and produce
exactly the same transcripts as the earlier recorded automatic/forced runs.
New logs remain ignored under `build/whistle/pr-prep/tokenizer-fix/`.

The reviewer also exercised the advertised extracted-Safetensors directory
through the production CLI with CPU, one thread, English, and the full sample.
It exits 0 with the expected English transcript, 786.915 ms inference wall time,
and peak process working set 447 MiB. This is an additional load-route check,
not a before/after performance comparison.

## GGML encoder rewrite (2026-10-07)

Maintainer review asked for framework-module reuse and GGML graphs. This revision moves
the mel frontend onto `NemoMelFrontend` and the whole encoder (stem, eight mHC layers,
final norm, positional table, cross-attention key/value projections) into one GGML graph
built from `Conv2dModule`, `DepthwiseConv2dModule`, `DepthwiseConv1dModule`,
`LinearModule`, `GemmaRMSNormModule`, `RoPEModule`, `SiluModule`, and `GLUModule`.
Whistle-specific math (mHC lane mixing with Sinkhorn normalization, the conditioned
Kronecker-factored Hadamard MLP with fixed permutations, and 48/64-wide grouped-query
attention) stays in `src/community_models/whistle_asr/encoder.cpp`. The autoregressive
decoder still runs on the host. The converter and GGUF layout are unchanged; the loader
transposes the JAX `[in, out]` kernels into the backend weight store.

Measured on macOS 27.0.1, Apple M3 Ultra, Apple clang, CMake 4.4.4, Ninja, Release,
`ENGINE_ENABLE_METAL=ON`, `GGML_METAL=ON`, OpenMP on, deployment build, custom model set
`whistle_asr`, `ENGINE_BUILD_TESTS=ON`. Exact commands from the repository root:

```bash
scripts/build_metal.sh --build-dir build/whistle --build-type Release --openmp auto --with-tests --deployment-build --model-set custom --models whistle_asr --target audiocpp_cli --target audiocpp_gguf --target whistle_frontend_test --target whistle_tokenizer_test --target whistle_assets_test
python3 tools/community_models/whistle_asr/convert.py --source build/whistle/pr-prep/source --cact /private/tmp/whistle-src/whistle.cact --converter build/bin/audiocpp_gguf --output build/whistle/pr-prep/model/whistle-f32.gguf
cmake -S . -B build/whistle -DAUDIOCPP_WHISTLE_TEST_MODEL=$PWD/build/whistle/pr-prep/model/whistle-f32.gguf
ctest --test-dir build/whistle -R whistle_ --output-on-failure
build/whistle/bin/whistle_frontend_test build/whistle/pr-prep/source/mel_filterbank_80.f32 build/whistle/pr-prep/oracle/sample_16k.f32
build/whistle/bin/whistle_assets_test build/whistle/pr-prep/model/whistle-f32.gguf assets/resources/sample_16k.wav --reference build/whistle/pr-prep/oracle --backend cpu --threads 4
build/whistle/bin/whistle_assets_test build/whistle/pr-prep/model/whistle-f32.gguf assets/resources/sample_16k.wav --reference build/whistle/pr-prep/oracle --backend metal
build/whistle/bin/audiocpp_cli --task asr --family whistle_asr --backend metal --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --audio assets/resources/sample_16k.wav --metrics
python3 tools/check_loader_catalog_sync.py --self-test && python3 tools/check_loader_catalog_sync.py
```

The locally converted GGUF has SHA-256
`80c42e88a1ad07203634b60d50e0763e0478bb6830544434b70f52ab53c9ab42` (220831648 bytes).
The hosted `bumpyclock/whistle-GGUF` file, SHA-256
`1167c811abad2fc539a1116e1b7031ba4a07c8f5b2e75ab3a389ae370efede8b`, was converted from an
earlier spec revision; it loads and produces the same English transcript on CPU and Metal.

### Numerical comparison against the host implementation

Reference tensors were dumped from the previous host-only implementation at commit
`6e0a40c4` through a temporary hook on the full 14.0719-second repository sample
(1407 mel frames, 176 encoder frames). That hook is not committed, so the dumps are
reproducible only by checking out that commit; they stay in ignored
`build/whistle/pr-prep/oracle/`. `whistle_assets_test --reference` reports the largest
absolute difference as a fraction of each reference tensor's largest magnitude:

| Tensor | Reference max | CPU fraction | Metal fraction |
|---|---:|---:|---:|
| mel features (1407 x 80) | 3.42311 | 6.37e-06 | 6.37e-06 |
| encoder memory (176 x 512) | 5.55704 | 6.01e-07 | 3.03e-04 |
| cross-attention keys, layer 0 | 7.92314 | 1.66e-06 | 1.20e-03 |
| cross-attention values, layer 0 | 86.5275 | 1.32e-06 | 5.06e-04 |
| cross-attention keys, layer 7 | 9.34248 | 1.35e-06 | 1.08e-03 |
| cross-attention values, layer 7 | 197.274 | 8.99e-07 | 3.51e-04 |

Per-layer probes during bring-up showed the Metal drift appears already after the stem
(2.5e-04 of scale) and stays between 1e-04 and 4e-04 of scale through the eight layers;
setting `GGML_PREC_F32` on every matmul did not change the Metal result, so it is not
matmul accumulation precision. The CPU graph tracks the host implementation to about
1e-06 of scale. `whistle_frontend_test` passes both its fixed vectors and the official
filterbank check on the real sample (tolerance 0.02). The default `--reference`
tolerance is 2e-3, which Metal passes with a 1.7x margin on the worst tensor.

Whistle's mHC residual logits reach gaps above 200 nats inside one 4x4 Sinkhorn block
(layer 1), so a float softmax underflows to exact zeros. The graph clamps the softmax to
`FLT_MIN` before the log; a float64 Sinkhorn with and without that clamp differs by at
most 9e-07 on all eight layers of the sample.

### Transcripts and tests

CTest passed 4/4 (`whistle_frontend_test`, `whistle_tokenizer_test`,
`whistle_asr_integration`, `whistle_asr_full_integration`). The full-sample integration
checks (expected transcript, decoder reuse, quiet and constant-RMS speech, silence,
30-second rejection, non-finite rejection) also pass with `--backend metal`. The English
sample transcript is unchanged on CPU with 1 and 4 threads and on Metal:

```text
Some call me Nature. Others call me Mother Nature. I've been here for over 4.5 billion years, twenty two thousand five hundred times longer than you.
```

Seven longer synthetic clips (20 to 26 seconds, one per supported language) produce
byte-identical transcripts across CPU 1 thread, CPU 4 threads, Metal, automatic and
forced language; see [the multilingual report](MULTILINGUAL_VALIDATION.md#longer-seven-language-clips-2026-10-07).

### Performance and process memory

CLI `metrics.wall_ms` covers preparation plus inference per request and excludes model
loading. Encoder-only times come from `whistle_assets_test --reference` (graph build,
allocation, upload, compute, readback) for the 176-frame sample; "first" includes
backend warm-up, "warm" is the second call. Single local observations, not guarantees.

| Measurement | Previous host code (same machine, 2026-10-07 baseline) | GGML encoder revision |
|---|---:|---:|
| Sample, CLI wall, CPU 1 thread | 579.7 ms (RTF 0.0412) | 479 ms (RTF 0.0341) |
| Sample, CLI wall, CPU 4 threads | 280.7 ms (RTF 0.0199) | 224 ms (RTF 0.0160) |
| Sample, CLI wall, Metal | not supported | 145 to 148 ms (RTF 0.0105) |
| Encoder only, CPU 1 thread | not measured separately | 313 ms first, 291 ms warm |
| Encoder only, CPU 4 threads | not measured separately | 103 ms first, 89 ms warm |
| Encoder only, Metal | not supported | 65 to 110 ms first, 29 ms warm |
| Sample, peak RSS | 615 to 619 MiB | 749 MiB Metal, 757 MiB CPU |

On Metal the encoder is now about 29 ms of a 145 ms sample request, so the host decoder
is the dominant remaining cost. Peak RSS grew because encoder weights now live in the
backend weight store while the decoder keeps its host copies.

Five identical English requests (20.43 s clip) in one loaded CLI session:

| Backend | Wall per request | Peak RSS | Peak footprint |
|---|---|---|---|
| CPU 4 threads | 392.6, 385.5, 385.0, 379.3, 379.9 ms | 770 to 774 MiB | 537 to 541 MiB |
| Metal | 275.6, 252.5, 262.2, 261.7, 261.0 ms | 746 to 748 MiB | 802 to 803 MiB |

```bash
build/whistle/bin/audiocpp_cli --task asr --family whistle_asr --backend metal --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --request-sequence build/whistle/pr-prep/longform/batch5.json --metrics
```

Catalog self-tests passed 2/2; synchronization passed. No framework, GGML, converter, or
GGUF-format changes are included. CUDA, HIP, and Vulkan remain unvalidated and are
rejected at session creation.

## Readiness checklist

- [x] Community layout, namespace, registration, build paths, and discoverability updated.
- [x] Fresh self-contained GGUF embeds the updated specification and license/assets.
- [x] Catalog checks, CPU builds, frontend, and opt-in integration tests passed.
- [x] GGML encoder matches the host implementation on CPU (about 1e-06 of scale) and Metal (at most 1.2e-03 of scale); macOS CTest 4/4 and Metal integration checks pass.
- [x] Actual CLI English/German, silence, duration rejection, streaming rejection, and session reuse checked.
- [x] Local latency, RTF, and CPU process memory recorded without excluding slow requests.
- [x] Final diff and untracked-file list inspected. No generated model/log/audio artifacts enter source changes.
- [x] Relative documentation links resolve, old core paths are absent, and `git diff --check` passes.
- [ ] Resolve documented source-specific licensing uncertainties before redistribution.
- [ ] Establish the local reference executable's exact source revision for a revision-identified comparison.
- [ ] Obtain a pinned FP32 reference runtime before claiming FP32 numerical parity.
- [x] French, Spanish, Italian, Dutch, and Polish synthetic spot checks completed through the actual CLI.
- [ ] Broaden native-speaker and real-recording coverage; current evidence is not a general accuracy benchmark.
- [ ] Force decode exhaustion in a focused behavioral test if required by reviewers; currently only code-inspected.

These validation runs did not publish model files or download large dependencies.
The report records preparation evidence before PR submission.
