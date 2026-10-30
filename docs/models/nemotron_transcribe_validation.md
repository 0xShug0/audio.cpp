# Nemotron Transcribe validation

This records a local bring-up on Windows using MSVC 14.44, CUDA 12.8.61,
RTX PRO 6000 Blackwell 96 GB and an i9-13900K. It is a small functional comparison,
not a held-out multilingual accuracy benchmark.

## Versions and settings

- Weights: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, revision `a99e33f698257b8b69ca7834f19c73582eb1fa7e`.
- Reference: `NVIDIA/NeMo-Speech.cpp` revision `66d50d91b75a406becd95fbaf39d2c9c13e95d7c`, llama.cpp submodule `bd4f514db14d87fded667787a7a963bfbaa98e89`.
- Reference build: `NEMO_SPEECH_GGML_PATCHED=OFF`, CUDA SM120, MTP/DFlash off, one sequence, batch 64, encoder batch 1, context and per-sequence pool 73,728.
- Package: IQ4_NL routed experts, Q8_0 dense decoder linears, remaining BF16/F32 tensors; F16 encoder and F32 CTC head. Decoder is about 20.37 GB; encoder is 1.67 GB.
- Port: native weight storage, four CPU threads, prefill 64, context 73,728, encoder 500/50/50 and speaker cache/FIFO/update 188/40/300.
- Greedy decoding, no repetition penalty. Output budgets matched the reference recognizer: `max(24,ceil(6*seconds-1e-9)+24)`, apart from the one-token limit check.

The port uses the exact stored BF16 Hann window and filterbank. Its frontend was
compared with the actual unmodified HF `NemotronAsrStreamingFeatureExtractor` at
Transformers revision `0c38601b69653fa2f6f4c88b287e7a8d7c57fb4e`. The 49.055-second
input had maximum log-mel error 0.000353 and relative L2 error 3.85e-7 over its
valid frames. Both frontend constants matched exactly.

## Inputs and reproduction

Short inputs were synthesized with Windows SAPI at 16 kHz, PCM16, mono, rate 0
and volume 100. Digital silence was one second. No private recordings were used.

| File | SAPI voice | Text |
| --- | --- | --- |
| `english-david.wav` | Microsoft David Desktop, English United States | The river is calm today. We will test transcription, translation, and questions about this recording. |
| `english-zira.wav` | Microsoft Zira Desktop, English United States | My train leaves at seven forty-five. Please meet me at the station after breakfast. |
| `japanese-haruka.wav` | Microsoft Haruka Desktop, Japanese | 今日は良い天気です。午後三時に駅で会いましょう。 |

For a single input, SAPI generation can be reproduced in PowerShell:

```powershell
$voice = New-Object -ComObject SAPI.SpVoice
$voice.Voice = @($voice.GetVoices()) | Where-Object { $_.GetDescription() -eq 'Microsoft David Desktop - English (United States)' } | Select-Object -First 1
$voice.Rate = 0
$voice.Volume = 100
$file = New-Object -ComObject SAPI.SpFileStream
$file.Format.Type = 18
$file.Open((Join-Path $PWD 'english-david.wav'), 3, $false)
try {
    $voice.AudioOutputStream = $file
    $voice.Speak('The river is calm today. We will test transcription, translation, and questions about this recording.') | Out-Null
} finally { $file.Close() }
```

The 56.44-second input repeats David then Zira four times, inserting 300 ms of
digital silence after each file. A separate 49.055-second input alternates the
same voices over these eight distinct turns, with 250 ms after each:

1. We left the hotel shortly after sunrise and followed the river toward the old stone bridge.
2. I bought two tickets for the museum, but the afternoon exhibition had already sold out.
3. The guide explained how the clocks were repaired, then showed us a workshop behind the courtyard.
4. For lunch we ordered vegetable soup, fresh bread, and a small pot of jasmine tea.
5. A sudden shower sent everyone under the awning, where a musician was playing a quiet tune.
6. When the rain stopped, I called my sister and arranged to meet her near the library.
7. Our train was delayed by twenty minutes, so we read the notice board and waited on platform six.
8. Before going home, please check that the kitchen window is closed and the blue bicycle is locked.

Use the [model guide](nemotron_transcribe.md) to build and package the pinned
weights. A CLI transcription with timestamps uses:

```sh
audiocpp_cli --task asr --family nemotron_transcribe --model models/audiocpp/nemotron-transcribe-iq4_nl.gguf \
  --backend cuda --threads 4 --audio english-david.wav \
  --session-option nemotron_transcribe.prefill_batch_size=64 \
  --request-option return_timestamps=true --text-out transcript.txt
```

For the native reference, use `nemo-speech transcribe` with `--word-times --format
json` and the settings above. Translation uses `--translate-to`; Q&A and language
identification use `--instruction`. Native JSON removes SOT tags from its text
and numbers speakers from one; port custom results retain the original SOT tags
and expose their Sortformer-column mapping separately.

## Observed results

One loaded CUDA session handled 11 sequential requests without exceptions. It
processed short English/Japanese transcription, English-to-French and
Japanese-to-English translation, audio Q&A, language identification, both long
dialogues, a repeat of the first input, digital silence and a one-token limit.
All nine paired text outputs matched the native reference after speaker-marker
removal and whitespace normalization. Both dialogue cases also matched the
reference's per-word SOT speaker tags. Repeating English after the longer cases
returned identical text.

Examples of matched generated output:

| Task | Output |
| --- | --- |
| Japanese ASR | 今日はいい天気です。午後三時に駅で会いましょう。 |
| Japanese to English | Today is nice weather. Let's meet at the station at 3 PM. |
| English to French | La rivière est calme aujourd'hui. Nous allons tester la transcription, la traduction et poser des questions à propos de cet enregistrement. |
| Q&A: Which three tasks are mentioned in the recording? | Transcription, translation, and questions about the recording. |
| Japanese language identification | Japanese |

The distinct dialogue included all eight turns. Both implementations omitted one
sentence pair in the four-repeat input. Silence returned empty text and an
unavailable alignment with `empty_transcript`; the forced one-token request
reported `token_limit`, not EOS.

Recorded English checks for the initial IQ4_NL package, with word timestamps:

| Input and backend | Audio duration | Processing time | Real-time factor | Matched reference words |
| --- | ---: | ---: | ---: | ---: |
| Short English, CUDA | 7.330 s | 0.556 s | 0.076 | 15/15 |
| Short English, CPU, four threads | 7.330 s | 6.250 s | 0.853 | 15/15 |
| Distinct two-speaker English, CUDA | 49.055 s | 1.232 s | 0.025 | 129/129 |

Real-time factor is processing time divided by audio duration. Model loading is
excluded; loader/session creation took 14.35 seconds separately. Reference-word
matching ignores case, punctuation and speaker markers. All four released expert
profiles also matched the same 15-word and 129-word fixtures. This small synthetic
set does not establish held-out English accuracy. Other GPU workloads were active;
the individual timings are not a throughput comparison with native CLI process
times, which include loading. The repeated-sentence omission described above and
the timestamp limitations below remain unresolved.

Word units/counts matched, but timings were not identical: maximum observed
differences were 10 ms for English, 119 ms for Japanese, 80 ms for the repeated
dialogue and 424 ms for the distinct dialogue. The largest difference follows a
native legacy-frontend speaker-activity gap at 26.66–27.02 seconds. Running the
same full encoder on identical mel inputs agrees with Torch; changing only the
frontend to native's periodic/right-aligned/+1-valid-frame policy reproduces that
exact gap. This is not a claim of bit-exact CTC parity.

Additional checks passed: 3,522 official tokenizer cases, 14 prompt ID pairs,
full real encoder components, actual CTC weights, frozen host-alignment fixtures,
native mixed-quantized decoder loading, repeated/reset calls, public BPE hook
regression, model-spec schema, silence/window policy, and GGUF package round-trip.

Full-model Metal/Vulkan, actual hour-long inference, broader multilingual quality
and a held-out accuracy benchmark remain untested. The current public packages
and conversion recipes are linked from the [model guide](nemotron_transcribe.md).

## Follow-up compatibility audit

The audit kept fixes within the new model implementation. Pre-existing code has
one necessary BPE dispatch change: an optional model-owned splitter replaces
the default split only when supplied. Existing variable names, parameters and
tokenizer defaults are preserved. Build, registration and documentation additions
do not alter other models. No GGML changes belong to this contribution.

The audit reproduced decoder allocation-failure recovery and oversized graph
configuration failures, plus a speaker-cache count overflow from extreme rates.
The model now rejects invalid bounds before graph construction, recovers from
failed allocation, and protects timestamp fallback so bad speaker activity cannot
discard a successful transcript. Self-generated tiny-network regressions exercise
the actual cache-compression, cancellation and allocation paths.

Six focused CTests passed on MSVC and WSL GCC 13.3: the BPE hook, silence/input/
residency policy, encoder bounds, decoder safety, alignment safety and model-spec
system tests. The self-generated Python GGUF package round-trip also passed.
Run the CTests from a build with `ENGINE_BUILD_TESTS=ON` and
`ENGINE_BUILD_MODEL_TESTS=ON`:

```sh
ctest --test-dir build/nemotron --output-on-failure \
  -R '^(nemotron_transcribe_.*|llama_bpe_pretokenizer_hook_test|model_spec_system_test)$'
```

A loopback CPU server with one lazy-loaded model, four threads and the same
native-storage package passed 24 HTTP checks. They covered transcription details,
translation, Q&A, Japanese language identification, multipart upload, repeated
requests, unload/reload, valid `top_p=0` sampling and eleven malformed requests.
Invalid sampling/boolean/seed/token-budget options, unknown options, arrays,
control-token injection and incompatible task/timestamp combinations returned
HTTP 400. Before the fix, invalid top-p returned 500 for speech and passed the
silence shortcut with HTTP 200. The shared server and sampler were unchanged.

English transcription and all 15 word sample spans matched the earlier CUDA
result exactly. A 48 kHz stereo copy, made by repeating each source sample three
times and duplicating the channel, returned identical words with every sample
span multiplied by three. This checks that timestamps retain the original audio
clock. CPU French translation used different wording from the earlier CUDA take;
this follow-up was not a new native-reference translation parity test.

The first CPU HTTP request took 148.3 seconds including model loading. Warm
functional requests took approximately 2.9 to 6.4 seconds. After unload/reload,
English text remained identical. Unloading reduced the server working set from
about 20.1 GiB to 182 MiB. The audit server was then stopped. An earlier CUDA HTTP
observation timed out during cold loading; that timeout was retained and was not
classified as an inference failure.

With the existing memory guard enabled at 512 MiB headroom, an F32 session override
on the actual package returned HTTP 503 before model allocation: estimated
188.42 GiB exceeded 84.18 GiB of available host memory. Estimates include both
GGUFs, expanded expert banks, mandatory F32 parameters and unaligned quant-row
fallback. The test did not load or evaluate a full F32 model. GPU headroom was
limited during this follow-up, so its full-model HTTP validation used CPU.

After merging upstream `46e3893b`, the Windows and Linux CPU CLI builds and all
six CTests passed again on source commit `b00a9ee4`. The isolated Linux build
used a private `model_specs` discovery link; no framework change was needed.
The merged CPU server repeated all 24 HTTP checks successfully. It also returned
all eight turns and 129 aligned words for the distinct 49.055-second dialogue,
with normalized text matching the earlier CUDA take. English remained identical
after this longer request. The model was unloaded and the owned server stopped.
