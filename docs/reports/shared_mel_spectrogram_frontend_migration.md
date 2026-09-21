# Shared mel spectrogram frontend migration

`MelSpectrogramFrontend` owns the waveform-to-mel sequence: optional channel
mixdown and resampling, waveform padding, windowed STFT, cached mel filterbank,
magnitude or power projection, log and dynamic-range scaling, frame trimming
and padding, and output layout. A model supplies `MelSpectrogramFrontendConfig`
and calls `extract_audio`, `extract_mono`, or `extract_planar`. Numerical choices
are explicit settings; the shared code does not dispatch on model names.

`ReferenceAudioFrontend` owns the **complete reference-audio feature pipeline**
shared by IndexTTS2 and Confucius4-TTS: channel mixdown, optional input
truncation, mel and speaker resampling, S2 log mel, CAMPPlus fbank, and
semantic fbank with normalization, frame pairing, and attention mask. The
models now configure their different resampling routes and adapt the shared
output to their existing public result types. Their `audio_features.cpp` files
are 71 and 64 lines, respectively. Model-specific reference selection, gain,
repetition, and stereo duration handling remain outside the common pipeline.

The 16 kHz CAMPPlus path now uses one cached float Kaldi mel bank and the
framework's cached Povey window across IndexTTS2, Confucius4-TTS, Seed-VC,
DotTTS, Chatterbox, and GLM-TTS. All six use the shared CAMPPlus extraction,
removing their duplicate framing, preemphasis, STFT, projection, and mean
subtraction. Their resampling and short-input behavior stay model-owned.
GLM-TTS now calls the framework's interleaved-to-mono Torchaudio resampler
directly for CAMPPlus, removing a single-use local wrapper while preserving
its float64 kernel setting and input validation.
Kroko ASR shares the Povey window
but keeps its 256-bin, piecewise filterbank. The framework's general
`extract_kaldi_fbank` also keeps its separate double-precision bank; it is not
bitwise interchangeable with this float bank.

The shared mel projection selects its accumulator and power path before the hot
frequency loop. Epsilon stabilization and square-before-weight transforms are
applied once in place to each STFT magnitude, matching the original Seed-VC and
IndexTTS2 operation order while avoiding repeated work across overlapping mel
bands. `mel_spectrogram_frontend_test` compares the optimized paths bit-for-bit
with the original generic loop across dense and sparse projection, all power
modes, both layouts, both log precisions, and all output transforms.

The common bank is built with fixed 16 kHz/512 FFT/80 mel/20 Hz settings.
During migration, passing the same constants through a runtime-parameterized
builder changed nine F32 coefficients because compile-time and runtime float
evaluation differed. Keeping the fixed bank construction preserved the
original coefficients and restored exact frontend parity.

| Production frontend | Model-owned preparation | Shared frontend output |
| --- | --- | --- |
| IndexTTS2 reference | Resampling route and 15-second limit | S2 mel, CAMPPlus, semantic features and mask |
| Confucius4-TTS reference | Target and speaker sample rates | S2 mel, CAMPPlus, semantic features and mask |
| Seed-VC | Waveform input | Log mel |
| Seed-VC Whisper content | Fixed-length waveform for content encoder | Whisper-scaled log mel |
| Chatterbox S3 tokenizer | Mono input | Resampled, scaled log mel |
| Chatterbox S3 prompt | Mono input | Resampled log mel |
| Chatterbox voice encoder | Voice encoder waveform | Time-major power mel |
| GLM-TTS prompt | Audio selection | Resampled log mel |
| Niagara ASR | Audio input | Log mel with frame padding |
| Qwen3-TTS speaker reference | Reference selection | Log mel |
| VieNeu-TTS speaker reference | Reference selection | Log mel |
| DramaBox audio VAE reference | Stereo resampling and duration selection | Two-channel log mel |
| Mira-TTS speaker reference | Gain and fixed-duration repetition | Linear mel |

The shared frontend caches the immutable filterbank and sparse representation
per configuration. The variants preserve each migrated path's original Hann
window convention, STFT centering and padding, dense or sparse accumulation,
log precision, power accumulation order, output layout, and frame trimming or
padding. It does not replace the separate NeMo mel
frontend, which also covers dither, preemphasis, checkpoint windows and
filterbanks, normalization, and streaming-specific frame rules.

## Local feature parity

The dedicated `tests/shared_mel_migration/probe.cpp` calls production frontend
functions with deterministic 2,048- and 22,187-sample fixtures. Captures were
made immediately before and after each migration. The probe writes complete
F32 output tensors and frame/bin metadata; `compare.py` checks the bytes and
metadata. Reference-audio captures include both IndexTTS2 speaker routes,
both resampled waveforms, all three feature outputs, and attention masks (whose
integer values are recorded as exactly representable F32 values). Every run
used a Debug build and saved its log. Niagara's published GGUF was loaded for
frontend configuration; no decoder or end-to-end model inference was run.

| Frontend | F32 values compared | Result |
| --- | ---: | --- |
| IndexTTS2 | 7,520 | Identical |
| Confucius4-TTS | 7,520 | Identical |
| Seed-VC | 7,520 | Identical |
| Niagara ASR | 8,640 | Identical |
| Qwen3-TTS | 13,056 | Identical |
| VieNeu-TTS | 13,056 | Identical |
| DramaBox | 6,656 | Identical |
| Mira-TTS | 77,056 | Identical |
| Seed-VC Whisper content | 480,000 | Identical |
| Seed-VC CAMPPlus | 11,840 | Identical |
| DotTTS CAMPPlus | 11,840 | Identical |
| Chatterbox S3 tokenizer | 13,952 | Identical |
| Chatterbox S3 prompt | 4,320 | Identical |
| Chatterbox voice encoder | 6,080 | Identical |
| Chatterbox CAMPPlus | 8,480 | Identical |
| Kroko ASR fbank | 8,800 | Identical |
| IndexTTS2 complete reference | 133,032 | Identical |
| Confucius4-TTS complete reference | 66,516 | Identical |
| **Total** | **885,884** | **Identical** |

The same 885,884-value comparison remained identical after the projection-loop
optimization. On the preserved 180-second Seed-VC v1 component case,
`seed_vc.v1.mel_ms` changed from 264.275 ms before optimization to 243.344 ms;
main measured 241.302 ms. Each timing is one Debug/OpenMP observation and only
covers the mel component.

The post-optimization Seed-VC v1 Whisper path also produced a WAV byte-identical
to the preserved main output: both files were 549,932 bytes with SHA-256
`166a458d8cb02d3c045b056764c50fd71ceb81a13ff53e90f48a4f30233ca6f4`.

The model-specific mel rows were compared after their mel migration. Seed-VC
Whisper and the Chatterbox paths were then compared with their
pre-migration captures.
Finally, the complete IndexTTS2 and Confucius4-TTS reference outputs were
compared before and after moving their shared pipeline. CAMPPlus paths from
Seed-VC, DotTTS, and Chatterbox, plus Kroko's fbank, were then captured before
and after the float Kaldi bank/window consolidation. Chatterbox voice mel and
complete CAMPPlus extraction were then captured before and after their
shared-frontend migration. The 885,884-value final comparison checks all
listed paths; the successive
before/after comparisons establish each migration step.

To capture a build at a particular revision, supply your Niagara GGUF path and
capture directory. Capture the baseline before changing a model's mel math,
then run the same command in a second directory after migration.

```bash
cmake --build build/debug --target shared_mel_migration_probe -j 8
mkdir -p "$CAPTURE_DIR"
build/debug/bin/shared_mel_migration_probe \
  --niagara-model "$MODEL_FILE" \
  --out "$CAPTURE_DIR" --log "$CAPTURE_DIR/probe.log" \
  > "$CAPTURE_DIR/probe.stdout" 2>&1
python3 tests/shared_mel_migration/compare.py "$BEFORE_DIR" "$AFTER_DIR"
```

This is exact **frontend tensor parity** for the recorded mono, 22,050 Hz
fixtures. They exercise the speaker-rate resampling routes but do not exercise
target resampling from another input rate or IndexTTS2's 15-second truncation.
Other channel layouts were not measured by this component probe.

## End-to-end path parity

Separate pre-refactor and refactor binaries were built with matching Debug
settings and verified through their version output. The cases ran sequentially
with the same backend and thread count for each pair. Across the included
paths, all 69 comparable artifacts from 31 cases were byte identical, and the
path comparator reported no artifact or normalized-text differences.

| Model family | Cases | Artifacts | Result |
| --- | ---: | ---: | --- |
| Chatterbox | 3 | 3 | Identical |
| Confucius4-TTS | 2 | 6 | Identical |
| DotTTS | 6 | 32 | Identical |
| DramaBox | 2 | 3 | Identical |
| IndexTTS2 | 6 | 6 | Identical |
| Kroko ASR | 1 | 2 | Identical |
| Mira-TTS | 1 | 1 | Identical |
| Niagara ASR | 1 | 1 | Identical |
| Qwen3-TTS | 4 | 10 | Identical |
| Seed-VC | 4 | 4 | Identical |
| VieNeu-TTS | 1 | 1 | Identical |
| **Total** | **31** | **69** | **Identical** |

The DotTTS coverage includes offline and streaming generation; the individual
chunks, flush output, and assembled stream matched. Kroko ASR's transcript and
emitted word boundaries matched, while Niagara ASR's short and long transcripts
matched. The tested paths did not emit speaker-turn records.

## End-to-end RTF regression screen

RTF is processing wall time divided by audio duration, so lower is faster. For
generation, the denominator is generated audio duration; for ASR, it is input
audio duration. The complete sequential sweep produced these family-level
observations, calculated from the summed wall time and audio duration of each
family's cases:

| Model | Cases | Audio | Baseline RTF | Refactor RTF | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chatterbox | 3 | 435.80 s | 0.119563 | 0.120293 | +0.61% |
| Confucius4-TTS | 2 | 112.12 s | 0.247817 | 0.246501 | -0.53% |
| DotTTS | 6 | 140.32 s | 0.264698 | 0.263529 | -0.44% |
| DramaBox | 2 | 25.63 s | 0.332562 | 0.334619 | +0.62% |
| IndexTTS2 | 6 | 1,367.14 s | 0.124929 | 0.124980 | +0.04% |
| Kroko ASR | 1 | 14.07 s input | 0.024779 | 0.024289 | -1.97% |
| Niagara ASR | 1 | 17.73 s input | 0.024122 | 0.025847 | +7.15% |
| Qwen3-TTS | 4 | 386.96 s | 0.151008 | 0.152142 | +0.75% |
| Seed-VC | 4 | 233.11 s | 0.097814 | 0.097327 | -0.50% |
| VieNeu-TTS | 1 | 10.16 s | 0.045349 | 0.045209 | -0.31% |

Repeated measurements suggest that the observed RTF differences are run noise.

The long-form Chatterbox, IndexTTS2, and Qwen3-TTS cases were then repeated
sequentially in after-before-before-after order. All six WAV files compared for
each model were byte identical.

| Model | Audio | Baseline RTF | Refactor RTF | Change | Combined change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chatterbox | 419.52 s | 0.121007 | 0.119331 | -1.39% | -0.70% |
| IndexTTS2 | 417.02 s | 0.154890 | 0.154533 | -0.23% | -0.16% |
| Qwen3-TTS | 327.60 s | 0.150723 | 0.149033 | -1.12% | -0.44% |

The combined column includes the original parity-sweep observation and the two
ordered observations per revision. These measurements found no end-to-end RTF
regression in the repeated models. They are a regression screen for the full
request path and do not isolate frontend execution time.
