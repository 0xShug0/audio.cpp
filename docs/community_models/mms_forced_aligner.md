# Meta MMS-300M Forced Aligner

`mms_forced_aligner` provides native GGML inference for the multilingual
[Meta MMS-300M CTC forced aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
(`Wav2Vec2ForCTC`, 315M parameters). It aligns an **exact transcript** to
mono/stereo audio and returns per-word start/end timestamps through the
generic `align` task as `TaskResult.word_timestamps`.

## Task and modes

- Task: `align`
- Mode: `offline` only
- Output: `word_timestamps` (original word strings with start/end samples in
  the source audio's sample rate, plus a mean aligned log-probability
  confidence)

## Text normalization

The checkpoint covers 1,130 languages, but its 31-symbol CTC alphabet
(`<blank>` + letters + apostrophe) requires romanized text. This port ships:

- `text_normalization=latin` (default): native normalization for Dutch
  (`nl`/`nld`) and English (`en`/`eng`). NFKD decomposition, lowercase,
  diacritic removal, apostrophe kept; punctuation and digits act as
  separators; standalone numbers are dropped. Unsupported non-Latin input is
  **rejected** rather than silently mis-aligned.
- `text_normalization=pre_romanized`: accepts caller-supplied ASCII
  romanization for any language (e.g. Uroman output produced offline).

Hyphenated compounds (`Twenty-two`) are preserved as one output word.

## Install

The checkpoint is CC-BY-NC-4.0. **No public audio.cpp GGUF redistribution is
approved** — convert locally.

Download the upstream safetensors package:

```bash
python tools/model_manager_v2.py install mms_forced_aligner_300m_safetensors --models-root models
```

Or download manually from
[`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
(commit `49402e9577b1158620820667c218cd494cc44486`) into
`models/mms-300m-1130-forced-aligner/` (requires `config.json`,
`model.safetensors`, `vocab.json`; tokenizer sidecars are optional).

## Local GGUF conversion

```powershell
build\mms\bin\Release\audiocpp_gguf.exe `
  --input models\mms-300m-1130-forced-aligner\model.safetensors `
  --root models\mms-300m-1130-forced-aligner `
  --family mms_forced_aligner `
  --model-spec model_specs\mms_forced_aligner.json `
  --output models\MMS-Forced-Aligner-GGUF\mms-forced-aligner-f16.gguf `
  --type f16 --overwrite
```

A Q8_0 package is converted the same way (`--type q8_0`); it keeps biases,
norms, charset lookups and non-matmul tensors unquantized and reproduces the
F32 timestamps bit-for-bit.

Do **not** pass `--fold-weight-norm`: the checkpoint stores the positional
convolution in PyTorch parametrization layout and the encoder folds it at
load time.

## Run

```bash
audiocpp_cli --task align --family mms_forced_aligner \
  --model models/mms-300m-1130-forced-aligner --backend cpu \
  --audio block.wav --text "De groep keert terug naar Reverdin." \
  --language nld --words-out words.json
```

The generic server accepts the same request fields (`text`, `language`,
`audio`) and returns `words`.

## Options

Request options:

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `language` | `nl`, `nld`, `en`, `eng` (latin); any code (pre_romanized) | — | Transcript language, canonicalized to ISO 639-3 |
| `text_normalization` | `latin`, `pre_romanized` | `latin` | Native Latin normalization or caller-supplied romanization |
| `star_frequency` | `segment`, `edges` | `segment` | Virtual `<star>` target placement (per-word prefix vs transcript edges) |
| `merge_threshold_sec` | float >= 0 | `0.0` | Merge words whose gap is at or below this many seconds |

Session options:

| Option | Default | Description |
| --- | --- | --- |
| `mms_forced_aligner.emission_window_sec` | `30.0` | Center window length for long waveforms |
| `mms_forced_aligner.emission_context_sec` | `2.0` | Left/right context per window; must be < window |
| `mms_forced_aligner.max_alignment_cells` | `50000000` | Hard cap on Viterbi cells (frames x states); fails before allocation |
| `mms_forced_aligner.max_target_tokens` | `8192` | Hard cap on flattened CTC targets per request |
| `mms_forced_aligner.weight_type` | `native` | Weight storage: `native`, `f32`, `f16` |

## Long-form alignment

Audio longer than one emission window is processed in 30 s center windows
with 2 s context; the DP is still O(frames x targets) in time. For long
recordings, split audio and transcript into **matching blocks** and send one
request per block (for example via the batch API or `--request-sequence`).
Standalone audio chunking is rejected by design: an audio-only chunker cannot
infer transcript boundaries.

Window handling is validated on a 42.2 s synthesized repeat of `sample.wav`
(24 words x 3, exact known transcript): the native output is monotonic, spans
the full audio, and preserves repeat periodicity; 57/72 word boundaries match
the pinned Python reference exactly and 71/72 are within one 320-sample
frame. The single outlier is a 60 ms sentence-gap tie on the final
`Twenty-two` end boundary.

## Limits and license

- Checkpoint license: **CC-BY-NC-4.0**. GGUF conversion is local-only; do not
  redistribute the GGUF publicly.
- Native normalization: Dutch and English Latin script only. Non-Latin input
  fails explicitly unless `pre_romanized` is used.
- No embedded Uroman engine.
- `confidence` is the mean aligned log-probability, not a calibrated
  probability.

## Tests

```bash
ctest --test-dir build/mms -R 'mms_' --output-on-failure
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py
build/mms/bin/audiocpp_cli --list-loaders
```

Parity evidence (F32 CPU, `resources/sample.wav`, English transcript): all 24
word boundaries match the pinned `ctc-forced-aligner` Python reference within
one 20 ms frame; the F16 GGUF reproduces the F32 safetensors timestamps
exactly.

| Weight type | vs pinned Python reference | vs F32 safetensors | Status |
| --- | --- | --- | --- |
| F32 (safetensors) | <= 20 ms (1 frame) | — | measured |
| F16 (local GGUF) | <= 20 ms (1 frame) | 0-sample max diff | measured |
| Q8_0 (local GGUF) | <= 20 ms (1 frame) | 0-sample max diff | measured |

Q8_0 parity was measured on the full validation set: all 24 words of the
English request, the 4-request warmbench set (apostrophe, Dutch-on-English,
full sample) and the 72-word windowed long-form are bit-identical to the F32
safetensors and F16 GGUF outputs (0-sample boundary diff on every request).

The Q8_0 GGUF is ~356 MB (vs ~631 MB F16): CTC forced alignment is tolerant
to weight quantization because the emitted per-frame logits only need to keep
an identical argmax path, and the converter leaves biases, norms, charset
lookups and positional-conv state unquantized.

Measured performance (Release MSVC build, CPU, F32): a full 14.07 s request
runs in ~1.8 s (~0.13x real-time) and the parity harness passes 4/4 exact
word parity; a 42.2 s long-form request runs at ~0.33x real-time with
~2.1 GB peak working set (the F32 checkpoint alone is ~1.26 GB; the F16 GGUF
halves that, Q8 less than a third). Debug builds are ~10x slower and are not
representative.
