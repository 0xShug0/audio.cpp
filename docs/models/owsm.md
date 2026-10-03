# OWSM

OWSM is ESPnet's multilingual speech recognition and speech translation family.
The supported version is v4.
Base (102M), Small (370M), and Medium (1B) use an E-Branchformer audio encoder
and an autoregressive Transformer text decoder.

F32 and Q8_0 packages are available for all three variants. Small and Medium
also provide Q4_K. Quantized packages reduce memory use but can change words
and timestamp segmentation. F32 is the default package; choose Q8_0 for a
smaller CPU-oriented option.

## Usage

```bash
audiocpp_cli --task asr --family owsm \
  --model /path/to/OWSM-GGUF/owsm-v4-base-102m-f32.gguf \
  --backend cpu --threads 8 --audio input.wav \
  --request-option language=eng --text-out transcript.txt --log
```

For translation, set the source language and add
`--request-option target_language=eng` (or another supported target).
For utterance timestamps, add `--request-option return_timestamps=true`
and `--segments-out segments.json`.

## Common Options

| Option | Description |
| --- | --- |
| `--audio` | Input audio. Non-16 kHz input is resampled with SOXR. |
| `--backend` | Execution backend. |
| `--mode` | `offline` (default) or `streaming` text output from a complete audio file. |
| `--threads` | CPU thread count. |
| `--text-out` | Transcript output file. |
| `--segments-out` | Utterance segment output file. Enable `return_timestamps`. |

## Request Options

Use with `--request-option name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `language` | `eng` | Source language code, or `auto` to identify it from the first window. |
| `target_language` | unset | Translation target. Omit for transcription. |
| `return_timestamps` | `false` | Return utterance-level segments. |
| `condition_on_previous_text` | `false` | Condition each window on its preceding transcript. |
| `beam_size` | `1` | Decoder beam width. `1` is greedy; `5` matches the official Python default. |
| `max_tokens` | `0` | Output token limit per window; `0` selects 374. |
| `audio_chunk_mode` | `auto` | `auto`, `fixed`, or `none`. |
| `audio_chunk_duration_sec` | `30` | Window duration, from 0.2 to 30 seconds. |

Translation targets: `ara`, `cat`, `ces`, `cym`, `deu`, `eng`, `est`, `fas`,
`fra`, `ind`, `ita`, `jpn`, `lav`, `mon`, `nld`, `por`, `ron`, `rus`, `slv`,
`spa`, `swe`, `tam`, `tur`, `vie`, and `zho`.

Automatic long-form decoding uses predicted timestamps to advance each window.
An unfinished final utterance near the window boundary is decoded again in the
next window. `fixed` uses consecutive windows; `none` accepts one window only.
Automatic long-form decoding predicts timestamps internally even when segment
output is disabled.

The decoder defaults to greedy search for CPU efficiency, corresponding to
official Python with `beam_size=1`. Use `beam_size=5` for the official Python
default. Beam search retains per-hypothesis KV states in host memory and reuses
the session's decoder graph.
Use `language=auto` when the source language is unknown. Detection uses the
first audio window and retains that language for the rest of the recording.
An explicit language avoids misclassification between similar languages.
## Streaming

Add `--mode streaming` for incremental text output from a complete audio file.
Greedy decoding emits token deltas. Beam search and automatic long-form
continuation emit committed window text, since unfinished hypotheses can change.
Concatenating the deltas gives the final transcript, with the same decoding
settings and output as offline mode. This is not live microphone streaming.

For the server, configure the model with `"mode": "streaming"` and send:

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F model=owsm -F file=@input.wav -F language=eng -F stream=true
```

The response uses `transcript.text.delta` followed by `transcript.text.done`.

## Session Options

Use with `--session-option owsm.name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `weight_type` | `native` | Matmul weight storage override. |

## Conversion

Download the selected variant's `config.yaml`, `valid.total_count.ave_5best.pth`,
`bpe.model`, and `feats_stats.npz`, preserving the upstream directory structure.

```bash
python tests/owsm/convert_gguf.py /path/to/official-snapshot \
  /path/to/OWSM-GGUF/owsm-v4-base-102m-f32.gguf \
  --staging-dir /path/to/conversion-staging
```

The converter embeds the tokenizer, runtime configuration, normalization weights,
and model spec. Use `--type q8_0` or `--type q4_k` to convert quantized weights.

## Upstream

- [Base](https://huggingface.co/espnet/owsm_v4_base_102M)
- [Small](https://huggingface.co/espnet/owsm_v4_small_370M)
- [Medium](https://huggingface.co/espnet/owsm_v4_medium_1B)
