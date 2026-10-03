# OWSM-CTC

OWSM-CTC is ESPnet's non-autoregressive multilingual speech recognition and
speech translation family. The v4 checkpoint is
[espnet/owsm_ctc_v4_1B](https://huggingface.co/espnet/owsm_ctc_v4_1B).
It uses an E-Branchformer encoder with self-conditioned CTC and is separate
from the autoregressive [OWSM](owsm.md) family.

## Usage

```bash
audiocpp_cli --task asr --family owsm_ctc \
  --model /path/to/OWSM-CTC-GGUF/owsm-ctc-v4-1b-f32.gguf \
  --backend cuda --audio input.wav \
  --request-option language=eng --text-out transcript.txt --log
```

Add `--request-option target_language=deu` to translate into German instead
of transcribing. Language values are the checkpoint's ISO 639-3 codes, such as
`eng`, `jpn`, and `zho`. `language=auto` uses unknown-language conditioning;
it does not report a detected language.

The model runs offline best-path CTC decoding. It does not generate timestamp
tokens, use an autoregressive beam, or condition on previous transcript text.

## Common Options

| CLI option | Description |
| --- | --- |
| `--audio` | Input audio; resampled to mono 16 kHz. |
| `--backend` | Execution backend. |
| `--threads` | CPU thread count. |
| `--text-out` | Transcript output path. |

## Request Options (Use With `--request-option`)

| Option | Default | Description |
| --- | --- | --- |
| `language` | `eng` | Input language code; `auto` uses unknown-language conditioning. |
| `target_language` | Unset | Translation target language; omit for transcription. |
| `audio_chunk_mode` | `auto` | `auto`, `fixed`, or `none`. |

## Session Options (Use With `--session-option`)

| Option | Default | Description |
| --- | --- | --- |
| `owsm_ctc.weight_type` | `native` | Matmul weight storage override. |

## Long Audio

`auto` pads short input to the trained 30-second window. Longer recordings use
overlapping 30-second windows with two seconds of context on each side.
CTC frame predictions are joined before collapsing repeated tokens and blanks.
`fixed` also applies this overlapping-window path to short audio. `none` rejects
input longer than 30 seconds. The result is a whole-recording transcript,
without word or segment timestamps.

## Conversion

```bash
python tests/owsm_ctc/convert_gguf.py /path/to/owsm_ctc_v4_1B \
  /path/to/OWSM-CTC-GGUF/owsm-ctc-v4-1b-f32.gguf \
  --staging-dir /path/to/staging --type orig
```

Use `--type q8_0` and a corresponding output filename for Q8. The GGUF includes
the tokenizer, configuration, normalization statistics, and model spec.
