# ASR Validation Assets

Small, public ASR/STT fixtures for smoke validation. These are not training data
and are intentionally tiny.

## LibriSpeech

Path: `assets/asr_validation/librispeech/`

The LibriSpeech examples are taken from the `openslr/librispeech_asr` Hugging Face
dataset mirror, with upstream source OpenSLR SLR12. LibriSpeech is distributed
under CC BY 4.0.

Each example includes:

- a 16 kHz mono WAV file
- a `.txt` reference transcript
- one row in `manifest.jsonl`

The manifest rows include source metadata, split, speaker/chapter ids, duration,
audio path, transcript path, and the reference text.

## Common Voice (Japanese)

Path: `assets/asr_validation/common_voice_ja/`

The Common Voice example is `common_voice_ja_20461197` from the Common Voice 8.0
Japanese test split, taken from the `japanese-asr/ja_asr.common_voice_8_0`
Hugging Face dataset mirror (row 264), with upstream source Mozilla Common
Voice. Common Voice is distributed under CC0 1.0. The 48 kHz MP3 was decoded
and resampled to 16 kHz. The mirror ends each transcription with a `.`, which
is kept as it is.

Each example includes the same files and manifest fields as LibriSpeech, with
the mirror's row in place of the speaker and chapter ids.
