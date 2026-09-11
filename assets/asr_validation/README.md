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
