# Parakeet TDT test fixtures

- `2086-149220-0033.wav`
  - 16 kHz mono, ~7.44 s speech clip, extracted from the LibriSpeech
    `test-clean` corpus, utterance `2086-149220-0033`
    ("Well, I don't wish to see it any more, observed Phoebe, turning away
    her eyes. It is certainly very like the old portrait.").
  - LibriSpeech (Panayotov et al., 2015) is distributed under CC BY 4.0:
    https://www.openslr.org/12
  - Used by `test_golden_transcription.cpp` as a fixed-input regression
    check for the Parakeet-TDT 0.6B v3 offline (full-context) ASR path.
    The expected transcription was verified against the actual NeMo
    reference model (`nvidia/parakeet-tdt-0.6b-v3` via
    `nemo.collections.asr.models.ASRModel.from_pretrained`) for this exact
    clip, not just eyeballed.
