# Test reference fixtures

These compact fixtures are independent correctness oracles, not a collection of
example recordings. General audio inputs live in `assets/resources/` and the
licensed audio/transcript corpus in `assets/asr_validation/`.

- `tokenizers/tokenizer-1.model`: SentencePiece model for tokenizer unit checks.
- `index_tts_tn_corpus.tsv`: text-normalization input/expected-output cases.
- `earshot_vad/upstream_reference.json`: per-frame scores and segment boundaries
  from the pinned Rust reference in `tests/earshot_vad/reference_rs`. Each entry
  identifies a shared WAV and its SHA-256. Regenerate from that reference when
  changing an input; never use the C++ detector to populate expected scores.
- `framework/audio_utilities/`: input and expected-output tensors for FlashSR,
  GTCRN, RNNoise and ZipEnhancer checks. The reference preparation scripts are
  under `tests/audio_utilities/`; runtime weights remain under
  `assets/framework/audio_utilities/`.
- `metal_im2col/`: source identity/provenance for the Metal regression check.

Generate silence, deterministic noise and format variants during test setup.
Changing a recorded input requires regenerating its independent expectations;
changing only a path while retaining identical bytes does not.
