# MOSS-TTS-Local codec comparison

`moss_tts_local_codec_parity.py` generates the same deterministic code matrix
as the native codec decoder. Run in `conda run -n qwen3-tts` and compare with:

```bash
conda run -n qwen3-tts python tests/moss_tts_local/moss_tts_local_codec_parity.py \
  --frames 64 --cpp-wav CPP.wav --min-cosine "$COSINE_BOUND" \
  --max-abs-error "$ABS_BOUND"
```

Comparison requires explicit finite acceptance bounds: cosine in [-1, 1] and
nonnegative maximum absolute error on normalized PCM16 samples. No unverified
numerical defaults are provided. Both WAVs must be complete, nonempty 48 kHz
stereo PCM16 with exactly matching shapes. No prefix truncation is performed;
nonfinite generated audio, zero signal norms and acceptance failures are errors.
Use `--compare-only --out REF.wav --cpp-wav CPP.wav` with the same acceptance
flags to compare existing WAVs without importing or running the model.
Without `--cpp-wav`, the script only generates a reference WAV.
