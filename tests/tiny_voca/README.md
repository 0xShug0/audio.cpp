# TinyVoca controlled-noise audio comparison

Use `reference_audio.py write --checkpoint MODEL --texts TEXTS --out REF` to
produce cases, durations, pitch and controlled noise.
Run the Debug `tiny_voca_synth_probe MODEL REF/cases.tsv CPP --log probe.log`.
One runtime is retained for the entire case sequence. Logging is optional;
when enabled it uses framework request timings and `session.wall_ms`.

Compare with:

```bash
python tests/tiny_voca/reference_audio.py compare --out REF --cpp CPP \
  --max-peak-relative-error "$WAVE_BOUND" --max-pitch-abs-error "$PITCH_HZ_BOUND"
```

Both bounds must be supplied explicitly and be finite and nonnegative. No
validated default bounds are claimed. Durations, sample counts, frame counts,
and the waveform and pitch array shapes must match exactly. Missing, empty, malformed or nonfinite
artifacts fail. Pitch acceptance is maximum absolute error in Hz; waveform
acceptance is maximum absolute difference divided by the reference peak.
